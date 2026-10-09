"""Fixed parent/candidate target-free generation; NEVER executes robot actions."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rl/memlite_online/code'))
sys.path.insert(0, str(REPO / 'scripts/eval/memlite_sft100'))
from recovery_corpus import file_sha, digest
from recovery_planner_data import VerifiedRecoveryPlannerDataset
from recovery_sft_data import raw_observation
from recovery_generation_metrics import score_event, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--model', choices=('parent', 'candidate'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    # Fail on an incomplete code closure before reading large model files.
    importlib.import_module('g05.utils.memlite_planner_format')
    cfg = json.loads(args.config.read_text()); root = Path(cfg['root'])
    if args.output.exists():
        raise FileExistsError(args.output)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Use clean frozen code')
    for item in (cfg['models'][args.model], cfg['feedback'], cfg['history'], cfg['inventory']):
        if file_sha(root / item['path']) != item['sha256']:
            raise ValueError('Changed pinned evaluation artifact: ' + item['path'])
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str, range(8))) or subprocess.check_output(
            ['nvidia-smi', '-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Select one verified idle GPU')
    import numpy as np
    import torch
    from g05.models.g05.qwen35 import vision
    from g05.utils.training.stage1_model import configuration, restore_model, make_processor
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    from g05.utils.memlite_skill_protocol import planner_input_projection
    from g05.utils.memlite_causal_feedback import single_frame_planner_prefix
    from sparse_cache import indexed_cache_freeze
    from g05.models.kv_cache import SparseKVCache
    vision._flash_attn_varlen = None; vision._flash_attn_backend = None
    torch.set_num_threads(2); torch.manual_seed(17)
    names = {}
    release = root / cfg['expert_release']
    for line in (release / 'episodes.jsonl').read_text().splitlines():
        ep = json.loads(line); names[ep['row']['task_index']] = ep['task_name']
    config = configuration(root, 'high', names)
    model_path = root / cfg['models'][args.model]['path']
    saved = torch.load(model_path, map_location='cpu', mmap=True, weights_only=False)
    model, restore = restore_model(config, 'high', state=saved['model_state_dict'])
    del saved; gc.collect()
    model.requires_grad_(False).eval().cuda(); processor = make_processor(config, False)
    expert = Stage1Dataset(release, config, 'high', 'eval')
    union = root / cfg['union']
    recovery = VerifiedRecoveryPlannerDataset(union/'admission', union/'raw',
        json.loads((root/cfg['inventory']['path']).read_text()), config, split='dev',
        admission_sha256=cfg['admission_sha256'],
        history=[json.loads(x) for x in (root/cfg['history']['path']).read_text().splitlines()],
        feedback=[json.loads(x) for x in (root/cfg['feedback']['path']).read_text().splitlines()],
        evidence_root=root, high_sha256=cfg['models']['parent']['sha256'])
    fixed = np.load(release/'fixed_eval_indices.npy').reshape(100,32)[:,0].tolist()
    schedule = [('original_heldout', int(i)) for i in fixed] + [('recovery_dev', i) for i in range(len(recovery))]
    args.output.mkdir(parents=True)
    receipt = dict(config_sha256=file_sha(args.config), checkpoint_sha256=cfg['models'][args.model]['sha256'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        restoration=restore, schedule_sha256=digest(schedule), target_free=True,
        recovery_dev_unseen_by_increment_only=True, no_physical_success_measurement=True)
    (args.output/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    started = time.monotonic(); rows = []
    for split, index in schedule:
        if split == 'original_heldout':
            raw, identity = expert.raw(index)
            target = raw.pop('model_projection')
        else:
            item = recovery.rows[index]
            raw = raw_observation(item['candidate'], recovery.reader, config)
            target = recovery.projections[index]
            identity = dict(sample_id=item['candidate']['sample_id'], source_group=item['candidate']['source_group'])
        # Targets remain on CPU in a separate dictionary. The actual model
        # receives only a whitelist prefix and the three processed images.
        prepared = processor._process_tensors(raw)
        prefix = single_frame_planner_prefix(processor.samples_builder, prepared, planner_input_projection(target))
        pixels = {k: v.unsqueeze(0).cuda() for k,v in prepared['pixel_values'].items()}
        event = None; error = None; text = None; tick = time.monotonic()
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), indexed_cache_freeze(model.model.ar_helper, SparseKVCache):
                result = model.generate_high_level([prefix], pixels, temperature=0.)
            event = result['planner_events'][0]
            text = result.get('high_level_text')
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            error = str(exc); text = getattr(exc, 'planner_texts', None)
        row = dict(split=split, identity=identity, score=score_event(event,target), event=event, error=error,
            text=text, seconds=time.monotonic()-tick,
            expected={k:target[k] for k in ('next_decision','target_parent_goal','active_skills_semantic_json','memory_update')})
        rows.append(row)
        with (args.output/'rows.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
        (args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows), total=len(schedule), summary=summarize(rows)))+'\n')
        print(json.dumps(dict(completed=len(rows), split=split, score=row['score'])), flush=True)
    (args.output/'result.json').write_text(json.dumps(dict(receipt, status='completed', seconds=time.monotonic()-started,
        summary=summarize(rows)), indent=2)+'\n')


if __name__ == '__main__':
    main()
