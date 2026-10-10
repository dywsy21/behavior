"""Fixed parent/candidate target-free generation; NEVER executes robot actions."""
import argparse
from collections import Counter
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
from recovery_generation_metrics import score_event, summarize, feedback_probe, select_original_indices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--model', choices=('parent', 'candidate'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--feedback-probe', action='store_true', help='Counterfactual shortcut diagnostic, not heldout accuracy')
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
    expert_class=Stage1Dataset
    original_feedback=cfg.get('original_feedback','none_v1')
    if original_feedback not in ('none_v1','causal_expert_unknown_v1'):
        raise ValueError('Unknown normal-state feedback protocol')
    if original_feedback=='causal_expert_unknown_v1':
        from recovery_expert_feedback import with_expert_feedback
        expert_class=with_expert_feedback(Stage1Dataset,repeat_stride_controls=cfg.get('original_feedback_repeat_stride',16))
    expert = expert_class(release, config, 'high', 'eval')
    union = root / cfg['union']
    recovery = VerifiedRecoveryPlannerDataset(union/'admission', union/'raw',
        json.loads((root/cfg['inventory']['path']).read_text()), config, split='dev',
        admission_sha256=cfg['admission_sha256'],
        history=[json.loads(x) for x in (root/cfg['history']['path']).read_text().splitlines()],
        feedback=[json.loads(x) for x in (root/cfg['feedback']['path']).read_text().splitlines()],
        evidence_root=root, high_sha256=cfg['models']['parent']['sha256'],
        uncertainty_protocol=cfg.get('uncertainty_protocol','raw_observer_confidence_v1'))
    original_schedule=cfg.get('original_schedule','first_fixed_one_per_task_v1')
    fixed=select_original_indices(np.load(release/'fixed_eval_indices.npy').reshape(100,32).tolist(),
        lambda i:expert.locate(i)[3],lambda i:int(expert.task_ids[i]),original_schedule)
    original_with_previous=sum(expert.locate(i)[3]!='None' for i in fixed)
    schedule = [('original_heldout', int(i)) for i in fixed] + [('recovery_dev', i) for i in range(len(recovery))]
    args.output.mkdir(parents=True)
    receipt = dict(config_sha256=file_sha(args.config), checkpoint_sha256=cfg['models'][args.model]['sha256'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        restoration=restore, schedule_sha256=digest(schedule), target_free=True,
        original_schedule=original_schedule,original_with_previous_intent=original_with_previous,
        original_feedback=original_feedback,uncertainty_protocol=cfg.get('uncertainty_protocol','raw_observer_confidence_v1'),
        original_feedback_repeat_stride=cfg.get('original_feedback_repeat_stride',16),
        probe_unknown_confidence=cfg.get('probe_unknown_confidence',0.) if args.feedback_probe else None,
        recovery_dev_unseen_by_increment_only=True, no_physical_success_measurement=True,
        counterfactual_feedback_probe=args.feedback_probe,
        synthetic_probe_metrics_are_not_real_world_accuracy=bool(args.feedback_probe))
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
        causal=planner_input_projection(target);probe_kind='observed'
        if args.feedback_probe:
            causal,probe_kind=feedback_probe(causal,split,unknown_confidence=cfg.get('probe_unknown_confidence',0.))
        prefix = single_frame_planner_prefix(processor.samples_builder, prepared, causal)
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
        row = dict(split=split, identity=identity, probe_kind=probe_kind, score=score_event(event,target), event=event, error=error,
            text=text, seconds=time.monotonic()-tick,
            expected={k:target[k] for k in ('next_decision','target_parent_goal','active_skills_semantic_json','memory_update')})
        rows.append(row)
        with (args.output/'rows.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
        (args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows), total=len(schedule), summary=summarize(rows)))+'\n')
        print(json.dumps(dict(completed=len(rows), split=split, score=row['score'])), flush=True)
    coverage=dict(Counter(r['probe_kind'] for r in rows))
    if (args.feedback_probe and original_schedule=='noninitial_fixed_one_per_task_v1'
            and coverage.get('synthetic_unknown_feedback_on_normal_state')!=100):
        raise ValueError('Requested noninitial feedback control was not actually exercised')
    (args.output/'result.json').write_text(json.dumps(dict(receipt, status='completed', seconds=time.monotonic()-started,
        probe_kind_counts=coverage,normal_feedback_probe_actually_exercised=bool(
            args.feedback_probe and coverage.get('synthetic_unknown_feedback_on_normal_state',0)),
        summary=summarize(rows)), indent=2)+'\n')


if __name__ == '__main__':
    main()
