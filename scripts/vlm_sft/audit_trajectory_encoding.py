"""Bounded CPU-only validation of real current RGB -> response-only VLM inputs.

Uses the 50 already reviewed TRAIN examples, not a new sampling/labeling run.
Loads processor assets only. Does not load a model, use CUDA, train, or execute.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from audit_expert_action_capacity import snapshot
from audit_trajectory_codec import CORPUS_SHA, MODEL_SPEC, check_row, percentiles
import native_trajectory_codec as codec
from prepare_full_annotation import forbid_source_output, packed, strict_json
from trajectory_images import checked_current_pngs
import trajectory_modeling as modeling

REPO = Path(__file__).resolve().parents[2]
REVIEW_SHA = '19900e355372542fbe3504692980fcaf41b58dee75ae4f2544bbd98cbf2447bc'
HUMAN_SHA = 'ca5401b5b789f4d880267608dbb22810cfd48fb2eb85767bb03d87b5e21ccb12'
MAX_SECONDS = 600
MAX_BYTES = 2 * 1024**2
PROCESSOR_FILES = ('config.json', 'chat_template.jinja', 'merges.txt', 'preprocessor_config.json',
                   'tokenizer.json', 'tokenizer_config.json', 'video_preprocessor_config.json', 'vocab.json')


def checked_cases(manifest, human):
    cases = manifest['cases']
    if (manifest['status'] != 'DECODED_HUMAN_REVIEW_PENDING' or manifest['training_eligible'] is not False
            or manifest['manifest_sha256'] != CORPUS_SHA or len(cases) != 50
            or human['status'] != 'SAMPLED_VISUAL_ALIGNMENT_REVIEW_COMPLETED'
            or human['review_manifest_sha256'] != REVIEW_SHA or human['corpus_manifest_sha256'] != CORPUS_SHA
            or human['reviewed_cases'] != 50 or human['confirmed_visual_alignment_issues_in_sample'] != 0):
        raise ValueError('Pinned completed parent review required, not a training-release certificate')
    ids = {c['id'] for c in cases}
    if len(ids) != 50 or ids != {c['id'] for c in human['cases']} or len(human['cases']) != 50:
        raise ValueError('Incomplete or repeated reviewed cohort')
    if (Counter(c['task'] for c in cases) != Counter({i: 10 for i in range(5)})
            or len({(c['task'], c['instance']) for c in cases}) != 50):
        raise ValueError('Expected ten independent TRAIN instances per task')
    for case in cases:
        if case['split'] != 'train' or case['row']['id'] != case['id'] or case['frame'] != case['row']['frame_index']:
            raise ValueError('Review/current row identity mismatch')
        check_row(case['row'], case)
        current = [r for r in case['images'] if r['offset'] == 0]
        if len(current) != 3: raise ValueError('Exactly three original current image receipts required')
    return cases


def check_encoding(processor, actor, images, target):
    """Independently process each view, then check the actual response mask."""
    import torch
    prefix = modeling.encode(processor, actor, images)
    training = modeling.encode(processor, actor, images, target=target)
    length = prefix['input_ids'].shape[1]
    seqkeys = {'input_ids', 'attention_mask', 'token_type_ids', 'mm_token_type_ids'}
    if set(training) != set(prefix) | {'labels'}: raise ValueError('Unexpected training fields')
    for key, value in prefix.items():
        actual = training[key][:, :length] if key in seqkeys else training[key]
        if not torch.equal(value, actual): raise ValueError(f'Train/inference prefix changed: {key}')
    labels = training['labels'][0]
    expected = processor.tokenizer.encode(target, add_special_tokens=False) + [modeling.eos_id(processor)]
    if (not torch.all(labels[:length] == -100) or labels[length:].tolist() != expected
            or not torch.equal(labels[length:], training['input_ids'][0, length:])
            or processor.tokenizer.decode(expected[:-1], skip_special_tokens=False) != target):
        raise ValueError('Actual assistant-only JSON/EOS supervision changed')
    # Counts alone do not prove view order or payload. Compare each independently
    # preprocessed current image against its respective chunk in the joint prefix.
    content = modeling.messages(actor, images)[1]['content']
    cursor = 0
    for index, view in enumerate(modeling.VIEWS):
        single = processor.image_processor(images=[content[2 * index + 1]['image']], return_tensors='pt')
        pixels = single['pixel_values']; count = pixels.shape[0]
        if (not torch.equal(prefix['image_grid_thw'][index:index + 1], single['image_grid_thw'])
                or not torch.equal(prefix['pixel_values'][cursor:cursor + count], pixels)):
            raise ValueError(f'Actual image tensor ordering/payload changed: {view}')
        cursor += count
    if cursor != prefix['pixel_values'].shape[0]: raise ValueError('Extra visual tensor rows')
    stats = {'prefix_tokens': length, 'target_tokens_including_eos': len(expected),
             'total_tokens': training['input_ids'].shape[1], 'supervised_tokens': int((labels != -100).sum()),
             'image_grid_thw': prefix['image_grid_thw'].tolist(),
             'pixel_shape': list(prefix['pixel_values'].shape),
             'prefix_ids_sha256': hashlib.sha256(packed(prefix['input_ids'].tolist())).hexdigest(),
             'pixels_tensor_sha256': hashlib.sha256(prefix['pixel_values'].cpu().numpy().tobytes()).hexdigest(),
             'target_sha256': hashlib.sha256(target.encode()).hexdigest()}
    return training, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic(); pins = {}
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('Explicit CPU-only, offline processor required')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean immutable source required')
    code = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    for name in ('audit_trajectory_encoding.py', 'trajectory_images.py', 'trajectory_modeling.py',
                 'native_trajectory_codec.py', 'modeling.py', 'audit_trajectory_codec.py',
                 'audit_expert_action_capacity.py', 'prepare_full_annotation.py'):
        snapshot(REPO / 'scripts/vlm_sft' / name, pins)
    manifest = strict_json(snapshot(args.review_root / 'manifest.json', pins, REVIEW_SHA))
    human = strict_json(snapshot(REPO / 'configs/vlm_sft/h85_parent_action_review_v1.json', pins, HUMAN_SHA))
    cases = checked_cases(manifest, human)
    spec = strict_json(snapshot(MODEL_SPEC, pins)); model_path = Path(spec['model'])
    for name in PROCESSOR_FILES: snapshot(model_path / name, pins, spec['model_files'][name])
    protected = [args.review_root, REPO, model_path]
    protected.extend(Path(r['video_path']).resolve().parent for c in cases for r in c['row']['image_references'])
    forbid_source_output(args.output, *protected)
    if args.output.exists(): raise FileExistsError('Preserve existing encoding run')
    if shutil.disk_usage(args.output.parent).free < 80 * 1024**3: raise ValueError('Disk reserve')
    os.sched_setaffinity(0, set(range(48, 52)))
    import torch
    import transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized(): raise ValueError('CUDA must not be initialized')
    args.output.mkdir()
    launch = {'schema': 'h85-real-trajectory-encoding-v1', 'code_commit': code, 'pid': os.getpid(),
              'utc': datetime.now(timezone.utc).isoformat(), 'protocol': codec.VERSION,
              'max_seconds': MAX_SECONDS, 'max_bytes': MAX_BYTES, 'max_cases': 50,
              'model': str(model_path), 'model_revision': spec['revision'], 'model_weight_loads': 0,
              'training_updates': 0, 'controls': 0, 'training_eligible': False,
              'review_manifest_sha256': REVIEW_SHA, 'parent_review_sha256': HUMAN_SHA,
              'corpus_manifest_sha256': CORPUS_SHA, 'torch': torch.__version__, 'transformers': transformers.__version__}
    (args.output / 'launch.json').write_bytes(packed(launch) + b'\n')
    records = []; shortest = longest = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        for case in cases:
            if time.monotonic() - started > MAX_SECONDS - 20: raise TimeoutError('CPU encoding deadline')
            start = time.monotonic(); row = case['row']
            receipts = [r for r in case['images'] if r['offset'] == 0]
            images = checked_current_pngs(row, receipts, args.review_root)
            for r in receipts: pins[str((args.review_root / r['path']).resolve())] = r['png_sha256']
            actor = codec.actor_from_state(row['task'], row['active_instruction'], row['observation_state'])
            target = codec.encode(row['expert_action'], row['observation_state'])
            training, stats = check_encoding(processor, actor, images, target)
            for img in images.values(): img.close()
            n = stats['total_tokens']
            if shortest is None or n < shortest['input_ids'].shape[1]: shortest = training
            if longest is None or n > longest['input_ids'].shape[1]: longest = training
            records.append({'id': case['id'], 'task': case['task'], 'instance': case['instance'],
                            'split': 'train', 'frame': row['frame_index'], **stats,
                            'current_images': receipts, 'cpu_encoding_seconds': time.monotonic() - start})
            if len(records) % 10 == 0:
                print(json.dumps({'cases': len(records), 'seconds': time.monotonic() - started}), flush=True)
        batch = modeling.collate([shortest, longest], processor.tokenizer.pad_token_id)
        if (not torch.any(batch['attention_mask'] == 0)
                or not torch.all(batch['labels'][batch['attention_mask'] == 0] == -100)
                or int((batch['labels'] != -100).sum()) != sum(int((v['labels'] != -100).sum()) for v in (shortest, longest))):
            raise ValueError('Real variable-length batch padding lost/supervised tokens')
        for path, digest in pins.items(): snapshot(path, {}, digest)
        if torch.cuda.is_initialized(): raise ValueError('CPU check unexpectedly initialized CUDA')
        data = b''.join(packed(r) + b'\n' for r in records)
        elapsed = time.monotonic() - started
        result = {**launch, 'status': 'CPU_MULTIMODAL_ENCODING_CHECKED_NOT_TRAIN_RELEASE',
                  'cases': len(records), 'current_images': len(records) * 3, 'wall_seconds': elapsed,
                  'per_task': dict(Counter(r['task'] for r in records)), 'cuda_initialized': False,
                  'prefix_tokens': percentiles([r['prefix_tokens'] for r in records]),
                  'target_tokens': percentiles([r['target_tokens_including_eos'] for r in records]),
                  'total_tokens': percentiles([r['total_tokens'] for r in records]),
                  'cpu_encoding_seconds': percentiles([r['cpu_encoding_seconds'] for r in records]),
                  'rows_sha256': hashlib.sha256(data).hexdigest(), 'rows_bytes': len(data), 'input_sha256': pins,
                  'all_current_image_bindings_checked': True, 'per_view_tensor_comparison': True,
                  'same_inference_prefix': True, 'assistant_json_eos_only': True, 'real_left_padding_checked': True,
                  'model_forward_backward': 'NOT_RUN', 'generation_latency': 'NOT_MEASURED',
                  'training_throughput_and_three_hours': 'NOT_MEASURED'}
        result_bytes = packed(result) + b'\n'
        if elapsed > MAX_SECONDS or len(result_bytes) + len(data) + (args.output / 'launch.json').stat().st_size > MAX_BYTES:
            raise ValueError('Encoding time/output budget exceeded')
        (args.output / 'rows.jsonl').write_bytes(data)
        (args.output / 'result.json').write_bytes(result_bytes)
    except BaseException as exc:
        (args.output / 'failure.json').write_bytes(packed({'error': repr(exc), 'cases': len(records),
                                                         'seconds': time.monotonic() - started}) + b'\n')
        raise


if __name__ == '__main__': main()
