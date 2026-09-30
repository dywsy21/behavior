"""CPU-only, fail-closed export of B-final and original TRAIN probe samples.

Never changes the original checkpoint, labels, source tree or environment.
Model-only export retains every tensor; it deliberately excludes Adam state.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

WORK = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910')
RUN = WORK / 'formal_b_parent_format_1500_v1'
SOURCE = WORK / 'b_parent_format_train_source_v2'
WEIGHT = RUN / 'checkpoints/step_1500.pt'
WEIGHT_SHA = 'd4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef'
WEIGHT_BYTES = 26593013632


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def current_frame(original):
    """Preserve every label, selecting only the final historical observation."""
    row = deepcopy(original)
    sample = row['samples']
    old = ''.join(f'<image{i}_image_!>' for i in range(18))
    new = ''.join(f'<image{i}_image_!>' for i in range(3))
    if sample['template'].count(old) != 1:
        raise ValueError('Unexpected eighteen-image planner template')
    sample['template'] = sample['template'].replace(old, new)
    sizes = [sample[f'image{i}'] for i in (5, 11, 17)]
    for key in list(sample):
        if re.fullmatch(r'image\d+', key):
            del sample[key]
    sample.update({f'image{i}': value for i, value in enumerate(sizes)})
    if sample['proprio']['value'].shape != (6, 27):
        raise ValueError('Unexpected original proprio history')
    sample['proprio']['value'] = sample['proprio']['value'][-1:].clone()
    if len(row['pixel_values']) != 3:
        raise ValueError('Expected three cameras')
    for key, value in row['pixel_values'].items():
        if value.shape != (6, 3, 256, 256):
            raise ValueError('Unexpected original image shape')
        row['pixel_values'][key] = value[-1:].clone()
    return {'samples': sample, 'pixel_values': row['pixel_values']}


def export_weights(output, torch):
    if WEIGHT.stat().st_size != WEIGHT_BYTES or digest(WEIGHT) != WEIGHT_SHA:
        raise ValueError('Original B-final full SHA/size changed')
    saved = torch.load(WEIGHT, map_location='cpu', mmap=True, weights_only=False)
    if saved['step'] != 1500 or not saved['model_state_dict']:
        raise ValueError('Not the registered final planner checkpoint')
    state = saved['model_state_dict']
    if any('lora_' in key for key in state):
        raise ValueError('B-final must be the full planner, not a low LoRA checkpoint')
    export = {'step': 1500, 'model_state_dict': state, 'source_sha256': WEIGHT_SHA,
              'coordination_training_identity': saved.get('coordination_training_identity')}
    target = output / 'B-final-model.pt'
    torch.save(export, target)
    restored = torch.load(target, map_location='cpu', mmap=True, weights_only=False)['model_state_dict']
    if set(restored) != set(state):
        raise ValueError('Model tensor coverage changed')
    for key in state:
        left, right = state[key], restored[key]
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8)):
            raise ValueError('Model tensor bytes changed: ' + key)
    for name, source in {
        'B-training-config.yaml': RUN / '.hydra/config.yaml',
        'B-dataset-stats.json': RUN / 'dataset_stats.json',
        'B-run-receipt.json': RUN / 'coordination_run_receipt.json',
        'B-trainability-receipt.json': RUN / 'coordination_trainability_receipt.json',
    }.items():
        shutil.copyfile(source, output / name)
    return dict(source=str(WEIGHT), source_sha256=WEIGHT_SHA, source_bytes=WEIGHT_BYTES,
                path=str(target), sha256=digest(target), bytes=target.stat().st_size,
                exact_model_tensors=len(state), model_only=True, optimizer_state_exported=False)


def export_inputs(output, torch):
    # The original Hydra config has reviewed relative oc.load parts-meta paths.
    # Resolve them against its immutable source, never the SSH login directory.
    os.chdir(SOURCE)
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'scripts'), str(SOURCE)]
    from omegaconf import OmegaConf
    import pyarrow.parquet as pq
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.data.processor_utils import instantiate_dataset, build_processors
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    register_default_resolvers()
    cfg = OmegaConf.load(RUN / '.hydra/config.yaml')
    # Model construction and wandb initialization are intentionally absent.
    spec = json.loads((WORK / 'formal_b_inputs_v1/train_sampling_index.json').read_text())
    if spec['split'] != 'train' or spec.get('public_test_training_ingest'):
        raise ValueError('Only the locked original TRAIN split may be exported')
    eligible = {(int(r['episode_index']), int(r['frame_index'])): r for r in spec['high']}
    release = json.loads((WORK / 'formal_b_inputs_v1/released_high_only_manifest.json').read_text())
    sidecar = Path(release['labels_path'])
    if digest(sidecar) != release['labels_sha256']:
        raise ValueError('Original released high sidecar changed')
    columns = ['episode_index', 'frame_index', 'memory', 'previous_intent',
               'active_skills_semantic_json', 'target_parent_goal', 'memory_update']
    records = pq.read_table(sidecar, columns=columns).to_pylist()
    selected = []
    for task in range(5):
        candidates = []
        for row in records:
            key = (int(row['episode_index']), int(row['frame_index']))
            if key in eligible and int(eligible[key]['stratum']['task_index']) == task:
                candidates.append((sum(len(str(row[k])) for k in columns[2:]), key))
        candidates.sort()
        if len(candidates) < 6:
            raise ValueError('Missing task in original TRAIN release')
        for fraction in (.10, .35, .60, .85, .95, 1.):
            size, key = candidates[round((len(candidates)-1)*fraction)]
            selected.append(dict(task=task, episode=key[0], frame=key[1], length_proxy=size,
                                 fraction=fraction, dataset_index=int(eligible[key]['dataset_index'])))
    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.datastatics_path))
    processor.eval()  # deterministic image preprocessing for a timing fixture
    dataset = instantiate_dataset(cfg, is_training_set=True)
    dataset.set_processor(processor)
    live = dataset.get_memlite_coordination_sampling_spec()
    if live['eligible_digest'] != spec['eligible_digest']:
        raise ValueError('Live reader and locked TRAIN index disagree')
    rows = []
    for locator in selected:
        original = dataset[(locator['dataset_index'], 'high')]
        if original['samples']['memlite_branch'] != 'high' or original['samples']['outcome_supervision_mask']:
            raise ValueError('Expected planner-only original TRAIN row')
        rows.append(current_frame(original))
        print(json.dumps({'input_prepared': len(rows), **locator}), flush=True)
    target = output / 'B-train-singleframe-samples.pt'
    torch.save(rows, target)
    return dict(path=str(target), sha256=digest(target), bytes=target.stat().st_size, rows=len(rows),
                source_sidecar_sha256=release['labels_sha256'], train_eligible_digest=spec['eligible_digest'],
                split='original_train_only', labels_modified=False, observation_frames=1,
                selection=selected, scope='Five original tasks, stratified text-length proxies; not 100-task labels')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kind', choices=('weights', 'inputs'), required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['WANDB_MODE'] = 'disabled'
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = export_weights(args.output, torch) if args.kind == 'weights' else export_inputs(args.output, torch)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only export initialized CUDA')
    result.update(status='complete', cuda_initialized=False, seconds=time.monotonic()-started,
                  commit=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                                                   'rev-parse', 'HEAD'], text=True).strip())
    publish(args.output / 'receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
