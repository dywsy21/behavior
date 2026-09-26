"""CPU-only preflight of the exact prospective 256-window training sample path.

No model weights, optimizer, GPU or training. Validates the real spawn workers
and largest TRAIN sequences before a GPU becomes available. CPU throughput is
not used to declare the three-hour training capacity proven.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import time

from audit_expert_action_capacity import snapshot
from audit_trajectory_encoding import PROCESSOR_FILES
from audit_trajectory_codec import MODEL_SPEC
from prepare_full_annotation import strict_json
from prepare_visual_review import atomic_json, sha
from train_visual_presence import BASE, clean_commit, environment_identity
from trajectory_training import (CONFIG, DATA, DATA_SHA, QA_SHA, checked_data, make_plan,
    LazyTrainingDataset, initialize_loader_worker, collate_training)

OUTPUT = DATA.parent / 'h85_benchmark_cpu_v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('Explicit CPU-only offline preflight required')
    if OUTPUT.exists(): raise FileExistsError('Preserve previous CPU preflight')
    code = clean_commit(); environment = environment_identity(); manifest, _ = checked_data()
    pins = {}; spec = strict_json(snapshot(MODEL_SPEC, pins))
    for name in PROCESSOR_FILES: snapshot(Path(BASE) / name, pins, spec['model_files'][name])
    os.sched_setaffinity(0, set(range(48, 56))); started = time.monotonic()
    OUTPUT.mkdir(); iterator = loader = None
    atomic_json(OUTPUT / 'launch.json', {'code_commit': code, 'config': CONFIG, 'pid': os.getpid(),
        'utc': datetime.now(timezone.utc).isoformat(), 'dataset_sha256': DATA_SHA, 'qa_sha256': QA_SHA,
        'wall_seconds': 600, 'max_bytes': 4 * 1024**2, 'model_weight_loads': 0, 'training_updates': 0})
    try:
        import torch
        from transformers import AutoProcessor
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized(): raise RuntimeError('Unexpected CUDA initialization')
        plan = make_plan(DATA, manifest); atomic_json(OUTPUT / 'sampling.json', plan)
        processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
        selected = plan['selected']; indices = [r['index'] for r in selected]
        by_index = {r['index']: r for r in selected}
        batches = [indices[i:i + CONFIG['microbatch']] for i in range(0, len(indices), CONFIG['microbatch'])]
        dataset = LazyTrainingDataset(DATA, DATA_SHA, BASE, plan['train_windows'])
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches,
            num_workers=CONFIG['loader_workers'], multiprocessing_context='spawn',
            worker_init_fn=initialize_loader_worker, pin_memory=False,
            collate_fn=partial(collate_training, pad_token_id=processor.tokenizer.pad_token_id),
            prefetch_factor=CONFIG['prefetch_factor'], timeout=120)
        records = []; iterator = iter(loader)
        for expected in batches:
            if time.monotonic() - started >= 570: raise TimeoutError('CPU loader deadline')
            item = next(iterator); tensors = item['batch']
            if (item['indices'] != expected
                    or item['target_tokens'] != sum(by_index[i]['target_tokens'] for i in expected)
                    or item['total_tokens'] != sum(by_index[i]['total_tokens'] for i in expected)
                    or any(not isinstance(v, torch.Tensor) or v.device.type != 'cpu' for v in tensors.values())
                    or not torch.all(tensors['labels'][tensors['attention_mask'] == 0] == -100)):
                raise RuntimeError('Spawn loader changed sample, mask, device, or token counts')
            records.append({'indices': expected, 'target_tokens': item['target_tokens'],
                            'total_tokens': item['total_tokens'], 'padded_length': tensors['input_ids'].shape[1]})
        try: next(iterator)
        except StopIteration: pass
        else: raise RuntimeError('Extra loader sample')
        checked_data()
        for path, digest in {**pins, **plan['shard_pins']}.items():
            if sha(Path(path)) != digest: raise RuntimeError('Source changed during CPU preflight')
        if torch.cuda.is_initialized(): raise RuntimeError('CPU preflight initialized CUDA')
        result = {'status': 'CPU_SPAWN_TRAINING_LOADER_VERIFIED_GPU_PENDING', 'code_commit': code,
            'config': CONFIG, 'environment': environment, 'dataset_sha256': DATA_SHA, 'qa_sha256': QA_SHA,
            'sampling_sha256': sha(OUTPUT / 'sampling.json'), 'unique_train_windows': len(set(indices)),
            'decoded_current_images': 3 * len(indices), 'train_instances': plan['train_instances'],
            'selected_instances': plan['selected_instances'], 'selected_task_counts': plan['selected_task_counts'],
            'gate_indices': plan['gate_indices'], 'model_weight_loads': 0, 'training_updates': 0,
            'cuda_initialized': False, 'three_hour_capacity': 'NOT_MEASURED',
            'wall_seconds': time.monotonic() - started, 'records': records, 'training_eligible': False}
        if result['wall_seconds'] > 600: raise TimeoutError('CPU preflight exceeded deadline')
        if sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file()) + len(json.dumps(result).encode()) > 4 * 1024**2:
            raise RuntimeError('CPU preflight artifact budget')
        atomic_json(OUTPUT / 'result.json', result)
    except BaseException as exc:
        atomic_json(OUTPUT / 'failure.json', {'error': repr(exc), 'wall_seconds': time.monotonic() - started})
        raise
    finally: del iterator, loader


if __name__ == '__main__': main()
