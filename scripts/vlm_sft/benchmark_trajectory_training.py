"""One guarded 32-update action-data benchmark; no checkpoint or deployment.

Launched only by launch_trajectory_benchmark.py on an otherwise empty GPU2.
This validates the real recipe, not a three-hour training or policy outcome.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from modeling import load_model
from prepare_visual_review import atomic_json, sha
from train_visual_presence import (BASE, GPU_UUID, clean_commit, model_identity,
                                  environment_identity)
from trajectory_dataset import TrajectoryDataset
from trajectory_modeling import collate, supervised_loss
from trajectory_training import (CONFIG, DATA, DATA_SHA, QA_SHA, ROOT, checked_data,
                                 make_plan, LazyTrainingDataset, initialize_loader_worker,
                                 collate_training, accumulated_backward, capacity_report)


def run():
    code = clean_commit(); launch = json.loads((ROOT / 'launch.json').read_text())
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or launch['code_commit'] != code
            or launch['config'] != CONFIG or launch['dataset_sha256'] != DATA_SHA or launch['qa_sha256'] != QA_SHA
            or hashlib.sha256(os.environ.get('H85_BENCHMARK_TOKEN', '').encode()).hexdigest() != launch['token_sha256']):
        raise ValueError('Exact guarded benchmark launch required')
    output = ROOT / 'training'; output.mkdir(exist_ok=False)
    started = time.monotonic(); updates = 0; gate_data = None; loader = iterator = None

    def budget():
        if time.monotonic() - started >= CONFIG['wall_seconds']: raise TimeoutError('Action benchmark wall budget')
        if shutil.disk_usage(ROOT).free < 80 * 1024**3: raise RuntimeError('Disk reserve')
        if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) >= CONFIG['max_bytes']:
            raise RuntimeError('Action benchmark artifact cap')

    try:
        manifest, qa = checked_data(); identity = model_identity(); environment = environment_identity()
        if identity != launch['model_identity'] or environment != launch['environment']:
            raise ValueError('Registered base/environment changed')
        plan = make_plan(DATA, manifest); atomic_json(output / 'sampling.json', plan); budget()
        import torch
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed']); random.seed(CONFIG['seed'])
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction((CONFIG['max_gpu_mib'] - 1024) * 1024**2 / total_memory)
        model, processor = load_model(BASE, train=True, cfg=CONFIG); budget()
        params = [p for p in model.parameters() if p.requires_grad]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not params or any('lora_' not in n or 'language_model' not in n for n in names):
            raise RuntimeError('Unexpected trainable parameters')
        if any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if 'lora_B' in n):
            raise RuntimeError('Fresh zero-effect language LoRA required')
        atomic_json(output / 'identity.json', {
            'code_commit': code, 'config': CONFIG, 'model_identity': identity, 'environment': environment,
            'dataset_sha256': DATA_SHA, 'qa_sha256': QA_SHA, 'physical_gpu': 2,
            'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(),
            'trainable_parameters': sum(p.numel() for p in params), 'trainable_names': names,
            'old_adapter_loaded': False, 'online_executor_changed': False})

        # Use global shortest/longest TRAIN sequences, not heldout losses.
        gate_data = TrajectoryDataset(DATA, DATA_SHA, processor, allow_prepared=True)
        gate_items = [gate_data[i] for i in plan['gate_indices']]
        gate_batch = collate(gate_items, processor.tokenizer.pad_token_id)
        if not (gate_batch['attention_mask'] == 0).any(): raise RuntimeError('Mixed-length loss gate missing')
        gate_batch = {k: v.to('cuda') for k, v in gate_batch.items()}
        model.eval()
        with torch.no_grad():
            native = model(**gate_batch, use_cache=False).loss
            custom = supervised_loss(model, gate_batch)
        if not torch.isfinite(native) or not torch.allclose(native, custom, rtol=1e-5, atol=1e-4):
            raise RuntimeError('Native/custom long-action CE disagreement')
        atomic_json(output / 'loss_gate.json', {'indices': plan['gate_indices'], 'native': float(native),
            'custom': float(custom), 'absolute_error': float((native - custom).abs()),
            'left_padding': True, 'supervised_tokens': int((gate_batch['labels'] != -100).sum())})
        del gate_batch, native, custom, gate_items
        gate_data.close(); gate_data = None; budget()

        selected = plan['selected']; indices = [r['index'] for r in selected]
        by_index = {r['index']: r for r in selected}
        batches = [indices[i:i + CONFIG['microbatch']] for i in range(0, len(indices), CONFIG['microbatch'])]
        worker_data = LazyTrainingDataset(DATA, DATA_SHA, BASE, plan['train_windows'])
        loader = torch.utils.data.DataLoader(worker_data, batch_sampler=batches,
            num_workers=CONFIG['loader_workers'], multiprocessing_context='spawn',
            worker_init_fn=initialize_loader_worker, pin_memory=True,
            collate_fn=partial(collate_training, pad_token_id=processor.tokenizer.pad_token_id),
            prefetch_factor=CONFIG['prefetch_factor'], timeout=120)
        iterator = iter(loader)
        optimizer = torch.optim.AdamW(params, lr=CONFIG['learning_rate'], betas=tuple(CONFIG['betas']),
                                      weight_decay=CONFIG['weight_decay'])
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        records = []; changed = 0; model.train(); torch.cuda.reset_peak_memory_stats()
        accumulation = CONFIG['effective_batch'] // CONFIG['microbatch']
        with (output / 'steps.jsonl').open('x', buffering=1) as log:
            for step in range(1, CONFIG['updates'] + 1):
                budget(); started_step = time.perf_counter()
                microbatches = [next(iterator) for _ in range(accumulation)]
                observed = [i for item in microbatches for i in item['indices']]
                expected = indices[(step - 1) * CONFIG['effective_batch']:step * CONFIG['effective_batch']]
                if observed != expected: raise RuntimeError('Loader sampling changed')
                for item in microbatches:
                    expected_tokens = sum(by_index[i]['target_tokens'] for i in item['indices'])
                    if item['target_tokens'] != expected_tokens: raise RuntimeError('Action target token count changed')
                torch.cuda.synchronize(); compute_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = accumulated_backward(model, microbatches,
                    move=lambda batch: {k: v.to('cuda', non_blocking=True) for k, v in batch.items()})
                gradient = torch.nn.utils.clip_grad_norm_(params, 1.)
                if not torch.isfinite(gradient) or float(gradient) <= 0: raise RuntimeError('Nonfinite/zero LoRA gradient')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError('Frozen base received gradient')
                optimizer.step(); torch.cuda.synchronize()
                compute_seconds = time.perf_counter() - compute_start
                end_to_end_seconds = time.perf_counter() - started_step; updates = step
                if torch.cuda.max_memory_reserved() > CONFIG['max_gpu_mib'] * 1024**2:
                    raise RuntimeError('Own GPU allocation exceeded cap')
                if step == 1:
                    changed = sum(not torch.equal(p, before[n]) for n, p in model.named_parameters() if p.requires_grad)
                    if not changed: raise RuntimeError('No LoRA tensor updated')
                    del before
                record = {'step': step, 'indices': observed, 'loss': loss, 'gradient_norm': float(gradient),
                    'compute_seconds': compute_seconds, 'end_to_end_seconds': end_to_end_seconds,
                    'target_tokens': sum(x['target_tokens'] for x in microbatches),
                    'input_and_target_tokens': sum(x['total_tokens'] for x in microbatches),
                    'elapsed_seconds': time.monotonic() - started}
                records.append(record); log.write(json.dumps(record, allow_nan=False) + '\n')
                print(json.dumps({k: v for k, v in record.items() if k != 'indices'}), flush=True)
                budget()
        try: next(iterator)
        except StopIteration: pass
        else: raise RuntimeError('Unexpected extra benchmark sample')
        report = capacity_report(records, plan['train_windows'])
        checked_data()
        for path, digest in plan['shard_pins'].items():
            if sha(Path(path)) != digest: raise RuntimeError('TRAIN source changed during benchmark')
        budget()
        atomic_json(output / 'result.json', {'status': 'TRAINING_PIPELINE_BENCHMARKED',
            'code_commit': code, 'config': CONFIG, 'dataset_sha256': DATA_SHA, 'qa_sha256': QA_SHA,
            'optimizer_updates': updates, 'unique_train_windows': len(set(indices)),
            'changed_lora_tensors': changed, 'capacity': report, 'train_windows': plan['train_windows'],
            'sampling_sha256': sha(output / 'sampling.json'), 'steps_sha256': sha(output / 'steps.jsonl'),
            'identity_sha256': sha(output / 'identity.json'), 'loss_gate_sha256': sha(output / 'loss_gate.json'),
            'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
            'wall_seconds': time.monotonic() - started, 'checkpoint_writes': 0,
            'three_hour_training_performed': False, 'training_eligible': False,
            'policy_success_rate_evaluated': False})
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'error': repr(exc), 'optimizer_updates': updates,
            'wall_seconds': time.monotonic() - started, 'checkpoint_writes': 0})
        raise
    finally:
        if gate_data is not None: gate_data.close()
        # Spawned loader processes belong to the supervisor-owned session;
        # normal exhaustion closes them, supervisor handles exceptional cleanup.
        del iterator, loader


if __name__ == '__main__': run()
