"""Offline composite-action training recipe and capacity benchmark helpers.

The sampler covers unique TRAIN windows without replacement. IDs remain in
audit receipts, never in model inputs. No online executor or legacy trainer is
changed, and this module does not authorize a long training run.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np

from audit_expert_action_capacity import snapshot
from prepare_full_annotation import packed, strict_json
from trajectory_dataset import TrajectoryDataset
from trajectory_modeling import collate, supervised_loss

DATA = Path('/mnt/nvme_tmp/robodojo_vlm_actions_20260926/h85_sft_v1')
DATA_SHA = 'fb95625b265b564cb07cb481615a2f3fead194c26c18bfd65a70da77441fbad9'
QA = DATA.parent / 'h85_dataset_check_v1/result.json'
QA_SHA = 'b5f59acf8a7243d06131791bb183873dab0cfaaf2b6da27b99db4a238caea73a'
ROOT = DATA.parent / 'h85_training_capacity_v1'
CONFIG = {'protocol': 'r1-composite16-v1', 'seed': 41,
          'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05,
          'learning_rate': 5e-5, 'weight_decay': .01, 'betas': [.9, .95],
          'microbatch': 2, 'effective_batch': 8, 'updates': 32, 'warmup_updates': 4,
          'loader_workers': 4, 'prefetch_factor': 2, 'wall_seconds': 1800,
          'max_gpu_mib': 32768, 'max_bytes': 1024**3,
          'required_training_seconds': 10800, 'checkpoint_writes': 0}


def epoch_order(length, seed=41, epoch=0):
    if type(length) is not int or length <= 0 or type(epoch) is not int or epoch < 0:
        raise ValueError('Positive dataset size and nonnegative epoch required')
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(length).tolist()


def training_sources(manifest):
    entries = []; ends = []; seen = set(); count = 0
    for source in manifest['sources']:
        if source['split'] != 'train': continue
        key = (source['task'], source['instance'])
        if key in seen: raise ValueError('TRAIN source instance duplicated')
        seen.add(key)
        files = [f for f in source['files'] if f['kind'] == 'shards']
        if len(files) != 1 or files[0]['samples'] != source['samples'] or source['samples'] <= 0:
            raise ValueError('Exactly one accepted TRAIN shard per source required')
        file = files[0]
        if Path(file['path']).parts != ('shards', 'train', f'episode_{source["episode"]:06d}.parquet'):
            raise ValueError('TRAIN source path mismatch')
        entries.append((file, source)); count += file['samples']; ends.append(count)
    if count != manifest['samples_by_split']['train'] or not count:
        raise ValueError('TRAIN totals mismatch')
    return entries, ends


def make_plan(root, manifest, *, cfg=CONFIG):
    """Only read TRAIN shards; pick gates by length, not loss or outcomes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    entries, ends = training_sources(manifest); total = ends[-1]
    count = cfg['updates'] * cfg['effective_batch']
    if not 2 <= count <= total: raise ValueError('Enough unique TRAIN windows required')
    order = epoch_order(total, cfg['seed']); metadata = {}; shortest = longest = None
    # Keep only prospective prefix candidates and two extremes in memory.
    wanted = set(order[:count + 2]); pins = {}; offset = 0
    for file, source in entries:
        data = snapshot(Path(root) / file['path'], pins, file['sha256'])
        if len(data) != file['bytes']: raise ValueError('TRAIN shard byte count changed')
        rows = pq.read_table(pa.BufferReader(data), columns=['id', 'total_tokens', 'target_tokens'], use_threads=False).to_pylist()
        if len(rows) != file['samples']: raise ValueError('TRAIN shard row count changed')
        for i, row in enumerate(rows):
            index = offset + i
            value = {'index': index, 'id': row['id'], 'task': source['task'], 'instance': source['instance'],
                     'episode': source['episode'], 'total_tokens': row['total_tokens'], 'target_tokens': row['target_tokens']}
            if index in wanted: metadata[index] = value
            if shortest is None or (value['total_tokens'], index) < (shortest['total_tokens'], shortest['index']): shortest = value
            if longest is None or (value['total_tokens'], -index) > (longest['total_tokens'], -longest['index']): longest = value
        offset += len(rows)
    gates = [shortest['index'], longest['index']]
    if len(set(gates)) != 2: raise ValueError('Mixed-length TRAIN gate required')
    metadata.update({shortest['index']: shortest, longest['index']: longest})
    chosen = gates + [i for i in order[:count + 2] if i not in gates][:count - 2]
    selected = [metadata[i] for i in chosen]
    if len({r['id'] for r in selected}) != count or {r['task'] for r in selected} != set(range(5)):
        raise ValueError('Unique, five-task TRAIN benchmark required')
    timed = selected[cfg['warmup_updates'] * cfg['effective_batch']:]
    if {r['task'] for r in timed} != set(range(5)): raise ValueError('Measured portion lacks a task')
    return {'schema': 'h85-train-capacity-sampling-v1', 'train_windows': total,
            'train_instances': len(entries), 'gate_indices': gates, 'selected': selected,
            'selected_task_counts': dict(Counter(str(r['task']) for r in selected)),
            'selected_instances': len({(r['task'], r['instance']) for r in selected}),
            'epoch_order_sha256': hashlib.sha256(packed(order)).hexdigest(), 'shard_pins': pins}


class LazyTrainingDataset:
    """Spawn-safe worker-local processor; never pickle open AV/CUDA handles."""
    def __init__(self, root, digest, model, length):
        self.root, self.digest, self.model, self.length = str(root), digest, str(model), length
        self._dataset = None

    def __len__(self): return self.length

    def __getstate__(self):
        return {**self.__dict__, '_dataset': None}

    def __getitem__(self, index):
        if self._dataset is None:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(self.model, local_files_only=True)
            processor.tokenizer.padding_side = 'left'
            self._dataset = TrajectoryDataset(self.root, self.digest, processor, allow_prepared=True)
            if len(self._dataset) != self.length: raise ValueError('Worker dataset length changed')
        return index, self._dataset[index]


def initialize_loader_worker(worker_id):
    import os
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    if torch.cuda.is_initialized(): raise RuntimeError('Loader worker initialized CUDA')


def collate_training(items, *, pad_token_id):
    indices, values = zip(*items)
    batch = collate(values, pad_token_id)
    return {'indices': list(indices), 'batch': batch,
            'target_tokens': int((batch['labels'] != -100).sum()),
            'total_tokens': int(batch['attention_mask'].sum())}


def accumulated_backward(model, microbatches, *, move):
    """One token-mean objective over the entire effective batch, not means of means."""
    import torch
    total = sum(x['target_tokens'] for x in microbatches)
    if total <= 0: raise ValueError('No action targets in effective batch')
    losses = []
    for item in microbatches:
        batch = move(item['batch'])
        if int((batch['labels'] != -100).sum()) != item['target_tokens']:
            raise ValueError('Effective-batch token count changed')
        loss = supervised_loss(model, batch)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite action CE')
        weight = item['target_tokens'] / total
        (loss * weight).backward(); losses.append(float(loss.detach()) * weight)
    return sum(losses)


def capacity_report(records, train_windows, *, cfg=CONFIG):
    if (len(records) != cfg['updates'] or [r['step'] for r in records] != list(range(1, cfg['updates'] + 1))
            or type(train_windows) is not int or train_windows <= 0):
        raise ValueError('Complete ordered benchmark required')
    measured = records[cfg['warmup_updates']:]
    all_ids = [i for r in records for i in r['indices']]
    if len(all_ids) != len(set(all_ids)): raise ValueError('Benchmark repeated a window')
    for r in records:
        if len(r['indices']) != cfg['effective_batch'] or any(type(i) is not int or not 0 <= i < train_windows for i in r['indices']):
            raise ValueError('Invalid TRAIN sample accounting')
        for key in ('compute_seconds', 'end_to_end_seconds'):
            if type(r[key]) not in (int, float) or not math.isfinite(r[key]) or r[key] <= 0:
                raise ValueError('Positive finite timing required')
        if r['compute_seconds'] > r['end_to_end_seconds']: raise ValueError('Inverted timing scope')
    samples = len(measured) * cfg['effective_batch']
    compute_rate = samples / sum(r['compute_seconds'] for r in measured)
    e2e_rate = samples / sum(r['end_to_end_seconds'] for r in measured)
    fastest_rate = max(cfg['effective_batch'] / r['compute_seconds'] for r in measured)
    one_pass_seconds = train_windows / fastest_rate
    return {'timed_updates': len(measured), 'timed_unique_windows': samples,
            'compute_samples_per_second': compute_rate, 'end_to_end_samples_per_second': e2e_rate,
            'fastest_observed_compute_samples_per_second': fastest_rate,
            'one_unique_pass_seconds_at_fastest_observed_compute': one_pass_seconds,
            'windows_for_three_hours_at_fastest_observed_compute': math.ceil(cfg['required_training_seconds'] * fastest_rate),
            'single_unique_pass_supports_three_hours': one_pass_seconds >= cfg['required_training_seconds'],
            'scope': 'Measured fixed recipe on one A100; not an optimized-throughput upper bound or policy-quality result'}


def checked_data():
    manifest = strict_json(snapshot(DATA / 'manifest.json', {}, DATA_SHA))
    qa = strict_json(snapshot(QA, {}, QA_SHA))
    if (qa['status'] != 'CPU_DATASET_LOADING_VERIFIED_CAPACITY_PENDING' or qa['dataset_manifest_sha256'] != DATA_SHA
            or qa['real_train_items_checked'] != 50 or not qa['all_pixels_equal_prior_checked_encoder']
            or qa['full_recount']['accepted_by_split'] != manifest['samples_by_split']):
        raise ValueError('Missing real dataset loading evidence')
    return manifest, qa
