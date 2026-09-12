"""Opt-in task-equal, branch- and event-stratified MEM-Lite sampling.

The old sampler is unchanged. This sampler consumes a separately audited index
over the *existing* training view; it never constructs targets or loads images.
Repeated draws, if needed, stay inside permanently disjoint DDP source shards.
"""
from __future__ import annotations

import bisect
from copy import deepcopy
import hashlib
import json
from numbers import Integral
from pathlib import Path

import torch

from .dist import ResumableDistributedBranchBalancedBatchSampler


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Sampling index requires integer coordinates, got {value!r}")
    return int(value)


def canonical_ranges(ranges, size):
    """Validate nonoverlap/bounds and merge adjacent half-open intervals."""
    rows = []
    for pair in ranges:
        if len(pair) != 2:
            raise ValueError("Each sampling range must have exactly two endpoints")
        start, end = map(_integer, pair)
        if not 0 <= start < end <= size:
            raise ValueError(f"Sampling range outside Dataset view: {(start, end)} / {size}")
        rows.append((start, end))
    result = []
    for start, end in sorted(rows):
        if result and start < result[-1][1]:
            raise ValueError("Overlapping sampling ranges would duplicate source rows")
        if result and start == result[-1][1]:
            result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return result


def branch_fingerprint(spec, size):
    high = sorted(_integer(x) for x in spec['high'])
    if len(set(high)) != len(high) or any(x < 0 or x >= size for x in high):
        raise ValueError("Invalid or duplicate high-level source indices")
    low = canonical_ranges(spec['low_ranges'], size)
    starts = [x[0] for x in low]
    for index in high:
        pos = bisect.bisect_right(starts, index) - 1
        if pos >= 0 and index < low[pos][1]:
            raise ValueError("High and low sampling pools overlap")
    raw = json.dumps({'dataset_length': size, 'high': high, 'low_ranges': low},
                     sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def build_memlite_train_sampler(dataset, *, sampling_config, **kwargs):
    """Explicit opt-in factory; legacy configurations retain their old sampler."""
    strategy = str(sampling_config.get('strategy', 'branch'))
    if strategy == 'branch':
        return ResumableDistributedBranchBalancedBatchSampler(dataset, **kwargs)
    if strategy in {'motion_recovery', 'fm_motion'}:
        from .motion_sampler import ResumableDistributedMotionBatchSampler
        index = sampling_config.get('sampling_index_path')
        if not index:
            raise ValueError("motion_recovery requires an audited sampling_index_path")
        if strategy == 'fm_motion':
            return ResumableDistributedMotionBatchSampler(dataset, sampling_index=index,
                low_only=True, low_quotas_override=sampling_config.get('low_quotas'), **kwargs)
        return ResumableDistributedMotionBatchSampler(dataset, sampling_index=index, **kwargs)
    if strategy != 'task_event':
        raise ValueError(f"Unknown MEM-Lite training sampling strategy: {strategy!r}")
    index = sampling_config.get('sampling_index_path')
    if not index:
        raise ValueError("task_event sampling requires an audited sampling_index_path")
    return ResumableDistributedTaskEventBatchSampler(dataset, sampling_index=index,
        critical_per_batch=sampling_config.get('critical_per_batch', 1), **kwargs)


class ResumableDistributedTaskEventBatchSampler(ResumableDistributedBranchBalancedBatchSampler):
    """Equal task exposure, with a fixed event quota in every per-rank batch.

    At batch=8/high_fraction=.125/critical_per_batch=1, every rank gets
    1 high + 1 low gripper-transition window + 6 other low rows. The event and
    other pools are a disjoint, complete partition of the old low pool. No
    samples are removed because they are stationary, near a boundary, or hard.

    Each pool cycles independently only after its rank-owned rows are exhausted.
    Unlike the legacy sampler, low rows may intentionally repeat within a long
    epoch because task equality/event oversampling changes the sampling measure.
    Resumption replays the exact prefix without loading examples.
    """

    def __init__(self, dataset, *, sampling_index, critical_per_batch=1, **kwargs):
        if bool(getattr(dataset, 'is_rank_sharded', False)):
            raise ValueError("Task/event sampling currently requires an unsharded Dataset view")
        super().__init__(dataset, **kwargs)
        self.critical_per_batch = _integer(critical_per_batch)
        self.other_per_batch = self.low_per_batch - self.critical_per_batch
        if self.critical_per_batch < 1 or self.other_per_batch < 1:
            raise ValueError("Task/event batches require at least one critical and one other low row")
        if isinstance(sampling_index, (str, Path)):
            index = json.loads(Path(sampling_index).read_text())
        else:
            index = deepcopy(sampling_index)
        if index.get('schema_version') != 1:
            raise ValueError("Unsupported task/event sampling index schema")
        size = len(dataset)
        if _integer(index.get('dataset_length')) != size:
            raise ValueError("Sampling index Dataset length does not match")
        expected = branch_fingerprint(dataset.get_memlite_branch_sampling_spec(), size)
        if index.get('branch_fingerprint') != expected:
            raise ValueError("Sampling index branch fingerprint does not match the actual Dataset view")
        self._verify_identity(dataset, index.get('view_identity'))
        tasks = index.get('tasks')
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("Task/event index has no tasks")
        self.task_ids = []
        self._task_pools = []
        all_high, all_low = [], []
        for task in tasks:
            task_id = str(task['task_id'])
            if task_id in self.task_ids or not task_id:
                raise ValueError("Task IDs must be nonempty and unique")
            self.task_ids.append(task_id)
            high = [_integer(x) for x in task['high']]
            critical = canonical_ranges(task['critical_low_ranges'], size)
            other = canonical_ranges(task['other_low_ranges'], size)
            pools = {'high': self._make_pool(indices=high),
                     'critical': self._make_pool(ranges=critical),
                     'other': self._make_pool(ranges=other)}
            for kind, pool in pools.items():
                if pool['size'] < self._global_replicas:
                    raise ValueError(f"Task {task_id} {kind} pool is smaller than DDP world size")
            self._task_pools.append(pools)
            all_high.extend(high)
            all_low.extend(critical + other)
        # This also rejects cross-task/category overlap, missing rows, invented
        # indices and accidental holdout-view coordinates.
        if branch_fingerprint({'high': all_high, 'low_ranges': all_low}, size) != expected:
            raise ValueError("Task/event pools do not exactly partition the existing branch pools")
        self.index_metadata = {k: deepcopy(index[k]) for k in
                               ('schema_version', 'dataset_length', 'branch_fingerprint', 'view_identity')}

    @staticmethod
    def _verify_identity(dataset, identity):
        if not isinstance(identity, dict):
            raise ValueError("Sampling index must declare its training-view identity")
        if not bool(getattr(dataset, 'is_training_set', False)):
            raise ValueError("Task/event sampling index is training-only; validation must remain unchanged")
        if bool(getattr(dataset, 'use_weight_for_sampling', False)):
            raise ValueError("Weighted logical Dataset views are not supported")
        children = getattr(dataset, 'datasets', [dataset])
        actual = {'dataset_roots': [], 'sidecar_paths': [], 'active_episodes': []}
        recovery_identity = getattr(dataset, 'recovery_identity', None)
        if recovery_identity is not None:
            actual['recovery'] = recovery_identity
        for child in children:
            roots = getattr(child, 'dataset_dirs', None)
            sidecar = getattr(child, '_memlite_sidecar', None)
            episodes = getattr(child, '_active_episode_indices', None)
            if not roots or sidecar is None or episodes is None:
                raise ValueError("Task/event index requires explicit dataset roots, sidecar and episode split")
            actual['dataset_roots'].append([str(Path(p).resolve()) for p in roots])
            actual['sidecar_paths'].append(str(Path(sidecar.path).resolve()))
            actual['active_episodes'].append([int(x) for x in episodes])
        if identity != actual:
            raise ValueError("Sampling index training-view identity does not match Dataset")

    @staticmethod
    def _make_pool(*, indices=None, ranges=None):
        if indices is not None:
            return {'indices': indices, 'ranges': None, 'ends': None, 'size': len(indices)}
        ends, total = [], 0
        for start, end in ranges:
            total += end - start
            ends.append(total)
        return {'indices': None, 'ranges': ranges, 'ends': ends, 'size': total}

    def _make_task_stream(self, kind, generator):
        count = len(self.task_ids)
        order = torch.randperm(count, generator=generator).tolist() if self.shuffle else list(range(count))
        affines = [self._shared_affine_permutation(pool[kind]['size'], generator, self.shuffle)
                   for pool in self._task_pools]

        def stream():
            cursors, position = [0] * count, 0
            while True:
                task = order[(position + self._global_rank) % count]
                pool = self._task_pools[task][kind]
                source = self._rank_position(cursors[task], pool['size'])
                multiplier, offset = affines[task]
                flat = (multiplier * source + offset) % pool['size']
                if pool['indices'] is not None:
                    yield pool['indices'][flat]
                else:
                    which = bisect.bisect_right(pool['ends'], flat)
                    previous = 0 if which == 0 else pool['ends'][which - 1]
                    yield pool['ranges'][which][0] + flat - previous
                cursors[task] += 1
                position += 1
        return stream()

    def _streams(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1000003 * self.epoch)
        high = self._make_task_stream('high', generator)
        critical = self._make_task_stream('critical', generator)
        other = self._make_task_stream('other', generator)

        def low():
            while True:
                for _ in range(self.critical_per_batch):
                    yield next(critical)
                for _ in range(self.other_per_batch):
                    yield next(other)
        return generator, high, low()
