"""Bounded, stateless task-balanced sampling of the admitted native anchors.

The optimizer's completed update count, not the prefetch iterator, is the
cursor. This plan never silently starts another epoch when a task runs out.
"""
import hashlib
import json

import numpy as np


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class BalancedPlan:
    def __init__(self, dataset, *, updates=8000, batch_size=256, seed=17, tasks=100):
        if dataset.split != 'train' or not 0 < tasks <= 100:
            raise ValueError('A protected TRAIN dataset and explicit task count are required')
        if updates < 1 or batch_size < tasks or seed < 0:
            raise ValueError('Invalid bounded mixed-task plan')
        self.updates, self.batch_size, self.seed, self.tasks = updates, batch_size, seed, tasks
        self.base, self.extra = divmod(batch_size, tasks)
        self.order = np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(tasks)
        self.rank = np.argsort(self.order)
        intervals = [[] for _ in range(tasks)]
        identities = []
        for serial, ep in enumerate(dataset.episodes):
            row = ep['row']; task = int(row['task_index'])
            if ep['split'] != 'train' or not 0 <= task < tasks:
                raise ValueError('Non-TRAIN or unexpected task in dataset')
            start, stop = map(int, dataset.offsets[serial:serial + 2])
            if stop <= start:
                raise ValueError('Empty admitted episode')
            intervals[task].append((start, stop))
            identities.append([int(row['episode_index']), task, int(row['task_instance_id']),
                               int(row['raw_episode_id']), int(row['length']), ep['phase'], start, stop])
        self.dataset_fingerprint = digest_json(identities)
        self.selected = []
        self.pool_sizes = []
        for task, spans in enumerate(intervals):
            if not spans:
                raise ValueError(f'Task {task} has no admitted anchors')
            starts, stops = np.asarray(spans, dtype=np.int64).T
            local_offsets = np.r_[0, np.cumsum(stops - starts)]
            size = int(local_offsets[-1]); self.pool_sizes.append(size)
            needed = self._consumed(updates, task)
            if needed > size:
                raise ValueError(f'Task {task} needs {needed} distinct anchors but has {size}; no automatic repetition')
            positions = np.random.default_rng(np.random.SeedSequence([seed, 1, task])).permutation(size)[:needed]
            serials = np.searchsorted(local_offsets, positions, side='right') - 1
            selected = starts[serials] + positions - local_offsets[serials]
            selected.flags.writeable = False
            self.selected.append(selected)
        self.contract = dict(schema='champion-balanced-v1', updates=updates, batch_size=batch_size,
                             seed=seed, tasks=tasks, task_order=self.order.tolist(),
                             dataset_fingerprint=self.dataset_fingerprint, pool_sizes=self.pool_sizes,
                             selected_sha256=[hashlib.sha256(x.astype('<i8').tobytes()).hexdigest()
                                              for x in self.selected])
        self.fingerprint = digest_json(self.contract)

    def _consumed(self, completed, task):
        cycles, remaining = divmod(completed * self.extra, self.tasks)
        return completed * self.base + cycles + int(self.rank[task] < remaining)

    def batch(self, update):
        if not 0 <= update < self.updates:
            raise IndexError('Update outside the registered plan')
        indices = np.concatenate([self.selected[t][self._consumed(update, t):self._consumed(update + 1, t)]
                                  for t in range(self.tasks)])
        if len(indices) != self.batch_size:
            raise ValueError('Balanced plan lost samples')
        np.random.default_rng(np.random.SeedSequence([self.seed, 2, update])).shuffle(indices)
        return indices

    def cursor(self, completed):
        if not 0 <= completed <= self.updates:
            raise ValueError('Invalid completed-update cursor')
        next_sha = None if completed == self.updates else hashlib.sha256(
            self.batch(completed).astype('<i8').tobytes()).hexdigest()
        return dict(plan_sha256=self.fingerprint, completed_updates=completed, next_batch_sha256=next_sha)

    def validate_cursor(self, value, completed):
        if value != self.cursor(completed):
            raise ValueError('Sampler plan, cursor or next batch changed on resume')

    def chunks(self, start, end=None, *, size=32):
        end = self.updates if end is None else end
        if not 0 <= start <= end <= self.updates or self.batch_size % size:
            raise ValueError('Invalid prefetch bounds/chunk size')
        for update in range(start, end):
            indices = self.batch(update)
            for offset in range(0, self.batch_size, size):
                yield indices[offset:offset + size].tolist()


class PlanChunks:
    """Pickleable DataLoader batch_sampler; workers cannot advance a saved cursor."""
    def __init__(self, plan, start=0, end=None, size=32):
        self.plan, self.start, self.end, self.size = plan, start, end, size

    def __iter__(self):
        return self.plan.chunks(self.start, self.end, size=self.size)

    def __len__(self):
        end = self.plan.updates if self.end is None else self.end
        return (end - self.start) * self.plan.batch_size // self.size
