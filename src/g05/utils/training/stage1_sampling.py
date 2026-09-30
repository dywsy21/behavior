"""Exact, task-mixed passes for the single-node MEM-Lite stage-1 recipe.

The schedule contains each eligible observation once, not once per noise draw.
Every rank's microbatch contains >=2 tasks. The last two updates are repacked
when necessary: unused slots are NOT fetched, forwarded, or optimized. Thus a
short epoch tail changes its denominator, never duplicates a training example.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


class MixedTaskPass:
    version = "mixed-task-exact-pass-v1"

    def __init__(self, task_ids, *, seed=17, epoch=0, world_size=8,
                 micro_batch=4, accumulation=8):
        self.tasks = np.asarray(task_ids, dtype=np.int32)
        self.world = int(world_size)
        self.micro = int(micro_batch)
        self.accumulation = int(accumulation)
        self.global_batch = self.world * self.micro * self.accumulation
        if self.tasks.ndim != 1 or self.global_batch != 256:
            raise ValueError("Stage 1 requires a 1-D candidate table and global batch 256")
        if self.world != 8 or (self.micro, self.accumulation) not in ((4, 8), (32, 1)):
            raise ValueError("Only the admitted 8x4x8 high / 8x32 low recipes are supported")
        n = len(self.tasks)
        if n < 256 or len(np.unique(self.tasks)) < 2:
            raise ValueError("An exact mixed pass needs >=256 candidates from >=2 tasks")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
        order = rng.permutation(n)
        self.updates = math.ceil(n / 256)
        # Groups of four are the smallest admitted microbatch. Build both
        # recipes from the same slot sequence; each low micro contains 8 groups.
        lengths = np.full(self.updates * 64, 4, dtype=np.int32)
        remainder = n % 256
        if remainder:
            # At least 257 real observations fill the last 128 groups, giving
            # every group 2..4 real rows. No rank has an empty forward/backward.
            tail = 256 + remainder
            lengths[-128:] = tail // 128
            lengths[-128:-128 + tail % 128 or None] += 1
        offsets = np.r_[0, np.cumsum(lengths)]
        if offsets[-1] != n or lengths.min() < 2 or lengths.max() > 4:
            raise AssertionError("Internal exact-tail allocation error")
        # Vectorized scan of the full part, small explicit scan of the tail.
        full = (self.updates - (2 if remainder else 0)) * 64
        block = self.tasks[order[:full * 4]].reshape(-1, 4)
        bad = list(np.flatnonzero(np.all(block == block[:, :1], axis=1)))
        bad.extend(g for g in range(full, len(lengths))
                   if len(np.unique(self.tasks[order[offsets[g]:offsets[g + 1]]])) < 2)
        for group in bad:
            a, b = offsets[group:group + 2]
            if len(np.unique(self.tasks[order[a:b]])) >= 2:
                continue
            wanted_not = self.tasks[order[a]]
            repaired = False
            # A swap is legal only if BOTH groups remain mixed. Never silently
            # oversample a minority task to make an impossible recipe pass.
            for distance in range(1, len(lengths)):
                donor = (group + distance) % len(lengths)
                c, d = offsets[donor:donor + 2]
                for pos in range(c, d):
                    if self.tasks[order[pos]] == wanted_not:
                        continue
                    trial = self.tasks[order[c:d]].copy()
                    trial[pos - c] = wanted_not
                    if len(np.unique(trial)) < 2:
                        continue
                    order[a], order[pos] = order[pos], order[a]
                    repaired = True
                    break
                if repaired:
                    break
            if not repaired:
                raise ValueError("Task proportions cannot satisfy exact microbatch mixing")
        slots = np.full((len(lengths), 4), -1, dtype=np.int64)
        for column in range(4):
            valid = lengths > column
            slots[valid, column] = order[offsets[:-1][valid] + column]
        self.slots = slots.reshape(self.updates, self.accumulation, self.world, self.micro)
        self.digest = hashlib.sha256(self.slots.tobytes()).hexdigest()
        self.epoch = int(epoch)
        self.real_count = n

    def micro_indices(self, update, accumulation, rank):
        row = self.slots[update, accumulation, rank]
        return row[row >= 0].tolist()

    def update_count(self, update):
        return int((self.slots[update] >= 0).sum())

    def audit(self):
        flat = self.slots[self.slots >= 0]
        if not np.array_equal(np.sort(flat), np.arange(self.real_count)):
            raise AssertionError("A candidate was duplicated or omitted")
        minimum = min(len(np.unique(self.tasks[row[row >= 0]]))
                      for row in self.slots.reshape(-1, self.micro))
        if minimum < 2:
            raise AssertionError("A rank microbatch contains a single task")
        return dict(version=self.version, epoch=self.epoch, candidates=self.real_count,
                    updates=self.updates, schedule_sha256=self.digest,
                    minimum_tasks_per_microbatch=minimum, duplicates=0, omitted=0,
                    final_update_counts=[self.update_count(i)
                                         for i in range(max(0, self.updates - 2), self.updates)])


class CommittedBatchSampler:
    """Prefetch may run ahead; only the trainer owns the committed update cursor."""

    def __init__(self, schedule, rank, next_update=0):
        if not 0 <= rank < schedule.world or not 0 <= next_update <= schedule.updates:
            raise ValueError("Invalid rank or committed resume cursor")
        self.schedule, self.rank, self.next_update = schedule, rank, next_update

    def __iter__(self):
        for update in range(self.next_update, self.schedule.updates):
            for accumulation in range(self.schedule.accumulation):
                yield self.schedule.micro_indices(update, accumulation, self.rank)

    def __len__(self):
        return (self.schedule.updates - self.next_update) * self.schedule.accumulation
