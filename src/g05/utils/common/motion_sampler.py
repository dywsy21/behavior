"""Complete, task-equal motion/recovery strata with disjoint DDP source shards."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from .task_event_sampler import (ResumableDistributedTaskEventBatchSampler,
                                 canonical_ranges, _integer)


class ResumableDistributedMotionBatchSampler(ResumableDistributedTaskEventBatchSampler):
    def __init__(self, dataset, *, sampling_index, low_quotas_override=None, **kwargs):
        raw = json.loads(Path(sampling_index).read_text()) if isinstance(sampling_index, (str, Path)) else deepcopy(sampling_index)
        if raw.get("schema_version") != 2:
            raise ValueError("Motion/recovery sampler requires schema 2")
        self.quotas = {k: _integer(v) for k, v in
                       (raw["low_quotas"] if low_quotas_override is None else low_quotas_override).items()}
        if set(self.quotas) != {"yaw_stop", "yaw_start_reverse", "gripper", "recovery", "other"}:
            raise ValueError("All motion/recovery sampling strata must be explicit")
        if any(v < 1 for v in self.quotas.values()):
            raise ValueError("No motion/recovery stratum may be silently removed")
        self.high_recovery_period = _integer(raw.get("high_recovery_period", 4))
        if self.high_recovery_period < 2:
            raise ValueError("Both original and recovery high-level supervision are required")
        # Reuse the mature full-view/branch/rank checks with a lossless two-pool
        # projection. The original schema stays version 2 in saved provenance.
        compatible = deepcopy(raw)
        compatible["schema_version"] = 1
        for task in compatible["tasks"]:
            task["critical_low_ranges"] = [pair for k, ranges in task["low_strata"].items()
                                            if k != "other" for pair in ranges]
            task["other_low_ranges"] = task["low_strata"]["other"]
        super().__init__(dataset, sampling_index=compatible, critical_per_batch=1, **kwargs)
        if sum(self.quotas.values()) != self.low_per_batch:
            raise ValueError("Motion quotas must exactly fill the low-level batch")
        pools = []
        for task in raw["tasks"]:
            if set(task["low_strata"]) != set(self.quotas):
                raise ValueError("Task has missing or unknown motion strata")
            high_original = [_integer(i) for i in task["high_original"]]
            high_recovery = [_integer(i) for i in task["high_recovery"]]
            if sorted(high_original + high_recovery) != sorted(task["high"]):
                raise ValueError("High original/recovery pools must partition all high rows")
            pool = {k: self._make_pool(ranges=canonical_ranges(r, len(dataset)))
                    for k, r in task["low_strata"].items()}
            pool.update(high_original=self._make_pool(indices=high_original),
                        high_recovery=self._make_pool(indices=high_recovery))
            if any(p["size"] < self._global_replicas for p in pool.values()):
                raise ValueError("Every task/stratum needs at least one source row per DDP rank")
            pools.append(pool)
        self._task_pools = pools
        self.index_metadata["schema_version"] = 2
        self.index_metadata["low_quotas"] = dict(self.quotas)

    def _streams(self):
        generator = torch.Generator().manual_seed(self.seed + 1000003 * self.epoch)
        original = self._make_task_stream("high_original", generator)
        recovery = self._make_task_stream("high_recovery", generator)
        low_streams = {k: self._make_task_stream(k, generator) for k in self.quotas}

        def high():
            step = 0
            while True:
                # All ranks execute the same branch pattern; source rows stay
                # rank-owned through the common task stream implementation.
                stream = recovery if step % self.high_recovery_period == self.high_recovery_period - 1 else original
                yield next(stream)
                step += 1

        def low():
            while True:
                for kind, quota in self.quotas.items():
                    for _ in range(quota):
                        yield next(low_streams[kind])
        return generator, high(), low()
