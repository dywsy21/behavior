import bisect
import itertools
import math
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    def __init__(self, *args, batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset_is_rank_sharded = bool(getattr(self.dataset, "is_rank_sharded", False))
        if self._dataset_is_rank_sharded:
            # Dataset already performed rank sharding, so sampler should operate in local mode.
            self.num_replicas = 1
            self.rank = 0
            self.num_samples = len(self.dataset)
            self.total_size = self.num_samples
        self.batch_size = batch_size
        self.start_batch_idx = 0  # per-rank dataloader batch index
        self._log_indices = False
        self._logger = None

    def set_start_batch(self, start_batch_idx: int):
        self.start_batch_idx = int(start_batch_idx)

    def __iter__(self):
        # super().__iter__() already returns the per-rank index stream (shuffled + padded)
        base_iter = super().__iter__()
        # Skip indices corresponding to already-consumed batches WITHOUT loading data
        skip = self.start_batch_idx * self.batch_size
        if skip > 0:
            base_iter = itertools.islice(base_iter, skip, None)
        return base_iter


class ResumableDistributedBranchBalancedBatchSampler(Sampler[List[int]]):
    """DDP-safe MEM-Lite batch sampler with a fixed high/low branch mix.

    Every *per-rank* batch contains at least one planner (``high``) and one
    action (``low``) row.  It deliberately oversamples the rare high-level
    boundaries rather than weakening MEM-Lite's fail-fast mixed-loss route.
    The dataset supplies individual high indices plus run-length encoded low
    ranges.  This keeps a 9M-frame sidecar to O(number of skill/gap ranges)
    Python memory, and the iterator is streaming rather than materializing an
    epoch's millions of Python lists.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        high_fraction: float = 0.125,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        low_only: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.high_fraction = float(high_fraction)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.low_only = bool(low_only)
        self.epoch = 0
        self.start_batch_idx = 0
        if self.batch_size < 2:
            raise ValueError("MEM-Lite mixed training requires batch_size >= 2")
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid DDP num_replicas/rank for branch-balanced sampler")
        if self.low_only and self.high_fraction != 0.0:
            raise ValueError("Explicit low-only training requires high_fraction=0")
        if not self.low_only and not 0.0 < self.high_fraction < 1.0:
            raise ValueError("high_fraction must be strictly between 0 and 1")
        if not hasattr(dataset, "get_memlite_branch_sampling_spec"):
            raise TypeError(
                "dataset must implement get_memlite_branch_sampling_spec() for MEM-Lite sampling"
            )
        spec = dataset.get_memlite_branch_sampling_spec()
        self._high_indices = [int(index) for index in spec.get("high", [])]
        self._low_ranges = [
            (int(start), int(end))
            for start, end in spec.get("low_ranges", [])
            if int(end) > int(start)
        ]
        if not self._high_indices or not self._low_ranges:
            raise ValueError(
                "MEM-Lite branch sampler requires non-empty high and low pools; "
                f"got high={len(self._high_indices)}, low_ranges={len(self._low_ranges)}"
            )
        self._dataset_is_rank_sharded = bool(getattr(dataset, "is_rank_sharded", False))
        self.effective_num_replicas = 1 if self._dataset_is_rank_sharded else self.num_replicas
        self.effective_rank = 0 if self._dataset_is_rank_sharded else self.rank
        # Round to the requested fraction but preserve both branches for every
        # local batch.  E.g. batch=8, fraction=.125 -> 1 high + 7 low.
        self.high_per_batch = 0 if self.low_only else min(
            self.batch_size - 1,
            max(1, int(round(self.batch_size * self.high_fraction))),
        )
        self.low_per_batch = self.batch_size - self.high_per_batch
        self._low_range_ends = []
        total = 0
        for start, end in self._low_ranges:
            total += end - start
            self._low_range_ends.append(total)
        self._low_total = total
        self._global_replicas = self.effective_num_replicas
        self._global_rank = self.effective_rank
        if len(self._high_indices) < self._global_replicas:
            raise ValueError(
                "MEM-Lite high pool is smaller than the number of DDP ranks; "
                "cannot keep high samples rank-disjoint while every local batch "
                f"contains one (high={len(self._high_indices)}, ranks={self._global_replicas}). "
                "Add planner boundaries or lower the DDP world size."
            )
        if self._low_total < self._global_replicas:
            raise ValueError(
                "MEM-Lite low pool is smaller than the number of DDP ranks; "
                "cannot keep low samples rank-disjoint "
                f"(low={self._low_total}, ranks={self._global_replicas})."
            )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch_idx: int):
        self.start_batch_idx = max(0, int(start_batch_idx))

    def _num_batches(self) -> int:
        per_rank_samples = math.ceil(len(self.dataset) / self.effective_num_replicas)
        if self.drop_last:
            requested = per_rank_samples // self.batch_size
        else:
            requested = math.ceil(per_rank_samples / self.batch_size)
        # Low/action examples do not repeat during an epoch.  Planner rows
        # intentionally may: their rare, rank-local pool is oversampled to
        # preserve one high sample in every mixed batch.
        # All ranks use the smallest shard capacity so DDP always executes the
        # same number of optimizer/eval collectives.
        low_capacity = (self._low_total // self._global_replicas) // self.low_per_batch
        batches = min(requested, low_capacity)
        if batches <= 0:
            raise ValueError(
                "MEM-Lite low pool cannot supply one rank-disjoint mixed batch; "
                f"low={self._low_total}, ranks={self._global_replicas}, "
                f"low_per_batch={self.low_per_batch}"
            )
        return batches

    def _rank_shard_size(self, total: int) -> int:
        """Size of this rank's ``rank::world_size`` shard of a global order."""
        remaining = total - self._global_rank
        return 0 if remaining <= 0 else 1 + (remaining - 1) // self._global_replicas

    @staticmethod
    def _shared_affine_permutation(
        total: int, generator: torch.Generator, shuffle: bool
    ) -> Tuple[int, int]:
        """Return a compact shared permutation of ``range(total)``.

        All ranks derive the same epoch order, then take ``rank::world_size``
        positions.  An affine bijection avoids allocating a 9M-element
        ``randperm`` tensor for the low-level sidecar pool.
        """
        if total <= 1 or not shuffle:
            return 1, 0
        offset = int(torch.randint(total, (1,), generator=generator).item())
        multiplier = int(torch.randint(1, total, (1,), generator=generator).item())
        while math.gcd(multiplier, total) != 1:
            multiplier = (multiplier + 1) % total
            if multiplier == 0:
                multiplier = 1
        return multiplier, offset

    def _rank_position(self, local_position: int, total: int) -> int:
        """Map a local cursor to a permanent rank-disjoint global shard.

        High rows may repeat only inside their owning rank after its unique
        shard is exhausted (the explicit rare-planner oversampling policy).
        Low rows are capacity-capped in ``_num_batches`` and hence cannot
        repeat during an epoch.
        """
        shard_size = self._rank_shard_size(total)
        if shard_size <= 0:
            raise RuntimeError("rank has no source rows in a required MEM-Lite branch")
        return self._global_rank + self._global_replicas * (local_position % shard_size)

    def _high_stream(self, affine: Tuple[int, int]):
        multiplier, offset = affine
        cursor = 0
        total = len(self._high_indices)
        while True:
            source_position = self._rank_position(cursor, total)
            yield self._high_indices[(multiplier * source_position + offset) % total]
            cursor += 1

    def _low_stream(self, affine: Tuple[int, int]):
        multiplier, offset = affine
        cursor = 0
        while True:
            source_position = self._rank_position(cursor, self._low_total)
            flat_offset = (multiplier * source_position + offset) % self._low_total
            range_idx = bisect.bisect_right(self._low_range_ends, flat_offset)
            prior = 0 if range_idx == 0 else self._low_range_ends[range_idx - 1]
            yield self._low_ranges[range_idx][0] + flat_offset - prior
            cursor += 1

    def _streams(self):
        # Deliberately rank-independent: DDP shuffles globally before slicing
        # into non-overlapping rank shards.
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1000003 * self.epoch)
        high_affine = self._shared_affine_permutation(
            len(self._high_indices), generator, self.shuffle
        )
        low_affine = self._shared_affine_permutation(
            self._low_total, generator, self.shuffle
        )
        return generator, self._high_stream(high_affine), self._low_stream(low_affine)

    def __iter__(self) -> Iterator[List[int]]:
        generator, high_stream, low_stream = self._streams()
        # Replaying the prefix is O(consumed batches), bounded by resume
        # progress, while preserving exact stream determinism without holding
        # the full epoch in memory.
        for _ in range(self.start_batch_idx):
            for _ in range(self.high_per_batch):
                next(high_stream)
            for _ in range(self.low_per_batch):
                next(low_stream)
            if self.shuffle:
                torch.randperm(self.batch_size, generator=generator)
        for _ in range(self.start_batch_idx, self._num_batches()):
            batch = [next(high_stream) for _ in range(self.high_per_batch)]
            batch.extend(next(low_stream) for _ in range(self.low_per_batch))
            if self.shuffle:
                batch = [batch[index] for index in torch.randperm(self.batch_size, generator=generator).tolist()]
            yield batch

    def __len__(self) -> int:
        # Finetune's resume loop compares its absolute checkpoint batch_idx
        # against DataLoader.__len__().  Keep the full-epoch contract here;
        # __iter__ alone removes the already-consumed prefix.
        return self._num_batches()


class ResumableDistributedGroupedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler for datasets with pre-defined group index ranges.
    It guarantees all indices within a batch come from the same group.

    Required dataset attribute:
        group_ranges: Dict[int, Tuple[int, int]], where each tuple is [start, end).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.start_batch_idx = 0

        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {self.batch_size}.")
        if self.num_replicas <= 0:
            raise ValueError(f"`num_replicas` must be > 0, got {self.num_replicas}.")
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"`rank` must be in [0, {self.num_replicas}), got {self.rank}.")
        if not hasattr(self.dataset, "group_ranges"):
            raise TypeError(
                "Dataset must expose `group_ranges: Dict[int, Tuple[int, int]]` for grouped batching."
            )
        self._dataset_is_rank_sharded = bool(getattr(self.dataset, "is_rank_sharded", False))
        if self._dataset_is_rank_sharded:
            self.effective_num_replicas = 1
            self.effective_rank = 0
        else:
            self.effective_num_replicas = self.num_replicas
            self.effective_rank = self.rank

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch_idx: int):
        self.start_batch_idx = int(start_batch_idx)

    def _build_group_batches(self) -> List[List[int]]:
        group_ranges: Dict[int, Tuple[int, int]] = getattr(self.dataset, "group_ranges")
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        group_batches: List[List[int]] = []
        for group_id in sorted(group_ranges):
            start, end = group_ranges[group_id]
            if end <= start:
                continue

            indices = list(range(start, end))
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]

            if self.drop_last:
                total_size = (
                    len(indices) // self.effective_num_replicas
                ) * self.effective_num_replicas
                if total_size <= 0:
                    continue
                indices = indices[:total_size]
            else:
                total_size = (
                    math.ceil(len(indices) / self.effective_num_replicas)
                    * self.effective_num_replicas
                )
                if total_size > len(indices):
                    pad_size = total_size - len(indices)
                    # If pad_size > len(indices), we need to repeat indices to fully pad to total_size.
                    pad = (indices * math.ceil(pad_size / len(indices)))[:pad_size]
                    indices = indices + pad

            rank_indices = indices[self.effective_rank : total_size : self.effective_num_replicas]
            for i in range(0, len(rank_indices), self.batch_size):
                batch = rank_indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                group_batches.append(batch)

        if self.shuffle and len(group_batches) > 1:
            perm = torch.randperm(len(group_batches), generator=generator).tolist()
            group_batches = [group_batches[i] for i in perm]
        return group_batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._build_group_batches()
        if self.start_batch_idx > 0:
            batches = batches[self.start_batch_idx :]
        yield from batches

    def __len__(self) -> int:
        group_ranges: Dict[int, Tuple[int, int]] = getattr(self.dataset, "group_ranges")
        total_batches = 0
        for start, end in group_ranges.values():
            n = max(0, end - start)
            if n == 0:
                continue
            if self.drop_last:
                total_size = (n // self.effective_num_replicas) * self.effective_num_replicas
                per_rank = total_size // self.effective_num_replicas
                total_batches += per_rank // self.batch_size
            else:
                total_size = (
                    math.ceil(n / self.effective_num_replicas) * self.effective_num_replicas
                )
                per_rank = total_size // self.effective_num_replicas
                total_batches += math.ceil(per_rank / self.batch_size)
        return max(0, total_batches - self.start_batch_idx)
