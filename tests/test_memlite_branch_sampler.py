from g05.utils.common.dist import ResumableDistributedBranchBalancedBatchSampler


class _Dataset:
    def __init__(self):
        self.is_rank_sharded = False

    def __len__(self):
        return 41

    def get_memlite_branch_sampling_spec(self):
        return {"high": [1, 9, 17], "low_ranges": [(20, 41)]}


def test_memlite_sampler_guarantees_both_branches_on_each_ddp_rank():
    dataset = _Dataset()
    samplers = [
        ResumableDistributedBranchBalancedBatchSampler(
            dataset,
            batch_size=8,
            num_replicas=2,
            rank=rank,
            high_fraction=0.25,
            seed=7,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        batches = list(sampler)
        assert batches
        assert all(len(batch) == 8 for batch in batches)
        for batch in batches:
            high = sum(index in {1, 9, 17} for index in batch)
            assert high == 2
            assert len(batch) - high == 6


def test_memlite_sampler_shared_shuffle_then_rank_shards_are_disjoint():
    class _LargeDataset:
        def __len__(self):
            return 120

        def get_memlite_branch_sampling_spec(self):
            return {"high": [1, 3, 5, 7], "low_ranges": [(20, 120)]}

    samplers = [
        ResumableDistributedBranchBalancedBatchSampler(
            _LargeDataset(),
            batch_size=8,
            num_replicas=2,
            rank=rank,
            high_fraction=0.25,
            shuffle=True,
            seed=11,
        )
        for rank in range(2)
    ]
    batches = [list(sampler) for sampler in samplers]
    high_pool = {1, 3, 5, 7}
    high_by_rank = [
        {index for batch in rank_batches for index in batch if index in high_pool}
        for rank_batches in batches
    ]
    low_by_rank = [
        [index for batch in rank_batches for index in batch if index not in high_pool]
        for rank_batches in batches
    ]
    # Planner rows can repeat inside a rank, but never leak to a different rank.
    assert high_by_rank[0].isdisjoint(high_by_rank[1])
    # Action rows are a no-replacement stream for this epoch and DDP-sharded.
    assert len(low_by_rank[0]) == len(set(low_by_rank[0]))
    assert len(low_by_rank[1]) == len(set(low_by_rank[1]))
    assert set(low_by_rank[0]).isdisjoint(low_by_rank[1])


def test_memlite_sampler_unshuffled_low_stream_strides_by_world_size():
    dataset = _Dataset()
    rank_zero = ResumableDistributedBranchBalancedBatchSampler(
        dataset, batch_size=8, num_replicas=2, rank=0, high_fraction=0.25, shuffle=False
    )
    rank_one = ResumableDistributedBranchBalancedBatchSampler(
        dataset, batch_size=8, num_replicas=2, rank=1, high_fraction=0.25, shuffle=False
    )
    high_pool = {1, 9, 17}
    first_low_zero = [x for x in next(iter(rank_zero)) if x not in high_pool]
    first_low_one = [x for x in next(iter(rank_one)) if x not in high_pool]
    assert first_low_zero == [20, 22, 24, 26, 28, 30]
    assert first_low_one == [21, 23, 25, 27, 29, 31]


def test_memlite_sampler_is_epoch_deterministic_and_resumable():
    sampler = ResumableDistributedBranchBalancedBatchSampler(
        _Dataset(), batch_size=8, num_replicas=1, rank=0, high_fraction=0.125, seed=123
    )
    first = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == first
    sampler.set_start_batch(1)
    assert list(sampler) == first[1:]
    # ResumableDistributedSampler's existing finetune contract keeps the full
    # epoch length while __iter__ skips the checkpoint prefix.
    assert len(sampler) == len(first)
    sampler.set_start_batch(0)
    sampler.set_epoch(1)
    second = list(sampler)
    assert second != first
    assert all(sum(index in {1, 9, 17} for index in batch) == 1 for batch in second)


def test_memlite_sampler_rejects_missing_branch_pool():
    class _Bad:
        def __len__(self):
            return 8

        def get_memlite_branch_sampling_spec(self):
            return {"high": [1], "low_ranges": []}

    try:
        ResumableDistributedBranchBalancedBatchSampler(
            _Bad(), batch_size=2, num_replicas=1, rank=0
        )
    except ValueError as exc:
        assert "non-empty high and low" in str(exc)
    else:
        raise AssertionError("missing low branch must fail before training")


def test_memlite_sampler_rejects_high_pool_smaller_than_world_size():
    class _TooFewHigh:
        def __len__(self):
            return 20

        def get_memlite_branch_sampling_spec(self):
            return {"high": [1], "low_ranges": [(2, 20)]}

    try:
        ResumableDistributedBranchBalancedBatchSampler(
            _TooFewHigh(), batch_size=2, num_replicas=2, rank=0
        )
    except ValueError as exc:
        assert "smaller than the number of DDP ranks" in str(exc)
    else:
        raise AssertionError("rank-disjoint high pool must fail before training")
