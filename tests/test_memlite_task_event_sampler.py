from collections import Counter
from copy import deepcopy
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from g05.utils.common.task_event_sampler import (
    ResumableDistributedTaskEventBatchSampler,
    branch_fingerprint,
    canonical_ranges,
    build_memlite_train_sampler,
)
from g05.utils.common.dist import ResumableDistributedBranchBalancedBatchSampler


class View:
    is_training_set = True
    is_rank_sharded = False
    use_weight_for_sampling = False

    def __init__(self, tasks=2, high=4, critical=8, other=48):
        self.dataset_dirs = ['/fake_memlite_dataset']
        self._memlite_sidecar = SimpleNamespace(path='/fake_memlite_sidecar.parquet')
        self._active_episode_indices = list(range(tasks))
        self.tasks = []
        self.lookup = {}
        width = high + critical + other + 10
        for task in range(tasks):
            start = task * width
            first = start + high + 3
            second = first + critical + 2
            row = {'task_id': task, 'high': list(range(start, start+high)),
                   'critical_low_ranges': [(first, first+critical)],
                   'other_low_ranges': [(second, second+other)]}
            self.tasks.append(row)
            for kind, values in (('high', row['high']), ('critical', range(first, first+critical)),
                                 ('other', range(second, second+other))):
                for index in values:
                    self.lookup[index] = (task, kind)
        self.size = width * tasks

    def __len__(self):
        return self.size

    def get_memlite_branch_sampling_spec(self):
        return {'high': [x for row in self.tasks for x in row['high']],
                'low_ranges': [x for row in self.tasks
                               for x in row['critical_low_ranges'] + row['other_low_ranges']]}

    def index(self):
        return {'schema_version': 1, 'dataset_length': len(self),
                'branch_fingerprint': branch_fingerprint(self.get_memlite_branch_sampling_spec(), len(self)),
                'view_identity': {'dataset_roots': [[str(Path(x).resolve()) for x in self.dataset_dirs]],
                                  'sidecar_paths': [str(Path(self._memlite_sidecar.path).resolve())],
                                  'active_episodes': [self._active_episode_indices.copy()]},
                'tasks': deepcopy(self.tasks)}


def make(view, **kwargs):
    options = {'batch_size': 8, 'num_replicas': 2, 'rank': 0,
               'high_fraction': .125, 'critical_per_batch': 1, 'seed': 17}
    options.update(kwargs)
    return ResumableDistributedTaskEventBatchSampler(view, sampling_index=view.index(), **options)


def test_each_rank_batch_has_exact_branch_and_event_quota():
    view = View()
    for rank in range(2):
        for batch in make(view, rank=rank):
            assert len(batch) == 8
            assert Counter(view.lookup[x][1] for x in batch) == {'high': 1, 'critical': 1, 'other': 6}


def test_four_rank_5000_schedule_task_equal_and_low_no_replacement():
    view = View(tasks=5, high=3000, critical=5000, other=40000)
    counts = Counter()
    low_seen, all_by_rank = set(), []
    for rank in range(4):
        rank_seen = set()
        for batch in itertools.islice(make(view, num_replicas=4, rank=rank), 5000):
            for index in batch:
                task, kind = view.lookup[index]
                counts[task, kind] += 1
                rank_seen.add(index)
                if kind != 'high':
                    assert index not in low_seen
                    low_seen.add(index)
        all_by_rank.append(rank_seen)
    for a in range(4):
        for b in range(a):
            assert all_by_rank[a].isdisjoint(all_by_rank[b])
    for task in range(5):
        assert counts[task, 'high'] == 4000
        assert counts[task, 'critical'] == 4000
        assert counts[task, 'other'] == 24000
    assert len(low_seen) == 140000


def test_oversampling_cycles_only_inside_rank_owned_pool():
    view = View(high=2, critical=4, other=48)
    draws = []
    for rank in range(2):
        rows = list(itertools.chain.from_iterable(make(view, rank=rank, batch_size=4)))
        assert len(rows) > len(set(rows))
        draws.append(set(rows))
    assert draws[0].isdisjoint(draws[1])


@pytest.mark.parametrize('shuffle', [True, False])
def test_resume_preserves_every_remaining_batch_and_full_epoch_len(shuffle):
    sampler = make(View(), shuffle=shuffle)
    first = list(sampler)
    assert list(sampler) == first
    sampler.set_start_batch(3)
    assert list(sampler) == first[3:]
    assert len(sampler) == len(first)
    sampler.set_start_batch(len(first) + 1)
    assert list(sampler) == []
    sampler.set_start_batch(0)
    sampler.set_epoch(2)
    if shuffle:
        assert list(sampler) != first


def test_input_index_is_not_mutated_or_retained_by_reference():
    view = View()
    index = view.index()
    copy = deepcopy(index)
    sampler = ResumableDistributedTaskEventBatchSampler(view, sampling_index=index,
        batch_size=8, num_replicas=2, rank=0)
    expected = list(sampler)
    assert index == copy
    index['tasks'][0]['high'][0] = 999999
    assert list(sampler) == expected


@pytest.mark.parametrize('critical', [0, 7, 8, 1.5, True])
def test_reject_invalid_critical_quota(critical):
    with pytest.raises(ValueError):
        make(View(), critical_per_batch=critical)


@pytest.mark.parametrize('change', ['length', 'fingerprint', 'schema', 'root', 'split', 'sidecar',
                                   'overlap', 'missing_low', 'duplicate_high', 'duplicate_task', 'empty_critical'])
def test_reject_bad_or_wrong_view_index(change):
    view = View()
    index = view.index()
    if change == 'length': index['dataset_length'] += 1
    elif change == 'fingerprint': index['branch_fingerprint'] = 'wrong'
    elif change == 'schema': index['schema_version'] = 2
    elif change == 'root': index['view_identity']['dataset_roots'] = [['/different_data']]
    elif change == 'split': index['view_identity']['active_episodes'] = [[999]]
    elif change == 'sidecar': index['view_identity']['sidecar_paths'] = ['/other_labels']
    elif change == 'overlap': index['tasks'][1]['other_low_ranges'] = index['tasks'][0]['other_low_ranges']
    elif change == 'missing_low':
        a, b = index['tasks'][0]['other_low_ranges'][0]
        index['tasks'][0]['other_low_ranges'][0] = (a+1, b)
    elif change == 'duplicate_high': index['tasks'][1]['high'] = index['tasks'][0]['high']
    elif change == 'duplicate_task': index['tasks'][1]['task_id'] = index['tasks'][0]['task_id']
    elif change == 'empty_critical': index['tasks'][0]['critical_low_ranges'] = []
    with pytest.raises(ValueError):
        ResumableDistributedTaskEventBatchSampler(view, sampling_index=index,
                                                 batch_size=8, num_replicas=2, rank=0)


@pytest.mark.parametrize('flag', ['is_rank_sharded', 'use_weight_for_sampling', 'eval'])
def test_reject_unsupported_data_views(flag):
    view = View()
    if flag == 'eval': view.is_training_set = False
    else: setattr(view, flag, True)
    with pytest.raises(ValueError):
        make(view)


def test_reject_task_pool_smaller_than_world_size():
    with pytest.raises(ValueError, match='smaller than DDP'):
        make(View(tasks=3, high=3), num_replicas=4)


def test_fingerprint_invariant_to_order_and_adjacent_segmentation():
    assert branch_fingerprint({'high': [1, 0], 'low_ranges': [(5,7), (7,10)]}, 12) == \
           branch_fingerprint({'high': [0, 1], 'low_ranges': [(5,10)]}, 12)


@pytest.mark.parametrize('ranges', [[(-1,3)], [(4,4)], [(1,5), (4,6)], [(1,99)], [(1.0,2)]])
def test_range_validation(ranges):
    with pytest.raises(ValueError):
        canonical_ranges(ranges, 10)


def test_high_low_overlap_is_not_a_valid_fingerprint():
    with pytest.raises(ValueError, match='overlap'):
        branch_fingerprint({'high': [3], 'low_ranges': [(2,4)]}, 10)


def test_factory_default_remains_exact_legacy_sampler():
    result = build_memlite_train_sampler(View(), sampling_config={},
        batch_size=8, num_replicas=2, rank=0)
    assert type(result) is ResumableDistributedBranchBalancedBatchSampler


@pytest.mark.parametrize('config', [{'strategy': 'typo'}, {'strategy': 'task_event'}])
def test_factory_rejects_ambiguous_or_unindexed_strategy(config):
    with pytest.raises(ValueError):
        build_memlite_train_sampler(View(), sampling_config=config,
            batch_size=8, num_replicas=2, rank=0)


def test_factory_reads_exact_index_file(tmp_path):
    import json
    view = View()
    path = tmp_path / 'sampling.json'
    path.write_text(json.dumps(view.index()))
    result = build_memlite_train_sampler(view,
        sampling_config={'strategy': 'task_event', 'sampling_index_path': str(path)},
        batch_size=8, num_replicas=2, rank=0)
    assert isinstance(result, ResumableDistributedTaskEventBatchSampler)
