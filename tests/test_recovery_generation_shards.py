import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_corpus import digest
from recovery_generation_shards import generation_shard, merge_generation_shards


class GenerationShardTests(unittest.TestCase):
    def fixture(self):
        schedule = [('boundary', i) for i in range(5)] + [('recovery', i) for i in range(2)]
        common = dict(schedule_sha256=digest(schedule), checkpoint_sha256='weights', source_commit='source',
                      config_sha256='config', target_free=True)
        parts = []
        for index in range(3):
            selected, sharding = generation_shard(schedule, index, 3)
            manifest = dict(common, sharding=sharding)
            result = dict(manifest, status='completed', seconds=10)
            rows = [dict(global_schedule_index=i, scheduled_sample_index=sample, split=split,
                         identity=dict(task=sample), score=dict(exact=sample % 2 == 0))
                    for i, split, sample in selected]
            parts.append((manifest, result, rows))
        return parts

    def test_strided_partition_is_complete_ordered_and_does_not_change_rows(self):
        parts = self.fixture()
        old = copy.deepcopy(parts)
        receipt, rows, seconds = merge_generation_shards(parts[::-1])
        self.assertEqual([r['global_schedule_index'] for r in rows], list(range(7)))
        self.assertEqual(parts, old)
        self.assertEqual(seconds, [10, 10, 10])
        self.assertTrue(receipt['sharding']['merged'])

    def test_invalid_empty_or_duplicate_schedule_refused(self):
        for index, count in ((-1, 2), (2, 2), (0, 0), (True, 2), (0, True), (0, 3)):
            with self.assertRaises(ValueError): generation_shard([('a', 0), ('a', 1)], index, count)
        for schedule in ([], [('a', 0), ('a', 0)]):
            with self.assertRaises(ValueError): generation_shard(schedule)

    def test_missing_extra_or_duplicate_shards_refused(self):
        parts = self.fixture()
        for changed in (parts[:-1], parts+parts[:1], [parts[0], parts[0], parts[2]]):
            with self.assertRaises(ValueError): merge_generation_shards(changed)

    def test_foreign_weight_config_source_or_schedule_refused(self):
        for field in ('checkpoint_sha256', 'source_commit', 'config_sha256', 'schedule_sha256'):
            parts = self.fixture()
            parts[1][0][field] = 'foreign'
            with self.assertRaises(ValueError): merge_generation_shards(parts)

    def test_incomplete_receipt_or_changed_rows_refused(self):
        mutations = [lambda p: p[0][1].update(status='running'),
                     lambda p: p[0][1].update(checkpoint_sha256='foreign'),
                     lambda p: p[0][2].pop(),
                     lambda p: p[0][2].reverse(),
                     lambda p: p[0][2][0].update(scheduled_sample_index=999),
                     lambda p: p[0][2][0].update(split='foreign')]
        for change in mutations:
            parts = self.fixture(); change(parts)
            with self.assertRaises(ValueError): merge_generation_shards(parts)

    def test_frozen_model_digest_checked_for_every_shard_and_retained(self):
        parts = self.fixture()
        for manifest, result, _ in parts:
            manifest['model_before_sha256'] = 'before'
            result.update(model_before_sha256='before', model_after_sha256='before')
        receipt, _, _ = merge_generation_shards(parts)
        self.assertEqual(receipt['model_before_sha256'], receipt['model_after_sha256'])
        bad = copy.deepcopy(parts); bad[1][1].pop('model_after_sha256')
        with self.assertRaises(ValueError): merge_generation_shards(bad)
        bad = copy.deepcopy(parts); bad[2][1]['model_after_sha256'] = 'changed'
        with self.assertRaises(ValueError): merge_generation_shards(bad)


if __name__ == '__main__':
    unittest.main()
