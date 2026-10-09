import unittest

from g05.utils.training.recovery_runtime import schedule_fingerprint, shard_finite_batch


class RecoveryRuntimeTests(unittest.TestCase):
    def test_unequal_tails_no_supervised_duplicates(self):
        for count in (1, 3, 8, 10, 33, 63, 64, 65):
            rows = list(range(count))
            shards = [shard_finite_batch(rows, rank=r, world_size=8, micro_batch=4) for r in range(8)]
            self.assertEqual(len({len(x) for x in shards}), 1)
            actual = [row for shard in shards for micro in shard if not micro['dummy'] for row in micro['rows']]
            self.assertEqual(sorted(actual), rows)
            self.assertEqual(sum(m['real_count'] for shard in shards for m in shard), count)
            self.assertTrue(all(m['real_count'] or len(m['rows']) == 1 for s in shards for m in s))

    def test_fingerprint_changes_on_data_world_and_order(self):
        binding = dict(component='L0', source_commit='a'*40, parent_sha256='b'*64,
                       admission_sha256='c'*64, recipe_sha256='d'*64, world_size=8, micro_batch=4)
        schedule = [dict(rows=[['new', 0], ['expert', 12]], event_pass=0)]
        old = schedule_fingerprint(schedule, binding=binding)
        self.assertEqual(old, schedule_fingerprint(schedule, binding=dict(reversed(list(binding.items())))))
        for key, value in (('world_size', 4), ('component', 'H0'), ('admission_sha256', 'e'*64)):
            self.assertNotEqual(old, schedule_fingerprint(schedule, binding=dict(binding, **{key:value})))
        self.assertNotEqual(old, schedule_fingerprint([dict(rows=list(reversed(schedule[0]['rows'])), event_pass=0)], binding=binding))
        for key, value in (('parent_sha256', 'missing'), ('source_commit', 'main'), ('world_size', True)):
            with self.assertRaises(ValueError):
                schedule_fingerprint(schedule, binding=dict(binding, **{key:value}))

    def test_invalid_shards(self):
        for rows, rank, world, micro in (([], 0, 8, 4), ([1], 8, 8, 4), ([1], 0, 8, 0)):
            with self.assertRaises(ValueError):
                shard_finite_batch(rows, rank=rank, world_size=world, micro_batch=micro)


if __name__ == '__main__':
    unittest.main()
