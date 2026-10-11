from collections import Counter
import types
import unittest

import numpy as np

from behavior_champion.sampling import BalancedPlan, PlanChunks


def dataset(tasks=100, anchors=80):
    return types.SimpleNamespace(split='train', episodes=[dict(split='train', phase=t % 16,
        row=dict(task_index=t, episode_index=t, raw_episode_id=t, task_instance_id=1,
                 length=anchors * 16 + t % 16)) for t in range(tasks)],
        offsets=np.arange(tasks + 1) * anchors)


class SamplingTests(unittest.TestCase):
    def test_every_batch_all_tasks_no_repeats_fair_rotation(self):
        plan = BalancedPlan(dataset(), updates=25)
        counts = Counter(); all_indices = []
        for step in range(25):
            indices = plan.batch(step); per_task = Counter(indices // 80)
            self.assertEqual(set(per_task), set(range(100)))
            self.assertEqual(sorted(per_task.values()), [2] * 44 + [3] * 56)
            counts.update(per_task); all_indices.extend(indices.tolist())
        self.assertEqual(set(counts.values()), {64})
        self.assertEqual(len(set(all_indices)), 25 * 256)

    def test_resume_not_advanced_by_prefetch(self):
        plan = BalancedPlan(dataset(), updates=25)
        prefetched = iter(PlanChunks(plan)); [next(prefetched) for _ in range(63)]
        cursor = plan.cursor(2)
        restored = BalancedPlan(dataset(), updates=25)
        restored.validate_cursor(cursor, 2)
        got = np.concatenate(list(PlanChunks(restored, start=2, end=3)))
        np.testing.assert_array_equal(got, plan.batch(2))
        self.assertEqual(len(PlanChunks(restored, start=2, end=3)), 8)
        with self.assertRaises(ValueError): restored.validate_cursor(cursor, 3)
        with self.assertRaises(ValueError): BalancedPlan(dataset(), updates=25, seed=18).validate_cursor(cursor, 2)
        changed = dataset(); changed.episodes[1]['phase'] = 8
        with self.assertRaises(ValueError): BalancedPlan(changed, updates=25).validate_cursor(cursor, 2)

    def test_fail_closed_on_exhaustion_or_nontrain(self):
        with self.assertRaises(ValueError): BalancedPlan(dataset(anchors=10), updates=25)
        data = dataset(); data.split = 'eval'
        with self.assertRaises(ValueError): BalancedPlan(data)
        plan = BalancedPlan(dataset(), updates=25)
        self.assertIsNone(plan.cursor(25)['next_batch_sha256'])
        with self.assertRaises(IndexError): plan.batch(25)


if __name__ == '__main__': unittest.main()
