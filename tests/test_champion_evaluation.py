from collections import Counter
import types
import unittest

import numpy as np

from behavior_champion.evaluation import FixedEvalPlan


def data():
    episodes = [dict(split='eval', phase=task % 16, row=dict(task_index=task, episode_index=task,
        task_instance_id=100, raw_episode_id=task + 1000)) for task in range(100)]
    def locate(index):
        task, local = divmod(int(index), 32)
        return episodes[task], task % 16 + 16 * local
    return types.SimpleNamespace(split='eval', episodes=episodes, offsets=np.arange(101) * 32,
                                 locate=locate, excluded_episode_counts={'train': 105, 'eval': 6})


class EvalTests(unittest.TestCase):
    def test_equal_task_counts_correct_model_grouping_and_fixed_resume(self):
        mapping = {task: (task + 23) % 100 for task in range(100)}
        plan = FixedEvalPlan(data(), mapping)
        self.assertEqual(plan.fingerprint, FixedEvalPlan(data(), mapping).fingerprint)
        self.assertEqual(set(Counter(row['task'] for row in plan.identities).values()), {16})
        self.assertEqual(len({row['index'] for row in plan.identities}), 1600)
        for row in plan.identities:
            self.assertEqual(row['group'], 'old50' if mapping[row['task']] < 50 else 'new50')
            self.assertEqual(row['frame'] % 16, row['phase'])
        for group in ('old50', 'new50'):
            batches = list(plan.batches(group))
            self.assertEqual(len(batches), 25)
            self.assertTrue(all(len(batch) == 32 for batch in batches))

    def test_train_or_missing_task_mapping_is_rejected(self):
        bad = data(); bad.split = 'train'
        with self.assertRaises(ValueError): FixedEvalPlan(bad, {task: task for task in range(100)})
        with self.assertRaises(ValueError): FixedEvalPlan(data(), {task: task for task in range(99)})


if __name__ == '__main__': unittest.main()
