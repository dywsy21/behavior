import unittest

import numpy as np

from behavior_champion.training_io import combine_chunks, EvalAccumulator


def chunk(indices, split='train'):
    return dict(identity=[dict(index=i, split=split, task=i % 100, frame=16*i+3,
                              phase=3, length=20000) for i in indices], state61=np.zeros((len(indices), 61)))


class TrainingIOTests(unittest.TestCase):
    def test_prefetch_order_and_phase_are_checked_not_only_set_membership(self):
        parts = [chunk([0, 1]), chunk([2, 3])]
        source, audit = combine_chunks(parts, [0, 1, 2, 3], split='train')
        self.assertEqual(source['state61'].shape, (4, 61))
        self.assertEqual(audit['task_counts'][:4], [1, 1, 1, 1])
        with self.assertRaises(ValueError): combine_chunks(parts, [1, 0, 2, 3], split='train')
        parts[1]['identity'][0]['split'] = 'eval'
        with self.assertRaises(ValueError): combine_chunks(parts, [0, 1, 2, 3], split='train')
        parts[1]['identity'][0]['split'] = 'train'
        parts[1]['identity'][0]['phase'] = 7
        with self.assertRaises(ValueError): combine_chunks(parts, [0, 1, 2, 3], split='train')

    def test_eval_exact_task_and_group_means(self):
        accumulator = EvalAccumulator({str(i): i for i in range(100)}, per_task=2)
        tasks = np.repeat(np.arange(100), 2)
        accumulator.add(np.arange(200), tasks, {'fm': tasks.astype(float)})
        result = accumulator.finish()
        self.assertEqual(result['all']['fm'], 49.5)
        self.assertEqual(result['old50']['fm'], 24.5)
        self.assertEqual(result['new50']['fm'], 74.5)
        self.assertEqual(result['task_007']['observations'], 2)
        self.assertEqual(result['task_007']['fm'], 7)

    def test_partial_eval_scalar_broadcast_duplicate_and_nonfinite_rejected(self):
        accumulator = EvalAccumulator({str(i): i for i in range(100)}, per_task=2)
        with self.assertRaises(ValueError): accumulator.add([0], [0], {'fm': np.array(2.)})
        with self.assertRaises(ValueError): accumulator.add([0], [0], {'fm': np.array([np.nan])})
        accumulator.add([0], [0], {'fm': np.array([1.])})
        with self.assertRaises(ValueError): accumulator.finish()
        with self.assertRaises(ValueError): accumulator.add([0], [0], {'fm': np.array([1.])})


if __name__ == '__main__': unittest.main()
