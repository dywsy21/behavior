from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import audit_trajectory_codec as audit
import native_trajectory_codec as codec


class AuditTests(unittest.TestCase):
    def test_sources_are_train_only_stable_and_unique(self):
        m = {'shards': [{'task': t, 'instance': i, 'episode': t * 100 + i, 'samples': 1,
                        'split': 'train' if i < 5 else 'test'} for t in range(5) for i in range(7)]}
        selected = audit.select_sources(m)
        self.assertEqual(len(selected), 20)
        self.assertTrue(all(s['split'] == 'train' for s in selected))
        m['shards'].reverse()
        self.assertEqual(selected, audit.select_sources(m))

    def test_sampling_is_bounded_and_includes_both_ends(self):
        for n in (1, 2, 599, 600, 601, 3000):
            got = audit.sampled_rows(list(range(n)))
            self.assertEqual(len(got), min(n, 600))
            self.assertEqual((got[0], got[-1]), (0, n - 1))
            self.assertEqual(len(set(got)), len(got))

    def test_wrong_identity_or_split_is_rejected(self):
        shard = {'task': 3, 'instance': 42, 'episode': 650}
        row = {'frame_index': 32, 'id': 't3_i42_e650_f000032', 'source_split': 'train',
               'task_index': 3, 'task_instance_id': 42, 'episode_index': 650, 'timestamp_s': 32 / 30}
        audit.check_row(row, shard)
        for k, v in [('source_split', 'test'), ('task_instance_id', 99), ('frame_index', 33), ('timestamp_s', 0.)]:
            with self.assertRaises(ValueError): audit.check_row({**row, k: v}, shard)

    def test_dense_comparator_reconstructs_all_quantized_controls(self):
        state = np.zeros(61)
        a = np.random.default_rng(41).uniform(-.7, .7, (16, 23))
        target = audit.dense_target(a, state)
        obj = codec.parse(target)
        self.assertTrue(all(len(obj[k]) == 16 for k in codec.GROUP_WIDTHS))
        errors = codec.errors(a, codec.decode(target, codec.current_anchor(state)))
        self.assertLessEqual(errors['joint_rad'], .00005 + 1e-12)
        self.assertLessEqual(errors['base_normalized'], .0005 + 1e-12)
        self.assertLessEqual(errors['gripper_command'], .0005 + 1e-12)


if __name__ == '__main__': unittest.main()
