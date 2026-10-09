from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_evaluation_partition import build_partition, rows_for_purpose, validate_partition


def row(group, split):
    return dict(candidate=dict(source_group=group, split=split))


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.old = [row('a:1', 'train'), row('b:2', 'dev')]
        self.new = [row('c:3', 'dev')]
        self.spec = dict(new_independent_groups=['c:3'], no_added_training_rows_expected=True)
        self.partition = build_partition(self.spec, [self.old, self.new])

    def test_default_training_excludes_frozen_test_but_keeps_old_dev(self):
        self.assertEqual(rows_for_purpose(self.old + self.new, self.partition, 'training'), self.old)
        self.assertEqual(rows_for_purpose(self.old + self.new, self.partition, 'frozen_evaluation'), self.new)
        self.assertEqual(rows_for_purpose(self.old + self.new, self.partition, 'feature_extraction'), self.old+self.new)

    def test_no_overlap_added_train_or_unassigned_new_group(self):
        for rows in [[row('c:3', 'train')], [row('d:4', 'dev')], [row('b:2', 'dev')]]:
            with self.assertRaises(ValueError):
                build_partition(self.spec, [self.old, rows])
        changed = dict(self.spec, new_independent_groups=['b:2'])
        with self.assertRaisesRegex(ValueError, 'previously exposed'):
            build_partition(changed, [self.old, [row('b:2', 'dev')]])

    def test_legacy_is_not_frozen_test_and_tampered_roles_fail(self):
        self.assertEqual(rows_for_purpose(self.old, None, 'training'), self.old)
        with self.assertRaises(ValueError):
            rows_for_purpose(self.old, None, 'frozen_evaluation')
        changed = deepcopy(self.partition); changed['training_groups'].append('c:3')
        with self.assertRaises(ValueError):
            validate_partition(changed, self.old + self.new)
        with self.assertRaises(ValueError):
            validate_partition(self.partition, [row('c:3', 'train')])


if __name__ == '__main__':
    unittest.main()
