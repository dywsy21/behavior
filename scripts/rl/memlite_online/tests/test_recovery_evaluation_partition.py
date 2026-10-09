from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_evaluation_partition import build_partition, rows_for_purpose, validate_partition
from recovery_corpus import file_sha
from recovery_sft_data import require_training_pool


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

    def independent(self, *, reserve=False):
        spec = dict(schema='recovery_evaluation_partition_spec_v2',
            calibration_groups=['c:3'], frozen_test_groups=['z:9'],
            reserved_frozen_test_groups=['z:9'] if reserve else [],
            no_added_training_rows_expected=True, calibration_may_not_select_checkpoints=True,
            cohort_sha256='a'*64)
        cal = [dict(row('c:3','dev'), approval=dict(usage_role='calibration',cohort_sha256='a'*64))]
        test = [] if reserve else [row('z:9','dev')]
        return spec, cal, test

    def test_calibration_never_trains_selects_or_supplies_test(self):
        spec, cal, test = self.independent()
        partition = build_partition(spec,[self.old,cal,test]); rows=self.old+cal+test
        self.assertEqual(rows_for_purpose(rows,partition,'training'),self.old)
        self.assertEqual(rows_for_purpose(rows,partition,'calibration'),cal)
        self.assertEqual(rows_for_purpose(rows,partition,'frozen_evaluation'),test)
        self.assertEqual(rows_for_purpose(rows,partition,'feature_extraction'),rows)

    def test_reserved_test_cannot_be_read_or_used_to_drop_denominator(self):
        spec, cal, _ = self.independent(reserve=True)
        partition = build_partition(spec,[self.old,cal])
        self.assertEqual(partition['frozen_test_groups'],['z:9'])
        with self.assertRaises(ValueError): validate_partition(partition,[row('z:9','dev')])
        with self.assertRaises(ValueError): build_partition(spec,[self.old,cal+[row('z:9','dev')]])
        with self.assertRaises(ValueError): build_partition(spec,[self.old,[]])
        transport = build_partition(dict(spec,base_unit_count=0),[cal])
        self.assertEqual(rows_for_purpose(cal,transport,'training'),[])

    def test_calibration_role_sha_overlap_duplicate_and_missing_contract_fail(self):
        spec,cal,test=self.independent();partition=build_partition(spec,[self.old,cal,test])
        for change in [dict(calibration_groups=['a:1']),dict(calibration_groups=['c:3','c:3']),
                       dict(calibration_groups=['z:9']),dict(calibration_may_not_select_checkpoints=False),
                       dict(cohort_sha256='b'*64),dict(base_unit_count=-1)]:
            with self.assertRaises(ValueError):build_partition(dict(spec,**change),[self.old,cal,test])
        with self.assertRaises(ValueError):build_partition(None,[self.old,cal])
        with self.assertRaises(ValueError):rows_for_purpose(cal,None,'training')
        with self.assertRaises(ValueError):rows_for_purpose(self.old,self.partition,'calibration')
        for change in [dict(calibration_may_not_select_checkpoints=False),dict(cohort_sha256='b'*64),
                       dict(calibration_groups=['c:3','c:3']),dict(training_groups=['a:1','c:3'])]:
            with self.assertRaises(ValueError):validate_partition(dict(partition,**change),self.old+cal+test)

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

    def test_real_loader_does_not_count_test_rows_as_training_gate_support(self):
        # Structural fixture only; no real trajectory or semantic approval.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [row(f't:{i}', 'train') for i in range(8)]
            dev = [row(f'd:{i}', 'dev') for i in range(4)]
            test = [row(f'f:{i}', 'dev') for i in range(3)]
            for items in (train, dev, test):
                for i, item in enumerate(items):
                    item['approval'] = dict(pool='outcome', event_id=item['candidate']['source_group'],
                                            label=dict(value=('IN_PROGRESS','SUCCEEDED','FAILED')[i%3]))
            spec = dict(new_independent_groups=[r['candidate']['source_group'] for r in test],
                        no_added_training_rows_expected=True)
            def write_release(old_dev):
                partition = build_partition(spec, [train+old_dev, test])
                (root/'evaluation_partition.json').write_text(json.dumps(partition))
                (root/'outcome.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in train+old_dev+test))
                receipt = dict(schema='recovery_admission_v1',
                    pools=dict(outcome=dict(training_ready=True, blockers=[])),
                    files={name:file_sha(root/name) for name in ('outcome.jsonl','evaluation_partition.json')},
                    evaluation_partition_file='evaluation_partition.json')
                (root/'admission.json').write_text(json.dumps(receipt))
                return file_sha(root/'admission.json')
            sha = write_release(dev)
            self.assertEqual(require_training_pool(root,'outcome',sha)[1],train+dev)
            self.assertEqual(require_training_pool(root,'outcome',sha,purpose='frozen_evaluation')[1],test)
            sha = write_release(dev[:1])
            with self.assertRaisesRegex(ValueError,'cannot satisfy'):
                require_training_pool(root,'outcome',sha)

    def test_calibration_rows_cannot_rescue_inadequate_selection_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train=[row(f't:{i}','train') for i in range(8)]
            dev=[row('d:0','dev')]
            cal=[row(f'c:{i}','dev') for i in range(8)]
            for items in (train,dev,cal):
                for i,item in enumerate(items):
                    item['approval']=dict(pool='outcome',event_id=item['candidate']['source_group'],
                        label=dict(value=('IN_PROGRESS','SUCCEEDED','FAILED')[i%3]))
            spec=dict(schema='recovery_evaluation_partition_spec_v2',no_added_training_rows_expected=True,
                calibration_groups=[r['candidate']['source_group'] for r in cal],
                frozen_test_groups=['z:9'],reserved_frozen_test_groups=['z:9'],
                calibration_may_not_select_checkpoints=True,cohort_sha256='a'*64)
            part=build_partition(spec,[train+dev,cal])
            (root/'evaluation_partition.json').write_text(json.dumps(part))
            (root/'outcome.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in train+dev+cal))
            receipt=dict(schema='recovery_admission_v1',pools=dict(outcome=dict(training_ready=True,blockers=[])),
                files={n:file_sha(root/n) for n in ('evaluation_partition.json','outcome.jsonl')},
                evaluation_partition_file='evaluation_partition.json')
            (root/'admission.json').write_text(json.dumps(receipt));sha=file_sha(root/'admission.json')
            self.assertEqual(require_training_pool(root,'outcome',sha,purpose='calibration')[1],cal)
            with self.assertRaisesRegex(ValueError,'cannot satisfy'):require_training_pool(root,'outcome',sha)


if __name__ == '__main__':
    unittest.main()
