from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_admission import independent_candidates, local_file, verify_approval
from recovery_corpus import file_sha


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'review.json').write_text('{"fixture_only":true}\n')
        self.row = dict(sample_id='s', source_episode=['run','episode'], task='t', source_group='t:1',
            split='train', control_step=32, label_status='candidate_pending_semantic_review',
            actor_input=dict(issued_skills_semantic_json=json.dumps([dict(verb='GRASP')]), parent_goal='g'),
            label_audit=dict(full_executed_32_step_target_available=True, outcome=dict(value='SUCCEEDED')))
        self.approval = dict(sample_id='s', pool='outcome', reviewer='unit-test-fixture-not-real-approval', event_id='e',
            source_group='t:1', reviewed_start=16, reviewed_end=64,
            evidence=[dict(path='review.json',sha256=file_sha(self.root/'review.json'),kind=kind) for kind in
                      ('original_media_review','physical_semantic_review')],
            label=dict(value='SUCCEEDED',member_index=0,available_control_step=32,evidence_end_control_step=32))

    def test_future_label_and_wrong_member_rejected(self):
        verify_approval(self.approval,self.row,self.root)
        for k,v in [('available_control_step',33),('evidence_end_control_step',33),('member_index',1)]:
            item=deepcopy(self.approval);item['label'][k]=v
            with self.assertRaises(ValueError):verify_approval(item,self.row,self.root)

    def test_independent_cohort_metadata_is_strict_and_dev_outcome_only(self):
        approved=dict(self.approval,usage_role='calibration',cohort_sha256='a'*64)
        dev=dict(self.row,split='dev')
        verify_approval(approved,dev,self.root)
        for change in [dict(usage_role='training'),dict(cohort_sha256='invalid'),dict(pool='action')]:
            with self.assertRaises(ValueError):verify_approval(dict(approved,**change),dev,self.root)
        with self.assertRaises(ValueError):verify_approval(approved,dict(dev,split='train'),self.root)
        del approved['cohort_sha256']
        with self.assertRaises(ValueError):verify_approval(approved,dev,self.root)

    def test_outcome_approval_is_not_bc_approval(self):
        item=deepcopy(self.approval);item['pool']='action'
        with self.assertRaises(ValueError):verify_approval(item,self.row,self.root)
        item['label']=dict(quality='verified_correct_execution',executed_controls=32)
        verify_approval(item,self.row,self.root)
        item['reviewed_end']=63
        with self.assertRaises(ValueError):verify_approval(item,self.row,self.root)
        item['reviewed_end']=64
        row=deepcopy(self.row);row['label_audit']['full_executed_32_step_target_available']=False
        with self.assertRaises(ValueError):verify_approval(item,row,self.root)

    def test_protected_and_quarantined_do_not_pass(self):
        for field,value in [('split','protected'),('label_status','quarantined_binding_mismatch')]:
            with self.assertRaises(ValueError):verify_approval(self.approval,dict(self.row,**{field:value}),self.root)

    def test_evidence_must_match_both_media_and_physics(self):
        item=deepcopy(self.approval);item['evidence']=item['evidence'][:1]
        with self.assertRaises(ValueError):verify_approval(item,self.row,self.root)
        (self.root/'review.json').write_text('changed')
        with self.assertRaises(ValueError):verify_approval(self.approval,self.row,self.root)
        with self.assertRaises(ValueError):local_file(self.root,'../escaped.json')

    def test_contiguous_success_frames_are_not_many_events(self):
        rows=[dict(self.row,control_step=k,sample_id=str(k)) for k in (16,32,48,96,112)]
        spans=independent_candidates(rows)
        self.assertEqual([s['anchors'] for s in spans],[3,2])
        self.assertFalse(any(s['verified_recovery'] for s in spans))
        rows[-1]['source_episode']=['run','another-episode']
        self.assertEqual(len(independent_candidates(rows)),3)

    def test_dense_four_control_images_are_not_independent_events(self):
        rows=[dict(self.row,control_step=k,sample_id=str(k)) for k in (16,20,24,28,32,96)]
        self.assertEqual([s['anchors'] for s in independent_candidates(rows)],[5,1])

    def test_predecision_result_requires_exact_old_attempt_evidence(self):
        approval=deepcopy(self.approval);approval['label']['history_role']='predecision'
        with self.assertRaises(ValueError):verify_approval(approval,self.row,self.root)
        row=deepcopy(self.row)
        row['label_audit']['predecision_outcome']=dict(value='SUCCEEDED',context_id='old',evidence_end_control_step=32)
        verify_approval(approval,row,self.root)
        approval['label']['value']='FAILED'
        with self.assertRaises(ValueError):verify_approval(approval,row,self.root)


if __name__ == '__main__':unittest.main()
