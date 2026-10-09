from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from sign_reviewed_grasp_outcomes import verify_causal_grasp


class GraspReviewTests(unittest.TestCase):
    def setUp(self):
        self.item = dict(sample_id='fixture', step=16, source_group='task:1', event_id='event',
            members=[dict(verb='GRASP', target='can_1', entity='can.n.01_1',
                          evidence_arms=['right'], value='SUCCEEDED')])
        self.review = dict(decision='approve_outcome_only', label='SUCCEEDED', notes='Test fixture only',
                           member_index=0, target='can_1', arm='right')
        self.evidence = dict(sample_id='fixture', label_anchor=16, source_group='task:1',
            independent_event_conservative='event', future_physics_used=False, members=self.item['members'],
            causal_physics=[dict(control_step=k, physical_audit=dict(grasp_states={
                'can.n.01_1': dict(right='TRUE', left='FALSE')})) for k in range(10,16)])

    def test_exact_owner_decision_only(self):
        verify_causal_grasp(self.evidence, self.item, self.review)
        for key, value in [('decision','approve_action'), ('label','FAILED'), ('target','can_2'),
                           ('arm','left'), ('member_index',-1), ('notes','')]:
            with self.assertRaises(ValueError):
                verify_causal_grasp(self.evidence, self.item, dict(self.review, **{key:value}))

    def test_no_future_or_skipped_causal_control(self):
        for position, step in [(5,16), (0,9)]:
            evidence=deepcopy(self.evidence);evidence['causal_physics'][position]['control_step']=step
            with self.assertRaises(ValueError):verify_causal_grasp(evidence,self.item,self.review)
        with self.assertRaises(ValueError):
            verify_causal_grasp(dict(self.evidence,future_physics_used=True),self.item,self.review)

    def test_one_failed_grasp_or_different_event_rejects(self):
        evidence=deepcopy(self.evidence)
        evidence['causal_physics'][3]['physical_audit']['grasp_states']['can.n.01_1']['right']='FALSE'
        with self.assertRaises(ValueError):verify_causal_grasp(evidence,self.item,self.review)
        with self.assertRaises(ValueError):
            verify_causal_grasp(dict(self.evidence,independent_event_conservative='other'),self.item,self.review)


if __name__ == '__main__':unittest.main()
