import copy
import json
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code')]
from recovery_handoff_feedback import requests_for_candidate
from recovery_handoff_inference import feedback_from_trace, tree_sha, unique_checks
import test_recovery_handoff_feedback as requests_fixture
import test_recovery_periodic_teacher as original_fixture


class HandoffInferenceTests(unittest.TestCase):
    def case(self):
        original, frames = requests_fixture.HandoffTests().example()
        row, _ = requests_for_candidate(original,frames,excluded_groups=set())
        _, episode = original_fixture.PeriodicTeacherTests().candidate()
        calibration = dict(temperature=1.774,high_sha256='a'*64,observer_sha256='b'*64)
        return row,episode,calibration

    def trace(self,row,values=None):
        return [dict(r,logits=(values or [[0.,20.,0.,0.]]*2)[n]) for n,r in enumerate(row['requests'])]

    def test_success_changes_only_estimate_not_target_or_clock(self):
        row,ep,cal = self.case(); before=copy.deepcopy(row)
        out=json.loads(feedback_from_trace(row,ep,self.trace(row),cal))
        self.assertEqual(out['estimated_bundle_outcome'],'SUCCEEDED')
        self.assertGreater(out['estimated_member_outcomes'][0]['confidence'],.99)
        self.assertEqual(out['same_intent_controls'],128)
        self.assertEqual(row,before)
        self.assertEqual(row['original_projection']['known_previous_outcome'],'UNKNOWN')
        self.assertIn('PLACE_ON',row['original_projection']['active_skills_text'])

    def test_disagreement_and_uncertainty_stay_unknown(self):
        row,ep,cal=self.case()
        for values in ([[0.,20.,0.,0.],[0.,0.,20.,0.]], [[0.,1.,0.,0.]]*2):
            out=json.loads(feedback_from_trace(row,ep,self.trace(row,values),cal))
            self.assertEqual(out['estimated_bundle_outcome'],'UNKNOWN')

    def test_failed_prediction_preserved_even_for_teacher_handoff(self):
        row,ep,cal=self.case()
        out=json.loads(feedback_from_trace(row,ep,self.trace(row,[[0.,0.,20.,0.]]*2),cal))
        self.assertEqual(out['estimated_bundle_outcome'],'FAILED')
        self.assertIn('PLACE_ON',row['original_projection']['active_skills_text'])

    def test_missing_duplicate_foreign_or_future_predictions_rejected(self):
        row,ep,cal=self.case(); trace=self.trace(row)
        bad=copy.deepcopy(trace);bad[1]['checks'][-1]['frame']+=16
        other=copy.deepcopy(trace);other[0]['checks'][0]['task_name']='other task'
        extra=copy.deepcopy(trace);extra[0]['reward']=1
        for invalid in (trace[:1],trace+trace[:1],bad,other,extra):
            with self.assertRaises(ValueError): feedback_from_trace(row,ep,invalid,cal)

    def test_same_target_label_does_not_force_outcome(self):
        row,ep,cal=self.case()
        row['original_projection']['active_skills_text']=row['original_projection']['previous_intent']
        out=json.loads(feedback_from_trace(row,ep,self.trace(row),cal))
        self.assertEqual(out['estimated_bundle_outcome'],'SUCCEEDED')

    def test_two_overlapping_windows_need_five_distinct_observations(self):
        row,_,_=self.case()
        self.assertEqual(len(unique_checks(row)),5)
        self.assertEqual([c['frame'] for c in unique_checks(row)],[192,208,224,240,256])
        self.assertNotEqual(tree_sha(row),tree_sha(dict(row,source_group='other')))

    def test_nonzero_original_video_phase_not_used_as_control_time(self):
        row,ep,cal=self.case(); original=copy.deepcopy(row)
        ep['phase']=13
        for segment in ep['segments']:
            segment['start']+=13;segment['end']+=13
        row['original_identity']['frame']+=13;row['annotation_trace']['first_frame']+=13
        for request in row['requests']:
            for check in request['checks']: check['frame']+=13
        shifted=feedback_from_trace(row,ep,self.trace(row),cal)
        _,unshifted,_=self.case()
        self.assertEqual(shifted,feedback_from_trace(original,unshifted,self.trace(original),cal))

    def test_original_clock_corruption_is_not_overwritten(self):
        row,ep,cal=self.case()
        feedback=json.loads(row['original_projection']['execution_feedback'])
        feedback['attempt_index']+=1
        row['original_projection']['execution_feedback']=json.dumps(feedback)
        with self.assertRaises(ValueError): feedback_from_trace(row,ep,self.trace(row),cal)

    def test_pixel_dtype_shape_and_value_all_bound(self):
        import torch
        x=torch.zeros(3,16,16,dtype=torch.uint8)
        self.assertEqual(tree_sha({'x':x}),tree_sha({'x':x.clone()}))
        y=x.clone();y[1,1,1]=1
        for v in (y,x.to(torch.int32),x.reshape(16,16,3)):
            self.assertNotEqual(tree_sha(x),tree_sha(v))


if __name__ == '__main__': unittest.main()
