import copy
import json
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code')]
from recovery_corpus import digest
from recovery_handoff_data import feedback_overlay, HandoffFeedbackPool, validate_handoff_audit
from recovery_handoff_feedback import requests_for_candidate
import test_recovery_handoff_feedback as fixtures


class HandoffDataTests(unittest.TestCase):
    def example(self, outcome='SUCCEEDED', confidence=.96):
        row, frames = fixtures.HandoffTests().example()
        request, _ = requests_for_candidate(row, frames, excluded_groups=set())
        p = row['projection']; feedback = json.loads(p['execution_feedback'])
        feedback['estimated_bundle_outcome'] = outcome
        feedback['estimated_member_outcomes'] = [dict(member=0, estimated_outcome=outcome, confidence=confidence)]
        prediction = dict(sample_id=request['sample_id'], source_group=request['source_group'],
            original_identity=copy.deepcopy(row['identity']), execution_feedback=json.dumps(feedback),
            original_execution_feedback=p['execution_feedback'], target_projection_sha256=digest(p),
            known_previous_outcome='UNKNOWN', physical_outcome_asserted=False, new_human_admission=False)
        return row, request, prediction

    def test_overlay_changes_input_only_and_preserves_every_teacher_field(self):
        row, request, predicted = self.example(); before = copy.deepcopy((row, request, predicted))
        index, actual = feedback_overlay(row, request, predicted)
        self.assertEqual(index, row['identity']['original_candidate_index'])
        self.assertEqual({k for k in actual if actual[k] != row['projection'][k]}, {'execution_feedback'})
        self.assertEqual((row, request, predicted), before)
        self.assertFalse(actual['outcome_supervision_mask']); self.assertFalse(actual['low_action_supervision_mask'])
        self.assertEqual(actual['known_previous_outcome'], 'UNKNOWN')

    def test_failed_or_unknown_predictions_are_not_filtered_for_a_switch_target(self):
        for outcome, confidence in (('SUCCEEDED', .97), ('FAILED', .96), ('UNKNOWN', 0.)):
            row, request, predicted = self.example(outcome, confidence)
            _, actual = feedback_overlay(row, request, predicted)
            self.assertIn('PLACE_ON', actual['active_skills_text'])
            self.assertEqual(json.loads(actual['execution_feedback'])['estimated_bundle_outcome'], outcome)

    def test_cross_task_frame_group_or_projection_rejected(self):
        for field in ('task', 'frame', 'episode', 'instance'):
            row, request, predicted = self.example(); predicted['original_identity'][field] += 1
            with self.assertRaises(ValueError): feedback_overlay(row, request, predicted)
        for field, value in (('sample_id', 'other'), ('source_group', 'other:1'),
                             ('target_projection_sha256', 'f'*64), ('original_execution_feedback', '{}')):
            row, request, predicted = self.example(); predicted[field] = value
            with self.assertRaises(ValueError): feedback_overlay(row, request, predicted)

    def test_causal_clock_and_truth_cannot_be_overwritten(self):
        for field, value in (('same_intent_controls', 999), ('attempt_index', 99),
                             ('source', 'oracle'), ('stalled_is_not_failed', False)):
            row, request, predicted = self.example()
            feedback = json.loads(predicted['execution_feedback']); feedback[field] = value
            predicted['execution_feedback'] = json.dumps(feedback)
            with self.assertRaises(ValueError): feedback_overlay(row, request, predicted)
        for field, value in (('physical_outcome_asserted', True), ('known_previous_outcome', 'SUCCEEDED')):
            row, request, predicted = self.example(); predicted[field] = value
            with self.assertRaises(ValueError): feedback_overlay(row, request, predicted)

    def test_invalid_unready_confidence_and_unreviewed_classes_refused(self):
        for outcome, confidence in (('UNKNOWN', -1.), ('SUCCEEDED', .84), ('FAILED', float('nan')),
                                    ('SUCCEEDED', 1.2), ('IN_PROGRESS', .98)):
            row, request, predicted = self.example(outcome, confidence)
            with self.assertRaises(ValueError): feedback_overlay(row, request, predicted)

    def test_production_agreed_gated_unknown_retains_nonzero_raw_confidence(self):
        from g05.utils.memlite_causal_feedback import CausalFeedbackLedger, FeedbackIdentity
        from g05.utils.memlite_skill_protocol import canonical_json, parse_active_skills_text
        row, request, predicted = self.example()
        p = row['projection']; identity = FeedbackIdentity('test',p['task_name'],1,'ep','a'*64,'b'*64)
        ledger = CausalFeedbackLedger(identity,uncertainty_protocol='unready_zero_confidence_v1')
        ledger.issued(identity,128,canonical_json(parse_active_skills_text(p['previous_intent'])),p['previous_parent_goal'])
        for at, conf in ((240,.61),(256,.72)):
            ledger.observe_clock(identity,at)
            ledger.estimated(identity,at,[dict(outcome='FAILED',confidence=conf)],calibrated=True)
        actual = json.loads(ledger.projection(identity,256))
        feedback = json.loads(predicted['execution_feedback'])
        for key in ('estimated_bundle_outcome','estimated_member_outcomes'): feedback[key]=actual[key]
        predicted['execution_feedback']=json.dumps(feedback)
        _, overlaid = feedback_overlay(row,request,predicted)
        member=json.loads(overlaid['execution_feedback'])['estimated_member_outcomes'][0]
        self.assertEqual(member['estimated_outcome'],'UNKNOWN'); self.assertEqual(member['confidence'],.61)

    def test_overlay_copy_no_cross_sample_memory_and_unselected_unchanged(self):
        row, request, predicted = self.example(); index, projection = feedback_overlay(row, request, predicted)
        pool = HandoffFeedbackPool.__new__(HandoffFeedbackPool); pool.by_index = {index: projection}
        a = pool.projection(index, row['projection']); a['memory'] = 'foreign'
        self.assertEqual(pool.projection(index, row['projection'])['memory'], row['projection']['memory'])
        self.assertEqual(pool.projection(index+1, row['projection']), row['projection'])
        changed = copy.deepcopy(row['projection']); changed['previous_intent'] = 'foreign'
        with self.assertRaises(ValueError): pool.projection(index, changed)

    def test_old_partial_foreign_or_supervised_audit_cannot_launch(self):
        ticket = dict(source_commit='source', files=dict(recipe=dict(sha256='recipe'),
            handoff_feedback=dict(sha256='feedback')))
        recipe = dict(H1=dict(periodic_teacher=dict(candidate_sha256='candidate',
            learned_handoff=dict(admission_sha256='admission'))))
        good = dict(schema='handoff_planner_processor_audit_v1',status='passed',
            source_commit='source',recipe_sha256='recipe',handoff_admission_sha256='admission',
            feedback_sha256='feedback',candidate_sha256='candidate',original_targets_reconstructed=1800,
            learned_feedback_rows=383,tokenized_rows=383,reviewed_current_images_matched=480,
            original_observations_match=True,estimates_recomputed_from_logits=True,
            training_serving_prefix_equal=True,same_periodic_control_schedule=True,
            outcome_tokens_supervised=0,physical_truth_added=0,low_action_supervision=False,
            model_forwards=0,optimizer_updates=0)
        validate_handoff_audit(good,ticket,recipe)
        for key, value in [('schema','periodic_planner_processor_audit_v1'),('source_commit','old'),
                           ('recipe_sha256','old'),('handoff_admission_sha256','old'),('feedback_sha256','foreign'),
                           ('tokenized_rows',382),('reviewed_current_images_matched',0),
                           ('training_serving_prefix_equal',False),('same_periodic_control_schedule',False),
                           ('outcome_tokens_supervised',1),('physical_truth_added',1),('low_action_supervision',True),
                           ('optimizer_updates',1)]:
            with self.assertRaises(ValueError): validate_handoff_audit(dict(good,**{key:value}),ticket,recipe)


if __name__ == '__main__': unittest.main()
