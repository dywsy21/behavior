import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_handoff_exposure import training_exposure, first_difference


def example():
    candidates=[]; requests=[]; feedback=[]; processed=[]; tokens=[]
    for i in range(6):
        identity=dict(original_candidate_index=i, task=i, split='train')
        p=dict(previous_intent='old', active_skills_text='new' if i%2 else 'old', task_name=str(i),
               outcome_supervision_mask=False, low_action_supervision_mask=False)
        candidates.append(dict(identity=identity, projection=p))
        requests.append(dict(sample_id=str(i), original_identity=identity, original_projection=p, source_group=str(i)))
        feedback.append(dict(sample_id=str(i), original_identity=identity, source_group=str(i),
                             execution_feedback=json.dumps(dict(estimated_bundle_outcome='SUCCEEDED'))))
        processed.append(dict(sample_id=str(i), identity=dict(candidate=i)))
        tokens.append(dict(task_name=str(i), memory_weight=.25, supervised_tokens=21,
                           supervised_field_tokens=dict(outcome_target=0, task_complete=0,
                                                       active_skills_semantic_json=5, memory_update=12, next_decision=2)))
    batch=dict(rows=[['expert',i] for i in range(6)]+[['new',0],['new',1]])
    schedule=dict(schedule=[batch])
    update=dict(update=1,real_samples=8,expert_examples=6,recovery_examples=2,world_size=8,
                loss=.1,grad_norm=.5,denominator=100.)
    audit=dict(status='passed',processed=processed,token_checks=tokens,training_serving_prefix_equal=True)
    fit=dict(status='completed_finite_schedule',step=1,planned_updates=1,
             frozen_before_sha256='a',frozen_after_sha256='a')
    return [schedule,copy.deepcopy(schedule),candidates,requests,feedback,audit,[update],fit]


class ExposureTests(unittest.TestCase):
    def test_complete_actual_exposure_and_weighted_token_mass(self):
        args=example(); before=copy.deepcopy(args)
        summary,rows=training_exposure(*args)
        self.assertEqual(summary['expert_exposures'],6)
        self.assertEqual(summary['recovery_exposures'],2)
        self.assertEqual(summary['learned_feedback_weighted_token_mass']['handoff']['memory_update'],9)
        self.assertEqual(summary['learned_feedback_weighted_token_mass']['handoff']['structural'],6)
        self.assertEqual(args,before); self.assertEqual(len(rows),6)

    def test_requests_may_be_in_independent_file_order(self):
        args=example(); args[3].reverse(); args[4].reverse()
        self.assertEqual(training_exposure(*args),training_exposure(*example()))

    def test_partial_or_nonfinite_fit_rejected(self):
        for change in ('missing','nan','denominator','step','frozen'):
            args=example()
            if change=='missing': args[6]=[]
            if change=='nan': args[6][0]['grad_norm']=float('nan')
            if change=='denominator': args[6][0]['denominator']=float('nan')
            if change=='step': args[6][0]['update']=2
            if change=='frozen': args[7]['frozen_after_sha256']='b'
            with self.assertRaises(ValueError): training_exposure(*args)

    def test_foreign_duplicate_eval_or_misaligned_rows_rejected(self):
        for change in ('eval','duplicate','identity','tokens','control','unmixed'):
            args=example()
            if change=='eval': args[2][0]['identity']['split']='eval'
            if change=='duplicate': args[3].append(copy.deepcopy(args[3][0]))
            if change=='identity': args[4][0]=dict(args[4][0],original_identity={'other':1})
            if change=='tokens': args[5]['token_checks'].reverse()
            if change=='control': args[1]['schedule'][0]['rows'].reverse()
            if change=='unmixed': args[2][0]['identity']['task']=1
            with self.assertRaises(ValueError): training_exposure(*args)

    def test_mask_and_weight_cannot_change(self):
        for key,value in (('memory_weight',1.),('supervised_tokens',1)):
            args=example();args[5]['token_checks'][0][key]=value
            with self.assertRaises(ValueError):training_exposure(*args)
        args=example();args[5]['token_checks'][0]['supervised_field_tokens']['outcome_target']=1
        with self.assertRaises(ValueError):training_exposure(*args)

    def test_first_divergent_token_not_sequence_average(self):
        self.assertEqual(first_difference([1,2,3,4],[1,2,9,4]),(2,3,9))
        self.assertIsNone(first_difference([1,2],[1,2]))
        self.assertIsNone(first_difference([1,2],[1,2,3]))


if __name__=='__main__':unittest.main()
