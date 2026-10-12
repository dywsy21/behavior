import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_handoff_exposure import (training_exposure, first_difference, choice_cache_index,
                                      preprocess_planner_example, diagnostic_model_identity,
                                      validate_diagnostic_checkpoint_step, ENGINEERING_PARENT_SHA)


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
    def test_engineering_parent_cannot_be_substituted_for_fixed_model_pair(self):
        common=dict(purpose='ENGINEERING_PARENT_ONLY_NOT_FINAL360_COMPARISON',
                    checkpoint_sha256=ENGINEERING_PARENT_SHA,optimizer_updates=0)
        validate_diagnostic_checkpoint_step(360,360,{})
        validate_diagnostic_checkpoint_step(48045,48045,common)
        for actual,expected in ((48045,360),(360,48045),(1,1)):
            with self.assertRaises(ValueError):validate_diagnostic_checkpoint_step(actual,expected,common)
        for key,value in (('purpose','comparison'),('checkpoint_sha256','different'),('optimizer_updates',1)):
            wrong=dict(common);wrong[key]=value
            with self.assertRaises(ValueError):validate_diagnostic_checkpoint_step(48045,48045,wrong)

    def test_readonly_ce_keeps_profile_but_hashes_both_groups(self):
        from types import SimpleNamespace
        model=SimpleNamespace(training=False,coordination_trainable_parameter_groups=lambda:{'planner_vlm':[0]*326},
                              parameters=lambda:[SimpleNamespace(grad=None)])
        calls=[]
        def hashes(actual,*,frozen):
            self.assertIs(actual,model);calls.append(frozen);return str(frozen)
        before=diagnostic_model_identity(model,hashes)
        self.assertEqual(calls,[False,True]);self.assertEqual(before['groups'],{'planner_vlm':326})
        self.assertEqual(before,diagnostic_model_identity(model,hashes))
        changed=diagnostic_model_identity(model,lambda _,frozen:'different' if frozen else str(frozen))
        self.assertNotEqual(before['sha256'],changed['sha256'])

    def test_disabled_profile_training_mode_or_gradient_refused(self):
        from types import SimpleNamespace
        for groups,training,grad in (({},False,None),({'planner_vlm':[0]*326},True,None),
                                     ({'planner_vlm':[0]*326},False,1)):
            model=SimpleNamespace(training=training,coordination_trainable_parameter_groups=lambda:groups,
                                  parameters=lambda:[SimpleNamespace(grad=grad)])
            with self.assertRaises(ValueError):diagnostic_model_identity(model,lambda _,frozen:'hash')

    def test_target_survives_processor_consuming_and_mutating_raw_metadata(self):
        class Processor:
            def preprocess(self,raw):
                projection=raw.pop('model_projection'); projection['memory']['value']='changed'
                return {'samples':projection}
        raw={'model_projection':{'memory':{'value':'original'}}}; identity={'frame':128}
        target, prepared=preprocess_planner_example(Processor(),raw,identity)
        self.assertEqual(target,{'memory':{'value':'original'}})
        self.assertEqual(prepared['samples']['memory']['value'],'changed')
        prepared['source_identity']['frame']=0
        self.assertEqual(identity,{'frame':128})

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

    def test_first_choice_uses_shifted_mask_not_raw_field_offset(self):
        labels=[-100,-100,7,8,-100,2,3,4]
        mask=[v!=-100 for v in labels[1:]]
        actual=choice_cache_index([2,3,4],[2,9,4],5,labels,mask)
        self.assertEqual(actual,dict(field_token_offset=1,full_token_position=6,
                                    ce_cache_index=3,target_token=3,repeated_token=9))
        with self.assertRaises(ValueError):choice_cache_index([2,3,4],[2,9,4],4,labels,mask)
        with self.assertRaises(ValueError):choice_cache_index([2,3,4],[2,9,4],5,labels,[True]*7)


if __name__=='__main__':unittest.main()
