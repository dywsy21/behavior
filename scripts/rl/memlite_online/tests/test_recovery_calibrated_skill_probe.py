from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'code'),str(Path(__file__).parent)]
from causal_skill_probe import CausalSkillProbe
from causal_skill_probe_audit import audit_joint_episode
from recovery_calibrated_observer import GraspCalibrationBinding,BoundCalibratedGraspInference
from skill_training_protocol import observation_hash


class CalibratedPhysicalBridgeTests(unittest.TestCase):
    def setUp(self):
        from test_recovery_causal_skill_probe import ProbeTests
        from test_recovery_causal_inference import Processor
        from test_recovery_observer_shadow import Observer,Head
        ProbeTests.setUp(self)
        self.start=lambda i=0:ProbeTests.start(self,i)
        self.apply=lambda p,j,n=16:ProbeTests.apply(self,p,j,n)
        self.binding=GraspCalibrationBinding(self.models,'0'*64,1.77)
        planner,low=self.probe.planner,self.probe.low
        original=planner.ensure_context
        def plan(session,*args,**kwargs):
            result=original(session,*args,**kwargs)
            result['observer_feedback_mode']=session.observer_feedback_mode
            return result
        planner.ensure_context=plan
        self.policy,self.head,self.processor=Observer(),Head(),Processor()
        config=dict(raw_shape=dict(state=[dict(key='left_arm',start_index=0,raw_shape=7)]))
        for module in ('recovery_causal_inference','recovery_observer_shadow'):
            m=patch(module+'.configuration_stats_sha256',return_value=self.models.observer_normalization)
            m.start();self.addCleanup(m.stop)
        def factory(session):
            return BoundCalibratedGraspInference(self.policy,self.processor,config,self.head,session,
                loaded_backbone_sha256=self.models.observer_backbone,
                loaded_adapter_sha256=self.models.observer_adapter,device='cpu')
        self.probe=CausalSkillProbe(self.cases,self.manifest,self.models,planner,low,
            log_event=lambda kind,value:self.log.append((kind,deepcopy(value))),
            calibration=self.binding,observer_factory=factory)

    def episode(self):
        p,j,_=self.start();controls=[];hashes={32:observation_hash(self.observation)}
        for i in range(9):
            if i:self.probe.goal(p,j,self.observation)
            controls+=self.apply(p,j);hashes[p.rollout.step]=observation_hash(self.observation)
        p.rollout.ended=True;self.probe.close(p,j,self.observation)
        events=[dict(kind=k,**v) for k,v in self.log]
        return events,controls,hashes

    def replay(self,events,controls,hashes):
        return audit_joint_episode(events,models=self.models,history_row=self.manifest['rows'][0],
            case=self.cases[self.jobs[0]['case']],job=self.jobs[0],controls=controls,observation_hashes=hashes,
            calibration=self.binding)

    def test_actual_ack_feedback_reaches_planner_never_physical_truth(self):
        events,controls,hashes=self.episode();out=self.replay(events,controls,hashes)
        self.assertEqual(out['applied_controls'],144)
        self.assertEqual(out['observer_predictions'],self.probe.observer_predictions)
        self.assertEqual(out['generations'],2)
        plans=[e['result'] for e in events if e['kind']=='plan' and not e['result']['reused']]
        self.assertEqual(json.loads(plans[0]['causal_input']['execution_feedback'])['estimated_bundle_outcome'],'UNKNOWN')
        self.assertEqual(json.loads(plans[1]['causal_input']['execution_feedback'])['estimated_bundle_outcome'],'SUCCEEDED')
        for plan in plans:
            self.assertEqual(plan['causal_input']['known_previous_outcome'],'UNKNOWN')
            self.assertEqual(json.loads(plan['causal_input']['memory'])['verified_world_facts'],[])
        self.assertEqual(self.probe.observers,{})
        self.assertFalse(out['physical_success_asserted_by_planner'])

    def test_interleaved_episode_feedback_and_caches_are_separate(self):
        p,j,_=self.start();other,oj,_=self.start(1)
        self.apply(p,j);self.probe.goal(p,j,self.observation)
        session=self.probe.active[j['id']][0];foreign=self.probe.active[oj['id']][0]
        def value(s):return json.loads(s.feedback.projection(s.identity.feedback_identity(),s.feedback.control_step))
        self.assertEqual(value(session)['estimated_bundle_outcome'],'SUCCEEDED')
        self.assertEqual(value(foreign)['estimated_bundle_outcome'],'UNKNOWN')
        self.assertEqual([x[0] for x in foreign.windows[0].rows],[32])
        with self.assertRaises(ValueError):self.probe.goal(p,oj,self.observation)

    def test_raw_logit_clock_member_token_and_feedback_corruption_rejected(self):
        events,controls,hashes=self.episode()
        index=next(i for i,e in enumerate(events) if e['kind']=='observer' and 'members' in e['result'])
        mutations=[lambda e:e.update(observation_sha256='bad'),
            lambda e:e.update(attempt_generation=999),lambda e:e.update(estimated_feedback='{}'),
            lambda e:e['result']['members'][0].update(request_token='foreign'),
            lambda e:e['result']['members'][0].update(context_steps=[0,16,32]),
            lambda e:e['result']['members'][0].update(raw_logits=[float('nan')]*4),
            lambda e:e['result'].update(calibration_sha256='1'*64)]
        for mutate in mutations:
            bad=deepcopy(events);mutate(bad[index])
            with self.assertRaises(ValueError):self.replay(bad,controls,hashes)
        with self.assertRaises(ValueError):self.replay(events[:index]+events[index+1:],controls,hashes)

    def test_final_partial_requires_real_final_observation(self):
        p,j,_=self.start();self.apply(p,j,3);p.rollout.ended=True
        with self.assertRaises(ValueError):self.probe.close(p,j)
        self.probe.close(p,j,self.observation)
        checks=[e for k,e in self.log if k=='observer']
        self.assertEqual(checks[-1]['result']['status'],'partial_chunk_skipped_by_original_cadence')

    def test_registered_model_and_receipt_bindings_keep_original_low_cases_and_solver(self):
        from recovery_corpus import file_sha
        repo=Path(__file__).resolve().parents[4]
        old=json.loads((repo/'configs/recovery_sft/a800_causal_joint_shadow_probes_v1.json').read_text())
        new=json.loads((repo/'configs/recovery_sft/a800_causal_calibrated_grasp_probes_v1.json').read_text())
        for k in ['model','cases','evaluation_seeds','ppo','stats_sha256','learning_rounds']:
            self.assertEqual(new[k],old[k])
        spec=new['causal_planner'];model=repo/spec['models_config']
        self.assertEqual(file_sha(model),spec['models_config_sha256'])
        self.assertTrue(spec['observer_predictions_enabled']);self.assertEqual(new['learning_rounds'],0)
        models=json.loads(model.read_text())
        ref=models['pilot_receipts']['observer_fit_config']
        self.assertEqual(file_sha(repo/ref['path']),ref['sha256'])
        self.assertNotEqual(models['planner']['sha256'],models['observer_backbone']['sha256'])


if __name__=='__main__':unittest.main()
