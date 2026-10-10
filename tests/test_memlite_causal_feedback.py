import json
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

from g05.utils.memlite_causal_feedback import CausalFeedbackLedger, CausalFeatureWindow, FeedbackIdentity, single_frame_member_prefix, last_context_hidden, single_frame_planner_prefix

# Test the torch-only head without importing the full VLM/Hydra deployment.
# A private package name avoids shadowing g05 in other tests in this process.
_helpers = Path(__file__).resolve().parents[1] / "src/g05/models/g05/helpers"
_package = types.ModuleType("_recovery_observer_test_helpers")
_package.__path__ = [str(_helpers)]
sys.modules[_package.__name__] = _package
_spec = importlib.util.spec_from_file_location(_package.__name__ + ".temporal_outcome", _helpers / "temporal_outcome.py")
_observer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_observer)
TemporalOutcomeObserver = _observer.TemporalOutcomeObserver


def identity(episode="e", task="t", high="a"):
    return FeedbackIdentity("session", task, 3, episode, high*64, "b"*64)


def bundle(target="cup", parallel=False):
    row = dict(verb="GRASP", target=target, source="", destination="", target_part="", arm="LEFT", unbound_relation="")
    return json.dumps([row, dict(row, target="plate", arm="RIGHT")] if parallel else [row], sort_keys=True, separators=(",", ":"))


class FeedbackTests(unittest.TestCase):
    def test_planner_prefix_requires_input_whitelist_and_has_no_answers(self):
        from g05.utils.memlite_skill_protocol import planner_input_projection
        builder=types.SimpleNamespace(num_input_images=3,_image_sizes={k:(256,256) for k in
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')},embodiment_type='galaxea_r1pro',
            template='<memory_text_!><EOC><outcome_target_text>|<next_decision_text>')
        memory=json.dumps(dict(task_name='t',issued_command_history=[],verified_world_facts=[]),sort_keys=True,separators=(',',':'))
        fields=dict(task_name='t',memory=memory,previous_intent='None',previous_parent_goal='none',
            known_previous_outcome='UNKNOWN',execution_feedback='none')
        prepared=dict(_instructions='t',proprio=torch.ones(1,27),proprio_dim_is_pad=torch.zeros(27,dtype=torch.bool),
            outcome_target='FAILED',physical_audit={'oracle':True})
        causal=planner_input_projection(fields)
        a=single_frame_planner_prefix(builder,prepared,causal)
        self.assertNotIn('outcome_target',a); self.assertNotIn('physical_audit',a)
        self.assertEqual(a['memlite_branch'],'high'); self.assertTrue(a['template'].endswith('<EOC>'))
        with self.assertRaises(ValueError): single_frame_planner_prefix(builder,prepared,dict(causal,next_decision='RETRY'))
        with self.assertRaises(ValueError): single_frame_planner_prefix(builder,prepared,dict(causal,task_name='other'))

    def test_last_context_uses_nonzero_positions_not_modality_sum(self):
        hidden=torch.arange(3*5*2).reshape(3,5,2).float()
        masks=torch.tensor([[1,2,4,4,0],[0,0,1,2,4],[1,0,4,0,0]])
        torch.testing.assert_close(last_context_hidden(hidden,masks),hidden[torch.arange(3),torch.tensor([3,4,2])])
        self.assertTrue((masks.sum(1)-1>=hidden.shape[1]).any())
        with self.assertRaises(ValueError):last_context_hidden(hidden,torch.zeros_like(masks))

    def test_single_frame_prefix_never_copies_training_answers(self):
        builder=types.SimpleNamespace(num_input_images=3,_image_sizes={k:(256,256) for k in
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')},embodiment_type='galaxea_r1pro',
            template='<memory_text_!><EOC><outcome_target_text>|<next_decision_text>')
        prepared=dict(_instructions='t',proprio=torch.ones(1,27),proprio_dim_is_pad=torch.zeros(27,dtype=torch.bool),
                      outcome_target='FAILED',physical_audit=dict(oracle='must not copy'))
        memory=json.dumps(dict(task_name='t',issued_command_history=[],verified_world_facts=[]),sort_keys=True,separators=(',',':'))
        result=single_frame_member_prefix(builder,prepared,task_name='t',parent_goal='Task goal: t',
            issued_bundle=bundle(parallel=True),member_index=1,memory=memory,served_controls=128)
        self.assertEqual(result['template'],'<memory_text_!><EOC>')
        self.assertNotIn('outcome_target',result);self.assertNotIn('physical_audit',result)
        self.assertIn('plate',result['previous_intent']);self.assertNotIn('cup',result['previous_intent'])
        self.assertEqual([k for k in result if k.startswith('image')],['image0','image1','image2'])
        with self.assertRaises(ValueError):single_frame_member_prefix(builder,prepared,task_name='other',parent_goal='Task goal: t',
            issued_bundle=bundle(),member_index=0,memory=memory,served_controls=0)

    def test_refresh_is_not_retry_and_elapsed_does_not_disappear(self):
        i=identity(); ledger=CausalFeedbackLedger(i)
        ledger.issued(i,0,bundle(),"goal")
        ledger.issued(i,128,bundle(),"goal")
        result=json.loads(ledger.projection(i,256))
        self.assertEqual(result['same_intent_controls'],256)
        self.assertEqual(result['attempt_index'],1)
        self.assertEqual(result['same_intent_planner_refreshes'],1)
        self.assertEqual(result['estimated_bundle_outcome'],'UNKNOWN')
        ledger.issued(i,256,bundle(),"goal",decision="RETRY")
        result=json.loads(ledger.projection(i,256))
        self.assertEqual(result['attempt_index'],2)
        self.assertEqual(result['same_intent_controls'],0)

    def test_cross_task_episode_model_are_rejected(self):
        i=identity(); ledger=CausalFeedbackLedger(i)
        for other in (identity(task="u"),identity(episode="new"),identity(high="c")):
            with self.assertRaises(ValueError):ledger.issued(other,0,bundle(),"g")
        fresh=CausalFeedbackLedger(identity(task="u"))
        self.assertEqual(fresh.projection(identity(task="u"),0),'none')

    def test_double_arm_results_need_all_members_and_distinct_checks(self):
        i=identity(); ledger=CausalFeedbackLedger(i);ledger.issued(i,0,bundle(parallel=True),"g")
        partial=[dict(outcome='SUCCEEDED',confidence=.99),dict(outcome='IN_PROGRESS',confidence=.99)]
        for step in (128,256):
            ledger.observe_clock(i,step);ledger.estimated(i,step,partial,calibrated=True)
        self.assertEqual(json.loads(ledger.projection(i,256))['estimated_bundle_outcome'],'IN_PROGRESS')
        with self.assertRaises(ValueError):ledger.estimated(i,256,partial,calibrated=True)
        self.assertEqual(json.loads(ledger.projection(i,384))['estimated_bundle_outcome'],'UNKNOWN')

    def test_uncalibrated_low_confidence_or_oracle_cannot_drive_transition(self):
        i=identity(); ledger=CausalFeedbackLedger(i);ledger.issued(i,0,bundle(),"g")
        for step in (0,128):
            ledger.observe_clock(i,step)
            ledger.estimated(i,step,[dict(outcome='SUCCEEDED',confidence=.99)],calibrated=False)
        self.assertEqual(json.loads(ledger.projection(i,128))['estimated_bundle_outcome'],'UNKNOWN')
        ledger.observe_clock(i,256)
        with self.assertRaises(ValueError):ledger.estimated(i,256,[dict(outcome='SUCCEEDED',confidence=.99,physical_audit={})],calibrated=True)
        with self.assertRaises(ValueError):ledger.observe_clock(i,16)

    def test_bounded_features_are_detached_and_isolated(self):
        i=identity(); cache=CausalFeatureWindow(i)
        for step in range(6):cache.append(i,step,torch.ones(8,requires_grad=True),torch.ones(27))
        data=cache.tensors(i,5)
        self.assertEqual(data['steps'].tolist(),[[2,3,4,5]])
        self.assertFalse(data['context'].requires_grad)
        with self.assertRaises(ValueError):cache.tensors(identity(episode='other'),5)
        with self.assertRaises(ValueError):cache.tensors(i,4)

    def test_unready_confidence_contract_is_explicit_and_not_a_new_outcome(self):
        for protocol,expected in [('raw_observer_confidence_v1',.99),('unready_zero_confidence_v1',0.)]:
            i=identity();ledger=CausalFeedbackLedger(i,uncertainty_protocol=protocol)
            ledger.issued(i,0,bundle(),'g')
            for step in (0,128):
                ledger.observe_clock(i,step)
                ledger.estimated(i,step,[dict(outcome='SUCCEEDED',confidence=.99)],calibrated=False)
            result=json.loads(ledger.projection(i,128))
            self.assertEqual(result['estimated_member_outcomes'],[dict(member=0,estimated_outcome='UNKNOWN',confidence=expected)])
            self.assertEqual(result['estimated_bundle_outcome'],'UNKNOWN')
            for step in (256,384):
                ledger.observe_clock(i,step)
                ledger.estimated(i,step,[dict(outcome='FAILED',confidence=.96)],calibrated=True)
            result=json.loads(ledger.projection(i,384))
            self.assertEqual(result['estimated_member_outcomes'],[dict(member=0,estimated_outcome='FAILED',confidence=.96)])
        with self.assertRaises(ValueError):CausalFeedbackLedger(identity(),uncertainty_protocol='oracle')

    def test_optional_control_age_requires_actual_same_attempt_on_every_append(self):
        i=identity();cache=CausalFeatureWindow(i,intent_started_control_step=32)
        for step in (32,48):
            cache.append(i,step,torch.ones(8),torch.ones(27),intent_started_control_step=32)
        self.assertEqual(cache.tensors(i,48)['served_controls'].tolist(),[[0,16]])
        with self.assertRaises(ValueError):cache.append(i,64,torch.ones(8),torch.ones(27))
        with self.assertRaises(ValueError):cache.append(i,64,torch.ones(8),torch.ones(27),intent_started_control_step=64)

    def test_h0_gradients_stop_before_frozen_vlm(self):
        torch.manual_seed(17)
        observer=TemporalOutcomeObserver(16,width=8)
        features=torch.randn(2,4,16,requires_grad=True)
        state=torch.randn(2,4,27,requires_grad=True)
        steps=torch.tensor([[0,128,256,384],[0,128,0,0]])
        valid=torch.tensor([[True]*4,[True,True,False,False]])
        logits=observer(features,state,steps,valid)
        logits.square().sum().backward()
        self.assertIsNone(features.grad);self.assertIsNone(state.grad)
        self.assertIsNotNone(observer.head.classifier.weight.grad)
        before=observer(features,state,steps,valid).detach()
        modified=features.detach().clone();modified[1,2:]=100
        torch.testing.assert_close(before,observer(modified,state,steps,valid))
        steps[0,2]=12
        with self.assertRaises(ValueError):observer(features,state,steps,valid)


if __name__ == '__main__':unittest.main()
