from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'code'), str(Path(__file__).parent)]
from test_recovery_causal_inference import Processor, Policy
from recovery_causal_inference import CausalPlannerInference
from recovery_observer_shadow import BoundObserverShadowInference
from g05.utils.memlite_causal_session import CausalModelIdentity, CausalSessionIdentity, CausalPlannerSession


class Observer(Policy):
    def outcome_context_from_prefix(self, samples, pixels):
        self.inputs.append(samples[0])
        return torch.full((1, 8), float(len(self.inputs)))


class Head(torch.nn.Module):
    include_absolute_proprio = True
    include_served_controls = False
    def __init__(self):
        super().__init__(); self.bad = False; self.calls = 0; self.eval()
    def forward(self, **values):
        self.calls += 1
        return torch.tensor([[0., 10., 0., float('nan') if self.bad else 0.]])


class ShadowTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.models = CausalModelIdentity(*[v*64 for v in 'abcdefe'])
        self.identity = CausalSessionIdentity('s', 'task', 1, 'episode', self.models)
        self.session = CausalPlannerSession(self.identity)
        self.config = dict(raw_shape=dict(state=[dict(key='left_arm', start_index=0, raw_shape=7)]))
        self.processor = Processor(); self.policy = Observer(); self.head = Head()
        for module in ('recovery_causal_inference', 'recovery_observer_shadow'):
            stub = patch(module+'.configuration_stats_sha256', return_value='e'*64)
            stub.start(); self.addCleanup(stub.stop)
        self.planner = CausalPlannerInference(Policy(), self.processor, self.config,
            models=self.models, loaded_planner_sha256='a'*64, device='cpu')
        self.obs = dict(images={k:np.zeros((3,32,32), dtype=np.uint8) for k in
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')}, proprio=np.zeros(61, dtype=np.float32))
        self.planner.plan(self.session, self.identity, self.obs, validate_low_goal=lambda g:None)
        self.shadow = self.bind(self.session)

    def bind(self, session, **kwargs):
        options = dict(loaded_backbone_sha256='c'*64, loaded_adapter_sha256='d'*64, device='cpu')
        options.update(kwargs)
        return BoundObserverShadowInference(self.policy,self.processor,self.config,self.head,session,**options)

    def test_actual_four_frame_history_never_enters_planner(self):
        for step in (0,16,32,48,64):
            self.session.observe(self.identity,step)
            result=self.shadow.check(self.identity,self.obs)
        self.assertEqual(result['members'][0]['context_steps'],[16,32,48,64])
        self.assertEqual(result['members'][0]['uncalibrated_prediction']['outcome'],'SUCCEEDED')
        _,causal=self.session.begin_planning(self.identity,64)
        feedback=json.loads(causal['execution_feedback'])
        self.assertEqual(feedback['estimated_bundle_outcome'],'UNKNOWN')
        self.assertEqual(feedback['estimated_member_outcomes'][0]['confidence'],0.)
        self.assertFalse(result['entered_planner'])
        for prefix in self.policy.inputs:
            self.assertTrue(prefix['template'].endswith('<EOC>'))
            self.assertNotIn('outcome_target',prefix);self.assertNotIn('physical_audit',prefix)

    def test_duplicates_short_partial_and_changed_same_clock(self):
        self.shadow.check(self.identity,self.obs)
        self.assertEqual(self.shadow.check(self.identity,self.obs)['status'],'duplicate_check_not_recomputed')
        changed=dict(self.obs,proprio=self.obs['proprio'].copy());changed['proprio'][0]=1.
        with self.assertRaisesRegex(ValueError,'Different RGB'):self.shadow.check(self.identity,changed)
        self.session.observe(self.identity,10)
        self.assertEqual(self.shadow.check(self.identity,self.obs)['status'],'partial_chunk_skipped_by_original_cadence')
        self.assertEqual(self.head.calls,1);self.assertEqual(len(self.session.windows[0].rows),1)

    def test_failed_head_does_not_commit_any_feature_or_feedback(self):
        self.head.bad=True
        with self.assertRaisesRegex(ValueError,'Finite'):self.shadow.check(self.identity,self.obs)
        self.assertEqual(len(self.session.windows[0].rows),0)
        self.assertEqual(len(self.session.feedback._proposals),0);self.assertIsNone(self.shadow.last)
        self.head.bad=False
        self.shadow.check(self.identity,self.obs)
        self.assertEqual(len(self.session.windows[0].rows),1)

    def test_second_parallel_member_failure_rolls_back_first_member(self):
        from g05.utils.memlite_skill_protocol import canonical_json,semantic_active_skills_text
        token,_=self.session.begin_planning(self.identity,0)
        proposal=self.planner.policy.generate_high_level([dict(task_name='task',memory=self.session.memory,
            previous_intent=self.session.previous_intent)],{},0.)['planner_events'][0]
        members=json.loads(proposal['active_skills_semantic_json'])
        members.append(dict(members[0],arm='RIGHT',target='plate'))
        proposal['active_skills_semantic_json']=canonical_json(members)
        proposal['active_skills_text']=semantic_active_skills_text(members)
        self.session.stage(self.identity,token,proposal);self.session.commit(self.identity,token)
        original=self.head.forward
        def broken(**values):
            self.head.bad=self.head.calls==1
            return original(**values)
        with patch.object(self.head,'forward',side_effect=broken):
            with self.assertRaises(ValueError):self.shadow.check(self.identity,self.obs)
        self.assertEqual([len(w.rows) for w in self.session.windows],[0,0])
        self.assertEqual(len(self.session.feedback._proposals),0)
        self.assertIsNone(self.shadow.last)

    def test_foreign_identity_or_oracle_is_rejected_before_model(self):
        for bad in (replace(self.identity,episode='other'),replace(self.identity,instance=2),
                    replace(self.identity,models=replace(self.models,observer_adapter='0'*64))):
            with self.assertRaises(ValueError):self.shadow.check(bad,self.obs)
        with self.assertRaises(ValueError):self.shadow.check(self.identity,dict(self.obs,reward=1))
        self.assertEqual(self.head.calls,0)

    def test_wrong_weights_history_or_trainable_module_rejected(self):
        for options in (dict(loaded_backbone_sha256='a'*64),dict(loaded_adapter_sha256='a'*64),
                        dict(history_protocol='attempt_start_short_v1')):
            with self.assertRaises(ValueError):self.bind(self.session,**options)
        self.head.train()
        with self.assertRaises(ValueError):self.bind(self.session)
        self.head.eval();self.head.parameter=torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):self.bind(self.session)

    def test_pending_planner_or_action_and_closed_session_rejected(self):
        token,_=self.session.begin_planning(self.identity,0)
        with self.assertRaises(ValueError):self.shadow.check(self.identity,self.obs)
        self.session.discard(self.identity,token);self.session.action_in_flight='pending'
        with self.assertRaises(ValueError):self.shadow.check(self.identity,self.obs)
        self.session.action_in_flight=None;self.session.close(self.identity,0)
        with self.assertRaises(ValueError):self.shadow.check(self.identity,self.obs)

    def test_interleaved_wrappers_share_only_frozen_models(self):
        other_id=replace(self.identity,session='s2',episode='second',task='another task')
        other=CausalPlannerSession(other_id)
        self.planner.plan(other,other_id,self.obs,validate_low_goal=lambda g:None)
        second=self.bind(other)
        self.shadow.check(self.identity,self.obs);second.check(other_id,self.obs)
        self.session.observe(self.identity,16);self.shadow.check(self.identity,self.obs)
        self.assertEqual([r[0] for r in other.windows[0].rows],[0])
        self.assertEqual([r[0] for r in self.session.windows[0].rows],[0,16])
        self.assertNotEqual(float(other.windows[0].rows[0][1][0]),float(self.session.windows[0].rows[0][1][0]))

    def test_retry_same_clock_clears_old_attempt_not_new_same_intent_refresh(self):
        self.shadow.check(self.identity,self.obs)
        self.session.observe(self.identity,16)
        self.shadow.check(self.identity,self.obs)
        token,_=self.session.begin_planning(self.identity,16)
        proposal=self.planner.policy.generate_high_level([dict(task_name='task',memory=self.session.memory,
            previous_intent=self.session.previous_intent)],{},0.)['planner_events'][0]
        proposal['decision']='RETRY'
        self.session.stage(self.identity,token,proposal);self.session.commit(self.identity,token)
        result=self.shadow.check(self.identity,self.obs)
        self.assertEqual(result['members'][0]['context_steps'],[16])
        self.assertEqual(self.session.feedback.attempt,2)
        self.assertEqual(len(self.session.feedback._proposals),1)


if __name__=='__main__':unittest.main()
