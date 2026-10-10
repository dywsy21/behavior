from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'code'), str(Path(__file__).parent)]
from test_recovery_causal_inference import Processor, Policy
from test_recovery_observer_shadow import Observer, Head
from recovery_causal_inference import CausalPlannerInference
from recovery_calibrated_observer import (GraspCalibrationBinding, load_grasp_calibration,
    calibrated_prediction, CalibratedGraspPilotSession, BoundCalibratedGraspInference)
from recovery_corpus import file_sha
from g05.utils.memlite_causal_session import CausalModelIdentity, CausalSessionIdentity, CausalPlannerSession
from g05.utils.memlite_skill_protocol import canonical_json, semantic_active_skills_text


class CalibratedObserverTests(unittest.TestCase):
    def setUp(self):
        self.models = CausalModelIdentity(*[v*64 for v in 'abcdefe'])
        self.identity = CausalSessionIdentity('s', 'task', 1, 'episode', self.models)
        self.binding = GraspCalibrationBinding(self.models, '0'*64, 1.77)
        self.session = CalibratedGraspPilotSession(self.identity, calibration=self.binding)
        self.config = dict(raw_shape=dict(state=[dict(key='left_arm', start_index=0, raw_shape=7)]))
        self.processor = Processor(); self.policy = Observer(); self.head = Head()
        for module in ('recovery_causal_inference', 'recovery_observer_shadow'):
            stub = patch(module+'.configuration_stats_sha256', return_value='e'*64)
            stub.start(); self.addCleanup(stub.stop)
        self.planner = CausalPlannerInference(Policy(), self.processor, self.config,
            models=self.models, loaded_planner_sha256='a'*64, device='cpu')
        self.obs = dict(images={k:np.zeros((3,32,32), dtype=np.uint8) for k in
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')}, proprio=np.zeros(61, dtype=np.float32))
        self.planner.plan(self.session,self.identity,self.obs,validate_low_goal=lambda _:None)
        self.observer = self.bind(self.session)

    def bind(self, session):
        return BoundCalibratedGraspInference(self.policy,self.processor,self.config,self.head,session,
            loaded_backbone_sha256='c'*64,loaded_adapter_sha256='d'*64,device='cpu')

    def projection(self, session=None, identity=None):
        session=session or self.session;identity=identity or self.identity
        return json.loads(session.feedback.projection(identity.feedback_identity(),session.feedback.control_step))

    def issue(self, *, decision='EXECUTE', extra=None):
        token,_=self.session.begin_planning(self.identity,self.session.feedback.control_step)
        event=self.planner.policy.generate_high_level([dict(task_name='task',memory=self.session.memory,
            previous_intent=self.session.previous_intent)],{},0.)['planner_events'][0]
        event['decision']=decision
        if extra:
            members=json.loads(event['active_skills_semantic_json']);members.append(extra)
            event['active_skills_semantic_json']=canonical_json(members)
            event['active_skills_text']=semantic_active_skills_text(members)
        self.session.stage(self.identity,token,event);self.session.commit(self.identity,token)

    def test_two_fresh_checks_and_known_outcome_never_becomes_truth(self):
        self.observer.check(self.identity,self.obs)
        self.assertEqual(self.projection()['estimated_bundle_outcome'],'UNKNOWN')
        self.observer.check(self.identity,self.obs)
        self.assertEqual(self.head.calls,1)
        self.session.observe(self.identity,16);report=self.observer.check(self.identity,self.obs)
        p=self.projection();self.assertEqual(p['estimated_bundle_outcome'],'SUCCEEDED')
        expected=calibrated_prediction([0.,10.,0.,0.],temperature=1.77,verb='GRASP')
        self.assertEqual(p['estimated_member_outcomes'][0]['confidence'],expected['confidence'])
        result=self.planner.plan(self.session,self.identity,self.obs,validate_low_goal=lambda _:None)
        self.assertEqual(result['causal_input']['known_previous_outcome'],'UNKNOWN')
        self.assertEqual(result['observer_feedback_mode'],'calibrated_grasp_estimate_pilot_v1')
        self.assertEqual(json.loads(self.session.memory)['verified_world_facts'],[])
        self.assertFalse(report['physical_success_asserted']);self.assertFalse(result['physical_success_asserted'])
        self.assertFalse(report['entered_planner'])

    def test_partial_clock_stales_and_retry_clears_confirmations(self):
        for step in (0,16):
            self.session.observe(self.identity,step);self.observer.check(self.identity,self.obs)
        self.session.observe(self.identity,24)
        result=self.observer.check(self.identity,self.obs)
        self.assertEqual(result['status'],'partial_chunk_skipped_by_original_cadence')
        self.assertEqual(self.projection()['estimated_bundle_outcome'],'UNKNOWN')
        self.issue(decision='RETRY');report=self.observer.check(self.identity,self.obs)
        self.assertEqual(report['members'][0]['context_steps'],[24])
        self.assertEqual(self.projection()['estimated_bundle_outcome'],'UNKNOWN')
        self.assertEqual(len(self.session.feedback._proposals),1)

    def test_parallel_open_is_unknown_not_grasp_calibration(self):
        self.issue(extra=dict(verb='OPEN_DOOR',target='washer',source='',destination='',
                             target_part='',arm='RIGHT',unbound_relation=''))
        for step in (0,16):
            self.session.observe(self.identity,step);self.observer.check(self.identity,self.obs)
        p=self.projection()
        self.assertEqual([r['estimated_outcome'] for r in p['estimated_member_outcomes']],['SUCCEEDED','UNKNOWN'])
        self.assertEqual(p['estimated_member_outcomes'][1]['confidence'],0.)
        self.assertEqual(p['estimated_bundle_outcome'],'UNKNOWN')
        self.session.observe(self.identity,32)
        with self.assertRaisesRegex(ValueError,'never licenses'):
            self.session.calibrated_outcomes(self.identity,32,[dict(outcome='SUCCEEDED',confidence=.99)]*2,
                                             calibration=self.binding)

    def test_wrong_session_identity_or_calibration_cannot_commit(self):
        with self.assertRaises(ValueError):self.bind(CausalPlannerSession(self.identity))
        with self.assertRaises(ValueError):CalibratedGraspPilotSession(self.identity,
            calibration=replace(self.binding,models=replace(self.models,planner='1'*64)))
        for identity in (replace(self.identity,episode='other'),replace(self.identity,task='other')):
            with self.assertRaises(ValueError):self.observer.check(identity,self.obs)
        with self.assertRaises(ValueError):self.session.calibrated_outcomes(self.identity,0,
            [dict(outcome='SUCCEEDED',confidence=.99)],calibration=replace(self.binding,calibration_sha256='1'*64))
        self.assertEqual(len(self.session.feedback._proposals),0)

    def test_interleaved_sessions_do_not_share_prediction_history(self):
        identity=replace(self.identity,session='other',episode='other',task='other task')
        second=CalibratedGraspPilotSession(identity,calibration=self.binding)
        self.planner.plan(second,identity,self.obs,validate_low_goal=lambda _:None);other=self.bind(second)
        self.observer.check(self.identity,self.obs);other.check(identity,self.obs)
        self.session.observe(self.identity,16);self.observer.check(self.identity,self.obs)
        self.assertEqual(self.projection()['estimated_bundle_outcome'],'SUCCEEDED')
        self.assertEqual(self.projection(second,identity)['estimated_bundle_outcome'],'UNKNOWN')
        self.assertEqual([r[0] for r in second.windows[0].rows],[0])

    def test_failure_does_not_publish_partial_feedback(self):
        self.head.bad=True
        with self.assertRaises(ValueError):self.observer.check(self.identity,self.obs)
        self.assertEqual(len(self.session.windows[0].rows),0)
        self.assertEqual(len(self.session.feedback._proposals),0)
        with self.assertRaises(ValueError):self.observer.check(self.identity,dict(self.obs,reward=1))

    def test_softmax_and_original_gates_are_not_loosened(self):
        raw=calibrated_prediction([0.,3.,0.,0.],temperature=1.,verb='GRASP')
        calibrated=calibrated_prediction([0.,3.,0.,0.],temperature=1.77,verb='GRASP')
        self.assertGreater(raw['confidence'],.85);self.assertLess(calibrated['confidence'],.85)
        for logits in ([0.,1.],[0.,float('nan'),1.,1.],[True,1.,2.,3.]):
            with self.assertRaises(ValueError):calibrated_prediction(logits,temperature=1.77,verb='GRASP')
        for change in (dict(temperature=True),dict(minimum_confidence=.8),dict(confirmations=1)):
            with self.assertRaises(ValueError):replace(self.binding,**change)

    def test_loader_checks_real_receipt_bytes_gate_and_observer_not_planner(self):
        cal=dict(schema='recovery_observer_calibration_v1',ready=True,blockers=[],mechanism='GRASP',
            certifies_other_mechanisms=False,minimum_confidence=.85,temperature=1.77,
            high_sha256='c'*64,observer_sha256='d'*64,events=90,false_success_count=1,
            source_groups=['cal:'+str(i) for i in range(90)],classes={
                name:dict(actual_events=30,confident_predictions=n,correct=c)
                for name,n,c in [('IN_PROGRESS',25,25),('SUCCEEDED',29,28),('FAILED',27,26)]})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json';p.write_text(json.dumps(cal));sha=file_sha(p)
            self.assertEqual(load_grasp_calibration(p,sha,self.models).temperature,1.77)
            with self.assertRaises(ValueError):load_grasp_calibration(p,'0'*64,self.models)
            with self.assertRaises(ValueError):load_grasp_calibration(p,sha,replace(self.models,observer_backbone='a'*64))
            cal['ready']=False;p.write_text(json.dumps(cal))
            with self.assertRaises(ValueError):load_grasp_calibration(p,file_sha(p),self.models)


if __name__=='__main__':unittest.main()
