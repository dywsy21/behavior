from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).resolve().parents[1]/'code'),str(Path(__file__).parent)]
from test_recovery_causal_inference import Policy,Processor
from test_recovery_observer_shadow import Observer,Head
from recovery_causal_inference import CausalPlannerInference
from recovery_calibrated_observer import GraspCalibrationBinding,BoundCalibratedGraspInference
from g05.utils.memlite_causal_session import CausalModelIdentity,CausalSessionIdentity
from full_episode_session import FullEpisodeSession


class FullEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.models=CausalModelIdentity(*[v*64 for v in 'abcdefe'])
        self.identity=CausalSessionIdentity('s','task',301,'episode',self.models)
        self.binding=GraspCalibrationBinding(self.models,'0'*64,1.77)
        self.config=dict(raw_shape=dict(state=[dict(key='left_arm',start_index=0,raw_shape=7)]))
        self.processor=Processor();self.policy=Policy();self.observer_policy=Observer();self.head=Head()
        for module in ('recovery_causal_inference','recovery_observer_shadow'):
            stub=patch(module+'.configuration_stats_sha256',return_value='e'*64)
            stub.start();self.addCleanup(stub.stop)
        self.planner=CausalPlannerInference(self.policy,self.processor,self.config,models=self.models,
            loaded_planner_sha256='a'*64,device='cpu')
        self.obs=dict(images={k:np.zeros((3,32,32),dtype=np.uint8) for k in
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')},proprio=np.zeros(61,dtype=np.float32))
        self.events=[];self.full=self.new(self.identity)

    def new(self,identity):
        return FullEpisodeSession(identity,calibration=self.binding,planner=self.planner,
            low=SimpleNamespace(prepare=lambda *args:None),
            observer_factory=lambda s:BoundCalibratedGraspInference(self.observer_policy,self.processor,
                self.config,self.head,s,loaded_backbone_sha256='c'*64,loaded_adapter_sha256='d'*64,device='cpu'),
            emit=lambda k,v:self.events.append((k,v)))

    def infer(self,s=None):
        s=s or self.full
        return s.infer(s.session.identity,self.obs,lambda *_:np.zeros((16,23),dtype=np.float32))

    def ack(self,s,result,n=16):
        start=s.session.feedback.control_step
        rows=[dict(control_step=start+i,simulator_apply_ack=True,
            action_executed_raw23=result['action_chunk'][i].tolist()) for i in range(n)]
        return s.acknowledge(s.session.identity,result['token'],rows)

    def test_fresh_start_no_teacher_and_true_128_control_plan_partial_final(self):
        self.assertEqual(json.loads(self.full.session.memory)['issued_command_history'],[])
        for _ in range(8):self.ack(self.full,self.infer())
        self.assertEqual(len(self.policy.inputs),1)
        r=self.infer();self.assertEqual(len(self.policy.inputs),2)
        self.assertEqual(r['display']['control_step'],128)
        self.ack(self.full,r,1)
        result=self.full.close(self.identity,self.obs)
        self.assertEqual(result['control_step'],129)
        self.assertEqual(self.full.session.revision,2)
        self.assertEqual(json.loads(self.full.session.memory)['verified_world_facts'],[])

    def test_cross_task_memories_heads_clocks_and_ack_cannot_mix(self):
        second=self.new(replace(self.identity,session='t',task='other task',episode='other'))
        first=self.infer();self.ack(self.full,first)
        r=self.infer(second)
        self.assertEqual(second.session.feedback.control_step,0)
        self.assertEqual(self.full.session.feedback.control_step,16)
        self.assertEqual(r['display']['observer']['members'][0]['context_steps'],[0])
        self.assertNotEqual(second.session.memory,self.full.session.memory)
        with self.assertRaises(ValueError):self.full.acknowledge(second.session.identity,r['token'],[])

    def test_pending_duplicate_oracle_and_wrong_action_rejected(self):
        r=self.infer()
        with self.assertRaises(ValueError):self.infer()
        with self.assertRaises(ValueError):self.full.close(self.identity,self.obs)
        with self.assertRaises(ValueError):self.full.acknowledge(self.identity,r['token'],[
            dict(control_step=0,simulator_apply_ack=True,action_executed_raw23=[1.]*23)])
        self.ack(self.full,r)
        with self.assertRaises(ValueError):self.ack(self.full,r)
        with self.assertRaises(ValueError):self.full.infer(self.identity,dict(self.obs,reward=1),lambda *_:None)


if __name__=='__main__':unittest.main()
