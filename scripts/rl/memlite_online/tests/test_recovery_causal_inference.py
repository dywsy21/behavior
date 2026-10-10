import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_causal_inference import CausalPlannerInference,observer_prefix
from g05.utils.memlite_causal_session import CausalModelIdentity,CausalSessionIdentity,CausalPlannerSession
from g05.utils.memlite_skill_protocol import canonical_json,append_b_memory_idempotent,semantic_active_skills_text


class Processor:
    def __init__(self):
        self.samples_builder=types.SimpleNamespace(num_input_images=3,_image_sizes={k:(256,256) for k in
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')},template='<memory_text_!><EOC><next_decision_text>',
            embodiment_type='galaxea_r1pro')
    def _process_tensors(self,raw):
        return dict(_instructions=raw['task'],proprio=torch.ones(1,27),proprio_dim_is_pad=torch.zeros(27,dtype=torch.bool),
            pixel_values=raw['images'],outcome_target='FAILED',physical_audit={'oracle':'MUST_NOT_LEAK'})


class Policy:
    planner_only=True
    training=False
    def __init__(self):self.inputs=[];self.broken=False
    def parameters(self):return []
    def generate_high_level(self,samples,pixels,temperature):
        self.inputs.append(samples[0]);s=samples[0]
        if self.broken:raise RuntimeError('generation failure')
        members=[dict(verb='GRASP',target='cup',source='',destination='',target_part='',arm='LEFT',unbound_relation='')]
        return dict(planner_events=[dict(validated=True,previous_outcome='UNKNOWN',task_complete=False,
            task_complete_claimed=False,decision='EXECUTE',parent_goal='Task goal: '+s['task_name'],
            active_skills_semantic_json=canonical_json(members),active_skills_text=semantic_active_skills_text(members),
            memory_update=append_b_memory_idempotent(s['memory'],s['previous_intent'],task_name=s['task_name']))])


class Tests(unittest.TestCase):
    def setUp(self):
        self.models=CausalModelIdentity('a'*64,'b'*64,'c'*64,'d'*64,'e'*64,'f'*64,'e'*64)
        self.i=CausalSessionIdentity('slot','task',3,'episode',self.models)
        self.session=CausalPlannerSession(self.i);self.policy=Policy();self.processor=Processor()
        self.config=dict(raw_shape=dict(state=[dict(key='left_arm',start_index=0,raw_shape=7)]))
        stats=patch('recovery_causal_inference.configuration_stats_sha256',return_value='e'*64)
        self.addCleanup(stats.stop);self.stats=stats.start()
        self.adapter=CausalPlannerInference(self.policy,self.processor,self.config,models=self.models,
            loaded_planner_sha256='a'*64,device='cpu')
        self.observation=dict(images={k:np.zeros((3,32,32),dtype=np.uint8) for k in
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')},proprio=np.zeros(61,dtype=np.float32))

    def test_actual_prefix_and_low_condition_have_no_targets_or_identity(self):
        goals=[]
        for step in (0,128):
            self.session.observe(self.i,step)
            result=self.adapter.plan(self.session,self.i,self.observation,validate_low_goal=goals.append)
        prefix=self.policy.inputs[-1]
        for key in ('outcome_target','physical_audit','next_decision','episode','source_group','identity'):
            self.assertNotIn(key,prefix)
        self.assertTrue(prefix['template'].endswith('<EOC>'))
        self.assertEqual(json.loads(prefix['execution_feedback'])['same_intent_controls'],128)
        self.assertEqual(set(goals[0]),{'task','parent_goal','semantic_bundle'})
        self.assertEqual(result['revision'],2)
        self.assertFalse(result['physical_success_asserted'])

    def test_failed_generation_or_low_check_never_commits(self):
        self.policy.broken=True
        with self.assertRaises(RuntimeError):self.adapter.plan(self.session,self.i,self.observation,validate_low_goal=lambda x:None)
        self.assertEqual(self.session.revision,0);self.assertIsNone(self.session.request)
        self.policy.broken=False
        def reject(goal):raise ValueError('bad downstream condition')
        with self.assertRaises(ValueError):self.adapter.plan(self.session,self.i,self.observation,validate_low_goal=reject)
        self.assertEqual(self.session.revision,0);self.assertIsNone(self.session.installed)
        self.assertIsNone(self.session.feedback.bundle)

    def test_oracle_fields_rejected_and_observer_cannot_use_planner_backbone(self):
        with self.assertRaises(ValueError):self.adapter.plan(self.session,self.i,dict(self.observation,reward=1),validate_low_goal=lambda x:None)
        self.assertIsNone(self.session.request)
        self.adapter.plan(self.session,self.i,self.observation,validate_low_goal=lambda x:None)
        with self.assertRaises(ValueError):observer_prefix(self.processor,self.config,self.session,self.i,0,self.observation,loaded_backbone_sha256='a'*64)
        _,prefix,_=observer_prefix(self.processor,self.config,self.session,self.i,0,self.observation,loaded_backbone_sha256='c'*64)
        self.assertIn('GRASP',prefix['previous_intent']);self.assertNotIn('outcome_target',prefix)

    def test_interleaved_slots_keep_memories_independent(self):
        other=CausalSessionIdentity('slot2','other task',4,'episode2',self.models);second=CausalPlannerSession(other)
        self.adapter.plan(self.session,self.i,self.observation,validate_low_goal=lambda x:None)
        self.session.observe(self.i,128)
        self.adapter.plan(second,other,self.observation,validate_low_goal=lambda x:None)
        self.adapter.plan(self.session,self.i,self.observation,validate_low_goal=lambda x:None)
        self.assertEqual(json.loads(second.memory)['task_name'],'other task')
        self.assertEqual(json.loads(second.memory)['issued_command_history'],[])
        self.assertEqual(len(json.loads(self.session.memory)['issued_command_history']),1)
        self.assertEqual(second.feedback.control_step,0)

    def test_planner_and_observer_each_reject_wrong_normalizer_even_with_right_weights(self):
        self.stats.return_value='f'*64
        with self.assertRaises(ValueError):CausalPlannerInference(self.policy,self.processor,self.config,
            models=self.models,loaded_planner_sha256='a'*64,device='cpu')
        self.stats.return_value='e'*64
        self.adapter.plan(self.session,self.i,self.observation,validate_low_goal=lambda x:None)
        self.stats.return_value='f'*64
        with self.assertRaises(ValueError):observer_prefix(self.processor,self.config,self.session,self.i,0,
            self.observation,loaded_backbone_sha256='c'*64)


if __name__=='__main__':unittest.main()
