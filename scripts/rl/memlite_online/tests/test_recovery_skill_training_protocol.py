from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest
from skill_training_protocol import SkillTrainingSession,actor_observation


def skill():return dict(verb='GRASP',arm='RIGHT')
def evidence(held=False):return dict(semantic_skill_sha256=digest(skill()),exact_binding_verified=True,
    arm='RIGHT',target_held_by_requested_arm=held,requested_eef_target_distance_m=.05)
def observation():return dict(images={k:np.zeros((3,16,16),np.uint8) for k in
    ('head_rgb','left_wrist_rgb','right_wrist_rgb')},proprio=np.zeros(61,np.float32))
def session(end=30):return SkillTrainingSession(dict(task='t',instance_id=1,start_control=0,end_control=end,
    context_id='ctx',semantic_bundle=json.dumps([skill()])),session='run',episode='ep',policy_version=0,
    policy_sha256='a'*64,initial_evidence=evidence())
def message(s,op,**extra):
    if op=='ack':extra.setdefault('observation_control_step',extra['controls'][-1]['control_step'])
    return dict(op=op,identity=asdict(s.identity),policy_version=0,policy_sha256='a'*64,
        control_step=s.rollout.step,observation=observation(),**extra)
def controls(n,held=False):return [dict(control_step=i+1,simulator_apply_ack=True,
    action_executed_raw23=[0.]*23,physical_evidence=evidence(held),official_terminal=False,official_truncated=False)
    for i in range(n)]


class TrainingProtocolTests(unittest.TestCase):
    def test_actor_whitelist_and_task_isolation(self):
        s=session();s.action_input(message(s,'action'))
        bad=observation();bad['reward']=1.
        with self.assertRaises(ValueError):actor_observation(bad)
        for key,value in [('task','wrong'),('episode','old'),('context_id','other')]:
            bad=message(s,'action');bad['identity'][key]=value
            with self.assertRaises(ValueError):s.action_input(bad)
        bad=message(s,'action');bad['policy_version']=1
        with self.assertRaises(ValueError):s.action_input(bad)

    def test_real_early_success_keeps_tail_and_exact_actions(self):
        s=session();s.emit(np.zeros((16,23),np.float32),experience_id=8,old_value=.1)
        _,rewards=s.ack(message(s,'ack',controls=controls(6,True)))
        self.assertEqual(len(rewards),6);self.assertTrue(rewards[-1]['skill_success'])
        with self.assertRaises(ValueError):s.finish_ack(.3)
        s.finish_ack(0.)
        self.assertEqual(s.training_targets()[0]['actual_controls'],6)
        self.assertEqual(s.training_targets()[0]['experience_id'],8)
        with self.assertRaises(ValueError):s.action_input(message(s,'action'))

    def test_truncation_bootstraps_and_not_failure(self):
        s=session(3);s.emit(np.zeros((16,23),np.float32),experience_id=3,old_value=0.)
        _,rewards=s.ack(message(s,'ack',controls=controls(3)))
        self.assertTrue(rewards[-1]['truncated']);self.assertEqual(rewards[-1]['outcome'],'UNKNOWN')
        self.assertEqual(s.finish_ack(.7)['next_value'],.7)
        self.assertEqual(s.training_targets()[0]['actual_controls'],3)

    def test_wrong_raw_action_missing_ack_and_partial_chunk_rejected(self):
        for mutation in ('action','ack','clock','skill','partial'):
            s=session();s.emit(np.zeros((16,23),np.float32),experience_id=1,old_value=0.)
            rows=controls(16)
            if mutation=='action':rows[0]['action_executed_raw23'][0]=.1
            if mutation=='ack':rows[0]['simulator_apply_ack']=False
            if mutation=='clock':rows[0]['control_step']=2
            if mutation=='skill':rows[0]['physical_evidence']['semantic_skill_sha256']='b'*64
            if mutation=='partial':rows=rows[:1]
            with self.assertRaises(ValueError):s.ack(message(s,'ack',controls=rows))

    def test_pending_episode_cannot_learn_or_emit_twice(self):
        s=session()
        with self.assertRaises(ValueError):s.training_targets()
        s.emit(np.zeros((16,23),np.float32),experience_id=1,old_value=0.)
        with self.assertRaises(ValueError):s.action_input(message(s,'action'))
        with self.assertRaises(ValueError):s.emit(np.zeros((16,23),np.float32),experience_id=2,old_value=0.)

    def test_next_action_and_bootstrap_use_actual_ack_frame(self):
        s=session();s.emit(np.zeros((16,23),np.float32),experience_id=1,old_value=0.)
        bad=message(s,'ack',controls=controls(16));bad['observation_control_step']=15
        with self.assertRaises(ValueError):s.ack(bad)
        s.ack(message(s,'ack',controls=controls(16)));s.finish_ack(.2)
        s.action_input(message(s,'action'))
        bad=message(s,'action');bad['observation']['images']['head_rgb'][0,0,0]=1
        with self.assertRaises(ValueError):s.action_input(bad)


if __name__=='__main__':unittest.main()
