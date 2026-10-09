from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest
from skill_aligned_reward import SkillIdentity, SkillReward, skill_measurement


def ident():return SkillIdentity('s','task',1,'ep','intent0','a'*64,0)
def obs(phi=0.,success=False):return dict(potential=phi,achieved=success)
def step(reward,t,phi=0.,success=False,**kw):
    return reward.advance(ident(),t,obs(phi,success),protected_values=kw.pop('protected_values',{}),**kw)


class RewardTests(unittest.TestCase):
    def test_wrong_task_intent_and_clock_rejected(self):
        for bad in (replace(ident(),task='other'),replace(ident(),context_id='retry'),replace(ident(),episode='new')):
            with self.assertRaises(ValueError):SkillReward(ident(),obs(),control_step=0).advance(bad,1,obs(),protected_values={})
        with self.assertRaises(ValueError):step(SkillReward(ident(),obs(),control_step=0),16)

    def test_one_success_bonus_requires_actual_distinct_controls(self):
        r=SkillReward(ident(),obs(),control_step=0,stable_controls=3)
        self.assertFalse(step(r,1,1.,True)['skill_success'])
        self.assertFalse(step(r,2,1.,True)['skill_success'])
        out=step(r,3,1.,True)
        self.assertTrue(out['skill_success']);self.assertFalse(out['bootstrap'])
        with self.assertRaises(ValueError):step(r,4,1.,True)

    def test_timeout_is_not_failure_and_keeps_bootstrap(self):
        out=step(SkillReward(ident(),obs(),control_step=0),1,.4,time_limit=True)
        self.assertEqual(out['outcome'],'UNKNOWN');self.assertEqual(out['sparse'],0.)
        self.assertTrue(out['bootstrap']);self.assertTrue(out['truncated'])

    def test_potential_telescopes_on_terminal_not_reward_farming(self):
        gamma=.99;r=SkillReward(ident(),obs(.2),control_step=0,gamma_per_control=gamma)
        results=[step(r,1,.8),step(r,2,.1),step(r,3,.9,official_terminal=True)]
        self.assertAlmostEqual(sum(gamma**i*x['shaping'] for i,x in enumerate(results)),-.2*.2)

    def test_preserved_progress_loss_is_not_success(self):
        r=SkillReward(ident(),obs(),control_step=0,protected_facts=('held_other_object',),stable_controls=2)
        step(r,1,1.,True,protected_values={'held_other_object':False})
        out=step(r,2,1.,True,protected_values={'held_other_object':False})
        self.assertTrue(out['physical_failure']);self.assertFalse(out['skill_success'])
        self.assertEqual(out['sparse'],-1.)

    def test_grasp_requested_hand_not_gripper_closure(self):
        s=dict(verb='GRASP',arm='LEFT'); e=dict(semantic_skill_sha256=digest(s),exact_binding_verified=True,
            arm='LEFT',target_held_by_requested_arm=False,requested_eef_target_distance_m=0.)
        self.assertFalse(skill_measurement(s,e)['achieved'])
        with self.assertRaises(ValueError):skill_measurement(s,dict(e,arm='RIGHT'))
        with self.assertRaises(ValueError):skill_measurement(s,dict(e,target_held_by_requested_arm=None))

    def test_heterogeneous_contracts_require_real_completion(self):
        for verb in ('PLACE_IN','PLACE_ON'):
            s=dict(verb=verb);e=dict(semantic_skill_sha256=digest(s),exact_binding_verified=True,
                relation='inside' if verb=='PLACE_IN' else 'ontop',official_relation=True,
                released_from_all_hands=False,object_speed_mps=0.,target_destination_distance_m=0.)
            self.assertFalse(skill_measurement(s,e)['achieved'])
            self.assertTrue(skill_measurement(s,dict(e,released_from_all_hands=True))['achieved'])
        s=dict(verb='OPEN_DOOR');e=dict(semantic_skill_sha256=digest(s),exact_binding_verified=True,
                                     official_open=True,directed_open_fraction=.05)
        self.assertFalse(skill_measurement(s,e)['achieved'])
        self.assertTrue(skill_measurement(s,dict(e,directed_open_fraction=.4))['achieved'])
        s=dict(verb='NAVIGATE');e=dict(semantic_skill_sha256=digest(s),exact_binding_verified=True,
            reachable_pose_distance_m=.03,heading_error_rad=.1,reachable_pose_verified=True,collision_free=True)
        self.assertTrue(skill_measurement(s,e)['achieved'])
        with self.assertRaises(ValueError):skill_measurement(s,dict(e,reachable_pose_verified=False))

    def test_no_rewards_for_already_solved_seed_or_unknown_predicate(self):
        with self.assertRaises(ValueError):SkillReward(ident(),obs(1.,True),control_step=0)
        with self.assertRaises(ValueError):step(SkillReward(ident(),obs(),control_step=0),1,float('nan'))


if __name__=='__main__':unittest.main()
