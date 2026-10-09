from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from skill_aligned_reward import SkillIdentity,SkillReward
from skill_rollout import SkillRollout


def ident():return SkillIdentity('s','task',1,'e','ctx','b'*64,0)
def state(phi=0,held=False):return dict(potential=phi,achieved=held)


class RolloutTests(unittest.TestCase):
    def start(self,step=0):
        return SkillRollout(ident(),policy_version=0,policy_sha256='a'*64,control_step=step,lambda_per_control=.9)
    def begin(self,r,i,value=0,n=16):
        r.begin_chunk(experience_id=i,old_value=value,policy_version=0,policy_sha256='a'*64,control_step=r.step,max_controls=n)

    def test_variable_ack_clock_and_discounted_reward(self):
        r=self.start();self.begin(r,0,n=3)
        physical=SkillReward(ident(),state(),control_step=0,gamma_per_control=.99)
        values=[]
        for t in range(1,4):
            row=physical.advance(ident(),t,state(t/10),protected_values={});r.acknowledge(row);values.append(row['reward'])
        chunk=r.finish_chunk(next_value=.4,observation_control_step=3)
        self.assertAlmostEqual(chunk['reward'],sum(.99**i*v for i,v in enumerate(values)))
        self.assertAlmostEqual(chunk['discount'],.99**3)
        self.assertAlmostEqual(chunk['trace_discount'],(.99*.9)**3)
        self.assertEqual(chunk['controls'],3)

    def test_early_success_retains_short_tail_and_zero_bootstrap(self):
        r=self.start();self.begin(r,0)
        physical=SkillReward(ident(),state(),control_step=0,stable_controls=2)
        for t in (1,2):r.acknowledge(physical.advance(ident(),t,state(1,True),protected_values={}))
        with self.assertRaises(ValueError):r.finish_chunk(next_value=.4,observation_control_step=2)
        last=r.finish_chunk(next_value=0,observation_control_step=2)
        self.assertTrue(last['terminated']);self.assertEqual(r.advantages()[0]['actual_controls'],2)
        with self.assertRaises(ValueError):self.begin(r,1)

    def test_truncation_bootstraps_without_fake_failure(self):
        r=self.start();self.begin(r,0,value=.3)
        physical=SkillReward(ident(),state(),control_step=0)
        row=physical.advance(ident(),1,state(.4),protected_values={},time_limit=True);r.acknowledge(row)
        chunk=r.finish_chunk(next_value=.7,observation_control_step=1)
        self.assertFalse(chunk['terminated']);self.assertTrue(chunk['truncated'])
        self.assertAlmostEqual(r.advantages()[0]['returns'],row['reward']+.999*.7)
        with self.assertRaises(ValueError):self.begin(r,1)

    def test_gae_stays_within_same_attempt_and_policy(self):
        r=self.start();physical=SkillReward(ident(),state(),control_step=0)
        for i in range(2):
            self.begin(r,i,value=.1*i,n=1)
            r.acknowledge(physical.advance(ident(),i+1,state(.1*(i+1)),protected_values={}))
            r.finish_chunk(next_value=.1*(i+1),observation_control_step=i+1)
        values=r.advantages();a,b=r.records
        delta_b=b['reward']+b['discount']*.2-.1
        self.assertAlmostEqual(values[0]['advantage'],a['reward']+a['discount']*.1+a['trace_discount']*delta_b)
        r.records[1]['identity']['episode']='other'
        with self.assertRaises(ValueError):r.advantages()

    def test_reused_cross_context_or_stale_rollouts_rejected(self):
        r=self.start();self.begin(r,0)
        physical=SkillReward(replace(ident(),context_id='wrong'),state(),control_step=0)
        with self.assertRaises(ValueError):r.acknowledge(physical.advance(replace(ident(),context_id='wrong'),1,state(),protected_values={}))
        with self.assertRaises(ValueError):r.finish_chunk(next_value=0,observation_control_step=0)
        other=self.start()
        with self.assertRaises(ValueError):other.begin_chunk(experience_id=0,old_value=0,policy_version=1,
            policy_sha256='a'*64,control_step=0)


if __name__=='__main__':unittest.main()
