from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g05.rl.protocol import validate_worker,validate_phase,next_prefix,paired_summary,check_reset,EVAL_LIMIT


class MethodProtocolTests(unittest.TestCase):
    def spec(self,evaluation):
        row=dict(worker=0,gpu=2,task_id=0,task='turning_on_radio',seed=0,
            split='public_test' if evaluation else 'train',evaluation_only=evaluation,
            instance=301 if evaluation else 1,prefix_controls=0 if evaluation else 1268)
        if not evaluation: row.update(episode=0,actions='/train/demo.npy',first_recorded_terminal=1364)
        return row

    def test_eval_cannot_carry_expert_or_train_role(self):
        row=self.spec(True); validate_worker(row,evaluation=True)
        for patch in ({'actions':'/expert.npy'},{'prefix_controls':16},{'split':'train'},
                      {'evaluation_only':False},{'instance':138},{'task_id':1}):
            with self.assertRaises(ValueError): validate_worker(dict(row,**patch),evaluation=True)
        for phase in ('policy','expert_prefix'):
            with self.assertRaises(ValueError): validate_phase(row,phase)
        validate_phase(row,'eval_parent_fp32')

    def test_training_holdout_and_terminal_prefix_rejected(self):
        row=self.spec(False); validate_worker(row,evaluation=False)
        for patch in ({'episode':190},{'prefix_controls':1364},{'prefix_controls':-1},{'split':'public_test'}):
            with self.assertRaises(ValueError): validate_worker(dict(row,**patch),evaluation=False)
        with self.assertRaises(ValueError): validate_phase(row,'eval_rl_fp32')

    def test_curriculum_requires_three_successes_and_has_floor(self):
        self.assertEqual(next_prefix(1364,1268,[True,True]),1268)
        self.assertEqual(next_prefix(1364,1268,[True,False,True]),1268)
        self.assertEqual(next_prefix(1364,1268,[True,True,True]),1172)
        self.assertEqual(next_prefix(1364,596,[True]*3),596)
        with self.assertRaises(ValueError): next_prefix(1364,1364,[])

    def matrix(self):
        return [dict(variant=v,instance=i,policy_seed=s,environment_seed=0,split='public_test',
                expert_prefix_controls=0,controls=EVAL_LIMIT,control_limit=EVAL_LIMIT,
                success=False,terminated=False,truncated=False)
                for v in ('parent_fp32','rl_fp32') for i in (301,302) for s in (17,23,41)]

    def test_complete_paired_raw_counts(self):
        rows=self.matrix(); rows[0].update(success=True,terminated=True,controls=2000)
        rows[6].update(success=True,terminated=True,controls=2100)
        rows[7].update(success=True,terminated=True,controls=2200)
        result=paired_summary(rows)
        self.assertEqual((result['before_successes'],result['after_successes']),(1,2))
        self.assertEqual((result['improved_pairs'],result['regressed_pairs']),(1,0))

    def test_partial_duplicate_prefix_and_short_failure_not_sr(self):
        rows=self.matrix()
        for bad in (rows[:-1],rows+[rows[-1]]):
            with self.assertRaises(ValueError): paired_summary(bad)
        for patch in ({'expert_prefix_controls':1},{'controls':128},{'invalid':True},
                      {'success':True,'terminated':False},{'environment_seed':1}):
            bad=deepcopy(rows); bad[-1].update(patch)
            with self.assertRaises(ValueError): paired_summary(bad)

    def test_reset_identity_and_finite_physics(self):
        ref={'radio':[1.,2.],'robot':[0.,0.]}
        self.assertLess(check_reset(ref,dict(ref,radio=[1.,2.000001])),1e-5)
        for bad in ({'robot':[0.,0.]},dict(ref,radio=[1.,3.]),dict(ref,radio=[1.,float('nan')])):
            with self.assertRaises(ValueError): check_reset(ref,bad)

    def test_independent_worker_noise_does_not_depend_on_other_episode_length(self):
        def noise(interleave):
            generators={w:torch.Generator().manual_seed(17) for w in (0,1)}
            values=[]
            for n in range(3):
                for _ in range(interleave): torch.randn(1,32,27,generator=generators[0])
                values.append(torch.randn(1,32,27,generator=generators[1]))
            return torch.stack(values)
        self.assertTrue(torch.equal(noise(0),noise(9)))


if __name__=='__main__': unittest.main()
