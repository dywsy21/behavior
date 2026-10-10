from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from skill_return_diagnostic import decompose_returns


class ReturnDiagnosticTests(unittest.TestCase):
    def records(self):
        return [dict(identity=dict(episode='a'),policy_version=1,policy_sha256='a'*64,
            experience_id=i,start_control=i,end_control=i+1,controls=1,reward=.1,discount=.9,
            trace_discount=.8,old_value=.4,next_value=.4,terminated=False,truncated=i==1) for i in range(2)]

    def test_decomposition_and_counterfactual_are_exact_and_nonmutating(self):
        rows=self.records();before=deepcopy(rows);got=decompose_returns(rows)
        self.assertEqual(rows,before)
        self.assertAlmostEqual(got['observed_discounted_reward'],.19)
        self.assertAlmostEqual(got['discounted_final_bootstrap'],.324)
        self.assertAlmostEqual(got['bootstrapped_monte_carlo_return'],.514)
        self.assertAlmostEqual(got['chunk_diagnostics'][0]['advantage'],.108)
        self.assertAlmostEqual(got['chunk_diagnostics'][0]['final_bootstrap_advantage_component'],.288)
        self.assertEqual(got['positive_advantage_fraction'],1.)
        self.assertEqual(got['zero_tail_positive_advantage_fraction'],0.)

    def test_terminal_success_has_no_final_bootstrap_contribution(self):
        rows=self.records();rows[-1].update(terminated=True,truncated=False,discount=0.,trace_discount=0.,next_value=0.,reward=1.)
        got=decompose_returns(rows)
        self.assertEqual(got['discounted_final_bootstrap'],0.)
        self.assertEqual(got['mean_final_bootstrap_advantage_component'],0.)

    def test_no_missing_controls_cross_episode_or_incomplete_inputs(self):
        for key,value in [('identity',dict(episode='b')),('start_control',3),('policy_version',2),
                          ('old_value',.2),('experience_id',0),('truncated',False)]:
            rows=self.records();rows[-1][key]=value
            with self.assertRaises(ValueError):decompose_returns(rows)


if __name__=='__main__':unittest.main()
