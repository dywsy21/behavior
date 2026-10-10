from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('_rl_action_windows',Path(__file__).resolve().parents[1]/'tools/review_short_skill_action_windows.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RlActionWindowTests(unittest.TestCase):
    def records(self):
        controls=[dict(control_step=t,simulator_apply_ack=True,policy_version=2,
            action_executed_raw23=[float(t)]*23,skill_reward=dict(terminated=t==150,truncated=False)) for t in range(101,151)]
        observations=[dict(control_step=t,observation_sha256=str(t)) for t in (100,116,132,148,150)]
        return controls,observations

    def test_real_observation_clock_and_short_tail(self):
        controls,observations=self.records();result=module.action_windows(controls,observations)
        self.assertEqual((result[0]['first_action_ack'],result[0]['last_action_ack']),(101,132))
        self.assertEqual([r['complete_32_applied_actions'] for r in result],[True,True,False,False,False])
        self.assertTrue(all(r['admitted_for_training'] is False for r in result))

    def test_no_gap_unacked_terminal_or_policy_switch(self):
        for kind in ('gap','unacked','terminal','switch'):
            controls,observations=self.records()
            if kind=='gap':del controls[10]
            elif kind=='unacked':controls[10]['simulator_apply_ack']=False
            elif kind=='terminal':controls[10]['skill_reward']['terminated']=True
            else:controls[10]['policy_version']=3
            with self.assertRaises(ValueError):module.action_windows(controls,observations)

    def test_missing_duplicate_or_shifted_observation(self):
        controls,obs=self.records()
        for bad in (obs[1:],obs[:-1],obs+[deepcopy(obs[-1])]):
            with self.assertRaises(ValueError):module.action_windows(controls,bad)


if __name__=='__main__':unittest.main()
