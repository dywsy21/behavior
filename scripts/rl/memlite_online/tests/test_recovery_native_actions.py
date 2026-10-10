from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_native_actions import checked_action_window
from recovery_corpus import digest


class NativeLearnerActionTests(unittest.TestCase):
    def rows(self):
        return [dict(control_step=t,simulator_apply_ack=True,policy_version=25,action_source='train',
            action_executed_raw23=np.full(23,t/100,dtype=np.float32).tolist(),
            skill_reward=dict(terminated=t==132,truncated=False,identity={'episode':'bound','bundle':'same'}))
            for t in range(101,133)]

    def read(self, rows, at=100):
        return checked_action_window(rows,at=at,identity={'episode':'bound','bundle':'same'},policy_version=25)

    def test_native_post_control_clock_exact23_no_terminal_padding(self):
        rows=self.rows();action,sha=self.read(rows)
        self.assertEqual(action.shape,(32,23));self.assertEqual(action.dtype,np.float32)
        self.assertEqual(action.tolist(),[r['action_executed_raw23'] for r in rows])
        self.assertEqual(sha,digest(action.tolist()))
        with self.assertRaises(ValueError):self.read(rows,101)
        with self.assertRaises(ValueError):self.read(rows[:-1])

    def test_cross_source_policy_and_nontraining_actions_fail(self):
        for mutate in (lambda r:r[10].update(simulator_apply_ack=False),
                       lambda r:r[10].update(policy_version=26),
                       lambda r:r[10].update(action_source='evaluation'),
                       lambda r:r[10]['skill_reward']['identity'].update(episode='foreign'),
                       lambda r:r[10]['skill_reward']['identity'].update(bundle='different'),
                       lambda r:r[10]['skill_reward'].update(terminated=True)):
            rows=self.rows();mutate(rows)
            with self.assertRaises(ValueError):self.read(rows)

    def test_dense_or_padded_or_nonfinite_or_requantized_actions_fail(self):
        for change in ([0.]*27,[0.]*22,[float('nan')]*23,[.1]*23):
            rows=self.rows();rows[0]['action_executed_raw23']=change
            with self.assertRaises(ValueError):self.read(rows)

    def test_duplicate_or_float_clock_fails(self):
        rows=self.rows()
        with self.assertRaises(ValueError):self.read(rows+[deepcopy(rows[-1])])
        rows[0]['control_step']=101.
        with self.assertRaises(ValueError):self.read(rows)


if __name__=='__main__':unittest.main()
