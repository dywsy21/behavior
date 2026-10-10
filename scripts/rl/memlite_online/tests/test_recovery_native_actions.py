from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_native_actions import checked_action_window,native_low_raw
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

    def test_raw_low_projection_only_current_observation_and_same_skill(self):
        import types
        import torch
        from recovery_corpus import canonical
        observation=dict(proprio=np.zeros(61,dtype=np.float32),images={k:np.zeros((3,32,32),dtype=np.uint8)
            for k in ('head_rgb','left_wrist_rgb','right_wrist_rgb')})
        goal=dict(task='some task',parent_goal='Task goal: some task',semantic_bundle=canonical([
            dict(verb='GRASP',target='cup',source='',destination='',target_part='',arm='LEFT',unbound_relation='')]))
        action=np.arange(32*23,dtype=np.float32).reshape(32,23)
        reader=types.SimpleNamespace(read=lambda _: (observation,action,goal))
        config=dict(raw_shape=dict(state=[dict(key='actual_state',start_index=0,raw_shape=61)],
            action=[dict(key='actual_controls',start_index=0,raw_shape=23)]))
        raw=native_low_raw(reader,0,config)
        self.assertEqual(raw['model_projection']['memlite_branch'],'low')
        self.assertTrue(torch.equal(raw['action']['actual_controls'],torch.from_numpy(action)))
        self.assertFalse(raw['action_is_pad'].any())
        for forbidden in ('reward','physical_evidence','source_policy_sha256','source_group','outcome_target'):
            self.assertNotIn(forbidden,raw)
        reader.read=lambda _: (dict(observation,reward=1),action,goal)
        with self.assertRaises(ValueError):native_low_raw(reader,0,config)


if __name__=='__main__':unittest.main()
