import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_articulation_teacher import CausalArticulation,servo,functional_goal


class ArticulationTests(unittest.TestCase):
    def test_no_segment_end_timeout_or_unknown_success(self):
        state=CausalArticulation()
        self.assertEqual([state.update(i,True) for i in range(30)],['UNKNOWN']*30)
        self.assertEqual(state.update(30,None),'UNKNOWN')
        with self.assertRaises(ValueError):state.update(32,True)

    def test_real_goal_loss_then_corrective_progress(self):
        state=CausalArticulation();state.update(0,False)
        self.assertEqual([state.update(i,True,moving=True) for i in range(1,13)][-1],'SUCCEEDED')
        self.assertEqual([state.update(i,False) for i in range(13,25)][-1],'FAILED')
        self.assertEqual(state.update(25,False,retry=True,moving=True),'IN_PROGRESS')
        self.assertEqual([state.update(i,True) for i in range(26,38)][-1],'SUCCEEDED')

    def test_raw23_body_frame_and_rate_limits(self):
        q=np.zeros(23);target=q.copy();target[7]=1;target[14]=-1
        action,info=servo(q,[0,0,np.pi/2],dict(q=target,base=[.1,0,np.pi/2]))
        self.assertAlmostEqual(action[0],0,places=6);self.assertLess(action[1],0)
        self.assertAlmostEqual(action[7],.015,places=6);self.assertEqual(action[14],-1)
        self.assertFalse(info['reached'])

    def test_barely_open_does_not_become_a_success_and_hysteresis(self):
        self.assertFalse(functional_goal(True,.06,True))
        self.assertIsNone(functional_goal(True,.25,True))
        self.assertTrue(functional_goal(True,.40,True))
        self.assertIsNone(functional_goal(True,.04,False))
        self.assertTrue(functional_goal(False,.01,False))
        with self.assertRaises(ValueError):functional_goal(True,1.5,True)


if __name__=='__main__':unittest.main()
