import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_local_teacher import LocalGraspTeacher,perturb,NonGraspingFixedPoint


class LocalTeacherTests(unittest.TestCase):
    def test_blocked_opening_is_a_measured_fixed_point_not_a_time_limit(self):
        detector=NonGraspingFixedPoint();q=np.zeros(23)
        for i in range(100):
            stopped=detector.observe(q,q+np.sin(i)*1e-5,[1,2,3],.03,dict(left='FALSE',right='FALSE'))
        self.assertTrue(stopped)
        self.assertFalse(detector.observe(q,q,[1,2,3],.03,dict(left='TRUE',right='FALSE')))

    def test_slow_real_progress_does_not_trip_fixed_point(self):
        detector=NonGraspingFixedPoint();q=np.zeros(23)
        for i in range(200):
            moved=q.copy();moved[7]=i*.0001
            self.assertFalse(detector.observe(q,moved,[1,2,3],.03,dict(left='FALSE',right='FALSE')))

    def teacher(self):
        q=np.zeros(23,dtype=np.float32);q[14]=-1;q[22]=1
        return LocalGraspTeacher(q,[1,2,3],arm='left',arm_indices=list(range(7,14)),gripper_index=14)

    def test_same_state_feedback_rate_limit_and_close(self):
        teacher=self.teacher();q=teacher.target_q.copy();q[7]=.1
        action,info=teacher.action(q,[1,2,3],.05,'FALSE')
        self.assertAlmostEqual(float(action[7]),.085,places=6)
        self.assertEqual(action[14],1);self.assertFalse(info['action_quality_approved'])
        action,_=teacher.action(teacher.target_q,[1,2,3],.05,'FALSE')
        self.assertEqual(action[14],-1)

    def test_moved_or_unknown_target_is_not_fake_expert(self):
        for pos,held in (([1.1,2,3],'FALSE'),([1,2,3],'UNKNOWN')):
            with self.assertRaises(ValueError):self.teacher().action(np.zeros(23),pos,.05,held)

    def test_noise_not_mislabelled_intended_and_clipping_accounted(self):
        q=self.teacher().target_q.copy()
        noise,executed=perturb(q,arm_indices=list(range(7,14)),gripper_index=14,
                              rng=np.random.default_rng(17),kind='open_gripper_joint_jitter')
        self.assertEqual(q[14],-1);self.assertEqual(executed[14],1)
        self.assertTrue(np.allclose(executed-q,noise));self.assertTrue(np.array_equal(executed[:7],q[:7]))


if __name__=='__main__':unittest.main()
