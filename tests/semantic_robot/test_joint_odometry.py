import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.joint_odometry import solve_joint_correspondences
from semantic_robot.v2.odometry import RGBDMotion
import test_odometry


class JointOdometryTests(unittest.TestCase):
    def args(self, **kwargs):
        return list(test_odometry.OdometryTests().correspondences(**kwargs)[:-1])

    def test_known_motion_both_signs_and_translation(self):
        for yaw in (-.12, 0., .12):
            args = self.args(yaw=yaw)
            result = solve_joint_correspondences(*args)
            self.assertTrue(result["valid"], result)
            np.testing.assert_allclose(result["body_delta"], [.03, -.01, yaw], atol=1e-6)
            self.assertFalse(result["solver_fallback"])
            self.assertTrue(result["fixed_support_no_reselection"])

    def test_duplicate_locations_cannot_create_twenty_five_points(self):
        args = self.args()
        for i in (0, 1, 2):
            args[i] = np.repeat(args[i][:12], 3, axis=0)
        result = solve_joint_correspondences(*args)
        self.assertFalse(result["valid"])
        self.assertEqual(result["unique_matches"], 12)
        self.assertNotIn("body_delta", result)

    def test_outliers_are_robust_but_inconsistent_depth_is_rejected(self):
        args = self.args(); args[1] = args[1].copy(); args[1][:20] += 100.
        self.assertTrue(solve_joint_correspondences(*args)["valid"])
        args = self.args(); args[2] = args[2] + [.07, 0., 0.]
        self.assertFalse(solve_joint_correspondences(*args)["valid"])

    def test_invalid_matrices_and_nonfinite_data(self):
        for index in (0, 1, 2, 3, 4, 5):
            args = self.args(); args[index] = args[index].copy(); args[index].flat[0] = np.nan
            with self.assertRaises(ValueError):
                solve_joint_correspondences(*args)
        args = self.args(); args[4] = np.eye(4); args[4][0, 0] = -1
        with self.assertRaises(ValueError):
            solve_joint_correspondences(*args)

    def test_large_motion_negative_depth_and_sparse_support_abstain(self):
        self.assertFalse(solve_joint_correspondences(*self.args(yaw=.45))["valid"])
        args = self.args(); args[0] = args[0].copy(); args[0][0, 2] = -.1
        self.assertFalse(solve_joint_correspondences(*args)["valid"])
        args = self.args(); args[:3] = [x[:20] for x in args[:3]]
        self.assertFalse(solve_joint_correspondences(*args)["valid"])

    def test_optimizer_failure_has_no_solver_fallback_or_displacement(self):
        with patch("scipy.optimize.least_squares", side_effect=ValueError("bad fit")):
            result = solve_joint_correspondences(*self.args())
        self.assertFalse(result["valid"])
        self.assertNotIn("body_delta", result)
        self.assertFalse(result["solver_fallback"])

    def test_explicit_runtime_route_and_old_default_unchanged(self):
        self.assertIs(RGBDMotion("rgbd_joint").solve, solve_joint_correspondences)
        self.assertNotEqual(RGBDMotion().solve, solve_joint_correspondences)


if __name__ == "__main__":
    unittest.main()
