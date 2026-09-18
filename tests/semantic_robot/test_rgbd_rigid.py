import unittest
import numpy as np
from semantic_robot.v2.odometry import rigid_fit,solve_rgbd_correspondences,RGBDMotion
import test_odometry


class RGBDRigidTests(unittest.TestCase):
    def test_known_motion_planar_support_and_outliers(self):
        args=test_odometry.OdometryTests().correspondences();raw=list(args[:-1]);truth=args[-1]
        for outliers in (0,30):
            data=[x.copy() for x in raw];data[2][:outliers]+=[.3,.1,0]
            r=solve_rgbd_correspondences(*data)
            self.assertTrue(r["valid"],r)
            np.testing.assert_allclose(r["body_transform_current_in_previous"],truth,atol=1e-9)
        p=np.c_[np.linspace(0,1,30),np.sin(np.arange(30)),np.ones(30)]
        fit=rigid_fit(p,p+[.1,-.1,.03]);np.testing.assert_allclose(fit[1],[.1,-.1,.03],atol=1e-10)

    def test_depth_rgb_contradiction_motion_bounds_and_collinear_abstention(self):
        args=list(test_odometry.OdometryTests().correspondences()[:-1]);args[2]+=np.array([.07,0,0])
        self.assertEqual(solve_rgbd_correspondences(*args)["reason"],"RGB_DEPTH_MOTION_DISAGREEMENT")
        self.assertFalse(solve_rgbd_correspondences(*test_odometry.OdometryTests().correspondences(yaw=.45)[:-1])["valid"])
        a=np.c_[np.arange(30),np.zeros((30,2))]
        self.assertIsNone(rigid_fit(a,a))
        with self.assertRaises(ValueError):RGBDMotion("automatic_accepting_fallback")


if __name__=="__main__":unittest.main()
