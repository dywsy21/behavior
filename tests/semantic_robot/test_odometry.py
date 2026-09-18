import copy
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.v2.odometry import RGBDMotion, body_motion, solve_correspondences
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import HOLD, Action
from semantic_robot.v2.search import CoverageSearch
from semantic_robot.v2.servo import SafeServo
from test_v2 import fixture, feedback


class OdometryTests(unittest.TestCase):
    def correspondences(self,yaw=.12,translation=(.03,-.01,0.)):
        rng=np.random.default_rng(413)
        K=np.array([[400.,0,320],[0,400.,320],[0,0,1.]])
        xyz=rng.uniform([-1,-1,1.5],[1,1,3.5],(120,3))
        camera=np.eye(4);camera[:3,3]=[.1,0,1.3]
        body=np.eye(4);body[:3,:3]=Rotation.from_euler("z",yaw).as_matrix();body[:3,3]=translation
        S=np.diag([1.,-1.,-1.,1.]);rel=S@np.linalg.inv(camera)@np.linalg.inv(body)@camera@S
        new=xyz@rel[:3,:3].T+rel[:3,3];uv=(new@K.T)[:,:2]/new[:,2,None]
        return xyz,uv,new,K,camera,camera,body

    def test_known_body_motion_and_depth_not_velocity_scale(self):
        for yaw in (.12,-.12,0.):
            for translation in ((.03,-.01,0.),(0.,0.,0.)):
                args=self.correspondences(yaw,translation);r=solve_correspondences(*args[:-1])
                self.assertTrue(r["valid"],r)
                np.testing.assert_allclose(r["body_delta"],[*translation[:2],yaw],atol=1e-6)

    def test_outliers_and_depth_inconsistency(self):
        args=list(self.correspondences()[:-1]);args[1]=args[1].copy();args[1][:20]+=100
        self.assertTrue(solve_correspondences(*args)["valid"])
        args=list(self.correspondences()[:-1]);args[2]=args[2]+[.07,0.,0.]
        self.assertEqual(solve_correspondences(*args)["reason"],"RGB_DEPTH_MOTION_DISAGREEMENT")
        self.assertFalse(solve_correspondences(*self.correspondences(yaw=.45)[:-1])["valid"])
        self.assertFalse(solve_correspondences([],[],[],np.eye(3),np.eye(4),np.eye(4))["valid"])

    def test_camera_joint_motion_does_not_count_as_body_motion(self):
        old=np.eye(4);new=np.eye(4);new[:3,:3]=Rotation.from_euler("x",.2).as_matrix();new[:3,3]=[.1,0.,0.]
        S=np.diag([1.,-1.,-1.,1.]);rel=S@np.linalg.inv(new)@old@S
        motion,delta=body_motion(old,new,rel)
        np.testing.assert_allclose(motion,np.eye(4),atol=1e-12)
        np.testing.assert_allclose(delta,0.,atol=1e-12)

    def test_textureless_views_abstain_and_snapshot_is_copied(self):
        model,state=fixture();est=RGBDMotion()
        images={"head_rgb":np.zeros((100,100,3),np.uint8)};depths={"head":np.ones((100,100),np.float32)}
        self.assertTrue(est.observe(images,depths,model,state.q)["initial"])
        depths["head"][:]=2.
        self.assertTrue(np.all(est.previous["depth"]==1.))
        receipt=est.observe(images,depths,model,state.q)
        self.assertFalse(receipt["valid"])
        self.assertTrue(receipt["no_scene_truth"])

    def test_controller_defers_coverage_until_real_image_motion_and_preserves_raw(self):
        model,state=fixture();h=GroundedHarness([Goal("navigate","table","both","near table")])
        c=GroundedController(model,SafeServo(model,state),h,visual_odometry=True)
        c.motion=Mock()
        c.motion.observe.side_effect=[{"valid":True,"initial":True,"body_delta":[0,0,0]},
                                      {"valid":True,"body_delta":[.001,0.,.12]}]
        c.update_motion({}, {},state)
        raw=feedback();raw["base_integral"]=[.03,0.,.16];preserved=copy.deepcopy(raw)
        c.executed(HOLD,raw)
        self.assertEqual(c.search.heading,0.)
        c.update_motion({}, {},state)
        self.assertAlmostEqual(c.search.heading,.12)
        self.assertEqual(raw,preserved)
        self.assertEqual(h.feedback["base_velocity_integral_raw"],[.03,0.,.16])
        self.assertEqual(h.feedback["base_integral"],[.001,0.,.12])

    def test_failed_visual_motion_never_falls_back_to_wrong_coverage(self):
        model,state=fixture();h=GroundedHarness([Goal("navigate","table","both","near table")])
        c=GroundedController(model,SafeServo(model,state),h,visual_odometry=True);c.motion=Mock()
        c.motion.observe.return_value={"valid":False,"reason":"NO_TEXTURE"}
        raw=feedback();raw["base_integral"]=[0,0,.16];c.executed(HOLD,raw)
        c.update_motion({}, {},state)
        self.assertEqual(h.stop_reason,"VISUAL_ODOMETRY_UNCERTAIN")
        self.assertEqual(c.search.heading,0.)

    def test_visual_displacement_has_previous_body_frame_not_midpoint_velocity_frame(self):
        search=CoverageSearch()
        search.executed({"base_integral":[.1,0.,.2],"base_motion_convention":"displacement_in_previous_body_frame"})
        np.testing.assert_allclose(search.xy,[.1,0.])

    def test_visual_base_residual_can_correct_only_old_velocity_based_status(self):
        for old,yaw,expected in (("BASE_TRACKING_FAILED",.12,"TARGET_REACHED"),
                                 ("TARGET_REACHED",0.,"BASE_TRACKING_FAILED"),
                                 ("TRACKING_DIVERGED",.12,"TRACKING_DIVERGED")):
            model,state=fixture();h=GroundedHarness([Goal("navigate","table","both","near table")])
            c=GroundedController(model,SafeServo(model,state),h,visual_odometry=True);c.motion=Mock()
            c.motion.observe.return_value={"valid":True,"body_delta":[0,0,yaw]}
            raw=feedback(status=old);raw["base_integral"]=[0,0,.3]
            c.executed(Action("base","yaw_plus","coarse"),raw)
            c.update_motion({}, {},state)
            self.assertEqual(h.feedback["status"],expected)
            self.assertEqual(raw["status"],old)


if __name__=="__main__":unittest.main()
