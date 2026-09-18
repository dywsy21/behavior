import unittest
from dataclasses import replace
import numpy as np
from PIL import Image
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.policy import verification_inputs
from semantic_robot.v2.protocol import Action,HOLD
from semantic_robot.v2.vision import VisualBundle,clip_segment
from test_v2 import fixture,evidence,feedback


class GraspFeedbackTests(unittest.TestCase):
    def make(self):
        _,s=fixture();s.gripper[:]=[.05,.045]
        h=GroundedHarness([Goal("pick","visible object","right","moves with hand")])
        h.stage="VERIFY_GRASP";h.stage_age=3;h.pending_grasp["right"]=True
        h.observation=evidence(enclosed=True);h.last_action=Action("right","up")
        h.feedback=feedback();h.feedback["eef_delta_m"]["right"]=[0,0,.01]
        return h,s

    def test_unknown_after_nonempty_close_preserves_load_not_success(self):
        h,s=self.make();h.observe(evidence(enclosed=True,co_moving=None),s,
                                  measured_progress={"metric_co_motion":True})
        self.assertIn("UNVERIFIED_LOAD_PRESERVED",h.stop_reason)
        self.assertEqual(h.palette(),(HOLD,));self.assertFalse(h.completed)
        self.assertTrue(h.pending_grasp["right"])

    def test_empty_probe_can_recover_by_opening(self):
        h,s=self.make();s.gripper[1]=1e-7
        h.observe(evidence(enclosed=None),s)
        self.assertEqual(h.stage,"RECOVER");self.assertIsNone(h.stop_reason)
        self.assertIn(Action("right","open"),h.palette())

    def test_one_nonempty_hand_prevents_bimanual_release(self):
        h,s=self.make();h.goals=[Goal("pick","plate","both","moves",True)]
        h.pending_grasp["left"]=True;s.gripper[:]=[0.,.045]
        h.observe(evidence(enclosed=None),s)
        self.assertEqual(h.palette(),(HOLD,));self.assertIn("PRESERVED",h.stop_reason)

    def test_unknown_aperture_and_interrupted_close_preserve_latch(self):
        h,s=self.make();h.stage="GRASP";h.pending_grasp["right"]=False
        h.executed(Action("right","close"),{"status":"TRACKING_DIVERGED","control_ticks":2})
        self.assertTrue(h.pending_grasp["right"])
        h.recover("TRACKING_DIVERGED");self.assertEqual(h.palette(),(HOLD,))

    def test_all_original_verification_gates_still_required(self):
        for metric,visual in ((False,True),(True,False),(True,None)):
            h,s=self.make();h.stage_age=0
            h.observe(evidence(enclosed=True,co_moving=visual),s,measured_progress={"metric_co_motion":metric})
            self.assertFalse(h.completed)
        h,s=self.make();h.observe(evidence(enclosed=True,co_moving=True),s,
                                 measured_progress={"metric_co_motion":True})
        self.assertEqual(len(h.completed),1);self.assertTrue(h.hold_verified["right"])
        self.assertFalse(h.pending_grasp["right"])

    def test_verification_pairs_keep_exact_pixels_no_new_images(self):
        h,_=self.make();labels=[t+"_"+v+"_"+s for v in ("HEAD","LEFT_WRIST","RIGHT_WRIST")
            for t,s in (("PREVIOUS","RAW"),("CURRENT","RAW"),("CURRENT","ROBOT_GUIDE_NOT_OBJECT_LABELS"))]
        imgs=[Image.fromarray(np.full((8,8,3),i,np.uint8)) for i in range(9)]
        b=VisualBundle(imgs,labels,{},{});sub,prompt=verification_inputs(h,b)
        self.assertEqual(sub.labels,["PREVIOUS_RIGHT_WRIST_RAW","CURRENT_RIGHT_WRIST_RAW","PREVIOUS_HEAD_RAW","CURRENT_HEAD_RAW"])
        for label,img in zip(sub.labels,sub.images):self.assertIs(img,imgs[labels.index(label)])
        self.assertIn("stationary",prompt)
        h.stage="ALIGN";self.assertIs(verification_inputs(h,b)[0],b)

    def test_segment_clipping_is_intersection_not_endpoint_clamping(self):
        a,b=clip_segment([-10,30],[30,10],[40,40]);np.testing.assert_allclose(a,[0,25]);np.testing.assert_allclose(b,[30,10])
        self.assertIsNone(clip_segment([-10,2],[-5,30],[40,40]))
        self.assertIsNone(clip_segment(None,[1,2],[40,40]))
        a,b=clip_segment([-10,20],[50,20],[40,40]);np.testing.assert_allclose([a,b],[[0,20],[39,20]])


if __name__=="__main__":unittest.main()
