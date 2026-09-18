import copy
from unittest.mock import patch
import unittest
import numpy as np
from semantic_robot.v2.grasp_motion import GraspMotionVerifier,robot_point_mask,measure_grasp_motion,point_tracks
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import Action
from test_v2 import fixture,evidence,feedback


def boxes():
    return {"source":"robot_visual_link_boxes_actual_joint_fk","scene_truth":False,
        "includes_actual_finger_positions":True,"margin_m":.003,
        "boxes":[{"T_base_link":np.eye(4).tolist(),"lower":[-1,-1,-1],"upper":[1,1,1]}]}


class GraspMotionTests(unittest.TestCase):
    def test_robot_only_exclusion_and_missing_finger_metadata(self):
        b=boxes();np.testing.assert_array_equal(robot_point_mask([[0,0,0],[2,2,2]],b),[True,False])
        self.assertEqual(len(robot_point_mask([],b)),0)
        for field,value in (("scene_truth",True),("includes_actual_finger_positions",False),("source","scene_objects"),("boxes",[])):
            bad=copy.deepcopy(b);bad[field]=value
            self.assertIsNone(robot_point_mask([[0,0,0]],bad))

    def test_identical_complete_images_are_not_temporal_motion(self):
        m,s=fixture();cam=m.spec["metadata"]["cameras"]["head"]
        img=np.zeros((100,100,3),np.uint8);dep=np.ones((100,100))
        r,a,b=point_tracks(img,img,dep,dep,cam,np.eye(4),np.eye(4),[0,0,1])
        self.assertFalse(r["valid"]);self.assertIsNone(a)

    def test_geometric_rigid_vs_stationary_points_and_unmasked_abstention(self):
        m,s=fixture();q1=s.q.copy();q1[13]=.01
        # right arm z is q13 in this fixture.
        p0=np.c_[np.linspace(.39,.41,12),np.full(12,-.30),np.full(12,.81)]
        frame=lambda q:{"q":q,"rgb":{"right_wrist":None},"depth":{"right_wrist":None},"self_geometry":boxes()}
        for delta,masked,expected in ((.01,True,True),(0.,True,False),(.01,False,False)):
            base={"valid":True,"robot_self_exclusion_checked":masked}
            p1=p0+[0,0,delta]
            with patch("semantic_robot.v2.grasp_motion.point_tracks",return_value=(base,p0,p1)):
                r=measure_grasp_motion(frame(s.q),frame(q1),m,"right",p0.mean(axis=0),np.eye(4))
            self.assertEqual(r["registered_pair_consistent"],expected)

    def make(self,both=False):
        m,s=fixture();s.gripper[:]=.035
        h=GroundedHarness([Goal("pick","object","both" if both else "right","moves")]);h.stage="VERIFY_GRASP"
        h.pending_grasp={"left":both,"right":True};h.last_action=Action(h.goal.hand,"up")
        h.feedback=feedback();h.feedback["eef_delta_m"]["right"]=[0,0,.008]
        v=GraspMotionVerifier();targets={a:{"valid":True,"point_base_m":[.4,0,.8]} for a in h.arms}
        motion={"valid":True,"body_transform_current_in_previous":np.eye(4).tolist()}
        return m,s,h,v,targets,motion

    def observe(self,x,obs=None):
        m,s,h,v,t,motion=x;h.executions+=1
        return v.observe(m,s,{}, {},boxes(),h,t,motion,obs or evidence(enclosed=True,co_moving=None))

    def test_requires_repeated_displacement_not_one_good_frame(self):
        x=self.make();row={"registered_pair_consistent":True,"hand_delta_m":[0,0,.008]}
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",return_value=row):
            self.assertFalse(self.observe(x)["verified"])
            self.assertFalse(self.observe(x)["verified"])
            self.assertTrue(self.observe(x)["verified"])
            m,s,h,v,t,motion=x
            r=v.observe(m,s,{}, {},boxes(),h,t,motion,evidence(enclosed=True,co_moving=None))
            self.assertFalse(r["verified"])

    def test_bad_pair_clears_accumulated_evidence(self):
        x=self.make();row={"registered_pair_consistent":True,"hand_delta_m":[0,0,.008]}
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",return_value=row):
            self.observe(x);self.observe(x)
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",return_value={"valid":False}):
            self.assertFalse(self.observe(x)["verified"])
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",return_value=row):
            self.assertFalse(self.observe(x)["verified"])

    def test_empty_hand_contradiction_or_wrong_motion_cannot_verify(self):
        for case in ("empty","contradiction","hold","different_goal","no_body_motion"):
            x=self.make();m,s,h,v,t,motion=x;self.observe(x)
            obs=evidence(enclosed=True,co_moving=None)
            if case=="empty":s.gripper[1]=0.
            if case=="contradiction":obs=evidence(enclosed=True,co_moving=False)
            if case=="hold":h.last_action=Action("all","hold")
            if case=="different_goal":h.index=1
            if case=="no_body_motion":motion["valid"]=False
            with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",return_value={"registered_pair_consistent":True,"hand_delta_m":[0,0,.02]}):
                self.assertFalse(self.observe(x,obs)["verified"])
                self.assertFalse(self.observe(x,obs)["verified"])

    def test_both_hands_need_independent_registered_targets(self):
        x=self.make(True)
        def measured(*args):return {"registered_pair_consistent":args[3]=="right","hand_delta_m":[0,0,.02]}
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",side_effect=measured):
            for _ in range(4):self.assertFalse(self.observe(x)["verified"])

    def test_registered_sensor_gate_does_not_rewrite_vlm_claim(self):
        _,s=fixture();s.gripper[:]=.03
        h=GroundedHarness([Goal("pick","object","right","moves")]);h.stage="VERIFY_GRASP"
        h.observation=evidence();h.feedback=feedback();h.feedback["eef_delta_m"]["right"]=[0,0,.01]
        obs=evidence(enclosed=True,co_moving=None)
        h.observe(obs,s,measured_progress={"metric_co_motion":False,"registered_grasp_motion":{
            "verified":True,"source":"registered_onboard_RGBD_grasp_motion","not_official_task_success":True}})
        self.assertEqual(len(h.completed),1);self.assertIsNone(h.completed[0]["evidence"]["co_moving"])
        self.assertIn("registered_motion_evidence",h.completed[0])


if __name__=="__main__":unittest.main()
