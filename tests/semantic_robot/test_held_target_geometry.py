from types import SimpleNamespace
import unittest
import numpy as np

from semantic_robot.v2.grounded_harness import GroundedHarness,GroundedController
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import SafeServo
from test_grounded import grounded_evidence
from test_v2 import fixture


class HeldTargetGeometryTests(unittest.TestCase):
    def make(self,scope="held_right"):
        m,s=fixture()
        h=GroundedHarness([Goal("press","visible button","left","effect")],held_inspection=True,reference_from_planner=True)
        h.held["right"]="object";h.hold_verified["right"]=True
        h.bind_reference(scope,"test")
        h.stage="APPROACH";h.observation=grounded_evidence(view="head")
        ctl=GroundedController(m,SafeServo(m,s,[1,-1]),h)
        ctl.centers=m.grasp_centers(s.q)
        point=ctl.centers["left"]+np.array([.08,.02,.016])
        ctl.target={"valid":True,"point_base_m":point.tolist(),"distance_to_active_closing_center_m":float(np.linalg.norm(point-ctl.centers["left"]))}
        ctl.depth_guard=SimpleNamespace(check=lambda *a:(True,"test_depth"),receipt=lambda:{})
        # Deliberately unrelated old surface anchor cannot replace the button.
        ctl.inspector.anchors["right"]={"point_hand_m":[9.,9.,9.]}
        return m,s,h,ctl

    def test_base_rigid_motion_has_zero_held_contact_gain_and_is_not_offered(self):
        _,s,h,ctl=self.make();before=ctl.target["distance_to_active_closing_center_m"]
        for move in ("forward","left","yaw_plus","yaw_minus"):
            self.assertAlmostEqual(ctl._expected_point(Action("base",move),s),before)
            with self.assertRaises(ValueError):h.authorize(Action("base",move))
        self.assertFalse(any(a.part=="base" for a in ctl.candidates(s)))

    def test_world_contact_keeps_original_body_geometry(self):
        _,s,h,ctl=self.make("world")
        self.assertLess(ctl._expected_point(Action("base","forward"),s),ctl.target["distance_to_active_closing_center_m"])
        self.assertTrue(any(a.part=="base" for a in h.palette()))

    def test_working_hand_can_close_gap_but_synchronized_motion_cannot(self):
        m,s,h,ctl=self.make();before=ctl.target["distance_to_active_closing_center_m"]
        self.assertLess(ctl._expected_point(Action("left","forward","coarse"),s),before)
        self.assertAlmostEqual(ctl._expected_point(Action("both","forward","coarse"),s),before)
        trial=SafeServo(m,s,[1,-1]);self.assertTrue(trial.begin(Action("both","forward","coarse"),s))
        self.assertAlmostEqual(ctl._expected_point(Action("both","forward","coarse"),s,trial),before,places=4)

    def test_current_button_follows_reference_joint_fk(self):
        m,s,h,ctl=self.make();point=np.array(ctl.target["point_base_m"])
        trial=SafeServo(m,s,[1,-1]);action=Action("right","forward")
        self.assertTrue(trial.begin(action,s))
        now=m.forward(s.q,"right");after=m.forward(trial.joint_plan[-1],"right")
        expected=after[:3,:3]@(now[:3,:3].T@(point-now[:3,3]))+after[:3,3]
        self.assertAlmostEqual(ctl._expected_point(action,s,trial),np.linalg.norm(expected-ctl.centers["left"]))

    def test_unknown_or_lost_verified_reference_never_defaults_to_world(self):
        _,s,h,ctl=self.make();h.hold_verified["right"]=False
        self.assertIsNone(ctl._expected_point(Action("base","forward"),s))
        self.assertFalse(any(a.part=="base" for a in h.palette()))
        h.target_references[h.index]="unknown"
        self.assertIsNone(ctl._expected_point(Action("left","forward"),s))
