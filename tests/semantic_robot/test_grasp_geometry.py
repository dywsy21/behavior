import copy
import unittest
import numpy as np
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.vision import prepare_views
from semantic_robot.v2.grounded_harness import GroundedController,GroundedHarness
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import Action
from test_v2 import fixture,evidence


class GraspGeometryTests(unittest.TestCase):
    def model(self):
        m,s=fixture();spec=copy.deepcopy(m.spec)
        spec["metadata"].update(grasp_region_reference_gripper_m=[.05,.05],
            grasp_region_reference_fully_open={"left":True,"right":True},
            grasp_regions_eef={arm:[[[0,-.045,-.01],[0,-.045,.01]],[[0,.045,-.01],[0,.045,.01]]] for arm in ("left","right")})
        return RobotModel(spec),s

    def test_regions_derived_from_current_robot_pose_and_immutable(self):
        m,s=self.model();r=m.grasp_regions(s.q,s.gripper)
        self.assertTrue(r["right"]["valid"])
        expected=np.array([[.4,-.345,.79],[.4,-.345,.81]])
        np.testing.assert_allclose(r["right"]["sides_base_m"][0],expected)
        r["right"]["sides_base_m"][0][0][0]=99
        self.assertNotEqual(m.grasp_regions(s.q,s.gripper)["right"]["sides_base_m"][0][0][0],99)

    def test_closed_unknown_or_unverified_reference_abstains(self):
        m,s=self.model()
        self.assertFalse(m.grasp_regions(s.q,[.05,0.])["right"]["valid"])
        self.assertFalse(m.grasp_regions(s.q,None)["right"]["valid"])
        m.spec["metadata"]["grasp_region_reference_fully_open"]["right"]=False
        self.assertFalse(m.grasp_regions(s.q,s.gripper)["right"]["valid"])

    def test_overlay_preserves_raw_and_distinguishes_robot_points(self):
        m,s=self.model();imgs={v+"_rgb":np.zeros((100,100,3),np.uint8) for v in ("head","left_wrist","right_wrist")}
        b=prepare_views(imgs,m,s.q,grounded=True,gripper=s.gripper)
        r=b.geometry["head"]["right"]["finger_contact_region"]
        self.assertTrue(r["valid"]);self.assertTrue(r["robot_geometry_not_target_or_enclosure"])
        self.assertEqual(len(r["sides_uv"]),2)
        self.assertEqual(np.count_nonzero(b.current_raw["head"]),0)

    def test_tool_translation_matches_actual_servo_transform(self):
        m,s=self.model();h=GroundedHarness([Goal("pick","object","right","moves")]);h.stage="ALIGN";h.observation=evidence()
        servo=SafeServo(m,s,[1,1]);c=GroundedController(m,servo,h)
        for move in ("forward","back","left","right","up","down"):
            a=Action("right",move,"micro","tool");self.assertIn(a,h.palette())
            direction=c._translation_direction(a,s)
            trial=SafeServo(m,s,[1,1]);self.assertTrue(trial.begin(a,s))
            np.testing.assert_allclose(trial.targets["right"][0]-s.poses["right"][0],direction*.002,atol=1e-12)

    def test_no_independent_tool_moves_with_pending_load(self):
        _,s=self.model();h=GroundedHarness([Goal("pick","object","right","moves")]);h.stage="ALIGN";h.observation=evidence()
        h.pending_grasp["right"]=True
        self.assertFalse(any(a.frame=="tool" and a.move=="forward" for a in h.palette()))


if __name__=="__main__":unittest.main()
