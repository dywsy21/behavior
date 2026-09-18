import copy
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from test_v2 import fixture
from semantic_robot.v2.hand_body_collision import HandBodyGuard, obb_gap
from semantic_robot.v2.grounding import LocalDepthGuard
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.servo import SafeServo, ServoLimits
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.og_calibration import r1pro_parallel_hand_supported


def calibrated_fixture():
    model, state = fixture()
    spec = copy.deepcopy(model.spec)
    boxes = []
    for arm in ("left", "right"):
        for suffix in ("_gripper_link", "_gripper_finger_link1", "_gripper_finger_link2"):
            boxes.append({"link": arm + suffix, "lower": [-.025] * 3, "upper": [.025] * 3,
                          "T_base_link": model.forward(state.q, arm).tolist()})
    for name in ("base_link", "torso_link1", "torso_link2", "torso_link3", "torso_link4"):
        boxes.append({"link": name, "lower": [-.06] * 3, "upper": [.06] * 3, "T_base_link": np.eye(4).tolist()})
        if name != "base_link":
            spec["links"]["link:" + name] = {"T_reference": np.eye(4).tolist(), "screws": np.zeros((6, 18)).tolist()}
    geometry = {"source": "robot_visual_link_boxes_actual_joint_fk", "scene_truth": False,
                "includes_actual_finger_positions": True, "margin_m": .003, "boxes": boxes}
    spec["metadata"].update(robot_visual_boxes_reference=geometry,
                             parallel_gripper_open_envelope="R1Pro_parallel_prismatic_jaws",
                             grasp_region_reference_fully_open={"left": True, "right": True})
    return RobotModel(spec), state, geometry


class HandBodyTests(unittest.TestCase):
    def test_installed_fixed_hand_has_no_end_effector_attribute(self):
        fixed = SimpleNamespace(model="r1pro", has_end_effector_variants=False)
        self.assertFalse(hasattr(fixed, "end_effector"))
        self.assertTrue(r1pro_parallel_hand_supported(fixed))
        self.assertTrue(r1pro_parallel_hand_supported(SimpleNamespace(
            model="r1pro", has_end_effector_variants=True, end_effector="gripper")))
        for robot in (SimpleNamespace(model="r1pro"),
                      SimpleNamespace(model="unknown", has_end_effector_variants=False),
                      SimpleNamespace(model="r1pro", has_end_effector_variants=True),
                      SimpleNamespace(model="r1pro", has_end_effector_variants=True, end_effector="suction")):
            with self.subTest(robot=robot):
                self.assertFalse(r1pro_parallel_hand_supported(robot))

    def test_separating_axis_handles_rotated_disjoint_and_overlapping_boxes(self):
        a, b = np.eye(4), np.eye(4)
        lo, hi = np.full(3, -.1), np.full(3, .1)
        self.assertLess(obb_gap(a, lo, hi, b, lo, hi), 0)
        b[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        b[0, 3] = .3
        self.assertGreater(obb_gap(a, lo, hi, b, lo, hi), 0)
        b[0, 3] = .2
        self.assertLess(obb_gap(a, lo, hi, b, lo, hi), 0)

    def test_finger_envelope_body_veto_is_robot_only_and_opt_in(self):
        model, state, _ = calibrated_fixture()
        guard = HandBodyGuard(model)
        self.assertGreater(guard.clearance(state.q)[0], 0)
        q = state.q.copy()
        q[4:7] = [-.4, -.3, -.8]
        self.assertLess(guard.clearance(q)[0], 0)
        bad = model.state(q, state.gripper, np.zeros(3))
        servo = SafeServo(model, bad, limits=ServoLimits(robot_geometry_guards=True))
        self.assertFalse(servo.begin(Action("left", "up"), bad))
        self.assertEqual(servo.status, "ROBOT_COLLISION_RISK")
        self.assertIsNone(SafeServo(model, state).collision.hand_body)

    def test_missing_or_closed_reference_fails_before_actuation(self):
        model, state = fixture()
        with self.assertRaises(ValueError):
            SafeServo(model, state, limits=ServoLimits(robot_geometry_guards=True))
        model, _, _ = calibrated_fixture()
        model.spec["metadata"]["grasp_region_reference_fully_open"]["right"] = False
        with self.assertRaises(ValueError):
            HandBodyGuard(model)

    def test_real_near_hand_obstacle_not_erased_as_self(self):
        model, state, geometry = calibrated_fixture()
        points = np.tile([.34, .3, .8], (50, 1))
        old = LocalDepthGuard(points, model, state.q, {})
        new = LocalDepthGuard(points, model, state.q, {}, self_geometry=geometry)
        self.assertEqual(len(old.environment_points), 0)
        self.assertEqual(len(new.environment_points), 50)
        own = LocalDepthGuard(np.tile([.4, .3, .8], (50, 1)), model, state.q, {}, self_geometry=geometry)
        self.assertEqual(len(own.environment_points), 0)
        with self.assertRaises(ValueError):
            LocalDepthGuard(points, model, state.q, {}, self_geometry={})

    def test_malformed_current_geometry_cannot_erase_obstacle_and_leave_free_rays(self):
        model, state, geometry = calibrated_fixture()
        points = np.concatenate((np.tile([.4, 0, .2], (50, 1)), np.tile([1.5, 0, .2], (50, 1))))
        action = Action("base", "forward")
        self.assertEqual(LocalDepthGuard(points, model, state.q, {}, self_geometry=geometry).check(action),
                         (False, "OBSERVED_BASE_OBSTACLE"))
        scaled, reflected, projective = np.eye(4), np.eye(4), np.eye(4)
        scaled[:3, :3] *= .1
        scaled[:3, 3] = [.4, 0, .2]
        reflected[0, 0] = -1
        projective[3, 0] = .01
        for matrix in (scaled, reflected, projective):
            corrupt = copy.deepcopy(geometry)
            corrupt["boxes"][0]["T_base_link"] = matrix.tolist()
            with self.subTest(matrix=matrix.tolist()), self.assertRaisesRegex(ValueError, "SE\\(3\\)"):
                LocalDepthGuard(points, model, state.q, {}, self_geometry=corrupt)


if __name__ == "__main__":
    unittest.main()
