import unittest
import numpy as np

from semantic_robot.v2.arm_observation_guard import ObservingArmGuard, box_distance


class LinearArm:
    links = {"link:left_arm_link1": None, "link:left_finger_link1": None}

    def forward(self, q, name):
        result = np.eye(4)
        result[0, 3] = q[4]
        return result


def geometry():
    return {"source": "robot_visual_link_boxes_actual_joint_fk", "scene_truth": False,
            "includes_actual_finger_positions": True, "margin_m": .003,
            "boxes": [{"link": "left_arm_link1", "T_base_link": np.eye(4).tolist(),
                       "lower": [-.02, -.02, -.02], "upper": [.02, .02, .02]}]}


class ArmGuardTests(unittest.TestCase):
    def make(self, extra=(), geom=None):
        # Background plus the explicitly specified visible obstruction.
        cloud = np.vstack([np.tile([2., 2., 2.], (50, 1)), np.asarray(extra).reshape(-1, 3)])
        return ObservingArmGuard(LinearArm(), np.zeros(18), cloud, geometry() if geom is None else geom, "left")

    def plan(self, distance=.10):
        plan = np.zeros((2, 18)); plan[:, 4] = [distance / 2, distance]
        return plan

    def test_box_distance_inside_and_outside(self):
        distance = box_distance(np.array([[0, 0, 0], [.04, 0, 0]]), np.eye(4), np.full(3, -.02), np.full(3, .02))
        np.testing.assert_allclose(distance, [-.02, .02])

    def test_endpoint_clear_but_midpath_obstacle_is_vetoed(self):
        ok, receipt = self.make([[.045, 0., 0.]]).check(self.plan())
        self.assertFalse(ok)
        self.assertEqual(receipt["reason"], "OBSERVED_FREE_ARM_SWEEP_OBSTACLE")

    def test_unknown_depth_and_missing_actual_geometry_fail_closed(self):
        for geom in ({}, {**geometry(), "scene_truth": True}, {**geometry(), "boxes": []}):
            ok, _ = self.make(geom=geom).check(self.plan())
            self.assertFalse(ok)
        guard = ObservingArmGuard(LinearArm(), np.zeros(18), np.zeros((3, 3)), geometry(), "left")
        self.assertFalse(guard.check(self.plan())[0])

    def test_other_hand_is_not_masked_as_self(self):
        geom = geometry()
        other = np.eye(4); other[0, 3] = .05
        geom["boxes"].append({"link": "right_finger_link1", "T_base_link": other.tolist(),
                              "lower": [-.02] * 3, "upper": [.02] * 3})
        guard = self.make([[.05, 0., 0.]], geom)
        self.assertEqual(len(guard.points), 51)
        self.assertFalse(guard.check(self.plan())[0])

    def test_only_current_free_arm_volume_removed_not_new_obstacles(self):
        guard = self.make([[0., 0., 0.]])
        self.assertEqual(len(guard.points), 50)
        ok, receipt = guard.check(self.plan())
        self.assertTrue(ok)
        self.assertTrue(receipt["unobserved_space_not_certified"])
        self.assertTrue(receipt["continuous_collision_clearance_not_certified"])

    def test_other_joint_motion_and_unbounded_paths_rejected(self):
        plan = self.plan(); plan[-1, 11] = .001
        self.assertEqual(self.make().check(plan)[1]["reason"], "OTHER_ARM_OR_TORSO_WOULD_MOVE")
        for plan in (np.zeros((65, 18)), np.full((1, 18), np.nan), np.zeros((1, 17))):
            with self.assertRaises(ValueError): self.make().check(plan)
