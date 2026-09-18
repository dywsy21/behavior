import copy
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.v2.substep_odometry import SubstepMotion, checked_transform
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.protocol import Action
from test_v2 import fixture, feedback


def measured(x=0., y=0., yaw=0.):
    T = np.eye(4); T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix(); T[:2, 3] = [x, y]
    return {"valid": True, "body_delta": [x, y, yaw], "body_transform_current_in_previous": T.tolist(),
            "inliers": 40, "median_reprojection_px": .2, "median_depth_correspondence_m": .002}


class SubstepMotionTests(unittest.TestCase):
    def setup_motion(self, receipts):
        self.model, self.state = fixture()
        self.images = {"head_rgb": np.zeros((3, 100, 100), np.uint8)}
        self.depths = {"head": np.ones((100, 100), np.float32)}
        self.base = Mock()
        self.base.observe.side_effect = [{"valid": True, "initial": True, "body_delta": [0, 0, 0]}, *receipts]
        self.motion = SubstepMotion(self.base)
        self.args = (self.images, self.depths, self.model, self.state.q)
        self.motion.observe(*self.args)

    def test_rigid_composition_tail_and_exactly_once_delivery(self):
        a, b, c = measured(yaw=.08), measured(x=.03), measured(x=.01)
        self.setup_motion([a, b, c, measured()])
        self.motion.begin(30)
        for control in (36, 42, 45):
            self.motion.sample(*self.args, control)
        result = self.motion.finish(45)
        expected = checked_transform(a) @ checked_transform(b) @ checked_transform(c)
        np.testing.assert_allclose(result["body_transform_current_in_previous"], expected)
        self.assertGreater(result["body_delta"][1], 0.)  # Not componentwise addition.
        self.assertEqual([(s["control_start"], s["control_end"]) for s in result["segments"]], [(30, 36), (36, 42), (42, 45)])
        with self.assertRaises(ValueError): self.motion.begin(45)
        delivered = self.motion.observe(*self.args)
        self.assertNotIn("segments", delivered)
        self.assertEqual(delivered["segment_count"], 3)
        self.assertEqual(self.base.observe.call_count, 4)  # Cached aggregate, no extra fit.
        np.testing.assert_allclose(self.motion.observe(*self.args)["body_delta"], 0.)
        self.assertEqual(self.base.observe.call_count, 5)  # Not the aggregate again.
        with self.assertRaises(ValueError): self.motion.begin(46)
        self.motion.begin(45)

    def test_failed_middle_segment_never_credits_prefix_or_restarts(self):
        self.setup_motion([measured(x=.02), {"valid": False, "reason": "LOW_TEXTURE"}])
        self.motion.begin(0); self.motion.sample(*self.args, 6)
        self.assertFalse(self.motion.sample(*self.args, 12)["valid"])
        with self.assertRaises(ValueError): self.motion.sample(*self.args, 18)
        result = self.motion.finish(12)
        self.assertFalse(result["valid"]); self.assertNotIn("body_delta", result)
        self.assertEqual(len(result["segments"]), 2)
        self.assertFalse(self.motion.observe(*self.args)["valid"])
        with self.assertRaises(ValueError): self.motion.begin(12)
        with self.assertRaises(ValueError): self.motion.observe(*self.args)

    def test_missing_final_tail_duplicate_gap_and_noninteger_rejected(self):
        self.setup_motion([measured()]); self.motion.begin(0)
        for tick in (0, 7, 6.0, True):
            with self.assertRaises(ValueError): self.motion.sample(*self.args, tick)
        self.motion.sample(*self.args, 6)
        with self.assertRaises(ValueError): self.motion.sample(*self.args, 6)
        with self.assertRaises(ValueError): self.motion.finish(7)
        with self.assertRaises(ValueError): self.motion.observe(*self.args)

    def test_end_pixels_depth_and_camera_fk_are_bound_to_receipt(self):
        for kind in ("rgb", "depth", "camera"):
            self.setup_motion([measured()]); self.motion.begin(0); self.motion.sample(*self.args, 6); self.motion.finish(6)
            images = copy.deepcopy(self.images); depths = copy.deepcopy(self.depths); model = self.model
            if kind == "rgb": images["head_rgb"][0, 0, 0] = 255
            if kind == "depth": depths["head"][0, 0] = 2.
            if kind == "camera":
                model = Mock(wraps=self.model); T = self.model.forward(self.state.q, "camera_head").copy(); T[0, 3] += .01
                model.spec = self.model.spec; model.forward.return_value = T
            with self.assertRaises(ValueError): self.motion.observe(images, depths, model, self.state.q)
            self.assertIsNotNone(self.motion.pending)

    def test_whole_action_bound_is_not_bypassed_by_small_segments(self):
        self.setup_motion([measured(x=.10), measured(x=.10)])
        self.motion.begin(0); self.motion.sample(*self.args, 6); self.motion.sample(*self.args, 12)
        result = self.motion.finish(12)
        self.assertFalse(result["valid"]); self.assertNotIn("body_delta", result)

    def test_interruption_invalidates_even_all_good_segments(self):
        self.setup_motion([measured(x=.01)]); self.motion.begin(0); self.motion.sample(*self.args, 3)
        result = self.motion.finish(3, interrupted=True)
        self.assertFalse(result["valid"]); self.assertNotIn("body_delta", result)
        self.assertEqual(result["reason"], "ACTION_INTERRUPTED_NO_FULL_MOTION_CERTIFICATE")

    def test_arbitrary_action_length_keeps_tail_and_exact_control_span(self):
        for length in (23, 24, 25):
            ticks = list(range(6, length+1, 6))
            if ticks[-1] != length: ticks.append(length)
            self.setup_motion([measured(x=.001) for _ in ticks]); self.motion.begin(0)
            for tick in ticks: self.motion.sample(*self.args, tick)
            result = self.motion.finish(length)
            self.assertEqual(result["control_end"], length)
            self.assertTrue(result["valid"])
            self.assertAlmostEqual(result["body_delta"][0], .001*len(ticks))

    def test_bad_or_initial_substep_transform_cannot_be_accepted(self):
        for key in ("nan", "reflection", "scale", "delta"):
            r = measured(); T = np.asarray(r["body_transform_current_in_previous"])
            if key == "nan": T[0, 0] = float("nan")
            if key == "reflection": T[0, 0] = -1
            if key == "scale": T[0, 0] = 2
            if key == "delta": r["body_delta"][0] = .03
            r["body_transform_current_in_previous"] = T.tolist()
            with self.assertRaises(ValueError): checked_transform(r)
        self.setup_motion([{"valid": True, "initial": True}]); self.motion.begin(0)
        self.assertFalse(self.motion.sample(*self.args, 6)["valid"])
        self.assertFalse(self.motion.finish(6)["valid"])

    def test_reference_missing_calibration_shape_and_interval_fail_closed(self):
        with self.assertRaises(ValueError): SubstepMotion(Mock(), 3)
        with self.assertRaises(ValueError): SubstepMotion(Mock()).begin(0)
        self.setup_motion([])
        with self.assertRaises(ValueError): self.motion.observe(self.images, {"head": np.ones((2, 2), np.float32)}, self.model, self.state.q)

    def test_real_controller_gets_complete_move_not_last_segment(self):
        model, state = fixture()
        images = {"head_rgb": np.zeros((3, 100, 100), np.uint8)}
        depths = {"head": np.ones((100, 100), np.float32)}
        base = Mock(); base.observe.side_effect = [{"valid": True, "initial": True}, *[measured(yaw=.03) for _ in range(4)]]
        h = GroundedHarness([Goal("navigate", "table", "both", "near table")])
        c = GroundedController(model, SafeServo(model, state), h, visual_odometry=True)
        c.motion = SubstepMotion(base); c.update_motion(images, depths, state)
        c.motion.begin(0)
        for tick in (6, 12, 18, 24): c.motion.sample(images, depths, model, state.q, tick)
        c.motion.finish(24)
        raw = feedback(status="BASE_TRACKING_FAILED"); raw["base_integral"] = [0, 0, .16]
        c.executed(Action("base", "yaw_plus", "coarse"), raw)
        self.assertEqual(c.search.heading, 0.)
        c.update_motion(images, depths, state)
        self.assertAlmostEqual(c.search.heading, .12)
        self.assertEqual(h.feedback["status"], "TARGET_REACHED")
        self.assertIsNone(c.pending_motion)
        self.assertNotIn("segments", h.context()["egocentric_motion"])
        self.assertEqual(raw["status"], "BASE_TRACKING_FAILED")


if __name__ == "__main__": unittest.main()
