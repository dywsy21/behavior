"""A slower carry must not implicitly shorten the permitted joint path."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import SafeServo, ServoLimits
from test_v2 import fixture


class CarryDurationTests(unittest.TestCase):
    def folded(self):
        model = RobotModel(json.loads((Path(__file__).parent / "fixtures/r1pro_folded_fk.json").read_text()))
        return model, model.state(model.reference, np.full(2, .05), np.zeros(3))

    def test_strict_explicit_flag(self):
        for bad in (1, 0, "true", None, np.bool_(True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ServoLimits(carry_duration_v1=bad)
        self.assertFalse(ServoLimits().carry_duration_v1)

    def test_only_carry_arm_translation_moving_time_scales(self):
        new, old = ServoLimits(carry_duration_v1=True), ServoLimits()
        for scale, before, after in (("micro", 24, 43), ("fine", 40, 75), ("coarse", 64, 123)):
            for part in ("left", "right", "both"):
                for direction in ("up", "down", "forward", "back", "left", "right"):
                    action = Action(part, direction, scale)
                    self.assertEqual(new.action_tick_limit(action, True), after)
                    self.assertEqual(new.action_tick_limit(action, False), before)
                    self.assertEqual(old.action_tick_limit(action, True), before)
            for part, move in (("base", "forward"), ("torso", "up"), ("right", "yaw_plus")):
                self.assertEqual(new.action_tick_limit(Action(part, move, scale), True), before)
        for move in ("open", "close", "hold"):
            self.assertEqual(new.action_tick_limit(Action("right", move), True), 40)

    def test_real_folded_carry_now_has_finite_safe_path(self):
        model, start = self.folded()
        action = Action("right", "up")
        old = SafeServo(model, start)
        self.assertFalse(old.begin(action, start, carry=True))
        self.assertEqual(old.status, "DURATION_LIMIT_EXCEEDED")
        new = SafeServo(model, start, gripper_command=[1, -1], limits=ServoLimits(carry_duration_v1=True))
        self.assertTrue(new.begin(action, start, carry=True))
        self.assertEqual(new.total_ticks, old.required_joint_trajectory_ticks)
        self.assertGreater(new.total_ticks, 40)
        self.assertLessEqual(new.total_ticks, 75)
        self.assertEqual(action.amount(True), .01)
        path = np.vstack((start.q, new.joint_plan))
        self.assertLessEqual(float(np.abs(np.diff(path, axis=0)).max()), .0125 + 1e-9)
        for q in path:
            self.assertGreaterEqual(new.collision.clearance(q), 0)
            self.assertTrue(np.all(q >= model.lower + new.limits.joint_margin - 1e-5))
            self.assertTrue(np.all(q <= model.upper - new.limits.joint_margin + 1e-5))
        state = start
        while not new.done and new.ticks < new.total_ticks:
            command = new.next_action(state)
            np.testing.assert_array_equal(command[[14, 22]], [1, -1])
            q = np.r_[command[3:14], command[15:22]]
            self.assertLessEqual(float(np.max(abs(q-state.q))), .0125 + 1e-6)
            state = model.state(q, state.gripper, np.zeros(3))
        feedback = new.finish(state)
        self.assertEqual(feedback["status"], "TARGET_REACHED")
        self.assertLess(feedback["target_error_m"]["right"], .0025)
        self.assertEqual(feedback["motion_timing"]["maximum_control_ticks"], 75)
        self.assertEqual(feedback["motion_timing"]["joint_step_limit"], .0125)
        self.assertFalse(feedback["motion_timing"]["success_claim"])
        self.assertEqual(feedback["holding"], "UNKNOWN")

    def test_enabled_does_not_change_unladen_plan(self):
        model, state = self.folded()
        versions = [SafeServo(model, state, limits=ServoLimits(carry_duration_v1=flag)) for flag in (False, True)]
        for servo in versions:
            self.assertTrue(servo.begin(Action("right", "up"), state, carry=False))
        self.assertEqual(versions[0].total_ticks, versions[1].total_ticks)
        np.testing.assert_array_equal(versions[0].joint_plan, versions[1].joint_plan)
        self.assertNotIn("motion_timing", versions[0].finish(state))

    def test_new_duration_still_fails_closed_at_finite_cap(self):
        model, state = fixture()
        servo = SafeServo(model, state, limits=ServoLimits(joint_tick=.00001, carry_duration_v1=True))
        self.assertFalse(servo.begin(Action("right", "up"), state, carry=True))
        self.assertEqual(servo.status, "DURATION_LIMIT_EXCEEDED")
        self.assertEqual(servo.ticks, 0)
        self.assertIsNone(servo.joint_plan)
        receipt = servo.finish(state)["motion_timing"]
        self.assertGreater(receipt["required_joint_trajectory_ticks"], receipt["maximum_control_ticks"])
        self.assertFalse(receipt["joint_trajectory_planned"])

    def test_old_accepted_carry_has_identical_plan(self):
        model, state = fixture()
        versions = [SafeServo(model, state, limits=ServoLimits(carry_duration_v1=flag)) for flag in (False, True)]
        for servo in versions:
            self.assertTrue(servo.begin(Action("right", "up"), state, carry=True))
        self.assertEqual(versions[0].total_ticks, versions[1].total_ticks)
        np.testing.assert_array_equal(versions[0].joint_plan, versions[1].joint_plan)

    def test_original_tracking_divergence_is_not_relaxed(self):
        model, state = self.folded()
        servo = SafeServo(model, state, limits=ServoLimits(carry_duration_v1=True))
        self.assertTrue(servo.begin(Action("right", "up"), state, carry=True))
        # No simulation: deliberately falsify a proprio pose after a command.
        servo.next_action(state)
        disturbed = model.state(state.q.copy(), state.gripper.copy(), np.zeros(3))
        disturbed.poses["right"][0][2] += .020
        for _ in range(servo.limits.emergency_ticks):
            servo.next_action(disturbed)
        self.assertTrue(servo.done)
        self.assertEqual(servo.status, "TRACKING_DIVERGED")

    def test_carry_rotation_prohibition_is_unchanged(self):
        model, state = fixture()
        servo = SafeServo(model, state, limits=ServoLimits(carry_duration_v1=True))
        with self.assertRaises(ValueError):
            servo.begin(Action("right", "yaw_plus"), state, carry=True)

    def test_budget_interruption_remains_failure(self):
        model, state = self.folded()
        servo = SafeServo(model, state, limits=ServoLimits(carry_duration_v1=True))
        self.assertTrue(servo.begin(Action("right", "up"), state, carry=True))
        self.assertEqual(servo.finish(state)["status"], "INTERRUPTED")

    def test_runner_exact_gate_profile_and_saved_identity(self):
        tree = ast.parse((Path(__file__).resolve().parents[2] / "scripts/semantic_robot/run_v2.py").read_text())
        condition = next(n for n in ast.walk(tree) if isinstance(n, ast.Compare)
                         and ast.unparse(n.left) == "g.get('carry_duration_v1', False)")
        code = compile(ast.Expression(body=condition), "carry_gate_profile", "eval")
        for gate, requested, expected in (({}, False, True), ({}, True, False),
                                         ({"carry_duration_v1": True}, True, True),
                                         ({"carry_duration_v1": True}, False, False)):
            self.assertEqual(eval(code, {"g": gate, "args": SimpleNamespace(carry_duration_v1=requested)}), expected)
        values = [v for n in ast.walk(tree) if isinstance(n, ast.Dict) for k, v in zip(n.keys, n.values)
                  if isinstance(k, ast.Constant) and k.value == "carry_duration_v1"]
        self.assertEqual([ast.unparse(v) for v in values], ["args.carry_duration_v1"])
        constructor = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Name) and n.func.id == "ServoLimits")
        self.assertEqual(next(ast.unparse(k.value) for k in constructor.keywords if k.arg == "carry_duration_v1"),
                         "args.carry_duration_v1")


if __name__ == "__main__":
    unittest.main()
