"""Command execution is distinct from nominal IK precision and grasp outcome."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import SafeServo, ServoLimits, execution_completed, GRIPPER_COMPLETED
from semantic_robot.v2.harness import Goal, TaskHarness
from semantic_robot.v2.grounded_harness import GroundedHarness
from test_v2 import fixture, evidence


def completed(action=Action("right", "close"), drift=.0029833103417, enabled=True):
    model, state = fixture()
    servo = SafeServo(model, state, limits=ServoLimits(gripper_completion_v1=enabled))
    assert servo.begin(action, state)
    commands = []
    for _ in range(servo.total_ticks):
        commands.append(servo.next_action(state))
    end = copy.deepcopy(state)
    if action.part in ("left", "right", "both"):
        for arm in (("left", "right") if action.part == "both" else (action.part,)):
            end.poses[arm][0][0] += drift
    return model, state, servo, end, commands


class GripperCompletionTests(unittest.TestCase):
    def test_opt_in_preserves_raw_pose_failure_and_unknown_holding(self):
        _, _, servo, end, commands = completed()
        feedback = servo.finish(end)
        self.assertEqual(feedback["status"], GRIPPER_COMPLETED)
        self.assertEqual(feedback["pose_tracking_status"], "TRACKING_FAILED")
        self.assertAlmostEqual(feedback["target_error_m"]["right"], .0029833103417)
        self.assertTrue(execution_completed(servo.action, feedback))
        self.assertFalse(feedback["gripper_execution"]["success_claim"])
        self.assertEqual(feedback["holding"], "UNKNOWN")
        self.assertEqual(feedback["official_success"], "NOT_AVAILABLE_TO_ACTOR")
        self.assertEqual(len(commands), 18)
        self.assertTrue(all(c[22] == -1 and c[14] == 1 for c in commands))

    def test_legacy_is_unchanged(self):
        _, _, servo, end, _ = completed(enabled=False)
        result = servo.finish(end)
        self.assertEqual(result["status"], "TRACKING_FAILED")
        self.assertNotIn("gripper_execution", result)
        self.assertFalse(execution_completed(servo.action, result))
        _, _, servo, end, _ = completed(enabled=False, drift=0)
        self.assertTrue(execution_completed(servo.action, servo.finish(end)))

    def test_new_profile_cannot_expand_divergence_limits(self):
        for values in ({"divergent_position": .0181}, {"divergent_angle": np.deg2rad(9.1)},
                       {"divergent_position": float("nan")}):
            with self.assertRaises(ValueError):
                ServoLimits(gripper_completion_v1=True, **values)

    def test_both_hands_open_close_and_pose_precision_are_separate(self):
        for arm in ("left", "right", "both"):
            for move in ("open", "close"):
                with self.subTest(arm=arm, move=move):
                    _, _, servo, end, _ = completed(Action(arm, move), drift=0)
                    result = servo.finish(end)
                    self.assertEqual(result["status"], GRIPPER_COMPLETED)
                    self.assertEqual(result["pose_tracking_status"], "TARGET_REACHED")
                    self.assertTrue(execution_completed(servo.action, result))

    def test_short_command_is_interrupted_even_when_pose_exact(self):
        model, state = fixture()
        servo = SafeServo(model, state, limits=ServoLimits(gripper_completion_v1=True))
        servo.begin(Action("right", "close"), state)
        for _ in range(17):
            servo.next_action(state)
        result = servo.finish(state)
        self.assertEqual(result["status"], "INTERRUPTED")
        self.assertFalse(execution_completed(servo.action, result))

    def test_prior_hard_stop_is_never_cleared(self):
        _, _, servo, end, _ = completed(drift=0)
        servo.abort("ROBOT_COLLISION_RISK")
        result = servo.finish(end)
        self.assertEqual(result["status"], "ROBOT_COLLISION_RISK")
        self.assertFalse(result["gripper_execution"]["command_complete"])

    def test_final_step_active_and_inactive_motion_guards(self):
        for arm, drift, reason in (("right", .019, "TRACKING_DIVERGED"),
                                   ("left", .003, "TRACKING_FAILED")):
            _, _, servo, end, _ = completed(drift=0)
            end.poses[arm][0][0] += drift
            result = servo.finish(end)
            self.assertEqual(result["status"], reason)
            self.assertFalse(execution_completed(servo.action, result))

    def test_final_step_angle_guard(self):
        from scipy.spatial.transform import Rotation
        for arm, deg in (("right", 9.1), ("left", 1.6)):
            _, _, servo, end, _ = completed(drift=0)
            end.poses[arm] = (end.poses[arm][0], Rotation.from_euler("z", deg, degrees=True).as_quat())
            result = servo.finish(end)
            self.assertFalse(execution_completed(servo.action, result))

    def test_post_step_joint_limit_collision_and_nonfinite_checks(self):
        _, _, servo, end, _ = completed(drift=0)
        end.q[17] = 2.1
        self.assertEqual(servo.finish(end)["status"], "JOINT_STATE_OUT_OF_BOUNDS")
        _, _, servo, end, _ = completed(drift=0)
        with patch.object(servo.collision, "clearance", return_value=-.01):
            self.assertEqual(servo.finish(end)["status"], "ROBOT_COLLISION_RISK")
        _, _, servo, end, _ = completed(drift=0)
        end.gripper[1] = np.nan
        self.assertEqual(servo.finish(end)["status"], "NONFINITE_PROPRIOCEPTION")

    def test_mid_command_joint_limit_and_divergence_still_stop(self):
        model, state = fixture()
        servo = SafeServo(model, state, limits=ServoLimits(gripper_completion_v1=True))
        servo.begin(Action("right", "close"), state)
        invalid = copy.deepcopy(state); invalid.q[17] = 2.1
        servo.next_action(invalid)
        self.assertEqual(servo.status, "JOINT_STATE_OUT_OF_BOUNDS")
        servo = SafeServo(model, state, limits=ServoLimits(gripper_completion_v1=True))
        servo.begin(Action("right", "close"), state)
        invalid = copy.deepcopy(state); invalid.poses["right"][0][0] += .019
        for _ in range(3):
            servo.next_action(invalid)
        self.assertEqual(servo.status, "TRACKING_DIVERGED")

    def test_translation_does_not_gain_gripper_exception(self):
        _, _, servo, end, _ = completed(Action("right", "up"), drift=0)
        result = servo.finish(end)
        self.assertNotIn("gripper_execution", result)
        self.assertNotEqual(result["status"], GRIPPER_COMPLETED)

    def test_receipt_cannot_be_transferred_or_relabelled(self):
        _, _, servo, end, _ = completed()
        result = servo.finish(end)
        self.assertFalse(execution_completed(Action("right", "up"), result))
        self.assertFalse(execution_completed(Action("left", "close"), result))
        self.assertFalse(execution_completed(Action("right", "open"), result))
        self.assertFalse(execution_completed(servo.action, {"status": GRIPPER_COMPLETED}))
        mutations = [
            lambda r: r.update(visual_gate_failure={"valid": False}),
            lambda r: r.update(pose_tracking_status="TARGET_REACHED"),
            lambda r: r.update(holding="TRUE"),
            lambda r: r.update(control_ticks=17),
            lambda r: r["gripper_execution"].update(command_complete=False),
            lambda r: r["gripper_execution"].update(executed_control_ticks=True),
            lambda r: r["gripper_execution"]["final_state_checks"].update(joint_bounds=False),
            lambda r: r["gripper_execution"]["limits"].update(active_position_m=.019),
            lambda r: r["target_error_m"].update(right=float("nan")),
            lambda r: r["target_error_m"].update(left=.003),
            lambda r: r["gripper_execution"].update(success_claim=True),
        ]
        for change in mutations:
            changed = copy.deepcopy(result); change(changed)
            self.assertFalse(execution_completed(servo.action, changed))

    def test_task_harness_verifies_after_close_without_claiming_hold(self):
        _, _, servo, end, _ = completed()
        result = servo.finish(end)
        for kind in (TaskHarness, GroundedHarness):
            h = kind([Goal("pick", "target", "right", "verified lift")])
            h.stage = "GRASP"
            h.executed(servo.action, result)
            self.assertEqual(h.stage, "VERIFY_GRASP")
            self.assertFalse(any(h.hold_verified.values()))
            self.assertEqual(h.completed, [])
            self.assertEqual(h.feedback["pose_tracking_status"], "TRACKING_FAILED")
            if kind is TaskHarness:
                h.observe(evidence(enclosed=None, co_moving=None), end)
                self.assertNotEqual(h.stage, "RECOVER")

    def test_grounded_probe_keeps_unverified_contact_and_requires_verification(self):
        _, _, servo, end, _ = completed()
        h = GroundedHarness([Goal("pick", "target", "right", "verified lift")])
        h.stage = "ALIGN"; h.grasp_probe = {"eligible": True}
        h.executed(servo.action, servo.finish(end))
        self.assertEqual(h.stage, "VERIFY_GRASP")
        self.assertTrue(h.pending_grasp["right"])
        self.assertTrue(h.possible_contact_after_close["right"])
        self.assertFalse(h.hold_verified["right"])

    def test_release_command_is_not_a_place_success(self):
        _, _, servo, end, _ = completed(Action("right", "open"))
        h = TaskHarness([Goal("place", "target", "right", "supported on table")])
        h.stage = "RELEASE"; h.hold_verified["right"] = True
        h.executed(servo.action, servo.finish(end))
        self.assertEqual(h.stage, "VERIFY_PLACE")
        self.assertTrue(h.hold_verified["right"])
        self.assertEqual(h.completed, [])


if __name__ == "__main__":
    unittest.main()
