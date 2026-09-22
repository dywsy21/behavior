"""Physical-state validity is distinct from the conservative planning inset."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import SafeServo, ServoLimits, execution_completed
from test_v2 import fixture


class JointBoundaryStartTests(unittest.TestCase):
    def boundary(self, lower=False):
        model, state = fixture()
        # A physically legal inactive wrist, matching the observed i1 failure.
        q = state.q.copy()
        q[16] = model.lower[16]+1.43e-6 if lower else model.upper[16]-1.43e-6
        state = model.state(q, state.gripper, np.zeros(3))
        return model, state

    def servo(self, model, state, enabled=True, **kwargs):
        return SafeServo(model, state, limits=ServoLimits(
            joint_boundary_start_v1=enabled, gripper_completion_v1=True, **kwargs))

    def test_opt_in_is_strict_boolean_and_default_off(self):
        self.assertFalse(ServoLimits().joint_boundary_start_v1)
        for value in (None, 0, 1, "true", np.bool_(True)):
            with self.assertRaises(ValueError):
                ServoLimits(joint_boundary_start_v1=value)
        for value in (-.001, np.nan, np.inf):
            with self.assertRaises(ValueError):
                ServoLimits(joint_boundary_start_v1=True, joint_margin=value)

    def test_legacy_rejects_boundary_but_new_holds_or_moves_inactive_hand(self):
        for lower in (False, True):
            model, state = self.boundary(lower)
            original_q = state.q.copy()
            for action in (Action("right", "hold"), Action("left", "up"),
                           Action("left", "close"), Action("base", "forward")):
                with self.subTest(lower=lower, action=action):
                    old = self.servo(model, state, False)
                    self.assertFalse(old.begin(action, state))
                    self.assertEqual(old.status, "JOINT_STATE_OUT_OF_BOUNDS")
                    new = self.servo(model, state)
                    self.assertTrue(new.begin(action, state))
                    expected_lo = np.minimum(model.lower+.001, state.q)
                    expected_hi = np.maximum(model.upper-.001, state.q)
                    np.testing.assert_array_equal(new.plan_lower, expected_lo)
                    np.testing.assert_array_equal(new.plan_upper, expected_hi)
                    np.testing.assert_array_equal(state.q, original_q)
                    if new.joint_plan is not None:
                        self.assertTrue(np.all(new.joint_plan >= expected_lo-1e-12))
                        self.assertTrue(np.all(new.joint_plan <= expected_hi+1e-12))

    def test_interior_accepted_plan_and_commands_identical(self):
        model, state = fixture()
        versions = [self.servo(model, state, flag) for flag in (False, True)]
        for s in versions:
            self.assertTrue(s.begin(Action("right", "up"), state))
        np.testing.assert_array_equal(versions[0].joint_plan, versions[1].joint_plan)
        self.assertEqual(versions[0].total_ticks, versions[1].total_ticks)
        current = state
        while versions[0].ticks < versions[0].total_ticks:
            commands = [s.next_action(current) for s in versions]
            np.testing.assert_array_equal(*commands)
            q = np.r_[commands[0][3:14], commands[0][15:22]]
            current = model.state(q, current.gripper, np.zeros(3))
        self.assertNotIn("joint_boundary_start", versions[0].finish(current))
        self.assertFalse(versions[1].finish(current)["joint_boundary_start"]["success_claim"])

    def test_outside_hard_bounds_and_nonfinite_start_never_admitted(self):
        model, state = fixture()
        for index, value in ((16, model.upper[16]+1e-8), (16, model.lower[16]-1e-8),
                             (3, np.nan), (8, np.inf)):
            bad = copy.deepcopy(state); bad.q[index] = value
            new = self.servo(model, bad)
            self.assertFalse(new.begin(Action("left", "hold"), bad))
            self.assertEqual(new.status, "JOINT_STATE_OUT_OF_BOUNDS")
            self.assertEqual(new.ticks, 0)

    def test_malformed_state_rejected_not_clipped(self):
        model, state = fixture()
        bad = copy.deepcopy(state); bad.q = bad.q[:-1]
        s = self.servo(model, bad)
        self.assertFalse(s.begin(Action("right", "hold"), bad))

    def test_occupied_bound_never_extended_by_active_ik(self):
        model, state = self.boundary()
        s = self.servo(model, state)
        self.assertTrue(s.begin(Action("right", "yaw_minus", "micro", "base"), state))
        self.assertLess(s.joint_plan[-1,16], state.q[16])
        self.assertLessEqual(float(s.joint_plan[:,16].max()), state.q[16])
        outward = self.servo(model, state)
        self.assertFalse(outward.begin(Action("right", "yaw_plus", "micro", "base"), state))
        self.assertEqual(outward.status, "UNREACHABLE_OR_COLLISION_BLOCKED")

    def test_gripper_completes_at_legal_boundary_without_claiming_grasp(self):
        model, state = self.boundary()
        s = self.servo(model, state)
        a = Action("left", "close")
        self.assertTrue(s.begin(a, state))
        for _ in range(s.total_ticks):
            command = s.next_action(state)
            np.testing.assert_allclose(np.r_[command[3:14], command[15:22]], state.q, atol=1e-7, rtol=0)
        feedback = s.finish(state)
        self.assertTrue(execution_completed(a, feedback))
        self.assertEqual(feedback["holding"], "UNKNOWN")
        self.assertFalse(feedback["gripper_execution"]["success_claim"])

    def test_runtime_and_final_physical_bounds_remain_enforced(self):
        model, state = self.boundary()
        for a in (Action("left", "close"), Action("left", "up"), Action("base", "forward")):
            s = self.servo(model, state); self.assertTrue(s.begin(a, state))
            bad = copy.deepcopy(state); bad.q[16] = model.upper[16]+2e-5
            with self.assertRaises(ValueError):
                s.next_action(bad)
            self.assertEqual(s.status, "JOINT_STATE_OUT_OF_BOUNDS")
            self.assertEqual(s.finish(bad)["status"], "JOINT_STATE_OUT_OF_BOUNDS")

    def test_collision_and_tracking_envelopes_not_relaxed(self):
        model, state = self.boundary()
        s = self.servo(model, state)
        with patch.object(s.collision, "clearance", return_value=-.001):
            self.assertFalse(s.begin(Action("left", "close"), state))
        self.assertEqual(s.status, "ROBOT_COLLISION_RISK")
        s = self.servo(model, state); s.begin(Action("left", "close"), state)
        for _ in range(s.total_ticks): s.next_action(state)
        bad = copy.deepcopy(state); bad.poses["left"][0][0] += .019
        self.assertFalse(execution_completed(s.action, s.finish(bad)))

    def test_each_begin_recomputes_bounds_from_actual_pose_not_previous_action(self):
        model, boundary = self.boundary()
        s = self.servo(model, boundary); s.begin(Action("right", "hold"), boundary)
        interior = model.state(np.zeros(18), boundary.gripper, np.zeros(3))
        self.assertTrue(s.begin(Action("right", "hold"), interior))
        np.testing.assert_array_equal(s.plan_upper, model.upper-.001)
        np.testing.assert_array_equal(s.plan_lower, model.lower+.001)


if __name__ == "__main__":
    unittest.main()
