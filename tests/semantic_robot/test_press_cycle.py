"""CPU contract tests: real controller/FK/IK, synthetic sensors and ideal plant."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.press_cycle import VERSION, checked_action_motion
from semantic_robot.v2.press_reference import FingerSurfaceAsset
from semantic_robot.v2.protocol import HOLD, Action
from semantic_robot.v2.servo import SafeServo, ServoLimits
from test_finger_kinematics import finger_spec, current
from test_grounded import grounded_evidence
from test_hand_body_collision import calibrated_fixture
from test_press_reference import surface_spec


def synthetic_motion(start, end, delta=(0., 0., 0.)):
    # Synthetic sensor chain, NOT an RGB-D estimator or native physical test.
    transform = np.eye(4); transform[:2, 3] = delta[:2]
    angle = delta[2]
    transform[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    segments = []
    for a in range(start, end, 6):
        b = min(a+6, end)
        T = transform if b == end else np.eye(4)
        segments.append({"control_start": a, "control_end": b,
            "measurement": {"valid": True, "body_transform_current_in_previous": T.tolist(),
                            "body_delta": list(delta) if b == end else [0., 0., 0.]}})
    return {"valid": True, "source": "fixed_six_control_RGBD_SE3_chain", "no_scene_truth": True,
            "velocity_integral_not_used": True, "robot_self_exclusion": True,
            "substep_interval_controls": 6, "control_start": start, "control_end": end,
            "segments": segments, "body_transform_current_in_previous": transform.tolist(), "body_delta": list(delta)}


class PressCycleTests(unittest.TestCase):
    def setUp(self):
        old, old_state, self.geometry = calibrated_fixture()
        spec = copy.deepcopy(old.spec)
        spec["metadata"]["finger_kinematics"] = finger_spec()
        self.model = RobotModel(spec)
        self.state = self.model.state(old_state.q, old_state.gripper, np.zeros(3), current())
        self.asset = FingerSurfaceAsset(surface_spec(), "a"*64)
        self.manager = GroundedHarness([Goal("press", "visible switch", "right", "visible effect")])
        self.servo = SafeServo(self.model, self.state, [1, 1], ServoLimits(robot_geometry_guards=True))
        self.c = GroundedController(self.model, self.servo, self.manager,
                                   press_finger_surfaces=self.asset, press_cycle_v1=True,
                                   visual_odometry=True, odometry_self_exclusion=True, odometry_estimator="rgbd_joint")
        self.motion_receipt = None
        self.c.motion = SimpleNamespace(observe=lambda *a, **kw: copy.deepcopy(self.motion_receipt))
        self.cycle = self.c.press_cycle
        _, transforms = self.asset.context(self.model, self.state.q, self.state.finger_qpos)
        self.target = (transforms["right_f0"] @ [.016, 0, 0, 1])[:3]
        self.depths = {v: np.full((100, 100), 1.2, np.float32) for v in ("head", "left_wrist", "right_wrist")}
        self.depth_receipt = {v: {"valid_fraction": 1.} for v in self.depths}
        self.control = 0

    def observe(self, **changes):
        if self.c.pending_motion is not None:
            self.c.update_motion({}, {}, self.state, control=self.control)
        point = {"valid": True, "point_base_m": self.target.tolist(), "views": []}
        with patch("semantic_robot.v2.grounded_harness.localize_target", return_value=point):
            self.c.observe(grounded_evidence(**changes), self.state, self.depths, self.depth_receipt,
                           self_geometry=self.geometry, control=self.control)

    def execute(self, action=None, *, limit=None, sample_edit=None, interrupted=False, body_delta=(0., 0., 0.)):
        if action is None:
            allowed = self.c.candidates(self.state)
            self.assertIsNone(self.manager.stop_reason, self.manager.stop_reason)
            self.assertEqual(len(allowed), 1)
            action = allowed[0]
        self.assertTrue(self.c.press_execution_check(self.state, action)["eligible"])
        self.assertTrue(self.servo.begin(action, self.state, self.manager.carry), self.servo.status)
        self.assertTrue(self.c.press_trial_check(self.state, action, self.servo)["eligible"])
        self.assertTrue(self.cycle.begin_execution(action, self.state, self.control), self.manager.stop_reason)
        start = self.control
        ticks = self.servo.total_ticks if limit is None else limit
        for i in range(ticks):
            command = self.servo.next_action(self.state)
            q = np.r_[command[3:14], command[15:22]]
            positions = dict(self.state.finger_qpos)
            if sample_edit:
                sample_edit(i, positions)
            self.state = self.model.state(q, self.state.gripper, np.zeros(3), positions)
            self.control += 1
            if not self.cycle.post_step(self.state, self.control, command):
                self.servo.abort(self.cycle.stop_reason)
                break
        feedback = self.servo.finish(self.state)
        self.motion_receipt = synthetic_motion(start, self.control, body_delta)
        feedback = self.cycle.finish_execution(action, feedback, self.state, self.control, interrupted=interrupted,
                                              motion_receipt=self.motion_receipt)
        self.c.executed(action, feedback)
        return feedback

    def prepare(self):
        self.observe()
        self.assertIsNone(self.c.press_reference)
        self.assertNotIn("distance_to_active_reference_m", self.c.target)
        self.assertEqual(self.cycle.phase, "PREPARE")
        feedback = self.execute()
        self.assertEqual(feedback["press_execution"]["purpose"], "PREPARE")
        self.assertTrue(self.cycle.tool_ready)
        self.observe()
        return feedback

    def reach_dwell(self):
        self.prepare()
        for _ in range(4):
            if self.cycle.phase == "DWELL":
                break
            self.assertEqual(self.cycle.phase, "ADVANCE")
            self.execute()
            self.observe()
        self.assertEqual(self.cycle.phase, "DWELL")

    def reach_verify(self):
        self.reach_dwell()
        feedback = self.execute()
        self.assertEqual(feedback["press_execution"]["purpose"], "DWELL")
        self.assertEqual(feedback["control_ticks"], 18)
        self.observe(effect=True)
        self.assertEqual(self.cycle.phase, "RETRACT")
        self.assertEqual(self.manager.completed, [])
        for _ in range(3):
            if self.cycle.phase == "VERIFY":
                break
            feedback = self.execute()
            if self.cycle.phase != "VERIFY":
                self.observe(effect=True)
        self.assertEqual(self.cycle.phase, "VERIFY")
        self.assertGreaterEqual(feedback["press_execution"]["measured_away_displacement_m"], .003)

    def test_default_disabled_and_flag_dependencies(self):
        other = GroundedHarness([Goal("press", "button", "right", "visible change")])
        c = GroundedController(self.model, self.servo, other, press_finger_surfaces=self.asset)
        self.assertIsNone(c.press_cycle)
        self.assertNotIn("press_cycle", other.context())
        with self.assertRaises(ValueError):
            GroundedController(self.model, self.servo, other, press_cycle_v1=True)
        bare = SafeServo(self.model, self.state)
        with self.assertRaises(ValueError):
            GroundedController(self.model, bare, other, press_finger_surfaces=self.asset, press_cycle_v1=True)

    def test_prepare_uses_actual_six_samples_and_binds_only_after_stable(self):
        original = self.servo.grips.copy()
        feedback = self.prepare()
        r = feedback["press_execution"]
        self.assertEqual(r["version"], VERSION)
        self.assertEqual(r["executed_control_ticks"], 18)
        self.assertEqual(len(r["last_finger_samples_m"]), 6)
        self.assertTrue(r["stable_last_six"])
        np.testing.assert_equal(self.servo.grips, original)
        self.assertIsNotNone(self.c.press_reference)
        self.assertFalse(r["physical_contact_evidence"])
        self.assertFalse(r["success_claim"])

    def test_complete_chain_requires_retraction_and_two_fresh_effect_views(self):
        self.reach_verify()
        self.observe(effect=True)
        self.assertEqual(self.cycle.confirmations, 1)
        self.assertFalse(self.manager.completed)
        self.observe(effect=True)  # Same frozen control stamp cannot confirm twice.
        self.assertEqual(self.cycle.confirmations, 1)
        self.execute()
        self.observe(effect=True)
        self.assertEqual(self.cycle.phase, "FINISHED")
        self.assertEqual(len(self.manager.completed), 1)
        result = self.manager.completed[0]
        self.assertEqual(result["status"], "OBSERVATION_VERIFIED_NOT_OFFICIAL_TRUTH")
        self.assertFalse(result["press_execution_and_observation"]["official_success_claim"])
        self.assertEqual(self.manager.stop_reason, "PLAN_EXHAUSTED_NOT_OFFICIAL_SUCCESS")

    def test_effect_claim_before_dwell_does_not_complete(self):
        self.prepare()
        for _ in range(3):
            self.observe(effect=True)
        self.assertFalse(self.manager.completed)
        self.assertEqual(self.cycle.phase, "ADVANCE")

    def test_unknown_effect_exhausts_three_views_not_infinite_holds(self):
        self.reach_verify()
        for i in range(3):
            self.observe(effect=None)
            if i < 2:
                self.execute()
        self.assertEqual(self.manager.stop_reason, "PRESS_EFFECT_UNVERIFIED")
        self.assertFalse(self.manager.completed)

    def test_grip_command_not_currently_at_fingers_is_not_auto_formed(self):
        self.servo.grips[1] = -1
        self.observe()
        self.assertEqual(self.manager.stop_reason, "PRESS_CURRENT_TOOL_NOT_AT_EXISTING_COMMAND")
        self.assertFalse(self.cycle.tool_ready)
        self.assertEqual(self.control, 0)
        self.assertEqual(self.servo.grips[1], -1)

    def test_every_load_latch_blocks_preparation_without_release(self):
        for field, value in (("held", "object"), ("hold_verified", True),
                             ("pending_grasp", True), ("possible_contact_after_close", True)):
            self.setUp()
            getattr(self.manager, field)["right"] = value
            self.observe()
            self.assertEqual(self.manager.stop_reason, "PRESS_HAND_POSSIBLY_LOADED")
            self.assertEqual(self.control, 0)
            np.testing.assert_equal(self.servo.grips, [1, 1])

    def test_other_held_hand_command_is_preserved(self):
        self.manager.held["left"] = "radio"
        self.manager.hold_verified["left"] = True
        self.servo.grips[0] = -1
        self.prepare()
        self.assertEqual(self.servo.grips[0], -1)
        self.assertEqual(self.manager.held["left"], "radio")

    def test_same_mean_finger_drift_does_not_pass_stability(self):
        self.observe()
        def perturb(i, positions):
            positions["right_j0"] = .05-(.0004 if i % 2 == 0 else 0)
            positions["right_j1"] = .05
        self.execute(sample_edit=perturb)
        self.assertEqual(self.manager.stop_reason, "PRESS_TOOL_UNSTABLE")
        self.assertFalse(self.cycle.tool_ready)

    def test_last_tick_drift_aborts_even_if_earlier_samples_stable(self):
        self.prepare()
        def perturb(i, positions):
            if i == self.servo.total_ticks-1:
                positions["right_j0"] -= .001
        feedback = self.execute(sample_edit=perturb)
        self.assertFalse(feedback["press_execution"]["command_complete"])
        self.assertEqual(self.manager.stop_reason, "PRESS_CURRENT_TOOL_NOT_AT_EXISTING_COMMAND")

    def test_partial_dwell_and_budget_stop_never_count_as_dwell(self):
        self.reach_dwell()
        feedback = self.execute(limit=5, interrupted=True)
        self.assertFalse(feedback["press_execution"]["command_complete"])
        self.assertIsNone(self.cycle.dwell_end)
        self.assertEqual(feedback["press_execution"]["executed_control_ticks"], 5)
        self.assertFalse(self.manager.completed)

    def test_actual_same_action_visual_body_drift_vetoes_before_retract(self):
        self.reach_dwell()
        feedback = self.execute(body_delta=(.005, 0., 0.))
        self.assertEqual(feedback["base_integral"], [0., 0., 0.])  # Raw servo not rewritten.
        r = feedback["press_execution"]
        self.assertEqual(r["measured_body_delta"], [.005, 0., 0.])
        self.assertEqual(r["base_motion_source"], "onboard_RGBD_not_joint_velocity_integration")
        self.assertFalse(r["command_complete"])
        self.assertIsNone(self.cycle.dwell_end)
        self.assertEqual(self.cycle.phase, "STOPPED")
        self.assertEqual(self.manager.stop_reason, "PRESS_BODY_MOVED_DURING_NEAR_INTERACTION")
        self.observe(effect=True)
        self.assertEqual(self.cycle.phase, "STOPPED")
        self.assertFalse(self.manager.completed)

    def test_motion_requires_same_complete_control_chain_not_bare_delta(self):
        receipt = synthetic_motion(12, 30)
        self.assertIs(checked_action_motion(receipt, 12, 30), receipt)
        edits = [lambda r: r.update(control_start=0), lambda r: r.update(robot_self_exclusion=False),
                 lambda r: r["segments"].pop(), lambda r: r["segments"][1].update(control_start=17),
                 lambda r: r["segments"][0]["measurement"].update(valid=False),
                 lambda r: r.update(body_delta=[.005, 0., 0.])]
        for edit in edits:
            changed = copy.deepcopy(receipt); edit(changed)
            with self.assertRaises(ValueError): checked_action_motion(changed, 12, 30)
        with self.assertRaises(ValueError):
            checked_action_motion({"valid": True, "body_delta": [0., 0., 0.]}, 12, 30)

    def test_actual_near_pose_overruns_checked_surface_before_next_observation(self):
        self.reach_dwell()
        self.c.candidates(self.state)
        self.servo.begin(HOLD, self.state)
        self.assertTrue(self.cycle.begin_execution(HOLD, self.state, self.control))
        command = self.servo.next_action(self.state)
        q = self.state.q.copy(); q[11] += .008
        actual = self.model.state(q, self.state.gripper, np.zeros(3), self.state.finger_qpos)
        self.assertFalse(self.cycle.post_step(actual, self.control+1, command))
        self.assertEqual(self.cycle.stop_reason, "PRESS_EXECUTED_FACE_OR_CHANNEL_INVALID")
        self.assertIsNone(self.cycle.dwell_end)

    def test_missing_selection_or_samples_cannot_forge_hold_completion(self):
        self.observe()
        self.servo.begin(HOLD, self.state)
        self.assertFalse(self.cycle.begin_execution(HOLD, self.state, self.control))
        self.assertEqual(self.manager.stop_reason, "PRESS_PURPOSE_NOT_SELECTED_BY_EXECUTOR")
        self.setUp(); self.observe()
        self.c.candidates(self.state)
        self.servo.begin(HOLD, self.state)
        self.assertTrue(self.cycle.begin_execution(HOLD, self.state, 0))
        forged = {"status": "TARGET_REACHED", "control_ticks": 18}
        feedback = self.cycle.finish_execution(HOLD, forged, self.state, 18)
        self.assertFalse(feedback["press_execution"]["command_complete"])
        self.assertFalse(self.cycle.tool_ready)

    def test_ordinary_hold_during_approach_does_not_become_dwell(self):
        self.target[0] += .1
        self.prepare()
        self.assertEqual(self.cycle.phase, "APPROACH")
        feedback = self.execute(HOLD)
        self.assertIsNone(feedback["press_execution"]["purpose"])
        self.observe(effect=True)
        self.assertEqual(self.cycle.phase, "APPROACH")
        self.assertIsNone(self.cycle.dwell_end)
        self.assertFalse(self.manager.completed)

    def test_goal_reference_and_calibration_changes_abort(self):
        self.prepare()
        self.manager.goals[0] = Goal("press", "another button", "right", "visible effect")
        self.observe()
        self.assertEqual(self.manager.stop_reason, "PRESS_GOAL_OR_TARGET_REFERENCE_CHANGED")
        self.setUp(); self.prepare()
        self.model.links["right"][0][0, 3] += .001
        self.observe()
        self.assertEqual(self.manager.stop_reason, "PRESS_CALIBRATION_CHANGED")

    def test_changed_joints_after_hold_selection_do_not_execute(self):
        self.observe(); self.c.candidates(self.state)
        q = self.state.q.copy(); q[11] += .002
        changed = self.model.state(q, self.state.gripper, np.zeros(3), self.state.finger_qpos)
        self.assertFalse(self.cycle.begin_execution(HOLD, changed, self.control))
        self.assertEqual(self.manager.stop_reason, "PRESS_OBSERVATION_ROBOT_CHANGED")

    def test_hazard_does_not_get_overridden_by_protocol(self):
        self.reach_dwell()
        self.observe(hazard="collision", effect=True)
        self.assertIsNotNone(self.manager.stop_reason)
        self.assertFalse(self.manager.completed)

    def test_exhausted_recovery_hazard_cannot_replan_into_the_same_dwell(self):
        for hazard in ("collision", "slip"):
            self.setUp(); self.reach_dwell()
            self.manager.recoveries = self.manager.max_recoveries
            self.observe(hazard=hazard)
            self.assertEqual(self.cycle.phase, "STOPPED")
            self.assertIsNotNone(self.cycle.stop_reason)
            self.assertIsNone(self.c.replan_needed)
            with self.assertRaisesRegex(ValueError, "stopped press"):
                self.c.apply_recovery({"strategy": "retry_approach", "visible_reason": "retry"})
            self.assertFalse(self.cycle.begin_execution(HOLD, self.state, self.control))
            self.assertIsNone(self.cycle.dwell_end)

    def test_target_loss_does_not_get_overridden_by_protocol(self):
        self.reach_dwell()
        with patch("semantic_robot.v2.grounded_harness.localize_target", return_value={"valid": False, "reason": "NO_TARGET"}):
            self.c.observe(grounded_evidence(visible=False, view="none", target_uv=None), self.state,
                           self.depths, self.depth_receipt, self_geometry=self.geometry, control=self.control)
        self.assertEqual(self.manager.stop_reason, "PRESS_FRESH_TARGET_UNAVAILABLE")
        self.assertFalse(self.manager.completed)

    def test_real_runner_wraps_actual_step_and_adjudicated_feedback(self):
        path = Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py"
        tree = ast.parse(path.read_text())
        calls = [(n.lineno, n.func.attr) for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr in
                 ("begin_execution", "post_step", "finish_execution")]
        self.assertEqual([name for _, name in sorted(calls)], ["begin_execution", "post_step", "finish_execution"])
        source = path.read_text()
        self.assertLess(source.index("ended = step(command"), source.index("press_cycle.post_step("))
        self.assertLess(source.index("feedback=observed_motion_feedback(action,feedback,motion_receipt)"),
                        source.index("press_cycle.finish_execution("))
        self.assertLess(source.index("press_cycle.finish_execution("), source.index("controller.executed(action,feedback)"))
        self.assertIn('g.get("press_cycle_v1",False)==args.press_cycle_v1', source)
        self.assertIn('motion_receipt=substep_result', source)


if __name__ == "__main__":
    unittest.main()
