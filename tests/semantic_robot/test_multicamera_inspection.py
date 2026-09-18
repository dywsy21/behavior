from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.multicamera_inspection import MultiCameraInspection, free_observing_hand, inspection_carry
from semantic_robot.v2.protocol import Action, HOLD, ROTATIONS
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.prompt_context import actor_context
import json
from test_v2 import fixture, evidence


class Guard:
    def receipt(self): return {}


class MultiCameraTests(unittest.TestCase):
    def make(self, level=False):
        model, state = fixture(); state.gripper[:] = .035
        # A wrist camera is rigidly attached to its moving hand in this test.
        for hand in ("left", "right"):
            camera, _ = model.links["camera_" + hand + "_wrist"]
            model.links["camera_" + hand + "_wrist"] = (camera, model.links[hand][1])
        h = GroundedHarness([Goal("press", "button on held object", "left", "effect")],
                            held_inspection=True, inspection_budget_aware=True, multicamera_inspection=True)
        h.held["right"] = "object"; h.hold_verified["right"] = True; h.level["right"] = level
        obs = GroundedEvidence(**evidence(visible=False, view="none", target_uv=None).as_dict(), target_reference="held_right")
        h.observe(obs, state)
        servo = SafeServo(model, state, [1, -1])
        ctl = GroundedController(model, servo, h)
        inspector = ctl.inspector
        inspector.remember(model, state, "right", {"valid": True, "point_base_m": model.forward(state.q, "right")[:3, 3].tolist()})
        inspector.observe(model, state, h)
        return model, state, h, servo, inspector

    def test_opt_in_requires_original_budget_contract(self):
        with self.assertRaises(ValueError):
            GroundedHarness([Goal("press", "button", "left", "effect")], multicamera_inspection=True)
        self.assertIsInstance(self.make()[-1], MultiCameraInspection)

    def test_palette_preserves_grip_base_and_held_level(self):
        _, _, h, _, _ = self.make(level=True)
        choices = h.palette()
        self.assertIn(Action("left", "roll_plus", "coarse", "tool"), choices)
        self.assertFalse(any(a.part == "right" and a.move in ROTATIONS for a in choices))
        self.assertFalse(any(a.move in ("open", "close") or a.part in ("base", "torso") for a in choices))
        self.assertFalse(inspection_carry(h, Action("left", "roll_plus", "coarse", "tool")))
        self.assertTrue(inspection_carry(h, Action("right", "up")))

    def test_free_rotation_holds_other_arm_and_grip(self):
        model, state, h, servo, _ = self.make(level=True)
        action = Action("left", "roll_plus", "coarse", "tool")
        self.assertTrue(servo.begin(action, state, inspection_carry(h, action)))
        fixed = list(range(4)) + list(range(11, 18))
        np.testing.assert_allclose(servo.joint_plan[:, fixed], 0.)
        np.testing.assert_allclose(servo.grips, [1, -1])

    def test_occupied_or_unverified_other_hand_not_used_as_observer(self):
        for kind in ("held", "hold_verified", "pending_grasp"):
            _, _, h, _, _ = self.make()
            getattr(h, kind)["left"] = "other" if kind == "held" else True
            self.assertIsNone(free_observing_hand(h))
            self.assertFalse(any(a.part == "left" for a in h.palette()))

    def test_ownership_change_and_missing_depth_fail_closed(self):
        model, state, h, servo, inspector = self.make()
        allowed, receipt = inspector.candidates(model, state, h, servo, Guard())
        self.assertFalse(any(a.part == "left" for a in allowed))
        self.assertTrue(any(r["reason"] == "CURRENT_FREE_ARM_DEPTH_GUARD_REQUIRED" for r in receipt["tested"]))
        h.pending_grasp["left"] = True
        self.assertFalse(inspector.observe(model, state, h)[1]["valid"])
        self.assertEqual(h.stop_reason, "OBSERVER_HAND_OWNERSHIP_CHANGED")

    def test_shared_attempt_bound_and_separate_motion_accounts(self):
        model, state, h, servo, inspector = self.make()
        q = state.q.copy(); q[4] = .012
        changed = model.state(q, state.gripper, state.base_velocity)
        inspector.executed(h); inspector.observe(model, changed, h)
        context = inspector.context(model, changed, h)
        self.assertAlmostEqual(context["free_arm_path_m"], .012)
        self.assertEqual(context["relative_path_m"], 0.)
        self.assertEqual(context["attempts"], 1)
        inspector.states[h.index]["attempts"] = 24
        self.assertEqual(inspector.candidates(model, changed, h, servo, Guard())[0], (HOLD,))

    def test_accepted_free_candidates_have_sweep_receipts_not_visibility_claims(self):
        model, state, h, servo, inspector = self.make()
        inspector.free_guard = SimpleNamespace(arm="left", check=lambda plan: (True, {"reason": "test_sweep"}))
        allowed, receipt = inspector.candidates(model, state, h, servo, Guard())
        self.assertTrue(any(a.part == "left" for a in allowed))
        for row in receipt["tested"]:
            if row["accepted"] and row["action"]["part"] == "left":
                self.assertEqual(row["visible_arm_sweep"]["reason"], "test_sweep")
                self.assertTrue(row["inspection_after"]["not_affordance_or_visibility_evidence"])
        self.assertFalse(h.observation.visible)

    def test_carry_exception_does_not_escape_inspection(self):
        _, _, h, _, _ = self.make(level=True)
        h.stage = "ALIGN"
        self.assertTrue(inspection_carry(h, Action("left", "roll_plus", "coarse", "tool")))
        h.stage = "SEARCH"; h.observation = replace(h.observation, visible=True, view="head", target_uv=[.5, .5])
        self.assertTrue(inspection_carry(h, Action("left", "roll_plus", "coarse", "tool")))

    def test_compact_scores_keep_bound_command_and_camera_not_repeated_names(self):
        model, state, h, servo, inspector = self.make()
        inspector.free_guard = SimpleNamespace(arm="left", check=lambda plan: (True, {"reason": "test_sweep"}))
        allowed, h.candidate_receipt = inspector.candidates(model, state, h, servo, Guard())
        payload=json.loads(actor_context(h,state,SimpleNamespace(geometry={}),allowed))
        receipt=payload["CURRENT preflight receipt"]
        self.assertTrue(receipt["inspection_anchor_is_not_affordance_or_visibility_evidence"])
        self.assertEqual([r["command_index"] for r in receipt["scores_for_allowed_commands"]],list(range(len(allowed))))
        for row in receipt["scores_for_allowed_commands"]:
            after=row.get("inspection_after")
            if after:
                self.assertIn("observer_camera",after);self.assertIn("pointing_gain_deg",after)
                self.assertNotIn("motion_hand",after);self.assertNotIn("reference_hand",after)
