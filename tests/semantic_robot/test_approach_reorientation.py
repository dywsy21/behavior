import copy
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.approach_reorientation import eligible, preview
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import Action, HOLD, ROTATIONS, TRANSLATIONS
from semantic_robot.v2.servo import SafeServo
from test_v2 import fixture, evidence


def setup():
    model, state = fixture()
    manager = GroundedHarness([Goal("pick", "visible object", "right", "moves with hand")],
                              approach_reorientation=True)
    manager.stage = "APPROACH"
    manager.last_gripper = state.gripper.copy()
    manager.observation = evidence(hazard="none")
    manager.grounding = {"valid": True, "point_base_m": [1., .3, 1.3],
                         "distance_to_active_closing_center_m": 1.}
    servo = SafeServo(model, state, [1., 1.])
    controller = GroundedController(model, servo, manager)
    controller.target = manager.grounding
    controller.centers = model.grasp_centers(state.q)
    controller.depth_guard = SimpleNamespace(check=lambda *a: (True, "FIXTURE_ONLY"), receipt=lambda: {})
    return model, state, manager, servo, controller


class ApproachReorientationTests(unittest.TestCase):
    def test_opt_in_and_unloaded_far_single_pick_only(self):
        _, _, h, _, _ = setup()
        self.assertTrue(eligible(h))
        rotations = {a for a in h.palette() if a.move in ROTATIONS and a.part == "right"}
        self.assertEqual(len(rotations), 12)
        h.approach_reorientation = False
        self.assertFalse(eligible(h))
        self.assertFalse(any(a.move in ROTATIONS and a.part == "right" for a in h.palette()))

    def test_unknown_load_stage_target_or_aperture_never_unlocks_rotation(self):
        mutations = [lambda h: h.pending_grasp.update(right=True),
                     lambda h: h.hold_verified.update(left=True),
                     lambda h: h.possible_contact_after_close.update(right=True),
                     lambda h: setattr(h, "last_gripper", None),
                     lambda h: setattr(h, "last_gripper", np.array([.05, .02])),
                     lambda h: setattr(h, "last_gripper", np.array([.05, np.nan])),
                     lambda h: h.grounding.update(valid=False),
                     lambda h: h.grounding.update(distance_to_active_closing_center_m=np.inf),
                     lambda h: setattr(h, "observation", None),
                     lambda h: setattr(h, "observation", evidence(hazard="collision")),
                     lambda h: h.grounding.update(distance_to_active_closing_center_m=.09),
                     lambda h: setattr(h, "stage", "GRASP"),
                     lambda h: setattr(h, "stop_reason", "STOP"),
                     lambda h: setattr(h, "goals", [Goal("pick", "object", "both", "held")]),
                     lambda h: setattr(h, "goals", [Goal("press", "button", "right", "effect")])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                _, _, h, _, _ = setup()
                mutate(h)
                self.assertFalse(eligible(h))

    def test_unknown_or_incomplete_load_maps_never_offer_rotation(self):
        for name in ("pending_grasp", "hold_verified", "possible_contact_after_close"):
            for value in (None, {}, {"left": False}, {"left": False, "right": None},
                          {"left": False, "right": 0},
                          {"left": False, "right": False, "extra": False}):
                with self.subTest(name=name, value=value):
                    _, state, h, _, controller = setup()
                    setattr(h, name, value)
                    self.assertFalse(eligible(h))
                    # The legacy general palette may itself reject malformed
                    # maps. Complete but unknown values must not unlock it.
                    if isinstance(value, dict) and set(value) == {"left", "right"}:
                        self.assertFalse(any(a.part == "right" and a.move in ROTATIONS
                                             for a in h.palette()))

    def test_out_of_range_or_nonfinite_command_latch_never_offers_rotation(self):
        for latch in ([1., 1.1], [1., np.inf], [np.nan, 1.]):
            with self.subTest(latch=latch):
                _, state, h, servo, controller = setup()
                servo.grips = np.asarray(latch)
                # Other actions may be rejected by the native servo. In
                # either case rotation must not pass this eligibility gate.
                try:
                    allowed = controller.candidates(state)
                except ValueError:
                    continue
                self.assertFalse(h.candidate_receipt["approach_reorientation"]["eligible"])
                self.assertFalse(any(a.part == "right" and a.move in ROTATIONS for a in allowed))

    def test_all_rotation_directions_survive_translation_failures_and_preview_is_not_offered(self):
        model, state, h, servo, controller = setup()
        before = state.q.copy()
        original = SafeServo.begin

        def blocked_until_reoriented(trial, action, current, carry=False):
            if (action.part == "right" and action.move in TRANSLATIONS and action.scale == "coarse"
                    and np.max(np.abs(current.q[14:17])) < 1e-4):
                return trial.abort("FIXTURE_COARSE_BLOCKED")
            return original(trial, action, current, carry)

        with patch.object(SafeServo, "begin", blocked_until_reoriented):
            allowed = controller.candidates(state)
        receipt = h.candidate_receipt
        tested_rotations = [r for r in receipt["tested"] if r["action"]["part"] == "right" and r["action"]["move"] in ROTATIONS]
        self.assertEqual({r["action"]["move"] for r in tested_rotations}, set(ROTATIONS))
        self.assertLessEqual(len(receipt["tested"])-1, 32)
        lookahead = receipt["approach_reorientation"]["preview"]
        self.assertIsNotNone(lookahead)
        self.assertLessEqual(lookahead["additional_preflights"], 18)
        self.assertTrue(lookahead["future_actions_not_authorized"])
        self.assertTrue(any(r["summary"]["feasible_coarse_followups"] for r in lookahead["rows"]))
        self.assertFalse(any(a.part == "right" and a.move in TRANSLATIONS and a.scale == "coarse" for a in allowed))
        np.testing.assert_array_equal(state.q, before)
        np.testing.assert_array_equal(servo.grips, [1., 1.])
        self.assertEqual(servo.status, "IDLE")
        # Actor sees the boundary, not full hypothetical joint arrays or a queue.
        value = json.loads(actor_context(h, state, SimpleNamespace(geometry={}), allowed))["CURRENT preflight receipt"]
        self.assertTrue(value["approach_reorientation"]["future_actions_not_authorized"])
        scores = [s for s in value["scores_for_allowed_commands"] if "reorientation_after" in s]
        self.assertTrue(scores)
        self.assertTrue(all(s["reorientation_after"]["prediction_not_execution_or_environment_safety"] for s in scores))

    def test_coarse_rotation_rejection_tries_fine_without_changing_motion_amount(self):
        _, state, h, _, controller = setup()
        original = SafeServo.begin

        def reject_coarse_rotations(trial, action, current, carry=False):
            if action.part == "right" and action.move in ROTATIONS and action.scale == "coarse":
                return trial.abort("FIXTURE_COARSE_ROTATION_BLOCKED")
            return original(trial, action, current, carry)

        with patch.object(SafeServo, "begin", reject_coarse_rotations):
            allowed = controller.candidates(state)
        self.assertEqual({a.move for a in allowed if a.part == "right" and a.move in ROTATIONS}, set(ROTATIONS))
        self.assertTrue(all(a.scale == "fine" for a in allowed if a.part == "right" and a.move in ROTATIONS))
        self.assertIsNone(h.candidate_receipt["approach_reorientation"]["preview"])  # coarse reach already works

    def test_no_preview_when_command_latch_is_closed_even_if_measured_fingers_open(self):
        _, state, h, servo, controller = setup()
        servo.grips[1] = -1
        allowed = controller.candidates(state)
        self.assertFalse(h.candidate_receipt["approach_reorientation"]["eligible"])
        self.assertFalse(any(a.part == "right" and a.move in ROTATIONS for a in allowed))

    def test_preview_rejects_limits_invalid_target_loaded_or_stale_state(self):
        model, state, h, servo, _ = setup()
        action = Action("right", "pitch_minus", "fine", "tool")
        self.assertTrue(servo.begin(action, state, False))
        args = (model, state, np.array([1., 1.]), servo.limits, "right", [1., 0., 1.])
        forward = Action("right", "forward", "coarse")
        original_q = state.q.copy()
        result = preview(*args, [(action, servo)], [forward])
        self.assertEqual(result["additional_preflights"], 1)
        np.testing.assert_array_equal(state.q, original_q)
        for rotations, translations in (([(action, servo)]*7, [forward]), ([(action, servo)], [forward]*4),
                                        ([(action, servo)], [Action("left", "forward", "coarse")])):
            with self.assertRaises(ValueError): preview(*args, rotations, translations)
        with self.assertRaises(ValueError): preview(model, state, [-1, 1], servo.limits, "right", [1, 0, 1], [(action, servo)], [forward])
        with self.assertRaises(ValueError): preview(model, state, [1, 1.1], servo.limits, "right", [1, 0, 1], [(action, servo)], [forward])
        with self.assertRaises(ValueError): preview(model, state, [1, 1], servo.limits, "right", [np.nan, 0, 1], [(action, servo)], [forward])
        altered = copy.deepcopy(state)
        altered.q[11] += .01
        with self.assertRaises(ValueError): preview(model, altered, [1, 1], servo.limits, "right", [1, 0, 1], [(action, servo)], [forward])


if __name__ == "__main__":
    unittest.main()
