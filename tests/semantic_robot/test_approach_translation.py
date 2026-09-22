import argparse
import ast
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.approach_translation import preview
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import Action, TRANSLATIONS
from semantic_robot.v2.servo import SafeServo, ServoLimits
from semantic_robot.v2.wall_budget import WallTimeBudgetReached
from test_hand_body_collision import calibrated_fixture
from test_v2 import evidence


def setup(enabled=True):
    model, state, _ = calibrated_fixture()
    h = GroundedHarness([Goal("pick", "object", "right", "moves with hand")],
                        approach_reorientation=True, approach_translation_preview=enabled)
    h.stage = "APPROACH"
    h.last_gripper = state.gripper.copy()
    h.observation = evidence()
    point = np.array([1., -.3, 1.3])
    h.grounding = {"valid": True, "point_base_m": point.tolist(),
                   "distance_to_active_closing_center_m": float(np.linalg.norm(point-model.grasp_centers(state.q)["right"]))}
    servo = SafeServo(model, state, [1., 1.], ServoLimits(robot_geometry_guards=True))
    controller = GroundedController(model, servo, h)
    controller.target = h.grounding
    controller.centers = model.grasp_centers(state.q)
    controller.depth_guard = SimpleNamespace(check=lambda *a: (True, "FIXTURE_ONLY"), receipt=lambda: {})
    return model, state, h, servo, controller


def block_coarse_until_up(model, state, reason="UNREACHABLE_OR_COLLISION_BLOCKED"):
    original = SafeServo.begin
    start_z = model.grasp_centers(state.q)["right"][2]

    def begin(trial, action, current, carry=False):
        if (action.part == "right" and action.move in TRANSLATIONS and action.scale == "coarse"
                and model.grasp_centers(current.q)["right"][2] < start_z+.004):
            return trial.abort(reason)
        return original(trial, action, current, carry)
    return begin


class TranslationPreviewTests(unittest.TestCase):
    def test_constructor_default_off_and_dependency(self):
        goals = [Goal("pick", "object", "right", "held")]
        self.assertFalse(GroundedHarness(goals).approach_translation_preview)
        with self.assertRaises(ValueError):
            GroundedHarness(goals, approach_translation_preview=True)

    def test_two_step_unlock_hint_no_new_action_or_state_mutation(self):
        model, state, h, servo, controller = setup()
        before = state.q.copy()
        with patch.object(SafeServo, "begin", block_coarse_until_up(model, state)):
            allowed = controller.candidates(state)
        receipt = h.candidate_receipt
        lookahead = receipt["approach_translation_preview"]["preview"]
        self.assertIsNotNone(lookahead)
        self.assertLessEqual(lookahead["additional_preflights"], 3)
        self.assertEqual(lookahead["additional_preflights"], len(lookahead["rows"]))
        self.assertTrue(lookahead["future_actions_not_authorized"])
        # A wrist-frame BACK can be base-frame UP; assert physical direction,
        # not a particular label surviving the existing direction deduplication.
        self.assertTrue(any(controller._translation_direction(Action(**r["translation"]),state)[2] > .9
                            and r["summary"]["feasible_coarse_followups"]
                            for r in lookahead["rows"]))
        self.assertTrue(all(Action(**r["translation"]) in allowed for r in lookahead["rows"]))
        self.assertFalse(any(a.part == "right" and a.move in TRANSLATIONS and a.scale == "coarse" for a in allowed))
        h.approach_translation_preview = False
        with patch.object(SafeServo, "begin", block_coarse_until_up(model, state)):
            unchanged = controller.candidates(state)
        self.assertEqual(allowed, unchanged)
        np.testing.assert_array_equal(state.q, before)
        np.testing.assert_array_equal(servo.grips, [1., 1.])
        self.assertEqual(servo.status, "IDLE")

    def test_prompt_reports_predictions_only_for_currently_offered_actions(self):
        model, state, h, _, controller = setup()
        with patch.object(SafeServo, "begin", block_coarse_until_up(model, state)):
            allowed = controller.candidates(state)
        value = json.loads(actor_context(h, state, SimpleNamespace(geometry={}), allowed))
        context = value["CURRENT preflight receipt"]
        self.assertTrue(context["approach_translation_preview"]["future_actions_not_authorized"])
        self.assertFalse(context["approach_translation_preview"]["scene_truth"])
        scores = context["scores_for_allowed_commands"]
        self.assertTrue(any("translation_after" in row for row in scores))
        for row in scores:
            if "translation_after" in row:
                self.assertEqual(allowed[row["command_index"]].scale, "fine")
                self.assertTrue(row["translation_after"]["prediction_not_execution_or_environment_safety"])
        self.assertNotIn("joint_plan", json.dumps(value))
        self.assertNotIn("following_robot_only_trial", json.dumps(value))

    def test_no_preview_for_obstacle_veto_or_when_coarse_already_works(self):
        for reason in (None, "OBSERVED_ARM_OBSTACLE"):
            model, state, h, _, controller = setup()
            if reason:
                with patch.object(SafeServo, "begin", block_coarse_until_up(model, state, reason)):
                    controller.candidates(state)
            else:
                controller.candidates(state)
            self.assertIsNone(h.candidate_receipt["approach_translation_preview"]["preview"])

    def test_loaded_near_hidden_closed_or_unconfirmed_states_cannot_preview(self):
        mutations = [lambda h, s, c: h.pending_grasp.update(right=True),
                     lambda h, s, c: h.possible_contact_after_close.update(left=True),
                     lambda h, s, c: h.hold_verified.update(right=True),
                     lambda h, s, c: setattr(h, "observation", None),
                     lambda h, s, c: h.grounding.update(distance_to_active_closing_center_m=.09),
                     lambda h, s, c: s.grips.__setitem__(1, -1),
                     lambda h, s, c: h.last_gripper.__setitem__(1, .02)]
        for mutate in mutations:
            model, state, h, servo, controller = setup()
            mutate(h, servo, controller)
            with patch.object(SafeServo, "begin", block_coarse_until_up(model, state)):
                controller.candidates(state)
            self.assertFalse(h.candidate_receipt["approach_translation_preview"]["eligible"])
            self.assertIsNone(h.candidate_receipt["approach_translation_preview"]["preview"])
        _, state, h, _, controller = setup()
        controller.near_contact_review = True
        controller.target = {"reason": "VISIBLE_TARGET_CONTACT_UNCONFIRMED", "near_contact_review": {}}
        self.assertEqual(len(controller.candidates(state)), 1)
        self.assertEqual(h.candidate_receipt["reason"], "CONTACT_REVIEW_ABSTAINED")

    def test_deadline_and_strict_trial_binding(self):
        model, state, _, servo, _ = setup()
        first, followup = Action("right", "up", "fine"), Action("right", "forward", "coarse")
        self.assertTrue(servo.begin(first, state))
        args = (model, state, [1, 1], servo.limits, "right", [1., -.3, 1.3])
        with self.assertRaises(WallTimeBudgetReached):
            preview(*args, [(first, servo)], followup, deadline=0)
        for trials, following in (([], followup), ([(first, servo)]*4, followup),
                                  ([(first, servo)]*2, followup),
                                  ([(Action("right", "forward", "fine"), servo)], followup),
                                  ([(first, servo)], Action("left", "forward", "coarse")),
                                  ([(first, servo)], Action("right", "forward", "fine"))):
            with self.subTest(trials=len(trials), following=following), self.assertRaises(ValueError):
                preview(*args, trials, following)
        for changes in ({"q": state.q+1e-4}, {"gripper": np.array([.05, .0496])}):
            changed = copy.deepcopy(state)
            for key, value in changes.items(): setattr(changed, key, value)
            with self.assertRaises(ValueError):
                preview(model, changed, [1, 1], servo.limits, "right", [1., -.3, 1.3], [(first, servo)], followup)
        for grips in ([1, -1], [1, 1.01], [1, np.nan]):
            with self.assertRaises(ValueError):
                preview(model, state, grips, servo.limits, "right", [1., -.3, 1.3], [(first, servo)], followup)
        with self.assertRaises(ValueError):
            preview(model, state, [1, 1], servo.limits, "right", [np.nan, 0, 1], [(first, servo)], followup)
        servo.carry = True
        with self.assertRaises(ValueError): preview(*args, [(first, servo)], followup)

    def test_deadline_expiring_during_followup_cannot_return_hint(self):
        model, state, _, servo, _ = setup()
        first, followup = Action("right", "up", "fine"), Action("right", "forward", "coarse")
        self.assertTrue(servo.begin(first, state))
        with patch("semantic_robot.v2.approach_translation.require_time",
                   side_effect=[None, None, WallTimeBudgetReached("expired")]):
            with self.assertRaises(WallTimeBudgetReached):
                preview(model, state, [1, 1], servo.limits, "right", [1., -.3, 1.3], [(first, servo)], followup)


class TranslationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text())

    def test_explicit_default_off_cli(self):
        node = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                    and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--approach-translation-preview")
        parser = argparse.ArgumentParser()
        eval(compile(ast.Expression(node), "cli", "eval"), {"p": parser})
        self.assertFalse(parser.parse_args([]).approach_translation_preview)
        self.assertTrue(parser.parse_args(["--approach-translation-preview"]).approach_translation_preview)

    def test_dependencies_and_exact_gate_identity(self):
        node = next(n for n in ast.walk(self.tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test).startswith("args.approach_translation_preview and"))
        code = compile(ast.Module(body=[node], type_ignores=[]), "dependencies", "exec")
        for enabled in (False, True):
            for geometry in (False, True):
                for posture in (False, True):
                    args = SimpleNamespace(approach_translation_preview=enabled,
                                           robot_geometry_guards=geometry, approach_reorientation=posture)
                    if enabled and not (geometry and posture):
                        with self.assertRaises(ValueError): exec(code, {"args": args})
                    else: exec(code, {"args": args})
        output = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "out.mkdir")
        self.assertLess(node.lineno, output.lineno)
        compare = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Compare)
                       and ast.unparse(n.left) == "g.get('approach_translation_preview', False)")
        code = compile(ast.Expression(compare), "gate", "eval")
        for gate in ({}, {"approach_translation_preview": False}, {"approach_translation_preview": True}):
            for value in (False, True):
                self.assertEqual(eval(code, {"g": gate, "args": SimpleNamespace(approach_translation_preview=value)}),
                                 gate.get("approach_translation_preview", False) == value)

    def test_option_reaches_harness_manifest_and_result(self):
        constructor = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Name) and n.func.id == "GroundedHarness")
        kw = next(k for k in constructor.keywords if k.arg == "approach_translation_preview")
        self.assertEqual(ast.unparse(kw.value), "args.approach_translation_preview")
        values = [v for n in ast.walk(self.tree) if isinstance(n, ast.Dict) for k, v in zip(n.keys, n.values)
                  if isinstance(k, ast.Constant) and k.value == "approach_translation_preview"]
        self.assertEqual([ast.unparse(v) for v in values], ["args.approach_translation_preview"]*2)


if __name__ == "__main__": unittest.main()
