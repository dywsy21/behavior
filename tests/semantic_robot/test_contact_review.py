import ast
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from semantic_robot.v2.contact_review import NearContactReview, apply_review
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.grounding import GroundedEvidence, localize_target
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.policy import RefinedGroundedPolicy
from semantic_robot.v2.protocol import Action, HOLD
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import VisualBundle
from test_v2 import fixture, evidence


class ContactReviewTests(unittest.TestCase):
    def setUp(self):
        self.model, self.state = fixture()
        self.model.links["right"][0][:3, 3] = [.1, 0, .8]
        self.h = GroundedHarness([Goal("pick", "object", "right", "observed lift")])
        self.h.stage = "APPROACH"
        self.h.motion_receipt = {"valid": True, "body_transform_current_in_previous": np.eye(4).tolist()}
        self.e = GroundedEvidence(**asdict(evidence(view="head", enclosed=None)))
        self.depths = {v: np.full((100,100), 1.2, np.float32) for v in ("head", "left_wrist", "right_wrist")}
        self.raw = {v: np.zeros((100,100,3), np.uint8) for v in self.depths}
        self.bundle = VisualBundle([], [], {}, self.raw)
        self.target = localize_target(self.e, self.depths, self.model, self.state.q)
        self.reviewer = NearContactReview()

    def request(self, evidence=None, target=None):
        return self.reviewer.request(self.h, self.state, evidence or self.e, target or self.target, self.model)

    def finish(self, request, chosen=True, evidence=None, target=None):
        return self.reviewer.finish(request, self.h, self.state, evidence or self.e,
                                    target or self.target, chosen, self.raw)

    def policy(self, choice=None, enabled=True):
        policy = RefinedGroundedPolicy.__new__(RefinedGroundedPolicy)
        policy.max_refinements, policy.refinements = 16, 0
        policy.contact_review = NearContactReview() if enabled else None
        policy._call = Mock(return_value=({"text": json.dumps({"candidate_id":choice})}, {}))
        return policy

    def controller(self):
        return GroundedController(self.model, SafeServo(self.model, self.state, [1,1]), self.h,
                                  visual_odometry=True, near_contact_review=True)

    def test_first_near_then_continuity_no_repeated_model_requirement(self):
        first = self.request()
        self.assertTrue(first["required"])
        self.finish(first)
        second = self.request()
        self.assertFalse(second["required"])
        self.assertEqual(second["reason"], "CURRENT_CONTACT_CONTINUITY")
        self.assertEqual(second["point_jump_m"], 0)

    def test_view_change_and_compensated_jump_require_fresh_review(self):
        self.finish(self.request())
        other = replace(self.e, view="right_wrist")
        self.assertEqual(self.request(evidence=other)["reason"], "CONTACT_VIEW_CHANGED")
        shifted = {**self.target, "point_base_m": [.08, 0, .8]}
        self.assertEqual(self.request(target=shifted)["reason"], "CURRENT_CONTACT_POINT_JUMP")
        body = np.eye(4); body[0,3] = -.08
        self.h.motion_receipt["body_transform_current_in_previous"] = body.tolist()
        self.assertFalse(self.request(target=shifted)["required"])

    def test_missing_or_nonrigid_motion_never_certifies_continuity(self):
        self.finish(self.request())
        for motion in ({}, {"valid":False}, {"valid":True,"body_transform_current_in_previous":np.zeros((4,4)).tolist()}):
            self.h.motion_receipt = motion
            self.assertEqual(self.request()["reason"], "NO_MEASURED_CONTACT_CONTINUITY")

    def test_far_and_invalid_contacts_reset_near_confirmation(self):
        self.finish(self.request())
        far = {**self.target, "point_base_m": [2,0,.8]}
        self.assertFalse(self.request(target=far)["required"])
        self.assertTrue(self.request()["required"])
        self.finish(self.request())
        self.assertFalse(self.request(target={"valid":False})["required"])
        self.assertTrue(self.request()["required"])

    def test_load_and_nonapproach_do_not_initiate_contact_review(self):
        for stage in ("GRASP", "VERIFY_GRASP", "VERIFY_PLACE"):
            self.h.stage = stage
            self.assertFalse(self.request()["required"])
        self.h.stage = "APPROACH"
        for field in ("pending_grasp", "hold_verified", "possible_contact_after_close"):
            getattr(self.h, field)["right"] = True
            self.assertFalse(self.request()["required"])
            getattr(self.h, field)["right"] = False
        self.h.goals = [Goal("pick", "object", "both", "observed lift")]
        self.assertFalse(self.request()["required"])

    def test_null_selection_keeps_visible_object_but_vetoes_ray_and_motion(self):
        policy = self.policy()
        obs, call, receipt, views = policy.refine(self.e, self.h, self.state, self.bundle, self.depths, self.model)
        self.assertTrue(obs.visible)
        self.assertTrue(receipt["original_target"]["valid"])
        self.assertTrue(receipt["attempted"])
        self.assertFalse(receipt["near_contact_review"]["confirmed_current_contact"])
        self.assertEqual(policy.refinements, 1)
        controller = self.controller()
        controller.observe(obs, self.state, self.depths, {v:{"valid_fraction":1} for v in self.depths},
                           images=self.raw, contact_review_receipt=receipt["near_contact_review"])
        self.assertTrue(self.h.observation.visible)
        self.assertFalse(controller.target["valid"])
        self.assertEqual(controller.target["reason"], "VISIBLE_TARGET_CONTACT_UNCONFIRMED")
        self.assertEqual(controller.candidates(self.state), (HOLD,))
        controller.reposition=Action("base","forward","fine")
        controller.reposition_left=3
        action, selection=controller.search_action(self.state)
        self.assertEqual(action,HOLD)
        self.assertEqual(selection["reason"],"CONTACT_REVIEW_ABSTAINED")
        self.assertFalse(any(self.h.hold_verified.values()))

    def test_selected_current_surface_is_not_grasp_success(self):
        policy = self.policy(choice=0)
        old = replace(self.e, enclosed=True, co_moving=True, effect=True)
        obs, _, receipt, _ = policy.refine(old, self.h, self.state, self.bundle, self.depths, self.model)
        target = localize_target(obs, self.depths, self.model, self.state.q)
        result = apply_review(target, receipt["near_contact_review"], self.h, self.state, obs, self.raw)
        self.assertTrue(result["valid"])
        self.assertFalse(result["near_contact_review"]["success_or_grasp_claim"])
        self.assertIsNone(obs.co_moving)
        self.assertIsNone(obs.enclosed); self.assertIsNone(obs.effect)
        self.assertFalse(any(self.h.hold_verified.values()))

    def test_search_and_recovery_cannot_bypass_first_near_review(self):
        for stage in ("SEARCH", "RECOVER"):
            self.h.stage=stage
            policy=self.policy()
            obs, _, receipt, _=policy.refine(self.e,self.h,self.state,self.bundle,self.depths,self.model)
            self.assertTrue(receipt["near_contact_review"]["required"])
            controller=self.controller()
            controller.observe(obs,self.state,self.depths,{v:{"valid_fraction":1} for v in self.depths},
                               images=self.raw,contact_review_receipt=receipt["near_contact_review"])
            self.assertEqual(controller.candidates(self.state),(HOLD,))

    def test_budget_exhaustion_vetoes_without_new_call(self):
        policy = self.policy(choice=0); policy.refinements = 16
        obs, call, receipt, _ = policy.refine(self.e, self.h, self.state, self.bundle, self.depths, self.model)
        self.assertEqual(receipt["reason"], "REFINEMENT_BUDGET_EXHAUSTED")
        self.assertIsNone(call); policy._call.assert_not_called()
        self.assertFalse(apply_review(self.target, receipt["near_contact_review"], self.h, self.state, obs, self.raw)["valid"])
        self.assertEqual(policy.refinements, 16)

    def test_disabled_preserves_valid_depth_skip_and_old_receipt(self):
        policy = self.policy(enabled=False)
        obs, call, receipt, views = policy.refine(self.e, self.h, self.state, self.bundle, self.depths, self.model)
        policy._call.assert_not_called()
        self.assertIs(obs, self.e); self.assertIsNone(call); self.assertIsNone(views)
        self.assertNotIn("near_contact_review", receipt)

    def test_same_frame_binding_rejects_image_depth_pose_and_evidence_changes(self):
        receipt = self.finish(self.request())
        apply_review(self.target, receipt, self.h, self.state, self.e, self.raw)
        changed_raw = copy.deepcopy(self.raw); changed_raw["head"][0,0,0] = 1
        cases = [(self.target, self.e, changed_raw),
                 ({**self.target,"point_base_m":[.001,0,.8]}, self.e, self.raw),
                 (self.target, replace(self.e,target_uv=(.51,.5)), self.raw)]
        for target, evidence_, raw in cases:
            with self.assertRaises(ValueError):
                apply_review(target, receipt, self.h, self.state, evidence_, raw)
        self.state.q[0] += .001
        with self.assertRaises(ValueError):apply_review(self.target, receipt, self.h, self.state, self.e, self.raw)

    def test_missing_malformed_and_disabled_receipts_fail_closed(self):
        receipt = self.finish(self.request())
        for bad in (None, {}, {**receipt,"required":1}, {**receipt,"success_or_grasp_claim":True}):
            with self.assertRaises(ValueError):apply_review(self.target, bad, self.h, self.state, self.e, self.raw)
        with self.assertRaises(ValueError):
            self.controller().observe(self.e, self.state, self.depths, {}, images=self.raw)
        with self.assertRaises(ValueError):
            GroundedController(self.model, SafeServo(self.model,self.state,[1,1]),self.h,near_contact_review=True)

    def test_runner_mode_identity_and_same_frame_wiring(self):
        source = (Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertTrue(any(isinstance(n.func,ast.Attribute) and n.func.attr=="add_argument" and
            n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=="--near-contact-review" for n in calls))
        self.assertTrue(any(isinstance(n.func,ast.Name) and n.func.id=="GroundedController" and
            any(k.arg=="near_contact_review" for k in n.keywords) for n in calls))
        self.assertTrue(any(isinstance(n.func,ast.Attribute) and n.func.attr=="observe" and
            any(k.arg=="contact_review_receipt" for k in n.keywords) for n in calls))
        self.assertIn('g.get("near_contact_review",False)==args.near_contact_review', source)
        self.assertEqual(source.count('"near_contact_review":args.near_contact_review'), 3)


if __name__ == "__main__":
    unittest.main()
