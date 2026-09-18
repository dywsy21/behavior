import json
import unittest
from unittest.mock import Mock

from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.target_reference import ReferenceChoice,REFERENCES,reference_context
from test_v2 import fixture,evidence
from test_v2_transport import service


class TargetReferenceTests(unittest.TestCase):
    def setup_case(self,choice="held_right"):
        h=GroundedHarness([Goal("press","button on held radio","left","visible effect"),
                           Goal("place","table","right","supported")],held_inspection=True,reference_from_planner=True)
        h.held["right"]="radio";h.hold_verified["right"]=True
        p=object.__new__(GroundedPolicy);p.identity={"finite_choice_kinds":["reference"]}
        p._call=Mock(return_value=({"text":ReferenceChoice(choice).text()},{}))
        return h,p

    def test_text_only_relationship_not_working_hand_or_visibility(self):
        h,p=self.setup_case();p.resolve_target_reference(h)
        args=p._call.call_args.args
        self.assertEqual(args[0],"reference");self.assertEqual(args[3].images,[])
        context=json.loads(args[2].split("\nChoices:")[0])
        self.assertEqual(set(context),{"current_goal","prior_verified_holding_claims_not_current_attachment_truth"})
        self.assertEqual(h.search_reference,"held_right");self.assertEqual(h.goal.hand,"left")
        _,s=fixture();h.observe(GroundedEvidence(**evidence(visible=False,view="none",target_uv=None).as_dict()),s)
        self.assertEqual(h.search_reference,"held_right");self.assertIsNone(h.stop_reason)

    def test_one_binding_per_goal_not_per_frame(self):
        h,p=self.setup_case();p.resolve_target_reference(h);p.resolve_target_reference(h)
        self.assertEqual(p._call.call_count,1)
        h.index=1;p._call.return_value=({"text":ReferenceChoice("world").text()},{})
        p.resolve_target_reference(h)
        self.assertEqual(p._call.call_count,2);self.assertEqual(h.search_reference,"world")

    def test_no_held_reference_uses_no_model_call(self):
        h,p=self.setup_case();h.hold_verified["right"]=False
        p.resolve_target_reference(h);p._call.assert_not_called()
        self.assertEqual(h.search_reference,"world")
        self.assertEqual(reference_context(h)["prior_verified_holding_claims_not_current_attachment_truth"],{})

    def test_unknown_stops_instead_of_world_fallback(self):
        h,p=self.setup_case("unknown");p.resolve_target_reference(h)
        self.assertEqual(h.stop_reason,"TARGET_REFERENCE_UNRESOLVED_WITH_HELD_LOAD")
        self.assertEqual(h.search_reference,"unknown")

    def test_reference_to_unverified_hand_is_rejected(self):
        h,p=self.setup_case("held_left");p.resolve_target_reference(h)
        self.assertEqual(h.stop_reason,"TARGET_REFERENCE_REQUIRES_VERIFIED_HOLD")
        self.assertNotIn(h.index,h.target_references)

    def test_lost_verification_invalidates_cached_reference(self):
        h,p=self.setup_case();p.resolve_target_reference(h);h.hold_verified["right"]=False
        p.resolve_target_reference(h)
        self.assertEqual(h.stop_reason,"TARGET_REFERENCE_REQUIRES_VERIFIED_HOLD")
        self.assertEqual(p._call.call_count,1)

    def test_missing_service_support_and_call_cap(self):
        h,p=self.setup_case();p.identity={}
        with self.assertRaises(ValueError):p.resolve_target_reference(h)
        h,p=self.setup_case();p.reference_calls=4;p.resolve_target_reference(h)
        p._call.assert_not_called();self.assertEqual(h.stop_reason,"SEMANTIC_REFERENCE_CALL_BUDGET_REACHED")

    def test_complete_finite_grammar_and_strict_enum(self):
        choices=[ReferenceChoice(r).text() for r in REFERENCES]
        service.validate_scoped_choices("reference",choices)
        for bad in (choices[:-1],choices[1:],choices+choices[:1]):
            with self.assertRaises(ValueError):service.validate_scoped_choices("reference",bad)
        for raw in ('{"target_reference":"button"}','{"target_reference":"world","pose":0}','{}'):
            with self.assertRaises(ValueError):ReferenceChoice.parse(raw)

    def test_zero_images_only_for_semantic_reference(self):
        service.validate_image_count("reference",[])
        service.validate_image_count("observe",[{}])
        with self.assertRaises(ValueError):service.validate_image_count("reference",[{}])
        with self.assertRaises(ValueError):service.validate_image_count("observe",[])
