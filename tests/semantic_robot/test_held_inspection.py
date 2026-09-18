import json
import unittest
from dataclasses import replace
import numpy as np
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.grounded_harness import GroundedHarness,GroundedController
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.held_inspection import HeldInspection,inspection_palette
from semantic_robot.v2.policy import GROUNDED_OBSERVE_CORE,reference_observation_system
from semantic_robot.v2.protocol import Action,HOLD,ROTATIONS
from semantic_robot.v2.servo import SafeServo
from test_v2 import fixture,evidence


class HeldInspectionTests(unittest.TestCase):
    def make(self,reference="held_right",level=False):
        m,s=fixture();s.gripper[:]=.035
        h=GroundedHarness([Goal("press","button of held object","left","visible effect")],held_inspection=True)
        h.held["right"]="object";h.hold_verified["right"]=True;h.level["right"]=level
        obs=GroundedEvidence(**evidence(visible=False,view="none",target_uv=None).as_dict(),target_reference=reference)
        h.observe(obs,s)
        return m,s,h,obs

    def test_reference_is_not_active_hand_or_camera(self):
        _,_,h,_=self.make();palette=h.palette()
        self.assertEqual(h.goal.hand,"left")
        self.assertTrue(all(a==HOLD or a.part=="right" for a in palette))
        self.assertFalse(any(a.part in ("base","torso") or a.move in ("open","close") for a in palette))

    def test_level_constraint_preserved(self):
        _,_,h,_=self.make(level=True)
        self.assertFalse(any(a.move in ROTATIONS for a in h.palette()))

    def test_world_goal_with_other_held_object_still_uses_world_search(self):
        _,_,h,_=self.make(reference="world")
        self.assertTrue(any(a.part=="base" for a in h.palette()))

    def test_missing_reference_does_not_silently_scan_with_load(self):
        _,_,h,_=self.make(reference="unknown")
        self.assertEqual(h.palette(),(HOLD,))

    def test_contradiction_and_unverified_reference_stop_without_open(self):
        _,s,h,obs=self.make();h.observe(replace(obs,target_reference="world"),s)
        self.assertEqual(h.stop_reason,"TARGET_REFERENCE_CONTRADICTION")
        _,s,h,obs=self.make(reference="unknown");h.hold_verified["right"]=False
        h.observe(obs.__class__(**{**obs.as_dict(),"target_reference":"held_right"}),s)
        self.assertEqual(h.stop_reason,"TARGET_REFERENCE_REQUIRES_VERIFIED_HOLD")

    def test_old_schema_defaults_unknown_and_new_enum_checked(self):
        raw=evidence().as_dict();parsed=GroundedEvidence.parse(json.dumps(raw))
        self.assertEqual(parsed.target_reference,"unknown")
        raw.update(other_views=[],target_reference="sim_object_pose")
        with self.assertRaises(ValueError):GroundedEvidence.parse(json.dumps(raw))

    def test_prompt_example_is_parseable_and_separates_reference_from_visibility(self):
        prompt=reference_observation_system(GROUNDED_OBSERVE_CORE)
        example=prompt.split("Return only JSON with exactly these fields:\n")[1].split("\n")[0]
        self.assertEqual(GroundedEvidence.parse(example).target_reference,"unknown")
        self.assertIn("even if the LEFT",prompt)

    def test_inspection_requires_prior_verified_anchor_and_actual_relative_change(self):
        m,s,h,_=self.make();inspector=HeldInspection()
        self.assertFalse(inspector.observe(m,s,h)[1]["valid"])
        inspector.remember(m,s,"right",{"valid":True,"point_base_m":m.forward(s.q,"right")[:3,3].tolist()})
        self.assertFalse(inspector.observe(m,s,h)[0])
        self.assertFalse(inspector.observe(m,s,h)[0])
        q=s.q.copy();q[13]+=.012;new=m.state(q,s.gripper,np.zeros(3))
        self.assertTrue(inspector.observe(m,new,h)[0])
        self.assertGreater(inspector.context(m,new,h)["relative_path_m"],.008)

    def test_world_search_cannot_be_called_for_relative_target(self):
        m,s,h,_=self.make();ctl=GroundedController(m,SafeServo(m,s,[-1,-1]),h)
        self.assertTrue(ctl.is_held_search)
        with self.assertRaises(ValueError):ctl.search_action(s)

    def test_attempt_budget_cannot_be_reset_by_same_goal_observation(self):
        m,s,h,_=self.make();inspector=HeldInspection()
        inspector.remember(m,s,"right",{"valid":True,"point_base_m":m.forward(s.q,"right")[:3,3].tolist()})
        inspector.observe(m,s,h)
        for _ in range(24):inspector.executed(h);inspector.observe(m,s,h)
        class Guard:
            def receipt(self):return {}
        allowed,_=inspector.candidates(m,s,h,SafeServo(m,s,[-1,-1]),Guard())
        self.assertEqual(allowed,(HOLD,));self.assertEqual(h.stop_reason,"HELD_INSPECTION_BUDGET_REACHED")

