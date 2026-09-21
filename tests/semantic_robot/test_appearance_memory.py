import ast
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from semantic_robot.v2.appearance_memory import AppearanceMemory, LABEL, SYSTEM
from semantic_robot.v2.contact_review import NearContactReview, apply_review
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.grounding import GroundedEvidence, localize_target
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.policy import GroundedPolicy, RefinedGroundedPolicy
from semantic_robot.v2.vision import VisualBundle
from test_v2 import fixture, evidence


class AppearanceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.model, self.state = fixture()
        self.model.links["right"][0][:3, 3] = [.1, 0, .8]
        self.h = GroundedHarness([Goal("pick", "object", "right", "observed lift")])
        self.h.stage = "APPROACH"
        self.h.motion_receipt = {"valid": True, "body_transform_current_in_previous": np.eye(4).tolist()}
        self.e = GroundedEvidence(**asdict(evidence(view="head", enclosed=None, co_moving=None, supported=None, effect=None)))
        self.depths = {v: np.full((100,100), 1.2, np.float32) for v in ("head", "left_wrist", "right_wrist")}
        self.raw = {v: np.zeros((100,100,3), np.uint8) for v in self.depths}
        self.raw["head"][40:60,40:60,0] = 255
        self.bundle = VisualBundle([], [], {"not_observer_input": True}, self.raw)
        self.p = RefinedGroundedPolicy.__new__(RefinedGroundedPolicy)
        self.p.max_refinements, self.p.refinements = 16, 0
        self.p.contact_review = NearContactReview()
        self.p.appearance_memory = AppearanceMemory()
        self.p._call = Mock(side_effect=self.call)

    def call(self, kind, system, text, bundle, *args):
        value = {"candidate_id": 0} if kind == "ground" else asdict(self.e)
        return {"text": json.dumps(value)}, {"system": system, "text": text}

    def refine(self):
        return self.p.refine(self.e,self.h,self.state,self.bundle,self.depths,self.model)

    def seed(self):
        self.assertIsNone(self.p.appearance_memory.begin(self.h,self.state,self.bundle))
        obs, call, receipt, views = self.refine()
        self.assertTrue(receipt["appearance_memory"]["updated"])
        return obs, receipt, views

    def test_seed_only_from_real_semantic_selection_exact_raw_crop(self):
        _, receipt, views = self.seed()
        entry = self.p.appearance_memory.entry
        index = views.labels.index("CURRENT_HEAD_CROP_RAW")
        self.assertEqual(entry["image"].tobytes(), views.images[index].tobytes())
        self.assertEqual(entry["pixels_sha256"],hashlib.sha256(views.images[index].tobytes()).hexdigest())
        self.assertNotIn("point",entry);self.assertNotIn("target_uv",entry)
        self.assertFalse(receipt["appearance_memory"]["current_detection_or_grasp_claim"])

    def test_observation_uses_current3_raw_and_separate_reference_no_coordinates(self):
        self.seed(); old = self.p.appearance_memory.entry["image"].tobytes()
        self.raw["head"][:] = 17
        obs, call = self.p.observe(self.h,self.state,self.bundle)
        args = self.p._call.call_args.args
        self.assertEqual(args[0],"observe");self.assertEqual(args[1],SYSTEM)
        self.assertEqual(args[3].labels,["CURRENT_HEAD_RAW","CURRENT_LEFT_WRIST_RAW","CURRENT_RIGHT_WRIST_RAW",LABEL])
        self.assertEqual(args[3].images[0].tobytes(),self.raw["head"].tobytes())
        self.assertEqual(args[3].images[-1].tobytes(),old)
        self.assertEqual(call["appearance_reference_image"].tobytes(),old)
        context=json.loads(args[2]);self.assertEqual(set(context),{"current_goal","images"})
        self.assertNotIn("point",args[2]);self.assertNotIn("target_uv",args[2])
        self.assertEqual(call["validation"]["age_observations"],1)
        self.assertFalse(any(self.h.hold_verified.values()))

    def test_reference_forces_fresh_semantic_query_despite_valid_stable_depth(self):
        self.seed();self.p.observe(self.h,self.state,self.bundle)
        obs, call, receipt, _ = self.refine()
        self.assertEqual(self.p.refinements,2)
        self.assertEqual(receipt["near_contact_review"]["reason"],"APPEARANCE_REQUIRES_CURRENT_SURFACE")
        self.assertTrue(receipt["near_contact_review"]["confirmed_current_contact"])
        self.assertTrue(receipt["appearance_memory"]["updated"])
        self.assertIsNotNone(call)

    def test_abstention_clears_reference_and_vetoes_current_ray(self):
        self.seed();self.p.observe(self.h,self.state,self.bundle)
        self.p._call=Mock(return_value=({"text":'{"candidate_id":null}'},{}))
        obs, _, receipt, _ = self.refine()
        target=localize_target(obs,self.depths,self.model,self.state.q)
        self.assertFalse(apply_review(target,receipt["near_contact_review"],self.h,self.state,obs,self.raw)["valid"])
        self.assertIsNone(self.p.appearance_memory.entry)
        self.assertEqual(self.p.refinements,2)

    def test_exhaustion_cannot_reuse_reference_or_add_query(self):
        self.seed();self.p.observe(self.h,self.state,self.bundle)
        self.p.refinements=16;self.p._call.reset_mock()
        _,call,receipt,_=self.refine()
        self.assertIsNone(call);self.p._call.assert_not_called()
        self.assertEqual(self.p.refinements,16)
        self.assertEqual(receipt["reason"],"REFINEMENT_BUDGET_EXHAUSTED")
        self.assertIsNone(self.p.appearance_memory.entry)

    def test_absence_clears_cache_without_grasp_claim(self):
        self.seed()
        self.e=replace(self.e,visible=False,view="none",target_uv=None)
        self.p.observe(self.h,self.state,self.bundle)
        _, _, receipt, _=self.refine()
        self.assertIsNone(self.p.appearance_memory.entry)
        self.assertFalse(any(self.h.hold_verified.values()))
        self.assertEqual(receipt["appearance_memory"]["reason"],"INELIGIBLE_OR_NOT_VISIBLE")

    def test_expiry_is_exactly_eight_observations_no_automatic_refresh(self):
        self.seed();memory=self.p.appearance_memory
        for age in range(1,9):
            prepared=memory.begin(self.h,self.state,self.bundle)
            self.assertEqual(prepared[2]["age_observations"],age)
        self.assertIsNone(memory.begin(self.h,self.state,self.bundle))
        self.assertIsNone(memory.entry)

    def test_same_index_changed_target_or_instruction_invalidates_identity(self):
        for field,value in (("target","different"),("done_when","different criterion"),("hand","left"),("level",True)):
            self.setUp();self.seed();self.h.goals[0]=replace(self.h.goal,**{field:value})
            self.assertIsNone(self.p.appearance_memory.begin(self.h,self.state,self.bundle))
            self.assertIsNone(self.p.appearance_memory.entry)

    def test_loading_and_verification_or_bimanual_stages_do_not_use_memory(self):
        for field in ("pending_grasp","hold_verified","possible_contact_after_close"):
            self.setUp();self.seed();getattr(self.h,field)["left"]=True
            self.assertIsNone(self.p.appearance_memory.begin(self.h,self.state,self.bundle))
        for stage in ("GRASP","VERIFY_GRASP","CARRY","RELEASE","VERIFY_EFFECT"):
            self.setUp();self.seed();self.h.stage=stage
            self.assertIsNone(self.p.appearance_memory.begin(self.h,self.state,self.bundle))
        self.setUp();self.seed();self.h.goals[0]=replace(self.h.goal,hand="both")
        self.assertIsNone(self.p.appearance_memory.begin(self.h,self.state,self.bundle))

    def test_wrong_current_state_or_image_cannot_reuse_pending_question(self):
        for change in ("q","grip","image","stage","goal"):
            self.setUp();self.seed();self.p.observe(self.h,self.state,self.bundle)
            if change=="q":self.state.q[0]+=.001
            elif change=="grip":self.state.gripper[0]+=.001
            elif change=="image":self.raw["head"][0,0,0]=1
            elif change=="stage":self.h.stage="ALIGN"
            else:self.h.goals[0]=replace(self.h.goal,target="different")
            with self.assertRaisesRegex(ValueError,"snapshot mismatch"):self.refine()

    def test_unconfirmed_selection_wrong_candidate_or_crop_is_rejected(self):
        _,receipt,_=self.seed();memory=self.p.appearance_memory
        target=localize_target(self.e,self.depths,self.model,self.state.q)
        # Exact refined evidence, not the original detector UV, is required.
        refined=GroundedEvidence.parse(json.dumps(receipt["refined_evidence"]))
        target=localize_target(refined,self.depths,self.model,self.state.q)
        for change in ("candidate","box","image_size"):
            bad=copy.deepcopy(receipt)
            if change=="candidate":bad["choice"]["candidate_id"]=100
            elif change=="box":bad["proposals"]["crop_box_pixels"][0]=-1
            else:bad["proposals"]["image_size_pixels"]=[1,1]
            with self.assertRaises(ValueError):memory.update(self.h,self.state,refined,target,self.bundle,self.model,bad)

    def test_appearance_query_cannot_return_old_grasp_claims(self):
        for field,value in (("enclosed",True),("co_moving",False),("supported",False),("effect",True),("target_reference","world")):
            self.setUp();self.seed();bad=asdict(self.e);bad[field]=value
            self.p._call=Mock(return_value=({"text":json.dumps(bad)},{}))
            with self.assertRaises(ValueError):self.p.observe(self.h,self.state,self.bundle)

    def test_default_or_missing_cache_keeps_original_observer(self):
        for memory in (None,AppearanceMemory()):
            self.p.appearance_memory=memory
            with patch.object(GroundedPolicy,"observe",return_value=("old",{})) as old:
                self.assertEqual(self.p.observe(self.h,self.state,self.bundle),("old",{}))
                old.assert_called_once_with(self.h,self.state,self.bundle)

    def test_flag_dependency_and_exact_bool(self):
        def init(p,*args,**kwargs):p.identity={"finite_choice_kinds":["ground"]}
        with patch.object(GroundedPolicy,"__init__",init):
            for near,appearance in ((False,True),(True,1),(True,"yes")):
                with self.assertRaises(ValueError):RefinedGroundedPolicy(near_contact_review=near,appearance_memory=appearance)
            self.assertIsNone(RefinedGroundedPolicy().appearance_memory)
            self.assertIsInstance(RefinedGroundedPolicy(near_contact_review=True,appearance_memory=True).appearance_memory,AppearanceMemory)

    def test_runner_mode_identity_and_raw_reference_artifact_wiring(self):
        source=(Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text()
        ast.parse(source)
        self.assertIn('"--appearance-memory"',source)
        self.assertIn('g.get("appearance_memory",False)==args.appearance_memory',source)
        self.assertEqual(source.count('"appearance_memory":args.appearance_memory'),3)
        self.assertIn('call["appearance_reference_image"].save(directory/"REFERENCE_APPEARANCE_NOT_CURRENT.png")',source)


if __name__=="__main__":unittest.main()
