"""Surface proposals cannot invent target identity or bypass depth validity."""
import copy
from dataclasses import asdict, replace
import json
import unittest

import numpy as np

from semantic_robot.v2.affordance import (SurfaceChoice, surface_candidates,
                                        refinement_bundle, select_surface)
from semantic_robot.v2.grounding import GroundedEvidence, localize_target
from semantic_robot.v2.vision import VisualBundle
from test_v2 import fixture, evidence


class AffordanceTests(unittest.TestCase):
    def setUp(self):
        self.model,self.state=fixture()
        self.evidence=GroundedEvidence(**asdict(evidence(view="head")))
        self.depths={v:np.ones((100,100),np.float32) for v in self.model.spec["metadata"]["cameras"]}
        self.depths["head"][49:52,:]=2.
        self.raw={v:np.zeros((100,100,3),np.uint8) for v in self.depths}
        self.bundle=VisualBundle([],[],{},self.raw)

    def proposals(self):
        return surface_candidates(self.evidence,self.depths,self.model,self.state.q)

    def test_choice_strict_and_abstention(self):
        for value in (None,0,11):
            choice=SurfaceChoice(value)
            self.assertEqual(choice,SurfaceChoice.parse(choice.text()))
        for value in (True,False,-1,12,1.,"0",[]):
            with self.subTest(value=value),self.assertRaises(ValueError):SurfaceChoice(value)
        for text in ('{}','{"candidate_id":null,"target_uv":[0,0]}',
                     '{"candidate_id":0,"candidate_id":1}','```json\n{}\n```'):
            with self.assertRaises(ValueError):SurfaceChoice.parse(text)

    def test_original_invalid_proposals_stable_bounded_and_nonmutating(self):
        before=copy.deepcopy(self.depths);q=self.state.q.copy()
        self.assertFalse(localize_target(self.evidence,self.depths,self.model,q)["valid"])
        receipt=self.proposals()
        self.assertEqual(len(receipt["candidates"]),12)
        self.assertFalse(receipt["automatic_target_assignment"])
        self.assertFalse(receipt["scene_truth_used"])
        json.dumps(receipt,allow_nan=False)
        for c in receipt["candidates"]:
            p=replace(self.evidence,target_uv=tuple(c["target_uv"]))
            self.assertTrue(localize_target(p,self.depths,self.model,q)["valid"])
        for v in before:np.testing.assert_array_equal(before[v],self.depths[v])
        np.testing.assert_array_equal(q,self.state.q)

    def test_holes_and_invisibility_do_not_invent_surfaces(self):
        self.depths["head"][:]=np.nan
        self.assertEqual(self.proposals()["candidates"],[])
        invisible=GroundedEvidence(**asdict(evidence(visible=False,view="none",target_uv=None)))
        self.assertEqual(surface_candidates(invisible,{},self.model,self.state.q)["candidates"],[])

    def test_selection_scoped_and_drops_uncorroborated_motion(self):
        receipt=self.proposals()
        original=replace(self.evidence,co_moving=True,other_views=({"view":"left_wrist","target_uv":(.5,.5)},))
        self.assertIs(select_surface(original,SurfaceChoice(),receipt),original)
        selected=select_surface(original,SurfaceChoice(0),receipt)
        self.assertEqual(selected.target_uv,tuple(receipt["candidates"][0]["target_uv"]))
        self.assertIsNone(selected.co_moving);self.assertEqual(selected.other_views,())
        self.assertFalse(selected.enclosed)
        with self.assertRaises(ValueError):select_surface(original,SurfaceChoice(1),{**receipt,"candidates":receipt["candidates"][:1]})
        with self.assertRaises(ValueError):select_surface(original,SurfaceChoice(0),{**receipt,"view":"left_wrist"})

    def test_crop_raw_and_annotated_are_separate_and_resolution_checked(self):
        receipt=self.proposals();views=refinement_bundle(self.bundle,receipt)
        self.assertEqual(len(views.images),6)
        self.assertEqual(views.images[-2].size,(512,512))
        self.assertIn("NOT_OBJECT_DETECTIONS",views.labels[-1])
        self.assertIn("NOT_OBJECT_BOX",views.labels[-3])
        self.assertEqual(int(np.asarray(views.images[-2]).sum()),0)
        self.assertGreater(int(np.asarray(views.images[-1]).sum()),0)
        self.assertEqual(sum(int(x.sum()) for x in self.raw.values()),0)
        self.raw["head"]=np.zeros((32,32,3),np.uint8)
        with self.assertRaises(ValueError):refinement_bundle(self.bundle,receipt)


if __name__=="__main__":unittest.main()
