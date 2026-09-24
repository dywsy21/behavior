import copy
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.finite_localization import SelectionReply
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.native_grounding import (NativeDetection, locate_native, native_pixels,
                                                native_request, parse_native)
from test_v2 import fixture


class NativeGroundingTests(unittest.TestCase):
    def setUp(self):
        self.model, self.state = fixture()
        self.raw = {v: np.zeros((100, 100, 3), np.uint8) for v in self.model.spec["metadata"]["cameras"]}
        self.depths = {v: np.ones((100, 100), np.float32) for v in self.raw}
        self.goal = Goal("pick", "named object", "right", "independent feedback required")
        self.requests = []

    def arguments(self):
        return dict(frame_id="native-17", goal=self.goal, raw=self.raw, depths=self.depths,
                    model=self.model, state=self.state)

    def run_locator(self, text='[{"point_2d":[500,500],"label":"object"}]', mode="point", **changes):
        def choose(request):
            self.requests.append(request)
            return SelectionReply(request.sha256, text)
        args = dict(**self.arguments(), view="head", mode=mode, choose=choose, deadline=time.monotonic()+10)
        args.update(changes)
        return locate_native(**args)

    def test_original_single_raw_and_explicit_native_protocol(self):
        result = self.run_locator()
        request, = self.requests
        self.assertEqual(request.phase, "native_point")
        self.assertEqual(request.allowed, ())
        self.assertEqual(request.labels, ("CURRENT_HEAD_RAW",))
        np.testing.assert_array_equal(request.images[0], self.raw["head"])
        self.assertIn("0 to 1000", request.text)
        self.assertIn("Return []", request.text)
        self.assertIn("point_2d", request.text)
        self.assertNotIn("done_when", request.text)
        self.assertNotIn("native-17", request.text)
        self.assertTrue(result.point_evidence().visible)
        np.testing.assert_allclose(result.point_evidence().target_uv, [50/99, 50/99])
        self.assertEqual(result.receipt["native_pixels"], [50,50])
        for key in ("enclosed", "co_moving", "supported", "effect"):
            self.assertIsNone(getattr(result.point_evidence(), key))
        self.assertEqual(result.receipt["robot_controls"], 0)

    def test_non_square_dimensions_follow_official_width_height_not_minus_one(self):
        point = NativeDetection("object", (250,750))
        self.assertEqual(native_pixels(point, "point", 192,96), (48,72))
        box = NativeDetection("object", (250,250,750,1000))
        self.assertEqual(native_pixels(box, "box", 192,96), (48,24,144,96))
        # Pixel edges round outwards for crops, not to a fabricated contact.
        self.assertEqual(native_pixels(NativeDetection("x", (1,1,2,2)), "box", 101,77), (0,0,1,1))

    def test_non_square_capture_point_roundtrip(self):
        spec = copy.deepcopy(self.model.spec)
        for view in self.raw:
            camera = spec["metadata"]["cameras"][view]
            camera.update(width=192, height=96, K=[[70,0,96],[0,70,48],[0,0,1]])
            self.raw[view] = np.zeros((96,192,3), np.uint8)
            self.depths[view] = np.ones((96,192), np.float32)
        self.model = RobotModel(spec)
        result = self.run_locator('[{"point_2d":[250,750],"label":"object"}]')
        self.assertEqual(result.receipt["native_pixels"], [48,72])
        np.testing.assert_allclose(result.point_evidence().target_uv, [48/191,72/95])

    def test_point_edge_1000_is_not_silently_clipped(self):
        for xy in ((1000,500),(500,1000),(1000,1000)):
            with self.subTest(xy=xy):
                result = self.run_locator(json.dumps([{"point_2d":xy,"label":"x"}]))
                self.assertFalse(result.point_evidence().visible)
                self.assertEqual(result.receipt["reason"], "POINT_OUTSIDE_SENSOR_PIXELS")
        self.assertEqual(native_pixels(NativeDetection("x", (999,999)), "point", 720,720), (719,719))
        self.assertEqual(native_pixels(NativeDetection("x", (0,0)), "point", 720,720), (0,0))

    def test_box_is_region_only_and_never_becomes_a_centre_point(self):
        result = self.run_locator('[{"bbox_2d":[100,200,600,1000],"label":"handle"}]', mode="box")
        self.assertEqual(result.bbox_pixels, (10,20,60,100))
        self.assertIsNone(result.evidence)
        self.assertNotIn("geometry", result.receipt)
        self.assertEqual(result.receipt["reason"], "MODEL_REGION_ONLY_NOT_A_CONTACT")
        with self.assertRaises(ValueError): result.point_evidence()

    def test_empty_and_multiple_detections_do_not_choose_first(self):
        for mode,key,coordinates in (("point","point_2d",[500,500]),("box","bbox_2d",[0,0,1000,1000])):
            for count in (0,2):
                text = json.dumps([{key:coordinates,"label":"object"}]*count)
                result = self.run_locator(text, mode=mode)
                self.assertEqual(result.receipt["reason"], "MODEL_ABSTAINED" if count==0 else "AMBIGUOUS_MULTIPLE_TARGETS")
                self.assertIsNone(result.bbox_pixels)
                if mode=="point": self.assertFalse(result.point_evidence().visible)
                else: self.assertIsNone(result.evidence)

    def test_strict_fields_duplicates_numbers_and_formats(self):
        bad = ('null', '{}', '[{"point_2d":[.5,.5],"label":"x"}]',
               '[{"point_2d":[true,500],"label":"x"}]', '[{"point_2d":[500.0,500],"label":"x"}]',
               '[{"point_2d":[-1,500],"label":"x"}]', '[{"point_2d":[1001,500],"label":"x"}]',
               '[{"point_2d":[NaN,500],"label":"x"}]', '[{"point_2d":[500,500,500],"label":"x"}]',
               '[{"point_2d":[500,500],"label":"x","effect":true}]',
               '[{"point_2d":[500,500],"label":"x","label":"other"}]',
               '[{"point_2d":[500,500],"label":""}]')
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError): parse_native(text,"point")
        with self.assertRaises(ValueError): parse_native(' '*8193,"point")
        with self.assertRaises(ValueError): parse_native(json.dumps([{"point_2d":[1,1],"label":"x"}]*9),"point")

    def test_reversed_zero_and_mixed_mode_boxes_are_not_repaired(self):
        for coords in ([500,0,400,500],[0,500,500,400],[1,1,1,2],[1,1,2,1]):
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                parse_native(json.dumps([{"bbox_2d":coords,"label":"x"}]),"box")
        with self.assertRaises(ValueError): parse_native('[{"point_2d":[500,500],"label":"x"}]',"box")
        with self.assertRaises(ValueError): native_pixels(NativeDetection("x", (1,2,3)),"point",100,100)

    def test_one_complete_json_fence_only_and_raw_text_retained(self):
        text='```json\n[{"point_2d":[500,500],"label":"x"}]\n```'
        result=self.run_locator(text)
        self.assertEqual(result.receipt["raw_text"],text)
        self.assertEqual(result.receipt["transport_normalization"],"single_markdown_json_fence")
        for bad in ("Answer:\n"+text,text+"\nSuccess",text+"\n"+text,text[:-3],"```python\n[]\n```"):
            with self.subTest(bad=bad),self.assertRaises(ValueError):parse_native(bad,"point")

    def test_point_with_no_depth_remains_unknown(self):
        self.depths["head"][:]=np.nan
        result=self.run_locator()
        self.assertFalse(result.point_evidence().visible)
        self.assertEqual(result.receipt["reason"],"POINT_DEPTH_REJECTED")
        self.assertEqual(len(self.requests),1)

    def test_stale_goal_and_same_pixels_different_clock_rejected(self):
        result=self.run_locator()
        self.assertIs(result.for_frame(**self.arguments()),result)
        for updates in ({"frame_id":"native-18"},{"goal":replace(self.goal,target="another object")}):
            with self.subTest(updates=updates),self.assertRaises(ValueError):
                result.for_frame(**(self.arguments()|updates))
        self.raw["right_wrist"][0,0]=1
        with self.assertRaises(ValueError):result.for_frame(**self.arguments())

    def test_callback_cannot_mutate_request_or_public_geometry(self):
        for mutation in ("wrong_reply","request_image","q","depth","live_fk"):
            self.setUp()
            def choose(request):
                digest=request.sha256
                if mutation=="request_image":request.images[0].putpixel((0,0),(1,2,3))
                if mutation=="q":self.state.q[0]+=.1
                if mutation=="depth":self.depths["head"][0,0]+=.1
                if mutation=="live_fk":self.model.links["camera_head"][0][0,3]+=.1
                return SelectionReply("wrong" if mutation=="wrong_reply" else digest,'[]')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.run_locator(choose=choose)

    def test_deadline_before_and_after_single_call(self):
        with self.assertRaises(TimeoutError):self.run_locator(deadline=time.monotonic()-1)
        self.assertFalse(self.requests)
        with patch("semantic_robot.v2.native_grounding.time.monotonic",side_effect=[0,0,2]):
            with self.assertRaises(TimeoutError):self.run_locator(deadline=1)
        self.assertEqual(len(self.requests),1)

    def test_unknown_mode_view_or_malformed_raw_does_not_call_model(self):
        for changes in ({"mode":"pose"},{"view":"world"},{"deadline":float('nan')}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.run_locator(**changes)
        self.raw["head"]=self.raw["head"].astype(np.float32)
        with self.assertRaises(ValueError):self.run_locator()
        self.assertFalse(self.requests)


if __name__=="__main__":unittest.main()
