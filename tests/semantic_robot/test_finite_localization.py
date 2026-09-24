import copy
from dataclasses import asdict, replace
import json
import time
import unittest

import numpy as np

from semantic_robot.v2.affordance import SurfaceChoice
from semantic_robot.v2.finite_localization import (SelectionReply, bind_frame, locate_target,
    region_boxes, region_request, region_surfaces, surface_request)
from semantic_robot.v2.grounding import localize_target
from semantic_robot.v2.harness import Goal
from test_v2 import fixture


class FiniteLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.model, self.state = fixture()
        self.raw = {view: np.zeros((100, 100, 3), np.uint8) for view in self.model.spec["metadata"]["cameras"]}
        self.depths = {view: np.ones((100, 100), np.float32) for view in self.raw}
        self.goal = Goal("pick", "named object", "right", "independent feedback required")
        self.requests = []

    def binding(self):
        return bind_frame("capture-17", self.goal, self.raw, self.depths, self.model, self.state)

    def run_locator(self, choices, **changes):
        iterator = iter(choices)
        def choose(request):
            self.requests.append(request)
            return SelectionReply(request.sha256, SurfaceChoice(next(iterator)).text())
        arguments = dict(frame_id="capture-17", goal=self.goal, raw=self.raw, depths=self.depths,
                         model=self.model, state=self.state, views=("head",), choose=choose,
                         deadline=time.monotonic() + 10)
        arguments.update(changes)
        return locate_target(**arguments)

    def test_cells_cover_odd_rectangular_frame_without_overlap(self):
        counts = np.zeros((77, 103), np.uint8)
        for x0, y0, x1, y1 in region_boxes(103, 77):
            counts[y0:y1, x0:x1] += 1
        np.testing.assert_array_equal(counts, 1)
        for dimensions in ((True, 100), (100., 100), (0, 100), (23, 100), (100, 1025)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                region_boxes(*dimensions)

    def test_surface_samples_cover_region_not_just_seed_centre(self):
        box = region_boxes(100, 100)[7]
        rows = region_surfaces("head", box, self.depths, self.model, self.state)
        self.assertEqual([c["id"] for c in rows], list(range(12)))
        self.assertEqual(len({c["pixel_xy"][0] for c in rows}), 4)
        self.assertEqual(len({c["pixel_xy"][1] for c in rows}), 3)
        for c in rows:
            x, y = c["pixel_xy"]
            self.assertTrue(box[0] <= x < box[2] and box[1] <= y < box[3])
            np.testing.assert_allclose(c["target_uv"], [x / 99, y / 99])
        with self.assertRaises(ValueError):
            region_surfaces("head", (1, 1, 20, 20), self.depths, self.model, self.state)

    def test_successful_choices_need_both_questions_and_no_temporal_claims(self):
        result = self.run_locator([7, 5])
        self.assertEqual([r.phase for r in self.requests], ["region", "surface"])
        self.assertTrue(result.evidence.visible)
        self.assertEqual(result.evidence.view, "head")
        for field in ("enclosed", "co_moving", "supported", "effect"):
            self.assertIsNone(getattr(result.evidence, field))
        self.assertEqual(result.evidence.other_views, ())
        self.assertEqual(result.evidence.target_reference, "unknown")
        self.assertTrue(localize_target(result.evidence, self.depths, self.model, self.state.q)["valid"])
        self.assertIs(result.for_frame("capture-17", self.goal, self.raw, self.depths, self.model, self.state), result.evidence)
        json.dumps(result.receipt, allow_nan=False)
        self.assertEqual(result.receipt["robot_controls"], 0)

    def test_depth_valid_does_not_skip_semantic_choice(self):
        result = self.run_locator([7, None])
        self.assertEqual(len(self.requests), 2)
        self.assertFalse(result.evidence.visible)
        self.assertIsNone(result.evidence.target_uv)
        self.assertEqual(result.receipt["attempts"][0]["reason"], "SURFACE_ABSTAINED")

    def test_region_abstention_never_returns_cell_centre(self):
        result = self.run_locator([None])
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(result.evidence.visible)

    def test_invalid_depth_never_autofills_or_issues_surface_call(self):
        self.depths["head"][:] = np.nan
        result = self.run_locator([7])
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(result.evidence.visible)
        self.assertEqual(result.receipt["attempts"][0]["reason"], "NO_VALID_SURFACE_SAMPLES")

    def test_sparse_candidates_keep_ids_and_reject_unoffered_id(self):
        self.depths["head"][:] = np.nan
        box = region_boxes(100, 100)[4]
        x = box[0] + int(.5 * (box[2] - box[0]) / 4)
        y = box[1] + int(.5 * (box[3] - box[1]) / 3)
        self.depths["head"][y-2:y+3, x-2:x+3] = 1
        with self.assertRaises(ValueError): self.run_locator([4, 11])
        self.assertEqual(self.requests[-1].allowed, (SurfaceChoice(), SurfaceChoice(0)))

    def test_explicit_view_order_bounded_and_stops_after_selection(self):
        result = self.run_locator([None, 4, 5], views=("head", "right_wrist", "left_wrist"))
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(result.evidence.view, "right_wrist")
        self.requests.clear()
        result = self.run_locator([4, None, 4, None, 4, None], views=tuple(self.raw))
        self.assertEqual(len(self.requests), 6)
        self.assertFalse(result.evidence.visible)
        for views in ((), ("head", "head"), ("world",), ["head"]):
            with self.subTest(views=views), self.assertRaises(ValueError):
                self.run_locator([], views=views)

    def test_images_do_not_mutate_native_pixels_and_guides_are_separate(self):
        original = copy.deepcopy(self.raw)
        result = self.run_locator([4, 5])
        self.assertTrue(result.evidence.visible)
        for view in original: np.testing.assert_array_equal(original[view], self.raw[view])
        first, second = self.requests
        self.assertEqual(len(first.images), 2)
        self.assertEqual(len(second.images), 4)
        self.assertEqual(np.asarray(first.images[0]).sum(), 0)
        self.assertGreater(np.asarray(first.images[1]).sum(), 0)
        self.assertEqual(np.asarray(second.images[2]).sum(), 0)
        self.assertGreater(np.asarray(second.images[3]).sum(), 0)
        self.assertNotIn("point_base_m", second.text)
        self.assertNotIn("done_when", second.text)

    def test_binding_rejects_other_capture_goal_camera_depth_q_or_grip(self):
        result = self.run_locator([4, 5])
        with self.assertRaises(ValueError):
            result.for_frame("capture-18", self.goal, self.raw, self.depths, self.model, self.state)
        with self.assertRaises(ValueError):
            result.for_frame("capture-17", replace(self.goal, target="other"), self.raw, self.depths, self.model, self.state)
        for changes in ("rgb", "depth", "q", "grip"):
            raw, depths, state = copy.deepcopy((self.raw, self.depths, self.state))
            if changes == "rgb": raw["left_wrist"][0, 0, 0] = 1
            elif changes == "depth": depths["head"][0, 0] += .1
            elif changes == "q": state.q[0] += .01
            else: state.gripper[0] += .01
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                result.for_frame("capture-17", self.goal, raw, depths, self.model, state)
        self.model.spec["metadata"]["cameras"]["head"]["K"][0][0] += 1
        with self.assertRaises(ValueError): self.binding()

    def test_callback_cannot_change_capture_or_return_stale_reply(self):
        for mode in ("wrong_reply", "raw", "request_image"):
            def choose(request):
                digest = request.sha256
                if mode == "raw": self.raw["head"][0, 0] = 1
                if mode == "request_image": request.images[0].putpixel((0, 0), (1, 2, 3))
                return SelectionReply("wrong" if mode == "wrong_reply" else digest, SurfaceChoice(0).text())
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_locator([], choose=choose)
            self.raw["head"][:] = 0

    def test_bad_and_noncanonical_choices_are_not_repaired(self):
        for text in ('{"candidate_id":9}', '{"candidate_id":true}', '{"candidate_id":0,"effect":true}',
                     '{"candidate_id": 0}', '{"candidate_id":0,"candidate_id":1}'):
            def choose(request): return SelectionReply(request.sha256, text)
            with self.subTest(text=text), self.assertRaises(ValueError): self.run_locator([], choose=choose)

    def test_live_fk_cannot_change_under_unchanged_spec_hash(self):
        for mutation in ("reference", "camera_pose", "camera_screw", "limits", "link_set"):
            self.model, self.state = fixture()
            def choose(request):
                if mutation == "reference": self.model.reference[0] += .125
                elif mutation == "camera_pose": self.model.links["camera_head"][0][0, 3] += .125
                elif mutation == "camera_screw": self.model.links["camera_head"][1][0, 0] += .125
                elif mutation == "limits": self.model.lower[0] += .125
                else: self.model.links.pop("camera_left_wrist")
                return SelectionReply(request.sha256, SurfaceChoice(0).text())
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_locator([], choose=choose)
            with self.assertRaises(ValueError): self.binding()

    def test_deadline_stops_before_call_and_after_slow_callback(self):
        with self.assertRaises(TimeoutError): self.run_locator([], deadline=time.monotonic() - 1)
        self.assertEqual(self.requests, [])
        from unittest.mock import patch
        with patch("semantic_robot.v2.finite_localization.time.monotonic", side_effect=[0, 0, 0, 2]):
            with self.assertRaises(TimeoutError): self.run_locator([0], deadline=1)
        self.assertEqual(len(self.requests), 1)

    def test_malformed_capture_rejected_before_any_model_request(self):
        for changes in ("missing_depth", "shape", "dtype", "q_nan", "extra_view"):
            raw, depths, state = copy.deepcopy((self.raw, self.depths, self.state))
            if changes == "missing_depth": depths.pop("head")
            elif changes == "shape": raw["head"] = raw["head"][:50]
            elif changes == "dtype": raw["head"] = raw["head"].astype(np.float32)
            elif changes == "q_nan": state.q[0] = np.nan
            else: raw["world"] = raw["head"]; depths["world"] = depths["head"]
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_locator([], raw=raw, depths=depths, state=state)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
