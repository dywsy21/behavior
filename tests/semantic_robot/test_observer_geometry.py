import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.v2.observer_geometry import camera_anchor_geometry, projected_anchor_depth, angle_degrees


class ObserverGeometryTests(unittest.TestCase):
    camera = {"K": [[70, 0, 50], [0, 70, 50], [0, 0, 1]], "width": 101, "height": 101}

    def geometry(self, camera=None, hand=None, anchor=(0., 0., -1.)):
        return camera_anchor_geometry(np.eye(4) if camera is None else camera,
                                      np.eye(4) if hand is None else hand, anchor, self.camera)

    def test_center_and_behind_camera_are_distinct(self):
        front = self.geometry()
        self.assertTrue(front["in_image_bounds"])
        self.assertEqual(front["bearing_error_deg"], 0.)
        np.testing.assert_allclose(front["prior_anchor_uv"], [.5, .5])
        back = self.geometry(anchor=(0., 0., 1.))
        self.assertIsNone(back["prior_anchor_uv"])
        self.assertFalse(back["in_image_bounds"])
        self.assertEqual(back["bearing_error_deg"], 180.)

    def test_camera_aiming_changes_framing_not_side(self):
        camera = np.eye(4); camera[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
        front, turned = self.geometry(), self.geometry(camera)
        self.assertAlmostEqual(turned["bearing_error_deg"], 30.)
        self.assertAlmostEqual(angle_degrees(front["camera_direction_in_reference_hand"], turned["camera_direction_in_reference_hand"]), 0.)

    def test_transverse_view_changes_side_even_before_reaiming(self):
        camera = np.eye(4); camera[0, 3] = np.tan(np.deg2rad(30))
        old, new = self.geometry(), self.geometry(camera)
        self.assertAlmostEqual(angle_degrees(old["camera_direction_in_reference_hand"], new["camera_direction_in_reference_hand"]), 30.)

    def test_common_rigid_transform_changes_neither_metric(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [13, 44, 77], degrees=True).as_matrix()
        transform[:3, 3] = [2., 3., 4.]
        initial, common = self.geometry(), self.geometry(transform, transform)
        for key in ("range_m", "bearing_error_deg", "optical_depth_m"):
            self.assertAlmostEqual(initial[key], common[key])
        np.testing.assert_allclose(initial["camera_direction_in_reference_hand"], common["camera_direction_in_reference_hand"], atol=1e-12)

    def test_depth_hole_occlusion_and_contradiction_are_not_visibility(self):
        geometry = self.geometry()
        for z, expected in ((np.nan, "DEPTH_UNKNOWN"), (.5, "OCCLUDED_BY_NEARER_SURFACE"),
                            (2., "OBSERVED_FREE_SPACE_CONTRADICTION"), (1., "SURFACE_DEPTH_COMPATIBLE_NOT_IDENTITY_PROOF")):
            result = projected_anchor_depth(geometry, np.full((101, 101), z, dtype=np.float32), self.camera)
            self.assertEqual(result["status"], expected)

    def test_invalid_anchor_or_camera_separation_fails(self):
        for anchor in ((0, 0, 0), (np.nan, 0, -1), (0, 1)):
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                self.geometry(anchor=anchor)
