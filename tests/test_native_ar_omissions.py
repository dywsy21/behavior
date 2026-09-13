"""CPU/stdlib parser audit checks; no models or remote files needed."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/experiments/probe_native_ar_omissions.py"
SPEC = importlib.util.spec_from_file_location("native_omission_test", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
GRAMMAR = dict(markers={10: ("arm", 0, 2), 11: ("arm", 1, 2), 12: ("base", 0, 2), 13: ("gripper", 0, 1)},
    codebook_size=10, offset=100)


class OmissionTests(unittest.TestCase):
    def test_present_groups_can_be_complete_without_all_groups(self):
        got = audit.block_layout([110, 100, 101, 111, 102, 103], **GRAMMAR)
        self.assertTrue(got["complete_present_blocks"])
        self.assertEqual(got["present"], ["arm"])
        self.assertEqual(got["absent"], ["base", "gripper"])

    def test_missing_residual_is_not_a_complete_present_group(self):
        got = audit.block_layout([110, 100, 101], **GRAMMAR)
        self.assertEqual(got["incomplete_residual_groups"], ["arm"])
        self.assertFalse(got["complete_present_blocks"])

    def test_codec_noop_does_not_excuse_missing_active_base(self):
        got = audit.block_layout([110, 100, 101, 111, 102, 103], **GRAMMAR)
        result = audit.omission_classification(got, ["gripper"])
        self.assertEqual(result["codec_noop_omissions"], ["gripper"])
        self.assertEqual(result["omitted_non_noop_target_groups"], ["base"])
        self.assertFalse(result["deployable_claim"])
        self.assertFalse(result["physical_inactivity_proven"])

    def test_duplicate_unknown_truncated_and_embedded_markers_rejected(self):
        for ids in ([], [110, 100], [110, 100, 112], [199], [110, 100, 101, 110, 102, 103], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                audit.block_layout(ids, **GRAMMAR)

    def test_fully_present_has_no_omissions(self):
        got = audit.block_layout([110, 100, 101, 111, 102, 103, 112, 101, 102, 113, 103], **GRAMMAR)
        self.assertEqual(got["absent"], [])
        self.assertEqual(got["block_count"], 4)
        self.assertTrue(got["complete_present_blocks"])

    def test_identity_includes_episode_frame_requested_index_and_bundle(self):
        source = dict(task=0, episode=99, frame=1782, requested_index=233382, bundle_id="audited")
        original = audit.source_key(source)
        for key in source:
            self.assertNotEqual(original, audit.source_key({**source, key: "different"}))

    def test_other_embodiment_noop_rejected(self):
        got = audit.block_layout([110, 100, 101, 111, 102, 103], **GRAMMAR)
        with self.assertRaises(ValueError):
            audit.omission_classification(got, ["another_robot"])


if __name__ == "__main__":
    unittest.main()
