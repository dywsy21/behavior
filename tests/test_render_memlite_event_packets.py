import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import build_memlite_event_index as builder  # noqa: E402
import render_memlite_event_packets as renderer  # noqa: E402
from test_build_memlite_event_index import coverage, write_release  # noqa: E402


class RenderEventPacketsTests(unittest.TestCase):
    def test_locator_packet_is_camera_separated_and_has_no_footer_or_label_in_actor_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            packets = root / "packets"
            result = renderer.create_packets(index, packets, event_ids=set(), limit=1, questions={},
                                             include_source_annotation_context=True, decode=False, raw_root=None,
                                             contact_sheets=False, max_seconds=10)
            self.assertEqual(result["status"], "LOCATORS_READY_RENDER_PENDING")
            row = json.loads((packets / "packets.jsonl").read_text())
            actor = row["actor_packet"]
            self.assertEqual(actor["images"], {})
            self.assertTrue(actor["question_context"]["source_annotation_is_not_truth"])
            self.assertNotIn("audit", actor)
            self.assertNotIn("review_only_contact_sheet", actor)
            self.assertEqual([item["view"] for item in row["audit"]["video_locators"]],
                             ["head", "left_wrist", "right_wrist"])
            self.assertFalse(result["review_only_contact_sheets"])
            self.assertEqual(set(result["files"]), {"packets.jsonl", "rendered_asset_receipts.jsonl"})
            self.assertEqual(result["rendered_asset_receipts"], 0)
            self.assertEqual(renderer.resume_packets(index, packets)["status"], "RESUME_VALIDATED")
            with (packets / "rendered_asset_receipts.jsonl").open("a") as stream:
                stream.write(json.dumps({"forged": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "rendered-asset receipt"):
                renderer.resume_packets(index, packets)

    def test_render_requires_bounded_selection_and_question_rows_cannot_carry_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            with self.assertRaisesRegex(ValueError, "select explicit"):
                renderer.create_packets(index, root / "bad", event_ids=set(), limit=None, questions={},
                                        include_source_annotation_context=False, decode=False, raw_root=None,
                                        contact_sheets=False, max_seconds=10)
            question_file = root / "questions.jsonl"
            question_file.write_text(json.dumps({"event_id": "a" * 64,
                                                  "question_context": {"ground_truth": "bad"}}) + "\n")
            with self.assertRaisesRegex(ValueError, "must not carry"):
                renderer._questions(question_file)
            with self.assertRaisesRegex(ValueError, "requires --decode"):
                renderer.create_packets(index, root / "bad-contact", event_ids=set(), limit=1, questions={},
                                        include_source_annotation_context=False, decode=False, raw_root=None,
                                        contact_sheets=True, max_seconds=10)
            with self.assertRaisesRegex(ValueError, "requires an existing"):
                renderer.create_packets(index, root / "bad-decode", event_ids=set(), limit=1, questions={},
                                        include_source_annotation_context=False, decode=True, raw_root=root / "missing",
                                        contact_sheets=False, max_seconds=10)

    def test_decoded_camera_native_pngs_have_sealed_hash_receipts(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            raw_root = root / "raw"
            raw_root.mkdir()

            def fake_decode(_av, _video, requested):
                return Image.new("RGB", (8, 6), "red"), {"requested_timestamp_s": requested,
                                                           "decoded_timestamp_s": requested, "pts_error_s": 0.0}

            with patch.object(renderer, "_require_decode_dependencies", return_value=(object(), Image, None)), \
                 patch.object(renderer, "_resolve_video", return_value=raw_root / "placeholder.mp4"), \
                 patch.object(renderer, "_decode_rgb", side_effect=fake_decode):
                packets = root / "packets"
                result = renderer.create_packets(index, packets, event_ids=set(), limit=1, questions={},
                                                 include_source_annotation_context=False, decode=True, raw_root=raw_root,
                                                 contact_sheets=False, max_seconds=10)
            self.assertEqual(result["rendered_asset_receipts"], 3)
            assets = [json.loads(line) for line in (packets / "rendered_asset_receipts.jsonl").read_text().splitlines()]
            self.assertEqual([row["view"] for row in assets], ["head", "left_wrist", "right_wrist"])
            for row in assets:
                path = packets / row["relative_path"]
                self.assertEqual(renderer._sha256(path), row["sha256"])
                self.assertEqual(path.stat().st_size, row["bytes"])
            with (packets / assets[0]["relative_path"]).open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "rendered PNG bytes changed"):
                renderer.resume_packets(index, packets)


if __name__ == "__main__":
    unittest.main()
