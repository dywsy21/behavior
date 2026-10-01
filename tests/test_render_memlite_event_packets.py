import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from fractions import Fraction


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
            self.assertEqual(renderer.resume_packets(
                index, packets, expected_packet_manifest_sha256=result["packet_manifest_sha256"])["status"],
                "RESUME_VALIDATED")
            with (packets / "rendered_asset_receipts.jsonl").open("a") as stream:
                stream.write(json.dumps({"forged": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "rendered-asset receipt"):
                renderer.resume_packets(index, packets,
                                        expected_packet_manifest_sha256=result["packet_manifest_sha256"])

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
            question_file.write_text(json.dumps({"event_id": "a" * 64,
                                                  "question_context": {"prompt": {"outcome": "SUCCESS"}}}) + "\n")
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

            def fake_decode(_av, _video, requested, **bounds):
                return Image.new("RGB", (8, 6), "red"), {
                    "resolved_path": "/fixture/shared.mp4", "bytes": 1, "mtime_ns": 1,
                    "requested_timestamp_s": requested, "decoded_timestamp_s": requested, "pts_error_s": 0.0,
                    "fps": 30, "resolution": [8, 6],
                    "episode_global_pts_bounds_s": [bounds["episode_start_timestamp_s"], bounds["episode_end_timestamp_s"]],
                    "actor_anchor_timestamp_s": bounds["actor_anchor_timestamp_s"], "full_video_sha256": None,
                }

            with patch.object(renderer, "_require_decode_dependencies", return_value=(object(), Image, None)), \
                 patch.object(renderer, "_resolve_video", return_value=raw_root / "placeholder.mp4"), \
                 patch.object(renderer, "_decode_rgb", side_effect=fake_decode):
                packets = root / "packets"
                result = renderer.create_packets(index, packets, event_ids=set(), limit=1, questions={},
                                                 include_source_annotation_context=False, decode=True, raw_root=raw_root,
                                                 contact_sheets=False, max_seconds=10)
            self.assertEqual(result["rendered_asset_receipts"], 21)
            assets = [json.loads(line) for line in (packets / "rendered_asset_receipts.jsonl").read_text().splitlines()]
            self.assertEqual([row["view"] for row in assets[:3]], ["head", "left_wrist", "right_wrist"])
            self.assertEqual({row["sample_index"] for row in assets}, set(range(7)))
            self.assertEqual({row["temporal_role"] for row in assets},
                             {"ACTOR_CAUSAL", "OFFLINE_FUTURE_AUDIT"})
            for row in assets:
                path = packets / row["relative_path"]
                self.assertEqual(renderer._sha256(path), row["sha256"])
                self.assertEqual(path.stat().st_size, row["bytes"])
            with (packets / assets[0]["relative_path"]).open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "rendered PNG bytes changed"):
                renderer.resume_packets(index, packets,
                                        expected_packet_manifest_sha256=result["packet_manifest_sha256"])

    def test_contact_sheet_is_audited_only_and_native_temporal_assets_are_complete(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            raw_root = root / "raw"
            raw_root.mkdir()

            def fake_decode(_av, _video, requested, **bounds):
                return Image.new("RGB", (8, 6), "blue"), {
                    "resolved_path": "/fixture/shared.mp4", "bytes": 1, "mtime_ns": 1,
                    "requested_timestamp_s": requested, "decoded_timestamp_s": requested, "pts_error_s": 0.0,
                    "fps": 30, "resolution": [8, 6],
                    "episode_global_pts_bounds_s": [bounds["episode_start_timestamp_s"], bounds["episode_end_timestamp_s"]],
                    "actor_anchor_timestamp_s": bounds["actor_anchor_timestamp_s"], "full_video_sha256": None,
                }

            with patch.object(renderer, "_require_decode_dependencies", return_value=(object(), Image, ImageDraw)), \
                 patch.object(renderer, "_resolve_video", return_value=raw_root / "placeholder.mp4"), \
                 patch.object(renderer, "_decode_rgb", side_effect=fake_decode):
                packets = root / "packets"
                result = renderer.create_packets(index, packets, event_ids=set(), limit=1, questions={},
                                                 include_source_annotation_context=False, decode=True, raw_root=raw_root,
                                                 contact_sheets=True, max_seconds=10)
            self.assertEqual(result["rendered_asset_receipts"], 22)
            row = json.loads((packets / "packets.jsonl").read_text())
            self.assertTrue(all(path.startswith("assets/") for path in row["actor_packet"]["images"].values()))
            self.assertTrue(row["audit"]["review_only_contact_sheet"].startswith("review_assets/"))
            self.assertNotIn(row["audit"]["review_only_contact_sheet"], row["actor_packet"]["images"].values())
            self.assertEqual(renderer.resume_packets(
                index, packets, expected_packet_manifest_sha256=result["packet_manifest_sha256"])["status"],
                "RESUME_VALIDATED")

    def test_temporal_clamping_and_pts_selection_never_cross_episode_or_actor_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            event = json.loads((index / "event_candidates.jsonl").read_text().splitlines()[0])
            start_samples = renderer._temporal_samples(event, renderer.TEMPORAL_SAMPLE_OFFSETS)
            self.assertEqual(len(start_samples), 7)  # clipped repeats remain one packet, not extra candidates
            self.assertEqual([sample["sample_frame"] for sample in start_samples[:5]], [0, 0, 0, 0, 0])
            late_event = json.loads(json.dumps(event))
            late_event["observation"]["frame"] = late_event["source"]["episode_length"] - 1
            end_samples = renderer._temporal_samples(late_event, renderer.TEMPORAL_SAMPLE_OFFSETS)
            self.assertEqual([sample["sample_frame"] for sample in end_samples[-2:]],
                             [late_event["source"]["episode_length"] - 1] * 2)

            from PIL import Image

            class Frame:
                def __init__(self, pts):
                    self.pts = pts

                def to_image(self):
                    return Image.new("RGB", (2, 2), "green")

            class Stream:
                average_rate = 30
                time_base = Fraction(1, 30)

            class Container:
                streams = type("Streams", (), {"video": [Stream()]})()

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def seek(self, *_args, **_kwargs):
                    return None

                def decode(self, _stream):
                    return iter((Frame(30), Frame(31)))

            class Av:
                @staticmethod
                def open(_path):
                    return Container()

            video = root / "shared-container.mp4"
            video.write_bytes(b"fixture")
            # At half-way rounding, frame 31 would be future of the actor
            # anchor.  The decoder chooses frame 30 instead.
            image, receipt = renderer._decode_rgb(
                Av(), video, 30.5 / 30.0, episode_start_timestamp_s=1.0,
                episode_end_timestamp_s=2.0, actor_anchor_timestamp_s=30.5 / 30.0)
            self.assertEqual(receipt["decoded_timestamp_s"], 1.0)
            image.close()
            with self.assertRaisesRegex(ValueError, "outside the source episode"):
                renderer._decode_rgb(Av(), video, 2.1, episode_start_timestamp_s=1.0,
                                     episode_end_timestamp_s=2.0, actor_anchor_timestamp_s=None)

    def test_resume_rejects_self_resigned_manifest_without_external_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            packets = root / "packets"
            result = renderer.create_packets(index, packets, event_ids=set(), limit=1, questions={},
                                             include_source_annotation_context=False, decode=False, raw_root=None,
                                             contact_sheets=False, max_seconds=10)
            forged = json.loads((packets / "manifest.json").read_text())
            forged["training_eligible"] = True
            forged["status"] = "AUDITED"
            (packets / "manifest.json").write_text(renderer.canonical_json(forged) + "\n")
            with self.assertRaisesRegex(ValueError, "externally pinned"):
                renderer.resume_packets(index, packets,
                                        expected_packet_manifest_sha256=result["packet_manifest_sha256"])
            with self.assertRaisesRegex(ValueError, "requires an externally pinned"):
                renderer.resume_packets(index, packets, expected_packet_manifest_sha256=None)

    def test_resume_rejects_semantically_forged_actor_payload_even_if_new_hash_is_supplied(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            packets = root / "packets"
            renderer.create_packets(index, packets, event_ids=set(), limit=1, questions={},
                                    include_source_annotation_context=False, decode=False, raw_root=None,
                                    contact_sheets=False, max_seconds=10)
            rows = [json.loads(line) for line in (packets / "packets.jsonl").read_text().splitlines()]
            rows[0]["actor_packet"]["question_context"] = {"outcome": "SUCCESS"}
            payload = "".join(renderer.canonical_json(row) + "\n" for row in rows)
            (packets / "packets.jsonl").write_text(payload)
            manifest = json.loads((packets / "manifest.json").read_text())
            manifest["files"]["packets.jsonl"] = {"sha256": renderer._sha256(packets / "packets.jsonl"),
                                                      "bytes": (packets / "packets.jsonl").stat().st_size,
                                                      "rows": 1}
            (packets / "manifest.json").write_text(renderer.canonical_json(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                renderer.resume_packets(index, packets,
                                        expected_packet_manifest_sha256=renderer._sha256(packets / "manifest.json"))

    def test_resume_rechecks_canonical_event_packet_id_and_exact_manifest_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            packets = root / "packets"
            renderer.create_packets(index, packets, event_ids=set(), limit=1, questions={},
                                    include_source_annotation_context=False, decode=False, raw_root=None,
                                    contact_sheets=False, max_seconds=10)
            rows = [json.loads(line) for line in (packets / "packets.jsonl").read_text().splitlines()]
            rows[0]["packet_id"] = "f" * 64
            rows[0]["actor_packet"]["packet_id"] = "f" * 64
            (packets / "packets.jsonl").write_text("".join(renderer.canonical_json(row) + "\n" for row in rows))
            manifest = json.loads((packets / "manifest.json").read_text())
            manifest["files"]["packets.jsonl"] = {"sha256": renderer._sha256(packets / "packets.jsonl"),
                                                      "bytes": (packets / "packets.jsonl").stat().st_size, "rows": 1}
            (packets / "manifest.json").write_text(renderer.canonical_json(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "packet_id"):
                renderer.resume_packets(index, packets,
                                        expected_packet_manifest_sha256=renderer._sha256(packets / "manifest.json"))

    def test_resume_rejects_resigned_mutable_actor_instruction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            packets = root / "packets"
            renderer.create_packets(index, packets, event_ids=set(), limit=1, questions={},
                                    include_source_annotation_context=False, decode=False, raw_root=None,
                                    contact_sheets=False, max_seconds=10)
            rows = [json.loads(line) for line in (packets / "packets.jsonl").read_text().splitlines()]
            rows[0]["actor_packet"]["actor_instruction"] = "The answer is SUCCESS."
            (packets / "packets.jsonl").write_text("".join(renderer.canonical_json(row) + "\n" for row in rows))
            manifest = json.loads((packets / "manifest.json").read_text())
            manifest["files"]["packets.jsonl"] = {"sha256": renderer._sha256(packets / "packets.jsonl"),
                                                      "bytes": (packets / "packets.jsonl").stat().st_size, "rows": 1}
            (packets / "manifest.json").write_text(renderer.canonical_json(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "instruction"):
                renderer.resume_packets(index, packets,
                                        expected_packet_manifest_sha256=renderer._sha256(packets / "manifest.json"))

            manifest["outcome"] = "SUCCESS"
            (packets / "manifest.json").write_text(renderer.canonical_json(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "sealed P107 packet directory"):
                renderer.resume_packets(index, packets,
                                        expected_packet_manifest_sha256=renderer._sha256(packets / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
