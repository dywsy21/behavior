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
    def test_sealed_ten_slot_calibration_request_keeps_actor_and_future_panels_separate(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            event = json.loads((index / "event_candidates.jsonl").read_text().splitlines()[0])
            self.assertEqual(event["usage_role"], "annotation_calibration")
            offsets = (-60, -45, -30, -15, 0, 1, 16, 31, 46, 60)
            samples = renderer._temporal_samples(event, offsets)
            self.assertEqual([sample["sample_frame"] for sample in samples], [0, 1, 16, 31, 46, 60])
            request = {
                "schema_version": "p107-camera-native-temporal-request-v1", "request_id": "1" * 64,
                "event_id": event["event_id"],
                "source_identity": {key: event["source"][key] for key in (
                    "source_release_manifest_sha256", "source_annotation_sha256", "task_index", "task_instance_id",
                    "raw_episode_id", "episode_index", "source_group_id")},
                "anchor_camera_locators": event["video_locators"],
                "actor_available_frame_indices": [0], "offline_review_before_frame_indices": [],
                "offline_review_after_frame_indices": [1, 16, 31, 46, 60],
                "requested_frame_indices": [0, 1, 16, 31, 46, 60],
                "camera_delivery": {"camera_native_only": True, "include_footer": False,
                                    "allow_contact_sheet_in_actor_input": False, "decoded_by_selector": False,
                                    "forbidden_footer_fields": ["outcome", "success", "failure", "recovery", "label", "evidence", "review"]},
                "frame_locator_resolution": "RENDERER_MUST_LOOK_UP_EACH_REQUESTED_FRAME_FROM_FROZEN_SOURCE_METADATA",
                "status": "PENDING_RENDERER_HANDSHAKE_NO_RGB_DECODED",
            }
            queue_root = root / "queue"
            queue_root.mkdir()
            request_path = queue_root / "camera_native_render_requests.jsonl"
            request_path.write_text(renderer.canonical_json(request) + "\n")
            request_sha = renderer._sha256(request_path)
            for filename, contents in {
                "annotation_calibration_queue.jsonl": b"{}\n",
                "counts.json": b"{}\n",
                "student_candidate_queue.jsonl": b"",
            }.items():
                (queue_root / filename).write_bytes(contents)
            expected_files = ["annotation_calibration_queue.jsonl", "camera_native_render_requests.jsonl",
                              "counts.json", "student_candidate_queue.jsonl"]
            payload_files = {}
            for filename in expected_files:
                source = queue_root / filename
                payload_files[filename] = {"sha256": renderer._sha256(source), "bytes": source.stat().st_size,
                                           "rows": len([line for line in source.read_bytes().splitlines() if line.strip()])}
            seal = {
                "schema_version": "p107-metadata-annotation-queue-seal-v1", "canonical_protocol_sha256": "2" * 64,
                "coverage_expectations_sha256": "3" * 64, "expected_payload_files": expected_files,
                "inventory_seal_sha256": "4" * 64, "payload_files": payload_files, "policy_sha256": "5" * 64,
                "queue_manifest_sha256": "6" * 64,
                "source_release_manifest_sha256": event["source"]["source_release_manifest_sha256"],
            }
            seal_path = queue_root / "queue_seal.json"
            seal_path.write_text(renderer.canonical_json(seal) + "\n")
            seal_sha = renderer._sha256(seal_path)
            requests, loaded_sha = renderer._read_render_requests(request_path, request_sha)
            queue_seal_sha, queue_source_sha = renderer._read_queue_seal(
                seal_path, seal_sha, render_requests_path=request_path, render_requests_sha256=loaded_sha)
            self.assertEqual(queue_seal_sha, seal_sha)
            self.assertEqual(queue_source_sha, event["source"]["source_release_manifest_sha256"])
            raw_root = root / "raw"
            raw_root.mkdir()

            def fake_decode(_av, _video, requested, **bounds):
                decoded = requested + 3e-12
                return Image.new("RGB", (8, 6), "red"), {
                    "resolved_path": "/fixture/shared.mp4", "bytes": 1, "mtime_ns": 1,
                    "requested_timestamp_s": requested, "decoded_timestamp_s": decoded,
                    "pts_error_s": abs(decoded - requested),
                    "fps": 30, "resolution": [8, 6],
                    "episode_global_pts_bounds_s": [bounds["episode_start_timestamp_s"], bounds["episode_end_timestamp_s"]],
                    "actor_anchor_timestamp_s": bounds["actor_anchor_timestamp_s"], "full_video_sha256": None,
                }

            with patch.object(renderer, "_require_decode_dependencies", return_value=(object(), Image, None)), \
                 patch.object(renderer, "_resolve_video", return_value=raw_root / "placeholder.mp4"), \
                 patch.object(renderer, "_decode_rgb", side_effect=fake_decode):
                packets = root / "packets"
                result = renderer.create_packets(
                    index, packets, event_ids={event["event_id"]}, limit=None, questions={},
                    include_source_annotation_context=False, decode=True, raw_root=raw_root, contact_sheets=False,
                    temporal_offsets_frames=offsets, render_requests=requests,
                    render_requests_sha256=loaded_sha, queue_seal_sha256=seal_sha,
                    queue_seal_source_release_manifest_sha256=queue_source_sha, max_seconds=10)
            row = json.loads((packets / "packets.jsonl").read_text())
            self.assertEqual(result["render_requests_sha256"], request_sha)
            self.assertEqual(result["queue_seal_sha256"], seal_sha)
            self.assertEqual(row["audit"]["requested_temporal_slot_count"], 10)
            self.assertEqual(row["audit"]["temporal_slot_count"], 6)
            self.assertEqual(row["audit"]["clamped_duplicate_sample_frame_count"], 4)
            self.assertEqual([sample["sample_frame"] for sample in row["actor_packet"]["causal_temporal_rgb"]], [0])
            self.assertEqual([sample["sample_frame"] for sample in row["audit"]["offline_future_rgb"]], [1, 16, 31, 46, 60])
            self.assertEqual(renderer.resume_packets(
                index, packets, expected_packet_manifest_sha256=result["packet_manifest_sha256"])["status"],
                "RESUME_VALIDATED")

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

    def test_protocol_reader_requires_exact_pinned_bytes_and_index_binding(self):
        """A historical index cannot be silently reinterpreted under DATA's default reader."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol_copy = root / "legacy_protocol.py"
            protocol_copy.write_bytes((REPO / "src/g05/data/memlite_event_protocol.py").read_bytes())
            protocol_sha = renderer._sha256(protocol_copy)
            binding = renderer._load_protocol(protocol_copy, expected_sha256=protocol_sha)
            self.assertEqual(binding.sha256, renderer.DEFAULT_PROTOCOL_SHA256)
            with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
                renderer._load_protocol(protocol_copy, expected_sha256="0" * 64)

            release = root / "release"
            fixture = write_release(release)
            index = root / "index"
            builder.build_index(release, index, coverage_expectations=coverage(fixture), max_seconds=10)
            manifest_path = index / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["protocol_sha256"] = "0" * 64
            manifest_path.write_text(renderer.canonical_json(manifest) + "\n")
            event = json.loads((index / "event_candidates.jsonl").read_text().splitlines()[0])
            with self.assertRaisesRegex(ValueError, "protocol does not match"):
                renderer.create_packets(
                    index, root / "packets", event_ids={event["event_id"]}, limit=None, questions={},
                    include_source_annotation_context=False, decode=False, raw_root=None, contact_sheets=False,
                    protocol_binding=binding, max_seconds=10)

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
            schedule = json.loads((packets / "packets.jsonl").read_text())["audit"]["temporal_schedule"]
            self.assertEqual(result["rendered_asset_receipts"], len(schedule) * 3)
            assets = [json.loads(line) for line in (packets / "rendered_asset_receipts.jsonl").read_text().splitlines()]
            self.assertEqual([row["view"] for row in assets[:3]], ["head", "left_wrist", "right_wrist"])
            self.assertEqual({row["sample_index"] for row in assets}, set(range(len(schedule))))
            self.assertEqual({row["temporal_role"] for row in assets},
                             {"ACTOR_CAUSAL", "OFFLINE_FUTURE_AUDIT"})
            for row in assets:
                path = packets / row["relative_path"]
                self.assertEqual(renderer._sha256(path), row["sha256"])
                self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(renderer.resume_packets(
                index, packets, expected_packet_manifest_sha256=result["packet_manifest_sha256"])["status"],
                "RESUME_VALIDATED")
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
            row = json.loads((packets / "packets.jsonl").read_text())
            self.assertEqual(result["rendered_asset_receipts"], len(row["audit"]["temporal_schedule"]) * 3 + 1)
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
            self.assertEqual([sample["sample_frame"] for sample in start_samples], [0, 15, 30])
            self.assertEqual([sample["offset_frames"] for sample in start_samples], [0, 15, 30])
            late_event = json.loads(json.dumps(event))
            late_event["observation"]["frame"] = late_event["source"]["episode_length"] - 1
            end_samples = renderer._temporal_samples(late_event, renderer.TEMPORAL_SAMPLE_OFFSETS)
            self.assertEqual(end_samples[-1]["sample_frame"], late_event["source"]["episode_length"] - 1)
            self.assertEqual(end_samples[-1]["offset_frames"], 0)

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

            # The exact lc3 failure: metadata's float seconds and the stream's
            # integral 30-Hz PTS identify the same frame but differ by 3e-12s.
            actor_anchor = 5885.166666666664
            same_frame_pts = 5885.166666666667
            self.assertTrue(renderer._clock_at_or_before(same_frame_pts, actor_anchor))
            self.assertFalse(renderer._clock_at_or_before(actor_anchor + 1e-6, actor_anchor))
            self.assertFalse(renderer._clock_at_or_before(actor_anchor + 1.0 / 30.0, actor_anchor))

            class LargeContainer:
                streams = type("Streams", (), {"video": [Stream()]})()

                def __init__(self, pts):
                    self._pts = pts

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def seek(self, *_args, **_kwargs):
                    return None

                def decode(self, _stream):
                    return iter((Frame(self._pts),))

            class LargeAv:
                def __init__(self, pts):
                    self._pts = pts

                def open(self, _path):
                    return LargeContainer(self._pts)

            image, receipt = renderer._decode_rgb(
                LargeAv(176555), video, actor_anchor, episode_start_timestamp_s=5880.0,
                episode_end_timestamp_s=5890.0, actor_anchor_timestamp_s=actor_anchor)
            self.assertEqual(receipt["decoded_timestamp_s"], same_frame_pts)
            image.close()
            with self.assertRaisesRegex(ValueError, "cannot decode"):
                renderer._decode_rgb(
                    LargeAv(176556), video, actor_anchor, episode_start_timestamp_s=5880.0,
                    episode_end_timestamp_s=5890.0, actor_anchor_timestamp_s=actor_anchor)

            class MicrosecondStream:
                average_rate = 30
                time_base = Fraction(1, 1_000_000)

            class MicrosecondContainer(LargeContainer):
                streams = type("Streams", (), {"video": [MicrosecondStream()]})()

            class MicrosecondAv:
                @staticmethod
                def open(_path):
                    return MicrosecondContainer(5_885_166_667)

            microsecond_earlier_anchor = 5885.166666
            with self.assertRaisesRegex(ValueError, "cannot decode"):
                renderer._decode_rgb(
                    MicrosecondAv(), video, microsecond_earlier_anchor, episode_start_timestamp_s=5880.0,
                    episode_end_timestamp_s=5890.0, actor_anchor_timestamp_s=microsecond_earlier_anchor)
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
