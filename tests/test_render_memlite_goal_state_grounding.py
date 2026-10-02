from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "data" / "render_memlite_goal_state_grounding.py"
SPEC = importlib.util.spec_from_file_location("private_goal_state_grounding_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
grounding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grounding)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _DistinctRenderer:
    SPECS = {
        "zed_link_camera_0": ((18, 12), 0.0, (17, 71, 131)),
        "left_realsense_link_camera_0": ((24, 14), 1.0 / 120.0, (41, 101, 53)),
        "right_realsense_link_camera_0": ((30, 16), -1.0 / 60.0, (113, 37, 89)),
    }

    def _require_decode_dependencies(self, *, contact_sheets: bool):
        del contact_sheets
        return object(), None, None

    def _decode_rgb(self, _av, video: Path, requested: float, **_kwargs):
        for suffix, (size, delta, colour) in self.SPECS.items():
            if suffix in str(video):
                image = Image.new("RGB", size, colour)
                return image, {
                    "resolved_path": str(video),
                    "bytes": video.stat().st_size,
                    "mtime_ns": video.stat().st_mtime_ns,
                    "requested_timestamp_s": requested,
                    "decoded_timestamp_s": requested + delta,
                    "pts_error_s": abs(delta),
                    "fps": 30.0,
                    "resolution": list(size),
                    "full_video_sha256": None,
                }
        raise AssertionError(f"unknown fake camera path: {video}")


class _MismatchedRenderer(_DistinctRenderer):
    def _decode_rgb(self, av, video: Path, requested: float, **kwargs):
        image, decoded = super()._decode_rgb(av, video, requested, **kwargs)
        if "left_realsense_link_camera_0" in str(video):
            decoded["resolved_path"] = str(video) + ".wrong"
        return image, decoded


class GoalStateGroundingTests(unittest.TestCase):
    def _fixture(self) -> tuple[Path, Path, Path, dict[str, object]]:
        root = Path(tempfile.mkdtemp(prefix="goal-state-grounding-test-"))
        batch_dir = root / "batch"
        index_dir = root / "index"
        raw_root = root / "raw"
        batch_dir.mkdir()
        index_dir.mkdir()
        raw_root.mkdir()

        source = {
            "episode_index": 7,
            "raw_episode_id": 70,
            "source_annotation_sha256": "b" * 64,
            "source_group_id": "c" * 64,
            "source_release_manifest_sha256": "a" * 64,
            "task_index": 3,
            "task_instance_id": 4,
            "episode_length": 1000,
            "original_split": "train",
        }
        locators = []
        for view, camera in grounding.CAMERA_BY_VIEW.items():
            relative = f"videos/{camera}/chunk-000/file-000.mp4"
            video = raw_root / relative
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"synthetic video placeholder")
            locators.append({
                "camera_key": camera,
                "episode_start_timestamp_s": 1000.0,
                "expected_fps": 30,
                "locator_status": "METADATA_ONLY_UNRESOLVED",
                "relative_path": relative,
                "requested_timestamp_s": 1000.0 + 10 / 30.0,
                "view": view,
            })
        parent_skill = {
            "arm": "UNSPECIFIED",
            "binding_confidence": "BOUND",
            "destination": "",
            "interval_id": 0,
            "raw_description": "pick up from",
            "raw_relation": {"manipulating_object_id": ["object_1"], "memory_prefix": [], "object_id": [["object_1", "table_0"]], "spatial_prefix": []},
            "skill_end": 100,
            "skill_id": 2,
            "skill_idx": 1,
            "skill_start": 10,
            "source": "table_0",
            "target": "object_1",
            "target_part": "",
            "verb": "GRASP",
        }
        event = {
            "source": source,
            "observation": {"frame": 10, "timestamp_s": 10 / 30.0},
            "video_locators": locators,
            "event_interval": {"start_frame": 10, "end_frame": 100},
            "event_kind": "ANNOTATED_SKILL_SEGMENT_PHASE_ENTRY",
            "schema_version": grounding.EVENT_SCHEMA,
            "record_kind": "event_candidate",
            "event_id": "d" * 64,
            "usage_role": "student_candidate",
            "phase_lineage": {"parent_event_id": "e" * 64, "parent_skill_identity": {"parent_skill_member_sha256": grounding._canonical_sha256(parent_skill)}, "goal_state_mode": "goal_unbound"},
        }
        fake_summary = {
            "schema_version": grounding.PREFLIGHT_SCHEMA,
            "status": "PRIVATE_GOAL_STATE_RGB_PREFLIGHT_PASS",
            "role": "private_verifier_only",
            "usage_role": "student_candidate",
            "split": "train",
            "private_goal_state_mode": "goal_unbound",
            "training_eligible": False,
            "action_bc_supervision": False,
            "outcome_supervision": False,
            "recovery_supervision": False,
            "dart_supervision": False,
            "attempt_status": "NOT_APPLICABLE",
            "outcome_status": "NOT_APPLICABLE",
            "recovery_status": "NOT_APPLICABLE",
            "release_status": "NOT_RELEASED",
            "actor_packet_included": False,
            "questions_included": False,
            "answers_included": False,
            "decode_performed": False,
            "parent_count": 1,
            "anchor_count": 1,
            "unique_frame_count": 1,
            "camera_view_count": 3,
            "native_png_count": 3,
            "unique_frames": [{"episode_index": 7, "frame_index": 10, "anchor_event_ids": [event["event_id"]]}],
            "offline_review_before_frames": [],
            "offline_review_after_frames": [],
        }
        preflight = {"summary": fake_summary, "selected_events": [event]}
        return batch_dir, index_dir, raw_root, preflight

    def _mutated_real_batch(self, file_name: str, mutate) -> tuple[Path, str]:
        source_batch = Path("/home/wsy/behavior-annotations/p107/natural-retry-grasp-goal-state-v1/formal-batch-v1")
        if not source_batch.exists():
            self.skipTest("local sealed metadata is unavailable")
        root = Path(tempfile.mkdtemp(prefix="goal-state-grounding-payload-negative-"))
        batch = root / "formal-batch-v1"
        shutil.copytree(source_batch, batch)
        payload = batch / file_name
        rows = [json.loads(line) for line in payload.read_text().splitlines() if line.strip()]
        for row in rows:
            source = row.get("source") or row.get("source_identity") or {}
            if row.get("episode_index", source.get("episode_index")) == 3607:
                mutate(row)
                break
        else:
            raise AssertionError(f"no selected episode row in {file_name}")
        payload.write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n" for row in rows))
        manifest_path = batch / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        receipt = manifest["files"][file_name]
        receipt["bytes"] = payload.stat().st_size
        receipt["rows"] = len(rows)
        receipt["sha256"] = _sha(payload)
        manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n")
        return batch, _sha(manifest_path)

    def test_metadata_preflight_contract_is_bounded_and_has_no_query_or_future(self) -> None:
        _batch, _index, _raw, preflight = self._fixture()
        summary = preflight["summary"]
        self.assertEqual(summary["native_png_count"], 3)
        self.assertEqual(summary["camera_view_count"], 3)
        self.assertEqual(summary["offline_review_before_frames"], [])
        self.assertEqual(summary["offline_review_after_frames"], [])
        self.assertFalse(summary["training_eligible"])
        self.assertNotIn("questions", summary)
        self.assertNotIn("answers", summary)

    def test_decode_binds_each_view_locator_pts_dimensions_and_png(self) -> None:
        batch, index, raw, preflight = self._fixture()
        output = batch.parent / "output"
        with patch.object(grounding, "validate_preflight", return_value=preflight):
            result = grounding.run_grounding(
                batch,
                index,
                output,
                raw_root=raw,
                expected_batch_manifest_sha256="a" * 64,
                expected_event_index_manifest_sha256="b" * 64,
                expected_event_candidates_sha256="c" * 64,
                expected_source_groups_sha256="d" * 64,
                episode_indices=[7],
                renderer=_DistinctRenderer(),
                av_backend=object(),
            )
        self.assertTrue(output.is_dir())
        self.assertEqual(result["native_png_count"], 3)
        rows = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
        self.assertEqual({row["camera_view"] for row in rows}, set(grounding.VIEWS))
        for row in rows:
            png = output / row["png_relative_path"]
            self.assertEqual(row["png_sha256"], _sha(png))
            with Image.open(png) as image:
                self.assertEqual(list(image.size), row["png_dimensions"])
            self.assertEqual(row["decoded_source_container"]["resolved_path"], row["camera_video_locator"]["relative_path"].replace("videos/", str(raw / "videos/") + "/", 1))
            self.assertEqual(row["png_bytes"], png.stat().st_size)
        self.assertEqual(result["artifact_bytes"], sum(path.stat().st_size for path in output.rglob("*") if path.is_file()))
        self.assertFalse(result["questions_included"])
        self.assertFalse(result["answers_included"])

    def test_decoder_locator_mismatch_fails_closed_without_output(self) -> None:
        batch, index, raw, preflight = self._fixture()
        output = batch.parent / "rejected"
        with patch.object(grounding, "validate_preflight", return_value=preflight):
            with self.assertRaises(grounding.InputValidationError):
                grounding.run_grounding(
                    batch,
                    index,
                    output,
                    raw_root=raw,
                    expected_batch_manifest_sha256="a" * 64,
                    expected_event_index_manifest_sha256="b" * 64,
                    expected_event_candidates_sha256="c" * 64,
                    expected_source_groups_sha256="d" * 64,
                    episode_indices=[7],
                    renderer=_MismatchedRenderer(),
                    av_backend=object(),
                )
        self.assertFalse(output.exists())

    def test_locator_schema_rejects_unknown_view_and_unsafe_path(self) -> None:
        _batch, _index, _raw, _preflight = self._fixture()
        locator = {
            "camera_key": grounding.CAMERA_BY_VIEW["head"],
            "episode_start_timestamp_s": 0.0,
            "expected_fps": 30,
            "locator_status": "METADATA_ONLY_UNRESOLVED",
            "relative_path": "../escape.mp4",
            "requested_timestamp_s": 0.0,
            "view": "unknown",
        }
        with self.assertRaises(grounding.InputValidationError):
            grounding._locators_by_view([locator], "fixture")

    def test_canonical_metadata_goal_query_is_allowed_but_query_fields_outside_it_are_not(self) -> None:
        grounding._reject_explicit_unbound_annotations(
            {
                "phase_lineage": {
                    "goal_query": {
                        "query_kind": "METADATA_DESCRIBED_SKILL_GOAL_RELATION",
                        "query_scope": "SAME_ORIGINAL_ANNOTATED_SEGMENT_SKILL",
                    }
                }
            },
            "fixture",
        )
        with self.assertRaises(grounding.InputValidationError):
            grounding._reject_explicit_unbound_annotations({"phase_lineage": {"goal_state_question": "pick it up"}}, "fixture")

    def test_recomputed_payload_and_manifest_pins_still_reject_explicit_annotation_fields(self) -> None:
        batch_dir = Path("/home/wsy/behavior-annotations/p107/natural-retry-grasp-goal-state-v1/formal-batch-v1")
        index_dir = Path("/home/wsy/behavior-annotations/p107/index-validation/full-v3-candidate-index")
        if not batch_dir.exists() or not index_dir.exists():
            self.skipTest("local sealed metadata is unavailable")
        mutations = (
            ("candidate_rows.jsonl", lambda row: row.update({"query_id": "injected-query"})),
            ("goal_unbound_events.jsonl", lambda row: row.update({"question": "injected-question"})),
            ("goal_unbound_events.jsonl", lambda row: row.setdefault("phase_lineage", {}).update({"goal_state_question": "injected-question"})),
            ("student_candidate_queue.jsonl", lambda row: row.update({"question": "injected-question"})),
            ("camera_native_render_requests.jsonl", lambda row: row.update({"placeholder": "injected-placeholder"})),
        )
        for file_name, mutate in mutations:
            with self.subTest(file_name=file_name, mutation=mutate):
                batch, batch_sha = self._mutated_real_batch(file_name, mutate)
                with self.assertRaises(grounding.InputValidationError):
                    grounding.validate_preflight(
                        batch,
                        index_dir,
                        expected_batch_manifest_sha256=batch_sha,
                        expected_event_index_manifest_sha256="351fa44b03fe881b328bbafd5626b986c93879d7f14da346ac8b62a079d200bf",
                        expected_event_candidates_sha256="c12bfa8ba9375b208525a25eb2c1b3944ef5c980a021599c174e18b186886329",
                        expected_source_groups_sha256="cdd1c37db3f7af3a743ad25b1a6fc8c4b0da41299ecf3bd8d10b9645193673e3",
                        episode_indices=[3607, 4512],
                    )

    def test_real_manifest_wrong_pin_rejects_before_full_index_read(self) -> None:
        batch = Path("/home/wsy/behavior-annotations/p107/natural-retry-grasp-goal-state-v1/formal-batch-v1")
        index = Path("/home/wsy/behavior-annotations/p107/index-validation/full-v3-candidate-index")
        if not batch.exists() or not index.exists():
            self.skipTest("local sealed metadata is unavailable")
        with self.assertRaises(grounding.InputValidationError):
            grounding.validate_preflight(
                batch,
                index,
                expected_batch_manifest_sha256="0" * 64,
                expected_event_index_manifest_sha256="1" * 64,
                expected_event_candidates_sha256="2" * 64,
                expected_source_groups_sha256="3" * 64,
                episode_indices=[3607, 4512],
            )


if __name__ == "__main__":
    unittest.main()
