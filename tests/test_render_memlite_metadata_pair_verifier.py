from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "data" / "render_memlite_metadata_pair_verifier.py"
SPEC = importlib.util.spec_from_file_location("metadata_pair_verifier_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class _FakeFrame:
    def __init__(self, pts: int) -> None:
        self.pts = pts

    def to_image(self) -> Image.Image:
        return Image.new("RGB", (20, 12), (33, 71, 131))


class _FakeStream:
    average_rate = 30
    time_base = Fraction(1, 30)


class _FakeContainer:
    streams = type("Streams", (), {"video": [_FakeStream()]})()

    def __init__(self, path: Path) -> None:
        self.path = path
        self.target = 0

    def __enter__(self) -> "_FakeContainer":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def seek(self, offset: int, **_kwargs: object) -> None:
        self.target = offset

    def decode(self, _stream: object):
        yield _FakeFrame(self.target)
        yield _FakeFrame(self.target + 1)


class _FakeAV:
    def open(self, path: str) -> _FakeContainer:
        return _FakeContainer(Path(path))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MetadataPairVerifierTests(unittest.TestCase):
    def _make_fixture(self, root: Path) -> dict[str, object]:
        inputs = root / "inputs"
        index = inputs / "event-index"
        raw = root / "raw"
        inputs.mkdir()
        index.mkdir()
        annotation = "a" * 64
        group = "b" * 64
        release = "c" * 64
        first = {
            "arm": "UNSPECIFIED", "destination": "", "skill_end": 2, "skill_id": 2,
            "skill_idx": 0, "skill_start": 0, "source": "table", "target": "box",
            "target_part": "", "verb": "GRASP",
        }
        second = {**first, "skill_end": 4, "skill_idx": 1, "skill_start": 2}
        candidate = {
            "action_supervision": False, "arm_first": "UNSPECIFIED", "arm_second": "UNSPECIFIED",
            "attempt_status": "NOT_APPLICABLE", "candidate_id": "ep00001-target-box-grasp-1-to-2",
            "candidate_kind": "same_target_repeat_strict_nonoverlap", "episode_index": 1,
            "first_grasp": first, "intervening_skills": [], "outcome_status": "NOT_APPLICABLE",
            "outcome_supervision": False, "raw_episode_id": 10, "recovery_status": "NOT_APPLICABLE",
            "recovery_supervision": False, "second_grasp": second, "source_annotation_sha256": annotation,
            "source_first": "table", "source_group_id": group, "source_second": "table", "split": "train",
            "strict_time_order": True, "target": "box", "task_index": 2, "task_instance_id": 3,
            "task_name": "putting box", "training_eligible": False, "usage_role": "student_candidate",
        }
        frozen_row = {
            "annotation_path": "annotations/task-0002/episode_00000010.json", "dataset_from_index": 1000,
            "dataset_to_index": 1008, "episode_index": 1, "length": 8, "raw_episode_id": 10,
            "task_index": 2, "task_instance_id": 3, "tasks": ["putting_box"],
        }
        starts = {"head": 100.0, "left_wrist": 200.0, "right_wrist": 300.0}
        for view, camera in verifier.CAMERA_NAMES.items():
            prefix = f"videos/observation.rgb.{camera}"
            frozen_row[f"{prefix}/chunk_index"] = 0
            frozen_row[f"{prefix}/file_index"] = 0
            frozen_row[f"{prefix}/from_timestamp"] = starts[view]
            frozen_row[f"{prefix}/to_timestamp"] = starts[view] + 8 / 30.0
        frozen = {
            "annotation_sha256": annotation, "phase": 1, "row": frozen_row, "segments": [
                {"end": 4, "skills": [first, second], "start": 0}
            ], "split": "train", "task_name": "putting box",
        }
        frozen_path = inputs / "episodes.jsonl"
        frozen_path.write_bytes((json.dumps(frozen, sort_keys=True, separators=(",", ":")) + "\n").encode())
        triage_summary = {
            "record_kind": "summary", "schema_version": verifier.TRIAGE_SCHEMA,
            "status": "strict_metadata_triage_only_nontraining", "selected_candidate_count": 1,
            "metadata_sha256": _sha(frozen_path), "source_release_manifest_sha256_values": [release],
            "status_fields": {"attempt_status": "NOT_APPLICABLE", "outcome_status": "NOT_APPLICABLE", "recovery_status": "NOT_APPLICABLE", "training_eligible": False},
        }
        triage_path = inputs / "triage.jsonl"
        triage_path.write_bytes(b"\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() for row in (triage_summary, candidate)) + b"\n")
        events = []
        for event_id, skill in (("event-first", first), ("event-second", second)):
            locators = []
            for view, camera in verifier.CAMERA_NAMES.items():
                relative = f"videos/observation.rgb.{camera}/chunk-000/file-000.mp4"
                (raw / relative).parent.mkdir(parents=True, exist_ok=True)
                (raw / relative).write_bytes(b"tiny video")
                start = starts[view]
                locators.append({
                    "camera_key": f"observation.rgb.{camera}", "episode_start_timestamp_s": start,
                    "expected_fps": 30, "locator_status": "METADATA_ONLY_UNRESOLVED", "relative_path": relative,
                    "requested_timestamp_s": start + skill["skill_start"] / 30.0, "view": view,
                })
            events.append({
                "event_id": event_id, "event_interval": {"start_frame": skill["skill_start"], "end_frame": skill["skill_end"]},
                "event_kind": "ANNOTATED_SKILL_SEGMENT", "observation": {"frame": skill["skill_start"], "timestamp_s": skill["skill_start"] / 30.0},
                "record_kind": "event_candidate", "schema_version": verifier.EVENT_SCHEMA,
                "skill_bundle": [skill], "source": {
                    "episode_index": 1, "episode_length": 8, "original_split": "train", "raw_episode_id": 10,
                    "source_annotation_sha256": annotation, "source_group_id": group,
                    "source_release_manifest_sha256": release, "task_index": 2, "task_instance_id": 3,
                }, "task_name": "putting box", "usage_role": "student_candidate", "video_locators": locators,
            })
        event_path = index / "event_candidates.jsonl"
        event_path.write_bytes(b"".join((json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode() for event in events))
        receipt = {"sha256": _sha(event_path), "rows": 2, "bytes": event_path.stat().st_size}
        manifest = {"schema_version": verifier.EVENT_INDEX_SCHEMA, "source_release_manifest_sha256": release, "files": {"event_candidates.jsonl": receipt}}
        manifest_path = index / "manifest.json"
        manifest_path.write_bytes((json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode())
        selection_doc = {
            "schema_version": verifier.SELECTION_SCHEMA, "selection_kind": "explicit_private_pair_windows",
            "source_pins": {"triage_sha256": _sha(triage_path), "frozen_metadata_sha256": _sha(frozen_path), "event_index_manifest_sha256": _sha(manifest_path), "source_release_manifest_sha256": release},
            "candidates": [{"candidate_id": candidate["candidate_id"], "window_local": [0, 4], "window_kind": "first_grasp_start_to_second_grasp_end_half_open", "review_role": "primary_structural_candidate", "private_review_may_include_future_or_cross_skill": True}],
        }
        selection_path = inputs / "selection.json"
        selection_path.write_bytes((json.dumps(selection_doc, sort_keys=True, separators=(",", ":")) + "\n").encode())
        return {
            "selection": selection_path, "triage": triage_path, "frozen": frozen_path, "index": index, "raw": raw,
            "output": root / "output", "selection_sha": _sha(selection_path), "triage_sha": _sha(triage_path),
            "frozen_sha": _sha(frozen_path), "manifest_sha": _sha(manifest_path), "release": release,
            "candidate_id": candidate["candidate_id"],
        }

    def test_synthetic_run_binds_each_camera_clock_and_writes_private_only(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._make_fixture(root)
            with patch.object(verifier, "TRIAGE_CANDIDATE_COUNT", 1), patch.object(verifier, "FROZEN_ROW_COUNT", 1):
                result = verifier.run_verifier(
                    fixture["selection"], fixture["triage"], fixture["frozen"], fixture["index"], fixture["raw"], fixture["output"],
                    expected_selection_sha256=fixture["selection_sha"], expected_triage_sha256=fixture["triage_sha"],
                    expected_frozen_metadata_sha256=fixture["frozen_sha"], expected_event_index_manifest_sha256=fixture["manifest_sha"],
                    expected_release_sha256=fixture["release"], candidate_ids={fixture["candidate_id"]}, av_backend=_FakeAV(),
                )
            self.assertEqual(result["native_png_count"], 12)
            rows = [json.loads(line) for line in (fixture["output"] / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["role"] == "private_verifier_only" and not row["actor_packet_included"] for row in rows))
            self.assertTrue(all(row["decoded_source_container"]["bytes"] == len(b"tiny video") for row in rows))
            by_view = {view: next(row for row in rows if row["camera_video_locator"]["camera_view"] == view) for view in verifier.VIEWS}
            self.assertAlmostEqual(by_view["head"]["requested_timestamp_s"], 100.0)
            self.assertAlmostEqual(by_view["left_wrist"]["requested_timestamp_s"], 200.0)
            self.assertAlmostEqual(by_view["right_wrist"]["requested_timestamp_s"], 300.0)
            self.assertEqual(len(list((fixture["output"] / "private_overviews").glob("*.png"))), 1)
            with self.assertRaises(FileExistsError):
                verifier.run_verifier(
                    fixture["selection"], fixture["triage"], fixture["frozen"], fixture["index"], fixture["raw"], fixture["output"],
                    expected_selection_sha256=fixture["selection_sha"], expected_triage_sha256=fixture["triage_sha"],
                    expected_frozen_metadata_sha256=fixture["frozen_sha"], expected_event_index_manifest_sha256=fixture["manifest_sha"],
                    expected_release_sha256=fixture["release"], candidate_ids={fixture["candidate_id"]}, av_backend=_FakeAV(),
                )

    def test_missing_camera_bytes_fail_closed_before_decoder_and_preserve_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._make_fixture(root)
            (fixture["raw"] / "videos/observation.rgb.right_realsense_link_camera_0/chunk-000/file-000.mp4").unlink()
            calls = {"count": 0}

            class CountingAV(_FakeAV):
                def open(self, path: str) -> _FakeContainer:
                    calls["count"] += 1
                    return super().open(path)

            with patch.object(verifier, "TRIAGE_CANDIDATE_COUNT", 1), patch.object(verifier, "FROZEN_ROW_COUNT", 1):
                with self.assertRaises(verifier.InputValidationError):
                    verifier.run_verifier(
                        fixture["selection"], fixture["triage"], fixture["frozen"], fixture["index"], fixture["raw"], fixture["output"],
                        expected_selection_sha256=fixture["selection_sha"], expected_triage_sha256=fixture["triage_sha"],
                        expected_frozen_metadata_sha256=fixture["frozen_sha"], expected_event_index_manifest_sha256=fixture["manifest_sha"],
                        expected_release_sha256=fixture["release"], candidate_ids={fixture["candidate_id"]}, av_backend=CountingAV(),
                    )
            self.assertEqual(calls["count"], 0)
            self.assertFalse(fixture["output"].exists())

    def test_metadata_preflight_can_scan_all_pinned_candidates_without_raw_root(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._make_fixture(root)
            with patch.object(verifier, "TRIAGE_CANDIDATE_COUNT", 1), patch.object(verifier, "FROZEN_ROW_COUNT", 1):
                result = verifier.metadata_preflight(
                    fixture["selection"], fixture["triage"], fixture["frozen"], fixture["index"],
                    expected_selection_sha256=fixture["selection_sha"], expected_triage_sha256=fixture["triage_sha"],
                    expected_frozen_metadata_sha256=fixture["frozen_sha"], expected_event_index_manifest_sha256=fixture["manifest_sha"],
                    expected_release_sha256=fixture["release"], all_triage_candidates=True,
                )
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["event_join_count"], 2)
            self.assertFalse(result["video_decoded"])

    def test_frozen_loader_rejects_two_selected_pairs_from_one_episode(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._make_fixture(root)
            candidates = {
                "pair-a": {"candidate_id": "pair-a", "episode_index": 1},
                "pair-b": {"candidate_id": "pair-b", "episode_index": 1},
            }
            with self.assertRaises(verifier.InputValidationError):
                verifier.load_frozen_rows(fixture["frozen"], fixture["frozen_sha"], candidates, fixture["release"])

    def test_control_role_requires_exact_frozen_intervening_target_switch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._make_fixture(root)
            candidate = json.loads(fixture["triage"].read_text().splitlines()[1])
            record = json.loads(fixture["frozen"].read_text())
            intermediate = {
                "arm": "UNSPECIFIED", "destination": "", "skill_end": 5, "skill_id": 1,
                "skill_idx": 2, "skill_start": 4, "source": "", "target": "box",
                "target_part": "", "verb": "NAVIGATE",
            }
            candidate["intervening_skills"] = [intermediate]
            record["segments"][0]["skills"].append(intermediate)
            frozen = verifier._validate_frozen_row(record, candidate, fixture["release"])
            selection = {
                "candidate_id": candidate["candidate_id"], "window_local": [0, 4],
                "review_role": "metadata_target_switch_control",
            }
            with self.assertRaises(verifier.InputValidationError):
                verifier._validate_selected_pairs([selection], {candidate["candidate_id"]: candidate}, {1: frozen})

            switched = {**intermediate, "target": "other_box"}
            candidate["intervening_skills"] = [switched]
            record["segments"][0]["skills"][-1] = switched
            frozen = verifier._validate_frozen_row(record, candidate, fixture["release"])
            normalized = verifier._validate_selected_pairs([selection], {candidate["candidate_id"]: candidate}, {1: frozen})
            self.assertEqual(len(normalized), 1)

    def test_sample_plan_keeps_half_open_end_minus_one_and_rejects_full_span_drift(self) -> None:
        common = {"skill_idx": 0, "skill_id": 2, "verb": "GRASP", "target": "box", "target_part": "", "arm": "UNSPECIFIED", "source": "table", "destination": ""}
        candidate = {"candidate_id": "x", "first_grasp": {**common, "skill_start": 10, "skill_end": 20}, "second_grasp": {**common, "skill_start": 30, "skill_end": 40}}
        selection = {"candidate_id": "x", "window_local": [10, 40]}
        plan = verifier._sample_plan(candidate, selection, 50)
        frames = {row["local_frame"] for row in plan}
        self.assertTrue({10, 19, 30, 39}.issubset(frames))
        with self.assertRaises(verifier.InputValidationError):
            verifier._sample_plan(candidate, {"candidate_id": "x", "window_local": [0, 50]}, 50)

    def test_native_pts_rejects_a_request_in_the_adjacent_episode(self) -> None:
        renderer = verifier._renderer()
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "camera.mp4"
            video.write_bytes(b"tiny video")
            with self.assertRaises(ValueError):
                renderer._decode_rgb(
                    _FakeAV(), video, 100.0 + 8 / 30.0,
                    episode_start_timestamp_s=100.0, episode_end_timestamp_s=100.0 + 7 / 30.0,
                    actor_anchor_timestamp_s=None,
                )


if __name__ == "__main__":
    unittest.main()
