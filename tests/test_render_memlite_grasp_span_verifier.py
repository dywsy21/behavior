from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "data" / "render_memlite_grasp_span_verifier.py"
SPEC = importlib.util.spec_from_file_location("private_single_grasp_verifier_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.write_bytes((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return _sha(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_bytes(
        b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    )
    return _sha(path)


class _FakeFrame:
    def __init__(self, pts: int, colour: tuple[int, int, int]) -> None:
        self.pts = pts
        self.colour = colour

    def to_image(self) -> Image.Image:
        return Image.new("RGB", (18, 12), self.colour)


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
        # The neutral decoder seeks one half-frame before a requested tick.
        # Emit the preceding and exact ticks so PTS selection is exercised
        # without installing PyAV or reading a real video.
        yield _FakeFrame(self.target, (17, 71, 131))
        yield _FakeFrame(self.target + 1, (41, 101, 53))


class _FakeAV:
    def open(self, path: str) -> _FakeContainer:
        return _FakeContainer(Path(path))


class SingleGraspVerifierTests(unittest.TestCase):
    def _fixture(self) -> dict[str, object]:
        root = Path(tempfile.mkdtemp(prefix="single-grasp-verifier-test-"))
        raw_root = root / "raw"
        raw_root.mkdir()
        release = "a" * 64
        annotation = "b" * 64
        length = 1000
        start, end = 100, 500
        source = {
            "source_release_manifest_sha256": release,
            "source_annotation_sha256": annotation,
            "task_index": 3,
            "task_instance_id": 4,
            "raw_episode_id": 70,
            "episode_index": 7,
            "original_split": "train",
            "episode_length": length,
        }
        source["source_group_id"] = verifier._PROTOCOL.source_group_id(source)
        clocks: dict[str, dict[str, object]] = {}
        locators: list[dict[str, object]] = []
        for view in verifier.VIEWS:
            camera = verifier.CAMERA_BY_VIEW[view]
            relative = f"videos/{camera}/chunk-000/file-000.mp4"
            video = raw_root / relative
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"synthetic tinyvideo placeholder")
            camera_start = 1000.0
            camera_end = camera_start + length / verifier.FPS
            clocks[camera] = {
                "chunk_index": 0,
                "file_index": 0,
                "from_timestamp_s": camera_start,
                "to_timestamp_s": camera_end,
            }
            locators.append(
                {
                    "camera_key": camera,
                    "episode_start_timestamp_s": camera_start,
                    "expected_fps": 30,
                    "locator_status": "METADATA_ONLY_UNRESOLVED",
                    "relative_path": relative,
                    "requested_timestamp_s": camera_start + start / verifier.FPS,
                    "view": view,
                }
            )
        skill = {
            "arm": "UNSPECIFIED",
            "binding_confidence": "BOUND",
            "destination": "",
            "interval_id": 0,
            "raw_description": "pick up from",
            "raw_relation": {
                "manipulating_object_id": ["object_1"],
                "memory_prefix": [],
                "object_id": [["object_1", "table_0"]],
                "spatial_prefix": [],
            },
            "skill_end": end,
            "skill_id": 2,
            "skill_idx": 8,
            "skill_start": start,
            "source": "table_0",
            "target": "object_1",
            "target_part": "",
            "verb": "GRASP",
        }
        action = {
            "actual_executed_length": None,
            "model_action_dim": 27,
            "model_padding_indices": [7, 8, 17, 18],
            "raw_action_dim": 23,
            "start_frame": start,
        }
        event: dict[str, object] = {
            "schema_version": verifier.EVENT_SCHEMA,
            "record_kind": "event_candidate",
            "event_id": "",
            "source": source,
            "event_kind": "ANNOTATED_SKILL_SEGMENT",
            "event_interval": {"start_frame": start, "end_frame": end},
            "observation": {"frame": start, "timestamp_s": start / verifier.FPS},
            "action": action,
            "bundle_id": "bundle-single",
            "skill_bundle": [skill],
            "parallel_bundle": False,
            "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
            "video_locators": locators,
            "usage_role": "student_candidate",
        }
        event["event_id"] = verifier._PROTOCOL.event_id(event)
        verifier._PROTOCOL.validate_event(event)
        base_candidate_id = "ep00007-grasp-8-100-500"
        candidate = {
            "candidate_id": f"{base_candidate_id}:right:aba-0000",
            "command_structure": "A_B_A_UNNAMED",
            "frame_span": [start, end],
            "run_indices": [0, 1, 2],
            "run_intervals": [
                {"action_index": 22, "channel": "right", "dwell_frames": 100, "end_frame": 200, "run_index": 0, "start_frame": 100, "value": 1.0},
                {"action_index": 22, "channel": "right", "dwell_frames": 100, "end_frame": 300, "run_index": 1, "start_frame": 200, "value": -1.0},
                {"action_index": 22, "channel": "right", "dwell_frames": 200, "end_frame": 500, "run_index": 2, "start_frame": 300, "value": 1.0},
            ],
            "semantic_interpretation": "NOT_ASSIGNED",
            "source_identity": {
                "candidate_id": base_candidate_id,
                "episode_index": source["episode_index"],
                "event_id": event["event_id"],
                "frame_interval": [start, end],
                "source_group_id": source["source_group_id"],
            },
            "values": [1.0, -1.0, 1.0],
        }
        event_binding = {
            "action_contract": action,
            "bundle_id": event["bundle_id"],
            "event_id": event["event_id"],
            "event_interval": event["event_interval"],
            "event_kind": event["event_kind"],
            "observation": event["observation"],
            "parallel_bundle": event["parallel_bundle"],
            "schema_version": event["schema_version"],
            "skill_bundle": event["skill_bundle"],
            "usage_role": event["usage_role"],
            "video_locators": event["video_locators"],
        }
        span = {"span_id": base_candidate_id, "frame_interval": [start, end], "event_binding": event_binding}
        result_episode = {
            "length": length,
            "candidate_id": base_candidate_id,
            "candidates": [candidate],
            "grasp_spans": [span],
            "source_identity": {
                "episode_index": source["episode_index"],
                "raw_episode_id": source["raw_episode_id"],
                "source_annotation_sha256": source["source_annotation_sha256"],
                "source_group_id": source["source_group_id"],
                "task_index": source["task_index"],
                "task_instance_id": source["task_instance_id"],
            },
        }
        event_dir = root / "event-index"
        event_dir.mkdir()
        event_path = event_dir / "event_candidates.jsonl"
        event_candidates_sha = _write_jsonl(event_path, [event])
        event_receipt = {"sha256": event_candidates_sha, "rows": 1, "bytes": event_path.stat().st_size}
        event_manifest = {
            "schema_version": verifier.EVENT_INDEX_SCHEMA,
            "source_release_manifest_sha256": release,
            "files": {"event_candidates.jsonl": event_receipt},
        }
        event_manifest_path = event_dir / "manifest.json"
        event_manifest_sha = _write_json(event_manifest_path, event_manifest)
        manifest_entry = {
            "candidate_id": base_candidate_id,
            "selection_id": "single_00007",
            "episode_index": source["episode_index"],
            "raw_episode_id": source["raw_episode_id"],
            "task_index": source["task_index"],
            "task_instance_id": source["task_instance_id"],
            "source_group_id": source["source_group_id"],
            "source_annotation_sha256": source["source_annotation_sha256"],
            "length": length,
            "dataset_from_index": 10000,
            "usage_role": "student_candidate",
            "immutable_split": "train",
            "private_only": True,
            "training_eligible": False,
            "action_supervision": False,
            "action_bc_supervision": False,
            "outcome_supervision": False,
            "recovery_supervision": False,
            "dart_supervision": False,
            "attempt_status": "NOT_APPLICABLE",
            "release_status": "NOT_RELEASED",
            "grasp_spans": [{"frame_interval": [start, end]}],
            "event_bindings": {"single_00007": event_binding},
            "camera_clock": clocks,
        }
        manifest = {
            "schema_version": verifier.MANIFEST_SCHEMA,
            "status": "AUTHENTICATED_PRIVATE_SINGLE_GRASP_ACTION_SCAN_INPUT",
            "release_manifest_sha256": release,
            "constraints": {
                "read_only": True,
                "split": "train",
                "usage_role": "student_candidate",
                "single_grasp_span_per_episode": True,
                "private_only": True,
                "training_eligible": False,
                "action_supervision": False,
                "action_bc_supervision": False,
                "outcome_supervision": False,
                "recovery_supervision": False,
                "dart_supervision": False,
                "attempt_status": "NOT_APPLICABLE",
                "outcome_status": "NOT_APPLICABLE",
                "recovery_status": "NOT_APPLICABLE",
                "release_status": "NOT_RELEASED",
            },
            "event_index_source": {
                "schema": verifier.EVENT_INDEX_SCHEMA,
                "sha256": event_candidates_sha,
            },
            "episodes": [manifest_entry],
        }
        manifest_path = root / "single-manifest.json"
        manifest_sha = _write_json(manifest_path, manifest)
        result = {
            "schema_version": verifier.RESULT_SCHEMA,
            "status": "PASS_STRUCTURAL_GRASP_SPAN_SCAN_NONTRAINABLE",
            "role": "diagnostic_candidate",
            "private_only": True,
            "training_eligible": False,
            "action_bc_supervision": False,
            "outcome_supervision": False,
            "recovery_supervision": False,
            "dart_supervision": False,
            "attempt_status": "NOT_APPLICABLE",
            "release_status": "NOT_RELEASED",
            "inputs": {"manifest_sha256": manifest_sha, "release_manifest_sha256": release},
            "episodes": [result_episode],
        }
        result_path = root / "structural-result.json"
        result_sha = _write_json(result_path, result)
        frames = [100, 170, 199, 200, 230, 260, 290, 299, 300, 320, 350, 499]
        selection = {
            "schema_version": verifier.SELECTION_SCHEMA,
            "status": "PRIVATE_SINGLE_GRASP_RGB_SELECTION_UNDECODED",
            "guards": {
                "private_verifier_only": True,
                "actor_packet_included": False,
                "training_eligible": False,
                "action_bc_supervision": False,
                "outcome_supervision": False,
                "recovery_supervision": False,
                "dart_supervision": False,
                "attempt_status": "NOT_APPLICABLE",
                "release_status": "NOT_RELEASED",
                "do_not_assign_command_polarity": True,
                "no_outcome_inference": True,
                "no_success_failure_recovery_inference": True,
                "native_camera_rgb_only": True,
            },
            "source_result": {"schema_version": verifier.RESULT_SCHEMA, "sha256": result_sha},
            "single_manifest": {"schema_version": verifier.MANIFEST_SCHEMA, "sha256": manifest_sha},
            "event_index": {
                "schema_version": verifier.EVENT_INDEX_SCHEMA,
                "manifest_sha256": event_manifest_sha,
                "event_candidates_sha256": event_candidates_sha,
            },
            "selected": {
                "selection_id": "single_00007",
                "candidate_id": base_candidate_id,
                "event_id": event["event_id"],
                "episode_index": source["episode_index"],
                "source_group_id": source["source_group_id"],
                "frame_interval": [start, end],
            },
            "frame_plan": {
                "local_frames": frames,
                "windows": [
                    {"frame_interval": [100, 101], "stride_frames": 1},
                    {"frame_interval": [170, 361], "stride_frames": 30},
                    {"frame_interval": [499, 500], "stride_frames": 1},
                ],
                "required_frames": [199, 200, 299, 300],
                "span_endpoints": [100, 499],
                "transition_frames": [199, 200, 299, 300],
                "middle_sample_frames": [230, 260, 290],
                "stride_frames": 30,
            },
        }
        selection_path = root / "selection.json"
        selection_sha = _write_json(selection_path, selection)
        return {
            "root": root,
            "raw_root": raw_root,
            "event_dir": event_dir,
            "event_manifest_sha": event_manifest_sha,
            "event_candidates_sha": event_candidates_sha,
            "manifest_path": manifest_path,
            "manifest_sha": manifest_sha,
            "result_path": result_path,
            "result_sha": result_sha,
            "selection_path": selection_path,
            "selection_sha": selection_sha,
            "selection": selection,
            "result": result,
            "manifest": manifest,
            "event": event,
        }

    def _run(self, fixture: dict[str, object], output: Path | None = None) -> dict[str, object]:
        if output is None:
            root = fixture["root"]
            output = root.parent / f"{root.name}-output"
        return verifier.run_verifier(
            fixture["selection_path"],
            fixture["result_path"],
            fixture["manifest_path"],
            fixture["event_dir"],
            fixture["raw_root"],
            output,
            expected_selection_sha256=fixture["selection_sha"],
            expected_source_result_sha256=fixture["result_sha"],
            expected_manifest_sha256=fixture["manifest_sha"],
            expected_event_index_manifest_sha256=fixture["event_manifest_sha"],
            expected_event_candidates_sha256=fixture["event_candidates_sha"],
            av_backend=_FakeAV(),
        )

    def test_mock_decode_preserves_student_role_and_bounded_native_artifact(self) -> None:
        fixture = self._fixture()
        result = self._run(fixture)
        output = Path(result["output_path"])
        self.assertEqual(result["frame_count"], 12)
        self.assertEqual(result["native_png_count"], 36)
        self.assertLessEqual(result["artifact_bytes"], verifier.MAX_OUTPUT_BYTES)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["role"], "private_verifier_only")
        self.assertEqual(manifest["usage_role"], "student_candidate")
        self.assertFalse(manifest["training_eligible"])
        rows = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 36)
        self.assertTrue(all(row["usage_role"] == "student_candidate" for row in rows))
        self.assertTrue(all(row["actor_packet_included"] is False for row in rows))
        self.assertEqual(len(list((output / "native_rgb").rglob("*.png"))), 36)
        self.assertEqual(len(list((output / "private_verifier_overviews").glob("*.png"))), 3)

    def test_wrong_pins_are_rejected_before_decode(self) -> None:
        fixture = self._fixture()
        for kwargs in (
            {"expected_source_result_sha256": "0" * 64},
            {"expected_manifest_sha256": "0" * 64},
            {"expected_event_index_manifest_sha256": "0" * 64},
        ):
            arguments = {
                "expected_selection_sha256": fixture["selection_sha"],
                "expected_source_result_sha256": fixture["result_sha"],
                "expected_manifest_sha256": fixture["manifest_sha"],
                "expected_event_index_manifest_sha256": fixture["event_manifest_sha"],
                "expected_event_candidates_sha256": fixture["event_candidates_sha"],
            }
            arguments.update(kwargs)
            with self.assertRaises(verifier.InputValidationError):
                verifier.run_verifier(
                    fixture["selection_path"], fixture["result_path"], fixture["manifest_path"],
                    fixture["event_dir"], fixture["raw_root"], fixture["root"] / "rejected",
                    av_backend=_FakeAV(), **arguments,
                )

    def test_frame_plan_rejects_out_of_span_duplicate_and_budget(self) -> None:
        fixture = self._fixture()
        selection_info = verifier._load_selection(
            fixture["selection_path"], fixture["selection_sha"],
            source_result_path=fixture["result_path"], source_result_sha256=fixture["result_sha"],
            manifest_path=fixture["manifest_path"], manifest_sha256=fixture["manifest_sha"],
            event_index_manifest_sha256=fixture["event_manifest_sha"], event_candidates_sha256=fixture["event_candidates_sha"],
        )
        _result, _episode, candidate_info = verifier._validate_candidate_result(
            fixture["result_path"], fixture["result_sha"], selection_info,
        )
        for frames in (
            [99] + fixture["selection"]["frame_plan"]["local_frames"][1:],
            fixture["selection"]["frame_plan"]["local_frames"] + [499],
            list(range(100, 133)),
        ):
            bad = copy.deepcopy(fixture["selection"])
            bad["frame_plan"]["local_frames"] = frames
            with self.assertRaises(verifier.InputValidationError):
                verifier._validate_frame_plan(bad, candidate_info)

    def test_wrong_locator_and_unsafe_selection_name_are_rejected(self) -> None:
        fixture = self._fixture()
        bad_event = copy.deepcopy(fixture["event"])
        bad_event["video_locators"][0]["relative_path"] = "videos/../escape.mp4"
        entry = fixture["manifest"]["episodes"][0]
        with self.assertRaises(verifier.InputValidationError):
            verifier._validate_camera_locators(bad_event, entry, 1000)

        bad_selection = copy.deepcopy(fixture["selection"])
        bad_selection["selected"]["selection_id"] = "../unsafe"
        bad_path = fixture["root"] / "bad-selection.json"
        bad_sha = _write_json(bad_path, bad_selection)
        with self.assertRaises(verifier.InputValidationError):
            verifier._load_selection(
                bad_path, bad_sha,
                source_result_path=fixture["result_path"], source_result_sha256=fixture["result_sha"],
                manifest_path=fixture["manifest_path"], manifest_sha256=fixture["manifest_sha"],
                event_index_manifest_sha256=fixture["event_manifest_sha"], event_candidates_sha256=fixture["event_candidates_sha"],
            )

    def test_existing_output_is_not_overwritten(self) -> None:
        fixture = self._fixture()
        root = fixture["root"]
        output = root.parent / f"{root.name}-existing-output"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self._run(fixture, output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
