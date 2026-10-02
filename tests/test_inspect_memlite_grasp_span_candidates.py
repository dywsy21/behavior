from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.data.inspect_memlite_grasp_span_candidates import (
    INPUT_SCHEMA,
    SINGLE_INPUT_SCHEMA,
    InputValidationError,
    RELEASE_SHA,
    build_manifest,
    build_single_manifest,
    load_single_selection,
    scan_grasp_span_rows,
    validate_single_manifest,
    validate_manifest,
)


def _rows(left: list[float], right: list[float], *, episode: int = 10133, start: int = 1000) -> list[dict]:
    assert len(left) == len(right)
    result = []
    for frame, (left_value, right_value) in enumerate(zip(left, right)):
        action = [0.0] * 23
        action[14] = left_value
        action[22] = right_value
        result.append(
            {
                "episode_index": episode,
                "task_index": 50,
                "frame_index": frame,
                "index": start + frame,
                "timestamp": (start + frame) / 30.0,
                "action": action,
                "observation.state": [0.0] * 61,
            }
        )
    return result


def _entry(spans: list[list[int]] | None = None) -> dict:
    spans = spans or [[2, 5], [6, 9]]
    return {
        "candidate_id": "ep10133-target-tupperware-grasp-23-to-24",
        "episode_index": 10133,
        "raw_episode_id": 501980,
        "task_index": 50,
        "task_instance_id": 198,
        "source_group_id": "group-10133",
        "source_annotation_sha256": "a" * 64,
        "length": 10,
        "dataset_from_index": 1000,
        "dataset_to_index": 1010,
        "immutable_split": "train",
        "usage_role": "student_candidate",
        "private_only": True,
        "training_eligible": False,
        "action_supervision": False,
        "action_bc_supervision": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "dart_supervision": False,
        "attempt_status": "NOT_APPLICABLE",
        "parquet_relative_path": "data/chunk-050/file-007.parquet",
        "grasp_spans": [
            {"span_id": "first_grasp", "frame_interval": spans[0], "skill": {"verb": "GRASP", "skill_start": spans[0][0], "skill_end": spans[0][1]}},
            {"span_id": "second_grasp", "frame_interval": spans[1], "skill": {"verb": "GRASP", "skill_start": spans[1][0], "skill_end": spans[1][1]}},
        ],
        "event_bindings": {
            "first_grasp": {"event_id": "event-first", "event_interval": {"start_frame": spans[0][0], "end_frame": spans[0][1]}},
            "second_grasp": {"event_id": "event-second", "event_interval": {"start_frame": spans[1][0], "end_frame": spans[1][1]}},
        },
    }


def _manifest(entry: dict) -> dict:
    return {
        "schema_version": INPUT_SCHEMA,
        "release_manifest_sha256": RELEASE_SHA,
        "official_snapshot_root": "/sealed/root",
        "official_info": {"path": "/sealed/info.json", "sha256": "b" * 64},
        "frozen_source": {"path": "/sealed/episodes.jsonl", "sha256": "c" * 64},
        "triage_source": {"path": "/sealed/triage.jsonl", "sha256": "d" * 64},
        "event_index_source": {"path": "/sealed/events.jsonl", "sha256": "e" * 64},
        "constraints": {
            "read_only": True,
            "grasp_spans_only": True,
            "raw_gripper_values_unnamed": True,
            "private_only": True,
            "cross_span_join_forbidden": True,
            "action_bc_supervision": False,
            "outcome_supervision": False,
            "recovery_supervision": False,
            "dart_supervision": False,
        },
        "episodes": [entry],
    }


def _single_entry() -> dict:
    entry = _entry([[2, 5], [6, 9]])
    span = entry["grasp_spans"][0]
    span.update(
        {
            "span_id": "single-selection-1",
            "segment_index": 0,
            "parent": "Task goal: test",
            "parent_supervised": False,
            "semantic": "[]",
            "text": "test",
        }
    )
    span["skill"].update(
        {
            "skill_id": 2,
            "skill_idx": 0,
            "target": "toy_1",
            "source": "table_1",
            "raw_relation": {"object_id": [["toy_1", "table_1"]]},
            "binding_confidence": "BOUND",
            "arm": "UNSPECIFIED",
        }
    )
    entry.update(
        {
            "selection_id": "single-selection-1",
            "candidate_id": "single-selection-1",
            "candidate_kind": "single_grasp_span",
            "source_group_id": "g" * 64,
            "annotation_relative_path": "annotations/task-0001/episode_00010001.json",
            "parent_goal": "Task goal: test",
            "outcome_status": "NOT_APPLICABLE",
            "recovery_status": "NOT_APPLICABLE",
            "release_status": "NOT_RELEASED",
            "camera_clock": {
                camera: {
                    "chunk_index": 1,
                    "file_index": 0,
                    "from_timestamp_s": 0.0,
                    "to_timestamp_s": 1.0,
                }
                for camera in (
                    "observation.rgb.zed_link_camera_0",
                    "observation.rgb.left_realsense_link_camera_0",
                    "observation.rgb.right_realsense_link_camera_0",
                )
            },
            "grasp_spans": [span],
            "event_bindings": {
                "single-selection-1": {
                    "event_id": "event-single",
                    "event_interval": {"start_frame": 2, "end_frame": 5},
                }
            },
        }
    )
    return entry


def _single_manifest(entry: dict) -> dict:
    return {
        "schema_version": SINGLE_INPUT_SCHEMA,
        "status": "AUTHENTICATED_PRIVATE_SINGLE_GRASP_ACTION_SCAN_INPUT",
        "release_manifest_sha256": RELEASE_SHA,
        "official_snapshot_root": "/sealed/root",
        "official_info": {"path": "/sealed/info.json", "sha256": "b" * 64},
        "frozen_source": {"path": "/sealed/episodes.jsonl", "sha256": "c" * 64},
        "source_groups_source": {"path": "/sealed/source_groups.jsonl", "sha256": "d" * 64, "schema": "memlite-event-index-v1"},
        "single_selection_source": {"path": "/sealed/selection.json", "sha256": "e" * 64, "schema": "p107-single-grasp-span-selection-v1"},
        "event_index_source": {"path": "/sealed/events.jsonl", "sha256": "f" * 64, "schema": "memlite-event-index-v1"},
        "constraints": {
            "read_only": True,
            "split": "train",
            "usage_role": "student_candidate",
            "max_episode_count": 8,
            "selected_episode_count": 1,
            "selected_span_count": 1,
            "single_grasp_span_per_episode": True,
            "grasp_spans_only": True,
            "raw_gripper_values_unnamed": True,
            "private_only": True,
            "cross_span_join_forbidden": True,
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
        "episodes": [entry],
    }


def test_grasp_span_keeps_episode_global_frame_indices_and_raw_values():
    rows = _rows([0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 10)
    result = scan_grasp_span_rows(rows, _entry())

    first = result["grasp_spans"][0]
    left = first["channels"]["left"]
    assert [run["start_frame"] for run in left["runs"]] == [2, 3, 4]
    assert left["runs"][0]["value"] == 1.0
    assert left["runs"][1]["value"] == -1.0
    assert left["runs"][2]["value"] == 1.0
    assert result["candidates"][0]["frame_span"] == [2, 5]
    assert result["candidates"][0]["source_identity"]["event_id"] == "event-first"


def test_spans_and_arms_are_independent_and_no_cross_span_candidate_is_created():
    rows = _rows(
        [0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0],
    )
    result = scan_grasp_span_rows(rows, _entry())

    assert result["candidate_count"] == 2
    assert {candidate["source_identity"]["event_id"] for candidate in result["candidates"]} == {
        "event-first",
        "event-second",
    }
    assert {candidate["candidate_id"].split(":")[2] for candidate in result["candidates"]} == {"left", "right"}
    assert all(candidate["frame_span"] in ([2, 5], [6, 9]) for candidate in result["candidates"])
    assert result["private_only"] is True
    assert result["training_eligible"] is False
    assert result["outcome_supervision"] is False
    assert result["recovery_supervision"] is False


def test_short_span_and_end_boundary_are_retained_without_resetting_clock():
    entry = _entry([[0, 3], [7, 10]])
    rows = _rows([1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0], [0.0] * 10)
    result = scan_grasp_span_rows(rows, entry)

    assert result["grasp_spans"][1]["channels"]["left"]["runs"][0]["start_frame"] == 7
    assert result["grasp_spans"][1]["channels"]["left"]["runs"][-1]["end_frame"] == 10
    assert all(candidate["frame_span"][1] <= 10 for candidate in result["candidates"])


def test_manifest_rejects_calibration_role_and_duplicate_source():
    manifest = _manifest(_entry())
    manifest["episodes"][0]["usage_role"] = "annotation_calibration"
    with pytest.raises(InputValidationError, match="role"):
        validate_manifest(manifest)

    bad = copy.deepcopy(manifest)
    bad["episodes"][0]["usage_role"] = "student_candidate"
    duplicate = copy.deepcopy(bad["episodes"][0])
    bad["episodes"].append(duplicate)
    with pytest.raises(InputValidationError, match="duplicate"):
        validate_manifest(bad)


def test_manifest_rejects_span_outside_episode_and_path_escape():
    manifest = _manifest(_entry([[2, 5], [6, 11]]))
    with pytest.raises(InputValidationError, match="outside episode"):
        validate_manifest(manifest)

    manifest = _manifest(_entry())
    manifest["episodes"][0]["parquet_relative_path"] = "../escape.parquet"
    with pytest.raises(InputValidationError, match="safe relative path"):
        validate_manifest(manifest)


def test_manifest_rejects_duplicate_span_id_and_reversed_or_overlapping_spans():
    manifest = _manifest(_entry())
    manifest["episodes"][0]["grasp_spans"][1]["span_id"] = "first_grasp"
    with pytest.raises(InputValidationError, match="duplicate span_id"):
        validate_manifest(manifest)

    manifest = _manifest(_entry())
    second = manifest["episodes"][0]["grasp_spans"][1]
    second["frame_interval"] = [4, 9]
    second["skill"]["skill_start"] = 4
    second["skill"]["skill_end"] = 9
    second_binding = manifest["episodes"][0]["event_bindings"]["second_grasp"]
    second_binding["event_interval"] = {"start_frame": 4, "end_frame": 9}
    with pytest.raises(InputValidationError, match="overlap or are reversed"):
        validate_manifest(manifest)


def test_manifest_rejects_event_interval_drift_from_named_span():
    manifest = _manifest(_entry())
    manifest["episodes"][0]["event_bindings"]["second_grasp"]["event_interval"]["start_frame"] = 5
    with pytest.raises(InputValidationError, match="event interval mismatch"):
        validate_manifest(manifest)


def test_row_clock_mismatch_fails_before_private_result():
    rows = _rows([1.0, -1.0, 1.0] + [0.0] * 7, [0.0] * 10)
    rows[2]["index"] += 1
    with pytest.raises(InputValidationError, match="frame_index/index mismatch"):
        scan_grasp_span_rows(rows, _entry())


def test_single_manifest_accepts_one_span_and_rejects_a_second_span():
    manifest = _single_manifest(_single_entry())
    assert len(validate_single_manifest(manifest)) == 1

    bad = copy.deepcopy(manifest)
    extra = copy.deepcopy(bad["episodes"][0]["grasp_spans"][0])
    extra["span_id"] = "single-selection-2"
    bad["episodes"][0]["grasp_spans"].append(extra)
    bad["episodes"][0]["event_bindings"]["single-selection-2"] = copy.deepcopy(
        bad["episodes"][0]["event_bindings"]["single-selection-1"]
    )
    with pytest.raises(InputValidationError, match="exactly one GRASP span"):
        validate_single_manifest(bad)


def test_single_scan_reuses_raw_rle_without_cross_span_join():
    entry = _single_entry()
    rows = _rows([0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 10)
    result = scan_grasp_span_rows(rows, entry)
    assert len(result["grasp_spans"]) == 1
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["source_identity"]["event_id"] == "event-single"
    assert result["private_only"] is True
    assert result["training_eligible"] is False
    assert result["outcome_supervision"] is False
    assert result["recovery_supervision"] is False


def test_sealed_single_selection_list_is_exactly_eight_bound_spans():
    path = Path(__file__).parents[1] / "scripts/data/p107_single_grasp_selection_v1.json"
    if not path.is_file():
        pytest.skip("single-GRASP selection list is unavailable")
    _, selections = load_single_selection(
        path,
        "e3edcf21e5748d7e3df0bc2c577449a0fec63dcca0535cfbb1e987d4ca3ea2e2",
    )
    assert len(selections) == 8
    assert len({item["episode_index"] for item in selections}) == 8
    assert all(item["verb"] == "GRASP" and item["arm"] == "UNSPECIFIED" and item["binding_confidence"] == "BOUND" for item in selections)


def _real_build_kwargs(tmp_path):
    root = Path("/home/wsy/behavior-annotations/p107")
    return {
        "triage_path": root / "natural-retry-metadata-triage-v2/natural_retry_triage_v2.jsonl",
        "frozen_path": root / "frozen-v4-metadata/episodes.jsonl",
        "event_index_path": root / "index-validation/full-v3-candidate-index/event_candidates.jsonl",
        "info_path": root / "natural-action-probe-v1/official-metadata/info.json",
        "expected_triage_sha256": "242086ce43147a711b73ac8da2762fbb7030db10c6ad0bb75df44e651956c25c",
        "expected_frozen_sha256": "c62fe885143bcdc07a9dcb302a5af294afb355db98d078a838c587f9efcc16ca",
        "expected_event_index_sha256": "c12bfa8ba9375b208525a25eb2c1b3944ef5c980a021599c174e18b186886329",
        "expected_info_sha256": "24c77f7a984bcee775e666203881a946b11a899f524fbc2405922b2109757874",
        "source_root": "/data/workspace/wsy/behavior2026/datasets/2026-challenge-demos/datasets/fduTristin--2026-challenge-demos/snapshots/master",
    }


def _real_single_build_kwargs():
    root = Path("/home/wsy/behavior-annotations/p107")
    return {
        "selection_path": Path(__file__).parents[1] / "scripts/data/p107_single_grasp_selection_v1.json",
        "frozen_path": root / "frozen-v4-metadata/episodes.jsonl",
        "source_groups_path": root / "index-validation/full-v3-candidate-index/source_groups.jsonl",
        "event_index_path": root / "index-validation/full-v3-candidate-index/event_candidates.jsonl",
        "info_path": root / "natural-action-probe-v1/official-metadata/info.json",
        "expected_selection_sha256": "e3edcf21e5748d7e3df0bc2c577449a0fec63dcca0535cfbb1e987d4ca3ea2e2",
        "expected_frozen_sha256": "c62fe885143bcdc07a9dcb302a5af294afb355db98d078a838c587f9efcc16ca",
        "expected_source_groups_sha256": "cdd1c37db3f7af3a743ad25b1a6fc8c4b0da41299ecf3bd8d10b9645193673e3",
        "expected_event_index_sha256": "c12bfa8ba9375b208525a25eb2c1b3944ef5c980a021599c174e18b186886329",
        "expected_info_sha256": "24c77f7a984bcee775e666203881a946b11a899f524fbc2405922b2109757874",
        "source_root": "/data/workspace/wsy/behavior2026/datasets/2026-challenge-demos/datasets/fduTristin--2026-challenge-demos/snapshots/master",
    }


def test_real_four_manifest_builder_roundtrip_is_validated_before_publish(tmp_path):
    kwargs = _real_build_kwargs(tmp_path)
    if not all(path.is_file() for path in (kwargs["triage_path"], kwargs["frozen_path"], kwargs["event_index_path"], kwargs["info_path"])):
        pytest.skip("sealed local metadata bundle is unavailable")
    output = tmp_path / "four.json"
    manifest = build_manifest(
        **kwargs,
        output_path=output,
        candidate_ids=[
            "ep10133-target-tupperware-grasp-23-to-24",
            "ep10925-target-toy_figure-grasp-1-to-2",
            "ep04471-target-gym_shoe_78-grasp-1-to-3",
            "ep04431-target-gym_shoe_78-grasp-1-to-3",
        ],
    )
    assert output.is_file()
    assert len(validate_manifest(manifest)) == 4
    assert len(validate_manifest(json.loads(output.read_text()))) == 4


def test_real_single_manifest_builder_roundtrip_is_validated_before_publish(tmp_path):
    kwargs = _real_single_build_kwargs()
    if not all(path.is_file() for path in (kwargs["selection_path"], kwargs["frozen_path"], kwargs["source_groups_path"], kwargs["event_index_path"], kwargs["info_path"])):
        pytest.skip("sealed single-GRASP metadata bundle is unavailable")
    output = tmp_path / "single.json"
    manifest = build_single_manifest(**kwargs, output_path=output)
    assert output.is_file()
    assert len(validate_single_manifest(manifest)) == 8
    assert len(validate_single_manifest(json.loads(output.read_text()))) == 8


def test_builder_duplicate_selected_id_fails_without_publishing(tmp_path):
    kwargs = _real_build_kwargs(tmp_path)
    if not all(path.is_file() for path in (kwargs["triage_path"], kwargs["frozen_path"], kwargs["event_index_path"], kwargs["info_path"])):
        pytest.skip("sealed local metadata bundle is unavailable")
    output = tmp_path / "duplicate.json"
    with pytest.raises(InputValidationError, match="unique"):
        build_manifest(
            **kwargs,
            output_path=output,
            candidate_ids=[
                "ep10133-target-tupperware-grasp-23-to-24",
                "ep10133-target-tupperware-grasp-23-to-24",
            ],
        )
    assert not output.exists()
