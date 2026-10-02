from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path

import pytest

from scripts.data.inspect_memlite_gripper_retry_candidates import (
    InputValidationError,
    build_rle_runs,
    find_reversal_candidates,
    load_selected_rows,
    parse_primitive_annotation,
    scan_episode_rows,
    validate_episode_rows,
    validate_manifest,
)


def _rows(
    left_values: list[float],
    right_values: list[float] | None = None,
    *,
    episode_index: int = 7,
    task_index: int = 3,
    index_start: int = 100,
) -> list[dict]:
    right_values = right_values if right_values is not None else [0.0] * len(left_values)
    assert len(left_values) == len(right_values)
    result = []
    for frame, (left, right) in enumerate(zip(left_values, right_values)):
        action = [0.0] * 23
        action[14] = left
        action[22] = right
        result.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "frame_index": frame,
                "index": index_start + frame,
                "timestamp": frame / 30.0,
                "action": action,
                "observation.state": [float(frame)] * 61,
            }
        )
    return result


def _entry(*, length: int, episode_index: int = 7, task_index: int = 3) -> dict:
    return {
        "event_id": "event-7",
        "selection_order": 0,
        "episode_index": episode_index,
        "raw_episode_id": 700,
        "task_index": task_index,
        "task_instance_id": 70,
        "source_group_id": "group-7",
        "length": length,
        "dataset_from_index": 100,
        "dataset_to_index": 100 + length,
        "parquet_relative_path": "data/chunk-000/file-000.parquet",
        "annotation_local_copy": "/tmp/annotation.json",
        "annotation_sha256": "a" * 64,
        "immutable_split": "train",
        "usage_role": "annotation_calibration",
        "training_eligible": False,
    }


def test_aba_in_first_and_last_runs_is_retained():
    rows = _rows([1.0, -1.0, 1.0])
    normalized = validate_episode_rows(rows, _entry(length=3))
    runs = build_rle_runs(normalized, "left", 14)
    candidates, exclusions, counts = find_reversal_candidates(runs)

    assert [run["start_frame"] for run in runs] == [0, 1, 2]
    assert candidates[0]["frame_span"] == [0, 3]
    assert candidates[0]["run_indices"] == [0, 1, 2]
    assert exclusions == []
    assert counts["structural_aba_matches"] == 1


def test_non_reversal_does_not_create_candidate():
    runs = build_rle_runs(validate_episode_rows(_rows([1.0, 1.0, -1.0, -1.0]), _entry(length=4)), "left", 14)
    candidates, exclusions, counts = find_reversal_candidates(runs)

    assert candidates == []
    assert exclusions == []
    assert counts["structural_aba_matches"] == 0


def test_left_and_right_channels_are_scanned_independently():
    rows = _rows([1.0, -1.0, 1.0], [1.0, 1.0, -1.0])
    result = scan_episode_rows(rows, _entry(length=3), parse_primitive_annotation(None, 3))

    assert len(result["channels"]["left"]["runs"]) == 3
    assert len(result["channels"]["right"]["runs"]) == 2
    assert [candidate["channel"] for candidate in result["candidates"]] == ["left"]


def test_short_dwell_is_kept_by_default_and_excluded_only_by_explicit_threshold():
    runs = build_rle_runs(validate_episode_rows(_rows([1.0, -1.0, 1.0]), _entry(length=3)), "left", 14)
    retained, no_exclusions, _ = find_reversal_candidates(runs)
    excluded, exclusions, _ = find_reversal_candidates(runs, min_dwell_frames=2)

    assert len(retained) == 1
    assert no_exclusions == []
    assert excluded == []
    assert exclusions[0]["reason"] == "dwell_below_threshold"
    assert exclusions[0]["min_dwell_frames"] == 2


def test_cross_primitive_boundary_is_retained_with_union_and_gaps():
    rows = _rows([1.0, -1.0, 1.0])
    annotation = {
        "primitive_annotation": [
            {"primitive_idx": 0, "frame_duration": [0, 1], "primitive_id": [1]},
            {"primitive_idx": 1, "frame_duration": [1, 2], "primitive_id": [2]},
            {"primitive_idx": 2, "frame_duration": [2, 3], "primitive_id": [3]},
        ]
    }
    result = scan_episode_rows(
        rows,
        _entry(length=3),
        parse_primitive_annotation(annotation, 3),
    )

    private = result["candidates"][0]["private_annotation_audit"]
    assert private["classification"] == "cross_primitive_boundary"
    assert private["covered_frames"] == 3
    assert private["gaps_within_candidate"] == []
    assert [item["raw_frame_duration"] for item in private["raw_primitive_intersections"]] == [
        [0, 1],
        [1, 2],
        [2, 3],
    ]


def test_outer_context_can_cross_while_flip_core_is_same_primitive():
    rows = _rows([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    annotation = {
        "primitive_annotation": [
            {"primitive_idx": 0, "frame_duration": [0, 3], "primitive_id": [1]},
            {"primitive_idx": 1, "frame_duration": [3, 10], "primitive_id": [2]},
        ]
    }
    result = scan_episode_rows(
        rows,
        _entry(length=10),
        parse_primitive_annotation(annotation, 10),
    )
    candidate = result["candidates"][0]
    private = candidate["private_annotation_audit"]

    # Legacy v1 outer context is unchanged: [first A start, last A end).
    assert candidate["frame_span"] == [0, 10]
    assert private["candidate_span_frame_interval"] == [0, 10]
    assert private["classification"] == "cross_primitive_boundary"
    # The core includes the frame before the first flip, all B, and the frame
    # after the second flip, excluding long A-side context.
    assert candidate["flip_core_span"] == [3, 7]
    assert candidate["flip_core_relation"] == "same_primitive"
    assert private["outer_context_relation"] == "cross_primitive_boundary"
    assert private["flip_core_relation"] == "same_primitive"
    assert result["candidate_count"] == 1
    assert result["outer_context_relation_counts"] == {"cross_primitive_boundary": 1}
    assert result["flip_core_relation_counts"] == {"same_primitive": 1}
    assert candidate["semantic_interpretation"] == "NOT_ASSIGNED"


def test_flip_core_clips_cleanly_at_episode_start_and_end():
    annotation = {"primitive_annotation": [{"primitive_idx": 0, "frame_duration": [0, 4]}]}
    for values, expected in (([1.0, -1.0, 1.0, 1.0], [0, 3]), ([1.0, 1.0, -1.0, 1.0], [1, 4])):
        result = scan_episode_rows(
            _rows(values),
            _entry(length=4),
            parse_primitive_annotation(annotation, 4),
        )
        candidate = result["candidates"][0]
        assert candidate["flip_core_span"] == expected
        assert candidate["flip_core_relation"] == "same_primitive"


def test_partial_annotation_is_not_silently_treated_as_same_primitive():
    annotation = {"primitive_annotation": [{"frame_duration": [1, 2], "primitive_id": [4]}]}
    audit = parse_primitive_annotation(annotation, 3)
    rows = _rows([1.0, -1.0, 1.0])
    result = scan_episode_rows(rows, _entry(length=3), audit)
    private = result["candidates"][0]["private_annotation_audit"]

    assert audit["join_semantics"] == "join_semantics_unverified"
    assert private["classification"] == "partial_annotation_with_gaps"
    assert private["join_semantics_warning"]
    assert private["gaps_within_candidate"] == [[0, 1], [2, 3]]


def test_no_annotation_candidate_is_explicitly_classified():
    result = scan_episode_rows(_rows([1.0, -1.0, 1.0]), _entry(length=3), parse_primitive_annotation(None, 3))
    private = result["candidates"][0]["private_annotation_audit"]

    assert private["classification"] == "no_annotation"
    assert private["covered_frames"] == 0
    assert private["gaps_within_candidate"] == [[0, 3]]


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda rows: rows[1].update(frame_index=3), "frame_index"),
        (lambda rows: rows[1].update(action=[math.nan] + [0.0] * 22), "finite"),
        (lambda rows: rows[1].update(episode_index=99), "membership"),
        (lambda rows: rows[1].update(task_index=99), "task mismatch"),
    ],
)
def test_row_contract_rejects_mismatch_and_nonfinite_values(mutator, match):
    rows = _rows([1.0, -1.0, 1.0])
    mutator(rows)
    with pytest.raises(InputValidationError, match=match):
        validate_episode_rows(rows, _entry(length=3))


def test_annotation_frame_mismatch_is_rejected():
    with pytest.raises(InputValidationError, match="outside"):
        parse_primitive_annotation({"primitive_annotation": [{"frame_duration": [0, 4]}]}, 3)


def test_duplicate_source_identity_is_rejected():
    base = {
        "schema_version": "p107-natural-retry-pilot-input-v1",
        "frozen_source": {
            "required_split": "train",
            "required_usage_role": "annotation_calibration",
            "all_training_eligible": False,
        },
        "official_info": {},
        "constraints": {"read_only": True},
        "episodes": [],
    }
    first = _entry(length=3)
    second = copy.deepcopy(first)
    base["episodes"] = [first, second]
    with pytest.raises(InputValidationError, match="duplicate source identity"):
        validate_manifest(base, require_eight=False)


def test_full_selector_hash_can_consume_only_the_eight_pinned_rows():
    entry = _entry(length=3)
    entry["event_id"] = "event-7"
    selected = {
        "event_id": "event-7",
        "selection_order": 0,
        "immutable_split": "train",
        "usage_role": "annotation_calibration",
        "training_eligible": False,
        "source_identity": {
            "episode_index": 7,
            "raw_episode_id": 700,
            "task_index": 3,
            "task_instance_id": 70,
            "source_group_id": "group-7",
            "source_annotation_sha256": "a" * 64,
            "source_release_manifest_sha256": "release",
        },
    }
    unrelated = copy.deepcopy(selected)
    unrelated["event_id"] = "other"
    unrelated["source_identity"]["episode_index"] = 99
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "selected_rows.jsonl"
        path.write_text(
            json.dumps(unrelated) + "\n" + json.dumps(selected) + "\n", encoding="utf-8"
        )
        audit = load_selected_rows(path, [entry], expected_release_manifest_sha256="release")
    assert audit["file_row_count"] == 2
    assert audit["consumed_row_count"] == 1
    assert audit["ignored_unrelated_row_count"] == 1


def test_absolute_or_parent_parquet_path_is_rejected():
    base = {
        "schema_version": "p107-natural-retry-pilot-input-v1",
        "frozen_source": {
            "required_split": "train",
            "required_usage_role": "annotation_calibration",
            "all_training_eligible": False,
        },
        "official_info": {},
        "constraints": {"read_only": True},
        "episodes": [],
    }
    bad = _entry(length=3)
    bad["parquet_relative_path"] = "../escape.parquet"
    base["episodes"] = [bad]
    with pytest.raises(InputValidationError, match="relative and safe"):
        validate_manifest(base, require_eight=False)
