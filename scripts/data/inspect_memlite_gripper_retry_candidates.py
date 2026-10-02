#!/usr/bin/env python3
"""Audit unnamed gripper-command reversals in a frozen set of episodes.

This is a read-only diagnostic scanner.  It reports maximal runs of the two
shape-meta gripper command components and adjacent ``A, B, A`` runs without
assigning open/close, grasp, failure, recovery, success, or BC meaning to any
value.  Primitive/object fields are retained only as private annotation
context for later review.

The real dataset is read only when the CLI is run.  Unit tests exercise the
schema and interval logic with synthetic rows, so a missing remote snapshot
cannot silently turn into an empty or passing result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "p107-natural-retry-pilot-output-v1"
REQUIRED_PARQUET_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
ACTION_DIM = 23
STATE_DIM = 61
GRIPPER_INDICES = {"left": 14, "right": 22}
GRIPPER_INDEX_SOURCE = (
    "configs/data/behavior5_r1pro_memlite.yaml shape-meta convention; "
    "command polarity/units/contact semantics are not proven by this scanner"
)


class InputValidationError(ValueError):
    """Raised when a frozen input cannot be proven to match its contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(f"{field} must be an integer, got {value!r}")
    return int(value)


def _require_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{field} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise InputValidationError(f"{field} must be finite, got {value!r}")
    return result


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InputValidationError(f"{field} must be a finite sequence")
    return value


def _copy_json(value: Any) -> Any:
    """Copy a JSON-compatible value without retaining a mutable input object."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputValidationError(f"JSON root must be an object: {path}")
    return value


def _manifest_entry_key(entry: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        _require_int(entry.get("episode_index"), "episode_index"),
        _require_int(entry.get("raw_episode_id"), "raw_episode_id"),
        str(entry.get("source_group_id")),
    )


def validate_manifest(manifest: Mapping[str, Any], *, require_eight: bool = True) -> list[dict[str, Any]]:
    """Validate the writer-owned source manifest and return copied episode rows."""

    if manifest.get("schema_version") != "p107-natural-retry-pilot-input-v1":
        raise InputValidationError(f"unexpected manifest schema: {manifest.get('schema_version')!r}")
    frozen = manifest.get("frozen_source")
    info = manifest.get("official_info")
    episodes = manifest.get("episodes")
    constraints = manifest.get("constraints")
    if not isinstance(frozen, Mapping) or not isinstance(info, Mapping) or not isinstance(episodes, list):
        raise InputValidationError("manifest is missing frozen_source, official_info, or episodes")
    if not isinstance(constraints, Mapping) or constraints.get("read_only") is not True:
        raise InputValidationError("manifest must explicitly require read-only scanning")
    if require_eight and len(episodes) != 8:
        raise InputValidationError(f"expected exactly 8 episodes, got {len(episodes)}")
    if frozen.get("required_split") != "train" or frozen.get("required_usage_role") != "annotation_calibration":
        raise InputValidationError("manifest split/usage role is outside the pilot contract")
    if frozen.get("all_training_eligible") is not False:
        raise InputValidationError("manifest must mark all sources non-training-eligible")
    seen_keys: set[tuple[int, int, str]] = set()
    seen_episode: set[int] = set()
    seen_group: set[str] = set()
    seen_paths: dict[str, int] = {}
    copied: list[dict[str, Any]] = []
    required = {
        "event_id",
        "episode_index",
        "raw_episode_id",
        "task_index",
        "task_instance_id",
        "source_group_id",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "parquet_relative_path",
        "annotation_local_copy",
        "annotation_sha256",
        "immutable_split",
        "usage_role",
        "training_eligible",
    }
    for raw in episodes:
        if not isinstance(raw, Mapping):
            raise InputValidationError("each manifest episode must be an object")
        missing = sorted(required - set(raw))
        if missing:
            raise InputValidationError(f"manifest episode is missing {missing}")
        entry = dict(raw)
        key = _manifest_entry_key(entry)
        episode_index, _, source_group_id = key
        if key in seen_keys or episode_index in seen_episode or source_group_id in seen_group:
            raise InputValidationError(f"duplicate source identity: {key}")
        seen_keys.add(key)
        seen_episode.add(episode_index)
        seen_group.add(source_group_id)
        length = _require_int(entry["length"], f"episode {episode_index}.length")
        data_from = _require_int(entry["dataset_from_index"], f"episode {episode_index}.dataset_from_index")
        data_to = _require_int(entry["dataset_to_index"], f"episode {episode_index}.dataset_to_index")
        if length <= 0 or data_from < 0 or data_to - data_from != length:
            raise InputValidationError(f"episode {episode_index} has inconsistent length/global range")
        if entry["immutable_split"] != "train" or entry["usage_role"] != "annotation_calibration":
            raise InputValidationError(f"episode {episode_index} has an invalid split or usage role")
        if entry["training_eligible"] is not False:
            raise InputValidationError(f"episode {episode_index} is training-eligible")
        if not isinstance(entry["parquet_relative_path"], str) or not entry["parquet_relative_path"]:
            raise InputValidationError(f"episode {episode_index} has no parquet path")
        parquet_path = PurePosixPath(entry["parquet_relative_path"])
        if parquet_path.is_absolute() or ".." in parquet_path.parts:
            raise InputValidationError(f"episode {episode_index} parquet path is not relative and safe")
        previous = seen_paths.get(entry["parquet_relative_path"])
        if previous is not None and previous != episode_index:
            # Shared parquet files are valid; retain this fact for the reader.
            pass
        seen_paths[entry["parquet_relative_path"]] = episode_index
        copied.append(entry)
    return copied


def validate_info(info: Mapping[str, Any]) -> dict[str, list[int]]:
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise InputValidationError("official info has no features object")
    action = features.get("action")
    state = features.get("observation.state")
    action_shape = action.get("shape") if isinstance(action, Mapping) else None
    state_shape = state.get("shape") if isinstance(state, Mapping) else None
    if action_shape != [ACTION_DIM] or state_shape != [STATE_DIM]:
        raise InputValidationError(f"unexpected info shapes: action={action_shape!r}, state={state_shape!r}")
    return {"action": list(action_shape), "observation.state": list(state_shape)}


def load_selected_rows(
    path: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_release_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and identity-check the selected-row JSONL without dropping raw hashes."""

    expected = {int(entry["episode_index"]): entry for entry in entries}
    if not path.is_file():
        raise InputValidationError(f"selected rows file is missing: {path}")
    raw_bytes = path.read_bytes()
    rows: list[dict[str, Any]] = []
    hashes: dict[str, dict[str, str]] = {}
    full_row_count = 0
    for line_number, raw_line in enumerate(raw_bytes.splitlines(keepends=True), start=1):
        if not raw_line.strip():
            continue
        full_row_count += 1
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise InputValidationError(f"selected rows line {line_number} is not JSON: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("source_identity"), Mapping):
            raise InputValidationError(f"selected rows line {line_number} lacks source_identity")
        identity = value["source_identity"]
        episode_index = _require_int(identity.get("episode_index"), f"selected row {line_number}.episode_index")
        entry = expected.get(episode_index)
        if entry is None:
            # The frozen selected_rows.jsonl is a larger train selector.  The
            # pilot manifest consumes only its eight pinned source rows; keep
            # the full-file SHA while ignoring unrelated selector rows.
            continue
        for field in ("raw_episode_id", "task_index", "task_instance_id", "source_group_id"):
            if identity.get(field) != entry.get(field):
                raise InputValidationError(f"selected row {episode_index} disagrees on {field}")
        if value.get("event_id") != entry.get("event_id"):
            raise InputValidationError(f"selected row {episode_index} disagrees on event_id")
        if value.get("selection_order") != entry.get("selection_order"):
            raise InputValidationError(f"selected row {episode_index} disagrees on selection_order")
        if value.get("immutable_split") != entry.get("immutable_split"):
            raise InputValidationError(f"selected row {episode_index} disagrees on split")
        if value.get("usage_role") != entry.get("usage_role") or value.get("training_eligible") is not False:
            raise InputValidationError(f"selected row {episode_index} disagrees on usage/training eligibility")
        if (
            expected_release_manifest_sha256 is not None
            and identity.get("source_release_manifest_sha256") != expected_release_manifest_sha256
        ):
            raise InputValidationError(f"selected row {episode_index} disagrees on release manifest SHA")
        if identity.get("source_annotation_sha256") != entry.get("annotation_sha256"):
            raise InputValidationError(f"selected row {episode_index} disagrees on annotation SHA")
        if episode_index in hashes:
            raise InputValidationError(f"duplicate selected row episode {episode_index}")
        hashes[str(episode_index)] = {
            "raw_line_sha256": hashlib.sha256(raw_line).hexdigest(),
            "canonical_json_sha256": canonical_sha256(value),
        }
        rows.append(value)
    if len(rows) != len(entries):
        raise InputValidationError(f"selected rows count {len(rows)} != manifest count {len(entries)}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "file_row_count": full_row_count,
        "consumed_row_count": len(rows),
        "ignored_unrelated_row_count": full_row_count - len(rows),
        "row_hashes_by_episode": hashes,
    }


def load_frozen_episode_rows(
    path: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate the eight selected identities against frozen episodes.jsonl."""

    expected = {int(entry["episode_index"]): entry for entry in entries}
    if not path.is_file():
        raise InputValidationError(f"frozen episodes JSONL is missing: {path}")
    digest = hashlib.sha256()
    found: dict[int, dict[str, Any]] = {}
    total_rows = 0
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            total_rows += 1
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(f"frozen episodes line {line_number} is not JSON: {exc}") from exc
            row = value.get("row") if isinstance(value, Mapping) else None
            if not isinstance(row, Mapping):
                continue
            episode = row.get("episode_index")
            if not isinstance(episode, int) or episode not in expected:
                continue
            if episode in found:
                raise InputValidationError(f"duplicate frozen episode row {episode}")
            entry = expected[episode]
            checks = {
                "raw_episode_id": row.get("raw_episode_id"),
                "task_index": row.get("task_index"),
                "task_instance_id": row.get("task_instance_id"),
                "length": row.get("length"),
                "dataset_from_index": row.get("dataset_from_index"),
                "dataset_to_index": row.get("dataset_to_index"),
            }
            for field, actual in checks.items():
                if actual != entry.get(field):
                    raise InputValidationError(f"frozen episode {episode} disagrees on {field}")
            if value.get("split") != entry.get("immutable_split"):
                raise InputValidationError(f"frozen episode {episode} disagrees on split")
            if value.get("annotation_sha256") != entry.get("annotation_sha256"):
                raise InputValidationError(f"frozen episode {episode} disagrees on annotation SHA")
            if row.get("annotation_path") != entry.get("annotation_relative_path"):
                raise InputValidationError(f"frozen episode {episode} disagrees on annotation path")
            chunk = row.get("data/chunk_index")
            file_index = row.get("data/file_index")
            if not isinstance(chunk, int) or not isinstance(file_index, int):
                raise InputValidationError(f"frozen episode {episode} has no data chunk/file identity")
            parquet_path = f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
            if parquet_path != entry.get("parquet_relative_path"):
                raise InputValidationError(f"frozen episode {episode} disagrees on parquet path")
            found[episode] = {
                "line_number": line_number,
                "canonical_json_sha256": canonical_sha256(value),
                "parquet_relative_path": parquet_path,
                "annotation_relative_path": row.get("annotation_path"),
            }
    actual_sha = digest.hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise InputValidationError(
            f"frozen episodes JSONL SHA mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    if set(found) != set(expected):
        missing = sorted(set(expected) - set(found))
        raise InputValidationError(f"frozen episodes JSONL is missing selected episodes {missing}")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "row_count": total_rows,
        "selected_row_count": len(found),
        "selected_rows_by_episode": {str(key): value for key, value in sorted(found.items())},
        "expected_sha256": expected_sha256,
    }


def _private_primitive_fields(raw: Mapping[str, Any], ordinal: int, start: int, end: int) -> dict[str, Any]:
    """Retain annotation referents without turning them into a label."""

    return {
        "primitive_ordinal": ordinal,
        "primitive_idx": raw.get("primitive_idx", ordinal),
        "raw_frame_duration": _copy_json(raw.get("frame_duration")),
        "frame_interval": [start, end],
        "primitive_id": _copy_json(raw.get("primitive_id")),
        "primitive_description": _copy_json(raw.get("primitive_description")),
        "skill_idxes": _copy_json(raw.get("skill_idxes")),
        "object_id": _copy_json(raw.get("object_id")),
        "manipulating_object_id": _copy_json(raw.get("manipulating_object_id")),
    }


def parse_primitive_annotation(annotation: Mapping[str, Any] | None, length: int) -> dict[str, Any]:
    """Validate raw primitive intervals and expose their observed join geometry."""

    if annotation is None:
        return {
            "status": "no_annotation",
            "join_semantics": "no_annotation",
            "intervals": [],
            "gaps": [[0, length]] if length else [],
            "union_intervals": [],
            "union_frames": 0,
        }
    raw_intervals = annotation.get("primitive_annotation")
    if raw_intervals in (None, []):
        return {
            "status": "no_annotation",
            "join_semantics": "no_annotation",
            "intervals": [],
            "gaps": [[0, length]] if length else [],
            "union_intervals": [],
            "union_frames": 0,
        }
    if not isinstance(raw_intervals, list):
        raise InputValidationError("primitive_annotation must be a list")
    intervals: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_intervals):
        if not isinstance(raw, Mapping):
            raise InputValidationError(f"primitive_annotation[{ordinal}] is not an object")
        duration = raw.get("frame_duration")
        if (
            not isinstance(duration, list)
            or len(duration) != 2
            or isinstance(duration[0], bool)
            or isinstance(duration[0], float)
            or isinstance(duration[1], bool)
            or isinstance(duration[1], float)
        ):
            raise InputValidationError(f"primitive_annotation[{ordinal}] has invalid frame_duration")
        start = _require_int(duration[0], f"primitive_annotation[{ordinal}].start")
        end = _require_int(duration[1], f"primitive_annotation[{ordinal}].end")
        if start < 0 or end > length or start >= end:
            raise InputValidationError(
                f"primitive_annotation[{ordinal}] frame_duration {duration!r} is outside [0,{length})"
            )
        intervals.append(_private_primitive_fields(raw, ordinal, start, end))
    intervals.sort(key=lambda value: (value["frame_interval"][0], value["frame_interval"][1]))
    overlaps: list[list[int]] = []
    for index, previous in enumerate(intervals):
        for current in intervals[index + 1 :]:
            overlap_start = max(previous["frame_interval"][0], current["frame_interval"][0])
            overlap_end = min(previous["frame_interval"][1], current["frame_interval"][1])
            if overlap_start < overlap_end:
                overlaps.append([overlap_start, overlap_end])
    gaps: list[list[int]] = []
    cursor = 0
    union_intervals: list[list[int]] = []
    for item in intervals:
        start, end = item["frame_interval"]
        if cursor < start:
            gaps.append([cursor, start])
        cursor = max(cursor, end)
        if not union_intervals or start > union_intervals[-1][1]:
            union_intervals.append([start, end])
        else:
            union_intervals[-1][1] = max(union_intervals[-1][1], end)
    if cursor < length:
        gaps.append([cursor, length])
    union_frames = sum(end - start for start, end in union_intervals)
    full_partition = (
        intervals[0]["frame_interval"][0] == 0
        and intervals[-1]["frame_interval"][1] == length
        and not gaps
        and not overlaps
    )
    # A complete, non-overlapping partition gives an observable half-open
    # interpretation.  Partial annotations retain all intervals but do not
    # authorize a same-primitive claim.
    join_semantics = "verified_half_open_partition" if full_partition else "join_semantics_unverified"
    return {
        "status": "present",
        "join_semantics": join_semantics,
        "intervals": intervals,
        "gaps": gaps,
        "overlaps": overlaps,
        "union_intervals": union_intervals,
        "union_frames": union_frames,
        "full_partition": full_partition,
    }


def validate_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
    *,
    action_dim: int = ACTION_DIM,
    state_dim: int = STATE_DIM,
) -> list[dict[str, Any]]:
    """Validate one selected episode's complete clock/action/state rows."""

    episode = _require_int(entry.get("episode_index"), "manifest episode_index")
    task = _require_int(entry.get("task_index"), "manifest task_index")
    length = _require_int(entry.get("length"), "manifest length")
    global_start = _require_int(entry.get("dataset_from_index"), "manifest dataset_from_index")
    if len(rows) != length:
        raise InputValidationError(f"episode {episode} row count {len(rows)} != length {length}")
    normalized: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise InputValidationError(f"episode {episode} row {position} is not an object")
        row_episode = _require_int(row.get("episode_index"), f"episode {episode} row episode_index")
        if row_episode != episode:
            raise InputValidationError(f"episode membership mismatch: expected {episode}, got {row_episode}")
        row_task = _require_int(row.get("task_index"), f"episode {episode} row task_index")
        if row_task != task:
            raise InputValidationError(f"episode {episode} task mismatch: expected {task}, got {row_task}")
        frame = _require_int(row.get("frame_index"), f"episode {episode} row frame_index")
        index = _require_int(row.get("index"), f"episode {episode} row index")
        if frame < 0 or frame >= length or index != global_start + frame:
            raise InputValidationError(f"episode {episode} frame_index/index mismatch at frame {frame}")
        timestamp = _require_finite(row.get("timestamp"), f"episode {episode} frame {frame} timestamp")
        action = _require_sequence(row.get("action"), f"episode {episode} frame {frame} action")
        state = _require_sequence(row.get("observation.state"), f"episode {episode} frame {frame} state")
        if len(action) != action_dim or len(state) != state_dim:
            raise InputValidationError(
                f"episode {episode} frame {frame} dimension mismatch: action={len(action)}, state={len(state)}"
            )
        action_values = [_require_finite(value, f"episode {episode} frame {frame} action[{i}]") for i, value in enumerate(action)]
        state_values = [_require_finite(value, f"episode {episode} frame {frame} state[{i}]") for i, value in enumerate(state)]
        normalized.append(
            {
                "episode_index": row_episode,
                "task_index": row_task,
                "frame_index": frame,
                "index": index,
                "timestamp": timestamp,
                "action": action_values,
                "observation.state": state_values,
            }
        )
    normalized.sort(key=lambda row: row["frame_index"])
    if [row["frame_index"] for row in normalized] != list(range(length)):
        raise InputValidationError(f"episode {episode} frame_index is not exactly 0..{length - 1}")
    if [row["index"] for row in normalized] != list(range(global_start, global_start + length)):
        raise InputValidationError(f"episode {episode} global index is not contiguous")
    timestamps = [row["timestamp"] for row in normalized]
    if any(next_timestamp <= timestamp for timestamp, next_timestamp in zip(timestamps, timestamps[1:])):
        raise InputValidationError(f"episode {episode} timestamps are not strictly increasing")
    return normalized


def build_rle_runs(rows: Sequence[Mapping[str, Any]], channel: str, action_index: int) -> list[dict[str, Any]]:
    """Build maximal command-value runs, preserving source values and frame clock."""

    if channel not in GRIPPER_INDICES:
        raise InputValidationError(f"unknown gripper channel {channel!r}")
    if not rows:
        return []
    values = [row["action"][action_index] for row in rows]
    runs: list[dict[str, Any]] = []
    start = 0
    for position in range(1, len(values) + 1):
        if position < len(values) and values[position] == values[start]:
            continue
        end = position
        start_timestamp = float(rows[start]["timestamp"])
        last_timestamp = float(rows[end - 1]["timestamp"])
        runs.append(
            {
                "run_index": len(runs),
                "channel": channel,
                "action_index": action_index,
                "start_frame": int(rows[start]["frame_index"]),
                "end_frame": int(rows[end - 1]["frame_index"]) + 1,
                "value": float(values[start]),
                "dwell_frames": end - start,
                "start_timestamp": start_timestamp,
                "last_observed_timestamp": last_timestamp,
                "dwell_seconds_observed": last_timestamp - start_timestamp,
            }
        )
        start = end
    return runs


def find_reversal_candidates(
    runs: Sequence[Mapping[str, Any]], *, min_dwell_frames: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Find adjacent maximal A-B-A runs and explicit dwell exclusions."""

    min_dwell_frames = _require_int(min_dwell_frames, "min_dwell_frames")
    if min_dwell_frames < 0:
        raise InputValidationError("min_dwell_frames cannot be negative")
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    possible = max(0, len(runs) - 2)
    structural_matches = 0
    for index in range(possible):
        first, middle, last = runs[index : index + 3]
        if first["value"] != last["value"] or first["value"] == middle["value"]:
            continue
        structural_matches += 1
        run_ids = [int(first["run_index"]), int(middle["run_index"]), int(last["run_index"])]
        if any(int(run["dwell_frames"]) < min_dwell_frames for run in (first, middle, last)):
            exclusions.append(
                {
                    "run_indices": run_ids,
                    "reason": "dwell_below_threshold",
                    "min_dwell_frames": min_dwell_frames,
                    "dwell_frames": [int(first["dwell_frames"]), int(middle["dwell_frames"]), int(last["dwell_frames"])],
                }
            )
            continue
        candidates.append(
            {
                "run_indices": run_ids,
                "values": [float(first["value"]), float(middle["value"]), float(last["value"])],
                "run_intervals": [_copy_json(first), _copy_json(middle), _copy_json(last)],
                "frame_span": [int(first["start_frame"]), int(last["end_frame"])],
                "command_structure": "A_B_A_UNNAMED",
                "semantic_interpretation": "NOT_ASSIGNED",
            }
        )
    return candidates, exclusions, {
        "run_count": len(runs),
        "possible_three_run_windows": possible,
        "structural_aba_matches": structural_matches,
        "candidate_count_after_filters": len(candidates),
        "excluded_count": len(exclusions),
    }


def _union_and_gaps(intervals: Iterable[tuple[int, int]], start: int, end: int) -> tuple[list[list[int]], int]:
    ordered = sorted((left, right) for left, right in intervals if left < right)
    if not ordered:
        return ([[start, end]] if start < end else [], 0)
    union: list[list[int]] = []
    for left, right in ordered:
        if not union or left > union[-1][1]:
            union.append([left, right])
        else:
            union[-1][1] = max(union[-1][1], right)
    gaps: list[list[int]] = []
    cursor = start
    for left, right in union:
        if cursor < left:
            gaps.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append([cursor, end])
    covered = sum(right - left for left, right in union)
    return gaps, covered


def _primitive_relation_audit(
    span_start: int, span_end: int, annotation_audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute one span's raw annotation relation without assigning task meaning."""

    if span_start < 0 or span_end <= span_start:
        raise InputValidationError(f"invalid relation span [{span_start}, {span_end})")
    intervals = annotation_audit.get("intervals", [])
    intersections: list[dict[str, Any]] = []
    covered_parts: list[tuple[int, int]] = []
    for interval in intervals:
        p_start, p_end = interval["frame_interval"]
        left, right = max(span_start, p_start), min(span_end, p_end)
        if left < right:
            intersections.append(
                {
                    "primitive_ordinal": interval["primitive_ordinal"],
                    "primitive_idx": interval["primitive_idx"],
                    "raw_frame_duration": _copy_json(interval["raw_frame_duration"]),
                    "intersection_frame_interval": [left, right],
                    "primitive_frame_interval": [p_start, p_end],
                    "primitive_id": _copy_json(interval["primitive_id"]),
                    "primitive_description": _copy_json(interval["primitive_description"]),
                    "skill_idxes": _copy_json(interval["skill_idxes"]),
                    "object_id": _copy_json(interval["object_id"]),
                    "manipulating_object_id": _copy_json(interval["manipulating_object_id"]),
                }
            )
            covered_parts.append((left, right))
    gaps, covered = _union_and_gaps(covered_parts, span_start, span_end)
    span_frames = span_end - span_start
    overlap_in_span = [
        overlap
        for overlap in annotation_audit.get("overlaps", [])
        if overlap[0] < span_end and span_start < overlap[1]
    ]
    if not intersections:
        classification = "no_annotation"
    elif overlap_in_span:
        classification = "ambiguous_overlapping_annotation"
    elif covered == span_frames and len(intersections) == 1:
        classification = "same_primitive"
    elif covered == span_frames:
        classification = "cross_primitive_boundary"
    else:
        classification = "partial_annotation_with_gaps"
    return {
        "classification": classification,
        "join_semantics": annotation_audit.get("join_semantics"),
        "join_semantics_warning": (
            "Do not make a primitive-boundary claim from this candidate."
            if annotation_audit.get("join_semantics") == "join_semantics_unverified"
            else None
        ),
        "candidate_span_frame_interval": [span_start, span_end],
        "candidate_span_frames": span_frames,
        "covered_frames": covered,
        "gaps_within_candidate": gaps,
        "overlapping_annotation_regions": overlap_in_span,
        "raw_primitive_intersections": intersections,
        "all_raw_annotation_intervals": _copy_json(intervals),
    }


def _flip_core_span(candidate: Mapping[str, Any]) -> list[int]:
    """Return the three-run flip core, clipped to the legacy outer context."""

    runs = candidate.get("run_intervals")
    if not isinstance(runs, list) or len(runs) != 3:
        raise InputValidationError("A-B-A candidate must contain exactly three run intervals")
    first, middle, last = runs
    outer_start = int(first["start_frame"])
    outer_end = int(last["end_frame"])
    start = max(outer_start, int(first["end_frame"]) - 1)
    end = min(outer_end, int(middle["end_frame"]) + 1)
    if end <= start:
        raise InputValidationError(f"invalid flip core [{start}, {end})")
    return [start, end]


def annotate_candidate(candidate: Mapping[str, Any], annotation_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Attach legacy outer and new flip-core primitive intersections."""

    outer_span = [int(value) for value in candidate["frame_span"]]
    outer_audit = _primitive_relation_audit(outer_span[0], outer_span[1], annotation_audit)
    core_span = _flip_core_span(candidate)
    core_audit = _primitive_relation_audit(core_span[0], core_span[1], annotation_audit)
    result = _copy_json(candidate)
    # ``frame_span`` and the old private audit remain the v1 outer-context
    # definition.  The explicit aliases make that definition auditable while
    # preserving every legacy field and count.
    result["outer_context_span"] = outer_span
    result["outer_context_relation"] = outer_audit["classification"]
    result["flip_core_span"] = core_span
    result["flip_core_relation"] = core_audit["classification"]
    result["private_annotation_audit"] = outer_audit
    result["private_annotation_audit"]["outer_context_span"] = outer_span
    result["private_annotation_audit"]["outer_context_relation"] = outer_audit["classification"]
    result["private_annotation_audit"]["flip_core_span"] = core_span
    result["private_annotation_audit"]["flip_core_relation"] = core_audit["classification"]
    result["private_annotation_audit"]["flip_core_annotation_audit"] = core_audit
    return result


def scan_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
    annotation_audit: Mapping[str, Any],
    *,
    min_dwell_frames: int = 0,
) -> dict[str, Any]:
    normalized = validate_episode_rows(rows, entry)
    episode = int(entry["episode_index"])
    channel_results: dict[str, Any] = {}
    all_candidates: list[dict[str, Any]] = []
    all_exclusions: list[dict[str, Any]] = []
    for channel, action_index in GRIPPER_INDICES.items():
        runs = build_rle_runs(normalized, channel, action_index)
        candidates, exclusions, counts = find_reversal_candidates(
            runs, min_dwell_frames=min_dwell_frames
        )
        for candidate_index, candidate in enumerate(candidates):
            enriched = annotate_candidate(candidate, annotation_audit)
            enriched["candidate_id"] = f"episode-{episode:06d}:{channel}:aba-{candidate_index:04d}"
            enriched["channel"] = channel
            all_candidates.append(enriched)
        for exclusion in exclusions:
            item = _copy_json(exclusion)
            item["episode_index"] = episode
            item["channel"] = channel
            all_exclusions.append(item)
        channel_results[channel] = {"runs": runs, "counts": counts}
    outer_context_relation_counts = Counter(
        str(candidate["outer_context_relation"]) for candidate in all_candidates
    )
    flip_core_relation_counts = Counter(
        str(candidate["flip_core_relation"]) for candidate in all_candidates
    )
    return {
        "source_identity": {
            "event_id": entry["event_id"],
            "episode_index": episode,
            "raw_episode_id": int(entry["raw_episode_id"]),
            "task_index": int(entry["task_index"]),
            "task_instance_id": int(entry["task_instance_id"]),
            "source_group_id": entry["source_group_id"],
        },
        "length": int(entry["length"]),
        "rows_scanned": len(normalized),
        "gripper_action_indices": dict(GRIPPER_INDICES),
        "gripper_action_indices_source": GRIPPER_INDEX_SOURCE,
        "annotation_audit": _copy_json(annotation_audit),
        "channels": channel_results,
        "candidates": all_candidates,
        "candidate_exclusions": all_exclusions,
        "candidate_count": len(all_candidates),
        "excluded_candidate_count": len(all_exclusions),
        "outer_context_relation_counts": dict(sorted(outer_context_relation_counts.items())),
        "flip_core_relation_counts": dict(sorted(flip_core_relation_counts.items())),
        "role": "diagnostic_candidate",
        "training_eligible": False,
        "action_bc_supervision": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "dart_supervision": False,
        "attempt_status": "NOT_APPLICABLE",
    }


def _read_parquet_rows(
    entries: Sequence[Mapping[str, Any]],
    source_root: Path,
    on_episode_rows: Any,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise InputValidationError("pyarrow is required for a real Parquet scan") from exc
    by_path: dict[Path, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        path = (source_root / str(entry["parquet_relative_path"])).resolve()
        by_path[path].append(entry)
    file_records: list[dict[str, Any]] = []
    parquet_rows_seen = 0
    for path, path_entries in sorted(by_path.items(), key=lambda item: str(item[0])):
        if not path.is_file():
            raise InputValidationError(f"Parquet source is missing: {path}")
        parquet_file = parquet.ParquetFile(path)
        names = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_PARQUET_COLUMNS) - names)
        if missing:
            raise InputValidationError(f"Parquet {path} is missing columns {missing}")
        wanted = {int(entry["episode_index"]): entry for entry in path_entries}
        selected_for_file: dict[int, list[dict[str, Any]]] = defaultdict(list)
        file_rows_seen = 0
        for batch in parquet_file.iter_batches(columns=list(REQUIRED_PARQUET_COLUMNS), use_threads=False):
            batch_rows = batch.to_pylist()
            parquet_rows_seen += len(batch_rows)
            file_rows_seen += len(batch_rows)
            for row in batch_rows:
                episode = row.get("episode_index")
                if isinstance(episode, int) and episode in wanted:
                    selected_for_file[episode].append(row)
        # Validate and release one source file's selected rows before reading
        # another file.  This keeps action/state Python lists bounded by one
        # Parquet group rather than retaining all eight episodes at once.
        for episode, entry in wanted.items():
            on_episode_rows(entry, selected_for_file.get(episode, []))
        file_records.append(
            {
                "path": str(path),
                "selected_episode_indices": sorted(wanted),
                "parquet_num_rows": parquet_file.metadata.num_rows,
                "parquet_rows_seen": file_rows_seen,
            }
        )
    return {
        "unique_parquet_files_read": len(by_path),
        "parquet_files": file_records,
        "parquet_rows_seen": parquet_rows_seen,
    }


def run_scan(
    manifest_path: Path,
    output_path: Path,
    *,
    source_root: Path | None = None,
    info_path: Path | None = None,
    selected_rows_path: Path | None = None,
    frozen_episodes_path: Path | None = None,
    annotation_root: Path | None = None,
    min_dwell_frames: int = 0,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Run a complete scan and return the machine result before writing it."""

    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise InputValidationError(
            f"manifest SHA mismatch: expected {expected_manifest_sha256}, got {manifest_sha}"
        )
    manifest = _load_json(manifest_path)
    entries = validate_manifest(manifest)
    selected_path = selected_rows_path or Path(str(manifest["frozen_source"]["selected_rows_path"]))
    selected_audit = load_selected_rows(
        selected_path,
        entries,
        expected_release_manifest_sha256=manifest["frozen_source"].get("release_manifest_sha256"),
    )
    expected_selected_sha = manifest["frozen_source"].get("selected_rows_sha256")
    if expected_selected_sha and selected_audit["sha256"] != expected_selected_sha:
        raise InputValidationError("selected rows SHA does not match manifest")
    frozen_path = frozen_episodes_path or Path(str(manifest["frozen_source"]["episodes_jsonl_path"]))
    frozen_audit = load_frozen_episode_rows(
        frozen_path,
        entries,
        expected_sha256=manifest["frozen_source"].get("episodes_jsonl_sha256"),
    )
    actual_info_path = info_path or Path(str(manifest["official_info"]["local_copy_path"]))
    info_sha = sha256_file(actual_info_path)
    expected_info_sha = manifest["official_info"].get("sha256")
    if expected_info_sha and info_sha != expected_info_sha:
        raise InputValidationError("official info SHA does not match manifest")
    info = _load_json(actual_info_path)
    info_shapes = validate_info(info)
    declared_root = Path(str(manifest["official_snapshot_root"])).resolve()
    if source_root is not None and source_root.resolve() != declared_root:
        raise InputValidationError("source-root override does not exactly match manifest official_snapshot_root")
    root = declared_root
    episode_results: list[dict[str, Any]] = []
    def consume_episode(entry: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
        episode = int(entry["episode_index"])
        if annotation_root is None:
            annotation_path = Path(str(entry["annotation_local_copy"]))
        else:
            relative_annotation = PurePosixPath(str(entry["annotation_relative_path"]))
            if relative_annotation.is_absolute() or ".." in relative_annotation.parts:
                raise InputValidationError(f"episode {episode} annotation path is not relative and safe")
            annotation_path = annotation_root / Path(*relative_annotation.parts)
        annotation_sha = sha256_file(annotation_path)
        if annotation_sha != entry["annotation_sha256"]:
            raise InputValidationError(f"episode {episode} annotation SHA mismatch")
        annotation = _load_json(annotation_path)
        annotation_audit = parse_primitive_annotation(annotation, int(entry["length"]))
        result = scan_episode_rows(
            rows,
            entry,
            annotation_audit,
            min_dwell_frames=min_dwell_frames,
        )
        result["annotation_path"] = str(annotation_path)
        result["annotation_sha256"] = annotation_sha
        episode_results.append(result)
    parquet_audit = _read_parquet_rows(entries, root, consume_episode)
    episode_results.sort(key=lambda result: next(
        int(entry["selection_order"]) for entry in entries
        if int(entry["episode_index"]) == int(result["source_identity"]["episode_index"])
    ))
    total_runs = sum(
        len(episode["channels"][channel]["runs"])
        for episode in episode_results
        for channel in GRIPPER_INDICES
    )
    total_candidates = sum(int(episode["candidate_count"]) for episode in episode_results)
    total_exclusions = sum(int(episode["excluded_candidate_count"]) for episode in episode_results)
    outer_context_relation_counts = Counter()
    flip_core_relation_counts = Counter()
    for episode in episode_results:
        outer_context_relation_counts.update(episode["outer_context_relation_counts"])
        flip_core_relation_counts.update(episode["flip_core_relation_counts"])
    script_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_STRUCTURAL_SCAN_NONTRAINABLE",
        "role": "diagnostic_candidate",
        "training_eligible": False,
        "action_bc_supervision": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "dart_supervision": False,
        "attempt_status": "NOT_APPLICABLE",
        "release_status": "NOT_RELEASED",
        "semantic_guard": (
            "A_B_A is an unnamed command-structure candidate only; values are not open/close, "
            "grasp, success, failure, recovery, corrective action, or BC labels."
        ),
        "gripper_index_contract": {
            "indices": dict(GRIPPER_INDICES),
            "source": GRIPPER_INDEX_SOURCE,
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "git_commit": _git_commit(script_root),
            "cli_output_path": str(output_path),
            "min_dwell_frames": min_dwell_frames,
        },
        "inputs": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "manifest_schema_version": manifest.get("schema_version"),
            "official_info_path": str(actual_info_path),
            "official_info_sha256": info_sha,
            "info_shapes": info_shapes,
            "selected_rows": selected_audit,
            "release_manifest_sha256": manifest["frozen_source"].get("release_manifest_sha256"),
            "frozen_episodes": frozen_audit,
            "source_root": str(root),
        },
        "scan": {
            "episodes_requested": len(entries),
            "episodes_scanned": len(episode_results),
            "rows_scanned": sum(int(episode["rows_scanned"]) for episode in episode_results),
            "total_rle_runs": total_runs,
            "candidate_count": total_candidates,
            "excluded_candidate_count": total_exclusions,
            "outer_context_relation_counts": dict(sorted(outer_context_relation_counts.items())),
            "flip_core_relation_counts": dict(sorted(flip_core_relation_counts.items())),
            **parquet_audit,
        },
        "episodes": episode_results,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--info", type=Path, default=None)
    parser.add_argument("--selected-rows-path", type=Path, default=None)
    parser.add_argument("--frozen-episodes-path", type=Path, default=None)
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--min-dwell-frames", type=int, default=0)
    parser.add_argument("--expected-manifest-sha256", required=True)
    return parser


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write only a complete result and refuse to replace an existing artifact."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        result = run_scan(
            args.manifest,
            args.output,
            source_root=args.source_root,
            info_path=args.info,
            selected_rows_path=args.selected_rows_path,
            frozen_episodes_path=args.frozen_episodes_path,
            annotation_root=args.annotation_root,
            min_dwell_frames=args.min_dwell_frames,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        result["implementation"]["cli_args"] = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        _atomic_write_json(args.output, result)
    except (InputValidationError, OSError, ImportError, FileExistsError) as exc:
        print(json.dumps({"status": "FAILED_INPUT_VALIDATION", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "PASS_STRUCTURAL_SCAN_NONTRAINABLE", "output": str(args.output)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
