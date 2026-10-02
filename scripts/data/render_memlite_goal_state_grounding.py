#!/usr/bin/env python3
"""Render a bounded, private RGB grounding view for goal-unbound anchors.

The formal GRASP batch contains metadata-only phase anchors.  This adapter
authenticates those derived rows against their canonical parent events in the
sealed full index, then optionally decodes only the explicitly requested
actor-available frames from the three native RGB cameras.  It deliberately
does not create actor packets, questions, labels, outcome/recovery evidence,
or training data.

The decoder is the existing neutral implementation in
``render_memlite_event_packets.py``.  This file only owns the goal-unbound
input contract, source binding, bounded frame plan, and private receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


def _load_module(name: str, path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required local module is not a regular file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load local module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parents[2]
_NEUTRAL = _load_module(
    "_p107_goal_state_neutral_verifier",
    Path(__file__).with_name("render_memlite_retry_verifier.py"),
)

InputValidationError = _NEUTRAL.InputValidationError
_finite_number = _NEUTRAL._finite_number
_load_json = _NEUTRAL._load_json
_regular_dir = _NEUTRAL._regular_dir
_regular_file = _NEUTRAL._regular_file
_resolve_video = _NEUTRAL._resolve_video
_safe_relative_path = _NEUTRAL._safe_relative_path
_sha256 = _NEUTRAL._sha256


SCRIPT_SCHEMA = "p107-goal-state-private-rgb-grounding-v1"
PREFLIGHT_SCHEMA = "p107-goal-state-private-rgb-grounding-preflight-v1"
BATCH_SCHEMA = "p107-grasp-goal-state-batch-manifest-v1"
EVENT_SCHEMA = "memlite-event-recovery-v1"
INDEX_SCHEMA = "memlite-event-index-v1"
REQUEST_SCHEMA = "p107-camera-native-temporal-request-v1"
QUEUE_SCHEMA = "p107-metadata-annotation-queue-v1"
VIEWS = ("head", "left_wrist", "right_wrist")
CAMERA_BY_VIEW = {
    "head": "observation.rgb.zed_link_camera_0",
    "left_wrist": "observation.rgb.left_realsense_link_camera_0",
    "right_wrist": "observation.rgb.right_realsense_link_camera_0",
}
FPS = 30.0
MAX_UNIQUE_FRAMES = 64
MAX_PNG_COUNT = MAX_UNIQUE_FRAMES * len(VIEWS)
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
SHA256_CHARS = frozenset("0123456789abcdef")


def _strict_json(data: bytes, *, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InputValidationError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise InputValidationError(f"non-finite JSON number in {name}: {value}")

    try:
        return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InputValidationError(f"invalid JSON in {name}") from error


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_CHARS


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise InputValidationError(f"{label} must be an integer")
    return value


def _require_pin(actual: str, expected: Any, label: str) -> None:
    if not _is_sha256(expected) or actual != expected:
        raise InputValidationError(f"{label} SHA-256 does not match its external pin")


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        total += path.stat().st_size
    return total


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        + b"\n"
    )


def _read_jsonl_receipt(path: Path, receipt: Mapping[str, Any], label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _regular_file(path, label)
    if type(receipt.get("bytes")) is not int or type(receipt.get("rows")) is not int or not _is_sha256(receipt.get("sha256")):
        raise InputValidationError(f"{label} receipt is malformed")
    if path.stat().st_size != receipt["bytes"]:
        raise InputValidationError(f"{label} byte count differs from its formal receipt")
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.strip():
                continue
            value = _strict_json(line, name=f"{path}:{number}")
            if not isinstance(value, Mapping):
                raise InputValidationError(f"{label} row {number} is not an object")
            rows.append(dict(value))
    if len(rows) != receipt["rows"] or digest.hexdigest() != receipt["sha256"]:
        raise InputValidationError(f"{label} bytes or row count differs from its formal receipt")
    return rows, {"path": str(path), "sha256": digest.hexdigest(), "rows": len(rows), "bytes": path.stat().st_size}


def _check_gate_map(
    value: Mapping[str, Any], label: str, *, require_status_fields: bool = False, require_supervision_fields: bool = True
) -> None:
    expected = {
        "training_eligible": False,
        "action_bc_supervision": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "dart_supervision": False,
        "attempt_status": "NOT_APPLICABLE",
        "outcome_status": "NOT_APPLICABLE",
        "recovery_status": "NOT_APPLICABLE",
    }
    for key, expected_value in expected.items():
        if not require_supervision_fields and key in {"action_bc_supervision", "outcome_supervision", "recovery_supervision", "dart_supervision"}:
            continue
        if key not in value:
            if require_status_fields or key in {"training_eligible"}:
                raise InputValidationError(f"{label} gate is missing: {key}")
            continue
        if value.get(key) != expected_value:
            raise InputValidationError(f"{label} gate drifted: {key}")


def _source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_index",
        "raw_episode_id",
        "source_annotation_sha256",
        "source_group_id",
        "source_release_manifest_sha256",
        "task_index",
        "task_instance_id",
    )
    return {key: source.get(key) for key in keys}


def _locators_by_view(locators: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(locators, list) or len(locators) != len(VIEWS):
        raise InputValidationError(f"{label} must contain exactly three camera locators")
    result: dict[str, dict[str, Any]] = {}
    for locator in locators:
        if not isinstance(locator, Mapping):
            raise InputValidationError(f"{label} contains a non-object locator")
        view = locator.get("view")
        if view not in VIEWS or view in result:
            raise InputValidationError(f"{label} has an invalid or duplicate view")
        camera = locator.get("camera_key")
        if camera != CAMERA_BY_VIEW[view]:
            raise InputValidationError(f"{label}.{view} camera key is not canonical")
        if locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED":
            raise InputValidationError(f"{label}.{view} locator status is not metadata-only")
        if locator.get("expected_fps") != 30:
            raise InputValidationError(f"{label}.{view} FPS is not 30")
        _safe_relative_path(locator.get("relative_path"), f"{label}.{view}.relative_path")
        _finite_number(locator.get("episode_start_timestamp_s"), f"{label}.{view}.episode_start_timestamp_s")
        _finite_number(locator.get("requested_timestamp_s"), f"{label}.{view}.requested_timestamp_s")
        result[view] = dict(locator)
    if set(result) != set(VIEWS):
        raise InputValidationError(f"{label} view set is incomplete")
    return result


def _locator_identity(locator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: locator.get(key)
        for key in ("camera_key", "view", "relative_path", "episode_start_timestamp_s", "expected_fps", "locator_status")
    }


def _check_source(source: Mapping[str, Any], *, expected: Mapping[str, Any], release_sha: str, label: str) -> None:
    if source.get("original_split") != "train" or source.get("source_release_manifest_sha256") != release_sha:
        raise InputValidationError(f"{label} source split/release mismatch")
    for key in ("episode_index", "raw_episode_id", "source_annotation_sha256", "source_group_id", "task_index", "task_instance_id"):
        if source.get(key) != expected.get(key):
            raise InputValidationError(f"{label} source mismatch: {key}")
    if not _is_sha256(source.get("source_annotation_sha256")) or not _is_sha256(source.get("source_group_id")):
        raise InputValidationError(f"{label} source SHA identity is malformed")


def _check_observation(event: Mapping[str, Any], *, interval: Mapping[str, Any], label: str) -> tuple[int, float]:
    observation = event.get("observation")
    if not isinstance(observation, Mapping):
        raise InputValidationError(f"{label} observation is missing")
    frame = _exact_int(observation.get("frame"), f"{label}.observation.frame")
    start = _exact_int(interval.get("start_frame"), f"{label}.interval.start_frame")
    end = _exact_int(interval.get("end_frame"), f"{label}.interval.end_frame")
    if not 0 <= start < end or not start <= frame < end:
        raise InputValidationError(f"{label} observation frame is outside its half-open interval")
    timestamp = _finite_number(observation.get("timestamp_s"), f"{label}.observation.timestamp_s")
    if abs(timestamp - frame / FPS) > 1e-9:
        raise InputValidationError(f"{label} observation timestamp does not bind local frame clock")
    return frame, timestamp


def _validate_batch_manifest(
    batch_dir: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    batch_dir = _regular_dir(batch_dir, "formal goal-state batch")
    manifest_path = _regular_file(batch_dir / "manifest.json", "formal goal-state batch manifest")
    _require_pin(_sha256(manifest_path), expected_sha256, "formal goal-state batch manifest")
    manifest = _load_json(manifest_path, "formal goal-state batch manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != BATCH_SCHEMA:
        raise InputValidationError("formal batch manifest schema is invalid")
    if manifest.get("status") != "GOAL_UNBOUND_GROUNDING_REQUIRED" or manifest.get("split") != "train":
        raise InputValidationError("formal batch is not the approved train goal-unbound batch")
    if manifest.get("usage_role") != "student_candidate" or manifest.get("private_goal_state_mode") != "goal_unbound":
        raise InputValidationError("formal batch role or private mode is invalid")
    if manifest.get("training_eligible") is not False or manifest.get("actor_event_count") != 0 or manifest.get("label_count") != 0:
        raise InputValidationError("formal batch is not non-training and label-free")
    if manifest.get("outcome_or_recovery_count") != 0:
        raise InputValidationError("formal batch contains outcome/recovery records")
    if manifest.get("training_eligible") is not False:
        raise InputValidationError("formal batch training gate is not closed")
    non_authorization = manifest.get("non_authorization")
    if not isinstance(non_authorization, Mapping) or any(non_authorization.get(key) is not False for key in (
        "attempt_outcome", "bc_or_dart_supervision", "canonical_labels_created", "corrective_action", "recovery_decision", "student_release"
    )):
        raise InputValidationError("formal batch authorization gates are not closed")
    query_stage = manifest.get("query_stage")
    if not isinstance(query_stage, Mapping) or query_stage.get("status") != "UNBOUND_NO_QUERY_OR_PLACEHOLDER":
        raise InputValidationError("formal batch unexpectedly contains query stage content")

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise InputValidationError("formal batch file receipts are missing")
    loaded: dict[str, list[dict[str, Any]]] = {}
    receipts: dict[str, Any] = {}
    for name in ("candidate_rows.jsonl", "goal_unbound_events.jsonl", "camera_native_render_requests.jsonl", "student_candidate_queue.jsonl"):
        receipt = files.get(name)
        if not isinstance(receipt, Mapping):
            raise InputValidationError(f"formal batch receipt is missing: {name}")
        rows, audit = _read_jsonl_receipt(batch_dir / name, receipt, name)
        loaded[name] = rows
        receipts[name] = audit
    if len(loaded["candidate_rows.jsonl"]) != 32 or len(loaded["goal_unbound_events.jsonl"]) != 64:
        raise InputValidationError("formal batch selection cardinalities drifted")
    if len(loaded["camera_native_render_requests.jsonl"]) != 64 or len(loaded["student_candidate_queue.jsonl"]) != 64:
        raise InputValidationError("formal batch request/queue cardinalities drifted")
    pins = manifest.get("input_pins")
    if not isinstance(pins, Mapping):
        raise InputValidationError("formal batch input pins are missing")
    return dict(manifest), receipts, loaded["candidate_rows.jsonl"], {
        "events": loaded["goal_unbound_events.jsonl"],
        "requests": loaded["camera_native_render_requests.jsonl"],
        "queue": loaded["student_candidate_queue.jsonl"],
    }, {"path": str(manifest_path), "sha256": expected_sha256, "pins": dict(pins)}


def _validate_index(
    event_index_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_event_candidates_sha256: str,
    expected_source_groups_sha256: str,
    parent_ids: set[str],
    group_ids: set[str],
    expected_release_sha: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    event_index_dir = _regular_dir(event_index_dir, "sealed full-v3 event index")
    manifest_path = _regular_file(event_index_dir / "manifest.json", "full-v3 event-index manifest")
    _require_pin(_sha256(manifest_path), expected_manifest_sha256, "full-v3 event-index manifest")
    index_manifest = _load_json(manifest_path, "full-v3 event-index manifest")
    if not isinstance(index_manifest, Mapping) or index_manifest.get("schema_version") != INDEX_SCHEMA:
        raise InputValidationError("full-v3 event-index manifest schema is invalid")
    if index_manifest.get("source_release_manifest_sha256") != expected_release_sha:
        raise InputValidationError("full-v3 event-index release pin differs from formal batch")
    files = index_manifest.get("files")
    if not isinstance(files, Mapping):
        raise InputValidationError("full-v3 event-index file receipts are missing")
    event_receipt = files.get("event_candidates.jsonl")
    group_receipt = files.get("source_groups.jsonl")
    if not isinstance(event_receipt, Mapping) or not isinstance(group_receipt, Mapping):
        raise InputValidationError("full-v3 event-index receipts are incomplete")
    if event_receipt.get("sha256") != expected_event_candidates_sha256 or group_receipt.get("sha256") != expected_source_groups_sha256:
        raise InputValidationError("full-v3 event-index receipt pins differ from formal batch")

    event_path = _regular_file(event_index_dir / "event_candidates.jsonl", "full-v3 event candidates")
    group_path = _regular_file(event_index_dir / "source_groups.jsonl", "full-v3 source groups")
    parents: dict[str, dict[str, Any]] = {}
    event_digest = hashlib.sha256()
    event_rows = 0
    with event_path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            event_digest.update(line)
            if not line.strip():
                continue
            event_rows += 1
            value = _strict_json(line, name=f"{event_path}:{number}")
            if not isinstance(value, Mapping):
                raise InputValidationError("full-v3 event row is not an object")
            event_id = value.get("event_id")
            if event_id in parent_ids:
                if event_id in parents:
                    raise InputValidationError("full-v3 event index duplicates a requested parent")
                parents[event_id] = dict(value)
    if event_path.stat().st_size != event_receipt.get("bytes") or event_rows != event_receipt.get("rows") or event_digest.hexdigest() != expected_event_candidates_sha256:
        raise InputValidationError("full-v3 event candidate bytes differ from sealed receipt")
    if set(parents) != parent_ids:
        raise InputValidationError(f"full-v3 event index is missing parent event IDs: {sorted(parent_ids - set(parents))}")

    groups: dict[str, dict[str, Any]] = {}
    group_digest = hashlib.sha256()
    group_rows = 0
    with group_path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            group_digest.update(line)
            if not line.strip():
                continue
            group_rows += 1
            value = _strict_json(line, name=f"{group_path}:{number}")
            if not isinstance(value, Mapping):
                raise InputValidationError("full-v3 source-group row is not an object")
            group_id = value.get("source_group_id")
            if group_id in group_ids:
                if group_id in groups:
                    raise InputValidationError("full-v3 source groups duplicate a requested group")
                groups[group_id] = dict(value)
    if group_path.stat().st_size != group_receipt.get("bytes") or group_rows != group_receipt.get("rows") or group_digest.hexdigest() != expected_source_groups_sha256:
        raise InputValidationError("full-v3 source-group bytes differ from sealed receipt")
    if set(groups) != group_ids:
        raise InputValidationError(f"full-v3 source groups are missing selected IDs: {sorted(group_ids - set(groups))}")
    for event_id, event in parents.items():
        if event.get("schema_version") != EVENT_SCHEMA or event.get("record_kind") != "event_candidate":
            raise InputValidationError(f"full-v3 parent {event_id} has an invalid event schema")
        if event.get("event_kind") != "ANNOTATED_SKILL_SEGMENT":
            raise InputValidationError(f"full-v3 parent {event_id} is not an annotated skill segment")
        if event.get("usage_role") != "student_candidate":
            raise InputValidationError(f"full-v3 parent {event_id} role is not student_candidate")
    for group_id, group in groups.items():
        if group.get("schema_version") != EVENT_SCHEMA or group.get("source_group_id") != group_id:
            raise InputValidationError(f"full-v3 source group {group_id} schema or identity is invalid")
        if group.get("original_split") != "train" or group.get("usage_role") != "student_candidate":
            raise InputValidationError(f"full-v3 source group {group_id} split or role is invalid")
        if group.get("source_release_manifest_sha256") != expected_release_sha:
            raise InputValidationError(f"full-v3 source group {group_id} release pin differs")
    return parents, groups, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "event_candidates_path": str(event_path),
        "event_candidates_sha256": expected_event_candidates_sha256,
        "event_candidates_rows": event_rows,
        "source_groups_path": str(group_path),
        "source_groups_sha256": expected_source_groups_sha256,
        "source_groups_rows": group_rows,
    }


def _validate_parent_binding(
    parent: Mapping[str, Any], group: Mapping[str, Any], source: Mapping[str, Any], interval: Mapping[str, Any], locators: Mapping[str, Mapping[str, Any]], *, release_sha: str, label: str
) -> None:
    _check_source(parent.get("source", {}), expected=source, release_sha=release_sha, label=f"{label}.source")
    if not _same_json(parent.get("event_interval"), interval):
        raise InputValidationError(f"{label} interval does not bind parent segment")
    parent_observation = parent.get("observation")
    if not isinstance(parent_observation, Mapping) or parent_observation.get("frame") != interval.get("start_frame"):
        raise InputValidationError(f"{label} observation does not bind parent start")
    parent_action = parent.get("action")
    if not isinstance(parent_action, Mapping) or parent_action.get("start_frame") != interval.get("start_frame"):
        raise InputValidationError(f"{label} action start does not bind parent clock")
    _locators_by_view(parent.get("video_locators"), f"{label}.video_locators")
    parent_locators = _locators_by_view(parent.get("video_locators"), f"{label}.video_locators")
    for view in VIEWS:
        if _locator_identity(parent_locators[view]) != _locator_identity(locators[view]):
            raise InputValidationError(f"{label}.{view} locator identity differs from derived anchor")
    skill_bundle = parent.get("skill_bundle")
    if not isinstance(skill_bundle, list) or not skill_bundle or not all(isinstance(skill, Mapping) for skill in skill_bundle):
        raise InputValidationError(f"{label} skill bundle is missing")
    binding = parent.get("binding")
    if not isinstance(binding, list) or not binding:
        raise InputValidationError(f"{label} binding is missing")
    member = skill_bundle[0]
    if not isinstance(member, Mapping) or _canonical_sha256(member) != source.get("_parent_skill_member_sha256"):
        raise InputValidationError(f"{label} parent skill member digest does not bind derived lineage")
    if group.get("source_group_id") != source.get("source_group_id"):
        raise InputValidationError(f"{label} source group differs from parent group")
    if group.get("source_episode_ids") != [source.get("episode_index")]:
        raise InputValidationError(f"{label} source group episode binding differs")
    if group.get("task_index") != source.get("task_index") or group.get("task_instance_id") != source.get("task_instance_id"):
        raise InputValidationError(f"{label} source group task binding differs")


def _validate_selected_batch(
    manifest: Mapping[str, Any],
    candidate_rows: list[dict[str, Any]],
    payloads: Mapping[str, list[dict[str, Any]]],
    *,
    episode_indices: tuple[int, ...],
    parents: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise InputValidationError("selected episode list must be non-empty and unique")
    if len(episode_indices) > MAX_UNIQUE_FRAMES:
        raise InputValidationError("selected parent count exceeds bounded grounding budget")
    if any(type(value) is not int or value < 0 for value in episode_indices):
        raise InputValidationError("selected episode index is invalid")
    release_sha = manifest.get("source_release_manifest_sha256")
    if not _is_sha256(release_sha):
        raise InputValidationError("formal batch release pin is malformed")
    selected_candidates = [row for row in candidate_rows if row.get("episode_index") in episode_indices]
    if len(selected_candidates) != len(episode_indices):
        raise InputValidationError("formal batch does not contain exactly one candidate per selected episode")
    by_episode: dict[int, dict[str, Any]] = {}
    for candidate in selected_candidates:
        episode = _exact_int(candidate.get("episode_index"), "candidate episode_index")
        if episode in by_episode:
            raise InputValidationError("formal batch contains duplicate selected candidates")
        _check_gate_map(
            candidate,
            f"candidate {candidate.get('candidate_id')}",
            require_status_fields=True,
            require_supervision_fields=False,
        )
        if candidate.get("structural_candidate_only") is not True or candidate.get("source_group_id") not in groups:
            raise InputValidationError("selected candidate is not a structural train candidate")
        if not isinstance(candidate.get("first_event_id"), str) or not isinstance(candidate.get("second_event_id"), str):
            raise InputValidationError("selected candidate event identities are missing")
        if candidate.get("first_event_id") not in parents:
            raise InputValidationError("selected candidate first event is not a pinned full-index parent")
        by_episode[episode] = candidate

    events = [row for row in payloads["events"] if row.get("source", {}).get("episode_index") in episode_indices]
    if len(events) != len(episode_indices) * 2:
        raise InputValidationError("selected episodes do not have exactly two goal-unbound anchors each")
    requests = {row.get("event_id"): row for row in payloads["requests"] if row.get("event_id") in {event.get("event_id") for event in events}}
    queues = {row.get("event_id"): row for row in payloads["queue"] if row.get("event_id") in {event.get("event_id") for event in events}}
    if len(requests) != len(events) or len(queues) != len(events):
        raise InputValidationError("selected goal-unbound anchors lack one-to-one requests or queue rows")

    selected_events: list[dict[str, Any]] = []
    selected_parent_ids: set[str] = set()
    phases: dict[int, set[str]] = {episode: set() for episode in episode_indices}
    for event in events:
        event_id = event.get("event_id")
        source = event.get("source")
        lineage = event.get("phase_lineage")
        interval = event.get("event_interval")
        if not isinstance(source, Mapping) or not isinstance(lineage, Mapping) or not isinstance(interval, Mapping):
            raise InputValidationError("selected derived anchor is structurally incomplete")
        episode = _exact_int(source.get("episode_index"), "derived source episode_index")
        candidate = by_episode.get(episode)
        if candidate is None:
            raise InputValidationError("selected derived anchor has no candidate row")
        if event.get("schema_version") != EVENT_SCHEMA or event.get("record_kind") != "event_candidate":
            raise InputValidationError("selected derived anchor schema is invalid")
        if event.get("usage_role") != "student_candidate":
            raise InputValidationError("selected derived anchor role is invalid")
        if event.get("training_eligible") is not None:
            raise InputValidationError("derived anchor must not invent a top-level training gate")
        if event.get("event_kind") not in ("ANNOTATED_SKILL_SEGMENT_PHASE_ENTRY", "ANNOTATED_SKILL_SEGMENT_PHASE_TERMINAL"):
            raise InputValidationError("selected derived anchor phase kind is invalid")
        _check_source(source, expected={**candidate, "source_annotation_sha256": source.get("source_annotation_sha256"), "raw_episode_id": source.get("raw_episode_id")}, release_sha=release_sha, label=f"derived anchor {event_id}.source")
        if source.get("source_group_id") != candidate.get("source_group_id") or source.get("task_index") != candidate.get("task_index") or source.get("task_instance_id") != candidate.get("task_instance_id"):
            raise InputValidationError("derived anchor source does not bind candidate row")
        if not _same_json(interval, candidate.get("first_interval")):
            raise InputValidationError("derived anchor interval does not bind candidate first segment")
        parent_id = lineage.get("parent_event_id")
        if parent_id != candidate.get("first_event_id") or parent_id not in parents:
            raise InputValidationError("derived anchor parent_event_id is not its canonical full-index parent")
        if not _same_json(lineage.get("parent_event_interval"), interval):
            raise InputValidationError("derived anchor parent interval is not exact")
        if lineage.get("goal_state_mode") != "goal_unbound" or lineage.get("observation_phase_is_not_outcome") is not True:
            raise InputValidationError("derived anchor goal-unbound lineage is invalid")
        if lineage.get("action_bc_supervision") is not False or lineage.get("outcome_supervision") is not False or lineage.get("recovery_supervision") is not False or lineage.get("dart_supervision") is not False:
            raise InputValidationError("derived anchor contains forbidden supervision")
        if lineage.get("successor_parent_event_id") is not None or lineage.get("transition_gap_frames") is not None:
            raise InputValidationError("derived anchor carries a successor or transition claim")
        parent = parents[parent_id]
        parent_skill = parent.get("skill_bundle", [None])[0] if isinstance(parent.get("skill_bundle"), list) else None
        derived_skill = event.get("skill_bundle", [None])[0] if isinstance(event.get("skill_bundle"), list) else None
        parent_identity = lineage.get("parent_skill_identity")
        if not isinstance(parent_skill, Mapping) or not isinstance(derived_skill, Mapping) or not isinstance(parent_identity, Mapping):
            raise InputValidationError("derived anchor parent skill identity is incomplete")
        if _canonical_sha256(derived_skill) != _canonical_sha256(parent_skill):
            raise InputValidationError("derived anchor skill member differs from full-index parent")
        source_with_digest = dict(source)
        source_with_digest["_parent_skill_member_sha256"] = parent_identity.get("parent_skill_member_sha256")
        locators = _locators_by_view(event.get("video_locators"), f"derived anchor {event_id}.video_locators")
        _validate_parent_binding(parent, groups[source.get("source_group_id")], source_with_digest, interval, locators, release_sha=release_sha, label=f"parent {parent_id}")
        if parent_identity.get("parent_skill_index") != 0 or parent_identity.get("skill_id") != parent_skill.get("skill_id") or parent_identity.get("skill_start_frame") != parent_skill.get("skill_start") or parent_identity.get("skill_end_frame") != parent_skill.get("skill_end") or parent_identity.get("parent_skill_member_sha256") != _canonical_sha256(parent_skill):
            raise InputValidationError("derived anchor parent skill identity does not bind full-index parent")
        frame, timestamp = _check_observation(event, interval=interval, label=f"derived anchor {event_id}")
        action = event.get("action")
        if not isinstance(action, Mapping) or action.get("start_frame") != frame:
            raise InputValidationError(f"derived anchor {event_id} action start does not bind observation clock")
        phase = "ENTRY" if event.get("event_kind").endswith("ENTRY") else "TERMINAL"
        if phase in phases[episode] or (phase == "ENTRY" and frame != interval.get("start_frame")) or (phase == "TERMINAL" and frame != interval.get("end_frame") - 1):
            raise InputValidationError("selected episode has invalid or duplicate phase anchors")
        phases[episode].add(phase)
        for view, locator in locators.items():
            expected_clock = _finite_number(locator["episode_start_timestamp_s"], f"{event_id}.{view}.episode_start") + frame / FPS
            if abs(_finite_number(locator["requested_timestamp_s"], f"{event_id}.{view}.requested") - expected_clock) > 1e-9:
                raise InputValidationError(f"derived anchor {event_id}.{view} requested clock is not frame-bound")
        request = requests[event_id]
        if request.get("schema_version") != REQUEST_SCHEMA or request.get("status") != "PENDING_RENDERER_HANDSHAKE_NO_RGB_DECODED":
            raise InputValidationError("selected render request status/schema is invalid")
        if request.get("source_identity") != _source_identity(source):
            raise InputValidationError("selected render request source identity differs")
        requested_frames = request.get("requested_frame_indices")
        if not isinstance(requested_frames, list) or not requested_frames or len(set(requested_frames)) != len(requested_frames) or any(type(value) is not int for value in requested_frames):
            raise InputValidationError("selected render request frame list is invalid")
        episode_length = _exact_int(source.get("episode_length"), "derived source episode_length")
        if request.get("actor_available_frame_indices") != requested_frames or frame not in requested_frames or any(value < 0 or value >= episode_length or value > frame for value in requested_frames):
            raise InputValidationError("selected render request is not causal actor-available data")
        if request.get("offline_review_before_frame_indices") != [] or request.get("offline_review_after_frame_indices") != []:
            raise InputValidationError("selected render request contains before/after frames")
        if not _same_json(request.get("anchor_camera_locators"), list(event.get("video_locators", []))):
            raise InputValidationError("selected render request camera locators differ from derived anchor")
        delivery = request.get("camera_delivery")
        if not isinstance(delivery, Mapping) or delivery.get("camera_native_only") is not True or delivery.get("decoded_by_selector") is not False or delivery.get("include_footer") is not False:
            raise InputValidationError("selected render request is not native-only")
        if request.get("frame_locator_resolution") != "RENDERER_MUST_LOOK_UP_EACH_REQUESTED_FRAME_FROM_FROZEN_SOURCE_METADATA":
            raise InputValidationError("selected render request locator resolution is invalid")
        queue = queues[event_id]
        if queue.get("schema_version") != QUEUE_SCHEMA or queue.get("status") != "CANDIDATE_MISSING_EVIDENCE" or queue.get("queue_kind") != "STUDENT_CANDIDATE_GOAL_STATE_GROUNDING":
            raise InputValidationError("selected queue row schema/status is invalid")
        if queue.get("immutable_split") != "train" or queue.get("usage_role") != "student_candidate" or queue.get("training_eligible") is not False:
            raise InputValidationError("selected queue row role/split is invalid")
        if queue.get("event_id") != event_id or queue.get("source_group_id") != source.get("source_group_id") or not _same_json(queue.get("event_interval"), interval):
            raise InputValidationError("selected queue row source binding differs")
        expected_phase = "GOAL_UNBOUND_ENTRY" if phase == "ENTRY" else "GOAL_UNBOUND_TERMINAL"
        if queue.get("selection_phase") != expected_phase:
            raise InputValidationError("selected queue phase does not bind derived anchor")
        windows = queue.get("temporal_windows")
        if not isinstance(windows, Mapping) or windows.get("anchor_frame") != frame or abs(_finite_number(windows.get("anchor_timestamp_s"), "queue anchor timestamp") - timestamp) > 1e-9:
            raise InputValidationError("selected queue anchor clock differs")
        actor_window = windows.get("actor_available_window")
        if not isinstance(actor_window, Mapping) or actor_window.get("sampled_frames") != requested_frames or actor_window.get("start_frame") != interval.get("start_frame"):
            raise InputValidationError("selected queue actor window differs from request")
        expected_actor_end = frame + 1 if phase == "ENTRY" else interval.get("end_frame")
        if actor_window.get("end_frame_exclusive") != expected_actor_end:
            raise InputValidationError("selected queue actor window end does not bind phase")
        for window_name in ("offline_review_before_window", "offline_review_after_window"):
            window = windows.get(window_name)
            if not isinstance(window, Mapping) or window.get("sampled_frames") != []:
                raise InputValidationError("selected queue has non-empty before/after window")
        selected_parent_ids.add(parent_id)
        selected_events.append(dict(event))

    if any(phases[episode] != {"ENTRY", "TERMINAL"} for episode in episode_indices):
        raise InputValidationError("selected episodes do not have one entry and one terminal anchor")
    unique_frame_bindings: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for event in selected_events:
        source = event["source"]
        frame = event["observation"]["frame"]
        unique_frame_bindings.setdefault((source["episode_index"], frame), []).append({
            "event_id": event["event_id"],
            "phase": "ENTRY" if event["event_kind"].endswith("ENTRY") else "TERMINAL",
            "parent_event_id": event["phase_lineage"]["parent_event_id"],
            "episode_index": source["episode_index"],
            "source_group_id": source["source_group_id"],
            "frame_index": frame,
            "timestamp_s": event["observation"]["timestamp_s"],
        })
    frame_keys = sorted(unique_frame_bindings)
    if not frame_keys or len(frame_keys) > MAX_UNIQUE_FRAMES:
        raise InputValidationError("selected unique frame plan exceeds bounded grounding budget")
    anchor_records = [
        {
            "event_id": event["event_id"],
            "parent_event_id": event["phase_lineage"]["parent_event_id"],
            "phase": "ENTRY" if event["event_kind"].endswith("ENTRY") else "TERMINAL",
            "source_identity": _source_identity(event["source"]),
            "event_interval": dict(event["event_interval"]),
            "observation": dict(event["observation"]),
        }
        for event in selected_events
    ]
    summary = {
        "schema_version": PREFLIGHT_SCHEMA,
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
        "parent_count": len(selected_parent_ids),
        "anchor_count": len(selected_events),
        "unique_frame_count": len(frame_keys),
        "camera_view_count": len(VIEWS),
        "native_png_count": len(frame_keys) * len(VIEWS),
        "unique_frames": [{"episode_index": ep, "frame_index": frame, "anchor_event_ids": [item["event_id"] for item in unique_frame_bindings[(ep, frame)]]} for ep, frame in frame_keys],
        "camera_views": list(VIEWS),
        "offline_review_before_frames": [],
        "offline_review_after_frames": [],
        "anchors": anchor_records,
    }
    return selected_events, unique_frame_bindings, [dict(row) for row in selected_events], {
        "summary": summary,
        "parents": sorted(selected_parent_ids),
        "candidates": by_episode,
        "requests": requests,
        "queues": queues,
        "frame_keys": frame_keys,
        "groups": groups,
    }


def validate_preflight(
    batch_dir: Path,
    event_index_dir: Path,
    *,
    expected_batch_manifest_sha256: str,
    expected_event_index_manifest_sha256: str,
    expected_event_candidates_sha256: str,
    expected_source_groups_sha256: str,
    episode_indices: Iterable[int],
) -> dict[str, Any]:
    manifest, payload_receipts, candidates, payloads, batch_audit = _validate_batch_manifest(batch_dir, expected_batch_manifest_sha256)
    pins = batch_audit["pins"]
    candidate_pool = pins.get("candidate_pool") if isinstance(pins, Mapping) else None
    if not isinstance(candidate_pool, Mapping):
        raise InputValidationError("formal candidate-pool pins are missing")
    candidate_pool_summary = candidate_pool.get("summary")
    if not isinstance(candidate_pool_summary, Mapping):
        raise InputValidationError("formal candidate-pool summary pins are missing")
    if not _is_sha256(candidate_pool_summary.get("metadata_sha256")) or candidate_pool_summary.get("source_groups_sha256") != expected_source_groups_sha256:
        raise InputValidationError("formal source metadata/group pin is missing or drifted")
    if pins.get("event_candidates_sha256") != expected_event_candidates_sha256 or pins.get("index_manifest_sha256") != expected_event_index_manifest_sha256:
        raise InputValidationError("formal full-index pins do not match supplied index")
    selected_episodes = tuple(episode_indices)
    candidate_for_parent = {row.get("episode_index"): row for row in candidates if row.get("episode_index") in selected_episodes}
    parent_ids = {row.get("first_event_id") for row in candidate_for_parent.values()}
    group_ids = {row.get("source_group_id") for row in candidate_for_parent.values()}
    if len(parent_ids) != len(selected_episodes) or len(group_ids) != len(selected_episodes):
        raise InputValidationError("selected formal candidates do not have unique parent/group identities")
    parents, groups, index_audit = _validate_index(
        event_index_dir,
        expected_manifest_sha256=expected_event_index_manifest_sha256,
        expected_event_candidates_sha256=expected_event_candidates_sha256,
        expected_source_groups_sha256=expected_source_groups_sha256,
        parent_ids={value for value in parent_ids if isinstance(value, str)},
        group_ids={value for value in group_ids if isinstance(value, str)},
        expected_release_sha=manifest["source_release_manifest_sha256"],
    )
    _events, _frames, _anchors, selection = _validate_selected_batch(
        manifest,
        candidates,
        payloads,
        episode_indices=selected_episodes,
        parents=parents,
        groups=groups,
    )
    summary = dict(selection["summary"])
    summary["batch_manifest"] = batch_audit
    summary["payload_files"] = payload_receipts
    summary["sealed_event_index"] = index_audit
    summary["input_pins"] = dict(pins)
    summary["parent_event_ids"] = selection["parents"]
    summary["source_group_ids"] = sorted(selection["groups"])
    summary["source_pins"] = {
        "formal_batch_manifest_sha256": batch_audit["sha256"],
        "candidate_rows_sha256": payload_receipts["candidate_rows.jsonl"]["sha256"],
        "goal_unbound_events_sha256": payload_receipts["goal_unbound_events.jsonl"]["sha256"],
        "render_requests_sha256": payload_receipts["camera_native_render_requests.jsonl"]["sha256"],
        "queue_sha256": payload_receipts["student_candidate_queue.jsonl"]["sha256"],
        "event_index_manifest_sha256": index_audit["manifest_sha256"],
        "event_candidates_sha256": index_audit["event_candidates_sha256"],
        "source_groups_sha256": index_audit["source_groups_sha256"],
        "source_release_manifest_sha256": manifest["source_release_manifest_sha256"],
        "metadata_sha256": candidate_pool_summary["metadata_sha256"],
    }
    summary["implementation"] = {
        "script": str(Path(__file__).resolve()),
        "git_commit": _git_commit(_ROOT),
        "decoder_module": str(Path(__file__).with_name("render_memlite_event_packets.py").resolve()),
        "fps": FPS,
        "max_unique_frames": MAX_UNIQUE_FRAMES,
        "max_png_count": MAX_PNG_COUNT,
        "max_output_bytes": MAX_OUTPUT_BYTES,
    }
    return {"summary": summary, "selected_events": _events, "frame_bindings": _frames, "selection": selection}


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _forbid_output(output: Path, sources: Iterable[Path]) -> None:
    resolved = output.resolve(strict=False)
    for source in sources:
        source_resolved = source.resolve(strict=True)
        if resolved == source_resolved or source_resolved in resolved.parents:
            raise InputValidationError("grounding output must be outside immutable input sources")


def _validate_decoded(decoded: Mapping[str, Any], image: Any, video: Path, requested: float, view: str) -> tuple[float, float]:
    if decoded.get("resolved_path") != str(video) or decoded.get("fps") != FPS:
        raise InputValidationError(f"decoder source or FPS mismatch for {view}")
    decoder_requested = _finite_number(decoded.get("requested_timestamp_s"), f"{view}.decoder.requested_timestamp_s")
    decoded_timestamp = _finite_number(decoded.get("decoded_timestamp_s"), f"{view}.decoded_timestamp_s")
    pts_error = _finite_number(decoded.get("pts_error_s"), f"{view}.pts_error_s")
    if abs(decoder_requested - requested) > 1e-12 or abs(decoded_timestamp - requested) > (1.0 / 60.0 + 1e-6) or abs(pts_error - abs(decoded_timestamp - requested)) > 1e-9:
        raise InputValidationError(f"decoder PTS is not bound to requested frame for {view}")
    resolution = decoded.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2 or any(type(value) is not int or value <= 0 for value in resolution):
        raise InputValidationError(f"decoder resolution is invalid for {view}")
    if list(image.size) != resolution:
        raise InputValidationError(f"decoder image dimensions do not bind receipt for {view}")
    return decoder_requested, decoded_timestamp


def _record_for_view(
    *,
    event_bindings: list[dict[str, Any]],
    event: Mapping[str, Any],
    locator: Mapping[str, Any],
    view: str,
    decoded: Mapping[str, Any],
    image: Any,
    video: Path,
    png_path: Path,
    relative: Path,
    requested_timestamp: float,
    source_pins: Mapping[str, Any],
) -> dict[str, Any]:
    decoder_requested, decoded_timestamp = _validate_decoded(decoded, image, video, requested_timestamp, view)
    return {
        "schema_version": SCRIPT_SCHEMA,
        "role": "private_verifier_only",
        "usage_role": "student_candidate",
        "split": "train",
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
        "episode_index": event["source"]["episode_index"],
        "frame_index": event["observation"]["frame"],
        "anchor_event_ids": [binding["event_id"] for binding in event_bindings],
        "parent_event_ids": sorted({binding["parent_event_id"] for binding in event_bindings}),
        "anchor_bindings": event_bindings,
        "source_pins": dict(source_pins),
        "source_identity": _source_identity(event["source"]),
        "camera_view": view,
        "camera_video_locator": dict(locator),
        "requested_timestamp_s": decoder_requested,
        "decoded_timestamp_s": decoded_timestamp,
        "pts_error_s": _finite_number(decoded.get("pts_error_s"), f"{view}.pts_error_s"),
        "decoded_source_container": {
            "resolved_path": decoded["resolved_path"],
            "bytes": decoded.get("bytes"),
            "mtime_ns": decoded.get("mtime_ns"),
            "resolution": list(image.size),
            "full_video_sha256": decoded.get("full_video_sha256"),
        },
        "png_relative_path": str(relative),
        "png_dimensions": list(image.size),
        "png_bytes": png_path.stat().st_size,
        "png_sha256": _sha256(png_path),
        "semantic_interpretation": "UNASSIGNED_PRIVATE_VISUAL_GROUNDING_ONLY",
    }


def run_grounding(
    batch_dir: Path,
    event_index_dir: Path,
    output_path: Path | None,
    *,
    raw_root: Path | None,
    expected_batch_manifest_sha256: str,
    expected_event_index_manifest_sha256: str,
    expected_event_candidates_sha256: str,
    expected_source_groups_sha256: str,
    episode_indices: Iterable[int],
    metadata_only: bool = False,
    av_backend: Any | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    preflight = validate_preflight(
        batch_dir,
        event_index_dir,
        expected_batch_manifest_sha256=expected_batch_manifest_sha256,
        expected_event_index_manifest_sha256=expected_event_index_manifest_sha256,
        expected_event_candidates_sha256=expected_event_candidates_sha256,
        expected_source_groups_sha256=expected_source_groups_sha256,
        episode_indices=episode_indices,
    )
    if metadata_only:
        if output_path is not None:
            raise InputValidationError("metadata-only preflight does not write a decoded artifact")
        return preflight["summary"]
    if output_path is None:
        raise InputValidationError("decode mode requires an output directory")
    if raw_root is None:
        raise InputValidationError("decode mode requires the frozen RGB dataset root")
    raw_root = _regular_dir(raw_root, "frozen RGB dataset root")
    output_path = Path(output_path)
    batch_dir = Path(batch_dir)
    event_index_dir = Path(event_index_dir)
    _forbid_output(output_path, (batch_dir, event_index_dir, raw_root))
    if output_path.exists():
        raise InputValidationError(f"grounding output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = _NEUTRAL._renderer() if renderer is None else renderer
    if av_backend is None:
        av_backend, _image_module, _draw_module = renderer._require_decode_dependencies(contact_sheets=False)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=str(output_path.parent)))
    records: list[dict[str, Any]] = []
    try:
        native_root = staging / "native_rgb"
        native_root.mkdir()
        for frame_spec in preflight["summary"]["unique_frames"]:
            episode = _exact_int(frame_spec["episode_index"], "preflight episode_index")
            frame = _exact_int(frame_spec["frame_index"], "preflight frame_index")
            event_bindings = [
                event
                for event in preflight["selected_events"]
                if event["source"]["episode_index"] == episode and event["observation"]["frame"] == frame
            ]
            if not event_bindings:
                raise InputValidationError("preflight frame has no derived anchor binding")
            event = event_bindings[0]
            locators = _locators_by_view(event["video_locators"], f"frame {episode}:{frame} locators")
            episode_length = _exact_int(event["source"].get("episode_length"), "source episode_length")
            for view in VIEWS:
                locator = locators[view]
                video = _resolve_video(raw_root, locator["relative_path"])
                episode_start = _finite_number(locator["episode_start_timestamp_s"], f"{view} episode start")
                requested = episode_start + frame / FPS
                episode_end = episode_start + (episode_length - 1) / FPS
                image, decoded = renderer._decode_rgb(
                    av_backend,
                    video,
                    requested,
                    episode_start_timestamp_s=episode_start,
                    episode_end_timestamp_s=episode_end,
                    actor_anchor_timestamp_s=None,
                )
                try:
                    _validate_decoded(decoded, image, video, requested, view)
                    relative = Path("native_rgb") / f"episode-{episode:05d}" / f"frame-{frame:06d}_{view}.png"
                    asset_path = staging / relative
                    asset_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(asset_path, format="PNG")
                    records.append(_record_for_view(
                        event_bindings=[
                            {
                                "event_id": binding["event_id"],
                                "phase": "ENTRY" if binding["event_kind"].endswith("ENTRY") else "TERMINAL",
                                "parent_event_id": binding["phase_lineage"]["parent_event_id"],
                                "observation_frame": binding["observation"]["frame"],
                                "observation_timestamp_s": binding["observation"]["timestamp_s"],
                            }
                            for binding in event_bindings
                        ],
                        event=event,
                        locator=locator,
                        view=view,
                        decoded=decoded,
                        image=image,
                        video=video,
                        png_path=asset_path,
                        relative=relative,
                        requested_timestamp=requested,
                        source_pins=preflight["summary"].get("source_pins", {}),
                    ))
                finally:
                    image.close()
        if len(records) != preflight["summary"]["native_png_count"] or len(records) > MAX_PNG_COUNT:
            raise InputValidationError("decoded PNG count does not match the bounded preflight")
        frames_path = staging / "frames.jsonl"
        with frames_path.open("xb") as stream:
            for record in records:
                stream.write(_json_line(record))
        final_manifest = dict(preflight["summary"])
        final_manifest.update({
            "schema_version": SCRIPT_SCHEMA,
            "status": "PRIVATE_GOAL_STATE_RGB_GROUNDING_DECODED_NONTRAINABLE",
            "decode_performed": True,
            "native_png_count": len(records),
            "frame_receipt": {"path": str(frames_path.relative_to(staging)), "sha256": _sha256(frames_path), "rows": len(records), "bytes": frames_path.stat().st_size},
            "artifact_bytes": 0,
        })
        output_bytes = -1
        for _attempt in range(4):
            (staging / "manifest.json").write_bytes(_json_line(final_manifest))
            measured_bytes = _tree_bytes(staging)
            if measured_bytes > MAX_OUTPUT_BYTES:
                raise InputValidationError("private grounding artifact exceeds 32 MiB")
            final_manifest["artifact_bytes"] = measured_bytes
            if measured_bytes == output_bytes:
                break
            output_bytes = measured_bytes
        else:
            raise InputValidationError("private grounding artifact size did not stabilize")
        (staging / "manifest.json").write_bytes(_json_line(final_manifest))
        os.replace(staging, output_path)
        return final_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _git_sha_arg(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(name, required=True, help="externally sealed SHA-256 pin")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--event-index-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-only", action="store_true", help="validate and print the exact frame/view preflight without decoding")
    parser.add_argument("--episode", type=int, action="append", required=True, dest="episodes", help="selected source episode; repeat for the approved subset")
    _git_sha_arg(parser, "--expected-batch-manifest-sha256")
    _git_sha_arg(parser, "--expected-event-index-manifest-sha256")
    _git_sha_arg(parser, "--expected-event-candidates-sha256")
    _git_sha_arg(parser, "--expected-source-groups-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        result = run_grounding(
            args.batch_dir,
            args.event_index_dir,
            args.output,
            raw_root=args.raw_root,
            expected_batch_manifest_sha256=args.expected_batch_manifest_sha256,
            expected_event_index_manifest_sha256=args.expected_event_index_manifest_sha256,
            expected_event_candidates_sha256=args.expected_event_candidates_sha256,
            expected_source_groups_sha256=args.expected_source_groups_sha256,
            episode_indices=args.episodes,
            metadata_only=args.metadata_only,
        )
    except (InputValidationError, OSError, RuntimeError, ValueError) as error:
        print(f"goal-state grounding rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
