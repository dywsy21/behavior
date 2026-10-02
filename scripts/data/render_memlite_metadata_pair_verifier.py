"""Private RGB verifier for metadata-authenticated repeated GRASP pairs.

The input rows in this module are deliberately *metadata candidates*.  A pair
window may contain the later GRASP and intervening skills, so its images are
private review evidence and are never an actor/causal packet, a label, or
training data.  Before any decoder call, the event locator is joined to the
frozen episode row (camera chunk/file and timestamps) and to an existing local
regular file.  ``METADATA_ONLY_UNRESOLVED`` is therefore not a permission to
decode by itself.

The native RGB/PTS implementation remains in ``render_memlite_event_packets``.
This file only owns the small v2 triage/frozen/event joins, explicit selection,
safe path resolution, bounded sampling, and private output bookkeeping.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCRIPT_SCHEMA = "p107-private-metadata-pair-verifier-v1"
SELECTION_SCHEMA = "p107-private-metadata-pair-selection-v1"
TRIAGE_SCHEMA = "p107-natural-retry-metadata-triage-v2"
EVENT_INDEX_SCHEMA = "memlite-event-index-v1"
EVENT_SCHEMA = "memlite-event-recovery-v1"
VIEWS = ("head", "left_wrist", "right_wrist")
CAMERA_NAMES = {
    "head": "zed_link_camera_0",
    "left_wrist": "left_realsense_link_camera_0",
    "right_wrist": "right_realsense_link_camera_0",
}
FPS = 30.0
STRIDE_FRAMES = 30
MAX_CANDIDATES = 4
MAX_LOCAL_FRAMES = 127
TRIAGE_CANDIDATE_COUNT = 32
FROZEN_ROW_COUNT = 20000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
VIDEO_RE = re.compile(
    r"^videos/observation\.rgb\.(?P<camera>[^/]+)/chunk-(?P<chunk>[0-9]+)/file-(?P<file>[0-9]+)\.mp4$"
)


def _load_legacy_helpers() -> Any:
    """Reuse the existing verifier's strict I/O and native decoder seams."""

    path = Path(__file__).with_name("render_memlite_retry_verifier.py")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"existing verifier helper is not a regular file: {path}")
    name = "_p107_metadata_pair_legacy_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load existing verifier helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_LEGACY = _load_legacy_helpers()
InputValidationError = _LEGACY.InputValidationError
_renderer = _LEGACY._renderer
_strict_json = _LEGACY._strict_json
_is_sha256 = _LEGACY._is_sha256
_regular_file = _LEGACY._regular_file
_regular_dir = _LEGACY._regular_dir
_sha256 = _LEGACY._sha256
_load_json = _LEGACY._load_json
_require_sha = _LEGACY._require_sha
_finite = _LEGACY._finite_number
_integer = _LEGACY._exact_int
_safe_relative_path = _LEGACY._safe_relative_path
_forbid_output = _LEGACY._forbid_output
_json_line = _LEGACY._json_line


def _git_commit() -> str | None:
    return _LEGACY._git_commit(Path(__file__).resolve().parents[2])


def _resolve_no_symlink(raw_root: Path, relative_path: str) -> Path:
    """Resolve an authenticated video only beneath a non-symlink raw root."""

    raw_root = _regular_dir(raw_root, "raw root")
    relative_path = _safe_relative_path(relative_path, "video relative_path")
    root = raw_root.resolve(strict=True)
    current = raw_root
    for component in PurePosixPath(relative_path).parts:
        current = current / component
        if current.is_symlink():
            raise InputValidationError(f"video locator traverses a symlink: {current}")
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError as exc:
        raise InputValidationError(f"video locator is missing: {current}") from exc
    if root not in resolved.parents or not resolved.is_file():
        raise InputValidationError(f"video locator escapes raw root or is not a regular file: {current}")
    return resolved


def _candidate_skill(candidate: Mapping[str, Any], which: str) -> Mapping[str, Any]:
    value = candidate.get(which)
    if not isinstance(value, Mapping):
        raise InputValidationError(f"candidate {candidate.get('candidate_id')} has no {which}")
    required = ("skill_idx", "skill_start", "skill_end", "skill_id", "verb", "target", "target_part", "arm", "source", "destination")
    for field in required:
        if field not in value:
            raise InputValidationError(f"candidate {candidate.get('candidate_id')} {which}.{field} is missing")
    for field in ("skill_idx", "skill_start", "skill_end", "skill_id"):
        _integer(value[field], f"candidate {candidate.get('candidate_id')} {which}.{field}")
    for field in ("verb", "target", "target_part", "arm", "source", "destination"):
        if not isinstance(value[field], str):
            raise InputValidationError(f"candidate {candidate.get('candidate_id')} {which}.{field} is not a string")
    if value["skill_start"] < 0 or value["skill_end"] <= value["skill_start"]:
        raise InputValidationError(f"candidate {candidate.get('candidate_id')} {which} interval is invalid")
    return value


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not SAFE_NAME_RE.fullmatch(candidate_id):
        raise InputValidationError("triage candidate_id is malformed")
    if candidate.get("candidate_kind") not in {"same_target_repeat_strict_nonoverlap", "short_adjacent_reapproach"}:
        raise InputValidationError(f"candidate {candidate_id} kind is not an authenticated v2 candidate")
    if candidate.get("split") != "train" or candidate.get("usage_role") != "student_candidate":
        raise InputValidationError(f"candidate {candidate_id} is not a train student_candidate")
    for field in ("training_eligible", "action_supervision", "outcome_supervision", "recovery_supervision"):
        if candidate.get(field) is not False:
            raise InputValidationError(f"candidate {candidate_id} guard drifted: {field}")
    if candidate.get("attempt_status") != "NOT_APPLICABLE" or candidate.get("outcome_status") != "NOT_APPLICABLE" or candidate.get("recovery_status") != "NOT_APPLICABLE":
        raise InputValidationError(f"candidate {candidate_id} carries an outcome/recovery status")
    if candidate.get("strict_time_order") is not True:
        raise InputValidationError(f"candidate {candidate_id} is not strict-time ordered")
    for field in ("episode_index", "raw_episode_id", "task_index", "task_instance_id"):
        _integer(candidate.get(field), f"candidate {candidate_id}.{field}")
    for field in ("source_annotation_sha256", "source_group_id", "split", "usage_role", "target", "task_name"):
        if not isinstance(candidate.get(field), str):
            raise InputValidationError(f"candidate {candidate_id}.{field} is malformed")
    first = _candidate_skill(candidate, "first_grasp")
    second = _candidate_skill(candidate, "second_grasp")
    if first["verb"] != "GRASP" or second["verb"] != "GRASP":
        raise InputValidationError(f"candidate {candidate_id} pair is not GRASP/GRASP")
    if first["target"] != second["target"] or first["target"] != candidate["target"]:
        raise InputValidationError(f"candidate {candidate_id} target identity is inconsistent")
    if candidate.get("source_first") != first["source"] or candidate.get("source_second") != second["source"]:
        raise InputValidationError(f"candidate {candidate_id} source identity is inconsistent")
    if first["skill_end"] > second["skill_start"]:
        raise InputValidationError(f"candidate {candidate_id} GRASP intervals overlap")
    if candidate.get("arm_first") != first["arm"] or candidate.get("arm_second") != second["arm"]:
        raise InputValidationError(f"candidate {candidate_id} arm metadata drifted")


def load_triage(
    path: Path,
    expected_sha256: str,
    expected_frozen_sha256: str,
    expected_release_sha256: str,
    candidate_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Stream/hash the v2 triage and retain only explicitly requested rows."""

    path = _regular_file(path, "v2 triage")
    actual = _sha256(path)
    _require_sha(actual, expected_sha256, "v2 triage")
    summary: dict[str, Any] | None = None
    candidates: dict[str, dict[str, Any]] = {}
    rows = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise InputValidationError(f"blank v2 triage line: {line_number}")
            value = _strict_json(line, name=f"{path}:{line_number}")
            if line_number == 1:
                if not isinstance(value, dict) or value.get("record_kind") != "summary" or value.get("schema_version") != TRIAGE_SCHEMA:
                    raise InputValidationError("v2 triage summary/schema is not pinned")
                summary = value
                continue
            if not isinstance(value, dict) or value.get("record_kind") == "summary":
                raise InputValidationError(f"v2 triage candidate row {line_number} is malformed")
            _validate_candidate(value)
            candidate_id = value["candidate_id"]
            if candidate_id in candidates:
                raise InputValidationError(f"duplicate v2 candidate_id: {candidate_id}")
            candidates[candidate_id] = value
            rows += 1
    if summary is None:
        raise InputValidationError("v2 triage has no summary")
    if summary.get("status") != "strict_metadata_triage_only_nontraining" or summary.get("selected_candidate_count") != TRIAGE_CANDIDATE_COUNT:
        raise InputValidationError("v2 triage summary status/count drifted")
    if summary.get("metadata_sha256") != expected_frozen_sha256 or summary.get("source_release_manifest_sha256_values") != [expected_release_sha256]:
        raise InputValidationError("v2 triage summary source pins drifted")
    status_fields = summary.get("status_fields")
    if status_fields != {"attempt_status": "NOT_APPLICABLE", "outcome_status": "NOT_APPLICABLE", "recovery_status": "NOT_APPLICABLE", "training_eligible": False}:
        raise InputValidationError("v2 triage status fields drifted")
    if rows != TRIAGE_CANDIDATE_COUNT or len(candidates) != TRIAGE_CANDIDATE_COUNT:
        raise InputValidationError(f"v2 triage does not contain exactly {TRIAGE_CANDIDATE_COUNT} unique candidates")
    if candidate_ids is not None:
        if not candidate_ids or len(candidate_ids) > MAX_CANDIDATES:
            raise InputValidationError(f"selection must explicitly name 1..{MAX_CANDIDATES} candidates")
        missing = candidate_ids - set(candidates)
        if missing:
            raise InputValidationError(f"selection names unknown candidate(s): {sorted(missing)}")
        candidates = {key: candidates[key] for key in candidate_ids}
    return {"path": str(path), "sha256": actual, "summary": summary, "row_count": rows}, candidates


def _candidate_pair_key(skill: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(skill[field] for field in ("skill_idx", "skill_start", "skill_end", "skill_id", "verb", "target", "target_part", "arm", "source", "destination"))


def _find_frozen_skill(record: Mapping[str, Any], skill: Mapping[str, Any], candidate_id: str) -> None:
    matches: list[Mapping[str, Any]] = []
    for segment in record.get("segments", []):
        if not isinstance(segment, Mapping):
            raise InputValidationError(f"frozen episode {candidate_id} has malformed segment")
        for item in segment.get("skills", []):
            if isinstance(item, Mapping) and _candidate_pair_key(item) == _candidate_pair_key(skill):
                matches.append(item)
    if len(matches) != 1:
        raise InputValidationError(f"frozen episode {candidate_id} has {len(matches)} exact matches for GRASP skill")


def _validate_frozen_row(record: Mapping[str, Any], candidate: Mapping[str, Any], expected_release_sha256: str) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    row = record.get("row")
    if not isinstance(row, Mapping):
        raise InputValidationError(f"frozen row missing nested row for {candidate_id}")
    for field in ("episode_index", "raw_episode_id", "task_index", "task_instance_id", "length", "dataset_from_index", "dataset_to_index"):
        _integer(row.get(field), f"frozen {candidate_id}.row.{field}")
    for field in ("episode_index", "raw_episode_id", "task_index", "task_instance_id"):
        if row[field] != candidate[field]:
            raise InputValidationError(f"frozen {candidate_id} identity mismatch: {field}")
    length = row["length"]
    if length <= 0 or row["dataset_to_index"] - row["dataset_from_index"] != length:
        raise InputValidationError(f"frozen {candidate_id} global half-open range is invalid")
    if record.get("annotation_sha256") != candidate["source_annotation_sha256"] or record.get("split") != "train":
        raise InputValidationError(f"frozen {candidate_id} annotation/split mismatch")
    if record.get("task_name") != candidate["task_name"]:
        raise InputValidationError(f"frozen {candidate_id} task name mismatch")
    if not isinstance(row.get("annotation_path"), str):
        raise InputValidationError(f"frozen {candidate_id} annotation path missing")
    _find_frozen_skill(record, _candidate_skill(candidate, "first_grasp"), candidate_id)
    _find_frozen_skill(record, _candidate_skill(candidate, "second_grasp"), candidate_id)
    cameras: dict[str, dict[str, Any]] = {}
    for view, camera in CAMERA_NAMES.items():
        prefix = f"videos/observation.rgb.{camera}"
        fields = {
            "chunk_index": f"{prefix}/chunk_index",
            "file_index": f"{prefix}/file_index",
            "from_timestamp": f"{prefix}/from_timestamp",
            "to_timestamp": f"{prefix}/to_timestamp",
        }
        chunk = _integer(row.get(fields["chunk_index"]), f"frozen {candidate_id} {view} chunk_index")
        file_index = _integer(row.get(fields["file_index"]), f"frozen {candidate_id} {view} file_index")
        start = _finite(row.get(fields["from_timestamp"]), f"frozen {candidate_id} {view} from_timestamp")
        end = _finite(row.get(fields["to_timestamp"]), f"frozen {candidate_id} {view} to_timestamp")
        if chunk < 0 or file_index < 0 or end <= start or abs((end - start) - length / FPS) > 1e-5:
            raise InputValidationError(f"frozen {candidate_id} {view} camera clock/chunk is inconsistent")
        cameras[view] = {"camera": camera, "chunk_index": chunk, "file_index": file_index, "from_timestamp": start, "to_timestamp": end}
    return {"record": dict(record), "row": dict(row), "length": length, "cameras": cameras}


def load_frozen_rows(
    path: Path,
    expected_sha256: str,
    candidates: Mapping[str, Mapping[str, Any]],
    expected_release_sha256: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Hash the complete frozen metadata while retaining only selected episodes."""

    path = _regular_file(path, "frozen episodes metadata")
    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for candidate in candidates.values():
        by_episode.setdefault(candidate["episode_index"], []).append(candidate)
    duplicate_episodes = {episode: rows for episode, rows in by_episode.items() if len(rows) > 1}
    if duplicate_episodes:
        names = sorted(candidate["candidate_id"] for rows in duplicate_episodes.values() for candidate in rows)
        raise InputValidationError(
            "selected pair candidates sharing one episode_index are rejected until every pair has an independent frozen join: "
            + ",".join(names)
        )
    wanted = {episode: rows[0] for episode, rows in by_episode.items()}
    found: dict[int, dict[str, Any]] = {}
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.strip():
                raise InputValidationError(f"blank frozen metadata line: {line_number}")
            value = _strict_json(line, name=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise InputValidationError(f"frozen metadata row {line_number} is not an object")
            row = value.get("row")
            episode_index = row.get("episode_index") if isinstance(row, Mapping) else None
            if episode_index in wanted:
                if episode_index in found:
                    raise InputValidationError(f"duplicate frozen episode_index: {episode_index}")
                found[episode_index] = _validate_frozen_row(value, wanted[episode_index], expected_release_sha256)
            rows += 1
    actual = digest.hexdigest()
    _require_sha(actual, expected_sha256, "frozen episodes metadata")
    if rows != FROZEN_ROW_COUNT or set(found) != set(wanted):
        raise InputValidationError(f"frozen metadata row/selection count mismatch: rows={rows}, selected={len(found)}/{len(wanted)}")
    return {"path": str(path), "sha256": actual, "row_count": rows}, found


def _event_key(event: Mapping[str, Any]) -> tuple[Any, ...] | None:
    bundle = event.get("skill_bundle")
    if not isinstance(bundle, list) or len(bundle) != 1 or not isinstance(bundle[0], Mapping):
        return None
    skill = bundle[0]
    fields = ("skill_idx", "skill_start", "skill_end", "skill_id", "verb", "target", "target_part", "arm", "source", "destination")
    if any(field not in skill for field in fields):
        return None
    return tuple(skill[field] for field in fields)


def _wanted_event_keys(candidates: Mapping[str, Mapping[str, Any]]) -> dict[tuple[Any, ...], tuple[str, str]]:
    wanted: dict[tuple[Any, ...], tuple[str, str]] = {}
    for candidate_id, candidate in candidates.items():
        episode = candidate["episode_index"]
        for which in ("first_grasp", "second_grasp"):
            skill = _candidate_skill(candidate, which)
            key = (episode,) + _candidate_pair_key(skill)
            if key in wanted:
                raise InputValidationError(f"duplicate wanted event key for {candidate_id}/{which}")
            wanted[key] = (candidate_id, which)
    return wanted


def _validate_event(
    event: Mapping[str, Any], candidate: Mapping[str, Any], frozen: Mapping[str, Any], which: str, expected_release_sha256: str,
) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    if event.get("schema_version") != EVENT_SCHEMA or event.get("record_kind") != "event_candidate":
        raise InputValidationError(f"event for {candidate_id}/{which} schema drifted")
    if event.get("usage_role") != "student_candidate":
        raise InputValidationError(f"event for {candidate_id}/{which} is not student_candidate")
    if event.get("task_name") != frozen["record"].get("task_name"):
        raise InputValidationError(f"event for {candidate_id}/{which} task name mismatch")
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise InputValidationError(f"event for {candidate_id}/{which} has no top-level event_id")
    source = event.get("source")
    if not isinstance(source, Mapping):
        raise InputValidationError(f"event for {candidate_id}/{which} has no source")
    row = frozen["row"]
    exact = {
        "episode_index": row["episode_index"],
        "episode_length": frozen["length"],
        "original_split": "train",
        "raw_episode_id": row["raw_episode_id"],
        "source_annotation_sha256": candidate["source_annotation_sha256"],
        "source_group_id": candidate["source_group_id"],
        "source_release_manifest_sha256": expected_release_sha256,
        "task_index": row["task_index"],
        "task_instance_id": row["task_instance_id"],
    }
    for field, expected in exact.items():
        if source.get(field) != expected:
            raise InputValidationError(f"event for {candidate_id}/{which} source mismatch: {field}")
    skill = _candidate_skill(candidate, which)
    if _event_key(event) != _candidate_pair_key(skill):
        raise InputValidationError(f"event for {candidate_id}/{which} skill identity mismatch")
    interval = event.get("event_interval")
    if not isinstance(interval, Mapping) or interval.get("start_frame") != skill["skill_start"] or interval.get("end_frame") != skill["skill_end"]:
        raise InputValidationError(f"event for {candidate_id}/{which} interval mismatch")
    observation = event.get("observation")
    if not isinstance(observation, Mapping) or observation.get("frame") != skill["skill_start"]:
        raise InputValidationError(f"event for {candidate_id}/{which} observation frame mismatch")
    timestamp = _finite(observation.get("timestamp_s"), f"event {candidate_id}/{which} observation timestamp")
    if abs(timestamp - skill["skill_start"] / FPS) > 1e-6:
        raise InputValidationError(f"event for {candidate_id}/{which} observation clock mismatch")
    return dict(event)


def load_events(
    index_dir: Path,
    expected_manifest_sha256: str,
    expected_release_sha256: str,
    candidates: Mapping[str, Mapping[str, Any]],
    frozen_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    """Stream/hash the full sealed event index and join both skills per pair."""

    index_dir = _regular_dir(index_dir, "full-v3 event index")
    manifest_path = _regular_file(index_dir / "manifest.json", "event index manifest")
    manifest_sha = _sha256(manifest_path)
    _require_sha(manifest_sha, expected_manifest_sha256, "event index manifest")
    manifest = _load_json(manifest_path, "event index manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != EVENT_INDEX_SCHEMA or manifest.get("source_release_manifest_sha256") != expected_release_sha256:
        raise InputValidationError("event index manifest schema/release drifted")
    receipt = manifest.get("files", {}).get("event_candidates.jsonl")
    if not isinstance(receipt, Mapping) or not _is_sha256(receipt.get("sha256")) or type(receipt.get("rows")) is not int or type(receipt.get("bytes")) is not int:
        raise InputValidationError("event index receipt is malformed")
    event_path = _regular_file(index_dir / "event_candidates.jsonl", "event candidates")
    if event_path.stat().st_size != receipt["bytes"]:
        raise InputValidationError("event candidates byte count does not match receipt")
    wanted = _wanted_event_keys(candidates)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    digest = hashlib.sha256()
    rows = 0
    with event_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.strip():
                raise InputValidationError(f"blank event index line: {line_number}")
            value = _strict_json(line, name=f"{event_path}:{line_number}")
            if not isinstance(value, Mapping):
                raise InputValidationError(f"event index row {line_number} is not an object")
            rows += 1
            source = value.get("source")
            episode = source.get("episode_index") if isinstance(source, Mapping) else None
            key = _event_key(value)
            wanted_key = (episode,) + key if episode is not None and key is not None else None
            if wanted_key in wanted:
                candidate_id, which = wanted[wanted_key]
                if (candidate_id, which) in selected:
                    raise InputValidationError(f"duplicate selected event for {candidate_id}/{which}")
                selected[(candidate_id, which)] = dict(value)
    actual_event_sha = digest.hexdigest()
    if rows != receipt["rows"] or actual_event_sha != receipt["sha256"]:
        raise InputValidationError("event candidates bytes/rows do not match sealed receipt")
    if set(selected) != set(wanted.values()):
        missing = sorted(set(wanted.values()) - set(selected))
        raise InputValidationError(f"event index is missing selected pair skill(s): {missing[:3]}")
    validated: dict[tuple[str, str], dict[str, Any]] = {}
    for key, event in selected.items():
        candidate_id, which = key
        candidate = candidates[candidate_id]
        frozen = frozen_rows[candidate["episode_index"]]
        validated[key] = _validate_event(event, candidate, frozen, which, expected_release_sha256)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "event_path": str(event_path),
        "event_file_sha256": actual_event_sha,
        "event_file_rows": rows,
        "source_release_manifest_sha256": expected_release_sha256,
    }, validated


def _resolve_camera_bindings(
    event: Mapping[str, Any], frozen: Mapping[str, Any], *, raw_root: Path | None,
) -> dict[str, dict[str, Any]]:
    """Authenticate locator path/chunk/file/clock, then optionally require bytes."""

    locators = event.get("video_locators")
    if not isinstance(locators, list) or len(locators) != len(VIEWS):
        raise InputValidationError("selected event must contain exactly three camera locators")
    by_view: dict[str, Mapping[str, Any]] = {}
    for locator in locators:
        if not isinstance(locator, Mapping) or locator.get("view") in by_view:
            raise InputValidationError("selected event camera locators are duplicate/malformed")
        view = locator.get("view")
        if view not in VIEWS:
            raise InputValidationError(f"selected event has unexpected camera view: {view}")
        by_view[view] = locator
    if set(by_view) != set(VIEWS):
        raise InputValidationError("selected event does not cover all three camera views")
    result: dict[str, dict[str, Any]] = {}
    for view in VIEWS:
        locator = by_view[view]
        metadata = frozen["cameras"][view]
        if locator.get("camera_key") != f"observation.rgb.{metadata['camera']}":
            raise InputValidationError(f"{view} camera key does not match frozen metadata")
        if locator.get("expected_fps") != 30 or locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED":
            raise InputValidationError(f"{view} locator status/fps is not the authenticated metadata-only contract")
        relative = _safe_relative_path(locator.get("relative_path"), f"{view} video relative_path")
        match = VIDEO_RE.fullmatch(relative)
        if match is None or match.group("camera") != metadata["camera"] or int(match.group("chunk")) != metadata["chunk_index"] or int(match.group("file")) != metadata["file_index"]:
            raise InputValidationError(f"{view} locator path does not match frozen camera chunk/file")
        episode_start = _finite(locator.get("episode_start_timestamp_s"), f"{view} locator start")
        if abs(episode_start - metadata["from_timestamp"]) > 1e-6:
            raise InputValidationError(f"{view} locator start does not match frozen metadata")
        observation = event.get("observation")
        if not isinstance(observation, Mapping):
            raise InputValidationError("event observation is missing")
        observation_frame = _integer(observation.get("frame"), "event observation frame")
        requested = _finite(locator.get("requested_timestamp_s"), f"{view} locator requested timestamp")
        expected_requested = episode_start + observation_frame / FPS
        if abs(requested - expected_requested) > 1e-6:
            raise InputValidationError(f"{view} locator requested timestamp does not match event frame")
        resolved: Path | None = None
        if raw_root is not None:
            resolved = _resolve_no_symlink(raw_root, relative)
        result[view] = {
            "view": view,
            "camera_key": locator["camera_key"],
            "relative_path": relative,
            "episode_start_timestamp_s": episode_start,
            "episode_end_exclusive_timestamp_s": metadata["to_timestamp"],
            "expected_fps": 30,
            "resolved_path": str(resolved) if resolved is not None else None,
            "resolution_status": "RAW_METADATA_AND_LOCAL_FILE_BOUND" if resolved is not None else "METADATA_ONLY_UNRESOLVED",
        }
    return result


def _load_selection(
    path: Path, expected_sha256: str, expected_triage_sha256: str, expected_frozen_sha256: str,
    expected_event_manifest_sha256: str, expected_release_sha256: str, candidate_ids: set[str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _regular_file(path, "private pair selection")
    actual = _sha256(path)
    _require_sha(actual, expected_sha256, "private pair selection")
    doc = _load_json(path, "private pair selection")
    if not isinstance(doc, Mapping) or doc.get("schema_version") != SELECTION_SCHEMA or doc.get("selection_kind") != "explicit_private_pair_windows":
        raise InputValidationError("private pair selection schema/kind is not pinned")
    pins = doc.get("source_pins")
    expected_pins = {
        "triage_sha256": expected_triage_sha256,
        "frozen_metadata_sha256": expected_frozen_sha256,
        "event_index_manifest_sha256": expected_event_manifest_sha256,
        "source_release_manifest_sha256": expected_release_sha256,
    }
    if not isinstance(pins, Mapping) or any(pins.get(key) != value for key, value in expected_pins.items()):
        raise InputValidationError("private pair selection source pins do not match external pins")
    rows = doc.get("candidates")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_CANDIDATES:
        raise InputValidationError(f"private pair selection must contain 1..{MAX_CANDIDATES} explicit candidates")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("candidate_id"), str) or row["candidate_id"] in seen:
            raise InputValidationError("private pair selection candidate IDs are malformed/duplicate")
        candidate_id = row["candidate_id"]
        window = row.get("window_local")
        if not isinstance(window, list) or len(window) != 2 or not all(type(item) is int for item in window) or window[0] < 0 or window[1] <= window[0]:
            raise InputValidationError(f"private pair selection window is invalid for {candidate_id}")
        if row.get("window_kind") != "first_grasp_start_to_second_grasp_end_half_open" or row.get("private_review_may_include_future_or_cross_skill") is not True:
            raise InputValidationError(f"private pair selection window/visibility guard drifted for {candidate_id}")
        if row.get("review_role") not in {"primary_structural_candidate", "metadata_target_switch_control"}:
            raise InputValidationError(f"private pair selection review role is not allowed for {candidate_id}")
        selected.append(dict(row))
        seen.add(candidate_id)
    if candidate_ids is not None and candidate_ids != seen:
        raise InputValidationError("--candidate-id set must exactly match the pinned selection rows")
    return {"path": str(path), "sha256": actual, "doc": dict(doc)}, selected


def _sample_plan(candidate: Mapping[str, Any], selection: Mapping[str, Any], length: int) -> list[dict[str, Any]]:
    first = _candidate_skill(candidate, "first_grasp")
    second = _candidate_skill(candidate, "second_grasp")
    window = selection["window_local"]
    if window != [first["skill_start"], second["skill_end"]] or window[1] > length:
        raise InputValidationError(f"selection window does not equal authenticated pair span for {candidate['candidate_id']}")
    samples = set(range(window[0], window[1], STRIDE_FRAMES))
    labels: dict[int, set[str]] = {frame: set() for frame in samples}
    explicit = {
        window[0]: "pair_window_start",
        window[1] - 1: "pair_window_end_minus_one",
        first["skill_start"]: "first_grasp_start",
        first["skill_end"] - 1: "first_grasp_end_minus_one",
        second["skill_start"]: "second_grasp_start",
        second["skill_end"] - 1: "second_grasp_end_minus_one",
    }
    samples.update(explicit)
    for frame in samples:
        if not window[0] <= frame < window[1] or not 0 <= frame < length:
            raise InputValidationError(f"sampled frame outside half-open pair window for {candidate['candidate_id']}")
        labels.setdefault(frame, set()).add(explicit.get(frame, "stride_30"))
    plan = []
    for local_frame in sorted(samples):
        role = sorted(labels.get(local_frame, {"stride_30"}))
        if local_frame in explicit:
            role = sorted(set(role) | {explicit[local_frame]})
        plan.append({
            "local_frame": local_frame,
            "dataset_global_frame": None,
            "sample_roles": role,
        })
    return plan


def _validate_selected_pairs(
    selection_rows: list[Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]], frozen_rows: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in selection_rows:
        candidate_id = row["candidate_id"]
        if candidate_id not in candidates:
            raise InputValidationError(f"selection candidate is not present in triage: {candidate_id}")
        candidate = candidates[candidate_id]
        frozen = frozen_rows.get(candidate["episode_index"])
        if frozen is None:
            raise InputValidationError(f"selection candidate has no frozen episode: {candidate_id}")
        if row["review_role"] == "metadata_target_switch_control":
            intervening = candidate.get("intervening_skills")
            if not isinstance(intervening, list) or not intervening:
                raise InputValidationError(f"control candidate {candidate_id} has no intervening skill metadata")
            target_switch = False
            for raw_skill in intervening:
                skill = _candidate_skill({"candidate_id": candidate_id, "control_skill": raw_skill}, "control_skill")
                _find_frozen_skill(frozen["record"], skill, candidate_id)
                target_switch = target_switch or skill["target"] != candidate["target"]
            if not target_switch:
                raise InputValidationError(f"control candidate {candidate_id} has no frozen intervening target switch")
        plan = _sample_plan(candidate, row, frozen["length"])
        for item in plan:
            item["dataset_global_frame"] = frozen["row"]["dataset_from_index"] + item["local_frame"]
        normalized.append({"selection": dict(row), "candidate": candidate, "frozen": frozen, "plan": plan})
    if sum(len(row["plan"]) for row in normalized) > MAX_LOCAL_FRAMES:
        raise InputValidationError(f"selection exceeds {MAX_LOCAL_FRAMES} sparse local-frame budget")
    return normalized


def _write_overviews(stage: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - decoder also requires Pillow
        raise RuntimeError("private metadata-pair overviews require Pillow") from exc
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["candidate_id"], []).append(record)
    pages: list[dict[str, Any]] = []
    for candidate_id, rows in sorted(grouped.items()):
        by_frame: dict[int, dict[str, dict[str, Any]]] = {}
        for row in rows:
            by_frame.setdefault(row["local_frame"], {})[row["camera_view"]] = row
        endpoints = sorted(by_frame)
        endpoints = endpoints if len(endpoints) <= 2 else [endpoints[0], endpoints[-1]]
        opened: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        try:
            for frame in endpoints:
                camera_rows = by_frame[frame]
                if set(camera_rows) != set(VIEWS):
                    raise InputValidationError(f"overview frame lacks one of three views: {candidate_id}/{frame}")
                images = {view: Image.open(stage / camera_rows[view]["png_relative_path"]).convert("RGB") for view in VIEWS}
                opened.append((frame, camera_rows, images))
            widths = [opened[0][2][view].width for view in VIEWS]
            row_height = max(opened[0][2][view].height for view in VIEWS) + 42
            sheet = Image.new("RGB", (sum(widths), row_height * len(opened)), "white")
            draw = ImageDraw.Draw(sheet)
            for index, (frame, camera_rows, images) in enumerate(opened):
                x = 0
                y = index * row_height
                for view in VIEWS:
                    image = images[view]
                    draw.text((x + 3, y + 2), f"{view} local={frame} global={camera_rows[view]['dataset_global_frame']}", fill="black")
                    sheet.paste(image, (x, y + 22))
                    x += image.width
            relative = Path("private_overviews") / f"{candidate_id}.png"
            output = stage / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(output, format="PNG")
            pages.append({"candidate_id": candidate_id, "relative_path": str(relative), "sha256": _sha256(output), "bytes": output.stat().st_size, "actor_packet_included": False})
        finally:
            for _frame, _rows, images in opened:
                for image in images.values():
                    image.close()
    return pages


def _record(
    selection: Mapping[str, Any], candidate: Mapping[str, Any], frozen: Mapping[str, Any], item: Mapping[str, Any],
    view: str, binding: Mapping[str, Any], decoded: Mapping[str, Any], relative: Path, png_path: Path,
) -> dict[str, Any]:
    first = _candidate_skill(candidate, "first_grasp")
    second = _candidate_skill(candidate, "second_grasp")
    return {
        "schema_version": SCRIPT_SCHEMA,
        "role": "private_verifier_only",
        "training_eligible": False,
        "action_bc_supervision": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "dart_supervision": False,
        "attempt_status": "NOT_APPLICABLE",
        "release_status": "NOT_RELEASED",
        "actor_packet_included": False,
        "candidate_id": candidate["candidate_id"],
        "review_role": selection["review_role"],
        "private_review_may_include_future_or_cross_skill": True,
        "candidate_target_raw": candidate["target"],
        "candidate_target_is_visual_instance_binding": False,
        "window_local_half_open": list(selection["window_local"]),
        "first_grasp_local_half_open": [first["skill_start"], first["skill_end"]],
        "second_grasp_local_half_open": [second["skill_start"], second["skill_end"]],
        "sample_roles": list(item["sample_roles"]),
        "local_frame": item["local_frame"],
        "dataset_global_frame": item["dataset_global_frame"],
        "camera_view": view,
        "source_episode": {
            "episode_index": candidate["episode_index"],
            "raw_episode_id": candidate["raw_episode_id"],
            "task_index": candidate["task_index"],
            "task_instance_id": candidate["task_instance_id"],
            "source_group_id": candidate["source_group_id"],
            "episode_length": frozen["length"],
            "dataset_from_index": frozen["row"]["dataset_from_index"],
            "dataset_to_index": frozen["row"]["dataset_to_index"],
            "annotation_sha256": candidate["source_annotation_sha256"],
        },
        "camera_video_locator": {
            "camera_key": binding["camera_key"],
            "camera_view": view,
            "relative_path": binding["relative_path"],
            "resolution_status": binding["resolution_status"],
            "episode_start_timestamp_s": binding["episode_start_timestamp_s"],
            "episode_end_exclusive_timestamp_s": binding["episode_end_exclusive_timestamp_s"],
            "expected_fps": 30,
        },
        "requested_timestamp_s": decoded["requested_timestamp_s"],
        "decoded_timestamp_s": decoded["decoded_timestamp_s"],
        "pts_error_s": decoded["pts_error_s"],
        "decoded_source_container": {
            "resolved_path": decoded["resolved_path"],
            "bytes": decoded["bytes"],
            "mtime_ns": decoded["mtime_ns"],
            "full_video_sha256": decoded.get("full_video_sha256"),
        },
        "png_relative_path": str(relative),
        "png_sha256": _sha256(png_path),
        "png_bytes": png_path.stat().st_size,
        "semantic_interpretation": "UNASSIGNED_PRIVATE_VISUAL_VERIFICATION_ONLY",
    }


def run_verifier(
    selection_path: Path, triage_path: Path, frozen_metadata_path: Path, event_index_dir: Path,
    raw_root: Path, output_path: Path, *, expected_selection_sha256: str, expected_triage_sha256: str,
    expected_frozen_metadata_sha256: str, expected_event_index_manifest_sha256: str,
    expected_release_sha256: str, candidate_ids: set[str] | None = None, av_backend: Any | None = None,
) -> dict[str, Any]:
    """Decode an explicit, bounded v2 selection and atomically publish private evidence."""

    selection_path = _regular_file(selection_path, "private pair selection")
    triage_path = _regular_file(triage_path, "v2 triage")
    frozen_metadata_path = _regular_file(frozen_metadata_path, "frozen metadata")
    event_index_dir = _regular_dir(event_index_dir, "full-v3 event index")
    raw_root = _regular_dir(raw_root, "raw video root")
    output_path = Path(output_path)
    _forbid_output(output_path, (selection_path.parent, triage_path.parent, frozen_metadata_path.parent, event_index_dir, raw_root))
    selection_meta, selection_rows = _load_selection(
        selection_path, expected_selection_sha256, expected_triage_sha256, expected_frozen_metadata_sha256,
        expected_event_index_manifest_sha256, expected_release_sha256, candidate_ids,
    )
    triage_meta, candidates = load_triage(
        triage_path, expected_triage_sha256, expected_frozen_metadata_sha256, expected_release_sha256,
        {row["candidate_id"] for row in selection_rows},
    )
    frozen_meta, frozen_rows = load_frozen_rows(frozen_metadata_path, expected_frozen_metadata_sha256, candidates, expected_release_sha256)
    event_meta, events = load_events(event_index_dir, expected_event_index_manifest_sha256, expected_release_sha256, candidates, frozen_rows)
    normalized = _validate_selected_pairs(selection_rows, candidates, frozen_rows)
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for key, event in events.items():
        candidate_id, _which = key
        candidate = candidates[candidate_id]
        frozen = frozen_rows[candidate["episode_index"]]
        resolved = _resolve_camera_bindings(event, frozen, raw_root=raw_root)
        existing = bindings.get(candidate_id)
        if existing is not None:
            for view in VIEWS:
                if any(existing[view].get(field) != resolved[view].get(field) for field in (
                    "camera_key", "relative_path", "episode_start_timestamp_s", "episode_end_exclusive_timestamp_s", "resolved_path"
                )):
                    raise InputValidationError(f"pair event camera binding drifted for {candidate_id}/{view}")
        else:
            bindings[candidate_id] = resolved
    renderer = _renderer()
    if av_backend is None:
        av_backend, _image, _draw = renderer._require_decode_dependencies(contact_sheets=False)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=str(output_path.parent)))
    records: list[dict[str, Any]] = []
    try:
        assets = staging / "native_rgb"
        assets.mkdir()
        for row in normalized:
            selection = row["selection"]
            candidate = row["candidate"]
            frozen = row["frozen"]
            candidate_id = candidate["candidate_id"]
            # Both authenticated GRASP event rows must resolve to the same
            # camera files/timeline for this source episode.
            camera_bindings = bindings[candidate_id]
            for item in row["plan"]:
                opened: dict[str, tuple[Any, Mapping[str, Any]]] = {}
                try:
                    for view in VIEWS:
                        binding = camera_bindings[view]
                        video = Path(binding["resolved_path"])
                        start = binding["episode_start_timestamp_s"]
                        end = binding["episode_end_exclusive_timestamp_s"] - 1.0 / FPS
                        requested = start + item["local_frame"] / FPS
                        image, decoded = renderer._decode_rgb(
                            av_backend, video, requested, episode_start_timestamp_s=start,
                            episode_end_timestamp_s=end, actor_anchor_timestamp_s=None,
                        )
                        try:
                            if decoded.get("resolved_path") != str(video) or abs(float(decoded.get("fps", 0.0)) - FPS) > 1e-9:
                                raise InputValidationError(f"native decoder identity/FPS mismatch for {candidate_id}/{view}")
                            if abs(float(decoded["decoded_timestamp_s"]) - requested) > (1.0 / 60.0 + 1e-6):
                                raise InputValidationError(f"native decoder PTS error exceeds half-frame for {candidate_id}/{view}")
                            decoder_requested = _finite(decoded.get("requested_timestamp_s"), f"{candidate_id}/{view} decoder request")
                            if abs(decoder_requested - requested) > 1e-12:
                                raise InputValidationError(f"native decoder requested timestamp drifted for {candidate_id}/{view}")
                            opened[view] = (image, decoded)
                        except BaseException:
                            image.close()
                            raise
                    for view in VIEWS:
                        binding = camera_bindings[view]
                        image, decoded = opened[view]
                        relative = Path("native_rgb") / candidate_id / f"local-{item['local_frame']:06d}_{view}.png"
                        png_path = staging / relative
                        png_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            image.save(png_path, format="PNG")
                        finally:
                            image.close()
                        records.append(_record(selection, candidate, frozen, item, view, binding, decoded, relative, png_path))
                    opened.clear()
                finally:
                    for image, _decoded in opened.values():
                        image.close()
        expected_records = sum(len(row["plan"]) for row in normalized) * len(VIEWS)
        if len(records) != expected_records:
            raise InputValidationError("private metadata-pair native PNG count drifted")
        pages = _write_overviews(staging, records)
        frames_path = staging / "frames.jsonl"
        with frames_path.open("xb") as stream:
            for record in records:
                stream.write(_json_line(record))
        manifest: dict[str, Any] = {
            "schema_version": SCRIPT_SCHEMA,
            "status": "PRIVATE_VERIFIER_RGB_DECODED_NONTRAINABLE",
            "role": "private_verifier_only",
            "training_eligible": False,
            "action_bc_supervision": False,
            "outcome_supervision": False,
            "recovery_supervision": False,
            "dart_supervision": False,
            "attempt_status": "NOT_APPLICABLE",
            "release_status": "NOT_RELEASED",
            "actor_packet_included": False,
            "actor_packet_frame_count": 0,
            "future_frames_in_actor_packet": 0,
            "private_review_may_include_future_or_cross_skill": True,
            "selection": selection_meta,
            "triage": triage_meta,
            "frozen_metadata": frozen_meta,
            "event_index": event_meta,
            "candidate_count": len(normalized),
            "event_join_count": len(events),
            "window_count": len(normalized),
            "local_frame_count": sum(len(row["plan"]) for row in normalized),
            "native_png_count": len(records),
            "overview_page_count": len(pages),
            "frames_jsonl": {"relative_path": "frames.jsonl", "sha256": _sha256(frames_path), "bytes": frames_path.stat().st_size, "rows": len(records)},
            "overview_pages": pages,
            "guards": {
                "private_verifier_only": True,
                "no_actor_or_causal_packet": True,
                "no_success_failure_recovery_inference": True,
                "no_action_bc_outcome_dart_training": True,
                "selection_windows_half_open": True,
                "sparse_frames_are_not_continuous_video": True,
                "raw_target_is_not_visual_instance_binding": True,
            },
            "implementation": {"script": str(Path(__file__).resolve()), "git_commit": _git_commit(), "stride_frames": STRIDE_FRAMES, "camera_views": list(VIEWS)},
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_json_line(manifest))
        manifest_sha = _sha256(manifest_path)
        if output_path.exists():
            raise FileExistsError(f"private verifier output appeared during run: {output_path}")
        os.rename(staging, output_path)
        return {**manifest, "manifest_sha256": manifest_sha, "output_path": str(output_path)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def metadata_preflight(
    selection_path: Path, triage_path: Path, frozen_metadata_path: Path, event_index_dir: Path, *,
    expected_selection_sha256: str, expected_triage_sha256: str, expected_frozen_metadata_sha256: str,
    expected_event_index_manifest_sha256: str, expected_release_sha256: str, candidate_ids: set[str] | None = None,
    all_triage_candidates: bool = False,
) -> dict[str, Any]:
    """Perform all 32 v2 identity/clock joins without touching video bytes."""

    selection_meta, selection_rows = _load_selection(
        selection_path, expected_selection_sha256, expected_triage_sha256, expected_frozen_metadata_sha256,
        expected_event_index_manifest_sha256, expected_release_sha256, candidate_ids,
    )
    if all_triage_candidates and candidate_ids is not None:
        raise InputValidationError("--preflight-all cannot be combined with --candidate-id")
    triage_meta, candidates = load_triage(
        triage_path, expected_triage_sha256, expected_frozen_metadata_sha256, expected_release_sha256,
        None if all_triage_candidates else {row["candidate_id"] for row in selection_rows},
    )
    frozen_meta, frozen_rows = load_frozen_rows(frozen_metadata_path, expected_frozen_metadata_sha256, candidates, expected_release_sha256)
    event_meta, events = load_events(event_index_dir, expected_event_index_manifest_sha256, expected_release_sha256, candidates, frozen_rows)
    selection_frame_count: int | None = None
    if not all_triage_candidates:
        _validate_selected_pairs(selection_rows, candidates, frozen_rows)
        selection_frame_count = sum(
            len(_sample_plan(candidates[row["candidate_id"]], row, frozen_rows[candidates[row["candidate_id"]]["episode_index"]]["length"]))
            for row in selection_rows
        )
    locator_count = 0
    binding_audit: dict[str, dict[str, dict[str, Any]]] = {}
    for (_candidate_id, _which), event in events.items():
        candidate = candidates[_candidate_id]
        frozen = frozen_rows[candidate["episode_index"]]
        resolved = _resolve_camera_bindings(event, frozen, raw_root=None)
        existing = binding_audit.get(_candidate_id)
        if existing is not None:
            for view in VIEWS:
                if any(existing[view].get(field) != resolved[view].get(field) for field in (
                    "camera_key", "relative_path", "episode_start_timestamp_s", "episode_end_exclusive_timestamp_s"
                )):
                    raise InputValidationError(f"pair event camera binding drifted for {_candidate_id}/{view}")
        else:
            binding_audit[_candidate_id] = resolved
        locator_count += len(event["video_locators"])
    return {
        "status": "METADATA_PREFLIGHT_PASS_NO_VIDEO_DECODE",
        "selection_sha256": selection_meta["sha256"],
        "triage_sha256": triage_meta["sha256"],
        "frozen_metadata_sha256": frozen_meta["sha256"],
        "event_index_manifest_sha256": event_meta["manifest_sha256"],
        "candidate_count": len(candidates),
        "selection_candidate_count": len(selection_rows),
        "selection_local_frame_count": selection_frame_count,
        "event_join_count": len(events),
        "camera_locator_count": locator_count,
        "frozen_row_count": frozen_meta["row_count"],
        "event_index_row_count": event_meta["event_file_rows"],
        "video_decoded": False,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--triage", type=Path, required=True)
    parser.add_argument("--frozen-metadata", type=Path, required=True)
    parser.add_argument("--event-index", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-triage-sha256", required=True)
    parser.add_argument("--expected-frozen-metadata-sha256", required=True)
    parser.add_argument("--expected-event-index-manifest-sha256", required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-all", action="store_true", help="metadata-only join for all 32 triage candidates")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    candidate_ids = set(args.candidate_ids) if args.candidate_ids else None
    try:
        common = {
            "selection_path": args.selection,
            "triage_path": args.triage,
            "frozen_metadata_path": args.frozen_metadata,
            "event_index_dir": args.event_index,
            "expected_selection_sha256": args.expected_selection_sha256,
            "expected_triage_sha256": args.expected_triage_sha256,
            "expected_frozen_metadata_sha256": args.expected_frozen_metadata_sha256,
            "expected_event_index_manifest_sha256": args.expected_event_index_manifest_sha256,
            "expected_release_sha256": args.expected_release_sha256,
            "candidate_ids": candidate_ids,
        }
        if args.preflight_only:
            if args.preflight_all and args.candidate_ids:
                raise InputValidationError("--preflight-all cannot be combined with --candidate-id")
            common["all_triage_candidates"] = args.preflight_all
            result = metadata_preflight(**common)
        else:
            if args.raw_root is None or args.output is None:
                raise InputValidationError("decode mode requires --raw-root and --output")
            result = run_verifier(raw_root=args.raw_root, output_path=args.output, **common)
    except (InputValidationError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"PRIVATE_METADATA_PAIR_REJECTED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("status", "output_path", "manifest_sha256", "candidate_count", "selection_candidate_count", "selection_local_frame_count", "event_join_count", "camera_locator_count", "native_png_count", "video_decoded") if key in result}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
