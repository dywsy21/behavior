#!/usr/bin/env python3
"""Private, unnamed gripper-command scan restricted to authenticated GRASP spans.

This module is intentionally separate from the original eight-source
``annotation_calibration`` pilot.  The builder joins the v2 metadata triage,
the frozen episode table, and the metadata-only event index before a Parquet
read is permitted.  The scanner then reads a selected Parquet source and
builds independent left/right raw-value RLEs *inside each authenticated GRASP
interval*.  It does not infer open/close, contact, attempt, success, failure,
or recovery, and it never joins two GRASP intervals into one action candidate.

The command is diagnostic/private-only.  A valid result is not a training
artifact and all supervision gates remain false.

``build-single-manifest`` is an explicit opt-in input mode for a sealed list
of at most eight episodes.  It carries one exact GRASP interval per episode;
it does not invent a paired retry or join spans across skills.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


def _load_legacy_helpers() -> Any:
    path = Path(__file__).with_name("inspect_memlite_gripper_retry_candidates.py")
    spec = importlib.util.spec_from_file_location("_p107_grasp_span_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy scanner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_event_protocol() -> Any:
    """Load the existing metadata event protocol without importing ML deps."""
    path = Path(__file__).resolve().parents[2] / "src/g05/data/memlite_event_protocol.py"
    spec = importlib.util.spec_from_file_location("_p107_single_grasp_event_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import event protocol: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_LEGACY = _load_legacy_helpers()
_EVENT_PROTOCOL = _load_event_protocol()
ACTION_DIM = _LEGACY.ACTION_DIM
GRIPPER_INDEX_SOURCE = _LEGACY.GRIPPER_INDEX_SOURCE
GRIPPER_INDICES = _LEGACY.GRIPPER_INDICES
InputValidationError = _LEGACY.InputValidationError
_copy_json = _LEGACY._copy_json
_read_parquet_rows = _LEGACY._read_parquet_rows
build_rle_runs = _LEGACY.build_rle_runs
find_reversal_candidates = _LEGACY.find_reversal_candidates
sha256_file = _LEGACY.sha256_file
validate_episode_rows = _LEGACY.validate_episode_rows
validate_info = _LEGACY.validate_info


INPUT_SCHEMA = "p107-grasp-span-action-input-v1"
SINGLE_SELECTION_SCHEMA = "p107-single-grasp-span-selection-v1"
SINGLE_INPUT_SCHEMA = "p107-single-grasp-span-action-input-v1"
OUTPUT_SCHEMA = "p107-grasp-span-private-output-v1"
TRIAGE_SCHEMA = "p107-natural-retry-metadata-triage-v2"
EVENT_SCHEMA = "memlite-event-recovery-v1"
RELEASE_SHA = "90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23"
EXPECTED_TRIAGE_COUNT = 32
MAX_SINGLE_EPISODES = 8
EXPECTED_FROZEN_COUNT = 20000
EXPECTED_EVENT_INDEX_SCHEMA = "memlite-event-index-v1"
MODEL_ACTION_DIM = 27
MODEL_PADDING_INDICES = (7, 8, 17, 18)
CAMERAS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputValidationError(f"{label} must be a JSON object: {path}")
    return value


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(f"{field} must be an integer")
    return int(value)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InputValidationError(f"{field} must be finite")
    return result


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or any(part == "" for part in path.parts):
        raise InputValidationError(f"{field} is not a safe relative path: {value!r}")
    return str(path)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_jsonl(path: Path, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise InputValidationError(f"cannot open {label}: {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(f"invalid {label} JSON at line {line_number}") from exc
            if not isinstance(value, dict):
                raise InputValidationError(f"{label} line {line_number} is not an object")
            yield line_number, value


def _required_sha(path: Path, expected: str, label: str) -> str:
    actual = _sha(path)
    if actual != expected:
        raise InputValidationError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return actual


def _validate_single_selection_item(item: Mapping[str, Any], order: int) -> dict[str, Any]:
    """Validate one frozen, single-span selection before any source join."""

    if not isinstance(item, Mapping):
        raise InputValidationError(f"single selection item {order} is not an object")
    selection_id = item.get("selection_id")
    if not isinstance(selection_id, str) or not selection_id:
        raise InputValidationError(f"single selection item {order} has no selection_id")
    if item.get("selection_order") != order:
        raise InputValidationError(f"single selection {selection_id} has noncanonical selection_order")
    for field in (
        "episode_index", "raw_episode_id", "task_index", "task_instance_id", "length",
        "dataset_from_index", "dataset_to_index", "segment_index", "skill_id", "skill_idx",
        "skill_start", "skill_end",
    ):
        _int(item.get(field), f"single selection {selection_id}.{field}")
    for field in (
        "task_name", "source_group_id", "source_annotation_sha256", "annotation_relative_path",
        "parquet_relative_path", "parent", "semantic", "text", "span_id", "verb", "target",
        "source", "raw_description", "event_id",
    ):
        if not isinstance(item.get(field), str) or not item[field]:
            raise InputValidationError(f"single selection {selection_id}.{field} is missing")
    if item["span_id"] != selection_id:
        raise InputValidationError(f"single selection {selection_id} span_id mismatch")
    if item.get("parent_supervised") is not False:
        raise InputValidationError(f"single selection {selection_id}.parent_supervised must be false")
    if item.get("verb") != "GRASP" or item.get("binding_confidence") != "BOUND":
        raise InputValidationError(f"single selection {selection_id} is not a bound GRASP")
    if not isinstance(item.get("parent"), str) or not item["parent"]:
        raise InputValidationError(f"single selection {selection_id}.parent is missing")
    for field in ("destination", "target_part", "arm", "binding_confidence"):
        if not isinstance(item.get(field), str):
            raise InputValidationError(f"single selection {selection_id}.{field} must be text")
    if item["arm"] != "UNSPECIFIED":
        raise InputValidationError(f"single selection {selection_id}.arm must preserve UNSPECIFIED")
    if len(item["source_annotation_sha256"]) != 64 or len(item["source_group_id"]) != 64:
        raise InputValidationError(f"single selection {selection_id} has malformed source pin")
    start, end = int(item["skill_start"]), int(item["skill_end"])
    if not (0 <= start < end <= int(item["length"])):
        raise InputValidationError(f"single selection {selection_id} has invalid half-open interval")
    if int(item["dataset_from_index"]) < 0 or int(item["dataset_to_index"]) - int(item["dataset_from_index"]) != int(item["length"]):
        raise InputValidationError(f"single selection {selection_id} has invalid global episode range")
    if not isinstance(item.get("raw_relation"), Mapping):
        raise InputValidationError(f"single selection {selection_id}.raw_relation is missing")
    if not isinstance(item.get("raw_description"), str):
        raise InputValidationError(f"single selection {selection_id}.raw_description is invalid")
    for path_field in ("annotation_relative_path", "parquet_relative_path"):
        _safe_relative(item[path_field], f"single selection {selection_id}.{path_field}")
    return dict(item)


def load_single_selection(
    path: Path,
    expected_sha256: str,
    *,
    expected_release_sha256: str = RELEASE_SHA,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and authenticate a bounded single-GRASP selection list."""

    actual_sha = _required_sha(path, expected_sha256, "single selection")
    payload = _load_object(path, "single selection")
    if payload.get("schema_version") != SINGLE_SELECTION_SCHEMA:
        raise InputValidationError("single selection schema is not v1")
    if payload.get("status") != "METADATA_ONLY_SINGLE_GRASP_SELECTION":
        raise InputValidationError("single selection status is not metadata-only")
    if payload.get("release_manifest_sha256") != expected_release_sha256:
        raise InputValidationError("single selection release SHA is not authenticated")
    constraints = payload.get("constraints")
    episodes = payload.get("episodes")
    if not isinstance(constraints, Mapping) or constraints.get("read_only") is not True:
        raise InputValidationError("single selection is not explicitly read-only")
    if constraints.get("split") != "train" or constraints.get("usage_role") != "student_candidate":
        raise InputValidationError("single selection split/role is outside the contract")
    if constraints.get("single_grasp_span_per_episode") is not True:
        raise InputValidationError("single selection does not enforce one GRASP span per episode")
    if any(
        constraints.get(field) is not expected
        for field, expected in (
            ("private_only", True), ("training_eligible", False), ("action_supervision", False),
            ("action_bc_supervision", False), ("outcome_supervision", False),
            ("recovery_supervision", False), ("dart_supervision", False),
        )
    ):
        raise InputValidationError("single selection supervision/private gates are invalid")
    if constraints.get("attempt_status") != "NOT_APPLICABLE" or constraints.get("outcome_status") != "NOT_APPLICABLE" or constraints.get("recovery_status") != "NOT_APPLICABLE":
        raise InputValidationError("single selection status fields are not NOT_APPLICABLE")
    if constraints.get("release_status") != "NOT_RELEASED":
        raise InputValidationError("single selection release status is not NOT_RELEASED")
    if not isinstance(episodes, list) or not episodes or len(episodes) > MAX_SINGLE_EPISODES:
        raise InputValidationError(f"single selection count is outside 1..{MAX_SINGLE_EPISODES}")
    if constraints.get("max_episode_count") != MAX_SINGLE_EPISODES or constraints.get("selected_episode_count", len(episodes)) not in (len(episodes), None):
        raise InputValidationError("single selection count constraint is inconsistent")
    seen_ids: set[str] = set()
    seen_episodes: set[int] = set()
    seen_groups: set[str] = set()
    selected: list[dict[str, Any]] = []
    for order, item in enumerate(episodes):
        value = _validate_single_selection_item(item, order)
        if value["selection_id"] in seen_ids or int(value["episode_index"]) in seen_episodes or value["source_group_id"] in seen_groups:
            raise InputValidationError("single selection repeats selection, episode, or source-group identity")
        seen_ids.add(value["selection_id"])
        seen_episodes.add(int(value["episode_index"]))
        seen_groups.add(value["source_group_id"])
        selected.append(value)
    return {"sha256": actual_sha, "schema_version": SINGLE_SELECTION_SCHEMA, "episode_count": len(selected)}, selected


def load_source_groups(
    path: Path,
    expected_sha256: str,
    *,
    expected_release_sha256: str = RELEASE_SHA,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Authenticate source-group split/role membership for single spans."""

    actual_sha = _required_sha(path, expected_sha256, "source groups")
    by_episode: dict[int, dict[str, Any]] = {}
    row_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(f"invalid source groups JSON at line {line_number}") from exc
            if not isinstance(value, Mapping) or value.get("schema_version") != EVENT_SCHEMA:
                raise InputValidationError(f"source groups line {line_number} has an unexpected schema")
            group = value.get("source_group_id")
            episodes = value.get("source_episode_ids")
            if not isinstance(group, str) or len(group) != 64 or not isinstance(episodes, list) or not episodes:
                raise InputValidationError(f"source groups line {line_number} has malformed identity")
            if value.get("source_release_manifest_sha256") != expected_release_sha256:
                raise InputValidationError(f"source groups line {line_number} has an unexpected release SHA")
            for episode_value in episodes:
                episode = _int(episode_value, f"source groups line {line_number}.episode_index")
                if episode in by_episode:
                    raise InputValidationError(f"source groups duplicate episode {episode}")
                by_episode[episode] = dict(value)
    if row_count != EXPECTED_FROZEN_COUNT or len(by_episode) != EXPECTED_FROZEN_COUNT:
        raise InputValidationError("source groups are not the authenticated 20,000-row seal")
    return {"sha256": actual_sha, "row_count": row_count}, by_episode


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    cid = candidate.get("candidate_id")
    if not isinstance(cid, str) or not cid:
        raise InputValidationError("triage candidate_id is missing")
    for field in ("episode_index", "raw_episode_id", "task_index", "task_instance_id"):
        _int(candidate.get(field), f"candidate {cid}.{field}")
    if candidate.get("split") != "train" or candidate.get("usage_role") != "student_candidate":
        raise InputValidationError(f"candidate {cid} is not train/student_candidate")
    for field in ("training_eligible", "action_supervision", "outcome_supervision", "recovery_supervision"):
        if candidate.get(field) is not False:
            raise InputValidationError(f"candidate {cid}.{field} must be false")
    for field in ("attempt_status", "outcome_status", "recovery_status"):
        if candidate.get(field) != "NOT_APPLICABLE":
            raise InputValidationError(f"candidate {cid}.{field} must be NOT_APPLICABLE")
    for field in ("source_annotation_sha256", "source_group_id", "target", "task_name"):
        if not isinstance(candidate.get(field), str) or not candidate[field]:
            raise InputValidationError(f"candidate {cid}.{field} is missing")
    for which in ("first_grasp", "second_grasp"):
        skill = candidate.get(which)
        if not isinstance(skill, Mapping):
            raise InputValidationError(f"candidate {cid}.{which} is missing")
        if skill.get("verb") != "GRASP":
            raise InputValidationError(f"candidate {cid}.{which} is not GRASP")
        start = _int(skill.get("skill_start"), f"candidate {cid}.{which}.skill_start")
        end = _int(skill.get("skill_end"), f"candidate {cid}.{which}.skill_end")
        if start < 0 or end <= start:
            raise InputValidationError(f"candidate {cid}.{which} interval is invalid")
        if skill.get("target") != candidate.get("target"):
            raise InputValidationError(f"candidate {cid}.{which}.target mismatch")
    first = candidate["first_grasp"]
    second = candidate["second_grasp"]
    if int(first["skill_end"]) > int(second["skill_start"]):
        raise InputValidationError(f"candidate {cid} GRASP spans overlap")


def load_triage_candidates(
    path: Path,
    expected_sha256: str,
    candidate_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Hash and retain only explicitly selected v2 candidates."""

    _required_sha(path, expected_sha256, "triage")
    wanted = set(candidate_ids)
    if not wanted or len(wanted) != len(candidate_ids):
        raise InputValidationError("candidate IDs must be non-empty and unique")
    summary: dict[str, Any] | None = None
    found: dict[str, dict[str, Any]] = {}
    row_count = 0
    for line_number, raw in _read_jsonl(path, "triage"):
        row_count += 1
        if raw.get("record_kind") == "summary":
            if summary is not None:
                raise InputValidationError("triage has duplicate summary rows")
            summary = raw
            continue
        cid = raw.get("candidate_id")
        if isinstance(cid, str) and cid in wanted:
            _validate_candidate(raw)
            if cid in found:
                raise InputValidationError(f"duplicate triage candidate: {cid}")
            found[cid] = raw
    if summary is None or summary.get("schema_version") != TRIAGE_SCHEMA:
        raise InputValidationError("triage summary schema is not v2")
    if summary.get("selected_candidate_count") != EXPECTED_TRIAGE_COUNT:
        raise InputValidationError("triage selected count is not the authenticated 32")
    missing = sorted(wanted - set(found))
    if missing:
        raise InputValidationError(f"selected triage candidates are missing: {missing}")
    return {"sha256": expected_sha256, "line_count": row_count, "summary": summary}, found


def _skill_matches(candidate_skill: Mapping[str, Any], skill: Mapping[str, Any]) -> bool:
    return (
        skill.get("verb") == "GRASP"
        and _int(skill.get("skill_start"), "frozen skill_start") == int(candidate_skill["skill_start"])
        and _int(skill.get("skill_end"), "frozen skill_end") == int(candidate_skill["skill_end"])
        and skill.get("target") == candidate_skill.get("target")
        and skill.get("source") == candidate_skill.get("source")
        and _int(skill.get("skill_idx"), "frozen skill_idx") == int(candidate_skill["skill_idx"])
    )


def _flatten_skills(frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    for segment in frozen.get("segments", []):
        if not isinstance(segment, Mapping):
            continue
        for skill in segment.get("skills", []):
            if isinstance(skill, Mapping):
                skills.append(dict(skill))
    return skills


def _camera_clock(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for camera in CAMERAS:
        prefix = f"videos/{camera}"
        chunk = _int(row.get(f"{prefix}/chunk_index"), f"{camera}.chunk_index")
        file_index = _int(row.get(f"{prefix}/file_index"), f"{camera}.file_index")
        start = _finite(row.get(f"{prefix}/from_timestamp"), f"{camera}.from_timestamp")
        end = _finite(row.get(f"{prefix}/to_timestamp"), f"{camera}.to_timestamp")
        if end <= start:
            raise InputValidationError(f"{camera} timestamp interval is invalid")
        result[camera] = {
            "chunk_index": chunk,
            "file_index": file_index,
            "from_timestamp_s": start,
            "to_timestamp_s": end,
        }
    return result


def _parquet_path(info: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    template = info.get("data_path")
    if not isinstance(template, str) or "{chunk_index" not in template or "{file_index" not in template:
        raise InputValidationError("official info data_path lacks chunk/file template")
    chunk = _int(row.get("data/chunk_index"), "data/chunk_index")
    file_index = _int(row.get("data/file_index"), "data/file_index")
    try:
        result = template.format(chunk_index=chunk, file_index=file_index)
    except (KeyError, ValueError) as exc:
        raise InputValidationError("official info data_path cannot be formatted") from exc
    return _safe_relative(result, "parquet_relative_path")


def _event_key(event: Mapping[str, Any], candidate_skill: Mapping[str, Any]) -> bool:
    source = event.get("source")
    if not isinstance(source, Mapping):
        return False
    skill_bundle = event.get("skill_bundle")
    if not isinstance(skill_bundle, list):
        return False
    return any(_skill_matches(candidate_skill, skill) for skill in skill_bundle if isinstance(skill, Mapping))


def _event_matches_source(
    event: Mapping[str, Any], candidate: Mapping[str, Any], row: Mapping[str, Any]
) -> bool:
    source = event.get("source")
    if not isinstance(source, Mapping):
        return False
    expected = {
        "episode_index": row.get("episode_index"),
        "raw_episode_id": row.get("raw_episode_id"),
        "task_index": row.get("task_index"),
        "task_instance_id": row.get("task_instance_id"),
        "source_group_id": candidate.get("source_group_id"),
        "source_annotation_sha256": candidate.get("source_annotation_sha256"),
    }
    return all(source.get(field) == value for field, value in expected.items())


def _private_event(event: Mapping[str, Any], *, retain_source_contract: bool = False) -> dict[str, Any]:
    result = {
        "event_id": event.get("event_id"),
        "event_interval": _copy_json(event.get("event_interval")),
        "event_kind": event.get("event_kind"),
        "schema_version": event.get("schema_version"),
        "bundle_id": event.get("bundle_id"),
        "observation": _copy_json(event.get("observation")),
        "action_contract": _copy_json(event.get("action")),
        "usage_role": event.get("usage_role"),
    }
    if retain_source_contract:
        result.update(
            {
                "skill_bundle": _copy_json(event.get("skill_bundle")),
                "parallel_bundle": event.get("parallel_bundle"),
                "video_locators": _copy_json(event.get("video_locators")),
            }
        )
    return result


def load_event_bindings(
    path: Path,
    expected_sha256: str,
    candidates: Mapping[str, Mapping[str, Any]],
    frozen_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Find exact event records for both candidate GRASP spans."""

    _required_sha(path, expected_sha256, "event index")
    wanted = {
        (int(candidate["episode_index"]), int(skill["skill_start"]), int(skill["skill_end"])): (cid, which)
        for cid, candidate in candidates.items()
        for which in ("first_grasp", "second_grasp")
        for skill in (candidate[which],)
    }
    found: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for _, event in _read_jsonl(path, "event index"):
        if event.get("schema_version") != EVENT_SCHEMA or event.get("record_kind") != "event_candidate":
            continue
        source = event.get("source")
        interval = event.get("event_interval")
        if not isinstance(source, Mapping) or not isinstance(interval, Mapping):
            continue
        ep = source.get("episode_index")
        try:
            key = (int(ep), int(interval.get("start_frame")), int(interval.get("end_frame")))
        except (TypeError, ValueError):
            continue
        target = wanted.get(key)
        if target is None:
            continue
        cid, which = target
        if cid in found and which in found[cid]:
            raise InputValidationError(f"duplicate event binding for {cid}/{which}")
        row = frozen_rows[int(candidates[cid]["episode_index"])]
        if not _event_matches_source(event, candidates[cid], row):
            raise InputValidationError(f"event source identity mismatch for {cid}/{which}")
        skill = candidates[cid][which]
        if not _event_key(event, skill):
            raise InputValidationError(f"event skill identity mismatch for {cid}/{which}")
        if event.get("usage_role") != "student_candidate":
            raise InputValidationError(f"event role mismatch for {cid}/{which}")
        found[cid][which] = _private_event(event)
    for cid in candidates:
        if set(found[cid]) != {"first_grasp", "second_grasp"}:
            raise InputValidationError(f"missing exact GRASP event binding for {cid}")
    return dict(found)


def _single_skill_matches(selection: Mapping[str, Any], skill: Mapping[str, Any]) -> bool:
    return (
        skill.get("verb") == "GRASP"
        and _int(skill.get("skill_id"), "event skill_id") == int(selection["skill_id"])
        and _int(skill.get("skill_idx"), "event skill_idx") == int(selection["skill_idx"])
        and _int(skill.get("skill_start"), "event skill_start") == int(selection["skill_start"])
        and _int(skill.get("skill_end"), "event skill_end") == int(selection["skill_end"])
        and skill.get("target") == selection.get("target")
        and skill.get("source") == selection.get("source")
        and skill.get("raw_relation") == selection.get("raw_relation")
    )


def load_single_event_bindings(
    path: Path,
    expected_sha256: str,
    selections: Sequence[Mapping[str, Any]],
    frozen_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Find one exact event record for every selected single GRASP span."""

    _required_sha(path, expected_sha256, "event index")
    wanted = {
        (int(selection["episode_index"]), int(selection["skill_start"]), int(selection["skill_end"])): selection
        for selection in selections
    }
    found: dict[str, dict[str, Any]] = {}
    for _, event in _read_jsonl(path, "event index"):
        if event.get("schema_version") != EVENT_SCHEMA or event.get("record_kind") != "event_candidate":
            continue
        source = event.get("source")
        interval = event.get("event_interval")
        if not isinstance(source, Mapping) or not isinstance(interval, Mapping):
            continue
        try:
            key = (int(source.get("episode_index")), int(interval.get("start_frame")), int(interval.get("end_frame")))
        except (TypeError, ValueError):
            continue
        selection = wanted.get(key)
        if selection is None:
            continue
        selection_id = str(selection["selection_id"])
        if selection_id in found:
            raise InputValidationError(f"duplicate event binding for {selection_id}")
        row = frozen_rows[int(selection["episode_index"])]
        expected = {
            "episode_index": row.get("episode_index"),
            "raw_episode_id": row.get("raw_episode_id"),
            "task_index": row.get("task_index"),
            "task_instance_id": row.get("task_instance_id"),
            "source_group_id": selection.get("source_group_id"),
            "source_annotation_sha256": selection.get("source_annotation_sha256"),
        }
        if any(source.get(field) != value for field, value in expected.items()):
            raise InputValidationError(f"event source identity mismatch for {selection_id}")
        bundle = event.get("skill_bundle")
        if not isinstance(bundle, list) or not any(
            _single_skill_matches(selection, skill) for skill in bundle if isinstance(skill, Mapping)
        ):
            raise InputValidationError(f"event skill identity mismatch for {selection_id}")
        if event.get("usage_role") != "student_candidate":
            raise InputValidationError(f"event role mismatch for {selection_id}")
        if event.get("event_id") != selection.get("event_id"):
            raise InputValidationError(f"event ID mismatch for {selection_id}")
        found[selection_id] = _private_event(event, retain_source_contract=True)
    if set(found) != {str(selection["selection_id"]) for selection in selections}:
        missing = sorted({str(selection["selection_id"]) for selection in selections} - set(found))
        raise InputValidationError(f"missing exact GRASP event bindings: {missing}")
    return found


def _source_group_matches_single(
    selection: Mapping[str, Any],
    group: Mapping[str, Any],
    *,
    expected_release_sha256: str,
) -> bool:
    """Return whether a source-group seal binds exactly one selected episode."""

    episode = int(selection["episode_index"])
    return (
        group.get("source_group_id") == selection.get("source_group_id")
        and group.get("original_split") == "train"
        and group.get("usage_role") == "student_candidate"
        and group.get("source_release_manifest_sha256") == expected_release_sha256
        and group.get("source_episode_count") == 1
        and group.get("source_episode_ids") == [episode]
        and group.get("task_index") == selection.get("task_index")
        and group.get("task_instance_id") == selection.get("task_instance_id")
    )


def _single_entry_skill(selection: Mapping[str, Any], segment: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the one selected skill after checking its segment identity."""

    skills = segment.get("skills")
    if not isinstance(skills, list):
        raise InputValidationError(f"single selection {selection['selection_id']} segment has no skills")
    matches = [skill for skill in skills if isinstance(skill, Mapping) and _single_skill_matches(selection, skill)]
    if len(matches) != 1:
        raise InputValidationError(
            f"single selection {selection['selection_id']} has {len(matches)} exact skill matches in its segment"
        )
    return _copy_json(matches[0])


def build_single_manifest(
    *,
    selection_path: Path,
    frozen_path: Path,
    source_groups_path: Path,
    event_index_path: Path,
    info_path: Path,
    output_path: Path,
    expected_selection_sha256: str,
    expected_frozen_sha256: str,
    expected_source_groups_sha256: str,
    expected_event_index_sha256: str,
    expected_info_sha256: str,
    source_root: str,
    expected_release_sha256: str = RELEASE_SHA,
) -> dict[str, Any]:
    """Build a bounded manifest containing one authenticated GRASP span/episode.

    This path deliberately does not read Parquet.  The resulting input is a
    private diagnostic manifest; it is not action, outcome, recovery, or
    training supervision.
    """

    source_root_path = Path(source_root)
    if not source_root_path.is_absolute() or not str(source_root_path):
        raise InputValidationError("source_root must be an explicit absolute path")
    selection_audit, selections = load_single_selection(
        selection_path,
        expected_selection_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    groups_audit, source_groups = load_source_groups(
        source_groups_path,
        expected_source_groups_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    frozen_sha = _required_sha(frozen_path, expected_frozen_sha256, "frozen episodes")
    info_sha = _required_sha(info_path, expected_info_sha256, "official info")
    info = _load_object(info_path, "official info")
    shapes = validate_info(info)
    if info.get("fps") != 30:
        raise InputValidationError("official info fps is not 30")

    wanted_episodes = {int(selection["episode_index"]) for selection in selections}
    frozen: dict[int, dict[str, Any]] = {}
    total_frozen = 0
    for _, raw in _read_jsonl(frozen_path, "frozen episodes"):
        total_frozen += 1
        row = raw.get("row")
        if not isinstance(row, Mapping):
            raise InputValidationError("frozen episode row is missing row object")
        episode = _int(row.get("episode_index"), "frozen episode_index")
        if episode not in wanted_episodes:
            continue
        if episode in frozen:
            raise InputValidationError(f"duplicate frozen episode {episode}")
        frozen[episode] = raw
    if total_frozen != EXPECTED_FROZEN_COUNT:
        raise InputValidationError(f"frozen episode count is not {EXPECTED_FROZEN_COUNT}: {total_frozen}")
    if set(frozen) != wanted_episodes:
        raise InputValidationError(f"frozen episodes missing: {sorted(wanted_episodes - set(frozen))}")

    entries: list[dict[str, Any]] = []
    event_rows = {episode: raw["row"] for episode, raw in frozen.items()}
    event_bindings = load_single_event_bindings(
        event_index_path,
        expected_event_index_sha256,
        selections,
        event_rows,
    )
    for selection in selections:
        selection_id = str(selection["selection_id"])
        episode = int(selection["episode_index"])
        raw_frozen = frozen[episode]
        row = raw_frozen["row"]
        group = source_groups.get(episode)
        if group is None or not _source_group_matches_single(
            selection, group, expected_release_sha256=expected_release_sha256
        ):
            raise InputValidationError(f"source-group seal mismatch for {selection_id}")
        if raw_frozen.get("annotation_sha256") != selection["source_annotation_sha256"]:
            raise InputValidationError(f"annotation SHA mismatch for {selection_id}")
        if raw_frozen.get("split") != "train":
            raise InputValidationError(f"frozen split mismatch for {selection_id}")
        for field in ("episode_index", "raw_episode_id", "task_index", "task_instance_id", "length"):
            if row.get(field) != selection.get(field):
                raise InputValidationError(f"frozen identity mismatch for {selection_id}: {field}")
        if raw_frozen.get("task_name") != selection.get("task_name"):
            raise InputValidationError(f"frozen task name mismatch for {selection_id}")
        if row.get("annotation_path") != selection.get("annotation_relative_path"):
            raise InputValidationError(f"annotation path mismatch for {selection_id}")
        if row.get("dataset_from_index") != selection.get("dataset_from_index") or row.get("dataset_to_index") != selection.get("dataset_to_index"):
            raise InputValidationError(f"frozen global range mismatch for {selection_id}")
        if int(row.get("dataset_to_index", 0)) - int(row.get("dataset_from_index", 0)) != int(row["length"]):
            raise InputValidationError(f"frozen global range is inconsistent for {selection_id}")
        parquet_relative_path = _parquet_path(info, row)
        if parquet_relative_path != selection.get("parquet_relative_path"):
            raise InputValidationError(f"parquet path mismatch for {selection_id}")

        segments = raw_frozen.get("segments")
        if not isinstance(segments, list):
            raise InputValidationError(f"frozen segments missing for {selection_id}")
        segment_index = int(selection["segment_index"])
        if not (0 <= segment_index < len(segments)) or not isinstance(segments[segment_index], Mapping):
            raise InputValidationError(f"segment index is invalid for {selection_id}")
        segment = segments[segment_index]
        if segment.get("parent") != selection.get("parent"):
            raise InputValidationError(f"parent mismatch for {selection_id}")
        if segment.get("parent_supervised") is not False or selection.get("parent_supervised") is not False:
            raise InputValidationError(f"parent supervision gate is invalid for {selection_id}")
        if segment.get("semantic") != selection.get("semantic") or segment.get("text") != selection.get("text"):
            raise InputValidationError(f"segment text/semantic mismatch for {selection_id}")
        skill = _single_entry_skill(selection, segment)
        entry = {
            "selection_order": int(selection["selection_order"]),
            "selection_id": selection_id,
            "candidate_id": selection_id,
            "candidate_kind": "single_grasp_span",
            "episode_index": episode,
            "raw_episode_id": int(row["raw_episode_id"]),
            "task_index": int(row["task_index"]),
            "task_instance_id": int(row["task_instance_id"]),
            "task_name": str(raw_frozen["task_name"]),
            "source_group_id": str(selection["source_group_id"]),
            "source_annotation_sha256": str(selection["source_annotation_sha256"]),
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
            "outcome_status": "NOT_APPLICABLE",
            "recovery_status": "NOT_APPLICABLE",
            "release_status": "NOT_RELEASED",
            "parent_goal": str(selection["parent"]),
            "parent_supervised": False,
            "parquet_relative_path": parquet_relative_path,
            "annotation_relative_path": _safe_relative(row["annotation_path"], f"{selection_id}.annotation_path"),
            "length": int(row["length"]),
            "dataset_from_index": int(row["dataset_from_index"]),
            "dataset_to_index": int(row["dataset_to_index"]),
            "camera_clock": _camera_clock(row),
            "grasp_spans": [
                {
                    "span_id": selection_id,
                    "frame_interval": [int(selection["skill_start"]), int(selection["skill_end"])],
                    "segment_index": segment_index,
                    "parent": str(selection["parent"]),
                    "parent_supervised": False,
                    "semantic": str(selection["semantic"]),
                    "text": str(selection["text"]),
                    "skill": skill,
                }
            ],
            "event_bindings": {selection_id: event_bindings[selection_id]},
        }
        entries.append(entry)

    manifest: dict[str, Any] = {
        "schema_version": SINGLE_INPUT_SCHEMA,
        "status": "AUTHENTICATED_PRIVATE_SINGLE_GRASP_ACTION_SCAN_INPUT",
        "release_manifest_sha256": expected_release_sha256,
        "constraints": {
            "read_only": True,
            "split": "train",
            "usage_role": "student_candidate",
            "max_episode_count": MAX_SINGLE_EPISODES,
            "selected_episode_count": len(entries),
            "selected_span_count": len(entries),
            "single_grasp_span_per_episode": True,
            "grasp_spans_only": True,
            "raw_gripper_values_unnamed": True,
            "cross_span_join_forbidden": True,
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
        "official_info": {
            "path": str(info_path),
            "sha256": info_sha,
            "shapes": shapes,
            "fps": info["fps"],
            "data_path": info["data_path"],
        },
        "frozen_source": {"path": str(frozen_path), "sha256": frozen_sha, "row_count": total_frozen},
        "source_groups_source": {
            "path": str(source_groups_path),
            "sha256": groups_audit["sha256"],
            "row_count": groups_audit["row_count"],
            "schema": EVENT_SCHEMA,
        },
        "single_selection_source": {
            "path": str(selection_path),
            "sha256": selection_audit["sha256"],
            "schema": SINGLE_SELECTION_SCHEMA,
            "selected_episode_count": selection_audit["episode_count"],
        },
        "event_index_source": {
            "path": str(event_index_path),
            "sha256": expected_event_index_sha256,
            "schema": EXPECTED_EVENT_INDEX_SCHEMA,
        },
        "official_snapshot_root": str(source_root_path),
        "gripper_index_contract": {"indices": dict(GRIPPER_INDICES), "source": GRIPPER_INDEX_SOURCE, "polarity": "UNPROVEN"},
        "episodes": entries,
    }
    validate_single_manifest(manifest)
    _atomic_json(output_path, manifest)
    return manifest


def build_manifest(
    *,
    triage_path: Path,
    frozen_path: Path,
    event_index_path: Path,
    info_path: Path,
    output_path: Path,
    expected_triage_sha256: str,
    expected_frozen_sha256: str,
    expected_event_index_sha256: str,
    expected_info_sha256: str,
    candidate_ids: Sequence[str],
    source_root: str,
    expected_release_sha256: str = RELEASE_SHA,
) -> dict[str, Any]:
    """Build one authenticated private input manifest without reading Parquet."""

    if not candidate_ids or len(candidate_ids) > EXPECTED_TRIAGE_COUNT:
        raise InputValidationError("select one through 32 explicit candidate IDs")
    source_root_path = Path(source_root)
    if not source_root_path.is_absolute() or not str(source_root_path):
        raise InputValidationError("source_root must be an explicit absolute path")
    frozen_sha = _required_sha(frozen_path, expected_frozen_sha256, "frozen episodes")
    info_sha = _required_sha(info_path, expected_info_sha256, "official info")
    info = _load_object(info_path, "official info")
    shapes = validate_info(info)
    if info.get("fps") != 30:
        raise InputValidationError("official info fps is not 30")
    triage_audit, candidates = load_triage_candidates(triage_path, expected_triage_sha256, candidate_ids)
    selected_episode_ids = [int(candidates[cid]["episode_index"]) for cid in candidate_ids]
    selected_group_ids = [str(candidates[cid]["source_group_id"]) for cid in candidate_ids]
    if len(set(selected_episode_ids)) != len(selected_episode_ids):
        raise InputValidationError("selected candidates contain duplicate episode_index")
    if len(set(selected_group_ids)) != len(selected_group_ids):
        raise InputValidationError("selected candidates contain duplicate source_group_id")
    frozen: dict[int, dict[str, Any]] = {}
    total_frozen = 0
    wanted_episodes = {int(candidate["episode_index"]) for candidate in candidates.values()}
    for _, raw in _read_jsonl(frozen_path, "frozen episodes"):
        total_frozen += 1
        row = raw.get("row")
        if not isinstance(row, Mapping):
            raise InputValidationError("frozen episode row is missing row object")
        episode = _int(row.get("episode_index"), "frozen episode_index")
        if episode not in wanted_episodes:
            continue
        if episode in frozen:
            raise InputValidationError(f"duplicate frozen episode {episode}")
        frozen[episode] = raw
    if total_frozen != EXPECTED_FROZEN_COUNT:
        raise InputValidationError(f"frozen episode count is not {EXPECTED_FROZEN_COUNT}: {total_frozen}")
    if set(frozen) != wanted_episodes:
        raise InputValidationError(f"frozen episodes missing: {sorted(wanted_episodes - set(frozen))}")

    entries: list[dict[str, Any]] = []
    for order, cid in enumerate(candidate_ids):
        candidate = candidates[cid]
        raw_frozen = frozen[int(candidate["episode_index"])]
        row = raw_frozen["row"]
        if raw_frozen.get("annotation_sha256") != candidate["source_annotation_sha256"]:
            raise InputValidationError(f"annotation SHA mismatch for {cid}")
        if raw_frozen.get("split") != "train":
            raise InputValidationError(f"frozen split mismatch for {cid}")
        for left, right in (
            ("raw_episode_id", "raw_episode_id"),
            ("task_index", "task_index"),
            ("task_instance_id", "task_instance_id"),
        ):
            if row.get(left) != candidate.get(right):
                raise InputValidationError(f"frozen identity mismatch for {cid}: {left}")
        if row.get("dataset_to_index", 0) - row.get("dataset_from_index", 0) != row.get("length"):
            raise InputValidationError(f"frozen global range mismatch for {cid}")
        frozen_skills = _flatten_skills(raw_frozen)
        for which in ("first_grasp", "second_grasp"):
            if not any(_skill_matches(candidate[which], skill) for skill in frozen_skills):
                raise InputValidationError(f"{cid} {which} is not exact in frozen skills")
        entries.append(
            {
                "selection_order": order,
                "candidate_id": cid,
                "candidate_kind": candidate["candidate_kind"],
                "episode_index": int(row["episode_index"]),
                "raw_episode_id": int(row["raw_episode_id"]),
                "task_index": int(row["task_index"]),
                "task_instance_id": int(row["task_instance_id"]),
                "task_name": candidate["task_name"],
                "source_group_id": candidate["source_group_id"],
                "source_annotation_sha256": candidate["source_annotation_sha256"],
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
                "parquet_relative_path": _parquet_path(info, row),
                "annotation_relative_path": _safe_relative(row["annotation_path"], f"{cid}.annotation_path"),
                "length": int(row["length"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "camera_clock": _camera_clock(row),
                "grasp_spans": [
                    {
                        "span_id": "first_grasp",
                        "frame_interval": [int(candidate["first_grasp"]["skill_start"]), int(candidate["first_grasp"]["skill_end"])],
                        "skill": _copy_json(candidate["first_grasp"]),
                    },
                    {
                        "span_id": "second_grasp",
                        "frame_interval": [int(candidate["second_grasp"]["skill_start"]), int(candidate["second_grasp"]["skill_end"])],
                        "skill": _copy_json(candidate["second_grasp"]),
                    },
                ],
            }
        )
    event_bindings = load_event_bindings(event_index_path, expected_event_index_sha256, candidates, {ep: data["row"] for ep, data in frozen.items()})
    for entry in entries:
        entry["event_bindings"] = event_bindings[entry["candidate_id"]]
    manifest: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA,
        "status": "AUTHENTICATED_PRIVATE_ACTION_SCAN_INPUT",
        "release_manifest_sha256": expected_release_sha256,
        "constraints": {
            "read_only": True,
            "max_candidates": EXPECTED_TRIAGE_COUNT,
            "selected_candidate_count": len(entries),
            "grasp_spans_only": True,
            "raw_gripper_values_unnamed": True,
            "cross_span_join_forbidden": True,
            "private_only": True,
            "training_eligible": False,
            "action_bc_supervision": False,
            "outcome_supervision": False,
            "recovery_supervision": False,
            "dart_supervision": False,
        },
        "official_info": {"path": str(info_path), "sha256": info_sha, "shapes": shapes, "fps": info["fps"], "data_path": info["data_path"]},
        "frozen_source": {"path": str(frozen_path), "sha256": frozen_sha, "row_count": total_frozen},
        "triage_source": {"path": str(triage_path), "sha256": expected_triage_sha256, "line_count": triage_audit["line_count"], "summary_schema": TRIAGE_SCHEMA},
        "event_index_source": {"path": str(event_index_path), "sha256": expected_event_index_sha256, "schema": EXPECTED_EVENT_INDEX_SCHEMA},
        "official_snapshot_root": str(source_root_path),
        "gripper_index_contract": {"indices": dict(GRIPPER_INDICES), "source": GRIPPER_INDEX_SOURCE, "polarity": "UNPROVEN"},
        "episodes": entries,
    }
    # Run exactly the same fail-closed gate used by scan before publishing any
    # manifest.  A malformed input must not leave a seemingly usable artifact.
    validate_manifest(manifest)
    _atomic_json(output_path, manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != INPUT_SCHEMA:
        raise InputValidationError("unexpected GRASP-span manifest schema")
    if manifest.get("release_manifest_sha256") != RELEASE_SHA:
        raise InputValidationError("manifest release SHA is not the authenticated release")
    constraints = manifest.get("constraints")
    entries = manifest.get("episodes")
    if not isinstance(constraints, Mapping) or constraints.get("read_only") is not True:
        raise InputValidationError("manifest is not explicitly read-only")
    if constraints.get("grasp_spans_only") is not True or constraints.get("raw_gripper_values_unnamed") is not True:
        raise InputValidationError("manifest does not enforce unnamed GRASP-span scanning")
    for field, expected in (
        ("private_only", True),
        ("action_bc_supervision", False),
        ("outcome_supervision", False),
        ("recovery_supervision", False),
        ("dart_supervision", False),
        ("cross_span_join_forbidden", True),
    ):
        if constraints.get(field) is not expected:
            raise InputValidationError(f"manifest {field} guard is not {expected}")
    snapshot_root = manifest.get("official_snapshot_root")
    if not isinstance(snapshot_root, str) or not Path(snapshot_root).is_absolute():
        raise InputValidationError("manifest official_snapshot_root must be absolute")
    info = manifest.get("official_info")
    if not isinstance(info, Mapping) or not isinstance(info.get("path"), str) or not isinstance(info.get("sha256"), str):
        raise InputValidationError("manifest official_info pin is missing")
    for source_name in ("frozen_source", "triage_source", "event_index_source"):
        source = manifest.get(source_name)
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str) or not isinstance(source.get("sha256"), str):
            raise InputValidationError(f"manifest {source_name} pin is missing")
    if not isinstance(entries, list) or not entries or len(entries) > EXPECTED_TRIAGE_COUNT:
        raise InputValidationError("manifest episode count is outside 1..32")
    seen_episode: set[int] = set()
    seen_group: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise InputValidationError("manifest episode is not an object")
        episode = _int(entry.get("episode_index"), "manifest episode_index")
        group = entry.get("source_group_id")
        if episode in seen_episode or not isinstance(group, str) or group in seen_group:
            raise InputValidationError("duplicate source identity in GRASP-span manifest")
        seen_episode.add(episode)
        seen_group.add(group)
        if entry.get("immutable_split") != "train" or entry.get("usage_role") != "student_candidate":
            raise InputValidationError(f"episode {episode} role is not train/student_candidate")
        for field in ("training_eligible", "action_supervision", "outcome_supervision", "recovery_supervision", "dart_supervision"):
            if entry.get(field) is not False:
                raise InputValidationError(f"episode {episode}.{field} must be false")
        if entry.get("attempt_status") != "NOT_APPLICABLE":
            raise InputValidationError(f"episode {episode} attempt status is not applicable")
        _safe_relative(entry.get("parquet_relative_path"), f"episode {episode}.parquet_relative_path")
        spans = entry.get("grasp_spans")
        if not isinstance(spans, list) or len(spans) != 2:
            raise InputValidationError(f"episode {episode} must have exactly two GRASP spans")
        length = _int(entry.get("length"), f"episode {episode}.length")
        by_span_id: dict[str, Mapping[str, Any]] = {}
        for span in spans:
            if not isinstance(span, Mapping) or not isinstance(span.get("span_id"), str):
                raise InputValidationError(f"episode {episode} has malformed span identity")
            span_id = str(span["span_id"])
            if span_id in by_span_id:
                raise InputValidationError(f"episode {episode} has duplicate span_id {span_id}")
            by_span_id[span_id] = span
            interval = span.get("frame_interval") if isinstance(span, Mapping) else None
            if not isinstance(interval, list) or len(interval) != 2:
                raise InputValidationError(f"episode {episode} has malformed GRASP interval")
            start, end = _int(interval[0], "GRASP start"), _int(interval[1], "GRASP end")
            if not (0 <= start < end <= length):
                raise InputValidationError(f"episode {episode} GRASP interval is outside episode")
            skill = span.get("skill")
            if not isinstance(skill, Mapping) or skill.get("verb") != "GRASP":
                raise InputValidationError(f"episode {episode} {span_id} skill is not GRASP")
            if skill.get("skill_start") != start or skill.get("skill_end") != end:
                raise InputValidationError(f"episode {episode} {span_id} skill interval mismatch")
        if set(by_span_id) != {"first_grasp", "second_grasp"}:
            raise InputValidationError(f"episode {episode} span IDs are not first/second GRASP")
        first_interval = by_span_id["first_grasp"]["frame_interval"]
        second_interval = by_span_id["second_grasp"]["frame_interval"]
        if int(first_interval[1]) > int(second_interval[0]):
            raise InputValidationError(f"episode {episode} GRASP spans overlap or are reversed")
        if not isinstance(entry.get("event_bindings"), Mapping):
            raise InputValidationError(f"episode {episode} has no event bindings")
        if set(entry["event_bindings"]) != {"first_grasp", "second_grasp"}:
            raise InputValidationError(f"episode {episode} event bindings are incomplete")
        for span_id, binding in entry["event_bindings"].items():
            if not isinstance(binding, Mapping) or not isinstance(binding.get("event_id"), str) or not binding["event_id"]:
                raise InputValidationError(f"episode {episode} {span_id} event binding is malformed")
            event_interval = binding.get("event_interval")
            if not isinstance(event_interval, Mapping):
                raise InputValidationError(f"episode {episode} {span_id} event interval is missing")
            expected_interval = by_span_id[span_id]["frame_interval"]
            if [event_interval.get("start_frame"), event_interval.get("end_frame")] != expected_interval:
                raise InputValidationError(f"episode {episode} {span_id} event interval mismatch")
        for field, expected in (
            ("private_only", True),
            ("action_bc_supervision", False),
            ("outcome_supervision", False),
            ("recovery_supervision", False),
            ("dart_supervision", False),
            ("cross_span_join_forbidden", True),
        ):
            if field == "cross_span_join_forbidden":
                actual = constraints.get(field)
            else:
                actual = entry.get(field)
            if actual is not expected:
                raise InputValidationError(f"episode {episode} {field} guard is not {expected}")
    return [dict(entry) for entry in entries]


def _validate_single_video_locators(
    binding: Mapping[str, Any], entry: Mapping[str, Any], observation_frame: int, selection_id: str
) -> None:
    locators = binding.get("video_locators")
    if not isinstance(locators, list) or len(locators) != len(CAMERAS):
        raise InputValidationError(f"single-GRASP {selection_id} event video locators are incomplete")
    by_camera: dict[str, Mapping[str, Any]] = {}
    for locator in locators:
        if not isinstance(locator, Mapping) or not isinstance(locator.get("camera_key"), str):
            raise InputValidationError(f"single-GRASP {selection_id} event video locator is malformed")
        camera = str(locator["camera_key"])
        if camera in by_camera:
            raise InputValidationError(f"single-GRASP {selection_id} event video locators are duplicated")
        by_camera[camera] = locator
    if set(by_camera) != set(CAMERAS):
        raise InputValidationError(f"single-GRASP {selection_id} event video cameras do not match the source clock")
    clock = entry.get("camera_clock")
    if not isinstance(clock, Mapping):
        raise InputValidationError(f"single-GRASP {selection_id} camera clock is missing")
    view_by_camera = {
        "observation.rgb.zed_link_camera_0": "head",
        "observation.rgb.left_realsense_link_camera_0": "left_wrist",
        "observation.rgb.right_realsense_link_camera_0": "right_wrist",
    }
    for camera in CAMERAS:
        values = clock.get(camera)
        locator = by_camera[camera]
        if not isinstance(values, Mapping):
            raise InputValidationError(f"single-GRASP {selection_id} camera clock is malformed")
        chunk = _int(values.get("chunk_index"), f"{selection_id}.{camera}.chunk_index")
        file_index = _int(values.get("file_index"), f"{selection_id}.{camera}.file_index")
        start = _finite(values.get("from_timestamp_s"), f"{selection_id}.{camera}.from_timestamp_s")
        expected_path = f"videos/{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        expected_timestamp = start + observation_frame / 30.0
        if (
            locator.get("view") != view_by_camera[camera]
            or locator.get("relative_path") != expected_path
            or locator.get("expected_fps") != 30
            or locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED"
            or not math.isclose(_finite(locator.get("episode_start_timestamp_s"), f"{selection_id}.{camera}.episode_start_timestamp_s"), start, abs_tol=1e-9)
            or not math.isclose(_finite(locator.get("requested_timestamp_s"), f"{selection_id}.{camera}.requested_timestamp_s"), expected_timestamp, abs_tol=1e-9)
        ):
            raise InputValidationError(f"single-GRASP {selection_id} event video clock does not bind source camera clock")


def _validate_single_event_binding(
    manifest: Mapping[str, Any],
    constraints: Mapping[str, Any],
    entry: Mapping[str, Any],
    span: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    selection_id = str(entry["selection_id"])
    interval = span["frame_interval"]
    start = int(interval[0])
    if not isinstance(binding.get("event_id"), str) or not binding["event_id"]:
        raise InputValidationError(f"single-GRASP {selection_id} event binding ID is missing")
    event_interval = binding.get("event_interval")
    if not isinstance(event_interval, Mapping):
        raise InputValidationError(f"single-GRASP {selection_id} event interval is missing")
    if [event_interval.get("start_frame"), event_interval.get("end_frame")] != interval:
        raise InputValidationError(f"single-GRASP {selection_id} event interval mismatch")
    if binding.get("schema_version") != EVENT_SCHEMA:
        raise InputValidationError(f"single-GRASP {selection_id} event schema is invalid")
    if binding.get("event_kind") != "ANNOTATED_SKILL_SEGMENT":
        raise InputValidationError(f"single-GRASP {selection_id} event kind is invalid")
    if binding.get("usage_role") != entry.get("usage_role") or binding.get("usage_role") != constraints.get("usage_role"):
        raise InputValidationError(f"single-GRASP {selection_id} event role does not bind entry role")

    observation = binding.get("observation")
    if not isinstance(observation, Mapping):
        raise InputValidationError(f"single-GRASP {selection_id} event observation is missing")
    observation_frame = _int(observation.get("frame"), f"single-GRASP {selection_id}.observation.frame")
    observation_timestamp = _finite(observation.get("timestamp_s"), f"single-GRASP {selection_id}.observation.timestamp_s")
    if observation_frame != start or not math.isclose(observation_timestamp, start / 30.0, abs_tol=1e-9):
        raise InputValidationError(f"single-GRASP {selection_id} event observation does not bind GRASP start clock")

    action = binding.get("action_contract")
    if not isinstance(action, Mapping):
        raise InputValidationError(f"single-GRASP {selection_id} event action contract is missing")
    if (
        action.get("start_frame") != observation_frame
        or action.get("actual_executed_length") is not None
        or action.get("raw_action_dim") != ACTION_DIM
        or action.get("model_action_dim") != MODEL_ACTION_DIM
        or action.get("model_padding_indices") != list(MODEL_PADDING_INDICES)
    ):
        raise InputValidationError(f"single-GRASP {selection_id} event action contract does not bind source clock/schema")

    skill = span.get("skill")
    skill_bundle = binding.get("skill_bundle")
    if not isinstance(skill, Mapping) or not isinstance(skill_bundle, list) or not skill_bundle or not all(
        isinstance(member, Mapping) for member in skill_bundle
    ):
        raise InputValidationError(f"single-GRASP {selection_id} event skill bundle is missing")
    if not any(_canonical(member) == _canonical(skill) for member in skill_bundle):
        raise InputValidationError(f"single-GRASP {selection_id} event skill bundle does not contain selected GRASP")
    if binding.get("parallel_bundle") is not (len(skill_bundle) > 1):
        raise InputValidationError(f"single-GRASP {selection_id} event bundle parallel flag is invalid")
    bundle_id = binding.get("bundle_id")
    expected_bundle_id = _EVENT_PROTOCOL.canonical_sha256(
        {"semantic": span.get("semantic"), "text": span.get("text"), "skills": skill_bundle}
    )
    if bundle_id != expected_bundle_id:
        raise InputValidationError(
            f"single-GRASP {selection_id} event bundle ID does not match source skill bundle"
        )

    # event_id is content-addressed by the pinned event protocol.  This binds
    # the retained bundle/clock fields together; full bundle provenance still
    # relies on the externally pinned event-index SHA used by the builder.
    source = {
        "source_release_manifest_sha256": manifest["release_manifest_sha256"],
        "source_group_id": entry["source_group_id"],
        "raw_episode_id": entry["raw_episode_id"],
        "episode_index": entry["episode_index"],
    }
    expected_event_id = _EVENT_PROTOCOL.event_id(
        {
            "schema_version": EVENT_SCHEMA,
            "source": source,
            "event_kind": binding["event_kind"],
            "event_interval": event_interval,
            "observation": observation,
            "bundle_id": bundle_id,
        }
    )
    if binding["event_id"] != expected_event_id:
        raise InputValidationError(f"single-GRASP {selection_id} event ID does not match protocol identity")
    _validate_single_video_locators(binding, entry, observation_frame, selection_id)


def validate_single_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fail-closed validation for the one-span-per-episode input mode."""

    if manifest.get("schema_version") != SINGLE_INPUT_SCHEMA:
        raise InputValidationError("unexpected single-GRASP manifest schema")
    if manifest.get("status") != "AUTHENTICATED_PRIVATE_SINGLE_GRASP_ACTION_SCAN_INPUT":
        raise InputValidationError("single-GRASP manifest status is not authenticated/private")
    if manifest.get("release_manifest_sha256") != RELEASE_SHA:
        raise InputValidationError("single-GRASP manifest release SHA is not authenticated")
    constraints = manifest.get("constraints")
    entries = manifest.get("episodes")
    if not isinstance(constraints, Mapping) or constraints.get("read_only") is not True:
        raise InputValidationError("single-GRASP manifest is not explicitly read-only")
    for field, expected in (
        ("split", "train"),
        ("usage_role", "student_candidate"),
        ("max_episode_count", MAX_SINGLE_EPISODES),
        ("single_grasp_span_per_episode", True),
        ("grasp_spans_only", True),
        ("raw_gripper_values_unnamed", True),
        ("cross_span_join_forbidden", True),
        ("private_only", True),
        ("training_eligible", False),
        ("action_supervision", False),
        ("action_bc_supervision", False),
        ("outcome_supervision", False),
        ("recovery_supervision", False),
        ("dart_supervision", False),
        ("attempt_status", "NOT_APPLICABLE"),
        ("outcome_status", "NOT_APPLICABLE"),
        ("recovery_status", "NOT_APPLICABLE"),
        ("release_status", "NOT_RELEASED"),
    ):
        if constraints.get(field) != expected:
            raise InputValidationError(f"single-GRASP manifest {field} guard is not {expected!r}")
    snapshot_root = manifest.get("official_snapshot_root")
    if not isinstance(snapshot_root, str) or not Path(snapshot_root).is_absolute():
        raise InputValidationError("single-GRASP manifest official_snapshot_root must be absolute")
    for source_name in (
        "official_info",
        "frozen_source",
        "source_groups_source",
        "single_selection_source",
        "event_index_source",
    ):
        source = manifest.get(source_name)
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str) or not isinstance(source.get("sha256"), str):
            raise InputValidationError(f"single-GRASP manifest {source_name} pin is missing")
    if not isinstance(entries, list) or not entries or len(entries) > MAX_SINGLE_EPISODES:
        raise InputValidationError(f"single-GRASP manifest episode count is outside 1..{MAX_SINGLE_EPISODES}")
    if constraints.get("selected_episode_count") != len(entries) or constraints.get("selected_span_count") != len(entries):
        raise InputValidationError("single-GRASP manifest selected counts are inconsistent")
    if manifest["single_selection_source"].get("schema") != SINGLE_SELECTION_SCHEMA:
        raise InputValidationError("single selection source schema pin is missing")
    if manifest["event_index_source"].get("schema") != EXPECTED_EVENT_INDEX_SCHEMA:
        raise InputValidationError("event index source schema pin is missing")

    seen_selection: set[str] = set()
    seen_episode: set[int] = set()
    seen_group: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise InputValidationError("single-GRASP manifest episode is not an object")
        selection_id = entry.get("selection_id")
        if not isinstance(selection_id, str) or not selection_id:
            raise InputValidationError("single-GRASP entry selection_id is missing")
        if selection_id in seen_selection:
            raise InputValidationError(f"duplicate single selection_id: {selection_id}")
        seen_selection.add(selection_id)
        if entry.get("candidate_id") != selection_id or entry.get("candidate_kind") != "single_grasp_span":
            raise InputValidationError(f"single-GRASP entry {selection_id} candidate identity is invalid")
        episode = _int(entry.get("episode_index"), f"single-GRASP {selection_id}.episode_index")
        if episode in seen_episode:
            raise InputValidationError(f"duplicate single-GRASP episode {episode}")
        seen_episode.add(episode)
        group = entry.get("source_group_id")
        if not isinstance(group, str) or len(group) != 64 or group in seen_group:
            raise InputValidationError(f"duplicate or malformed single-GRASP source group for {selection_id}")
        seen_group.add(group)
        for field, expected in (
            ("immutable_split", "train"),
            ("usage_role", "student_candidate"),
            ("private_only", True),
            ("training_eligible", False),
            ("action_supervision", False),
            ("action_bc_supervision", False),
            ("outcome_supervision", False),
            ("recovery_supervision", False),
            ("dart_supervision", False),
            ("attempt_status", "NOT_APPLICABLE"),
            ("outcome_status", "NOT_APPLICABLE"),
            ("recovery_status", "NOT_APPLICABLE"),
            ("release_status", "NOT_RELEASED"),
            ("parent_supervised", False),
        ):
            if entry.get(field) != expected:
                raise InputValidationError(f"single-GRASP {selection_id}.{field} is not {expected!r}")
        length = _int(entry.get("length"), f"single-GRASP {selection_id}.length")
        from_index = _int(entry.get("dataset_from_index"), f"single-GRASP {selection_id}.dataset_from_index")
        to_index = _int(entry.get("dataset_to_index"), f"single-GRASP {selection_id}.dataset_to_index")
        if not (0 <= from_index < to_index and to_index - from_index == length):
            raise InputValidationError(f"single-GRASP {selection_id} global range is invalid")
        _safe_relative(entry.get("parquet_relative_path"), f"single-GRASP {selection_id}.parquet_relative_path")
        _safe_relative(entry.get("annotation_relative_path"), f"single-GRASP {selection_id}.annotation_relative_path")
        spans = entry.get("grasp_spans")
        if not isinstance(spans, list) or len(spans) != 1:
            raise InputValidationError(f"single-GRASP {selection_id} must have exactly one GRASP span")
        span = spans[0]
        if not isinstance(span, Mapping) or span.get("span_id") != selection_id:
            raise InputValidationError(f"single-GRASP {selection_id} span identity is invalid")
        interval = span.get("frame_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise InputValidationError(f"single-GRASP {selection_id} interval is malformed")
        start, end = _int(interval[0], f"single-GRASP {selection_id}.start"), _int(interval[1], f"single-GRASP {selection_id}.end")
        if not (0 <= start < end <= length):
            raise InputValidationError(f"single-GRASP {selection_id} interval is outside episode")
        if span.get("segment_index") is None:
            raise InputValidationError(f"single-GRASP {selection_id} segment index is missing")
        _int(span.get("segment_index"), f"single-GRASP {selection_id}.segment_index")
        if span.get("parent") != entry.get("parent_goal") or span.get("parent_supervised") is not False:
            raise InputValidationError(f"single-GRASP {selection_id} parent metadata is invalid")
        skill = span.get("skill")
        if not isinstance(skill, Mapping) or skill.get("verb") != "GRASP":
            raise InputValidationError(f"single-GRASP {selection_id} skill is not GRASP")
        if skill.get("skill_start") != start or skill.get("skill_end") != end:
            raise InputValidationError(f"single-GRASP {selection_id} skill interval mismatch")
        if skill.get("binding_confidence") != "BOUND" or skill.get("arm") != "UNSPECIFIED":
            raise InputValidationError(f"single-GRASP {selection_id} skill binding/arm is not preserved")
        for field in ("skill_id", "skill_idx", "target", "source", "raw_relation"):
            if field not in skill:
                raise InputValidationError(f"single-GRASP {selection_id} skill {field} is missing")
        event_bindings = entry.get("event_bindings")
        if not isinstance(event_bindings, Mapping) or set(event_bindings) != {selection_id}:
            raise InputValidationError(f"single-GRASP {selection_id} event bindings are incomplete")
        binding = event_bindings[selection_id]
        if not isinstance(binding, Mapping):
            raise InputValidationError(f"single-GRASP {selection_id} event binding is malformed")
        clock = entry.get("camera_clock")
        if not isinstance(clock, Mapping) or set(clock) != set(CAMERAS):
            raise InputValidationError(f"single-GRASP {selection_id} camera clock is incomplete")
        for camera in CAMERAS:
            values = clock[camera]
            if not isinstance(values, Mapping):
                raise InputValidationError(f"single-GRASP {selection_id} camera clock is malformed")
            _int(values.get("chunk_index"), f"{selection_id}.{camera}.chunk_index")
            _int(values.get("file_index"), f"{selection_id}.{camera}.file_index")
            from_timestamp = _finite(values.get("from_timestamp_s"), f"{selection_id}.{camera}.from_timestamp_s")
            to_timestamp = _finite(values.get("to_timestamp_s"), f"{selection_id}.{camera}.to_timestamp_s")
            if to_timestamp <= from_timestamp:
                raise InputValidationError(f"single-GRASP {selection_id} camera timestamp interval is invalid")
            if not math.isclose(to_timestamp - from_timestamp, length / 30.0, abs_tol=1e-9):
                raise InputValidationError(f"single-GRASP {selection_id} camera clock duration does not bind episode length")
        _validate_single_event_binding(manifest, constraints, entry, span, binding)
    return [dict(entry) for entry in entries]


def scan_grasp_span_rows(rows: Sequence[Mapping[str, Any]], entry: Mapping[str, Any]) -> dict[str, Any]:
    """Scan each manifest span independently while retaining episode frame IDs."""

    normalized = validate_episode_rows(rows, entry)
    result_spans: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    total_runs = 0
    total_reversals = 0
    for span in entry["grasp_spans"]:
        span_id = str(span["span_id"])
        start, end = map(int, span["frame_interval"])
        span_rows = [row for row in normalized if start <= int(row["frame_index"]) < end]
        if len(span_rows) != end - start:
            raise InputValidationError(f"episode {entry['episode_index']} {span_id} span has missing rows")
        channel_results: dict[str, Any] = {}
        span_candidates: list[dict[str, Any]] = []
        for channel, action_index in GRIPPER_INDICES.items():
            runs = build_rle_runs(span_rows, channel, action_index)
            candidates, exclusions, counts = find_reversal_candidates(runs)
            for index, candidate in enumerate(candidates):
                item = _copy_json(candidate)
                item["candidate_id"] = f"{entry['candidate_id']}:{span_id}:{channel}:aba-{index:04d}"
                item["source_identity"] = {
                    "candidate_id": entry["candidate_id"],
                    "event_id": entry["event_bindings"][span_id]["event_id"],
                    "episode_index": int(entry["episode_index"]),
                    "source_group_id": entry["source_group_id"],
                    "frame_interval": [start, end],
                }
                span_candidates.append(item)
                all_candidates.append(item)
            channel_results[channel] = {"runs": runs, "counts": counts, "exclusions": exclusions}
            total_runs += len(runs)
            total_reversals += int(counts["structural_aba_matches"])
        result_spans.append(
            {
                "span_id": span_id,
                "frame_interval": [start, end],
                "event_binding": _copy_json(entry["event_bindings"][span_id]),
                "channels": channel_results,
                "candidate_count": len(span_candidates),
                "internal_reversal_count": sum(item["counts"]["structural_aba_matches"] for item in channel_results.values()),
            }
        )
    return {
        "candidate_id": entry["candidate_id"],
        "source_identity": {key: entry[key] for key in ("episode_index", "raw_episode_id", "task_index", "task_instance_id", "source_group_id", "source_annotation_sha256")},
        "length": int(entry["length"]),
        "rows_scanned": len(normalized),
        "grasp_spans": result_spans,
        "candidates": all_candidates,
        "candidate_count": len(all_candidates),
        "total_rle_runs": total_runs,
        "internal_reversal_count": total_reversals,
        "gripper_action_indices": dict(GRIPPER_INDICES),
        "gripper_action_indices_source": GRIPPER_INDEX_SOURCE,
        "semantic_guard": "Raw A-B-A command structure only; no polarity, grasp, attempt, outcome, failure, recovery, or training meaning.",
        "role": "diagnostic_candidate",
        "private_only": True,
        "training_eligible": False,
        "action_bc_supervision": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "dart_supervision": False,
        "attempt_status": "NOT_APPLICABLE",
    }


def run_scan(manifest_path: Path, output_path: Path, *, source_root: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest_sha = _required_sha(manifest_path, expected_manifest_sha256, "input manifest")
    manifest = _load_object(manifest_path, "input manifest")
    if manifest.get("schema_version") == SINGLE_INPUT_SCHEMA:
        entries = validate_single_manifest(manifest)
        input_mode = "single_grasp_span"
    else:
        entries = validate_manifest(manifest)
        input_mode = "paired_grasp_spans"
    info_path = Path(str(manifest["official_info"]["path"]))
    if _sha(info_path) != manifest["official_info"]["sha256"]:
        raise InputValidationError("official info SHA changed after manifest creation")
    info = _load_object(info_path, "official info")
    validate_info(info)
    root = source_root.resolve()
    declared_root = manifest.get("official_snapshot_root")
    if declared_root is not None and root != Path(str(declared_root)).resolve():
        raise InputValidationError("source root differs from manifest authenticated root")
    episode_results: list[dict[str, Any]] = []

    def consume(entry: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
        episode_results.append(scan_grasp_span_rows(rows, entry))

    parquet_audit = _read_parquet_rows(entries, root, consume)
    result = {
        "schema_version": OUTPUT_SCHEMA,
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
        "inputs": {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha, "official_info_sha256": manifest["official_info"]["sha256"], "episode_count": len(entries), "input_mode": input_mode},
        "scan": {"episodes_scanned": len(episode_results), "rows_scanned": sum(item["rows_scanned"] for item in episode_results), "candidate_count": sum(item["candidate_count"] for item in episode_results), **parquet_audit},
        "episodes": sorted(episode_results, key=lambda item: item["candidate_id"]),
    }
    _atomic_json(output_path, result)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build-manifest")
    builder.add_argument("--triage", type=Path, required=True)
    builder.add_argument("--frozen", type=Path, required=True)
    builder.add_argument("--event-index", type=Path, required=True)
    builder.add_argument("--info", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--source-root", required=True)
    builder.add_argument("--candidate-id", action="append", required=True)
    builder.add_argument("--expected-triage-sha256", required=True)
    builder.add_argument("--expected-frozen-sha256", required=True)
    builder.add_argument("--expected-event-index-sha256", required=True)
    builder.add_argument("--expected-info-sha256", required=True)
    single_builder = sub.add_parser("build-single-manifest")
    single_builder.add_argument("--selection", type=Path, required=True)
    single_builder.add_argument("--frozen", type=Path, required=True)
    single_builder.add_argument("--source-groups", type=Path, required=True)
    single_builder.add_argument("--event-index", type=Path, required=True)
    single_builder.add_argument("--info", type=Path, required=True)
    single_builder.add_argument("--output", type=Path, required=True)
    single_builder.add_argument("--source-root", required=True)
    single_builder.add_argument("--expected-selection-sha256", required=True)
    single_builder.add_argument("--expected-frozen-sha256", required=True)
    single_builder.add_argument("--expected-source-groups-sha256", required=True)
    single_builder.add_argument("--expected-event-index-sha256", required=True)
    single_builder.add_argument("--expected-info-sha256", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--manifest", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    scan.add_argument("--source-root", type=Path, required=True)
    scan.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "build-manifest":
            build_manifest(
                triage_path=args.triage,
                frozen_path=args.frozen,
                event_index_path=args.event_index,
                info_path=args.info,
                output_path=args.output,
                expected_triage_sha256=args.expected_triage_sha256,
                expected_frozen_sha256=args.expected_frozen_sha256,
                expected_event_index_sha256=args.expected_event_index_sha256,
                expected_info_sha256=args.expected_info_sha256,
                candidate_ids=args.candidate_id,
                source_root=args.source_root,
            )
        elif args.command == "build-single-manifest":
            build_single_manifest(
                selection_path=args.selection,
                frozen_path=args.frozen,
                source_groups_path=args.source_groups,
                event_index_path=args.event_index,
                info_path=args.info,
                output_path=args.output,
                expected_selection_sha256=args.expected_selection_sha256,
                expected_frozen_sha256=args.expected_frozen_sha256,
                expected_source_groups_sha256=args.expected_source_groups_sha256,
                expected_event_index_sha256=args.expected_event_index_sha256,
                expected_info_sha256=args.expected_info_sha256,
                source_root=args.source_root,
            )
        else:
            run_scan(args.manifest, args.output, source_root=args.source_root, expected_manifest_sha256=args.expected_manifest_sha256)
    except (InputValidationError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
