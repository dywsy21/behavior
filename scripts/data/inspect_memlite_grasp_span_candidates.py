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


_LEGACY = _load_legacy_helpers()
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
OUTPUT_SCHEMA = "p107-grasp-span-private-output-v1"
TRIAGE_SCHEMA = "p107-natural-retry-metadata-triage-v2"
EVENT_SCHEMA = "memlite-event-recovery-v1"
RELEASE_SHA = "90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23"
EXPECTED_TRIAGE_COUNT = 32
EXPECTED_FROZEN_COUNT = 20000
EXPECTED_EVENT_INDEX_SCHEMA = "memlite-event-index-v1"
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


def _private_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "event_interval": _copy_json(event.get("event_interval")),
        "event_kind": event.get("event_kind"),
        "schema_version": event.get("schema_version"),
        "bundle_id": event.get("bundle_id"),
        "observation": _copy_json(event.get("observation")),
        "action_contract": _copy_json(event.get("action")),
        "usage_role": event.get("usage_role"),
    }


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
    entries = validate_manifest(manifest)
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
        "inputs": {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha, "official_info_sha256": manifest["official_info"]["sha256"], "episode_count": len(entries)},
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
        else:
            run_scan(args.manifest, args.output, source_root=args.source_root, expected_manifest_sha256=args.expected_manifest_sha256)
    except (InputValidationError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
