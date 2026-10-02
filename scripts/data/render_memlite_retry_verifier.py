"""Decode a small, private RGB verifier view for the P107 retry candidates.

This is deliberately narrower than the actor packet renderer.  It authenticates
the immutable private-selection document, the v1 structural candidate result,
and a sealed event index, then decodes only the selection's sparse local frame
plan from the three native camera videos.  The output is a private visual
verification artifact; it is never an actor packet, a label, or training data.

The actual PTS decoder is the implementation in
``render_memlite_event_packets.py``.  This module only supplies the bounded
selection, identity, path, output, and private-review bookkeeping around it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCRIPT_SCHEMA = "p107-private-rgb-verifier-v1"
SELECTION_SCHEMA = "p107-natural-retry-pilot-private-rgb-selection-v1"
SOURCE_RESULT_SCHEMA = "p107-natural-retry-pilot-output-v1"
EVENT_INDEX_SCHEMA = "memlite-event-index-v1"
EVENT_SCHEMA = "memlite-event-recovery-v1"
VIEWS = ("head", "left_wrist", "right_wrist")
FPS = 30.0
STRIDE_FRAMES = 30
EXPECTED_SELECTIONS = 8
EXPECTED_WINDOWS = 9
EXPECTED_LOCAL_FRAMES = 161
EXPECTED_PNGS = EXPECTED_LOCAL_FRAMES * len(VIEWS)
EXPECTED_OVERVIEW_PAGES = EXPECTED_WINDOWS
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class InputValidationError(ValueError):
    """Raised when a pinned source or bounded selection is not admissible."""


_RENDERER: Any | None = None


def _renderer() -> Any:
    """Load the existing packet renderer without importing an installed package named ``scripts``."""

    global _RENDERER
    if _RENDERER is not None:
        return _RENDERER
    path = Path(__file__).with_name("render_memlite_event_packets.py")
    if path.is_symlink() or not path.is_file():
        raise InputValidationError(f"native packet renderer is not a regular file: {path}")
    name = "_p107_native_packet_renderer_for_private_verifier"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise InputValidationError("cannot load the existing native packet renderer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _RENDERER = module
    return module


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
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputValidationError(f"invalid JSON in {name}") from exc


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise InputValidationError(f"{label} must be a regular file: {path}")
    return path


def _regular_dir(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise InputValidationError(f"{label} must be a regular directory: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path, "hash input").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Any:
    path = _regular_file(path, label)
    return _strict_json(path.read_bytes(), name=str(path))


def _require_sha(actual: str, expected: Any, label: str) -> None:
    if not _is_sha256(expected) or actual != expected:
        raise InputValidationError(f"{label} SHA-256 does not match its external pin")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InputValidationError(f"{label} must be finite")
    return float(value)


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise InputValidationError(f"{label} must be an integer")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise InputValidationError(f"{label} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise InputValidationError(f"{label} is not a safe relative path")
    return value


def _resolve_video(raw_root: Path, relative_path: str) -> Path:
    """Resolve a video without accepting symlinked files or path escapes."""

    raw_root = _regular_dir(raw_root, "raw root")
    relative_path = _safe_relative_path(relative_path, "video locator relative_path")
    candidate = raw_root / Path(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink():
        raise InputValidationError(f"video locator resolves through a symlink: {candidate}")
    try:
        path = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise InputValidationError(f"video locator is missing: {candidate}") from exc
    root = raw_root.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise InputValidationError(f"video locator escapes raw root or is not a regular file: {candidate}")
    return path


def _forbid_output(output: Path, sources: Iterable[Path]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"private verifier output already exists: {output}")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise InputValidationError(f"output parent must be an existing regular directory: {parent}")
    resolved = output.resolve(strict=False)
    for source in sources:
        source = Path(source).resolve(strict=True)
        if resolved == source or source in resolved.parents:
            raise InputValidationError("private verifier output must be outside immutable input directories")


def _source_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    source = entry.get("source_identity")
    if not isinstance(source, Mapping):
        raise InputValidationError("source result episode has no source_identity")
    fields = ("event_id", "episode_index", "raw_episode_id", "task_index", "task_instance_id", "source_group_id")
    if set(source) != set(fields):
        raise InputValidationError("source result source_identity fields drifted")
    if not isinstance(source["event_id"], str) or not isinstance(source["source_group_id"], str):
        raise InputValidationError("source result string identity field is malformed")
    for field in fields[1:5]:
        _exact_int(source[field], f"source_identity.{field}")
    return dict(source)


def _load_source_result(path: Path, expected_sha256: str, selection_doc: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    actual_sha = _sha256(path)
    _require_sha(actual_sha, expected_sha256, "source result")
    result = _load_json(path, "source result")
    if not isinstance(result, dict) or result.get("schema_version") != SOURCE_RESULT_SCHEMA:
        raise InputValidationError("source result schema is not the pinned v1 structural result")
    if (result.get("status") != "PASS_STRUCTURAL_SCAN_NONTRAINABLE" or result.get("role") != "diagnostic_candidate" or
            result.get("training_eligible") is not False or result.get("attempt_status") != "NOT_APPLICABLE" or
            result.get("release_status") != "NOT_RELEASED"):
        raise InputValidationError("source result is not the expected non-training diagnostic result")
    for field in ("action_bc_supervision", "outcome_supervision", "recovery_supervision", "dart_supervision"):
        if result.get(field) is not False:
            raise InputValidationError(f"source result {field} guard drifted")
    pinned = selection_doc.get("source_result")
    if not isinstance(pinned, Mapping) or pinned.get("schema_version") != SOURCE_RESULT_SCHEMA:
        raise InputValidationError("selection source_result pin is malformed")
    if pinned.get("sha256") != actual_sha or pinned.get("candidate_count") != 73:
        raise InputValidationError("selection does not pin the expected 73-candidate source result")
    if not isinstance(result.get("episodes"), list):
        raise InputValidationError("source result episodes are not a list")
    candidates: dict[str, dict[str, Any]] = {}
    episode_count = 0
    candidate_count = 0
    for episode in result["episodes"]:
        if not isinstance(episode, dict):
            raise InputValidationError("source result episode is not an object")
        source = _source_identity(episode)
        annotation_sha = episode.get("annotation_sha256")
        if not _is_sha256(annotation_sha):
            raise InputValidationError("source result annotation SHA is malformed")
        length = _exact_int(episode.get("length"), "source result episode length")
        if length <= 0 or not isinstance(episode.get("candidates"), list):
            raise InputValidationError("source result episode length/candidates are malformed")
        episode_count += 1
        for candidate in episode["candidates"]:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
                raise InputValidationError("source result candidate is malformed")
            candidate_id = candidate["candidate_id"]
            if candidate_id in candidates:
                raise InputValidationError(f"duplicate source result candidate_id: {candidate_id}")
            if candidate.get("semantic_interpretation") != "NOT_ASSIGNED" or candidate.get("command_structure") != "A_B_A_UNNAMED":
                raise InputValidationError("source candidate carries a forbidden semantic interpretation")
            runs = candidate.get("run_intervals")
            if not isinstance(runs, list) or len(runs) != 3:
                raise InputValidationError(f"candidate {candidate_id} does not contain three runs")
            for run in runs:
                if not isinstance(run, dict):
                    raise InputValidationError(f"candidate {candidate_id} run is malformed")
                start = _exact_int(run.get("start_frame"), f"candidate {candidate_id} run start")
                end = _exact_int(run.get("end_frame"), f"candidate {candidate_id} run end")
                if start < 0 or end <= start or end > length:
                    raise InputValidationError(f"candidate {candidate_id} run lies outside its episode")
            if candidate.get("frame_span") != [runs[0]["start_frame"], runs[2]["end_frame"]]:
                raise InputValidationError(f"candidate {candidate_id} outer span drifted")
            candidates[candidate_id] = {
                "candidate": candidate,
                "source": source,
                "annotation_sha256": annotation_sha,
                "length": length,
            }
            candidate_count += 1
    if episode_count != EXPECTED_SELECTIONS or candidate_count != 73:
        raise InputValidationError("source result does not contain the expected 8 episodes/73 candidates")
    release_sha = result.get("inputs", {}).get("release_manifest_sha256")
    if not _is_sha256(release_sha):
        raise InputValidationError("source result has no pinned release manifest SHA")
    return {"result": result, "release_manifest_sha256": release_sha}, candidates


def _validate_guards(selection: Mapping[str, Any]) -> None:
    if selection.get("document_kind") != SELECTION_SCHEMA:
        raise InputValidationError("private selection document kind is not the pinned schema")
    guards = selection.get("global_guards")
    if not isinstance(guards, Mapping):
        raise InputValidationError("private selection global_guards are missing")
    exact = {
        "future_image_use": "private_verifier_only_after_separate_decode_authorization",
        "training_eligible": False,
        "action_bc_supervision": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "dart_supervision": False,
        "attempt_status": "NOT_APPLICABLE",
        "release_status": "NOT_RELEASED",
        "do_not_assign_command_polarity": True,
        "do_not_treat_source_skill_instruction_as_observed_success_or_failure": True,
        "no_rgb_was_viewed_when_this_plan_was_written": True,
    }
    for key, value in exact.items():
        if guards.get(key) != value:
            raise InputValidationError(f"private selection guard drifted: {key}")


def _validate_event_index(
    index_dir: Path,
    expected_manifest_sha256: str,
    wanted_event_ids: set[str],
    expected_release_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    index_dir = _regular_dir(index_dir, "sealed event index")
    manifest_path = _regular_file(index_dir / "manifest.json", "event index manifest")
    manifest_sha = _sha256(manifest_path)
    _require_sha(manifest_sha, expected_manifest_sha256, "event index manifest")
    manifest = _load_json(manifest_path, "event index manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != EVENT_INDEX_SCHEMA:
        raise InputValidationError("event index schema is not the pinned memlite index")
    if manifest.get("source_release_manifest_sha256") != expected_release_sha256:
        raise InputValidationError("event index source release does not match the structural source result")
    receipt = manifest.get("files", {}).get("event_candidates.jsonl")
    if not isinstance(receipt, Mapping) or set(receipt) != {"sha256", "rows", "bytes"} or not _is_sha256(receipt.get("sha256")):
        raise InputValidationError("event index event-candidate receipt is malformed")
    if type(receipt.get("rows")) is not int or type(receipt.get("bytes")) is not int or receipt["rows"] < 1 or receipt["bytes"] < 1:
        raise InputValidationError("event index event-candidate receipt dimensions are malformed")
    event_path = _regular_file(index_dir / "event_candidates.jsonl", "event-candidate file")
    if event_path.stat().st_size != receipt["bytes"]:
        raise InputValidationError("event-candidate file byte count does not match its receipt")
    selected: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    rows = 0
    with event_path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            value = _strict_json(line, name=f"{event_path}:{number}")
            if not isinstance(value, dict):
                raise InputValidationError(f"event-candidate row {number} is not an object")
            event_id = value.get("event_id")
            if event_id in wanted_event_ids:
                if event_id in selected:
                    raise InputValidationError(f"duplicate selected event_id in event index: {event_id}")
                selected[event_id] = value
    if rows != receipt["rows"] or digest.hexdigest() != receipt["sha256"] or set(selected) != wanted_event_ids:
        raise InputValidationError("event-candidate bytes/rows/selected identities do not match their sealed receipt")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_schema_version": manifest["schema_version"],
        "source_release_manifest_sha256": manifest["source_release_manifest_sha256"],
        "event_path": str(event_path),
        "event_file_sha256": digest.hexdigest(),
        "event_file_rows": rows,
    }, selected


def _identity_fields() -> tuple[str, ...]:
    return ("event_id", "episode_index", "raw_episode_id", "task_index", "task_instance_id", "source_group_id")


def _validate_video_locators(
    event: Mapping[str, Any], source: Mapping[str, Any], length: int,
    expected_release_sha256: str, expected_annotation_sha256: str,
) -> dict[str, dict[str, Any]]:
    if event.get("schema_version") != EVENT_SCHEMA or event.get("record_kind") != "event_candidate":
        raise InputValidationError("selected event has an unexpected schema")
    if event.get("usage_role") != "annotation_calibration":
        raise InputValidationError("selected event is not TRAIN annotation_calibration")
    event_source = event.get("source")
    if not isinstance(event_source, Mapping):
        raise InputValidationError("selected event has no source identity")
    for field in _identity_fields():
        if event_source.get(field) != source.get(field):
            raise InputValidationError(f"selected event source identity mismatch: {field}")
    if event_source.get("original_split") != "train" or event_source.get("episode_length") != length:
        raise InputValidationError("selected event split/episode length mismatch")
    if event_source.get("source_release_manifest_sha256") != expected_release_sha256:
        raise InputValidationError("selected event release source mismatch")
    if event_source.get("source_annotation_sha256") != expected_annotation_sha256:
        raise InputValidationError("selected event annotation source mismatch")
    locators = event.get("video_locators")
    if not isinstance(locators, list) or len(locators) != len(VIEWS):
        raise InputValidationError("selected event must have exactly three camera locators")
    by_view: dict[str, dict[str, Any]] = {}
    for locator in locators:
        if not isinstance(locator, Mapping) or locator.get("view") in by_view:
            raise InputValidationError("selected event camera locators are duplicate or malformed")
        view = locator.get("view")
        if view not in VIEWS:
            raise InputValidationError(f"selected event has unexpected camera view: {view}")
        if locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED" or locator.get("expected_fps") != 30:
            raise InputValidationError(f"selected event locator contract drifted for {view}")
        _safe_relative_path(locator.get("relative_path"), f"{view} video path")
        _finite_number(locator.get("episode_start_timestamp_s"), f"{view} episode start timestamp")
        by_view[view] = dict(locator)
    if set(by_view) != set(VIEWS):
        raise InputValidationError("selected event does not cover all three camera views")
    return by_view


def _intervals(selection: Mapping[str, Any], length: int) -> tuple[list[list[int]], list[int], list[int]]:
    windows = selection.get("render_windows_local")
    global_windows = selection.get("render_windows_dataset_global")
    if not isinstance(windows, list) or not isinstance(global_windows, list) or len(windows) != len(global_windows):
        raise InputValidationError(f"{selection.get('selection_id')} render windows are malformed")
    if not windows or len(windows) > EXPECTED_WINDOWS:
        raise InputValidationError("selection window count is outside the bounded plan")
    normalized: list[list[int]] = []
    offsets: list[int] = []
    previous_end = -1
    for local, global_window in zip(windows, global_windows):
        if (not isinstance(local, list) or len(local) != 2 or not all(type(value) is int for value in local) or
                not isinstance(global_window, list) or len(global_window) != 2 or not all(type(value) is int for value in global_window)):
            raise InputValidationError("render window endpoint type drifted")
        start, end = local
        global_start, global_end = global_window
        if start < 0 or start >= end or end > length or previous_end >= start:
            raise InputValidationError("render windows overlap, are empty, or leave episode bounds")
        if global_end - global_start != end - start:
            raise InputValidationError("local/global render window length mismatch")
        offset = global_start - start
        if global_end - end != offset:
            raise InputValidationError("local/global frame offset drifted inside render window")
        normalized.append([start, end])
        offsets.append(offset)
        previous_end = end
    if len(set(offsets)) != 1:
        raise InputValidationError("one candidate uses more than one local/global frame offset")
    return normalized, [offsets[0]], [previous_end]


def build_sparse_frame_plan(selection: Mapping[str, Any], length: int) -> list[dict[str, int]]:
    """Build exactly the bounded stride/endpoint/flip/core plan from one selection."""

    windows, offsets, _ = _intervals(selection, length)
    frames: dict[tuple[int, int], dict[str, int]] = {}
    for window_index, (start, end) in enumerate(windows):
        offset = offsets[0]
        for local_frame in set(range(start, end, STRIDE_FRAMES)) | {start, end - 1}:
            frames[(window_index, local_frame)] = {
                "window_index": window_index,
                "local_frame": local_frame,
                "dataset_global_frame": local_frame + offset,
            }
    flip_frames = selection.get("flip_frames_local")
    core = selection.get("flip_core_frame_interval")
    if not isinstance(flip_frames, list) or len(flip_frames) != 2 or not all(type(value) is int for value in flip_frames):
        raise InputValidationError("selection flip frames are malformed")
    if not isinstance(core, list) or len(core) != 2 or not all(type(value) is int for value in core):
        raise InputValidationError("selection core interval is malformed")
    explicit_frames = set(flip_frames) | {core[0], core[1] - 1}
    if core[0] < 0 or core[0] >= core[1] or core[1] > length or not explicit_frames:
        raise InputValidationError("selection core interval lies outside the episode")
    for local_frame in explicit_frames:
        matches = [index for index, (start, end) in enumerate(windows) if start <= local_frame < end]
        if len(matches) != 1:
            raise InputValidationError("flip/core boundary is not inside exactly one render window")
        window_index = matches[0]
        offset = offsets[0]
        frames[(window_index, local_frame)] = {
            "window_index": window_index,
            "local_frame": local_frame,
            "dataset_global_frame": local_frame + offset,
        }
    expected_global_flips = selection.get("flip_dataset_global_indices")
    if expected_global_flips != [frame + offsets[0] for frame in flip_frames]:
        raise InputValidationError("selection local/global flip frame mapping drifted")
    ordered = [frames[key] for key in sorted(frames)]
    for item in ordered:
        if not any(start <= item["local_frame"] < end for start, end in windows):
            raise InputValidationError("sparse plan frame escaped its window")
    return ordered


def _validate_selection_and_candidates(
    selection_doc: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]], release_sha: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _validate_guards(selection_doc)
    selections = selection_doc.get("selections")
    if not isinstance(selections, list) or len(selections) != EXPECTED_SELECTIONS:
        raise InputValidationError("private selection must contain exactly eight selections")
    normalized: list[dict[str, Any]] = []
    wanted_event_ids: dict[str, dict[str, Any]] = {}
    for selection in selections:
        if not isinstance(selection, dict) or not isinstance(selection.get("selection_id"), str):
            raise InputValidationError("selection row is malformed")
        selection_id = selection["selection_id"]
        if not SAFE_NAME_RE.fullmatch(selection_id) or selection_id in {
            row["selection"]["selection_id"] for row in normalized
        }:
            raise InputValidationError("selection IDs must be unique safe names")
        source = selection.get("source")
        candidate_id = selection.get("candidate_id")
        if not isinstance(source, Mapping) or not isinstance(candidate_id, str) or candidate_id not in candidates:
            raise InputValidationError(f"selection {selection_id} is not bound to a source candidate")
        source_record = candidates[candidate_id]
        candidate = source_record["candidate"]
        for field in _identity_fields():
            if source.get(field) != source_record["source"].get(field):
                raise InputValidationError(f"selection {selection_id} identity mismatch: {field}")
        if source.get("annotation_sha256") != source_record["annotation_sha256"]:
            raise InputValidationError(f"selection {selection_id} annotation SHA mismatch")
        if selection.get("channel") != candidate.get("channel") or selection.get("run_indices") != candidate.get("run_indices"):
            raise InputValidationError(f"selection {selection_id} channel/run identity mismatch")
        if selection.get("unnamed_command_values") != candidate.get("values"):
            raise InputValidationError(f"selection {selection_id} command values mismatch")
        if selection.get("outer_frame_span") != candidate.get("frame_span"):
            raise InputValidationError(f"selection {selection_id} outer frame span mismatch")
        runs = candidate["run_intervals"]
        expected_core = [runs[0]["end_frame"] - 1, runs[1]["end_frame"] + 1]
        if selection.get("flip_core_frame_interval") != expected_core:
            raise InputValidationError(f"selection {selection_id} core interval mismatch")
        if selection.get("flip_frames_local") != [runs[0]["end_frame"], runs[1]["end_frame"]]:
            raise InputValidationError(f"selection {selection_id} flip frames mismatch")
        if selection.get("middle_run_frames") != runs[1].get("dwell_frames"):
            raise InputValidationError(f"selection {selection_id} middle dwell mismatch")
        _intervals(selection, source_record["length"])
        plan = build_sparse_frame_plan(selection, source_record["length"])
        if not plan:
            raise InputValidationError(f"selection {selection_id} has no sparse frames")
        event_id = source_record["source"]["event_id"]
        existing_event = wanted_event_ids.get(event_id)
        if existing_event is not None:
            # Multiple selected candidates may share one source episode (the
            # paired controls in the frozen plan).  They must share exactly
            # the same authenticated source identity and length, while their
            # candidate/window plans remain separate in ``normalized``.
            if (existing_event["source_record"]["source"] != source_record["source"] or
                    existing_event["source_record"]["length"] != source_record["length"]):
                raise InputValidationError(f"selection {selection_id} disagrees with a sibling source episode")
        else:
            wanted_event_ids[event_id] = {"source_record": source_record}
        normalized.append({"selection": selection, "source_record": source_record, "plan": plan})
    if len({row["selection"]["candidate_id"] for row in normalized}) != EXPECTED_SELECTIONS:
        raise InputValidationError("the eight selections must use distinct candidates")
    if sum(len(row["plan"]) for row in normalized) != EXPECTED_LOCAL_FRAMES or sum(
        len(row["selection"]["render_windows_local"]) for row in normalized
    ) != EXPECTED_WINDOWS:
        raise InputValidationError("sparse plan is not the pinned 161-frame/9-window budget")
    # s08 is intentionally two windows; never silently merge or reorder them.
    s08 = next((row for row in normalized if row["selection"]["selection_id"] == "s08"), None)
    if s08 is None or len(s08["selection"]["render_windows_local"]) != 2:
        raise InputValidationError("s08 must retain two non-contiguous verifier windows")
    if s08["selection"]["render_windows_local"][1][0] <= s08["selection"]["render_windows_local"][0][1]:
        raise InputValidationError("s08 verifier windows must remain separated")
    return normalized, wanted_event_ids


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_overview_pages(stage: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("private verifier overview pages require Pillow") from exc
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["selection_id"], record["window_index"]), []).append(record)
    overview_dir = stage / "private_verifier_overviews"
    overview_dir.mkdir()
    pages: list[dict[str, Any]] = []
    for (selection_id, window_index), group in sorted(grouped.items()):
        by_frame: dict[int, dict[str, dict[str, Any]]] = {}
        for record in group:
            by_frame.setdefault(record["local_frame"], {})[record["camera_view"]] = record
        endpoint_frames = sorted(by_frame)
        if len(endpoint_frames) > 2:
            endpoint_frames = [endpoint_frames[0], endpoint_frames[-1]]
        images: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        try:
            for local_frame in endpoint_frames:
                camera_records = by_frame[local_frame]
                if set(camera_records) != set(VIEWS):
                    raise InputValidationError("overview endpoint does not have all three camera PNGs")
                opened: dict[str, Any] = {}
                for view in VIEWS:
                    record = camera_records[view]
                    opened[view] = Image.open(stage / record["png_relative_path"]).convert("RGB")
                images.append((local_frame, camera_records, opened))
            widths = [images[0][2][view].width for view in VIEWS]
            row_height = max(images[0][2][view].height for view in VIEWS) + 42
            sheet = Image.new("RGB", (sum(widths), row_height * len(images)), "white")
            draw = ImageDraw.Draw(sheet)
            for row_index, (local_frame, camera_records, opened) in enumerate(images):
                y = row_index * row_height
                x = 0
                global_frame = camera_records[VIEWS[0]]["dataset_global_frame"]
                for view in VIEWS:
                    image = opened[view]
                    draw.text((x + 3, y + 2), f"{view} local={local_frame} global={global_frame}", fill="black")
                    sheet.paste(image, (x, y + 22))
                    x += image.width
            relative = Path("private_verifier_overviews") / f"{selection_id}_window-{window_index:02d}.png"
            output_path = stage / relative
            sheet.save(output_path, format="PNG")
            pages.append({
                "kind": "private_verifier_overview_png",
                "selection_id": selection_id,
                "window_index": window_index,
                "endpoint_local_frames": endpoint_frames,
                "relative_path": str(relative),
                "sha256": _sha256(output_path),
                "bytes": output_path.stat().st_size,
                "actor_packet_included": False,
            })
        finally:
            for _local_frame, _camera_records, opened in images:
                for image in opened.values():
                    image.close()
    return pages


def run_verifier(
    selection_path: Path,
    source_result_path: Path,
    event_index_dir: Path,
    raw_root: Path,
    output_path: Path,
    *,
    expected_selection_sha256: str,
    expected_source_result_sha256: str,
    expected_event_index_manifest_sha256: str,
    av_backend: Any | None = None,
) -> dict[str, Any]:
    """Decode the pinned private plan and atomically publish native PNGs/pages."""

    selection_path = _regular_file(selection_path, "private selection")
    source_result_path = _regular_file(source_result_path, "source result")
    event_index_dir = _regular_dir(event_index_dir, "sealed event index")
    raw_root = _regular_dir(raw_root, "raw video root")
    output_path = Path(output_path)
    _forbid_output(output_path, (selection_path.parent, source_result_path.parent, event_index_dir, raw_root))
    selection_sha = _sha256(selection_path)
    _require_sha(selection_sha, expected_selection_sha256, "private selection")
    selection = _load_json(selection_path, "private selection")
    if not isinstance(selection, dict):
        raise InputValidationError("private selection is not an object")
    source_pinned = selection.get("source_result")
    if not isinstance(source_pinned, Mapping) or source_pinned.get("sha256") != expected_source_result_sha256:
        raise InputValidationError("selection source-result SHA is not the requested external pin")
    source_meta, candidate_map = _load_source_result(source_result_path, expected_source_result_sha256, selection)
    normalized, wanted = _validate_selection_and_candidates(selection, candidate_map, source_meta["release_manifest_sha256"])
    event_audit, events = _validate_event_index(
        event_index_dir,
        expected_event_index_manifest_sha256,
        set(wanted),
        source_meta["release_manifest_sha256"],
    )
    event_locators: dict[str, dict[str, dict[str, Any]]] = {}
    for event_id, row in wanted.items():
        event_locators[event_id] = _validate_video_locators(
            events[event_id], row["source_record"]["source"], row["source_record"]["length"],
            source_meta["release_manifest_sha256"], row["source_record"]["annotation_sha256"],
        )
    renderer = _renderer()
    if av_backend is None:
        av_backend, _image, _draw = renderer._require_decode_dependencies(contact_sheets=False)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=str(output_path.parent)))
    records: list[dict[str, Any]] = []
    try:
        assets_dir = staging / "native_rgb"
        assets_dir.mkdir()
        for row in normalized:
            selection_row = row["selection"]
            source_record = row["source_record"]
            event_id = source_record["source"]["event_id"]
            locators = event_locators[event_id]
            selection_id = selection_row["selection_id"]
            if not SAFE_NAME_RE.fullmatch(selection_id):
                raise InputValidationError("selection ID is not safe for output paths")
            length = source_record["length"]
            for item in row["plan"]:
                local_frame = item["local_frame"]
                requested_by_view: dict[str, tuple[Any, dict[str, Any], float]] = {}
                for view in VIEWS:
                    locator = locators[view]
                    video = _resolve_video(raw_root, locator["relative_path"])
                    episode_start = _finite_number(locator["episode_start_timestamp_s"], f"{view} episode start")
                    episode_end = episode_start + (length - 1) / FPS
                    requested = episode_start + local_frame / FPS
                    if not (episode_start <= requested <= episode_end + 1e-9):
                        raise InputValidationError("requested timestamp lies outside the source episode")
                    image, decoded = renderer._decode_rgb(
                        av_backend,
                        video,
                        requested,
                        episode_start_timestamp_s=episode_start,
                        episode_end_timestamp_s=episode_end,
                        actor_anchor_timestamp_s=None,
                    )
                    try:
                        if decoded.get("resolved_path") != str(video) or decoded.get("fps") != FPS:
                            raise InputValidationError(f"native decoder identity/FPS mismatch for {selection_id}/{view}")
                        if abs(float(decoded["decoded_timestamp_s"]) - requested) > (1.0 / 60.0 + 1e-6):
                            raise InputValidationError(f"native decoder PTS error exceeds half-frame for {selection_id}/{view}")
                        decoder_requested = _finite_number(
                            decoded.get("requested_timestamp_s"), f"{selection_id}/{view} decoder request"
                        )
                        if abs(decoder_requested - requested) > 1e-12:
                            raise InputValidationError(f"native decoder request drifted for {selection_id}/{view}")
                        requested_by_view[view] = (image, decoded, decoder_requested)
                    except BaseException:
                        image.close()
                        raise
                for view in VIEWS:
                    image, decoded, decoder_requested = requested_by_view[view]
                    relative = Path("native_rgb") / selection_id / f"window-{item['window_index']:02d}_local-{local_frame:06d}_{view}.png"
                    asset_path = staging / relative
                    asset_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        image.save(asset_path, format="PNG")
                    finally:
                        image.close()
                    png_sha = _sha256(asset_path)
                    records.append({
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
                        "selection_id": selection_id,
                        "candidate_id": selection_row["candidate_id"],
                        "window_index": item["window_index"],
                        "local_frame": local_frame,
                        "dataset_global_frame": item["dataset_global_frame"],
                        "camera_view": view,
                        "source_episode": {
                            **source_record["source"],
                            "episode_length": length,
                            "annotation_sha256": source_record["annotation_sha256"],
                        },
                        "camera_video_locator": {
                            "camera_key": locators[view]["camera_key"],
                            "view": view,
                            "relative_path": locators[view]["relative_path"],
                            "episode_start_timestamp_s": locators[view]["episode_start_timestamp_s"],
                            "expected_fps": 30,
                        },
                        "requested_timestamp_s": decoder_requested,
                        "decoded_timestamp_s": decoded["decoded_timestamp_s"],
                        "pts_error_s": decoded["pts_error_s"],
                        "png_relative_path": str(relative),
                        "png_sha256": png_sha,
                        "png_bytes": asset_path.stat().st_size,
                        "semantic_interpretation": "UNASSIGNED_PRIVATE_VISUAL_VERIFICATION_ONLY",
                    })
        if len(records) != EXPECTED_PNGS:
            raise InputValidationError("decoded PNG count exceeded or fell below the fixed 483-image budget")
        overview_pages = _write_overview_pages(staging, records)
        if len(overview_pages) != EXPECTED_OVERVIEW_PAGES:
            raise InputValidationError("private verifier overview page count drifted")
        frames_path = staging / "frames.jsonl"
        with frames_path.open("xb") as stream:
            for record in records:
                stream.write(_json_line(record))
        manifest = {
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
            "source_selection": {"path": str(selection_path), "sha256": selection_sha},
            "source_result": {"path": str(source_result_path), "sha256": expected_source_result_sha256},
            "sealed_event_index": event_audit,
            "selection_count": EXPECTED_SELECTIONS,
            "candidate_count": EXPECTED_SELECTIONS,
            "window_count": EXPECTED_WINDOWS,
            "local_frame_count": EXPECTED_LOCAL_FRAMES,
            "native_png_count": len(records),
            "overview_page_count": len(overview_pages),
            "frames_jsonl": {"relative_path": "frames.jsonl", "sha256": _sha256(frames_path), "bytes": frames_path.stat().st_size, "rows": len(records)},
            "overview_pages": overview_pages,
            "guards": {
                "private_verifier_only": True,
                "native_camera_rgb_only": True,
                "no_success_failure_recovery_inference": True,
                "no_actor_packet": True,
                "no_contact_sheet_in_actor_packet": True,
                "no_full_video_hashing": True,
                "s08_windows_remain_noncontiguous": True,
            },
            "implementation": {
                "script": str(Path(__file__).resolve()),
                "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
                "stride_frames": STRIDE_FRAMES,
                "camera_views": list(VIEWS),
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_json_line(manifest).rstrip(b"\n") + b"\n")
        manifest_sha = _sha256(manifest_path)
        if output_path.exists():
            raise FileExistsError(f"private verifier output appeared during run: {output_path}")
        os.rename(staging, output_path)
        return {**manifest, "manifest_sha256": manifest_sha, "output_path": str(output_path)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _git_commit(root: Path) -> str | None:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip()
        return value if SHA256_RE.fullmatch(value) is None else value
    except (OSError, subprocess.CalledProcessError):
        return None


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--event-index", type=Path, required=True, help="sealed event-index directory")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-source-result-sha256", required=True)
    parser.add_argument("--expected-event-index-manifest-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        result = run_verifier(
            args.selection,
            args.source_result,
            args.event_index,
            args.raw_root,
            args.output,
            expected_selection_sha256=args.expected_selection_sha256,
            expected_source_result_sha256=args.expected_source_result_sha256,
            expected_event_index_manifest_sha256=args.expected_event_index_manifest_sha256,
        )
    except (InputValidationError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"PRIVATE_VERIFIER_REJECTED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_path": result["output_path"], "manifest_sha256": result["manifest_sha256"],
                      "native_png_count": result["native_png_count"], "overview_page_count": result["overview_page_count"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
