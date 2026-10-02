#!/usr/bin/env python3
"""Private RGB verifier for one sealed single-GRASP span.

This is an opt-in adapter for the single-span structural scanner.  It consumes
an externally SHA-pinned explicit frame-list selection, the corresponding
non-trainable structural result, its single-span manifest, and the sealed full
event index.  It deliberately does not translate the input into the older
73-candidate/calibration selection schema.  Native RGB decoding, PTS choice,
safe video resolution, overview construction, and atomic output bookkeeping
are reused from the existing neutral private verifier/packet renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping


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
    "_p107_single_grasp_neutral_verifier",
    Path(__file__).with_name("render_memlite_retry_verifier.py"),
)
_PROTOCOL = _load_module(
    "_p107_single_grasp_event_protocol",
    _ROOT / "src/g05/data/memlite_event_protocol.py",
)

InputValidationError = _NEUTRAL.InputValidationError
_forbid_output = _NEUTRAL._forbid_output
_finite_number = _NEUTRAL._finite_number
_json_line = _NEUTRAL._json_line
_load_json = _NEUTRAL._load_json
_regular_dir = _NEUTRAL._regular_dir
_regular_file = _NEUTRAL._regular_file
_resolve_video = _NEUTRAL._resolve_video
_safe_relative_path = _NEUTRAL._safe_relative_path
_sha256 = _NEUTRAL._sha256


SELECTION_SCHEMA = "p107-single-grasp-private-rgb-selection-v1"
RESULT_SCHEMA = "p107-grasp-span-private-output-v1"
MANIFEST_SCHEMA = "p107-single-grasp-span-action-input-v1"
EVENT_INDEX_SCHEMA = "memlite-event-index-v1"
EVENT_SCHEMA = "memlite-event-recovery-v1"
SCRIPT_SCHEMA = "p107-single-grasp-private-rgb-verifier-v1"
VIEWS = ("head", "left_wrist", "right_wrist")
CAMERA_BY_VIEW = {
    "head": "observation.rgb.zed_link_camera_0",
    "left_wrist": "observation.rgb.left_realsense_link_camera_0",
    "right_wrist": "observation.rgb.right_realsense_link_camera_0",
}
FPS = 30.0
MAX_FRAME_COUNT = 32
MAX_PNG_COUNT = MAX_FRAME_COUNT * len(VIEWS)
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
SHA256_RE = set("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_RE


def _exact_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise InputValidationError(f"{name} must be an integer")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same_json(left: Any, right: Any) -> bool:
    return _canonical(left) == _canonical(right)


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        total += path.stat().st_size
    return total


def _require_pin(actual: str, expected: Any, label: str) -> None:
    if not _is_sha256(expected) or actual != expected:
        raise InputValidationError(f"{label} SHA-256 does not match its external pin")


def _check_gate_map(gates: Mapping[str, Any], *, name: str) -> None:
    expected = {
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
    }
    for field, value in expected.items():
        if gates.get(field) != value:
            raise InputValidationError(f"{name} gate drifted: {field}")


def _load_selection(
    path: Path,
    expected_sha256: str,
    *,
    source_result_path: Path,
    source_result_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    event_index_manifest_sha256: str,
    event_candidates_sha256: str,
) -> dict[str, Any]:
    path = _regular_file(path, "single-GRASP RGB selection")
    _require_pin(_sha256(path), expected_sha256, "single-GRASP RGB selection")
    selection = _load_json(path, "single-GRASP RGB selection")
    if not isinstance(selection, Mapping) or selection.get("schema_version") != SELECTION_SCHEMA:
        raise InputValidationError("single-GRASP RGB selection schema is invalid")
    if selection.get("status") != "PRIVATE_SINGLE_GRASP_RGB_SELECTION_UNDECODED":
        raise InputValidationError("single-GRASP RGB selection status is invalid")
    guards = selection.get("guards")
    if not isinstance(guards, Mapping):
        raise InputValidationError("single-GRASP RGB selection guards are missing")
    _check_gate_map(guards, name="single-GRASP RGB selection")
    for key in ("no_success_failure_recovery_inference", "native_camera_rgb_only"):
        if guards.get(key) is not True:
            raise InputValidationError(f"single-GRASP RGB selection guard drifted: {key}")

    source_pin = selection.get("source_result")
    if (
        not isinstance(source_pin, Mapping)
        or source_pin.get("schema_version") != RESULT_SCHEMA
        or source_pin.get("sha256") != source_result_sha256
        or not _is_sha256(source_pin.get("sha256"))
    ):
        raise InputValidationError("single-GRASP RGB selection source-result pin is invalid")
    manifest_pin = selection.get("single_manifest")
    if (
        not isinstance(manifest_pin, Mapping)
        or manifest_pin.get("schema_version") != MANIFEST_SCHEMA
        or manifest_pin.get("sha256") != manifest_sha256
        or not _is_sha256(manifest_pin.get("sha256"))
    ):
        raise InputValidationError("single-GRASP RGB selection manifest pin is invalid")
    index_pin = selection.get("event_index")
    if (
        not isinstance(index_pin, Mapping)
        or index_pin.get("schema_version") != EVENT_INDEX_SCHEMA
        or index_pin.get("manifest_sha256") != event_index_manifest_sha256
        or index_pin.get("event_candidates_sha256") != event_candidates_sha256
        or not _is_sha256(index_pin.get("manifest_sha256"))
        or not _is_sha256(index_pin.get("event_candidates_sha256"))
    ):
        raise InputValidationError("single-GRASP RGB selection event-index pins are invalid")
    if not isinstance(source_result_path, Path) or not isinstance(manifest_path, Path):
        raise InputValidationError("single-GRASP RGB input paths are invalid")

    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise InputValidationError("single-GRASP RGB selection selected record is missing")
    required = ("selection_id", "candidate_id", "event_id", "episode_index", "source_group_id", "frame_interval")
    if any(field not in selected for field in required):
        raise InputValidationError("single-GRASP RGB selected identity is incomplete")
    if not isinstance(selected.get("selection_id"), str) or not selected["selection_id"]:
        raise InputValidationError("single-GRASP RGB selection_id is invalid")
    if not isinstance(selected.get("candidate_id"), str) or not selected["candidate_id"]:
        raise InputValidationError("single-GRASP RGB candidate_id is invalid")
    if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for character in selected["selection_id"]):
        raise InputValidationError("single-GRASP RGB selection_id is unsafe")
    if not _is_sha256(selected.get("event_id")) or not _is_sha256(selected.get("source_group_id")):
        raise InputValidationError("single-GRASP RGB selected SHA identity is malformed")
    episode = _exact_int(selected.get("episode_index"), "single-GRASP RGB selected episode_index")
    interval = selected.get("frame_interval")
    if not isinstance(interval, list) or len(interval) != 2 or any(type(value) is not int for value in interval):
        raise InputValidationError("single-GRASP RGB selected frame interval is malformed")
    if interval[0] < 0 or interval[0] >= interval[1]:
        raise InputValidationError("single-GRASP RGB selected frame interval is invalid")

    return {"selection": dict(selection), "selected": dict(selected), "selection_sha256": expected_sha256, "episode_index": episode}


def _validate_candidate_result(
    result_path: Path, expected_sha256: str, selection: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result_path = _regular_file(result_path, "single-GRASP structural result")
    _require_pin(_sha256(result_path), expected_sha256, "single-GRASP structural result")
    result = _load_json(result_path, "single-GRASP structural result")
    if not isinstance(result, Mapping) or result.get("schema_version") != RESULT_SCHEMA:
        raise InputValidationError("single-GRASP structural result schema is invalid")
    if result.get("status") != "PASS_STRUCTURAL_GRASP_SPAN_SCAN_NONTRAINABLE" or result.get("role") != "diagnostic_candidate":
        raise InputValidationError("single-GRASP structural result is not the private diagnostic result")
    for field, expected in (
        ("private_only", True),
        ("training_eligible", False),
        ("action_bc_supervision", False),
        ("outcome_supervision", False),
        ("recovery_supervision", False),
        ("dart_supervision", False),
        ("attempt_status", "NOT_APPLICABLE"),
        ("release_status", "NOT_RELEASED"),
    ):
        if result.get(field) != expected:
            raise InputValidationError(f"single-GRASP structural result gate drifted: {field}")
    inputs = result.get("inputs")
    if not isinstance(inputs, Mapping) or not _is_sha256(inputs.get("manifest_sha256")):
        raise InputValidationError("single-GRASP structural result manifest pin is missing")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise InputValidationError("single-GRASP structural result episodes are missing")

    selected = selection["selected"]
    wanted_candidate = str(selected["candidate_id"])
    wanted_event = str(selected["event_id"])
    found: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise InputValidationError("single-GRASP structural result episode is malformed")
        source = episode.get("source_identity")
        if not isinstance(source, Mapping):
            raise InputValidationError("single-GRASP structural result source identity is missing")
        candidates = episode.get("candidates")
        if not isinstance(candidates, list):
            raise InputValidationError("single-GRASP structural result candidates are missing")
        spans = episode.get("grasp_spans")
        if not isinstance(spans, list):
            raise InputValidationError("single-GRASP structural result GRASP spans are missing")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise InputValidationError("single-GRASP structural result candidate is malformed")
            candidate_id = candidate.get("candidate_id")
            if candidate.get("semantic_interpretation") != "NOT_ASSIGNED" or candidate.get("command_structure") != "A_B_A_UNNAMED":
                raise InputValidationError("single-GRASP candidate carries a forbidden command interpretation")
            runs = candidate.get("run_intervals")
            if not isinstance(runs, list) or len(runs) == 0:
                raise InputValidationError("single-GRASP candidate run intervals are malformed")
            frame_span = candidate.get("frame_span")
            if not isinstance(frame_span, list) or len(frame_span) != 2 or any(type(value) is not int for value in frame_span):
                raise InputValidationError("single-GRASP candidate frame span is malformed")
            previous_run_end = frame_span[0]
            for run in runs:
                if not isinstance(run, Mapping) or type(run.get("start_frame")) is not int or type(run.get("end_frame")) is not int:
                    raise InputValidationError("single-GRASP candidate run interval is malformed")
                if (
                    run["start_frame"] < previous_run_end
                    or run["start_frame"] < frame_span[0]
                    or run["start_frame"] >= run["end_frame"]
                    or run["end_frame"] > frame_span[1]
                ):
                    raise InputValidationError("single-GRASP candidate run interval is outside its span")
                previous_run_end = run["end_frame"]
            if frame_span != [runs[0]["start_frame"], runs[-1]["end_frame"]]:
                raise InputValidationError("single-GRASP candidate frame span drifted from runs")
            candidate_source = candidate.get("source_identity")
            if not isinstance(candidate_source, Mapping):
                raise InputValidationError("single-GRASP candidate source identity is missing")
            if candidate_source.get("frame_interval") != frame_span:
                raise InputValidationError("single-GRASP candidate source frame interval drifted")
            # The single-span producer names the episode/span with the base
            # candidate_id, while the structural ABA row may append its
            # channel/run suffix.  Accept either exact spelling only when the
            # immutable source identity selects the same event.
            candidate_base_id = candidate_source.get("candidate_id")
            episode_base_id = episode.get("candidate_id")
            id_matches = candidate_id == wanted_candidate or candidate_base_id == wanted_candidate or episode_base_id == wanted_candidate
            if not id_matches or candidate_source.get("event_id") != wanted_event:
                continue
            if found is not None:
                raise InputValidationError("single-GRASP candidate_id is duplicated")
            span = next(
                (
                    item
                    for item in spans
                    if isinstance(item, Mapping)
                    and item.get("span_id") in {candidate_id, candidate_base_id, episode_base_id}
                    and item.get("frame_interval") == frame_span
                ),
                None,
            )
            if span is None:
                raise InputValidationError("single-GRASP candidate has no matching GRASP span")
            if span.get("frame_interval") != frame_span or not isinstance(span.get("event_binding"), Mapping):
                raise InputValidationError("single-GRASP candidate span binding is inconsistent")
            found = (dict(episode), dict(candidate), dict(span))
    if found is None:
        raise InputValidationError("selected candidate is absent from the structural result")
    episode, candidate, span = found
    source = episode["source_identity"]
    for field in ("episode_index", "source_group_id"):
        if source.get(field) != selected.get(field) or candidate["source_identity"].get(field) != selected.get(field):
            raise InputValidationError(f"single-GRASP selected identity mismatch: {field}")
    if candidate["source_identity"].get("event_id") != selected.get("event_id"):
        raise InputValidationError("single-GRASP selected event identity mismatch")
    if candidate.get("frame_span") != selected.get("frame_interval"):
        raise InputValidationError("single-GRASP selected frame interval mismatch")
    return dict(result), episode, {"candidate": candidate, "span": span, "source": dict(source)}


def _validate_manifest_entry(
    manifest_path: Path,
    expected_sha256: str,
    selection: Mapping[str, Any],
    result: Mapping[str, Any],
    candidate_info: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = _regular_file(manifest_path, "single-GRASP input manifest")
    _require_pin(_sha256(manifest_path), expected_sha256, "single-GRASP input manifest")
    manifest = _load_json(manifest_path, "single-GRASP input manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise InputValidationError("single-GRASP input manifest schema is invalid")
    if manifest.get("status") != "AUTHENTICATED_PRIVATE_SINGLE_GRASP_ACTION_SCAN_INPUT":
        raise InputValidationError("single-GRASP input manifest status is invalid")
    release_sha = manifest.get("release_manifest_sha256")
    if not _is_sha256(release_sha):
        raise InputValidationError("single-GRASP input manifest release pin is malformed")
    constraints = manifest.get("constraints")
    if not isinstance(constraints, Mapping) or constraints.get("read_only") is not True:
        raise InputValidationError("single-GRASP input manifest is not read-only")
    for field, expected in (
        ("split", "train"),
        ("usage_role", "student_candidate"),
        ("single_grasp_span_per_episode", True),
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
            raise InputValidationError(f"single-GRASP input manifest gate drifted: {field}")
    result_inputs = result.get("inputs")
    if not isinstance(result_inputs, Mapping) or result_inputs.get("manifest_sha256") != expected_sha256:
        raise InputValidationError("structural result does not bind the supplied input manifest")
    event_index_source = manifest.get("event_index_source")
    index_pin = selection["selection"].get("event_index")
    if (
        not isinstance(event_index_source, Mapping)
        or event_index_source.get("schema") != EVENT_INDEX_SCHEMA
        or event_index_source.get("sha256") != index_pin.get("event_candidates_sha256")
    ):
        raise InputValidationError("single-GRASP input manifest event-index pin mismatch")

    selected = selection["selected"]
    entries = manifest.get("episodes")
    if not isinstance(entries, list):
        raise InputValidationError("single-GRASP input manifest episodes are missing")
    entry = next((item for item in entries if isinstance(item, Mapping) and item.get("candidate_id") == selected["candidate_id"]), None)
    if entry is None:
        raise InputValidationError("selected candidate is absent from the single-GRASP input manifest")
    source = candidate_info["source"]
    for field in ("episode_index", "raw_episode_id", "task_index", "task_instance_id", "source_group_id"):
        if entry.get(field) != source.get(field) or entry.get(field) != selected.get(field, entry.get(field)):
            raise InputValidationError(f"single-GRASP manifest identity mismatch: {field}")
    if entry.get("usage_role") != "student_candidate" or entry.get("immutable_split") != "train":
        raise InputValidationError("single-GRASP manifest role/split is not student_candidate/train")
    for field in ("private_only", "training_eligible", "action_supervision", "action_bc_supervision", "outcome_supervision", "recovery_supervision", "dart_supervision"):
        if entry.get(field) is not (True if field == "private_only" else False):
            raise InputValidationError(f"single-GRASP manifest entry gate drifted: {field}")
    if entry.get("attempt_status") != "NOT_APPLICABLE" or entry.get("release_status") != "NOT_RELEASED":
        raise InputValidationError("single-GRASP manifest entry status gate drifted")
    spans = entry.get("grasp_spans")
    if not isinstance(spans, list) or len(spans) != 1 or not isinstance(spans[0], Mapping) or spans[0].get("frame_interval") != selected["frame_interval"]:
        raise InputValidationError("single-GRASP manifest selected GRASP span mismatch")
    binding = entry.get("event_bindings", {}).get(entry.get("selection_id")) if isinstance(entry.get("event_bindings"), Mapping) else None
    if not isinstance(binding, Mapping) or binding.get("event_id") != selected["event_id"]:
        raise InputValidationError("single-GRASP manifest selected event binding is missing")
    if not _same_json(binding, candidate_info["span"]["event_binding"]):
        raise InputValidationError("single-GRASP manifest/result event binding drifted")
    return dict(manifest), dict(entry)


def _validate_frame_plan(
    selection: Mapping[str, Any], candidate_info: Mapping[str, Any]
) -> tuple[list[int], list[dict[str, int]], dict[int, int]]:
    candidate = candidate_info["candidate"]
    start, end = candidate["frame_span"]
    plan = selection.get("frame_plan")
    if not isinstance(plan, Mapping):
        raise InputValidationError("single-GRASP frame plan is missing")
    frames = plan.get("local_frames")
    if not isinstance(frames, list) or not frames or len(frames) > MAX_FRAME_COUNT or any(type(frame) is not int for frame in frames):
        raise InputValidationError("single-GRASP frame plan exceeds the bounded frame-list contract")
    if frames != sorted(frames) or len(set(frames)) != len(frames):
        raise InputValidationError("single-GRASP frame plan has duplicate or unsorted frames")
    if any(frame < start or frame >= end for frame in frames):
        raise InputValidationError("single-GRASP frame plan contains an out-of-span frame")
    windows = plan.get("windows")
    if not isinstance(windows, list) or not windows:
        raise InputValidationError("single-GRASP frame plan windows are missing")
    normalized_windows: list[dict[str, int]] = []
    previous_end = -1
    frame_set = set(frames)
    window_index_by_frame: dict[int, int] = {}
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise InputValidationError("single-GRASP frame plan window is malformed")
        interval = window.get("frame_interval")
        if not isinstance(interval, list) or len(interval) != 2 or any(type(value) is not int for value in interval):
            raise InputValidationError("single-GRASP frame plan window interval is malformed")
        window_start, window_end = interval
        stride = _exact_int(window.get("stride_frames"), "single-GRASP frame plan stride")
        if window_start < start or window_start >= window_end or window_end > end or stride <= 0 or previous_end >= window_start:
            raise InputValidationError("single-GRASP frame plan window is outside or overlaps the span")
        expected = set(range(window_start, window_end, stride))
        if not expected <= frame_set:
            raise InputValidationError("single-GRASP frame plan omitted a declared temporal sample")
        for frame in frame_set:
            if window_start <= frame < window_end:
                if frame in window_index_by_frame:
                    raise InputValidationError("single-GRASP frame belongs to multiple windows")
                window_index_by_frame[frame] = index
        normalized_windows.append({"start": window_start, "end": window_end, "stride": stride})
        previous_end = window_end
    if set(window_index_by_frame) != frame_set:
        raise InputValidationError("single-GRASP frame is not bound to exactly one declared window")
    required = plan.get("required_frames")
    if (
        not isinstance(required, list)
        or any(type(frame) is not int for frame in required)
        or len(set(required)) != len(required)
        or not set(required) <= frame_set
    ):
        raise InputValidationError("single-GRASP required frame list is malformed")
    expected_endpoints = [start, end - 1]
    if plan.get("span_endpoints") != expected_endpoints or not set(expected_endpoints) <= frame_set:
        raise InputValidationError("single-GRASP span endpoints are not explicitly sampled")
    runs = candidate.get("run_intervals")
    if not isinstance(runs, list) or len(runs) < 3:
        raise InputValidationError("single-GRASP candidate has no three-run transition contract")
    expected_transitions = [runs[0]["end_frame"] - 1, runs[0]["end_frame"], runs[1]["end_frame"] - 1, runs[1]["end_frame"]]
    if plan.get("transition_frames") != expected_transitions or not set(expected_transitions) <= frame_set:
        raise InputValidationError("single-GRASP transition boundary frames are incomplete")
    middle = plan.get("middle_sample_frames")
    if not isinstance(middle, list) or len(middle) < 2 or any(type(frame) is not int for frame in middle):
        raise InputValidationError("single-GRASP middle run lacks temporal samples")
    if len(set(middle)) != len(middle) or middle != sorted(middle):
        raise InputValidationError("single-GRASP middle samples are not unique and ordered")
    if any(frame not in frame_set or not runs[1]["start_frame"] <= frame < runs[1]["end_frame"] for frame in middle):
        raise InputValidationError("single-GRASP middle samples are outside the middle run")
    if max(middle) - min(middle) < _exact_int(plan.get("stride_frames"), "single-GRASP frame plan stride_frames"):
        raise InputValidationError("single-GRASP middle samples do not span a temporal interval")
    if _exact_int(plan.get("stride_frames"), "single-GRASP frame plan stride_frames") != 30:
        raise InputValidationError("single-GRASP frame plan must use 30-frame temporal sampling")
    if len(frames) * len(VIEWS) > MAX_PNG_COUNT:
        raise InputValidationError("single-GRASP frame plan exceeds the native PNG budget")
    return frames, normalized_windows, window_index_by_frame


def _validate_camera_locators(event: Mapping[str, Any], entry: Mapping[str, Any], length: int) -> dict[str, dict[str, Any]]:
    locators = event.get("video_locators")
    if not isinstance(locators, list) or len(locators) != len(VIEWS):
        raise InputValidationError("single-GRASP event does not contain exactly three camera locators")
    by_view: dict[str, dict[str, Any]] = {}
    clock = entry.get("camera_clock")
    if not isinstance(clock, Mapping):
        raise InputValidationError("single-GRASP manifest camera clock is missing")
    for locator in locators:
        if not isinstance(locator, Mapping) or locator.get("view") not in VIEWS or locator["view"] in by_view:
            raise InputValidationError("single-GRASP camera locator identity is malformed")
        view = str(locator["view"])
        camera = CAMERA_BY_VIEW[view]
        if locator.get("camera_key") != camera or locator.get("expected_fps") != 30 or locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED":
            raise InputValidationError(f"single-GRASP camera locator contract drifted for {view}")
        relative = _safe_relative_path(locator.get("relative_path"), f"single-GRASP {view} video path")
        values = clock.get(camera)
        if not isinstance(values, Mapping):
            raise InputValidationError(f"single-GRASP manifest camera clock missing {camera}")
        chunk = _exact_int(values.get("chunk_index"), f"single-GRASP {camera} chunk")
        file_index = _exact_int(values.get("file_index"), f"single-GRASP {camera} file")
        start = _finite_number(values.get("from_timestamp_s"), f"single-GRASP {camera} start")
        finish = _finite_number(values.get("to_timestamp_s"), f"single-GRASP {camera} end")
        if finish <= start or not math.isclose(finish - start, length / FPS, abs_tol=1e-9):
            raise InputValidationError(f"single-GRASP manifest camera clock duration is invalid for {view}")
        expected_path = f"videos/{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        observation = event.get("observation")
        if not isinstance(observation, Mapping) or type(observation.get("frame")) is not int:
            raise InputValidationError("single-GRASP event observation frame is malformed")
        expected_request = start + observation["frame"] / FPS
        if (
            relative != expected_path
            or not math.isclose(_finite_number(locator.get("episode_start_timestamp_s"), f"single-GRASP {view} episode start"), start, abs_tol=1e-9)
            or not math.isclose(_finite_number(locator.get("requested_timestamp_s"), f"single-GRASP {view} requested timestamp"), expected_request, abs_tol=1e-9)
        ):
            raise InputValidationError(f"single-GRASP camera locator does not bind manifest clock for {view}")
        by_view[view] = dict(locator)
    if set(by_view) != set(VIEWS):
        raise InputValidationError("single-GRASP event camera view set is incomplete")
    return by_view


def _load_event_index(
    event_index_dir: Path,
    expected_manifest_sha256: str,
    expected_candidates_sha256: str,
    expected_release_sha256: str,
    selected_event_id: str,
    expected_event: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_index_dir = _regular_dir(event_index_dir, "sealed single-GRASP event index")
    manifest_path = _regular_file(event_index_dir / "manifest.json", "single-GRASP event index manifest")
    _require_pin(_sha256(manifest_path), expected_manifest_sha256, "single-GRASP event index manifest")
    index_manifest = _load_json(manifest_path, "single-GRASP event index manifest")
    if not isinstance(index_manifest, Mapping) or index_manifest.get("schema_version") != EVENT_INDEX_SCHEMA:
        raise InputValidationError("single-GRASP event index manifest schema is invalid")
    if index_manifest.get("source_release_manifest_sha256") != expected_release_sha256:
        raise InputValidationError("single-GRASP event index release pin mismatch")
    receipt = index_manifest.get("files", {}).get("event_candidates.jsonl") if isinstance(index_manifest.get("files"), Mapping) else None
    event_path = _regular_file(event_index_dir / "event_candidates.jsonl", "single-GRASP event candidates")
    if not isinstance(receipt, Mapping) or receipt.get("sha256") != expected_candidates_sha256 or not _is_sha256(receipt.get("sha256")):
        raise InputValidationError("single-GRASP event candidate receipt is invalid")
    if type(receipt.get("bytes")) is not int or type(receipt.get("rows")) is not int or event_path.stat().st_size != receipt["bytes"]:
        raise InputValidationError("single-GRASP event candidate receipt dimensions are invalid")
    digest = hashlib.sha256()
    rows = 0
    selected_event: dict[str, Any] | None = None
    with event_path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            value = _NEUTRAL._strict_json(line, name=f"{event_path}:{number}")
            if not isinstance(value, Mapping):
                raise InputValidationError("single-GRASP event candidate row is not an object")
            if value.get("event_id") == selected_event_id:
                if selected_event is not None:
                    raise InputValidationError("single-GRASP selected event is duplicated")
                selected_event = dict(value)
    if rows != receipt["rows"] or digest.hexdigest() != expected_candidates_sha256 or selected_event is None:
        raise InputValidationError("single-GRASP full event-index bytes or selected identity drifted")
    try:
        _PROTOCOL.validate_event(selected_event)
    except Exception as exc:
        raise InputValidationError(f"single-GRASP selected event fails the pinned protocol: {exc}") from exc
    if selected_event.get("usage_role") != "student_candidate":
        raise InputValidationError("single-GRASP selected event role is not student_candidate")
    source = selected_event.get("source")
    if not isinstance(source, Mapping):
        raise InputValidationError("single-GRASP selected event source is missing")
    for field in ("episode_index", "raw_episode_id", "task_index", "task_instance_id", "source_group_id"):
        if source.get(field) != entry.get(field):
            raise InputValidationError(f"single-GRASP selected event source mismatch: {field}")
    if source.get("source_annotation_sha256") != entry.get("source_annotation_sha256"):
        raise InputValidationError("single-GRASP selected event source mismatch: source_annotation_sha256")
    if source.get("source_release_manifest_sha256") != expected_release_sha256 or source.get("original_split") != "train":
        raise InputValidationError("single-GRASP selected event split/release mismatch")
    if not _same_json(selected_event.get("event_interval"), expected_event.get("event_interval")):
        raise InputValidationError("single-GRASP selected event interval mismatch")
    for field in ("event_id", "event_kind", "schema_version", "bundle_id", "observation", "action", "skill_bundle", "parallel_bundle", "video_locators"):
        expected_field = expected_event.get("action_contract") if field == "action" else expected_event.get(field)
        if not _same_json(selected_event.get(field), expected_field):
            raise InputValidationError(f"single-GRASP selected event binding mismatch: {field}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "event_path": str(event_path),
        "event_candidates_sha256": expected_candidates_sha256,
        "event_candidates_rows": rows,
    }, selected_event


def _records_manifest(
    *,
    selection_path: Path,
    result_path: Path,
    manifest_path: Path,
    event_audit: Mapping[str, Any],
    selection_sha256: str,
    result_sha256: str,
    manifest_sha256: str,
    selected: Mapping[str, Any],
    records: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    frame_count: int,
    output_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCRIPT_SCHEMA,
        "status": "PRIVATE_SINGLE_GRASP_RGB_DECODED_NONTRAINABLE",
        "role": "private_verifier_only",
        "training_eligible": False,
        "action_bc_supervision": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "dart_supervision": False,
        "attempt_status": "NOT_APPLICABLE",
        "release_status": "NOT_RELEASED",
        "actor_packet_included": False,
        "source_selection": {"path": str(selection_path), "sha256": selection_sha256},
        "source_result": {"path": str(result_path), "sha256": result_sha256},
        "single_manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "sealed_event_index": dict(event_audit),
        "selection_id": selected["selection_id"],
        "candidate_id": selected["candidate_id"],
        "event_id": selected["event_id"],
        "usage_role": "student_candidate",
        "frame_count": frame_count,
        "native_png_count": len(records),
        "overview_page_count": len(pages),
        "artifact_bytes": output_bytes,
        "guards": {
            "private_verifier_only": True,
            "native_camera_rgb_only": True,
            "no_success_failure_recovery_inference": True,
            "no_actor_packet": True,
            "student_role_preserved": True,
            "no_command_polarity_assignment": True,
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "git_commit": _git_commit(_ROOT),
            "camera_views": list(VIEWS),
            "fps": FPS,
            "max_frame_count": MAX_FRAME_COUNT,
            "max_png_count": MAX_PNG_COUNT,
            "max_output_bytes": MAX_OUTPUT_BYTES,
        },
    }


def _git_commit(root: Path) -> str | None:
    try:
        import subprocess

        completed = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_verifier(
    selection_path: Path,
    source_result_path: Path,
    manifest_path: Path,
    event_index_dir: Path,
    raw_root: Path,
    output_path: Path,
    *,
    expected_selection_sha256: str,
    expected_source_result_sha256: str,
    expected_manifest_sha256: str,
    expected_event_index_manifest_sha256: str,
    expected_event_candidates_sha256: str,
    av_backend: Any | None = None,
) -> dict[str, Any]:
    selection_path = Path(selection_path)
    source_result_path = Path(source_result_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    selection_info = _load_selection(
        selection_path,
        expected_selection_sha256,
        source_result_path=source_result_path,
        source_result_sha256=expected_source_result_sha256,
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest_sha256,
        event_index_manifest_sha256=expected_event_index_manifest_sha256,
        event_candidates_sha256=expected_event_candidates_sha256,
    )
    result, episode, candidate_info = _validate_candidate_result(source_result_path, expected_source_result_sha256, selection_info)
    manifest, entry = _validate_manifest_entry(manifest_path, expected_manifest_sha256, selection_info, result, candidate_info)
    frames, windows, window_index_by_frame = _validate_frame_plan(selection_info["selection"], candidate_info)
    expected_event = candidate_info["span"]["event_binding"]
    event_audit, event = _load_event_index(
        event_index_dir,
        expected_event_index_manifest_sha256,
        expected_event_candidates_sha256,
        manifest["release_manifest_sha256"],
        selection_info["selected"]["event_id"],
        expected_event,
        entry,
    )
    locators = _validate_camera_locators(event, entry, _exact_int(entry["length"], "single-GRASP manifest length"))
    raw_root = _regular_dir(raw_root, "single-GRASP RGB raw root")
    _forbid_output(output_path, (selection_path.parent, source_result_path.parent, manifest_path.parent, Path(event_index_dir), raw_root))
    renderer = _NEUTRAL._renderer()
    if av_backend is None:
        av_backend, _image, _draw = renderer._require_decode_dependencies(contact_sheets=False)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=str(output_path.parent)))
    records: list[dict[str, Any]] = []
    try:
        assets = staging / "native_rgb" / selection_info["selected"]["selection_id"]
        assets.mkdir(parents=True)
        length = _exact_int(entry["length"], "single-GRASP manifest length")
        for local_frame in frames:
            window_index = window_index_by_frame[local_frame]
            requested_by_view: dict[str, tuple[Any, dict[str, Any], float]] = {}
            for view in VIEWS:
                locator = locators[view]
                video = _resolve_video(raw_root, locator["relative_path"])
                episode_start = _finite_number(locator["episode_start_timestamp_s"], f"{view} episode start")
                episode_end = episode_start + (length - 1) / FPS
                requested = episode_start + local_frame / FPS
                if not episode_start <= requested <= episode_end + 1e-9:
                    raise InputValidationError("single-GRASP requested frame lies outside the source episode")
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
                        raise InputValidationError(f"single-GRASP decoder identity/FPS mismatch for {view}")
                    decoded_timestamp = _finite_number(decoded.get("decoded_timestamp_s"), f"{view} decoded PTS")
                    pts_error = _finite_number(decoded.get("pts_error_s"), f"{view} PTS error")
                    if abs(decoded_timestamp - requested) > (1.0 / 60.0 + 1e-6) or abs(pts_error - abs(decoded_timestamp - requested)) > 1e-9:
                        raise InputValidationError(f"single-GRASP decoder PTS error exceeds half-frame for {view}")
                    decoder_requested = _finite_number(decoded.get("requested_timestamp_s"), f"{view} decoder request")
                    if abs(decoder_requested - requested) > 1e-12:
                        raise InputValidationError(f"single-GRASP decoder request drifted for {view}")
                    requested_by_view[view] = (image, decoded, decoder_requested)
                except BaseException:
                    image.close()
                    raise
            for view in VIEWS:
                image, decoded, decoder_requested = requested_by_view[view]
                relative = Path("native_rgb") / selection_info["selected"]["selection_id"] / f"window-{window_index:02d}_local-{local_frame:06d}_{view}.png"
                asset_path = staging / relative
                try:
                    image.save(asset_path, format="PNG")
                finally:
                    image.close()
                records.append(
                    {
                        "schema_version": SCRIPT_SCHEMA,
                        "role": "private_verifier_only",
                        "usage_role": "student_candidate",
                        "training_eligible": False,
                        "action_bc_supervision": False,
                        "outcome_supervision": False,
                        "recovery_supervision": False,
                        "dart_supervision": False,
                        "attempt_status": "NOT_APPLICABLE",
                        "release_status": "NOT_RELEASED",
                        "actor_packet_included": False,
                        "selection_id": selection_info["selected"]["selection_id"],
                        "candidate_id": selection_info["selected"]["candidate_id"],
                        "event_id": selection_info["selected"]["event_id"],
                        "episode_index": selection_info["selected"]["episode_index"],
                        "window_index": window_index,
                        "local_frame": local_frame,
                        "dataset_global_frame": _exact_int(entry["dataset_from_index"], "single-GRASP dataset_from_index") + local_frame,
                        "camera_view": view,
                        "source_episode": {
                            "episode_index": entry["episode_index"],
                            "raw_episode_id": entry["raw_episode_id"],
                            "task_index": entry["task_index"],
                            "task_instance_id": entry["task_instance_id"],
                            "source_group_id": entry["source_group_id"],
                            "source_annotation_sha256": entry["source_annotation_sha256"],
                            "source_release_manifest_sha256": manifest["release_manifest_sha256"],
                            "original_split": "train",
                            "usage_role": "student_candidate",
                        },
                        "camera_video_locator": {
                            "camera_key": locator["camera_key"],
                            "view": view,
                            "relative_path": locator["relative_path"],
                            "episode_start_timestamp_s": locator["episode_start_timestamp_s"],
                            "expected_fps": 30,
                        },
                        "requested_timestamp_s": decoder_requested,
                        "decoded_timestamp_s": decoded_timestamp,
                        "pts_error_s": pts_error,
                        "png_relative_path": str(relative),
                        "png_sha256": _sha256(asset_path),
                        "png_bytes": asset_path.stat().st_size,
                        "semantic_interpretation": "UNASSIGNED_PRIVATE_VISUAL_VERIFICATION_ONLY",
                    }
                )
        if len(records) != len(frames) * len(VIEWS) or len(records) > MAX_PNG_COUNT:
            raise InputValidationError("single-GRASP native PNG count exceeded the bounded budget")
        frames_path = staging / "frames.jsonl"
        with frames_path.open("xb") as stream:
            for record in records:
                stream.write(_json_line(record))
        pages = _NEUTRAL._write_overview_pages(staging, records)
        manifest_out = staging / "manifest.json"
        output_bytes = 0
        final_manifest: dict[str, Any] | None = None
        for _attempt in range(5):
            final_manifest = _records_manifest(
                selection_path=selection_path,
                result_path=source_result_path,
                manifest_path=manifest_path,
                event_audit=event_audit,
                selection_sha256=selection_info["selection_sha256"],
                result_sha256=expected_source_result_sha256,
                manifest_sha256=expected_manifest_sha256,
                selected=selection_info["selected"],
                records=records,
                pages=pages,
                frame_count=len(frames),
                output_bytes=output_bytes,
            )
            manifest_out.write_bytes(_json_line(final_manifest))
            measured_bytes = _tree_bytes(staging)
            if measured_bytes > MAX_OUTPUT_BYTES:
                raise InputValidationError("single-GRASP private RGB artifact exceeds 64 MiB")
            if measured_bytes == output_bytes:
                break
            output_bytes = measured_bytes
        else:
            raise InputValidationError("single-GRASP private RGB artifact size did not stabilize")
        if final_manifest is None or final_manifest["artifact_bytes"] != output_bytes:
            raise InputValidationError("single-GRASP private RGB manifest byte count is not sealed")
        if output_path.exists():
            raise FileExistsError(f"single-GRASP private RGB output appeared during run: {output_path}")
        os.rename(staging, output_path)
        return {**final_manifest, "output_path": str(output_path), "artifact_bytes": output_bytes}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--event-index", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-source-result-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-event-index-manifest-sha256", required=True)
    parser.add_argument("--expected-event-candidates-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        result = run_verifier(
            args.selection,
            args.source_result,
            args.manifest,
            args.event_index,
            args.raw_root,
            args.output,
            expected_selection_sha256=args.expected_selection_sha256,
            expected_source_result_sha256=args.expected_source_result_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_event_index_manifest_sha256=args.expected_event_index_manifest_sha256,
            expected_event_candidates_sha256=args.expected_event_candidates_sha256,
        )
    except (InputValidationError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"PRIVATE_SINGLE_GRASP_VERIFIER_REJECTED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_path": result["output_path"], "artifact_bytes": result["artifact_bytes"], "frame_count": result["frame_count"], "native_png_count": result["native_png_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
