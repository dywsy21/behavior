#!/usr/bin/env python3
"""Build deterministic, actor-causal visual review pages from sealed packets.

The packet renderer is the authority for the temporal sample set.  This
builder consumes only ``actor_packet.causal_temporal_rgb`` for its default
output, validates the packet's absolute source-frame clock as well as its
semantic temporal roles, and verifies every referenced native PNG against the
sealed receipt.  Future samples can be emitted only with ``--audit-output``;
that output is a separate audit-only directory and is never linked from the
default actor manifest.

Question text is intentionally not copied into the helper output.  When an
optional query registry is supplied, exact event/query IDs and a hash of the
original text are retained so a downstream annotator can join by IDs without
using display helper names (``qNNN``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


SCHEMA_VERSION = "p107.memlite.causal-review-pages.v1"
ACTOR_ROLE = "ACTOR_CAUSAL"
FUTURE_ROLE = "OFFLINE_FUTURE_AUDIT"
CAMERAS = ("head", "left_wrist", "right_wrist")
CAMERA_SIZES = {"head": (720, 720), "left_wrist": (480, 480), "right_wrist": (480, 480)}
CAMERA_POSITIONS = {"head": (0, 0), "left_wrist": (720, 0), "right_wrist": (1200, 0)}
CANVAS_SIZE = (1680, 1440)
PACKET_MANIFEST_FIELDS = frozenset({
    "schema_version", "status", "training_eligible", "index_manifest_sha256", "index_event_file_sha256",
    "files", "packets", "rendered_asset_receipts", "decoded_camera_native_rgb",
    "temporal_sample_offsets_frames", "temporal_slots", "distinct_temporal_sample_frames",
    "clamped_duplicate_temporal_samples", "render_requests_sha256", "queue_seal_sha256",
    "review_only_contact_sheets", "full_video_hashing", "renderer_sha256", "protocol_sha256", "wall_seconds",
})
QUEUE_SEAL_FIELDS = frozenset({
    "schema_version", "selection_manifest_sha256", "parent_inventory_seal_sha256",
    "source_release_manifest_sha256", "protocol_sha256", "payload_files", "training_eligible",
})
QUEUE_SEAL_PAYLOAD_FILES = (
    "phase_balanced_queue.jsonl",
    "phase_candidate_index/event_candidates.jsonl",
    "phase_candidate_index/source_groups.jsonl",
    "phase_candidate_index/manifest.json",
    "phase_candidate_index/inventory_seal.json",
)
STANDARD_QUEUE_SEAL_FIELDS = frozenset({
    "schema_version", "queue_manifest_sha256", "inventory_seal_sha256", "source_release_manifest_sha256",
    "canonical_protocol_sha256", "coverage_expectations_sha256", "policy_sha256", "expected_payload_files",
    "payload_files",
})
STANDARD_QUEUE_PAYLOAD_FILES = (
    "annotation_calibration_queue.jsonl", "camera_native_render_requests.jsonl", "counts.json",
    "student_candidate_queue.jsonl",
)
STANDARD_QUEUE_MANIFEST_FIELDS = frozenset({
    "schema_version", "status", "training_eligible", "all_jobs_status", "source_release_manifest_sha256",
    "index_manifest_sha256", "inventory_seal_sha256", "canonical_protocol_sha256",
    "coverage_expectations_sha256", "index_event_file_sha256", "index_source_group_file_sha256", "policy",
    "files", "renderer_handshake", "design_handoff",
})
COVERAGE_MANIFEST_FIELDS = frozenset({
    "schema_version", "status", "training_eligible", "selection_role", "usage_role", "immutable_split",
    "source_release_manifest_sha256", "index_manifest_sha256", "inventory_seal_sha256",
    "coverage_expectations_sha256", "canonical_protocol_sha256", "prior_source_windows_sha256", "policy",
    "files", "renderer_compatibility", "no_outcome_or_action_labels",
})
RENDER_REQUEST_FIELDS = frozenset({
    "schema_version", "request_id", "status", "event_id", "source_identity", "requested_frame_indices",
    "actor_available_frame_indices", "offline_review_before_frame_indices", "offline_review_after_frame_indices",
    "anchor_camera_locators", "frame_locator_resolution", "camera_delivery",
})
RENDER_REQUEST_SOURCE_FIELDS = frozenset({
    "source_release_manifest_sha256", "source_annotation_sha256", "task_index", "task_instance_id",
    "raw_episode_id", "episode_index", "source_group_id",
})
RENDER_REQUEST_DELIVERY_FIELDS = frozenset({
    "camera_native_only", "include_footer", "allow_contact_sheet_in_actor_input", "decoded_by_selector",
    "forbidden_footer_fields",
})
INDEX_MANIFEST_FILES = ("event_candidates.jsonl", "source_groups.jsonl")
INDEX_INVENTORY_FIELDS = frozenset({
    "schema_version", "index_manifest_sha256", "source_release_manifest_sha256",
    "coverage_expectations_sha256", "expected_payload_files", "payload_files",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {label} line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{suffix}")
    return value


def _strict_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _sha_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_asset_path(value: Any) -> bool:
    """Return whether a packet path is a relative path below ``assets``."""

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    relative = Path(value)
    return not relative.is_absolute() and ".." not in relative.parts and relative.parts[:1] == ("assets",)


def _resolve_asset(sealed_root: Path, relative_path: str) -> Path:
    if not _safe_asset_path(relative_path):
        raise ValueError(f"native asset path is not a safe assets-relative path: {relative_path!r}")
    assets_root = (sealed_root / "assets").resolve()
    resolved = (sealed_root / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(assets_root)
    except ValueError as exc:
        raise ValueError(f"native asset path escapes assets root: {relative_path!r}") from exc
    if not resolved.is_file():
        raise ValueError(f"native asset is missing: {relative_path}")
    return resolved


def _camera_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(CAMERAS):
        raise ValueError(f"{name} must contain exactly the three camera views {CAMERAS}")
    return {camera: value[camera] for camera in CAMERAS}


def _verify_image(path: Path, *, expected_size: tuple[int, int], expected_sha: str, expected_bytes: Any,
                  relative_path: str) -> None:
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"native asset SHA-256 mismatch: {relative_path}")
    if expected_bytes is not None and _strict_int(expected_bytes, name=f"bytes for {relative_path}", minimum=0) != path.stat().st_size:
        raise ValueError(f"native asset byte count mismatch: {relative_path}")
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != expected_size:
                raise ValueError(
                    f"native asset geometry/mode mismatch for {relative_path}: "
                    f"format={image.format!r}, mode={image.mode!r}, size={image.size!r}"
                )
            image.load()
    except OSError as exc:
        raise ValueError(f"cannot decode native PNG: {relative_path}") from exc


def _packet_source_binding(packet: Mapping[str, Any], event: Mapping[str, Any], event_id: str) -> None:
    actor = packet.get("actor_packet")
    audit = packet.get("audit")
    if not isinstance(actor, Mapping) or not isinstance(audit, Mapping):
        raise ValueError(f"packet for {event_id} is missing actor_packet/audit")
    if packet.get("packet_id") != actor.get("packet_id"):
        raise ValueError(f"packet/actor packet_id mismatch for {event_id}")
    if actor.get("event_id") != event_id or audit.get("event_id") != event_id:
        raise ValueError(f"packet event_id binding mismatch for {event_id}")
    observation = event.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError(f"event observation is missing for {event_id}")
    anchor = _strict_int(observation.get("frame"), name=f"event observation frame for {event_id}", minimum=0)
    if actor.get("observation_frame") != anchor:
        raise ValueError(f"packet observation frame does not bind event for {event_id}")
    if audit.get("source") != event.get("source"):
        raise ValueError(f"packet source identity does not bind event for {event_id}")


def _validate_schedule(packet: Mapping[str, Any], *, event_id: str, anchor: int) -> tuple[
    list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]
]:
    audit = packet["audit"]
    schedule = audit.get("temporal_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(f"packet temporal schedule is empty or malformed for {event_id}")
    schedule_by_index: dict[int, dict[str, Any]] = {}
    seen_frames: set[int] = set()
    for position, raw_sample in enumerate(schedule):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"packet temporal schedule sample is malformed for {event_id}")
        required = {"sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role", "video_locators"}
        if set(raw_sample) != required:
            raise ValueError(f"packet temporal schedule fields drift for {event_id}")
        sample = dict(raw_sample)
        sample_index = _strict_int(sample["sample_index"], name=f"schedule sample_index for {event_id}", minimum=0)
        if sample_index in schedule_by_index or sample_index != position:
            raise ValueError(f"packet temporal schedule sample indices are not ordered for {event_id}")
        offset = _strict_int(sample["offset_frames"], name=f"schedule offset_frames for {event_id}")
        sample_frame = _strict_int(sample["sample_frame"], name=f"schedule sample_frame for {event_id}", minimum=0)
        _strict_number(sample["timestamp_s"], name=f"schedule timestamp_s for {event_id}")
        if sample_frame != anchor + offset:
            raise ValueError(f"packet temporal source frame/offset contradiction for {event_id}")
        if sample_frame in seen_frames:
            raise ValueError(f"packet temporal schedule contains duplicate source frame for {event_id}")
        seen_frames.add(sample_frame)
        expected_role = ACTOR_ROLE if sample_frame <= anchor else FUTURE_ROLE
        if sample["temporal_role"] != expected_role:
            raise ValueError(f"packet temporal role/frame contradiction for {event_id}")
        locators = sample["video_locators"]
        if not isinstance(locators, list) or len(locators) != len(CAMERAS):
            raise ValueError(f"packet temporal schedule has asymmetric camera locators for {event_id}")
        locator_views: list[Any] = []
        for locator in locators:
            if not isinstance(locator, Mapping) or not isinstance(locator.get("view"), str):
                raise ValueError(f"packet temporal schedule has malformed camera locator for {event_id}")
            locator_views.append(locator["view"])
        if set(locator_views) != set(CAMERAS) or len(set(locator_views)) != len(CAMERAS):
            raise ValueError(f"packet temporal schedule has asymmetric camera locators for {event_id}")
        schedule_by_index[sample_index] = sample
    causal = {index: sample for index, sample in schedule_by_index.items() if sample["temporal_role"] == ACTOR_ROLE}
    future = {index: sample for index, sample in schedule_by_index.items() if sample["temporal_role"] == FUTURE_ROLE}
    if not causal or sum(1 for sample in causal.values() if sample["offset_frames"] == 0) != 1:
        raise ValueError(f"packet temporal schedule must contain exactly one causal anchor for {event_id}")
    return [schedule_by_index[index] for index in range(len(schedule_by_index))], causal, future


def _validate_sample_list(value: Any, *, expected_role: str, event_id: str, anchor: int) -> dict[int, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{expected_role} sample list is malformed for {event_id}")
    samples: dict[int, dict[str, Any]] = {}
    for raw_sample in value:
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"{expected_role} sample is malformed for {event_id}")
        required = {"sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role", "images"}
        if set(raw_sample) != required:
            raise ValueError(f"{expected_role} sample fields drift for {event_id}")
        sample = dict(raw_sample)
        index = _strict_int(sample["sample_index"], name=f"{expected_role} sample_index for {event_id}", minimum=0)
        if index in samples:
            raise ValueError(f"duplicate {expected_role} sample_index for {event_id}")
        offset = _strict_int(sample["offset_frames"], name=f"{expected_role} offset_frames for {event_id}")
        frame = _strict_int(sample["sample_frame"], name=f"{expected_role} sample_frame for {event_id}", minimum=0)
        _strict_number(sample["timestamp_s"], name=f"{expected_role} timestamp_s for {event_id}")
        if frame != anchor + offset:
            raise ValueError(f"{expected_role} source frame/offset contradiction for {event_id}")
        if expected_role == ACTOR_ROLE and frame > anchor:
            raise ValueError(f"actor causal sample is in the future for {event_id}")
        if expected_role == FUTURE_ROLE and frame <= anchor:
            raise ValueError(f"future audit sample is not strictly future for {event_id}")
        if sample["temporal_role"] != expected_role:
            raise ValueError(f"{expected_role} sample has contradictory temporal_role for {event_id}")
        images = _camera_mapping(sample["images"], name=f"{expected_role} images for {event_id}")
        for camera, relative_path in images.items():
            if not _safe_asset_path(relative_path):
                raise ValueError(f"{expected_role} {camera} image path is unsafe for {event_id}")
        samples[index] = sample
    return samples


def _validate_packet_temporal(packet: Mapping[str, Any], event: Mapping[str, Any], event_id: str) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]]
]:
    _packet_source_binding(packet, event, event_id)
    actor = packet["actor_packet"]
    audit = packet["audit"]
    anchor = actor["observation_frame"]
    schedule, causal_schedule, future_schedule = _validate_schedule(packet, event_id=event_id, anchor=anchor)
    causal = _validate_sample_list(actor.get("causal_temporal_rgb"), expected_role=ACTOR_ROLE,
                                   event_id=event_id, anchor=anchor)
    future = _validate_sample_list(audit.get("offline_future_rgb"), expected_role=FUTURE_ROLE,
                                   event_id=event_id, anchor=anchor)
    if set(causal) != set(causal_schedule) or set(future) != set(future_schedule):
        raise ValueError(f"actor/audit temporal lists do not exactly bind packet schedule for {event_id}")
    for index, sample in causal.items():
        scheduled = causal_schedule[index]
        for key in ("offset_frames", "sample_frame", "timestamp_s", "temporal_role"):
            if sample[key] != scheduled[key]:
                raise ValueError(f"actor causal sample disagrees with schedule for {event_id}")
    for index, sample in future.items():
        scheduled = future_schedule[index]
        for key in ("offset_frames", "sample_frame", "timestamp_s", "temporal_role"):
            if sample[key] != scheduled[key]:
                raise ValueError(f"future audit sample disagrees with schedule for {event_id}")
    anchor_samples = [sample for sample in causal.values() if sample["offset_frames"] == 0]
    if len(anchor_samples) != 1:
        raise ValueError(f"actor causal list must contain exactly one anchor for {event_id}")
    actor_images = actor.get("images")
    if actor_images not in ({}, anchor_samples[0]["images"]):
        raise ValueError(f"actor anchor images do not bind causal sample for {event_id}")
    audit_schedule_count = _strict_int(audit.get("temporal_slot_count"), name=f"temporal_slot_count for {event_id}", minimum=1)
    if audit_schedule_count != len(schedule):
        raise ValueError(f"packet temporal_slot_count drift for {event_id}")
    distinct_count = _strict_int(audit.get("distinct_sample_frame_count"), name=f"distinct_sample_frame_count for {event_id}", minimum=1)
    if distinct_count != len(schedule):
        raise ValueError(f"packet distinct_sample_frame_count drift for {event_id}")
    requested_count = _strict_int(audit.get("requested_temporal_slot_count"), name=f"requested_temporal_slot_count for {event_id}", minimum=len(schedule))
    clamped_count = _strict_int(audit.get("clamped_duplicate_sample_frame_count"), name=f"clamped_duplicate_sample_frame_count for {event_id}", minimum=0)
    if clamped_count != requested_count - len(schedule):
        raise ValueError(f"packet clamped-frame provenance drift for {event_id}")
    packet_samples = {**causal, **future}
    expected_receipt_keys = {(sample["sample_index"], camera) for sample in packet_samples.values() for camera in CAMERAS}
    return ([causal[index] for index in sorted(causal)], [future[index] for index in sorted(future)],
            {(sample_index, camera): {"sample": packet_samples[sample_index], "camera": camera}
             for sample_index, camera in expected_receipt_keys})


def _validate_query_registry(path: Path, events: Mapping[str, Mapping[str, Any]],
                             expected_event_ids: Iterable[str] | None = None) -> dict[str, dict[str, str]]:
    rows = _read_jsonl(path, label="query registry")
    event_ids = set(events) if expected_event_ids is None else set(expected_event_ids)
    by_event: dict[str, dict[str, str]] = {}
    seen_query_ids: set[str] = set()
    for row in rows:
        event_id = row.get("event_id")
        query_id = row.get("prelabel_query_id", row.get("query_id"))
        query_text = row.get("query_text")
        if query_text is None and isinstance(row.get("current_visible_goal_relation"), Mapping):
            query_text = row["current_visible_goal_relation"].get("query_text")
        if not isinstance(event_id, str) or event_id not in event_ids:
            raise ValueError("query registry contains an event_id outside the selected queue")
        if event_id in by_event:
            raise ValueError(f"query registry duplicates event_id: {event_id}")
        if not isinstance(query_id, str):
            raise ValueError(f"query registry query ID is missing for {event_id}")
        _sha_text(query_id, name=f"query registry query ID for {event_id}")
        if query_id in seen_query_ids:
            raise ValueError(f"query registry query ID is duplicated for {event_id}")
        if not isinstance(query_text, str) or not query_text:
            raise ValueError(f"query registry query_text is missing for {event_id}")
        event = events[event_id]
        event_frame = event.get("observation", {}).get("frame") if isinstance(event.get("observation"), Mapping) else None
        if row.get("observation_frame", event_frame) != event_frame:
            raise ValueError(f"query registry observation frame does not bind event for {event_id}")
        event_source = event.get("source")
        if (row.get("source_group_id") is not None and isinstance(event_source, Mapping) and
                row["source_group_id"] != event_source.get("source_group_id")):
            raise ValueError(f"query registry source_group_id does not bind event for {event_id}")
        text_sha = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        supplied_sha = row.get("query_text_sha256", row.get("query_sha256"))
        if supplied_sha is not None and _sha_text(supplied_sha, name=f"query text SHA for {event_id}") != text_sha:
            raise ValueError(f"query registry query text SHA mismatch for {event_id}")
        by_event[event_id] = {"query_id": query_id, "query_text_sha256": text_sha}
        seen_query_ids.add(query_id)
    if set(by_event) != event_ids:
        raise ValueError("query registry event IDs are not exactly the selected queue IDs")
    return by_event


def _safe_payload_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    relative = Path(value)
    return not relative.is_absolute() and ".." not in relative.parts and "." not in relative.parts and bool(relative.parts)


def _resolve_payload(base: Path, relative_path: str, *, label: str) -> Path:
    if not _safe_payload_path(relative_path):
        raise ValueError(f"{label} has an unsafe relative path: {relative_path!r}")
    resolved = (base / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its authenticated directory: {relative_path!r}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _verify_file_claim(path: Path, claim: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(claim, Mapping) or set(claim) != {"sha256", "rows", "bytes"}:
        raise ValueError(f"{label} is missing its exact sha256/rows/bytes receipt")
    digest = _sha_text(claim.get("sha256"), name=f"{label} SHA-256")
    rows = _strict_int(claim.get("rows"), name=f"{label} rows", minimum=0)
    byte_count = _strict_int(claim.get("bytes"), name=f"{label} bytes", minimum=0)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != byte_count or sha256(path) != digest:
        raise ValueError(f"{label} bytes or SHA-256 do not match its authenticated receipt")
    actual_rows = sum(1 for line in path.read_bytes().splitlines() if line.strip())
    if actual_rows != rows:
        raise ValueError(f"{label} row count does not match its authenticated receipt")
    return {"sha256": digest, "rows": rows, "bytes": byte_count}


def _verify_manifest_files(manifest: Mapping[str, Any], paths: Mapping[str, Path], *, expected_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    files = manifest.get("files")
    names = tuple(expected_names)
    if not isinstance(files, Mapping) or set(files) != set(names):
        raise ValueError("sealed packet manifest file inventory is missing or expanded")
    return {name: _verify_file_claim(paths[name], files[name], label=f"sealed packet {name}") for name in names}


def _validate_render_requests(render_requests_path: Path | None, expected_render_requests_sha256: str | None,
                              packet_manifest: Mapping[str, Any], queue_meta: Mapping[str, Any],
                              expected_queue_seal_sha256: str, events: Mapping[str, Mapping[str, Any]],
                              ordered_queue: list[Mapping[str, Any]],
                              packets: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]] | None:
    packet_request_sha = packet_manifest.get("render_requests_sha256")
    request_receipt = queue_meta.get("request_receipt")
    if packet_request_sha is None:
        if render_requests_path is not None or expected_render_requests_sha256 is not None:
            raise ValueError("render-request path/hash supplied but the pinned packet manifest claims no requests")
        if request_receipt is not None:
            raise ValueError("queue seal claims render requests but the pinned packet manifest claims none")
        return None
    packet_request_sha = _authenticated_sha(packet_request_sha, name="packet render-request SHA-256")
    if render_requests_path is None:
        raise ValueError("packet manifest claims render requests; --render-requests is required")
    if request_receipt is None:
        raise ValueError("packet manifest claims render requests but the authenticated queue seal has no request receipt")
    if expected_render_requests_sha256 is not None:
        if _authenticated_sha(expected_render_requests_sha256, name="expected render-request SHA-256") != packet_request_sha:
            raise ValueError("expected render-request SHA disagrees with the pinned packet manifest")
    render_requests_path = Path(render_requests_path)
    if render_requests_path.is_symlink() or not render_requests_path.is_file():
        raise ValueError("render-request path must be a regular file")
    expected_path = queue_meta.get("render_requests_path")
    if expected_path is None or render_requests_path.resolve() != Path(expected_path).resolve():
        raise ValueError("render-request path is not the request payload authenticated by the queue seal")
    verified_receipt = _verify_file_claim(render_requests_path, request_receipt, label="queue render requests")
    if verified_receipt["sha256"] != packet_request_sha:
        raise ValueError("packet manifest and queue seal render-request SHA-256 values disagree")
    request_rows = _read_jsonl(render_requests_path, label="camera-native render requests")
    event_ids = {row["event_id"] for row in ordered_queue}
    requests: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        if set(row) != set(RENDER_REQUEST_FIELDS) or row.get("schema_version") != "p107-camera-native-temporal-request-v1":
            raise ValueError("camera-native render request schema has missing or unrecognized fields")
        request_id = row.get("request_id")
        event_id = row.get("event_id")
        if (not isinstance(request_id, str) or _sha_text(request_id, name="render request ID") != request_id or
                not isinstance(event_id, str) or event_id not in event_ids or event_id in requests):
            raise ValueError("camera-native render request has a missing, duplicate, or unselected event binding")
        if row.get("status") != "PENDING_RENDERER_HANDSHAKE_NO_RGB_DECODED":
            raise ValueError("camera-native render request is not pending renderer handshake")
        source = events[event_id].get("source")
        request_source = row.get("source_identity")
        if not isinstance(source, Mapping) or not isinstance(request_source, Mapping) or set(request_source) != set(RENDER_REQUEST_SOURCE_FIELDS):
            raise ValueError(f"render request source identity is malformed for {event_id}")
        for key in RENDER_REQUEST_SOURCE_FIELDS:
            expected = source.get(key)
            if key.endswith("_sha256"):
                _sha_text(request_source.get(key), name=f"render request {key} for {event_id}")
            if request_source.get(key) != expected:
                raise ValueError(f"render request source identity does not bind event for {event_id}")
        frame_lists: dict[str, list[int]] = {}
        for key in ("requested_frame_indices", "actor_available_frame_indices", "offline_review_before_frame_indices",
                    "offline_review_after_frame_indices"):
            value = row.get(key)
            if (not isinstance(value, list) or any(type(frame) is not int or frame < 0 for frame in value)):
                raise ValueError(f"render request frame list is malformed for {event_id}: {key}")
            frame_lists[key] = value
        anchor = _strict_int(events[event_id].get("observation", {}).get("frame"),
                             name=f"render request event frame for {event_id}", minimum=0)
        actor_frames = frame_lists["actor_available_frame_indices"]
        before_frames = frame_lists["offline_review_before_frame_indices"]
        after_frames = frame_lists["offline_review_after_frame_indices"]
        requested_frames = frame_lists["requested_frame_indices"]
        if (len(actor_frames) not in {1, 5} or before_frames != actor_frames[:-1] or len(after_frames) != 5 or
                requested_frames != actor_frames + after_frames or len(requested_frames) not in {6, 10} or
                requested_frames != sorted(requested_frames) or len(set(requested_frames)) != len(requested_frames) or
                actor_frames[-1] != anchor or any(frame > anchor for frame in actor_frames) or
                any(frame > anchor for frame in before_frames) or any(frame <= anchor for frame in after_frames)):
            raise ValueError(f"render request temporal schedule is not causal/after partitioned for {event_id}")
        locators = row.get("anchor_camera_locators")
        if (not isinstance(locators, list) or len(locators) != len(CAMERAS) or
                not all(isinstance(locator, Mapping) for locator in locators) or
                {locator.get("view") for locator in locators} != set(CAMERAS)):
            raise ValueError(f"render request camera locators are asymmetric for {event_id}")
        event_locators = events[event_id].get("video_locators")
        if locators != event_locators:
            raise ValueError(f"render request camera locators do not bind event for {event_id}")
        delivery = row.get("camera_delivery")
        if (not isinstance(delivery, Mapping) or set(delivery) != set(RENDER_REQUEST_DELIVERY_FIELDS) or
                delivery.get("camera_native_only") is not True or delivery.get("include_footer") is not False or
                delivery.get("allow_contact_sheet_in_actor_input") is not False or
                delivery.get("decoded_by_selector") is not False or
                set(delivery.get("forbidden_footer_fields", [])) !=
                {"outcome", "success", "failure", "recovery", "label", "evidence", "review"}):
            raise ValueError(f"render request delivery contract is malformed for {event_id}")
        if row.get("frame_locator_resolution") != "RENDERER_MUST_LOOK_UP_EACH_REQUESTED_FRAME_FROM_FROZEN_SOURCE_METADATA":
            raise ValueError(f"render request frame locator contract is malformed for {event_id}")
        requests[event_id] = row
    if set(requests) != event_ids:
        raise ValueError("camera-native render requests are not bijective with the selected queue events")
    queue_seal_sha = _authenticated_sha(expected_queue_seal_sha256, name="expected queue seal SHA-256")
    for event_id in (row["event_id"] for row in ordered_queue):
        request = requests[event_id]
        packet = packets.get(event_id)
        if packet is None:
            raise ValueError(f"render request has no sealed packet binding for {event_id}")
        audit = packet.get("audit")
        if (not isinstance(audit, Mapping) or audit.get("render_request_id") != request["request_id"]):
            raise ValueError(f"packet render-request ID does not bind request for {event_id}")
        if packet_manifest.get("queue_seal_sha256") != queue_seal_sha:
            raise ValueError("request-backed packet manifest does not bind the explicit queue seal")
        schedule = audit.get("temporal_schedule")
        causal = packet.get("actor_packet", {}).get("causal_temporal_rgb") if isinstance(packet.get("actor_packet"), Mapping) else None
        future = audit.get("offline_future_rgb")
        if not isinstance(schedule, list) or not isinstance(causal, list) or not isinstance(future, list):
            raise ValueError(f"packet temporal schedule is missing for request binding {event_id}")
        schedule_frames = [sample.get("sample_frame") for sample in schedule]
        causal_frames = [sample.get("sample_frame") for sample in causal]
        future_frames = [sample.get("sample_frame") for sample in future]
        if (request["requested_frame_indices"] != schedule_frames or
                request["actor_available_frame_indices"] != causal_frames or
                request["offline_review_after_frame_indices"] != future_frames):
            raise ValueError(f"render request schedule does not bind sealed packet samples for {event_id}")
    return requests


def _authenticated_sha(value: Any, *, name: str) -> str:
    return _sha_text(value, name=name)


def _require_directory(path: Path, *, label: str) -> Path:
    original = Path(path)
    if original.is_symlink() or not original.is_dir():
        raise ValueError(f"{label} must be a regular directory: {original}")
    return original.resolve()


def _queue_source_sha(queue_seal: Mapping[str, Any]) -> str:
    field = "source_release_manifest_sha256"
    return _authenticated_sha(queue_seal.get(field), name=f"queue seal {field}")


def _queue_protocol_sha(queue_seal: Mapping[str, Any]) -> str:
    field = "protocol_sha256" if "protocol_sha256" in queue_seal else "canonical_protocol_sha256"
    return _authenticated_sha(queue_seal.get(field), name=f"queue seal {field}")


def _read_external_queue_seal(queue_seal_path: Path, expected_queue_seal_sha256: str) -> dict[str, Any]:
    queue_seal_path = Path(queue_seal_path)
    if queue_seal_path.is_symlink() or not queue_seal_path.is_file():
        raise ValueError(f"queue seal must be a regular file: {queue_seal_path}")
    expected_queue_seal_sha256 = _authenticated_sha(expected_queue_seal_sha256, name="expected queue seal SHA-256")
    if sha256(queue_seal_path) != expected_queue_seal_sha256:
        raise ValueError("queue seal bytes do not match the externally pinned SHA-256")
    return _read_json(queue_seal_path, label="queue seal")


def _authenticate_phase_queue_and_index(index_root: Path, queue_path: Path, queue_seal_path: Path,
                                        queue_seal: Mapping[str, Any],
                                        render_requests_path: Path | None) -> tuple[
    dict[str, Any], dict[str, Any], Path, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]
]:
    if set(queue_seal) != set(QUEUE_SEAL_FIELDS) or queue_seal.get("schema_version") != "p107-phase-balanced-calibration-seal-v1":
        raise ValueError("queue seal does not use the authenticated phase-calibration schema")
    if queue_seal.get("training_eligible") is not False:
        raise ValueError("queue seal cannot self-promote training eligibility")
    seal_source_sha = _queue_source_sha(queue_seal)
    seal_protocol_sha = _queue_protocol_sha(queue_seal)
    _authenticated_sha(queue_seal.get("selection_manifest_sha256"), name="queue seal selection-manifest SHA-256")
    _authenticated_sha(queue_seal.get("parent_inventory_seal_sha256"), name="queue seal parent-inventory SHA-256")
    payload_files = queue_seal.get("payload_files")
    if not isinstance(payload_files, Mapping) or set(payload_files) != set(QUEUE_SEAL_PAYLOAD_FILES):
        raise ValueError("queue seal payload inventory does not use the exact phase-calibration file set")
    queue_seal_root = queue_seal_path.parent
    payload_paths: dict[str, Path] = {}
    for relative_name in QUEUE_SEAL_PAYLOAD_FILES:
        payload_path = _resolve_payload(queue_seal_root, relative_name, label=f"queue seal payload {relative_name}")
        _verify_file_claim(payload_path, payload_files[relative_name], label=f"queue seal payload {relative_name}")
        payload_paths[relative_name] = payload_path
    queue_path = Path(queue_path).resolve()
    index_root = _require_directory(Path(index_root), label="event index root")
    if queue_path != payload_paths["phase_balanced_queue.jsonl"]:
        raise ValueError("explicit selection queue is not the queue payload authenticated by the seal")
    if index_root != payload_paths["phase_candidate_index/event_candidates.jsonl"].parent:
        raise ValueError("explicit event index root is not the index payload directory authenticated by the seal")

    selection_path = queue_seal_root / "phase_selection_manifest.json"
    if selection_path.is_symlink() or not selection_path.is_file() or sha256(selection_path) != queue_seal["selection_manifest_sha256"]:
        raise ValueError("queue seal selection manifest is missing or does not match its authenticated SHA-256")
    selection_manifest = _read_json(selection_path, label="phase selection manifest")
    if (selection_manifest.get("schema_version") != "p107-phase-balanced-calibration-manifest-v1" or
            selection_manifest.get("status") != "METADATA_CANDIDATES_READY_FOR_INDEPENDENT_RENDER_REVIEW" or
            selection_manifest.get("training_eligible") is not False or
            selection_manifest.get("source_release_manifest_sha256") != seal_source_sha):
        raise ValueError("phase selection manifest schema is not authenticated")
    if selection_manifest.get("files") != payload_files:
        raise ValueError("phase selection manifest payload receipts do not match the queue seal")
    mini_index = selection_manifest.get("mini_index")
    if (not isinstance(mini_index, Mapping) or
            mini_index.get("manifest_sha256") != payload_files["phase_candidate_index/manifest.json"]["sha256"] or
            mini_index.get("inventory_seal_sha256") != payload_files["phase_candidate_index/inventory_seal.json"]["sha256"]):
        raise ValueError("phase selection manifest does not bind the authenticated mini-index files")
    parent_index = selection_manifest.get("parent_index")
    if (not isinstance(parent_index, Mapping) or
            parent_index.get("inventory_seal_sha256") != queue_seal["parent_inventory_seal_sha256"]):
        raise ValueError("phase selection manifest does not bind the authenticated parent inventory")
    policy = selection_manifest.get("policy")
    if (not isinstance(policy, Mapping) or policy.get("source_release_manifest_sha256") != seal_source_sha or
            policy.get("protocol_sha256") != seal_protocol_sha):
        raise ValueError("phase selection policy does not bind queue source/protocol provenance")
    selection = selection_manifest.get("selection")
    if (not isinstance(selection, Mapping) or
            selection.get("exact_budget") != payload_files["phase_balanced_queue.jsonl"]["rows"] or
            selection.get("all_usage_role_annotation_calibration") is not True or
            selection.get("all_immutable_split_train") is not True or
            selection.get("all_evidence_missing") is not True):
        raise ValueError("phase selection manifest does not retain the authenticated TRAIN calibration gate")

    index_manifest_path = index_root / "manifest.json"
    index_manifest = _read_json(index_manifest_path, label="event index manifest")
    if (index_manifest.get("schema_version") != "memlite-event-index-v1" or
            index_manifest.get("status") != "METADATA_ONLY_READY_FOR_PACKET_RENDER" or
            index_manifest.get("training_eligible") is not False):
        raise ValueError("event index manifest schema is not authenticated")
    index_files = index_manifest.get("files")
    if not isinstance(index_files, Mapping) or set(index_files) != set(INDEX_MANIFEST_FILES):
        raise ValueError("event index manifest file inventory is missing or expanded")
    for filename in INDEX_MANIFEST_FILES:
        _verify_file_claim(index_root / filename, index_files[filename], label=f"event index {filename}")
        payload_name = f"phase_candidate_index/{filename}"
        if index_files[filename] != payload_files[payload_name]:
            raise ValueError(f"queue seal and event index receipts disagree for {filename}")
    index_manifest_sha = sha256(index_manifest_path)
    if payload_files["phase_candidate_index/manifest.json"]["sha256"] != index_manifest_sha:
        raise ValueError("queue seal does not bind the event index manifest bytes")
    if (index_manifest.get("source_release_manifest_sha256") != seal_source_sha or
            index_manifest.get("protocol_sha256") != seal_protocol_sha):
        raise ValueError("event index manifest does not bind queue source/protocol provenance")

    inventory_path = index_root / "inventory_seal.json"
    inventory = _read_json(inventory_path, label="event index inventory seal")
    if set(inventory) != set(INDEX_INVENTORY_FIELDS) or inventory.get("schema_version") != "memlite-event-inventory-seal-v1":
        raise ValueError("event index inventory seal schema is not authenticated")
    if (inventory.get("index_manifest_sha256") != index_manifest_sha or
            inventory.get("source_release_manifest_sha256") != seal_source_sha or
            inventory.get("expected_payload_files") != list(INDEX_MANIFEST_FILES) or
            inventory.get("payload_files") != {name: index_files[name] for name in INDEX_MANIFEST_FILES} or
            payload_files["phase_candidate_index/inventory_seal.json"]["sha256"] != sha256(inventory_path)):
        raise ValueError("event index inventory seal does not bind its manifest/payload/source")
    source_groups = {
        row["source_group_id"]: row
        for row in _read_jsonl(index_root / "source_groups.jsonl", label="event index source groups")
        if isinstance(row.get("source_group_id"), str)
    }
    if len(source_groups) != index_files["source_groups.jsonl"]["rows"]:
        raise ValueError("event index source groups contain missing or duplicate source_group_id values")
    for group in source_groups.values():
        if (group.get("source_release_manifest_sha256") != seal_source_sha or
                group.get("original_split") != "train" or group.get("usage_role") != "annotation_calibration"):
            raise ValueError("event index source group does not bind the authenticated TRAIN calibration gate")
    queue_meta = {
        "mode": "phase",
        "selection_manifest_path": selection_path,
        "source_release_manifest_sha256": seal_source_sha,
        "protocol_sha256": seal_protocol_sha,
        "request_receipt": None,
        "render_requests_path": None,
    }
    if render_requests_path is not None:
        raise ValueError("phase queue seal has no authenticated camera-native render-request payload")
    return queue_seal, selection_manifest, selection_path, index_manifest, source_groups, queue_meta


def _authenticate_standard_queue_and_index(index_root: Path, queue_path: Path, queue_seal_path: Path,
                                           queue_seal: Mapping[str, Any],
                                           render_requests_path: Path | None) -> tuple[
    dict[str, Any], dict[str, Any], Path, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]
]:
    if (set(queue_seal) != set(STANDARD_QUEUE_SEAL_FIELDS) or
            queue_seal.get("schema_version") != "p107-metadata-annotation-queue-seal-v1"):
        raise ValueError("queue seal does not use the authenticated canonical annotation schema")
    seal_source_sha = _queue_source_sha(queue_seal)
    seal_protocol_sha = _queue_protocol_sha(queue_seal)
    seal_policy_sha = _authenticated_sha(queue_seal.get("policy_sha256"), name="queue seal policy SHA-256")
    _authenticated_sha(queue_seal.get("inventory_seal_sha256"), name="queue seal inventory SHA-256")
    _authenticated_sha(queue_seal.get("coverage_expectations_sha256"), name="queue seal coverage SHA-256")
    payload_files = queue_seal.get("payload_files")
    if (not isinstance(payload_files, Mapping) or set(payload_files) != set(STANDARD_QUEUE_PAYLOAD_FILES) or
            queue_seal.get("expected_payload_files") != sorted(STANDARD_QUEUE_PAYLOAD_FILES)):
        raise ValueError("canonical queue seal payload inventory is not the exact TRAIN handoff set")
    queue_seal_root = Path(queue_seal_path).parent
    payload_paths: dict[str, Path] = {}
    for relative_name in STANDARD_QUEUE_PAYLOAD_FILES:
        payload_path = _resolve_payload(queue_seal_root, relative_name, label=f"canonical queue payload {relative_name}")
        _verify_file_claim(payload_path, payload_files[relative_name], label=f"canonical queue payload {relative_name}")
        payload_paths[relative_name] = payload_path
    queue_path = Path(queue_path).resolve()
    index_root = _require_directory(Path(index_root), label="event index root")
    if queue_path != payload_paths["annotation_calibration_queue.jsonl"]:
        raise ValueError("explicit selection queue is not the authenticated TRAIN calibration payload")
    request_payload_path = payload_paths["camera_native_render_requests.jsonl"]
    if render_requests_path is None:
        raise ValueError("canonical queue seal claims render requests; --render-requests is required")
    if Path(render_requests_path).resolve() != request_payload_path:
        raise ValueError("explicit render-request path is not the request payload authenticated by the queue seal")

    selection_path = queue_seal_root / "manifest.json"
    if (selection_path.is_symlink() or not selection_path.is_file() or
            sha256(selection_path) != _authenticated_sha(queue_seal.get("queue_manifest_sha256"),
                                                         name="queue seal queue-manifest SHA-256")):
        raise ValueError("canonical queue manifest is missing or does not match its authenticated SHA-256")
    selection_manifest = _read_json(selection_path, label="canonical queue manifest")
    manifest_schema = selection_manifest.get("schema_version")
    if manifest_schema not in {"p107-metadata-annotation-queue-manifest-v1", "p107-diagnostic-coverage-selection-v1"}:
        raise ValueError("canonical queue manifest schema is not authenticated")
    expected_manifest_fields = (STANDARD_QUEUE_MANIFEST_FIELDS if manifest_schema ==
                                "p107-metadata-annotation-queue-manifest-v1" else COVERAGE_MANIFEST_FIELDS)
    if set(selection_manifest) != expected_manifest_fields:
        raise ValueError("canonical queue manifest field inventory is not authenticated")
    if (selection_manifest.get("training_eligible") is not False or
            selection_manifest.get("source_release_manifest_sha256") != seal_source_sha or
            selection_manifest.get("index_manifest_sha256") is None or
            selection_manifest.get("inventory_seal_sha256") != queue_seal.get("inventory_seal_sha256") or
            selection_manifest.get("canonical_protocol_sha256") != seal_protocol_sha):
        raise ValueError("canonical queue manifest does not bind candidate-only source/protocol provenance")
    if manifest_schema == "p107-metadata-annotation-queue-manifest-v1":
        if (selection_manifest.get("status") != "METADATA_CANDIDATES_READY_FOR_RENDERER_HANDSHAKE" or
                selection_manifest.get("all_jobs_status") != "CANDIDATE_MISSING_EVIDENCE"):
            raise ValueError("canonical queue manifest is not a candidate-only TRAIN handoff")
        handshake = selection_manifest.get("renderer_handshake")
        if (not isinstance(handshake, Mapping) or handshake.get("schema_version") != "p107-camera-native-temporal-request-v1" or
                handshake.get("selector_decoded_rgb") is not False or handshake.get("no_footer_truth") is not True):
            raise ValueError("canonical queue renderer handshake is not authenticated")
        policy = selection_manifest.get("policy")
        if (not isinstance(policy, Mapping) or
                policy.get("frozen_source_release_manifest_sha256") != seal_source_sha or
                policy.get("sealed_index_manifest_sha256") != selection_manifest.get("index_manifest_sha256") or
                policy.get("canonical_protocol_sha256") != seal_protocol_sha or
                policy.get("all_jobs_status") != "CANDIDATE_MISSING_EVIDENCE"):
            raise ValueError("canonical queue policy does not bind the TRAIN source/protocol gate")
    else:
        if (selection_manifest.get("status") != "DIAGNOSTIC_CANDIDATES_READY_FOR_INDEPENDENT_REVIEW" or
                selection_manifest.get("selection_role") != "train" or
                selection_manifest.get("usage_role") != "annotation_calibration" or
                selection_manifest.get("immutable_split") != "train" or
                selection_manifest.get("renderer_compatibility") != "EXISTING_CALIBRATION_RENDERER_COMPATIBLE" or
                selection_manifest.get("no_outcome_or_action_labels") is not True):
            raise ValueError("canonical coverage handoff is not an authenticated TRAIN renderer input")
        policy = selection_manifest.get("policy")
        if (not isinstance(policy, Mapping) or policy.get("usage_role") != "annotation_calibration" or
                policy.get("immutable_split") != "train" or policy.get("source_release_manifest_sha256") != seal_source_sha or
                policy.get("canonical_protocol_sha256") != seal_protocol_sha or policy.get("training_eligible") is not False):
            raise ValueError("canonical coverage policy does not bind the TRAIN source/protocol gate")
    if not isinstance(policy, Mapping) or policy.get("policy_sha256") != seal_policy_sha:
        raise ValueError("canonical queue seal policy hash does not bind the authenticated queue manifest")
    manifest_files = selection_manifest.get("files")
    if not isinstance(manifest_files, Mapping):
        raise ValueError("canonical queue manifest file inventory is missing")
    for relative_name, claim in manifest_files.items():
        path = _resolve_payload(queue_seal_root, relative_name, label=f"canonical queue manifest file {relative_name}")
        _verify_file_claim(path, claim, label=f"canonical queue manifest file {relative_name}")
    for relative_name in STANDARD_QUEUE_PAYLOAD_FILES:
        if manifest_files.get(relative_name) != payload_files[relative_name]:
            raise ValueError(f"canonical queue manifest and queue seal receipts disagree for {relative_name}")

    index_manifest_path = index_root / "manifest.json"
    index_manifest = _read_json(index_manifest_path, label="event index manifest")
    if (index_manifest.get("schema_version") != "memlite-event-index-v1" or
            index_manifest.get("status") != "METADATA_ONLY_READY_FOR_PACKET_RENDER" or
            index_manifest.get("training_eligible") is not False):
        raise ValueError("event index manifest schema is not authenticated")
    index_files = index_manifest.get("files")
    if not isinstance(index_files, Mapping) or set(index_files) != set(INDEX_MANIFEST_FILES):
        raise ValueError("event index manifest file inventory is missing or expanded")
    for filename in INDEX_MANIFEST_FILES:
        _verify_file_claim(index_root / filename, index_files[filename], label=f"event index {filename}")
    index_manifest_sha = sha256(index_manifest_path)
    if (selection_manifest.get("index_manifest_sha256") != index_manifest_sha or
            index_manifest.get("source_release_manifest_sha256") != seal_source_sha or
            index_manifest.get("protocol_sha256") != seal_protocol_sha or
            selection_manifest.get("index_event_file_sha256") != index_files["event_candidates.jsonl"]["sha256"] or
            selection_manifest.get("index_source_group_file_sha256") != index_files["source_groups.jsonl"]["sha256"]):
        raise ValueError("canonical queue/index manifest bindings disagree")
    inventory_path = index_root / "inventory_seal.json"
    inventory = _read_json(inventory_path, label="event index inventory seal")
    if set(inventory) != set(INDEX_INVENTORY_FIELDS) or inventory.get("schema_version") != "memlite-event-inventory-seal-v1":
        raise ValueError("event index inventory seal schema is not authenticated")
    if (inventory.get("index_manifest_sha256") != index_manifest_sha or
            inventory.get("source_release_manifest_sha256") != seal_source_sha or
            inventory.get("expected_payload_files") != list(INDEX_MANIFEST_FILES) or
            inventory.get("payload_files") != {name: index_files[name] for name in INDEX_MANIFEST_FILES} or
            queue_seal.get("inventory_seal_sha256") != sha256(inventory_path)):
        raise ValueError("event index inventory seal does not bind its manifest/payload/source")
    source_groups = {
        row["source_group_id"]: row
        for row in _read_jsonl(index_root / "source_groups.jsonl", label="event index source groups")
        if isinstance(row.get("source_group_id"), str)
    }
    if len(source_groups) != index_files["source_groups.jsonl"]["rows"]:
        raise ValueError("event index source groups contain missing or duplicate source_group_id values")
    for group in source_groups.values():
        if group.get("source_release_manifest_sha256") != seal_source_sha:
            raise ValueError("event index source group does not bind the authenticated source release")
    queue_meta = {
        "mode": "canonical_train",
        "selection_manifest_path": selection_path,
        "source_release_manifest_sha256": seal_source_sha,
        "protocol_sha256": seal_protocol_sha,
        "request_receipt": payload_files["camera_native_render_requests.jsonl"],
        "render_requests_path": request_payload_path,
    }
    return queue_seal, selection_manifest, selection_path, index_manifest, source_groups, queue_meta


def _authenticate_queue_and_index(index_root: Path, queue_path: Path, queue_seal_path: Path,
                                  expected_queue_seal_sha256: str,
                                  render_requests_path: Path | None) -> tuple[
    dict[str, Any], dict[str, Any], Path, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]
]:
    queue_seal_path = Path(queue_seal_path).resolve()
    queue_seal = _read_external_queue_seal(queue_seal_path, expected_queue_seal_sha256)
    if queue_seal.get("schema_version") == "p107-phase-balanced-calibration-seal-v1":
        return _authenticate_phase_queue_and_index(index_root, queue_path, queue_seal_path, queue_seal, render_requests_path)
    if queue_seal.get("schema_version") == "p107-metadata-annotation-queue-seal-v1":
        return _authenticate_standard_queue_and_index(index_root, queue_path, queue_seal_path, queue_seal, render_requests_path)
    raise ValueError("queue seal schema is not an authenticated TRAIN queue handoff")


def _authenticate_packet_manifest(sealed_root: Path, index_root: Path, index_manifest: Mapping[str, Any],
                                  queue_seal: Mapping[str, Any], expected_packet_manifest_sha256: str,
                                  expected_queue_seal_sha256: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = sealed_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("sealed packet manifest must be a regular file")
    expected_packet_manifest_sha256 = _authenticated_sha(expected_packet_manifest_sha256, name="expected packet manifest SHA-256")
    if sha256(manifest_path) != expected_packet_manifest_sha256:
        raise ValueError("sealed packet manifest bytes do not match the externally pinned SHA-256")
    manifest = _read_json(manifest_path, label="sealed packet manifest")
    if set(manifest) != set(PACKET_MANIFEST_FIELDS) or manifest.get("schema_version") != "memlite-event-packet-index-v1":
        raise ValueError("sealed packet manifest does not use the authenticated packet-index schema")
    if manifest.get("training_eligible") is not False or manifest.get("status") != "CPU_READY_LABELS_PENDING":
        raise ValueError("sealed packet manifest is not a candidate-only decoded packet set")
    if manifest.get("decoded_camera_native_rgb") is not True or manifest.get("full_video_hashing") is not False:
        raise ValueError("sealed packet manifest does not declare decoded native RGB without full-video hashing")
    packet_paths = {"packets.jsonl": sealed_root / "packets.jsonl", "rendered_asset_receipts.jsonl": sealed_root / "rendered_asset_receipts.jsonl"}
    file_receipts = _verify_manifest_files(manifest, packet_paths, expected_names=packet_paths)
    _authenticated_sha(manifest.get("renderer_sha256"), name="packet renderer SHA-256")
    packet_protocol_sha = _authenticated_sha(manifest.get("protocol_sha256"), name="packet protocol SHA-256")
    if (packet_protocol_sha != index_manifest.get("protocol_sha256") or
            packet_protocol_sha != _queue_protocol_sha(queue_seal)):
        raise ValueError("packet/index/queue protocol provenance disagrees")
    index_manifest_sha = sha256(index_root / "manifest.json")
    if manifest.get("index_manifest_sha256") != index_manifest_sha:
        raise ValueError("packet manifest does not bind the explicit event index manifest")
    index_event_sha = index_manifest["files"]["event_candidates.jsonl"]["sha256"]
    if manifest.get("index_event_file_sha256") != index_event_sha or index_event_sha != sha256(index_root / "event_candidates.jsonl"):
        raise ValueError("packet manifest does not bind the explicit event-candidate payload")
    queue_seal_sha = _authenticated_sha(expected_queue_seal_sha256, name="expected queue seal SHA-256")
    packet_queue_seal_sha = manifest.get("queue_seal_sha256")
    if packet_queue_seal_sha is not None and packet_queue_seal_sha != queue_seal_sha:
        raise ValueError("packet manifest queue-seal provenance disagrees with the explicit queue seal")
    packet_request_sha = manifest.get("render_requests_sha256")
    if packet_request_sha is not None:
        _authenticated_sha(packet_request_sha, name="packet render-request SHA-256")
    offsets = manifest.get("temporal_sample_offsets_frames")
    if (not isinstance(offsets, list) or not offsets or any(type(value) is not int for value in offsets) or
            len(set(offsets)) != len(offsets) or 0 not in offsets):
        raise ValueError("packet manifest temporal offset provenance is malformed")
    packets_count = _strict_int(manifest.get("packets"), name="packet manifest packets", minimum=1)
    receipts_count = _strict_int(manifest.get("rendered_asset_receipts"), name="packet manifest receipts", minimum=1)
    if packets_count != file_receipts["packets.jsonl"]["rows"] or receipts_count != file_receipts["rendered_asset_receipts.jsonl"]["rows"]:
        raise ValueError("packet manifest counters do not match authenticated file receipts")
    temporal_slots = _strict_int(manifest.get("temporal_slots"), name="packet manifest temporal_slots", minimum=packets_count)
    distinct_frames = _strict_int(manifest.get("distinct_temporal_sample_frames"), name="packet manifest distinct frames", minimum=0)
    clamped_frames = _strict_int(manifest.get("clamped_duplicate_temporal_samples"), name="packet manifest clamped frames", minimum=0)
    if temporal_slots != len(offsets) * packets_count or distinct_frames + clamped_frames != temporal_slots:
        raise ValueError("packet manifest temporal counters do not bind its declared offsets and clamp count")
    _strict_number(manifest.get("wall_seconds"), name="packet manifest wall_seconds")
    return manifest, file_receipts


def _load_inputs(sealed_root: Path, index_root: Path, queue_path: Path, queue_seal_path: Path,
                 expected_queue_seal_sha256: str, expected_packet_manifest_sha256: str,
                 render_requests_path: Path | None, expected_render_requests_sha256: str | None,
                 query_registry: Path | None, expected_query_registry_sha256: str | None) -> tuple[
    dict[str, Any], dict[str, Any], Path, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any],
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]],
    dict[tuple[str, int, str], dict[str, Any]], dict[str, dict[str, str]] | None
]:
    sealed_root = _require_directory(Path(sealed_root), label="sealed packet root")
    index_root = _require_directory(Path(index_root), label="event index root")
    queue_path = Path(queue_path)
    if queue_path.is_symlink() or not queue_path.is_file():
        raise ValueError(f"selection queue must be a regular file: {queue_path}")
    queue_path = queue_path.resolve()
    queue_seal, selection_manifest, selection_manifest_path, index_manifest, source_groups, queue_meta = _authenticate_queue_and_index(
        index_root, queue_path, queue_seal_path, expected_queue_seal_sha256, render_requests_path)
    manifest, _file_receipts = _authenticate_packet_manifest(
        sealed_root, index_root, index_manifest, queue_seal, expected_packet_manifest_sha256,
        expected_queue_seal_sha256)
    packets_path = sealed_root / "packets.jsonl"
    receipts_path = sealed_root / "rendered_asset_receipts.jsonl"
    events_rows = _read_jsonl(index_root / "event_candidates.jsonl", label="event candidate index")
    events: dict[str, dict[str, Any]] = {}
    for event in events_rows:
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event candidate is missing event_id")
        if event_id in events:
            raise ValueError(f"event candidate index duplicates event_id: {event_id}")
        source = event.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"event candidate source identity is missing: {event_id}")
        if source.get("source_release_manifest_sha256") != queue_seal["source_release_manifest_sha256"]:
            raise ValueError(f"event candidate source release does not bind queue seal: {event_id}")
        group = source_groups.get(source.get("source_group_id"))
        if group is None:
            raise ValueError(f"event candidate references an unauthenticated source group: {event_id}")
        for key in ("source_release_manifest_sha256", "source_group_id", "task_index", "task_instance_id"):
            if source.get(key) != group.get(key):
                raise ValueError(f"event candidate/source-group identity mismatch for {event_id}")
        events[event_id] = event
    queue = _read_jsonl(queue_path, label="selection queue")
    if not queue:
        raise ValueError("selection queue is empty")
    if queue_meta["mode"] == "phase":
        ordered_queue = sorted(queue, key=lambda row: _strict_int(row.get("selection_order"), name="selection_order", minimum=0))
    else:
        # Canonical annotation queue rows are serialized in the selector's
        # deterministic order but do not carry a display-only qNNN field.
        # Keep the queue order only for helper display; all semantic joins
        # below remain event_id/source based and never use row position.
        ordered_queue = [dict(row, selection_order=position) for position, row in enumerate(queue)]
    orders = [_strict_int(row.get("selection_order"), name="selection_order", minimum=0) for row in ordered_queue]
    if orders != list(range(len(ordered_queue))):
        raise ValueError("selection_order must be exactly 0..N-1")
    queue_by_event: dict[str, dict[str, Any]] = {}
    for row in ordered_queue:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or event_id in queue_by_event:
            raise ValueError("selection queue has a missing or duplicate event_id")
        expected_queue_schema = ("p107-phase-balanced-calibration-queue-v1" if queue_meta["mode"] == "phase"
                                 else "p107-metadata-annotation-queue-v1")
        if (row.get("schema_version") != expected_queue_schema or
                row.get("training_eligible") is not False or row.get("immutable_split") != "train" or
                row.get("usage_role") != "annotation_calibration" or
                (queue_meta["mode"] != "phase" and row.get("queue_kind") != "ANNOTATION_CALIBRATION")):
            raise ValueError("selection queue row is outside the authenticated TRAIN calibration gate")
        if event_id not in events:
            raise ValueError(f"selection queue event_id is absent from event index: {event_id}")
        queue_by_event[event_id] = row
        event = events[event_id]
        event_frame = event.get("observation", {}).get("frame") if isinstance(event.get("observation"), Mapping) else None
        if row.get("observation_frame", event_frame) != event_frame:
            raise ValueError(f"selection queue observation frame does not bind event: {event_id}")
        source = event.get("source")
        queue_source = row.get("source_identity")
        if not isinstance(source, Mapping) or not isinstance(queue_source, Mapping):
            raise ValueError(f"selection queue source identity is missing: {event_id}")
        if queue_meta["mode"] == "canonical_train":
            # The canonical selector keeps source_group_id at the queue-row
            # top level.  It is the authoritative identity binding; older
            # sealed40 rows keep it nested in source_identity and use the
            # legacy branch below.  A nested canonical copy is optional for
            # compatibility, but if present it must agree exactly.
            queue_group = row.get("source_group_id")
            if not isinstance(queue_group, str) or not queue_group:
                raise ValueError(f"canonical queue source_group_id is missing or invalid for {event_id}")
            _sha_text(queue_group, name=f"canonical queue source_group_id for {event_id}")
            if ("source_group_id" in queue_source and
                    queue_source.get("source_group_id") != queue_group):
                raise ValueError(f"canonical queue nested source_group_id does not match top-level field for {event_id}")
            if queue_group != source.get("source_group_id"):
                raise ValueError(f"selection queue source_group_id does not bind event for {event_id}")
        else:
            queue_group = queue_source.get("source_group_id")
            if queue_group != source.get("source_group_id"):
                raise ValueError(f"selection queue/source identity mismatch for {event_id}")
        for key in ("source_release_manifest_sha256", "source_annotation_sha256",
                    "task_index", "task_instance_id", "raw_episode_id", "episode_index"):
            if queue_source.get(key) != source.get(key):
                raise ValueError(f"selection queue/source identity mismatch for {event_id}")
        group = source_groups.get(source.get("source_group_id"))
        if (group is None or group.get("original_split") != "train" or
                group.get("usage_role") != "annotation_calibration"):
            raise ValueError(f"selection queue event is not bound to an authenticated TRAIN calibration source group: {event_id}")
    packet_rows = _read_jsonl(packets_path, label="sealed packets")
    if len(packet_rows) != manifest["packets"]:
        raise ValueError("sealed packet row count does not match the authenticated packet manifest")
    packets: dict[str, dict[str, Any]] = {}
    packet_ids: set[str] = set()
    for packet in packet_rows:
        packet_id = packet.get("packet_id")
        event_id = packet.get("audit", {}).get("event_id") if isinstance(packet.get("audit"), Mapping) else None
        if (not isinstance(packet_id, str) or not isinstance(event_id, str) or event_id in packets or
                packet_id in packet_ids):
            raise ValueError("sealed packets contain a missing or duplicate event binding")
        if event_id not in queue_by_event:
            raise ValueError(f"sealed packet is outside the explicit selection queue: {event_id}")
        packets[event_id] = packet
        packet_ids.add(packet_id)
    if set(packets) != set(queue_by_event):
        raise ValueError("sealed packets and explicit selection queue event IDs are not bijective")
    _validate_render_requests(
        render_requests_path, expected_render_requests_sha256, manifest, queue_meta,
        expected_queue_seal_sha256, events, ordered_queue, packets)
    receipts = _read_jsonl(receipts_path, label="native asset receipts")
    if len(receipts) != manifest["rendered_asset_receipts"]:
        raise ValueError("native receipt row count does not match the authenticated packet manifest")
    native: dict[tuple[str, int, str], dict[str, Any]] = {}
    for receipt in receipts:
        if receipt.get("asset_kind") != "camera_native_rgb_png":
            continue
        packet_id = receipt.get("packet_id")
        sample_index = _strict_int(receipt.get("sample_index"), name="native receipt sample_index", minimum=0)
        view = receipt.get("view")
        if not isinstance(packet_id, str) or view not in CAMERAS:
            raise ValueError("native receipt has an invalid packet/view binding")
        key = (packet_id, sample_index, view)
        if key in native:
            raise ValueError(f"duplicate native receipt key: {key}")
        _sha_text(receipt.get("sha256"), name=f"native receipt SHA for {key}")
        if not _safe_asset_path(receipt.get("relative_path")):
            raise ValueError(f"native receipt has an unsafe relative_path: {key}")
        native[key] = receipt
    if not native:
        raise ValueError("sealed packet output contains no native RGB receipts")
    query_bindings = None
    if query_registry is not None:
        if expected_query_registry_sha256 is None:
            raise ValueError("--query-registry requires --expected-query-registry-sha256")
        query_registry = Path(query_registry)
        if query_registry.is_symlink() or not query_registry.is_file():
            raise ValueError("query registry must be a regular file")
        if sha256(query_registry) != _authenticated_sha(expected_query_registry_sha256, name="expected query registry SHA-256"):
            raise ValueError("query registry bytes do not match the externally pinned SHA-256")
        query_bindings = _validate_query_registry(query_registry, events, expected_event_ids=queue_by_event)
    return (manifest, queue_seal, selection_manifest_path, selection_manifest, index_manifest, source_groups,
            queue_meta, ordered_queue, events, packets, native, query_bindings)


def _verify_packet_receipts(sealed_root: Path, packet: Mapping[str, Any], event: Mapping[str, Any],
                            native: Mapping[tuple[str, int, str], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_id = event["event_id"]
    causal, future, receipt_keys = _validate_packet_temporal(packet, event, event_id)
    packet_id = packet["packet_id"]
    for (sample_index, camera), binding in receipt_keys.items():
        receipt = native.get((packet_id, sample_index, camera))
        if receipt is None:
            raise ValueError(f"missing native receipt for {(packet_id, sample_index, camera)}")
        sample = binding["sample"]
        expected_path = sample["images"].get(camera)
        if receipt.get("relative_path") != expected_path:
            raise ValueError(f"native receipt path does not bind packet sample for {event_id}")
        if receipt.get("sample_frame") != sample["sample_frame"] or receipt.get("temporal_role") != sample["temporal_role"]:
            raise ValueError(f"native receipt frame/role does not bind packet sample for {event_id}")
        path = _resolve_asset(sealed_root, receipt["relative_path"])
        _verify_image(path, expected_size=CAMERA_SIZES[camera], expected_sha=receipt["sha256"],
                      expected_bytes=receipt.get("bytes"), relative_path=receipt["relative_path"])
    expected_keys = {(packet_id, sample_index, camera) for sample_index, camera in receipt_keys}
    actual_packet_keys = {key for key in native if key[0] == packet_id}
    if actual_packet_keys != expected_keys:
        raise ValueError(f"native receipt set is asymmetric or has unexpected samples for {event_id}")
    return causal, future


def _source_asset(sealed_root: Path, receipt: Mapping[str, Any], *, camera: str, sample: Mapping[str, Any],
                  row_index: int) -> dict[str, Any]:
    size = CAMERA_SIZES[camera]
    x, y = CAMERA_POSITIONS[camera]
    return {
        "view": camera,
        "sample_index": sample["sample_index"],
        "sample_frame": sample["sample_frame"],
        "temporal_role": sample["temporal_role"],
        "relative_path": receipt["relative_path"],
        "sha256": receipt["sha256"],
        "native_size": list(size),
        "canvas_xy": [x, y + 720 * row_index],
    }


def _write_pages(staging: Path, helper_id: str, event_id: str, packet_id: str, selection_order: int,
                 samples: list[dict[str, Any]], native: Mapping[tuple[str, int, str], Mapping[str, Any]],
                 sealed_root: Path, *, audit_only: bool) -> tuple[list[dict[str, Any]], set[tuple[str, int, str]]]:
    pages_dir = staging / ("future_audit_pages" if audit_only else "pages")
    pages_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    covered: set[tuple[str, int, str]] = set()
    directory_prefix = "future-audit" if audit_only else helper_id
    for page_index, pair_start in enumerate(range(0, len(samples), 2)):
        pair = samples[pair_start:pair_start + 2]
        canvas = Image.new("RGB", CANVAS_SIZE, color=(0, 0, 0))
        source_assets: list[dict[str, Any]] = []
        for row_index, sample in enumerate(pair):
            for camera in CAMERAS:
                key = (packet_id, sample["sample_index"], camera)
                receipt = native[key]
                path = _resolve_asset(sealed_root, receipt["relative_path"])
                with Image.open(path) as image:
                    image.load()
                    x, y = CAMERA_POSITIONS[camera]
                    canvas.paste(image, (x, y + 720 * row_index))
                covered.add(key)
                source_assets.append(_source_asset(sealed_root, receipt, camera=camera, sample=sample,
                                                   row_index=row_index))
        page_name = f"{directory_prefix}-p{page_index:02d}.png"
        page_rel = f"{'future_audit_pages' if audit_only else 'pages'}/{page_name}"
        page_path = staging / page_rel
        canvas.save(page_path, format="PNG", optimize=False)
        rows.append({
            "helper_id": helper_id,
            "selection_order": selection_order,
            "event_id": event_id,
            "packet_id": packet_id,
            "page_index": page_index,
            "relative_path": page_rel,
            "sha256": sha256(page_path),
            "bytes": page_path.stat().st_size,
            "canvas_size": list(CANVAS_SIZE),
            "layout": "two_times_by_three_cameras_native_pixels_no_resize",
            "temporal_view": "OFFLINE_FUTURE_AUDIT_ONLY" if audit_only else "ACTOR_CAUSAL_ONLY",
            "source_assets": source_assets,
        })
    return rows, covered


def _event_output_row(queue_row: Mapping[str, Any], event: Mapping[str, Any], packet: Mapping[str, Any],
                      query: Mapping[str, str] | None) -> dict[str, Any]:
    source = event.get("source")
    source_identity = {}
    if isinstance(source, Mapping):
        for key in ("source_release_manifest_sha256", "source_annotation_sha256", "source_group_id",
                    "task_index", "task_instance_id", "raw_episode_id", "episode_index"):
            if key in source:
                source_identity[key] = source[key]
    row: dict[str, Any] = {
        "helper_id": f"q{queue_row['selection_order']:03d}",
        "selection_order": queue_row["selection_order"],
        "event_id": event["event_id"],
        "packet_id": packet["packet_id"],
        "observation_frame": event["observation"]["frame"],
        "source_identity": source_identity,
    }
    audit = packet.get("audit")
    if isinstance(audit, Mapping):
        # Keep clamping provenance without copying any future image/path or
        # query text into the actor-facing helper output.
        row["temporal_provenance"] = {
            "causal_packet_field": "actor_packet.causal_temporal_rgb",
            "requested_temporal_slot_count": audit.get("requested_temporal_slot_count"),
            "temporal_slot_count": audit.get("temporal_slot_count"),
            "distinct_sample_frame_count": audit.get("distinct_sample_frame_count"),
            "clamped_duplicate_sample_frame_count": audit.get("clamped_duplicate_sample_frame_count"),
            "actor_causal_sample_indices": [
                sample.get("sample_index") for sample in packet.get("actor_packet", {}).get("causal_temporal_rgb", [])
            ],
        }
    if "task_name" in event:
        row["task_name"] = event["task_name"]
    if query is not None:
        row["prelabel_query_id"] = query["query_id"]
        row["query_text_sha256"] = query["query_text_sha256"]
    return row


def _build_one_output(staging: Path, *, event_rows: list[dict[str, Any]], page_rows: list[dict[str, Any]],
                      manifest: Mapping[str, Any], packets_path: Path, receipts_path: Path,
                      queue_path: Path, queue_seal_path: Path, selection_manifest_path: Path,
                      source_protocol_sha256: str, source_release_manifest_sha256: str,
                      index_root: Path, query_registry: Path | None,
                      temporal_view: str, covered: set[tuple[str, int, str]]) -> None:
    _write_jsonl(staging / "ordered_events.jsonl", event_rows)
    _write_jsonl(staging / "pages.jsonl", page_rows)
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "CAUSAL_REVIEW_HELPERS_ONLY" if temporal_view == "ACTOR_CAUSAL_ONLY" else "FUTURE_AUDIT_HELPERS_ONLY",
        "usage_role": "ACTOR_CAUSAL_REVIEW" if temporal_view == "ACTOR_CAUSAL_ONLY" else "OFFLINE_FUTURE_AUDIT_ONLY",
        "sealed_payload_unchanged": True,
        "questions_embedded": False,
        "downstream_question_join_key": ["event_id", "query_id"],
        "selection_order_is_not_question_join_key": True,
        "temporal_view": temporal_view,
        "future_pages_in_default_output": temporal_view != "ACTOR_CAUSAL_ONLY",
        "events": len(event_rows),
        "pages": len(page_rows),
        "native_source_images_covered_once": len(covered),
        "source_sealed_manifest_sha256": sha256(packets_path.parent / "manifest.json"),
        "source_packets_sha256": sha256(packets_path),
        "source_native_receipts_sha256": sha256(receipts_path),
        "source_queue_sha256": sha256(queue_path),
        "source_queue_seal_sha256": sha256(queue_seal_path),
        "source_selection_manifest_sha256": sha256(selection_manifest_path),
        "source_event_index_sha256": sha256(index_root / "event_candidates.jsonl"),
        "source_index_manifest_sha256": sha256(index_root / "manifest.json"),
        "source_protocol_sha256": source_protocol_sha256,
        "source_release_manifest_sha256": source_release_manifest_sha256,
        "query_registry_sha256": sha256(query_registry) if query_registry else None,
        "query_registry_external_only": query_registry is not None,
        "packet_manifest_status": manifest.get("status"),
    }
    (staging / "manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _reject_output_overlap(targets: Iterable[Path], *, sealed_root: Path, index_root: Path,
                           queue_path: Path, queue_seal_path: Path,
                           render_requests_path: Path | None = None) -> None:
    targets = tuple(targets)
    if len(targets) == 2 and _paths_overlap(targets[0], targets[1]):
        raise ValueError("--output and --audit-output must be disjoint sibling trees")
    input_roots = tuple(path for path in (sealed_root, index_root, queue_path, queue_seal_path, render_requests_path)
                        if path is not None)
    for target in targets:
        for source in input_roots:
            if _paths_overlap(target, source):
                raise ValueError(f"output target overlaps authenticated input: {target} vs {source}")


def _write_owner_marker(staging: Path) -> str:
    token = hashlib.sha256(str(staging).encode("utf-8")).hexdigest()
    (staging / ".causal-review-builder-owner").write_text(token + "\n", encoding="ascii")
    return token


def _remove_owned_final(path: Path, token: str, expected_stat: os.stat_result | None) -> None:
    if not path.is_dir() or path.is_symlink():
        return
    marker = path / ".causal-review-builder-owner"
    try:
        stat = path.stat()
        owner = marker.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return
    if expected_stat is not None and (stat.st_dev, stat.st_ino) != (expected_stat.st_dev, expected_stat.st_ino):
        return
    if owner == token:
        shutil.rmtree(path)


def build(*, sealed_root: Path, index_root: Path, queue_path: Path, output: Path,
          queue_seal_path: Path | None = None, expected_queue_seal_sha256: str | None = None,
          expected_packet_manifest_sha256: str | None = None, render_requests_path: Path | None = None,
          expected_render_requests_sha256: str | None = None, audit_output: Path | None = None,
          query_registry: Path | None = None, expected_query_registry_sha256: str | None = None) -> dict[str, Any]:
    """Build actor-causal pages, optionally followed by a separate future audit set."""

    if queue_seal_path is None or expected_queue_seal_sha256 is None or expected_packet_manifest_sha256 is None:
        raise ValueError("explicit queue seal path/SHA and packet manifest SHA are required")
    sealed_root_raw = Path(sealed_root)
    index_root_raw = Path(index_root)
    queue_path_raw = Path(queue_path)
    queue_seal_path_raw = Path(queue_seal_path)
    render_requests_path_raw = Path(render_requests_path) if render_requests_path is not None else None
    if (sealed_root_raw.is_symlink() or index_root_raw.is_symlink() or queue_path_raw.is_symlink() or
            queue_seal_path_raw.is_symlink() or
            (render_requests_path_raw is not None and render_requests_path_raw.is_symlink())):
        raise ValueError("authenticated input roots/files must not be symlinks")
    sealed_root = sealed_root_raw.resolve(strict=False)
    index_root = index_root_raw.resolve(strict=False)
    queue_path = queue_path_raw.resolve(strict=False)
    queue_seal_path = queue_seal_path_raw.resolve(strict=False)
    render_requests_path = (render_requests_path_raw.resolve(strict=False)
                            if render_requests_path_raw is not None else None)
    output_raw = Path(output)
    if output_raw.is_symlink():
        raise FileExistsError(f"refusing symlink output target: {output_raw}")
    output = output_raw.resolve()
    if audit_output is not None:
        audit_output_raw = Path(audit_output)
        if audit_output_raw.is_symlink():
            raise FileExistsError(f"refusing symlink audit output target: {audit_output_raw}")
        audit_output = audit_output_raw.resolve()
    _reject_output_overlap(tuple(target for target in (output, audit_output) if target is not None),
                           sealed_root=sealed_root, index_root=index_root, queue_path=queue_path,
                           queue_seal_path=queue_seal_path, render_requests_path=render_requests_path)
    for target in (output, audit_output):
        if target is not None and target.exists():
            raise FileExistsError(f"refusing to overwrite existing output directory: {target}")
    (manifest, queue_seal, selection_manifest_path, _selection_manifest, index_manifest, source_groups,
     queue_meta, ordered_queue, events, packets, native, query_bindings) = _load_inputs(
        sealed_root, index_root, queue_path, queue_seal_path, expected_queue_seal_sha256,
        expected_packet_manifest_sha256, render_requests_path, expected_render_requests_sha256,
        query_registry, expected_query_registry_sha256)
    # Authenticate every source image before creating any output directory.
    validated_temporal: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for queue_row in ordered_queue:
        event_id = queue_row["event_id"]
        validated_temporal[event_id] = _verify_packet_receipts(sealed_root, packets[event_id], events[event_id], native)
    requested_slots = sum(packet["audit"]["requested_temporal_slot_count"] for packet in packets.values())
    distinct_frames = sum(packet["audit"]["distinct_sample_frame_count"] for packet in packets.values())
    clamped_slots = sum(packet["audit"]["clamped_duplicate_sample_frame_count"] for packet in packets.values())
    if (manifest["temporal_slots"] != requested_slots or
            manifest["distinct_temporal_sample_frames"] != distinct_frames or
            manifest["clamped_duplicate_temporal_samples"] != clamped_slots):
        raise ValueError("packet manifest temporal counters do not match authenticated packet rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    if audit_output is not None:
        audit_output.parent.mkdir(parents=True, exist_ok=True)
    actor_tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    audit_tmp = (Path(tempfile.mkdtemp(prefix=f".{audit_output.name}.staging-", dir=str(audit_output.parent)))
                 if audit_output else None)
    actor_token = _write_owner_marker(actor_tmp)
    audit_token = _write_owner_marker(audit_tmp) if audit_tmp is not None else None
    actor_pages: list[dict[str, Any]] = []
    audit_pages: list[dict[str, Any]] = []
    actor_events: list[dict[str, Any]] = []
    audit_events: list[dict[str, Any]] = []
    actor_covered: set[tuple[str, int, str]] = set()
    audit_covered: set[tuple[str, int, str]] = set()
    committed_actor_stat: os.stat_result | None = None
    committed_audit_stat: os.stat_result | None = None
    try:
        for queue_row in ordered_queue:
            event_id = queue_row["event_id"]
            event = events[event_id]
            packet = packets[event_id]
            causal, future = validated_temporal[event_id]
            query = query_bindings.get(event_id) if query_bindings is not None else None
            event_row = _event_output_row(queue_row, event, packet, query)
            actor_events.append(event_row)
            pages, covered = _write_pages(actor_tmp, event_row["helper_id"], event_id, packet["packet_id"],
                                          queue_row["selection_order"], causal, native, sealed_root, audit_only=False)
            actor_pages.extend(pages)
            actor_covered.update(covered)
            if audit_tmp is not None:
                audit_events.append(event_row)
                pages, covered = _write_pages(audit_tmp, event_row["helper_id"], event_id, packet["packet_id"],
                                              queue_row["selection_order"], future, native, sealed_root, audit_only=True)
                audit_pages.extend(pages)
                audit_covered.update(covered)
        _build_one_output(actor_tmp, event_rows=actor_events, page_rows=actor_pages, manifest=manifest,
                          packets_path=sealed_root / "packets.jsonl", receipts_path=sealed_root / "rendered_asset_receipts.jsonl",
                          queue_path=queue_path, queue_seal_path=queue_seal_path,
                          selection_manifest_path=selection_manifest_path,
                          source_protocol_sha256=queue_meta["protocol_sha256"],
                          source_release_manifest_sha256=queue_meta["source_release_manifest_sha256"],
                          index_root=index_root, query_registry=query_registry,
                          temporal_view="ACTOR_CAUSAL_ONLY", covered=actor_covered)
        if audit_tmp is not None:
            _build_one_output(audit_tmp, event_rows=audit_events, page_rows=audit_pages, manifest=manifest,
                              packets_path=sealed_root / "packets.jsonl", receipts_path=sealed_root / "rendered_asset_receipts.jsonl",
                              queue_path=queue_path, queue_seal_path=queue_seal_path,
                              selection_manifest_path=selection_manifest_path,
                              source_protocol_sha256=queue_meta["protocol_sha256"],
                              source_release_manifest_sha256=queue_meta["source_release_manifest_sha256"],
                              index_root=index_root, query_registry=query_registry,
                              temporal_view="OFFLINE_FUTURE_AUDIT_ONLY", covered=audit_covered)
        if output.exists() or (audit_output is not None and audit_output.exists()):
            raise FileExistsError("an output target appeared during authenticated build")
        os.rename(actor_tmp, output)
        committed_actor_stat = output.stat()
        if audit_tmp is not None:
            os.rename(audit_tmp, audit_output)
            committed_audit_stat = audit_output.stat()
        (output / ".causal-review-builder-owner").unlink(missing_ok=True)
        if audit_output is not None:
            (audit_output / ".causal-review-builder-owner").unlink(missing_ok=True)
    except BaseException:
        shutil.rmtree(actor_tmp, ignore_errors=True)
        if audit_tmp is not None:
            shutil.rmtree(audit_tmp, ignore_errors=True)
        _remove_owned_final(output, actor_token, committed_actor_stat)
        if audit_output is not None and audit_token is not None:
            _remove_owned_final(audit_output, audit_token, committed_audit_stat)
        raise
    return {
        "events": len(actor_events),
        "actor_pages": len(actor_pages),
        "actor_native_source_images": len(actor_covered),
        "audit_pages": len(audit_pages) if audit_output is not None else 0,
        "audit_native_source_images": len(audit_covered) if audit_output is not None else 0,
        "output": str(output),
        "audit_output": str(audit_output) if audit_output is not None else None,
        "query_registry_rows": len(query_bindings) if query_bindings is not None else 0,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-root", type=Path, required=True,
                        help="sealed packet directory containing manifest.json, packets.jsonl, and native receipts")
    parser.add_argument("--index-root", type=Path, required=True,
                        help="explicit event index directory containing event_candidates.jsonl")
    parser.add_argument("--queue-path", type=Path, required=True,
                        help="explicit selection queue JSONL; no adjacent queue is inferred")
    parser.add_argument("--queue-seal-path", "--queue-seal", dest="queue_seal_path", type=Path, required=True,
                        help="explicit authenticated phase queue seal JSON")
    parser.add_argument("--expected-queue-seal-sha256", required=True,
                        help="external SHA-256 for --queue-seal-path")
    parser.add_argument("--expected-packet-manifest-sha256", required=True,
                        help="external SHA-256 for the sealed packet manifest.json")
    parser.add_argument("--render-requests", type=Path, default=None,
                        help="explicit camera-native render-request JSONL when packet/queue seals claim requests")
    parser.add_argument("--expected-render-requests-sha256", default=None,
                        help="optional external SHA-256 for --render-requests; must match packet and queue receipts")
    parser.add_argument("--output", type=Path, required=True,
                        help="new default actor-causal review directory (must not already exist)")
    parser.add_argument("--audit-output", type=Path, default=None,
                        help="optional separate future audit-only directory (must not already exist)")
    parser.add_argument("--query-registry", type=Path, default=None,
                        help="optional exact event/query registry; query text is validated but not copied to output")
    parser.add_argument("--expected-query-registry-sha256", default=None,
                        help="external SHA-256 for --query-registry")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        summary = build(sealed_root=args.sealed_root, index_root=args.index_root, queue_path=args.queue_path,
                        queue_seal_path=args.queue_seal_path,
                        expected_queue_seal_sha256=args.expected_queue_seal_sha256,
                        expected_packet_manifest_sha256=args.expected_packet_manifest_sha256,
                        render_requests_path=args.render_requests,
                        expected_render_requests_sha256=args.expected_render_requests_sha256,
                        output=args.output, audit_output=args.audit_output, query_registry=args.query_registry,
                        expected_query_registry_sha256=args.expected_query_registry_sha256)
    except (FileExistsError, ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
