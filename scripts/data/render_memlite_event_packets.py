"""Create P107 camera-native annotation packets from an immutable event index.

The default mode only seals source-video locators.  ``--decode`` is an
explicit, bounded operation that writes one RGB PNG per camera and a PTS
receipt; it never hashes whole videos or accepts QA contact-sheet composites.
Contact sheets, when requested, are separately marked review-only and are
never referenced from actor input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_SHA256 = "efdd20642fed24241f38bbdeb4abff6cf4faf1c72a86fe7c2acb32ba7496193b"
_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ProtocolBinding:
    """Exact protocol bytes used to interpret an immutable event index."""

    module: Any
    path: Path
    sha256: str


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _load_protocol(path: Path | None = None, *, expected_sha256: str | None = None) -> ProtocolBinding:
    """Compile one verified protocol byte stream without importing ``g05.data``.

    The current DATA protocol is pinned by default.  A protocol outside this
    checkout is accepted only when its externally recorded SHA-256 is supplied;
    this permits the legacy full-v3 index reader without silently reinterpreting
    its canonical event IDs under the newer protocol.
    """
    default_path = REPO / "src/g05/data/memlite_event_protocol.py"
    explicit_path = path is not None
    path = default_path if path is None else Path(path)
    if explicit_path and not _is_sha256(expected_sha256):
        raise ValueError("--protocol-path requires an externally pinned --expected-protocol-sha256")
    expected_sha256 = DEFAULT_PROTOCOL_SHA256 if not explicit_path else expected_sha256
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"P107 protocol must be a regular file: {path}")
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_sha256:
        raise ValueError("P107 protocol bytes do not match the pinned SHA-256")
    name = f"p107_memlite_event_protocol_{digest}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    required = ("PACKET_SCHEMA_VERSION", "canonical_json", "canonical_sha256", "event_id",
                "validate_event", "validate_source_ref")
    if (module.PACKET_SCHEMA_VERSION != "memlite-event-packet-v1" or
            any(not callable(getattr(module, field, None)) for field in required[1:])):
        raise ValueError("P107 protocol does not expose the exact packet-v1 compatibility API")
    return ProtocolBinding(module=module, path=path.resolve(strict=True), sha256=digest)


_protocol_binding = _load_protocol()
_protocol = _protocol_binding.module
PACKET_SCHEMA_VERSION = _protocol.PACKET_SCHEMA_VERSION
canonical_json = _protocol.canonical_json
canonical_sha256 = _protocol.canonical_sha256
event_id = _protocol.event_id
validate_event = _protocol.validate_event
validate_source_ref = _protocol.validate_source_ref


VIEWS = ("head", "left_wrist", "right_wrist")
PACKET_INDEX_SCHEMA = "memlite-event-packet-index-v1"
PACKET_PAYLOAD_FILES = frozenset(("packets.jsonl", "rendered_asset_receipts.jsonl"))
PACKET_MANIFEST_FIELDS = frozenset((
    "schema_version", "status", "training_eligible", "index_manifest_sha256", "index_event_file_sha256",
    "files", "packets", "rendered_asset_receipts", "decoded_camera_native_rgb",
    "temporal_sample_offsets_frames", "temporal_slots", "distinct_temporal_sample_frames",
    "clamped_duplicate_temporal_samples", "render_requests_sha256", "queue_seal_sha256",
    "review_only_contact_sheets", "full_video_hashing", "expected_usage_role",
    "private_goal_state_review",
    "renderer_sha256", "protocol_sha256", "wall_seconds",
))
TEMPORAL_SAMPLE_OFFSETS = (-90, -60, -30, -15, 0, 15, 30)
MAX_TEMPORAL_OFFSET_FRAMES = 240  # Eight seconds at the source 30 Hz clock.
ACTOR_INSTRUCTION = "Use only the camera-native RGB images and question context. Source annotation context is not ground truth."
GOAL_STATE_REVIEW_QUERY = "At the anchor, is the brown animal toy visibly held by the robot’s right gripper?"
GOAL_STATE_REVIEW_QUERY_SHA256 = hashlib.sha256(GOAL_STATE_REVIEW_QUERY.encode("utf-8")).hexdigest()
ROLE_ANNOTATION_CALIBRATION = "annotation_calibration"
ROLE_STUDENT_CANDIDATE = "student_candidate"
# PTS comes from an integer stream tick while source metadata uses floating
# seconds.  Permit representation error only; this is many orders below one
# 30-Hz frame and never admits a genuine future sample.
CLOCK_ABSOLUTE_EPSILON_S = 1e-9


def _strict_json(data: bytes, *, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON number in {name}: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def _clock_at_or_before(value: float, upper: float) -> bool:
    """Absolute-only source-clock comparison; no epoch-size relative tolerance."""
    return value <= upper + CLOCK_ABSOLUTE_EPSILON_S


def _clock_in_episode(value: float, start: float, end: float) -> bool:
    return _clock_at_or_before(start, value) and _clock_at_or_before(value, end)


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular file is missing: {path}")
    return _strict_json(path.read_bytes(), name=str(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular file is missing: {path}")
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if line.strip():
            value = _strict_json(line, name=f"{path}:{number}")
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(value)
    return rows


def _select_index_events(path: Path, receipt: Mapping[str, Any], event_ids: set[str],
                         limit: int | None) -> list[dict[str, Any]]:
    """Stream a large sealed event JSONL; never materialize the full index for a bounded packet batch."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("event index payload must be a regular file")
    if not isinstance(receipt, Mapping) or set(receipt) != {"sha256", "rows", "bytes"}:
        raise ValueError("event index manifest has an invalid event-candidate receipt")
    if path.stat().st_size != receipt["bytes"]:
        raise ValueError("event index payload does not match its sealed manifest")
    digest, rows, selected, found_ids = hashlib.sha256(), 0, [], set()
    with path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            # Once a finite prefix has been retained, only integrity accounting
            # remains; this keeps peak memory independent of full index size.
            if not event_ids and limit is not None and len(selected) >= limit:
                continue
            value = _strict_json(line, name=f"{path}:{number}")
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            if event_ids:
                event_id_value = value.get("event_id")
                if event_id_value in event_ids:
                    if event_id_value in found_ids:
                        raise ValueError("sealed event index contains duplicate selected event IDs")
                    found_ids.add(event_id_value)
                    selected.append(value)
            else:
                selected.append(value)
    if digest.hexdigest() != receipt["sha256"] or rows != receipt["rows"]:
        raise ValueError("event index payload does not match its sealed manifest")
    if event_ids and found_ids != event_ids:
        raise ValueError("requested event_id absent from sealed event index")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbid_source_output(output: Path, *sources: Path) -> None:
    resolved = output.resolve(strict=False)
    for source in sources:
        source = source.resolve(strict=True)
        if resolved == source or source in resolved.parents:
            raise ValueError("packet output must be outside immutable index/raw source directories")


def _read_event_ids(path: Path | None, direct: Iterable[str]) -> set[str]:
    selected = set(direct)
    if path is None:
        return selected
    if path.is_symlink() or not path.is_file():
        raise ValueError("--event-ids must be a regular text/JSONL file")
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            value = _strict_json(line.encode(), name=f"{path}:{number}")
            line = value.get("event_id") if isinstance(value, dict) else None
        if not isinstance(line, str) or len(line) != 64 or any(c not in "0123456789abcdef" for c in line):
            raise ValueError(f"Invalid event_id at {path}:{number}")
        selected.add(line)
    return selected


def _read_render_requests(path: Path | None, expected_sha256: str | None, *,
                          expected_usage_role: str = ROLE_ANNOTATION_CALIBRATION,
                          private_goal_state_review: bool = False) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Load the sealed per-event temporal request interface; it is not a label source."""
    if expected_usage_role not in {ROLE_ANNOTATION_CALIBRATION, "evaluation_only", ROLE_STUDENT_CANDIDATE}:
        raise ValueError("render request role is not an approved P107 role")
    if private_goal_state_review != (expected_usage_role == ROLE_STUDENT_CANDIDATE):
        raise ValueError("student_candidate render requests require the explicit private review opt-in")
    if path is None and expected_sha256 is None:
        return {}, None
    if path is None or not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or \
            any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ValueError("--render-requests requires an externally pinned --expected-render-requests-sha256")
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("render-request bytes do not match the externally pinned SHA-256")
    requests: dict[str, dict[str, Any]] = {}
    required = {"schema_version", "request_id", "event_id", "source_identity", "anchor_camera_locators",
                "actor_available_frame_indices", "offline_review_before_frame_indices",
                "offline_review_after_frame_indices", "requested_frame_indices", "camera_delivery",
                "frame_locator_resolution", "status"}
    source_keys = {"source_release_manifest_sha256", "source_annotation_sha256", "task_index",
                   "task_instance_id", "raw_episode_id", "episode_index", "source_group_id"}
    delivery_keys = {"camera_native_only", "include_footer", "allow_contact_sheet_in_actor_input",
                     "decoded_by_selector", "forbidden_footer_fields"}
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        request = _strict_json(line, name=f"{path}:{number}")
        if not isinstance(request, dict) or set(request) != required or request.get("schema_version") != "p107-camera-native-temporal-request-v1":
            raise ValueError("render request does not use the exact P107 temporal request schema")
        if (not isinstance(request["event_id"], str) or len(request["event_id"]) != 64 or
                not isinstance(request["request_id"], str) or len(request["request_id"]) != 64 or
                not isinstance(request["source_identity"], dict) or set(request["source_identity"]) != source_keys):
            raise ValueError("render request has invalid event/request/source identity fields")
        source = request["source_identity"]
        if (not isinstance(source["source_release_manifest_sha256"], str) or
                not isinstance(source["source_annotation_sha256"], str) or
                not isinstance(source["source_group_id"], str) or
                any(type(source[key]) is not int for key in ("task_index", "task_instance_id", "raw_episode_id", "episode_index"))):
            raise ValueError("render request source identity is malformed")
        actor, before, after, frames = (request["actor_available_frame_indices"], request["offline_review_before_frame_indices"],
                                        request["offline_review_after_frame_indices"], request["requested_frame_indices"])
        if not all(isinstance(value, list) and all(type(frame) is int and frame >= 0 for frame in value)
                   for value in (actor, before, after, frames)):
            raise ValueError("render request temporal frames are malformed")
        if private_goal_state_review:
            if (expected_usage_role != ROLE_STUDENT_CANDIDATE or len(actor) not in {1, 2} or
                    before != [] or after != [] or frames != actor or frames != sorted(frames) or
                    len(set(frames)) != len(frames)):
                raise ValueError("private student render requests must contain one or two causal frames only")
        elif (len(actor) not in {1, 5} or len(after) != 5 or before != actor[:-1] or
              frames != actor + after or len(frames) not in {6, 10} or frames != sorted(frames) or
              len(set(frames)) != len(frames)):
            raise ValueError("render request temporal frames must be exact unique actor-plus-offline slots")
        delivery = request["camera_delivery"]
        if (not isinstance(delivery, dict) or set(delivery) != delivery_keys or
                delivery["camera_native_only"] is not True or delivery["include_footer"] is not False or
                delivery["allow_contact_sheet_in_actor_input"] is not False or delivery["decoded_by_selector"] is not False or
                set(delivery["forbidden_footer_fields"]) != {"outcome", "success", "failure", "recovery", "label", "evidence", "review"} or
                request["frame_locator_resolution"] != "RENDERER_MUST_LOOK_UP_EACH_REQUESTED_FRAME_FROM_FROZEN_SOURCE_METADATA" or
                request["status"] != "PENDING_RENDERER_HANDSHAKE_NO_RGB_DECODED" or
                not isinstance(request["anchor_camera_locators"], list) or len(request["anchor_camera_locators"]) != 3):
            raise ValueError("render request attempts to alter the camera-native/no-footer contract")
        if request["event_id"] in requests:
            raise ValueError("render request file has duplicate event_id")
        requests[request["event_id"]] = request
    if not requests:
        raise ValueError("render request file contains no bounded temporal jobs")
    return requests, expected_sha256


def _read_queue_seal(path: Path | None, expected_sha256: str | None, *,
                     render_requests_path: Path | None,
                     render_requests_sha256: str | None) -> tuple[str | None, str | None]:
    """Verify the queue seal that authorizes a bounded temporal request file.

    This is deliberately a byte-pinned transport check, not a label or outcome
    authority.  The request file remains independently checked against the
    immutable event index below.
    """
    if path is None and expected_sha256 is None and render_requests_sha256 is None and render_requests_path is None:
        return None, None
    if (path is None or render_requests_path is None or render_requests_path.name != "camera_native_render_requests.jsonl" or
            render_requests_sha256 is None or not isinstance(expected_sha256, str) or
            len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256)):
        raise ValueError("--render-requests requires a queue seal and externally pinned queue/request SHA-256 values")
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("queue-seal bytes do not match the externally pinned SHA-256")
    seal = _read_json(path)
    required = {"schema_version", "canonical_protocol_sha256", "coverage_expectations_sha256",
                "expected_payload_files", "inventory_seal_sha256", "payload_files", "policy_sha256",
                "queue_manifest_sha256", "source_release_manifest_sha256"}
    if not isinstance(seal, Mapping) or set(seal) != required or seal.get("schema_version") != "p107-metadata-annotation-queue-seal-v1":
        raise ValueError("queue seal does not use the exact P107 annotation queue schema")
    payload_files = seal.get("payload_files")
    expected_files = ["annotation_calibration_queue.jsonl", "camera_native_render_requests.jsonl",
                      "counts.json", "student_candidate_queue.jsonl"]
    if (seal.get("expected_payload_files") != expected_files or not isinstance(payload_files, Mapping) or
            set(payload_files) != set(expected_files)):
        raise ValueError("queue seal does not retain the exact queue payload inventory")
    for filename in expected_files:
        receipt = payload_files[filename]
        source = path.parent / filename
        if (not isinstance(receipt, Mapping) or set(receipt) != {"sha256", "rows", "bytes"} or
                not isinstance(receipt.get("sha256"), str) or len(receipt["sha256"]) != 64 or
                any(char not in "0123456789abcdef" for char in receipt["sha256"]) or
                type(receipt.get("rows")) is not int or type(receipt.get("bytes")) is not int or
                receipt["rows"] < 0 or receipt["bytes"] < 0 or source.is_symlink() or not source.is_file() or
                source.stat().st_size != receipt["bytes"] or _sha256(source) != receipt["sha256"] or
                sum(1 for line in source.read_bytes().splitlines() if line.strip()) != receipt["rows"]):
            raise ValueError("queue seal payload inventory does not match exact local bytes/rows")
    request_receipt = payload_files["camera_native_render_requests.jsonl"]
    if request_receipt["sha256"] != render_requests_sha256 or render_requests_path.resolve() != (path.parent / "camera_native_render_requests.jsonl").resolve():
        raise ValueError("queue seal does not bind the supplied temporal render-request bytes")
    for field in ("canonical_protocol_sha256", "coverage_expectations_sha256", "inventory_seal_sha256",
                  "policy_sha256", "queue_manifest_sha256", "source_release_manifest_sha256"):
        value = seal.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("queue seal contains an invalid immutable provenance digest")
    return expected_sha256, seal["source_release_manifest_sha256"]


def _parse_temporal_offsets(value: str) -> tuple[int, ...]:
    try:
        offsets = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--temporal-offset-frames must be seven or ten comma-separated integers") from error
    if not _valid_temporal_offsets(offsets):
        raise argparse.ArgumentTypeError("--temporal-offset-frames requires seven or ten unique values including 0 within +/-240")
    return offsets


def _valid_temporal_offsets(offsets: Any) -> bool:
    """The packet schedule is deliberately bounded before decode."""
    return (isinstance(offsets, (tuple, list)) and len(offsets) in {7, 10} and len(set(offsets)) == len(offsets) and
            0 in offsets and all(type(offset) is int and abs(offset) <= MAX_TEMPORAL_OFFSET_FRAMES
                                 for offset in offsets))


def _questions(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load actor prompt context, not labels; fail on answer-like payload keys."""
    if path is None:
        return {}
    prompts: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if set(row) != {"event_id", "question_context"} or not isinstance(row["event_id"], str):
            raise ValueError("question rows require exactly event_id and question_context")
        context = row["question_context"]
        if not isinstance(context, dict):
            raise ValueError("question_context must be an object")
        if _actor_has_forbidden_key(context):
            raise ValueError("actor question_context must not carry a label/review/evidence field")
        if row["event_id"] in prompts:
            raise ValueError("duplicate actor question context")
        prompts[row["event_id"]] = context
    return prompts


def _source_annotation_context(event: Mapping[str, Any]) -> dict[str, Any]:
    """Expose a source instruction as a question context, never as visual text or truth."""
    return {
        "task_context": event.get("task_name"),
        "source_annotation_context": event.get("skill_bundle"),
        "source_annotation_is_not_truth": True,
    }


def _temporal_identity(samples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: sample[key] for key in ("sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role")}
            for sample in samples]


def _packet_id(index_sha256: str, event: Mapping[str, Any], context: Mapping[str, Any], decode: bool,
               temporal_samples: Iterable[Mapping[str, Any]], render_request_id: str | None,
               *, protocol_binding: ProtocolBinding) -> str:
    protocol = protocol_binding.module
    return protocol.canonical_sha256({
        "schema_version": protocol.PACKET_SCHEMA_VERSION,
        "index_manifest_sha256": index_sha256,
        "event_id": event["event_id"],
        "source_release_manifest_sha256": event["source"]["source_release_manifest_sha256"],
        "source_group_id": event["source"]["source_group_id"],
        "observation_frame": event["observation"]["frame"],
        "question_context": context,
        "decoded_camera_native_rgb": decode,
        "temporal_samples": _temporal_identity(temporal_samples),
        "render_request_id": render_request_id,
    })


def _locators_by_view(event: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = event["video_locators"]
    by_view = {row.get("view"): row for row in rows if isinstance(row, Mapping)}
    if len(rows) != 3 or set(by_view) != set(VIEWS) or len(by_view) != len(rows):
        raise ValueError("packet rendering requires exactly three unique source camera locators")
    for view, locator in by_view.items():
        if locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED":
            raise ValueError(f"unexpected source locator status for {view}")
        requested = locator.get("requested_timestamp_s")
        if type(requested) not in (int, float) or isinstance(requested, bool):
            raise ValueError("source locator has no finite requested timestamp")
        expected = locator.get("episode_start_timestamp_s") + event["observation"]["frame"] / 30.0
        if abs(float(requested) - expected) > 1e-9 or locator.get("expected_fps") != 30:
            raise ValueError("source video locator clock mismatch")
    return by_view


def _temporal_samples(event: Mapping[str, Any], offsets_frames: tuple[int, ...]) -> list[dict[str, Any]]:
    if not _valid_temporal_offsets(offsets_frames):
        raise ValueError("temporal packet rendering requires seven or ten unique offsets including zero within +/-240 frames")
    anchor = event["observation"]["frame"]
    length = event["source"]["episode_length"]
    # At an episode edge multiple requested slots can clamp to the same source
    # row.  Render it once, retaining the closest requested offset (therefore
    # the actual zero-offset anchor), rather than forging duplicate windows.
    retained: dict[int, tuple[int, int]] = {}
    for position, offset in enumerate(offsets_frames):
        frame = min(length - 1, max(0, anchor + offset))
        current = retained.get(frame)
        if current is None or (abs(offset), position) < (abs(current[1]), current[0]):
            retained[frame] = (position, offset)
    samples = []
    for sample_index, (_position, offset) in enumerate(sorted(retained.values())):
        frame = min(length - 1, max(0, anchor + offset))
        samples.append({"sample_index": sample_index, "offset_frames": offset, "sample_frame": frame,
                        "timestamp_s": frame / 30.0,
                        "temporal_role": "ACTOR_CAUSAL" if frame <= anchor else "OFFLINE_FUTURE_AUDIT"})
    return samples


def _student_goal_state_temporal_samples(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the two-anchor causal schedule for the explicit private pilot.

    The phase producer seals the parent interval and phase name.  ENTRY keeps
    its one anchor; TERMINAL includes only the parent's first frame and the
    terminal anchor.  This intentionally does not synthesize a post-terminal
    frame (for the pilot that would be 540/689).
    """
    lineage = event.get("phase_lineage")
    if (not isinstance(lineage, Mapping) or lineage.get("private_goal_state_review") is not True or
            lineage.get("goal_state_question") != GOAL_STATE_REVIEW_QUERY or
            lineage.get("goal_state_question_sha256") != GOAL_STATE_REVIEW_QUERY_SHA256):
        raise ValueError("student private packet requires a sealed goal-state phase lineage")
    interval = lineage.get("parent_event_interval")
    if (not isinstance(interval, Mapping) or type(interval.get("start_frame")) is not int or
            type(interval.get("end_frame")) is not int or interval["end_frame"] <= interval["start_frame"]):
        raise ValueError("student private packet has no valid parent event interval")
    anchor = event.get("observation", {}).get("frame")
    length = event.get("source", {}).get("episode_length")
    if type(anchor) is not int or type(length) is not int or not 0 <= anchor < length:
        raise ValueError("student private packet has an invalid episode anchor")
    start, end = interval["start_frame"], interval["end_frame"]
    if not start <= anchor < end or start < 0 or end > length:
        raise ValueError("student private packet parent interval escapes its source episode")
    phase = lineage.get("observation_phase")
    if phase == "ENTRY":
        if anchor != start:
            raise ValueError("student ENTRY anchor is not the sealed parent start")
        frames = [anchor]
    elif phase == "TERMINAL":
        if anchor != end - 1 or start == anchor:
            raise ValueError("student TERMINAL anchor is not the sealed parent end-1")
        frames = [start, anchor]
    else:
        raise ValueError("student private packet supports only ENTRY and TERMINAL phases")
    return [{"sample_index": index, "offset_frames": frame - anchor, "sample_frame": frame,
             "timestamp_s": frame / 30.0, "temporal_role": "ACTOR_CAUSAL"}
            for index, frame in enumerate(frames)]


def _request_temporal_samples(event: Mapping[str, Any], request: Mapping[str, Any], *,
                              protocol_binding: ProtocolBinding,
                              expected_usage_role: str = ROLE_ANNOTATION_CALIBRATION,
                              private_goal_state_review: bool = False) -> list[dict[str, Any]]:
    """Validate each sealed queue job against its canonical indexed source before decode."""
    if expected_usage_role not in {ROLE_ANNOTATION_CALIBRATION, "evaluation_only", ROLE_STUDENT_CANDIDATE}:
        raise ValueError("render request role is not an approved P107 role")
    if private_goal_state_review != (expected_usage_role == ROLE_STUDENT_CANDIDATE):
        raise ValueError("student_candidate render requests require the explicit private review opt-in")
    source = event["source"]
    fields = ("source_release_manifest_sha256", "source_annotation_sha256", "task_index", "task_instance_id",
              "raw_episode_id", "episode_index", "source_group_id")
    if (request["event_id"] != event["event_id"] or any(request["source_identity"][key] != source[key] for key in fields) or
            event.get("usage_role") != expected_usage_role or
            protocol_binding.module.canonical_json(request["anchor_camera_locators"]) !=
            protocol_binding.module.canonical_json(event["video_locators"])):
        raise ValueError("sealed render request does not bind the canonical event role/source/camera locators")
    expected_split = "eval" if expected_usage_role == "evaluation_only" else "train"
    if source.get("original_split") != expected_split:
        raise ValueError("sealed render request role does not bind the canonical immutable split")
    frames, anchor = request["requested_frame_indices"], event["observation"]["frame"]
    actor_frames = request.get("actor_available_frame_indices")
    if (not frames or not isinstance(actor_frames, list) or not actor_frames or
            actor_frames[-1] != anchor or
            any(frame >= source["episode_length"] for frame in frames)):
        raise ValueError("sealed render request crosses its canonical source episode or observation anchor")
    if private_goal_state_review:
        if (request["actor_available_frame_indices"] != frames or
                request["offline_review_before_frame_indices"] or
                request["offline_review_after_frame_indices"] or
                any(frame > anchor for frame in frames)):
            raise ValueError("private student render request contains non-causal or offline frames")
        expected = _student_goal_state_temporal_samples(event)
        if [sample["sample_frame"] for sample in expected] != frames:
            raise ValueError("private student render request does not match the sealed two-anchor schedule")
        return [{"sample_index": index, "offset_frames": frame - anchor, "sample_frame": frame,
                 "timestamp_s": frame / 30.0, "temporal_role": "ACTOR_CAUSAL"}
                for index, frame in enumerate(frames)]
    return [{"sample_index": index, "offset_frames": frame - anchor, "sample_frame": frame,
             "timestamp_s": frame / 30.0,
             "temporal_role": "ACTOR_CAUSAL" if frame <= anchor else "OFFLINE_FUTURE_AUDIT"}
            for index, frame in enumerate(frames)]


def _temporal_locator(locator: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    return {**locator, "requested_timestamp_s": locator["episode_start_timestamp_s"] + sample["timestamp_s"],
            "sample_index": sample["sample_index"], "offset_frames": sample["offset_frames"],
            "sample_frame": sample["sample_frame"], "temporal_role": sample["temporal_role"]}


def _resolve_video(raw_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("video locator must be a safe relative source path")
    path = (raw_root / relative).resolve(strict=True)
    if raw_root.resolve(strict=True) not in path.parents or not path.is_file():
        raise ValueError("video locator escapes raw root or is not a regular file")
    return path


def _require_decode_dependencies(*, contact_sheets: bool) -> tuple[Any, Any | None, Any | None]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("--decode requires PyAV; install it in an isolated data environment, not the shared training env") from error
    if not contact_sheets:
        return av, None, None
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("--contact-sheets requires Pillow together with --decode") from error
    return av, Image, ImageDraw


def _decode_rgb(av: Any, video: Path, requested: float, *, episode_start_timestamp_s: float,
                episode_end_timestamp_s: float,
                actor_anchor_timestamp_s: float | None) -> tuple[Any, dict[str, Any]]:
    """Decode a bounded source PTS without crossing an episode in a shared container.

    ``episode_*`` are source-row global PTS bounds, not MP4 duration.  A
    causal actor frame may round up by less than half a frame, but never past
    the actor anchor; ties choose the non-future PTS.
    """
    if not _clock_in_episode(requested, episode_start_timestamp_s, episode_end_timestamp_s):
        raise ValueError("requested frame lies outside the source episode global PTS bounds")
    if actor_anchor_timestamp_s is not None and not _clock_at_or_before(requested, actor_anchor_timestamp_s):
        raise ValueError("actor camera request cannot be later than the pre-action observation clock")
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError(f"source has no video stream: {video}")
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate is not None else float("nan")
        if abs(rate - 30.0) > 1e-6:
            raise ValueError(f"source FPS changed: {video} ({rate})")
        if stream.time_base is None:
            raise ValueError(f"source stream has no time base: {video}")
        tolerance = 1.0 / 60.0 + 1e-6
        container.seek(max(0, int(max(episode_start_timestamp_s, requested - tolerance) /
                                  float(stream.time_base))), stream=stream, backward=True, any_frame=False)
        best = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = float(frame.pts * stream.time_base)
            # A file can contain contiguous demonstrations.  Never use an
            # adjacent episode merely because it is close to the requested
            # PTS, and never expose a rounded future frame at the actor clock.
            if actual < episode_start_timestamp_s - tolerance:
                continue
            if actual > episode_end_timestamp_s + tolerance:
                break
            distance = abs(actual - requested)
            allowed = (_clock_in_episode(actual, episode_start_timestamp_s, episode_end_timestamp_s) and
                       (actor_anchor_timestamp_s is None or _clock_at_or_before(actual, actor_anchor_timestamp_s)))
            candidate_rank = (distance, not _clock_at_or_before(actual, requested), actual)
            if allowed and distance <= tolerance and (best is None or candidate_rank < best[0]):
                image = frame.to_image().convert("RGB")
                if best is not None:
                    best[3].close()
                best = (candidate_rank, actual, image, image)
            if actual > requested + tolerance:
                break
        if best is None:
            raise ValueError(f"cannot decode requested source PTS within half-frame: {video}")
        stat = video.stat()
        return best[3], {
            "resolved_path": str(video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "requested_timestamp_s": requested, "decoded_timestamp_s": best[1],
            "pts_error_s": best[0][0], "fps": rate, "resolution": list(best[3].size),
            "episode_global_pts_bounds_s": [episode_start_timestamp_s, episode_end_timestamp_s],
            "actor_anchor_timestamp_s": actor_anchor_timestamp_s,
            "full_video_sha256": None,
        }


def _review_contact_sheet(samples: list[tuple[Mapping[str, Any], Mapping[str, Any]]], path: Path,
                          Image: Any, ImageDraw: Any) -> None:
    """Timestamped review-only composite; no text is present in actor-native PNGs."""
    sizes = [samples[0][1][view].size for view in VIEWS]
    width = sum(size[0] for size in sizes)
    row_height = max(size[1] for size in sizes) + 30
    height = row_height * len(samples)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (sample, images) in enumerate(samples):
        x, y = 0, row_index * row_height
        for view in VIEWS:
            image = images[view]
            sheet.paste(image, (x, y + 30))
            draw.text((x + 4, y + 6), f"{view} f={sample['sample_frame']} t={sample['timestamp_s']:.3f}s", fill="black")
            x += image.size[0]
    sheet.save(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *,
                 protocol_binding: ProtocolBinding) -> dict[str, Any]:
    count = 0
    with path.open("xb") as stream:
        for row in rows:
            stream.write(protocol_binding.module.canonical_json(row).encode() + b"\n")
            count += 1
    return {"sha256": _sha256(path), "rows": count, "bytes": path.stat().st_size}


def create_packets(index: Path, output: Path, *, event_ids: set[str], limit: int | None,
                   questions: Mapping[str, Mapping[str, Any]], include_source_annotation_context: bool,
                   decode: bool, raw_root: Path | None, contact_sheets: bool,
                   temporal_offsets_frames: tuple[int, ...] = TEMPORAL_SAMPLE_OFFSETS,
                   render_requests: Mapping[str, Mapping[str, Any]] | None = None,
                   render_requests_sha256: str | None = None, queue_seal_sha256: str | None = None,
                   queue_seal_source_release_manifest_sha256: str | None = None,
                   expected_usage_role: str = ROLE_ANNOTATION_CALIBRATION,
                   private_goal_state_review: bool = False,
                   protocol_binding: ProtocolBinding | None = None,
                   max_seconds: float = 1800.0) -> dict[str, Any]:
    index, output = Path(index), Path(output)
    protocol_binding = _protocol_binding if protocol_binding is None else protocol_binding
    protocol = protocol_binding.module
    if expected_usage_role not in {ROLE_ANNOTATION_CALIBRATION, "evaluation_only", ROLE_STUDENT_CANDIDATE}:
        raise ValueError("render request role is not an approved P107 role")
    if private_goal_state_review != (expected_usage_role == ROLE_STUDENT_CANDIDATE):
        raise ValueError("student_candidate rendering requires the explicit private review opt-in")
    if output.exists():
        raise FileExistsError("packet output already exists; use --resume only to verify its sealed receipt")
    if index.is_symlink() or not index.is_dir():
        raise ValueError("index must be an existing regular directory")
    if not event_ids and limit is None:
        raise ValueError("select explicit --event-id/--event-ids or provide a finite --limit")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if not private_goal_state_review and not _valid_temporal_offsets(temporal_offsets_frames):
        raise ValueError("temporal packet rendering requires seven or ten unique offsets including zero within +/-240 frames")
    if render_requests is None:
        render_requests = {}
    if render_requests:
        if (not event_ids or limit is not None or set(render_requests) != event_ids or
                not isinstance(render_requests_sha256, str) or not isinstance(queue_seal_sha256, str) or
                not isinstance(queue_seal_source_release_manifest_sha256, str)):
            raise ValueError("sealed render requests require exactly their explicit event IDs, no --limit, and pinned queue receipts")
    elif (render_requests_sha256 is not None or queue_seal_sha256 is not None or
          queue_seal_source_release_manifest_sha256 is not None):
        raise ValueError("queue/request receipts require sealed per-event render requests")
    if contact_sheets and not decode:
        raise ValueError("--contact-sheets requires --decode camera-native images")
    if private_goal_state_review and (contact_sheets or include_source_annotation_context):
        raise ValueError("private student packets cannot include contact sheets or source annotation context")
    if decode:
        if raw_root is None or raw_root.is_symlink() or not raw_root.is_dir():
            raise ValueError("--decode requires an existing non-symlink --raw-root")
        av, Image, ImageDraw = _require_decode_dependencies(contact_sheets=contact_sheets)
    else:
        av = Image = ImageDraw = None
    _forbid_source_output(output, index, *( [raw_root] if raw_root is not None else [] ))
    index_manifest_path = index / "manifest.json"
    manifest = _read_json(index_manifest_path)
    if manifest.get("schema_version") != "memlite-event-index-v1":
        raise ValueError("input is not a sealed P107 event index")
    if manifest.get("protocol_sha256") != protocol_binding.sha256:
        raise ValueError("sealed event index protocol does not match the pinned compatibility reader")
    if (render_requests and queue_seal_source_release_manifest_sha256 !=
            manifest.get("source_release_manifest_sha256")):
        raise ValueError("sealed temporal queue does not bind the input index source release")
    index_sha = _sha256(index_manifest_path)
    receipt = manifest.get("files", {}).get("event_candidates.jsonl", {})
    selected = _select_index_events(index / "event_candidates.jsonl", receipt, event_ids, limit)
    if not selected:
        raise ValueError("selection produced no event packets")
    selected_ids = {event["event_id"] for event in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("sealed event index contains duplicate selected event IDs")
    if set(questions) - selected_ids:
        raise ValueError("question context references an event outside the bounded packet selection")
    if private_goal_state_review and set(questions) != selected_ids:
        raise ValueError("private student packets require one pre-frozen question context per selected event")
    started = time.monotonic()
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    packets = []
    asset_receipts = []
    seen_packet_ids = set()
    native_asset_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    expected_split = "eval" if expected_usage_role == "evaluation_only" else "train"
    try:
        if decode:
            (staging / "assets").mkdir()
            if contact_sheets:
                (staging / "review_assets").mkdir()
        for event in selected:
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("packet rendering CPU budget reached; no packet directory was published")
            if (event.get("usage_role") != expected_usage_role or
                    event.get("source", {}).get("original_split") != expected_split):
                raise ValueError("selected event does not match the requested immutable render role")
            protocol.validate_event(event)
            if event["event_id"] != protocol.event_id(event):
                raise ValueError("event ID drift in sealed index")
            context = dict(questions.get(event["event_id"], {}))
            if _actor_has_forbidden_key(context):
                raise ValueError("actor question_context must not carry nested outcome/label/review/evidence material")
            if private_goal_state_review:
                lineage = event.get("phase_lineage")
                if (not isinstance(lineage, Mapping) or lineage.get("private_goal_state_review") is not True or
                        lineage.get("goal_state_question") != GOAL_STATE_REVIEW_QUERY or
                        lineage.get("goal_state_question_sha256") != GOAL_STATE_REVIEW_QUERY_SHA256 or
                        context != {"question": GOAL_STATE_REVIEW_QUERY}):
                    raise ValueError("private student packet requires the exact sealed goal-state question")
            if include_source_annotation_context:
                if context:
                    raise ValueError("use either explicit question context or source annotation context, not both")
                context = _source_annotation_context(event)
            if _actor_has_forbidden_key(context):
                raise ValueError("actor question_context must not carry nested outcome/label/review/evidence material")
            samples = (_student_goal_state_temporal_samples(event) if private_goal_state_review
                       else _temporal_samples(event, temporal_offsets_frames))
            render_request_id = None
            if render_requests:
                requested_samples = _request_temporal_samples(
                    event, render_requests[event["event_id"]], protocol_binding=protocol_binding,
                    expected_usage_role=expected_usage_role,
                    private_goal_state_review=private_goal_state_review)
                if _temporal_identity(samples) != _temporal_identity(requested_samples):
                    raise ValueError("static temporal schedule does not exactly match the sealed render request")
                render_request_id = render_requests[event["event_id"]]["request_id"]
            packet_id = _packet_id(index_sha, event, context, decode, samples, render_request_id,
                                   protocol_binding=protocol_binding)
            if packet_id in seen_packet_ids:
                raise ValueError("duplicate packet_id in selected immutable events")
            seen_packet_ids.add(packet_id)
            by_view = _locators_by_view(event)
            actor_images: dict[str, str] = {}
            actor_temporal_rgb: list[dict[str, Any]] = []
            offline_future_rgb: list[dict[str, Any]] = []
            decoded_receipts: list[dict[str, Any]] = []
            review_asset = None
            if decode:
                review_images: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
                try:
                    for sample in samples:
                        images, paths = {}, {}
                        for view in VIEWS:
                            locator = _temporal_locator(by_view[view], sample)
                            cache_key = (event["source"]["source_group_id"],
                                         event["source"]["episode_index"], locator["relative_path"],
                                         view, sample["sample_frame"])
                            cached = native_asset_cache.get(cache_key) if private_goal_state_review else None
                            image = None
                            if cached is not None:
                                decoded = dict(cached["decoded"])
                                # The PNG bytes are frame-identical, but the
                                # receipt's causal anchor belongs to this
                                # packet's observation (420 is reused by the
                                # terminal packet whose anchor is 539).
                                decoded["actor_anchor_timestamp_s"] = (
                                    float(locator["episode_start_timestamp_s"]) +
                                    event["observation"]["frame"] / 30.0)
                                relative_path = cached["relative_path"]
                                png_sha256 = cached["sha256"]
                                png_bytes = cached["bytes"]
                            else:
                                video = _resolve_video(raw_root, locator["relative_path"])
                                episode_start = float(locator["episode_start_timestamp_s"])
                                episode_end = episode_start + (event["source"]["episode_length"] - 1) / 30.0
                                actor_anchor = (episode_start + event["observation"]["frame"] / 30.0
                                                if sample["temporal_role"] == "ACTOR_CAUSAL" else None)
                                image, decoded = _decode_rgb(
                                    av, video, float(locator["requested_timestamp_s"]),
                                    episode_start_timestamp_s=episode_start, episode_end_timestamp_s=episode_end,
                                    actor_anchor_timestamp_s=actor_anchor)
                                native_name = f"{packet_id}_t{sample['sample_index']:02d}_{view}.png"
                                asset_path = staging / "assets" / native_name
                                image.save(asset_path)
                                relative_path = f"assets/{native_name}"
                                png_sha256 = _sha256(asset_path)
                                png_bytes = asset_path.stat().st_size
                                if private_goal_state_review:
                                    native_asset_cache[cache_key] = {
                                        "relative_path": relative_path, "sha256": png_sha256,
                                        "bytes": png_bytes, "decoded": dict(decoded),
                                    }
                            paths[view] = relative_path
                            decoded_receipts.append({**decoded, "view": view, **sample,
                                                     "camera_native_png_sha256": png_sha256,
                                                     "camera_native_png_bytes": png_bytes})
                            asset_receipts.append({
                                "schema_version": protocol.PACKET_SCHEMA_VERSION, "packet_id": packet_id, "view": view,
                                "asset_kind": "camera_native_rgb_png", "temporal_role": sample["temporal_role"],
                                "sample_index": sample["sample_index"], "sample_frame": sample["sample_frame"],
                                "requested_timestamp_s": locator["requested_timestamp_s"],
                                "decoded_timestamp_s": decoded["decoded_timestamp_s"], "pts_error_s": decoded["pts_error_s"],
                                "relative_path": relative_path, "sha256": png_sha256,
                                "bytes": png_bytes,
                            })
                            if image is not None:
                                if private_goal_state_review and not contact_sheets:
                                    image.close()
                                else:
                                    images[view] = image
                        temporal_record = {**sample, "images": paths}
                        if sample["temporal_role"] == "ACTOR_CAUSAL":
                            actor_temporal_rgb.append(temporal_record)
                            if sample["offset_frames"] == 0:
                                actor_images = dict(paths)
                        else:
                            offline_future_rgb.append(temporal_record)
                        if contact_sheets:
                            review_images.append((sample, images))
                        else:
                            for image in images.values():
                                image.close()
                    if contact_sheets:
                        review_name = f"{packet_id}_contact_sheet.png"
                        _review_contact_sheet(review_images, staging / "review_assets" / review_name, Image, ImageDraw)
                        review_asset = f"review_assets/{review_name}"
                        review_path = staging / review_asset
                        asset_receipts.append({
                            "schema_version": protocol.PACKET_SCHEMA_VERSION, "packet_id": packet_id, "view": None,
                            "asset_kind": "review_only_contact_sheet_png", "relative_path": review_asset,
                            "sha256": _sha256(review_path), "bytes": review_path.stat().st_size,
                        })
                finally:
                    for _, images in review_images:
                        for image in images.values():
                            image.close()
            actor_packet = {
                "packet_id": packet_id, "event_id": event["event_id"],
                "observation_frame": event["observation"]["frame"],
                "images": actor_images,
                "causal_temporal_rgb": actor_temporal_rgb,
                "question_context": context,
                "actor_instruction": ACTOR_INSTRUCTION,
            }
            # Actor input has no source outcome, evidence, review, label, or
            # contact-sheet path.  Those remain in the audit section below.
            packets.append({
                "schema_version": protocol.PACKET_SCHEMA_VERSION,
                "packet_id": packet_id,
                "status": "DECODED_PENDING_LABELS" if decode else "LOCATOR_READY_NO_RGB",
                "training_eligible": False,
                "actor_packet": actor_packet,
                "audit": {
                    "event_id": event["event_id"], "source": event["source"],
                    "usage_role": event.get("usage_role"), "video_locators": [by_view[view] for view in VIEWS],
                    "temporal_schedule": [{**sample, "video_locators": [
                        _temporal_locator(by_view[view], sample) for view in VIEWS]} for sample in samples],
                    "render_request_id": render_request_id,
                    "requested_temporal_slot_count": (len(samples) if private_goal_state_review
                                                       else len(temporal_offsets_frames)),
                    "temporal_slot_count": len(samples),
                    "distinct_sample_frame_count": len({sample["sample_frame"] for sample in samples}),
                    "clamped_duplicate_sample_frame_count": (0 if private_goal_state_review
                                                              else len(temporal_offsets_frames) - len(samples)),
                    "decoded_pts_receipts": decoded_receipts, "offline_future_rgb": offline_future_rgb,
                    "review_only_contact_sheet": review_asset,
                    **({"phase_lineage": event["phase_lineage"]} if private_goal_state_review else {}),
                },
            })
        files = {"packets.jsonl": _write_jsonl(staging / "packets.jsonl", packets, protocol_binding=protocol_binding),
                 "rendered_asset_receipts.jsonl": _write_jsonl(
                     staging / "rendered_asset_receipts.jsonl", asset_receipts, protocol_binding=protocol_binding)}
        result = {
            "schema_version": PACKET_INDEX_SCHEMA,
            "status": "CPU_READY_LABELS_PENDING" if decode else "LOCATORS_READY_RENDER_PENDING",
            "training_eligible": False,
            "index_manifest_sha256": index_sha,
            "index_event_file_sha256": receipt["sha256"],
            "files": files, "packets": len(packets), "rendered_asset_receipts": len(asset_receipts),
            "decoded_camera_native_rgb": decode,
            "temporal_sample_offsets_frames": (None if private_goal_state_review else list(temporal_offsets_frames)),
            "temporal_slots": sum(packet["audit"]["requested_temporal_slot_count"] for packet in packets),
            "distinct_temporal_sample_frames": sum(
                packet["audit"]["distinct_sample_frame_count"] for packet in packets),
            "clamped_duplicate_temporal_samples": sum(
                packet["audit"]["clamped_duplicate_sample_frame_count"] for packet in packets),
            "render_requests_sha256": render_requests_sha256,
            "queue_seal_sha256": queue_seal_sha256,
            "review_only_contact_sheets": contact_sheets, "full_video_hashing": False,
            "expected_usage_role": expected_usage_role,
            "private_goal_state_review": private_goal_state_review,
            "renderer_sha256": _sha256(Path(__file__)),
            "protocol_sha256": protocol_binding.sha256,
            "wall_seconds": time.monotonic() - started,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(protocol.canonical_json(result).encode() + b"\n")
        packet_manifest_sha256 = _sha256(manifest_path)
        os.rename(staging, output)
        # The manifest deliberately does not self-hash.  Resume callers must
        # retain this externally supplied value; recomputing a replacement is
        # not a valid authority action.
        return {**result, "packet_manifest_sha256": packet_manifest_sha256}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _actor_has_forbidden_key(value: Any) -> bool:
    forbidden = ("answer", "label", "outcome", "ground_truth", "success", "failure", "review", "evidence",
                 "audit", "privileged")
    if isinstance(value, Mapping):
        return any(any(token in str(key).lower() for token in forbidden) or _actor_has_forbidden_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_actor_has_forbidden_key(item) for item in value)
    return False


def _safe_asset_path(value: Any, *, directory: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    relative = Path(value)
    return not relative.is_absolute() and ".." not in relative.parts and relative.parts[:1] == (directory,)


def _validate_packet_semantics(packet: Mapping[str, Any], *, decoded: bool,
                               expected_offsets: tuple[int, ...] | None,
                               expected_usage_role: str = ROLE_ANNOTATION_CALIBRATION,
                               private_goal_state_review: bool = False,
                               protocol_binding: ProtocolBinding) -> None:
    protocol = protocol_binding.module
    required = {"schema_version", "packet_id", "status", "training_eligible", "actor_packet", "audit"}
    if set(packet) != required or packet["schema_version"] != protocol.PACKET_SCHEMA_VERSION:
        raise ValueError("packet row does not use the exact P107 packet schema")
    if packet["training_eligible"] is not False or packet["status"] != (
            "DECODED_PENDING_LABELS" if decoded else "LOCATOR_READY_NO_RGB"):
        raise ValueError("packet rows are candidate-only and cannot self-promote training eligibility")
    actor = packet["actor_packet"]
    actor_required = {"packet_id", "event_id", "observation_frame", "images", "causal_temporal_rgb",
                      "question_context", "actor_instruction"}
    if not isinstance(actor, Mapping) or set(actor) != actor_required or actor["packet_id"] != packet["packet_id"]:
        raise ValueError("packet actor input has an invalid schema")
    if not isinstance(actor["question_context"], Mapping) or actor["actor_instruction"] != ACTOR_INSTRUCTION:
        raise ValueError("packet actor question/instruction schema is invalid")
    if expected_usage_role not in {ROLE_ANNOTATION_CALIBRATION, "evaluation_only", ROLE_STUDENT_CANDIDATE}:
        raise ValueError("packet role is not an approved P107 role")
    if private_goal_state_review != (expected_usage_role == ROLE_STUDENT_CANDIDATE):
        raise ValueError("student_candidate packets require the explicit private review opt-in")
    if private_goal_state_review and actor["question_context"] != {"question": GOAL_STATE_REVIEW_QUERY}:
        raise ValueError("private student actor input must retain the exact pre-frozen question")
    if _actor_has_forbidden_key(actor):
        raise ValueError("actor input contains forbidden outcome/label/review/evidence material")
    if type(actor["observation_frame"]) is not int or actor["observation_frame"] < 0:
        raise ValueError("actor packet observation clock is invalid")
    if not isinstance(actor["images"], Mapping) or set(actor["images"]) - set(VIEWS):
        raise ValueError("actor packet image paths are invalid")
    if not isinstance(actor["causal_temporal_rgb"], list):
        raise ValueError("actor temporal RGB must be a list")
    causal_paths = set()
    causal_samples: dict[int, Mapping[str, Any]] = {}
    for sample in actor["causal_temporal_rgb"]:
        fields = {"sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role", "images"}
        if not isinstance(sample, Mapping) or set(sample) != fields or sample["temporal_role"] != "ACTOR_CAUSAL":
            raise ValueError("actor temporal RGB sample has an invalid causal schema")
        if (type(sample["sample_index"]) is not int or sample["sample_index"] in causal_samples or
                type(sample["sample_frame"]) is not int or sample["sample_frame"] > actor["observation_frame"]):
            raise ValueError("actor temporal RGB contains a future frame")
        if not isinstance(sample["images"], Mapping) or set(sample["images"]) != set(VIEWS):
            raise ValueError("actor temporal RGB must provide all three camera-native views")
        if any(not _safe_asset_path(path, directory="assets") for path in sample["images"].values()):
            raise ValueError("actor temporal RGB must name only camera-native asset paths")
        causal_samples[sample["sample_index"]] = sample
        causal_paths.update(sample["images"].values())
    if decoded and (set(actor["images"]) != set(VIEWS) or not set(actor["images"].values()) <= causal_paths):
        raise ValueError("decoded actor anchor images must be causal temporal camera-native assets")
    if decoded and any(not _safe_asset_path(path, directory="assets") for path in actor["images"].values()):
        raise ValueError("actor anchor images must name only camera-native asset paths")
    if not decoded and (actor["images"] or actor["causal_temporal_rgb"]):
        raise ValueError("locator-only packet cannot claim decoded actor RGB")
    audit = packet["audit"]
    audit_required = {"event_id", "source", "usage_role", "video_locators", "temporal_schedule",
                      "render_request_id", "requested_temporal_slot_count", "temporal_slot_count",
                      "distinct_sample_frame_count", "clamped_duplicate_sample_frame_count", "decoded_pts_receipts",
                      "offline_future_rgb", "review_only_contact_sheet"}
    if private_goal_state_review:
        audit_required.add("phase_lineage")
    if not isinstance(audit, Mapping) or set(audit) != audit_required or audit["event_id"] != actor["event_id"]:
        raise ValueError("packet audit has an invalid schema")
    protocol.validate_source_ref(audit["source"])
    if not isinstance(audit["video_locators"], list) or len(audit["video_locators"]) != 3:
        raise ValueError("packet audit must retain exactly three source camera locators")
    by_view = {locator.get("view"): locator for locator in audit["video_locators"] if isinstance(locator, Mapping)}
    if len(by_view) != 3 or set(by_view) != set(VIEWS):
        raise ValueError("packet audit camera locators are malformed")
    for locator in by_view.values():
        expected_time = locator.get("episode_start_timestamp_s") + actor["observation_frame"] / 30.0
        if (locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED" or locator.get("expected_fps") != 30 or
                type(locator.get("requested_timestamp_s")) not in (int, float) or
                abs(locator["requested_timestamp_s"] - expected_time) > 1e-9):
            raise ValueError("packet audit base locator is not a source-clock camera locator")
    schedule = audit["temporal_schedule"]
    expected_schedule = (_student_goal_state_temporal_samples({
        "observation": {"frame": actor["observation_frame"]},
        "source": audit["source"],
        "phase_lineage": audit["phase_lineage"],
    }) if private_goal_state_review else _temporal_samples(
        {"observation": {"frame": actor["observation_frame"]}, "source": audit["source"]},
        expected_offsets))
    if not isinstance(schedule, list) or len(schedule) != len(expected_schedule):
        raise ValueError("packet audit must retain the exact bounded temporal samples")
    request_id = audit["render_request_id"]
    if request_id is not None and (not isinstance(request_id, str) or len(request_id) != 64 or
                                   any(char not in "0123456789abcdef" for char in request_id)):
        raise ValueError("packet temporal request identity is invalid")
    expected_requested_count = len(expected_schedule) if private_goal_state_review else len(expected_offsets)
    if (audit.get("usage_role") != expected_usage_role or
            audit["requested_temporal_slot_count"] != expected_requested_count or
            audit["temporal_slot_count"] != len(expected_schedule) or
            type(audit["distinct_sample_frame_count"]) is not int or
            audit["distinct_sample_frame_count"] != len(expected_schedule) or
            audit["clamped_duplicate_sample_frame_count"] != (0 if private_goal_state_review else
                                                               len(expected_offsets) - len(expected_schedule))):
        raise ValueError("packet temporal slot/duplicate accounting is invalid")
    expected_sample_by_index: dict[int, Mapping[str, Any]] = {}
    for position, sample in enumerate(schedule):
        fields = {"sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role", "video_locators"}
        if not isinstance(sample, Mapping) or set(sample) != fields or sample["sample_index"] != position:
            raise ValueError("packet temporal schedule is malformed")
        expected_sample = expected_schedule[position]
        if (_temporal_identity((sample,)) != _temporal_identity((expected_sample,)) or
                not isinstance(sample["video_locators"], list) or len(sample["video_locators"]) != 3):
            raise ValueError("packet temporal schedule crosses its causal boundary")
        locators = {locator.get("view"): locator for locator in sample["video_locators"] if isinstance(locator, Mapping)}
        if len(locators) != 3 or set(locators) != set(VIEWS):
            raise ValueError("packet temporal schedule must preserve all three camera views")
        bare_sample = {key: sample[key] for key in ("sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role")}
        if any(dict(locators[view]) != _temporal_locator(by_view[view], bare_sample) for view in VIEWS):
            raise ValueError("packet temporal camera locator does not bind the source clock")
        expected_sample_by_index[position] = sample
    if audit["distinct_sample_frame_count"] != len({sample["sample_frame"] for sample in schedule}):
        raise ValueError("packet temporal duplicate accounting does not match the clipped schedule")
    expected_causal_indices = {index for index, sample in expected_sample_by_index.items()
                                if sample["temporal_role"] == "ACTOR_CAUSAL"}
    if decoded and set(causal_samples) != expected_causal_indices:
        raise ValueError("actor temporal RGB does not exactly match causal packet samples")
    anchor_index = next(index for index, sample in expected_sample_by_index.items() if sample["offset_frames"] == 0)
    if decoded and dict(actor["images"]) != dict(causal_samples[anchor_index]["images"]):
        raise ValueError("actor anchor images must bind the zero-offset camera sample")
    offline_samples: dict[int, Mapping[str, Any]] = {}
    if not isinstance(audit["offline_future_rgb"], list):
        raise ValueError("offline audit RGB must be a list")
    for sample in audit["offline_future_rgb"]:
        if (not isinstance(sample, Mapping) or sample.get("temporal_role") != "OFFLINE_FUTURE_AUDIT" or
                sample.get("sample_frame", -1) <= actor["observation_frame"] or
                sample.get("sample_index") in offline_samples):
            raise ValueError("offline audit RGB must be strictly future of the actor observation")
        if set(sample) != {"sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role", "images"}:
            raise ValueError("offline audit RGB has an invalid temporal schema")
        if not isinstance(sample["images"], Mapping) or set(sample["images"]) != set(VIEWS):
            raise ValueError("offline audit RGB must preserve all camera-native views")
        if any(not _safe_asset_path(path, directory="assets") for path in sample["images"].values()):
            raise ValueError("offline audit RGB must name only camera-native asset paths")
        offline_samples[sample["sample_index"]] = sample
        if causal_paths & set(sample.get("images", {}).values()):
            raise ValueError("future offline audit RGB leaked into actor assets")
    expected_offline_indices = {index for index, sample in expected_sample_by_index.items()
                                if sample["temporal_role"] == "OFFLINE_FUTURE_AUDIT"}
    if decoded and set(offline_samples) != expected_offline_indices:
        raise ValueError("offline audit RGB does not exactly match future packet samples")
    if not decoded and (causal_samples or offline_samples or audit["review_only_contact_sheet"] is not None):
        raise ValueError("locator-only packet cannot contain decoded temporal or review image paths")
    decoded_fields = {"resolved_path", "bytes", "mtime_ns", "requested_timestamp_s", "decoded_timestamp_s",
                      "pts_error_s", "fps", "resolution", "episode_global_pts_bounds_s",
                      "actor_anchor_timestamp_s", "full_video_sha256", "view", "sample_index", "offset_frames",
                      "sample_frame", "timestamp_s", "temporal_role", "camera_native_png_sha256",
                      "camera_native_png_bytes"}
    receipts = audit["decoded_pts_receipts"]
    if not isinstance(receipts, list) or (decoded and len(receipts) != len(schedule) * len(VIEWS)) or (not decoded and receipts):
        raise ValueError("packet decoded PTS receipt count is inconsistent with its camera render mode")
    seen_receipts = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or set(receipt) != decoded_fields:
            raise ValueError("decoded PTS receipt does not use the exact bounded camera schema")
        key = (receipt["sample_index"], receipt["view"])
        sample = expected_sample_by_index.get(receipt["sample_index"])
        locator = (by_view.get(receipt["view"]) if receipt["view"] in VIEWS else None)
        if key in seen_receipts or sample is None or locator is None or any(
                receipt[field] != sample[field] for field in ("sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role")):
            raise ValueError("decoded PTS receipt does not bind a unique scheduled camera sample")
        seen_receipts.add(key)
        expected_requested = locator["episode_start_timestamp_s"] + sample["timestamp_s"]
        start = locator["episode_start_timestamp_s"]
        end = start + (audit["source"]["episode_length"] - 1) / 30.0
        decoded_time, requested, error = receipt["decoded_timestamp_s"], receipt["requested_timestamp_s"], receipt["pts_error_s"]
        if (type(requested) not in (int, float) or type(decoded_time) not in (int, float) or
                type(error) not in (int, float) or abs(requested - expected_requested) > 1e-9 or
                abs(error - abs(decoded_time - requested)) > 1e-9 or error > 1.0 / 60.0 + 1e-6 or
                not _clock_in_episode(decoded_time, start, end) or receipt["episode_global_pts_bounds_s"] != [start, end] or
                receipt["fps"] != 30 or receipt["full_video_sha256"] is not None or
                not isinstance(receipt["resolved_path"], str) or type(receipt["bytes"]) is not int or
                type(receipt["mtime_ns"]) is not int or not isinstance(receipt["resolution"], list) or
                len(receipt["resolution"]) != 2 or any(type(value) is not int or value < 1 for value in receipt["resolution"]) or
                not isinstance(receipt["camera_native_png_sha256"], str) or type(receipt["camera_native_png_bytes"]) is not int):
            raise ValueError("decoded PTS receipt has invalid source-bound timing or image metadata")
        expected_anchor = start + actor["observation_frame"] / 30.0 if sample["temporal_role"] == "ACTOR_CAUSAL" else None
        if (receipt["actor_anchor_timestamp_s"] != expected_anchor or
                (expected_anchor is not None and not _clock_at_or_before(decoded_time, expected_anchor))):
            raise ValueError("decoded PTS receipt crosses the actor causal observation clock")
    if decoded and seen_receipts != {(sample_index, view) for sample_index in expected_sample_by_index for view in VIEWS}:
        raise ValueError("decoded PTS receipts do not cover the complete temporal camera panel")
    if audit["review_only_contact_sheet"] is not None and not _safe_asset_path(
            audit["review_only_contact_sheet"], directory="review_assets"):
        raise ValueError("review-only contact sheet must remain outside actor camera assets")


def _validate_packet_index_binding(packet: Mapping[str, Any], event: Mapping[str, Any], *,
                                   index_manifest_sha256: str, decoded: bool,
                                   temporal_offsets: tuple[int, ...] | None,
                                   expected_usage_role: str = ROLE_ANNOTATION_CALIBRATION,
                                   private_goal_state_review: bool = False,
                                   protocol_binding: ProtocolBinding) -> None:
    """A resigned packet cannot swap its source event, camera locators, or ID."""
    protocol = protocol_binding.module
    protocol.validate_event(event)
    actor, audit = packet["actor_packet"], packet["audit"]
    if (actor["event_id"] != event["event_id"] or actor["observation_frame"] != event["observation"]["frame"] or
            protocol.canonical_json(audit["source"]) != protocol.canonical_json(event["source"]) or
            audit["usage_role"] != event.get("usage_role") or
            protocol.canonical_json(audit["video_locators"]) != protocol.canonical_json(event["video_locators"])):
        raise ValueError("packet does not bind the exact canonical event/source/camera record from the sealed index")
    if event.get("usage_role") != expected_usage_role:
        raise ValueError("packet event role does not match the sealed packet role")
    if private_goal_state_review:
        lineage = event.get("phase_lineage")
        if (not isinstance(lineage, Mapping) or lineage.get("private_goal_state_review") is not True or
                lineage.get("goal_state_question") != GOAL_STATE_REVIEW_QUERY or
                lineage.get("goal_state_question_sha256") != GOAL_STATE_REVIEW_QUERY_SHA256 or
                protocol.canonical_json(audit.get("phase_lineage")) != protocol.canonical_json(lineage)):
            raise ValueError("private packet does not bind the sealed goal-state phase lineage")
    expected_packet_id = _packet_id(index_manifest_sha256, event, actor["question_context"], decoded,
                                    packet["audit"]["temporal_schedule"], packet["audit"]["render_request_id"],
                                    protocol_binding=protocol_binding)
    if packet["packet_id"] != expected_packet_id:
        raise ValueError("packet_id does not match its canonical sealed event/context/temporal identity")


def _validate_packet_assets(packet: Mapping[str, Any], assets: list[Mapping[str, Any]], *, decoded: bool,
                            review_sheets: bool) -> None:
    """Bind every native PNG receipt to its packet clock; review composites stay audit-only."""
    actor, audit = packet["actor_packet"], packet["audit"]
    camera = [asset for asset in assets if asset["asset_kind"] == "camera_native_rgb_png"]
    contacts = [asset for asset in assets if asset["asset_kind"] == "review_only_contact_sheet_png"]
    if not decoded:
        if assets:
            raise ValueError("locator-only packet cannot contain rendered RGB asset receipts")
        return
    if len(camera) != len(audit["temporal_schedule"]) * len(VIEWS):
        raise ValueError("decoded packet must seal every scheduled timestamp by three native cameras")
    if len(contacts) != (1 if review_sheets else 0):
        raise ValueError("decoded packet has an invalid review-only contact-sheet receipt count")
    schedule = {sample["sample_index"]: sample for sample in audit["temporal_schedule"]}
    decoded_receipts = {(receipt["sample_index"], receipt["view"]): receipt
                        for receipt in audit["decoded_pts_receipts"]}
    paths: dict[int, dict[str, str]] = {}
    for asset in camera:
        sample = schedule.get(asset["sample_index"])
        scheduled_locators = ({locator.get("view"): locator for locator in sample["video_locators"]}
                              if sample is not None else {})
        decoded_receipt = decoded_receipts.get((asset["sample_index"], asset["view"]))
        if (sample is None or asset["view"] not in VIEWS or asset["sample_frame"] != sample["sample_frame"] or
                asset["temporal_role"] != sample["temporal_role"] or
                asset["requested_timestamp_s"] != scheduled_locators[asset["view"]]["requested_timestamp_s"] or
                decoded_receipt is None or asset["requested_timestamp_s"] != decoded_receipt["requested_timestamp_s"] or
                asset["decoded_timestamp_s"] != decoded_receipt["decoded_timestamp_s"] or
                asset["pts_error_s"] != decoded_receipt["pts_error_s"] or
                asset["sha256"] != decoded_receipt["camera_native_png_sha256"] or
                asset["bytes"] != decoded_receipt["camera_native_png_bytes"]):
            raise ValueError("camera-native receipt does not bind its scheduled source timestamp")
        paths.setdefault(asset["sample_index"], {})[asset["view"]] = asset["relative_path"]
    if set(paths) != set(schedule) or any(set(views) != set(VIEWS) for views in paths.values()):
        raise ValueError("decoded packet does not seal a complete native three-camera temporal panel")
    causal = {sample["sample_index"]: sample for sample in actor["causal_temporal_rgb"]}
    if any(dict(sample["images"]) != paths[index] for index, sample in causal.items()):
        raise ValueError("actor causal image paths do not match sealed native camera receipts")
    offline = {sample["sample_index"]: sample for sample in audit["offline_future_rgb"]}
    if any(dict(sample["images"]) != paths[index] for index, sample in offline.items()):
        raise ValueError("offline audit image paths do not match sealed native camera receipts")
    if contacts:
        if contacts[0]["relative_path"] != audit["review_only_contact_sheet"]:
            raise ValueError("review contact-sheet receipt must remain audit-only")
    elif audit["review_only_contact_sheet"] is not None:
        raise ValueError("packet references a review contact sheet without its sealed receipt")


def resume_packets(index: Path, output: Path, *, expected_packet_manifest_sha256: str | None,
                   protocol_binding: ProtocolBinding | None = None) -> dict[str, Any]:
    index, output = Path(index), Path(output)
    protocol_binding = _protocol_binding if protocol_binding is None else protocol_binding
    if (not isinstance(expected_packet_manifest_sha256, str) or len(expected_packet_manifest_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in expected_packet_manifest_sha256)):
        raise ValueError("--resume requires an externally pinned --expected-packet-manifest-sha256")
    manifest_path = output / "manifest.json"
    if _sha256(manifest_path) != expected_packet_manifest_sha256:
        raise ValueError("packet manifest does not match the externally pinned authority SHA-256")
    result = _read_json(manifest_path)
    if not isinstance(result, Mapping) or set(result) != PACKET_MANIFEST_FIELDS or result.get("schema_version") != PACKET_INDEX_SCHEMA:
        raise ValueError("--resume requires a sealed P107 packet directory")
    if result.get("training_eligible") is not False or result.get("status") not in {
            "CPU_READY_LABELS_PENDING", "LOCATORS_READY_RENDER_PENDING"}:
        raise ValueError("packet manifest cannot self-promote candidate packets to training/audited status")
    if result.get("protocol_sha256") != protocol_binding.sha256:
        raise ValueError("packet manifest protocol does not match the pinned compatibility reader")
    expected_usage_role = result.get("expected_usage_role")
    private_goal_state_review = result.get("private_goal_state_review")
    if (expected_usage_role not in {ROLE_ANNOTATION_CALIBRATION, "evaluation_only", ROLE_STUDENT_CANDIDATE} or
            type(private_goal_state_review) is not bool or
            private_goal_state_review != (expected_usage_role == ROLE_STUDENT_CANDIDATE)):
        raise ValueError("packet manifest has an invalid immutable usage-role/private-review binding")
    offsets = result.get("temporal_sample_offsets_frames")
    if private_goal_state_review:
        if offsets is not None:
            raise ValueError("private student packet manifest must not claim a static calibration offset schedule")
        temporal_offsets = None
    elif not _valid_temporal_offsets(offsets):
        raise ValueError("packet manifest has an invalid bounded temporal sample schedule")
    else:
        temporal_offsets = tuple(offsets)
    if type(result.get("review_only_contact_sheets")) is not bool or result.get("full_video_hashing") is not False:
        raise ValueError("packet manifest has invalid review/video-hashing policy fields")
    request_sha, queue_seal_sha = result.get("render_requests_sha256"), result.get("queue_seal_sha256")
    if ((request_sha is None) != (queue_seal_sha is None) or
            any(value is not None and (not isinstance(value, str) or len(value) != 64 or
                                       any(char not in "0123456789abcdef" for char in value))
                for value in (request_sha, queue_seal_sha))):
        raise ValueError("packet manifest has invalid sealed temporal queue provenance")
    if result.get("index_manifest_sha256") != _sha256(index / "manifest.json"):
        raise ValueError("source index manifest changed; refusing resume")
    index_manifest = _read_json(index / "manifest.json")
    if index_manifest.get("protocol_sha256") != protocol_binding.sha256:
        raise ValueError("sealed event index protocol does not match the pinned compatibility reader")
    index_receipt = index_manifest.get("files", {}).get("event_candidates.jsonl", {}) if isinstance(index_manifest, Mapping) else {}
    if result.get("index_event_file_sha256") != index_receipt.get("sha256"):
        raise ValueError("packet manifest does not bind the sealed event-candidate payload")
    receipt = result.get("files", {}).get("packets.jsonl", {})
    if set(result.get("files", {})) != PACKET_PAYLOAD_FILES:
        raise ValueError("packet manifest must seal the exact packet and rendered-asset receipt inventory")
    packets = _read_jsonl(output / "packets.jsonl")
    if _sha256(output / "packets.jsonl") != receipt.get("sha256") or len(packets) != receipt.get("rows"):
        raise ValueError("sealed packet payload changed")
    if (result.get("packets") != len(packets) or not isinstance(result.get("decoded_camera_native_rgb"), bool) or
            type(result.get("temporal_slots")) is not int or type(result.get("distinct_temporal_sample_frames")) is not int or
            type(result.get("clamped_duplicate_temporal_samples")) is not int or
            result["temporal_slots"] != sum(packet.get("audit", {}).get("requested_temporal_slot_count", -1)
                                              for packet in packets) or
            result["clamped_duplicate_temporal_samples"] != result["temporal_slots"] - result["distinct_temporal_sample_frames"]):
        raise ValueError("packet manifest counters or decode mode are invalid")
    decoded = result["decoded_camera_native_rgb"]
    packet_ids, event_ids = set(), set()
    for packet in packets:
        _validate_packet_semantics(packet, decoded=decoded, expected_offsets=temporal_offsets,
                                   expected_usage_role=expected_usage_role,
                                   private_goal_state_review=private_goal_state_review,
                                   protocol_binding=protocol_binding)
        if (request_sha is None) != (packet["audit"]["render_request_id"] is None):
            raise ValueError("packet temporal request identity does not match its sealed manifest provenance")
        if packet["packet_id"] in packet_ids or packet["actor_packet"]["event_id"] in event_ids:
            raise ValueError("packet receipt contains duplicate packet/event identities")
        packet_ids.add(packet["packet_id"])
        event_ids.add(packet["actor_packet"]["event_id"])
    selected_events = _select_index_events(index / "event_candidates.jsonl", index_receipt, event_ids, limit=None)
    events_by_id = {event["event_id"]: event for event in selected_events}
    if len(events_by_id) != len(packets):
        raise ValueError("sealed event index has duplicate/missing packet source identities")
    for packet in packets:
        _validate_packet_index_binding(packet, events_by_id[packet["actor_packet"]["event_id"]],
                                       index_manifest_sha256=result["index_manifest_sha256"], decoded=decoded,
                                       temporal_offsets=temporal_offsets,
                                       expected_usage_role=expected_usage_role,
                                       private_goal_state_review=private_goal_state_review,
                                       protocol_binding=protocol_binding)
    if (result["temporal_slots"] != sum(packet["audit"]["requested_temporal_slot_count"] for packet in packets) or
            result["distinct_temporal_sample_frames"] != sum(packet["audit"]["distinct_sample_frame_count"] for packet in packets) or
            result["clamped_duplicate_temporal_samples"] != sum(
                packet["audit"]["clamped_duplicate_sample_frame_count"] for packet in packets)):
        raise ValueError("packet manifest temporal-slot accounting does not match its packet rows")
    asset_receipt = result["files"]["rendered_asset_receipts.jsonl"]
    assets = _read_jsonl(output / "rendered_asset_receipts.jsonl")
    if (_sha256(output / "rendered_asset_receipts.jsonl") != asset_receipt.get("sha256") or
            len(assets) != asset_receipt.get("rows")):
        raise ValueError("sealed rendered-asset receipt payload changed")
    if result.get("rendered_asset_receipts") != len(assets):
        raise ValueError("packet manifest rendered-asset count is invalid")
    packet_ids = {packet["packet_id"] for packet in packets}
    assets_by_packet: dict[str, list[Mapping[str, Any]]] = {packet_id_value: [] for packet_id_value in packet_ids}
    seen_asset_keys = set()
    for row in assets:
        camera_fields = {"schema_version", "packet_id", "view", "asset_kind", "temporal_role", "sample_index",
                         "sample_frame", "requested_timestamp_s", "decoded_timestamp_s", "pts_error_s",
                         "relative_path", "sha256", "bytes"}
        contact_fields = {"schema_version", "packet_id", "view", "asset_kind", "relative_path", "sha256", "bytes"}
        if set(row) not in (camera_fields, contact_fields):
            raise ValueError("rendered asset receipt has an unsealed field")
        if row.get("asset_kind") == "camera_native_rgb_png":
            if set(row) != camera_fields or row["view"] not in VIEWS or row["temporal_role"] not in {
                    "ACTOR_CAUSAL", "OFFLINE_FUTURE_AUDIT"} or type(row["sample_index"]) is not int or \
                    type(row["sample_frame"]) is not int or type(row["pts_error_s"]) not in (int, float) or \
                    row["pts_error_s"] < 0 or row["pts_error_s"] > 1.0 / 60.0 + 1e-6:
                raise ValueError("camera-native asset receipt has invalid temporal/PTS semantics")
            key = (row["packet_id"], row["sample_index"], row["view"])
            if key in seen_asset_keys:
                raise ValueError("duplicate packet temporal camera asset receipt")
            seen_asset_keys.add(key)
        elif row.get("asset_kind") != "review_only_contact_sheet_png" or set(row) != contact_fields:
            raise ValueError("rendered asset receipt has an unsupported artifact kind")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("rendered asset receipt escapes packet root")
        path = (output / relative).resolve(strict=True)
        if output.resolve() not in path.parents:
            raise ValueError("rendered asset receipt escapes packet root")
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise ValueError("sealed rendered PNG bytes changed")
        if row["packet_id"] not in assets_by_packet:
            raise ValueError("rendered asset receipt refers to an absent packet")
        assets_by_packet[row["packet_id"]].append(row)
    for packet in packets:
        _validate_packet_assets(packet, assets_by_packet[packet["packet_id"]], decoded=decoded,
                                review_sheets=result["review_only_contact_sheets"])
    return {"status": "RESUME_VALIDATED", "manifest": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path,
                        help="exact legacy protocol source for a sealed historical index; requires its external SHA-256")
    parser.add_argument("--expected-protocol-sha256",
                        help="external SHA-256 for --protocol-path; default DATA protocol is pinned in this renderer")
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--event-ids", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--questions", type=Path, help="JSONL actor question contexts, never labels")
    parser.add_argument("--include-source-annotation-context", action="store_true")
    parser.add_argument("--expected-usage-role", choices=(ROLE_ANNOTATION_CALIBRATION, "evaluation_only",
                                                            ROLE_STUDENT_CANDIDATE),
                        default=ROLE_ANNOTATION_CALIBRATION)
    parser.add_argument("--private-goal-state-review", action="store_true",
                        help="explicit private student_candidate causal goal-state packet mode")
    parser.add_argument("--decode", action="store_true", help="decode bounded native RGB and PTS receipts")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--temporal-offset-frames", type=_parse_temporal_offsets,
                        default=TEMPORAL_SAMPLE_OFFSETS,
                        help="exact seven or ten source-frame offsets incl. 0, max +/-240; future samples are audit-only")
    parser.add_argument("--render-requests", type=Path,
                        help="sealed per-event temporal request JSONL; only accepted with its external SHA and queue seal")
    parser.add_argument("--expected-render-requests-sha256",
                        help="external SHA-256 for --render-requests")
    parser.add_argument("--queue-seal", type=Path,
                        help="sealed queue JSON that binds --render-requests")
    parser.add_argument("--expected-queue-seal-sha256",
                        help="external SHA-256 for --queue-seal")
    parser.add_argument("--contact-sheets", action="store_true", help="review-only composites; excluded from actor input")
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-packet-manifest-sha256",
                        help="external SHA-256 returned at packet creation; mandatory with --resume")
    args = parser.parse_args()
    if args.protocol_path is None and args.expected_protocol_sha256 is not None:
        parser.error("--expected-protocol-sha256 is only valid with --protocol-path")
    try:
        protocol_binding = _load_protocol(args.protocol_path, expected_sha256=args.expected_protocol_sha256)
    except ValueError as error:
        parser.error(str(error))
    if args.resume and args.expected_packet_manifest_sha256 is None:
        parser.error("--resume requires --expected-packet-manifest-sha256 from an external authority record")
    request_flags = (args.render_requests, args.expected_render_requests_sha256,
                     args.queue_seal, args.expected_queue_seal_sha256)
    if args.private_goal_state_review != (args.expected_usage_role == ROLE_STUDENT_CANDIDATE):
        parser.error("student_candidate requires --private-goal-state-review and no other role may use it")
    if args.resume and any(value is not None for value in request_flags):
        parser.error("--resume validates the sealed packet manifest; do not provide creation-time render-request flags")
    if not args.resume and any(value is not None for value in request_flags) and any(value is None for value in request_flags):
        parser.error("--render-requests requires its expected SHA plus --queue-seal and its expected SHA")
    if args.resume:
        result = resume_packets(args.index, args.output,
                                expected_packet_manifest_sha256=args.expected_packet_manifest_sha256,
                                protocol_binding=protocol_binding)
    else:
        requests, request_sha = _read_render_requests(
            args.render_requests, args.expected_render_requests_sha256,
            expected_usage_role=args.expected_usage_role,
            private_goal_state_review=args.private_goal_state_review)
        queue_seal_sha, queue_source_manifest_sha = _read_queue_seal(
            args.queue_seal, args.expected_queue_seal_sha256, render_requests_path=args.render_requests,
            render_requests_sha256=request_sha)
        result = create_packets(
            args.index, args.output, event_ids=_read_event_ids(args.event_ids, args.event_id), limit=args.limit,
            questions=_questions(args.questions), include_source_annotation_context=args.include_source_annotation_context,
            decode=args.decode, raw_root=args.raw_root, contact_sheets=args.contact_sheets,
            temporal_offsets_frames=args.temporal_offset_frames, render_requests=requests,
            render_requests_sha256=request_sha, queue_seal_sha256=queue_seal_sha,
            queue_seal_source_release_manifest_sha256=queue_source_manifest_sha,
            expected_usage_role=args.expected_usage_role,
            private_goal_state_review=args.private_goal_state_review,
            protocol_binding=protocol_binding,
            max_seconds=args.max_seconds)
    print(protocol_binding.module.canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
