"""Select deterministic P107 annotation *candidates* from a sealed metadata index.

This is deliberately upstream of rendering and labeling.  It consumes the
immutable event/source-group identifiers produced by the P107 index, emits
camera-native temporal render requests, and never opens RGB, video, actions,
or simulator state.  Its heuristics find review candidates only: an annotation
boundary, a repeated source skill/object, or a long annotated interval is not
an outcome, failure, recovery, or trainable action label.  Action-sequence
heuristics are disabled here until an owner-sealed actual-execution contract
exists.

The output queues are external, content-addressed sidecars.  They keep the
frozen source split and usage role intact, explicitly exclude eval/public-test
groups, and make causal actor-visible frames distinct from reviewer-only
before/after frames.  A renderer must handshake on the temporal request schema
before it resolves or decodes any RGB.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import types
from typing import Any, Iterable, Mapping, Sequence


INDEX_SCHEMA = "memlite-event-index-v1"
INDEX_STATUS = "METADATA_ONLY_READY_FOR_PACKET_RENDER"
QUEUE_SCHEMA = "p107-metadata-annotation-queue-v1"
RENDER_REQUEST_SCHEMA = "p107-camera-native-temporal-request-v1"
COUNTS_SCHEMA = "p107-metadata-annotation-counts-v1"
MANIFEST_SCHEMA = "p107-metadata-annotation-queue-manifest-v1"
QUEUE_SEAL_SCHEMA = "p107-metadata-annotation-queue-seal-v1"
STATUS = "CANDIDATE_MISSING_EVIDENCE"
DEFAULT_SOURCE_MANIFEST_SHA256 = "90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23"
DEFAULT_SEED = "p107-metadata-candidate-selection-20261001"
FRAME_RATE_HZ = 30
ABSOLUTE_MAX_RETAINED_CANDIDATES = 50000
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = REPOSITORY_ROOT / "src" / "g05" / "data" / "memlite_event_protocol.py"
QUEUE_PAYLOAD_FILES = frozenset((
    "annotation_calibration_queue.jsonl",
    "camera_native_render_requests.jsonl",
    "counts.json",
    "student_candidate_queue.jsonl",
))
QUEUE_JOB_FIELDS = frozenset((
    "schema_version", "job_id", "queue_kind", "status", "training_eligible", "source_kind",
    "candidate_heuristics_are_not_truth", "event_id", "source_group_id", "immutable_split", "usage_role",
    "source_identity", "event_interval", "candidate_signals", "selection_stratum", "selection_phase",
    "repeat_same_skill_target_distinct_episode_count", "annotation_boundary", "action_heuristic_status",
    "long_annotated_interval_candidate", "skill_ids", "temporal_windows", "current_actor_evidence",
    "review_constraints", "camera_native_render_request_id",
))
RENDER_REQUEST_FIELDS = frozenset((
    "schema_version", "request_id", "status", "event_id", "source_identity", "requested_frame_indices",
    "actor_available_frame_indices", "offline_review_before_frame_indices", "offline_review_after_frame_indices",
    "anchor_camera_locators", "frame_locator_resolution", "camera_delivery",
))
COUNTS_FIELDS = frozenset((
    "schema_version", "status", "training_eligible", "source_role_inventory", "events_excluded_by_role",
    "events_excluded_by_unrecognized_official_vocabulary", "student_candidate_queue",
    "annotation_calibration_queue", "existing_local_single_frame_pilot",
))
MANIFEST_FIELDS = frozenset((
    "schema_version", "status", "training_eligible", "all_jobs_status", "source_release_manifest_sha256",
    "index_manifest_sha256", "inventory_seal_sha256", "canonical_protocol_sha256",
    "coverage_expectations_sha256", "index_event_file_sha256", "index_source_group_file_sha256", "policy",
    "files", "renderer_handshake", "design_handoff",
))
POLICY_CORE_FIELDS = frozenset((
    "schema_version", "algorithm", "seed", "frozen_source_release_manifest_sha256", "sealed_index_manifest_sha256",
    "inventory_seal_sha256", "canonical_protocol_sha256", "coverage_expectations_sha256", "candidate_budget",
    "calibration_budget", "max_per_episode", "max_per_source_group", "min_separation_frames",
    "boundary_gap_frames", "long_interval_frames", "normal_control_fraction", "actor_history_frames",
    "review_before_frames", "review_after_frames", "review_sample_stride_frames", "no_rgb_decode_by_selector",
    "action_sequence_heuristic", "supported_source_kinds", "all_jobs_status", "max_retained_candidates",
))
PRIVATE_PHASE_DERIVATION_SCHEMA = "p107-phase-balanced-mini-index-v1"
PRIVATE_GOAL_STATE_REVIEW_QUERY = "At the anchor, is the brown animal toy visibly held by the robot’s right gripper?"
PRIVATE_GOAL_STATE_REVIEW_QUERY_SHA256 = hashlib.sha256(
    PRIVATE_GOAL_STATE_REVIEW_QUERY.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """The only JSON encoding used for P107 sidecar identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_regular_binary(path: Path, *, name: str):
    """Open one regular-file inode without a pathname re-open race.

    Consumers that bind a digest to bytes must read those same bytes from this
    descriptor. ``lstat``/``fstat`` covers platforms without ``O_NOFOLLOW``;
    on Linux the latter also prevents a symlink swap at open time.
    """
    path = Path(path)
    try:
        entry = path.lstat()
    except OSError as error:
        raise ValueError(f"Required regular input is missing: {name}") from error
    if not stat.S_ISREG(entry.st_mode):
        raise ValueError(f"Required regular input is missing: {name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Required regular input is missing: {name}") from error
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or entry.st_dev != opened.st_dev or
                entry.st_ino != opened.st_ino):
            raise ValueError(f"Required regular input changed while opening: {name}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def read_regular_file_bytes(path: Path, *, name: str) -> bytes:
    """Read the exact regular-file bytes that a hash/consumer will use."""
    with _open_regular_binary(path, name=name) as stream:
        return stream.read()


def _strict_json(data: bytes, *, name: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON number in {name}: {value}")

    return json.loads(data, object_pairs_hook=reject_duplicate, parse_constant=reject_nonfinite)


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular JSON input is missing: {path}")
    return _strict_json(path.read_bytes(), name=str(path))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with _open_regular_binary(path, name=str(path)) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = _strict_json(line, name=f"{path}:{line_number}")
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def forbid_output_inside(output: Path, *sources: Path) -> None:
    resolved_output = output.resolve(strict=False)
    for source in sources:
        resolved_source = source.resolve(strict=True)
        if resolved_output == resolved_source or resolved_source in resolved_output.parents:
            raise ValueError("queue output must be outside immutable index/source directories")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_json(row).encode("utf-8") + b"\n")
            count += 1
    return {"sha256": sha256_file(path), "rows": count, "bytes": path.stat().st_size}


def _expected_receipt(expected: Mapping[str, Any], *, name: str) -> tuple[str, int, int]:
    expected = require_mapping(expected, f"sealed receipt for {name}")
    if set(expected) != {"sha256", "rows", "bytes"}:
        raise ValueError(f"Sealed receipt is malformed: {name}")
    digest, rows, byte_count = expected.get("sha256"), expected.get("rows"), expected.get("bytes")
    if not is_sha256(digest) or type(rows) is not int or rows < 0 or type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"Sealed receipt is malformed: {name}")
    return digest, rows, byte_count


def iter_jsonl_verified(path: Path, *, expected: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Parse and digest one JSONL descriptor, rejecting non-sealed bytes at EOF."""
    expected_digest, _, expected_bytes = _expected_receipt(expected, name=Path(path).name)
    digest = hashlib.sha256()
    byte_count = 0
    with _open_regular_binary(path, name=str(path)) as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            byte_count += len(line)
            if not line.strip():
                continue
            row = _strict_json(line, name=f"{path}:{line_number}")
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield row
    if digest.hexdigest() != expected_digest or byte_count != expected_bytes:
        raise ValueError(f"Sealed index payload hash mismatch: {Path(path).name}")


def read_jsonl_verified(path: Path, *, expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(iter_jsonl_verified(path, expected=expected))


def file_receipt(path: Path, *, expected: Mapping[str, Any]) -> None:
    expected_digest, _, expected_bytes = _expected_receipt(expected, name=path.name)
    if sha256_file(path) != expected_digest:
        raise ValueError(f"Sealed index payload hash mismatch: {path.name}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Sealed index payload byte count mismatch: {path.name}")


def load_canonical_protocol(path: Path, *, expected_sha256: str) -> Any:
    """Load the data owner's stdlib-only P107 authority module by exact hash."""
    path = Path(path)
    if not is_sha256(expected_sha256):
        raise ValueError("--expected-protocol-sha256 must be a lowercase SHA-256")
    source = read_regular_file_bytes(path, name="--protocol-path")
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected_sha256:
        raise ValueError("canonical protocol module does not match the externally pinned SHA-256")
    module_name = "p107_canonical_event_protocol"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None, origin=str(path))
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
    for name in ("canonical_json", "canonical_sha256", "event_id", "source_group_id", "validate_event"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"canonical protocol is missing required validator: {name}")
    return module


def validate_coverage_expectations(value: Any) -> dict[str, Any]:
    """Consume owner v3 skill identities without lexical reinterpretation."""
    value = require_mapping(value, "coverage expectations")
    required = {
        "schema_version", "official_task_metadata_sha256", "official_skill_vocabulary_sha256",
        "expected_task_ids", "expected_skill_vocabulary", "required_task_skill_pairs",
    }
    if set(value) != required or value.get("schema_version") != "p107-official-coverage-expectations-v3":
        raise ValueError("coverage expectations must use the exact P107 official-metadata schema")
    for name in ("official_task_metadata_sha256", "official_skill_vocabulary_sha256"):
        if not is_sha256(value.get(name)):
            raise ValueError(f"coverage expectations {name} must be a SHA-256")
    tasks, skills = value["expected_task_ids"], value["expected_skill_vocabulary"]
    if (not isinstance(tasks, list) or not tasks or any(type(task) is not int or task < 0 for task in tasks) or
            tasks != sorted(set(tasks))):
        raise ValueError("coverage expected_task_ids must be sorted unique nonnegative integers")
    if not isinstance(skills, list) or not skills:
        raise ValueError("coverage expected_skill_vocabulary must be a nonempty official skill ID/description list")
    previous_id = -1
    descriptions: set[str] = set()
    for skill in skills:
        skill = require_mapping(skill, "coverage official skill")
        if set(skill) != {"skill_id", "skill_description"}:
            raise ValueError("coverage official skills must be exact skill_id/skill_description objects")
        skill_id, description = skill["skill_id"], skill["skill_description"]
        if type(skill_id) is not int or skill_id < 0 or skill_id <= previous_id:
            raise ValueError("coverage official skill IDs must be strictly increasing nonnegative integers")
        if not isinstance(description, str) or not description or description in descriptions:
            raise ValueError("coverage official skill descriptions must be nonempty and unique")
        previous_id = skill_id
        descriptions.add(description)
    pairs = value["required_task_skill_pairs"]
    if pairs is not None:
        if not isinstance(pairs, list):
            raise ValueError("coverage required_task_skill_pairs must be null or a list")
        skill_ids = {skill["skill_id"] for skill in skills}
        previous: tuple[int, int] | None = None
        for pair in pairs:
            pair = require_mapping(pair, "coverage required task/skill pair")
            if (set(pair) != {"task_index", "skill_id"} or type(pair["task_index"]) is not int or
                    type(pair["skill_id"]) is not int or pair["task_index"] not in tasks or
                    pair["skill_id"] not in skill_ids):
                raise ValueError("coverage required task/skill pair is outside the official vocabulary")
            key = pair["task_index"], pair["skill_id"]
            if previous is not None and key <= previous:
                raise ValueError("coverage required task/skill pairs must be sorted unique")
            previous = key
    return dict(value)


def read_coverage_expectations(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if not is_sha256(expected_sha256):
        raise ValueError("coverage expectations seal must be a SHA-256")
    expectations = validate_coverage_expectations(read_json(path))
    if canonical_sha256(expectations) != expected_sha256:
        raise ValueError("official coverage expectations differ from the inventory-sealed contract")
    return expectations


@dataclass(frozen=True)
class SourceGroup:
    source_group_id: str
    source_release_manifest_sha256: str
    task_id: int
    task_instance_id: int
    original_split: str
    usage_role: str
    source_episode_ids: tuple[int, ...]


@dataclass(frozen=True)
class Candidate:
    event: Mapping[str, Any]
    event_id: str
    source_group_id: str
    task_id: int
    task_instance_id: int
    episode_key: tuple[str, int, int]
    episode_index: int
    episode_length: int
    anchor_frame: int
    interval_start: int
    interval_end: int
    skill_ids: tuple[int, ...]
    raw_source_verbs: tuple[str, ...]
    skill_keys: tuple[str, ...]
    repeat_attempt_count: int
    boundary_before: bool
    boundary_after: bool
    long_interval: bool
    candidate_signals: tuple[str, ...]
    selection_stratum: str

    @property
    def is_normal_control(self) -> bool:
        return self.selection_stratum == "NORMAL_CONTROL_CANDIDATE"


def source_identity(source: Mapping[str, Any], *, event_id: str) -> tuple[str, int, int]:
    """Keep source identity at the episode level; never collapse to task alone."""
    raw_episode = require_int(source.get("raw_episode_id"), f"{event_id}.source.raw_episode_id")
    episode_index = require_int(source.get("episode_index"), f"{event_id}.source.episode_index")
    group = source.get("source_group_id")
    if not is_sha256(group):
        raise ValueError(f"{event_id}.source.source_group_id must be a SHA-256")
    return str(group), raw_episode, episode_index


def skill_key(skill: Mapping[str, Any]) -> str:
    """A metadata grouping key, not a label or an inferred semantic fact."""
    public_fields = ("verb", "target", "source", "destination", "target_part", "arm")
    values = {field: skill.get(field, "") for field in public_fields}
    if not isinstance(values["verb"], str) or not values["verb"]:
        raise ValueError("source skill must carry a nonempty verb")
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("source skill binding fields must be strings when present")
    return canonical_sha256({"kind": "metadata_skill_target_key", "fields": values})


def validate_index(index: Path, *, expected_source_manifest_sha256: str,
                   canonical_protocol: Any, expected_protocol_sha256: str) -> tuple[dict[str, Any], Path, dict[str, SourceGroup]]:
    """Verify the exact sealed input before sampling any metadata candidate."""
    index = Path(index)
    if index.is_symlink() or not index.is_dir():
        raise ValueError("--index must be an existing regular sealed index directory")
    if not is_sha256(expected_source_manifest_sha256):
        raise ValueError("--expected-source-release-manifest-sha256 must be a lowercase SHA-256")
    manifest_path = index / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("index manifest must be an object")
    if manifest.get("schema_version") != INDEX_SCHEMA or manifest.get("status") != INDEX_STATUS:
        raise ValueError("input is not a sealed P107 metadata-only event index")
    if manifest.get("source_release_manifest_sha256") != expected_source_manifest_sha256:
        raise ValueError("frozen v4 source manifest hash does not match the predeclared queue policy")
    if manifest.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("index was not built with the externally pinned canonical protocol")
    if manifest.get("training_eligible") is not False:
        raise ValueError("candidate input must explicitly be non-trainable")
    files = require_mapping(manifest.get("files"), "index.files")
    source_groups_path, events_path = index / "source_groups.jsonl", index / "event_candidates.jsonl"
    source_rows = read_jsonl_verified(source_groups_path, expected=files.get("source_groups.jsonl", {}))
    if files["source_groups.jsonl"].get("rows") != len(source_rows):
        raise ValueError("sealed source-group row count mismatch")
    source_groups: dict[str, SourceGroup] = {}
    for row in source_rows:
        group_id = row.get("source_group_id")
        if not is_sha256(group_id) or group_id in source_groups:
            raise ValueError("source_groups.jsonl has an absent or duplicate immutable source_group_id")
        source_release = row.get("source_release_manifest_sha256")
        if source_release != expected_source_manifest_sha256:
            raise ValueError("source group release identity drifted from the sealed index")
        task_id = require_int(row.get("task_index"), "source_group.task_index")
        task_instance_id = require_int(row.get("task_instance_id"), "source_group.task_instance_id", minimum=1)
        original_split, usage_role = row.get("original_split"), row.get("usage_role")
        if original_split not in {"train", "eval"}:
            raise ValueError("source group has an unknown immutable split")
        if usage_role not in {"student_candidate", "annotation_calibration", "evaluation_only"}:
            raise ValueError("source group has an unknown usage role")
        if (original_split == "eval") != (usage_role == "evaluation_only"):
            raise ValueError("source-group eval role cannot be relabeled for annotation")
        if task_instance_id >= 301:
            raise ValueError("public-test source instance entered the candidate index")
        episodes = row.get("source_episode_ids")
        if (not isinstance(episodes, list) or not episodes or
                any(type(episode) is not int or episode < 0 for episode in episodes) or
                episodes != sorted(set(episodes)) or row.get("source_episode_count") != len(episodes)):
            raise ValueError("source group has an invalid immutable source-episode inventory")
        expected_group_id = canonical_protocol.source_group_id({
            "source_release_manifest_sha256": source_release,
            "task_index": task_id,
            "task_instance_id": task_instance_id,
        })
        if group_id != expected_group_id:
            raise ValueError("source group ID is not canonical for its immutable release/task/instance")
        source_groups[str(group_id)] = SourceGroup(str(group_id), str(source_release), task_id, task_instance_id,
                                                    str(original_split), str(usage_role), tuple(episodes))
    # Event rows are parsed in a single streaming pass by parse_candidates;
    # retaining a 1GiB full index here would make a 40-window queue unbounded.
    return manifest, events_path, source_groups


def validate_inventory_seal(index: Path, *, expected_inventory_seal_sha256: str) -> tuple[Mapping[str, Any], str]:
    """Pin the inventory seal supplied by the index owner, not just the files.

    The producer owns the seal; the consumer verifies its fixed contract in
    addition to pinning exact reviewed bytes.  Nothing in a queue invocation
    may override the event/group payload receipts it seals.
    """
    if not is_sha256(expected_inventory_seal_sha256):
        raise ValueError("--expected-inventory-seal-sha256 must be a lowercase SHA-256")
    seal_path = index / "inventory_seal.json"
    seal = read_json(seal_path)
    if not isinstance(seal, Mapping):
        raise ValueError("inventory_seal.json must be an object")
    actual = sha256_file(seal_path)
    if actual != expected_inventory_seal_sha256:
        raise ValueError("inventory seal hash does not match the externally recorded handoff")
    expected_fields = {
        "schema_version", "index_manifest_sha256", "source_release_manifest_sha256",
        "coverage_expectations_sha256", "expected_payload_files", "payload_files",
    }
    if set(seal) != expected_fields or seal.get("schema_version") != "memlite-event-inventory-seal-v1":
        raise ValueError("inventory seal schema/field set is not the fixed P107 contract")
    manifest_path = index / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("sealed index manifest must be an object")
    if seal.get("index_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("inventory seal does not bind the exact index manifest")
    if seal.get("source_release_manifest_sha256") != manifest.get("source_release_manifest_sha256"):
        raise ValueError("inventory seal source release identity drifted from index manifest")
    if not is_sha256(seal.get("coverage_expectations_sha256")):
        raise ValueError("inventory seal coverage expectations hash is invalid")
    expected_payload_files = seal.get("expected_payload_files")
    if expected_payload_files != ["event_candidates.jsonl", "source_groups.jsonl"]:
        raise ValueError("inventory seal payload file set/order is not the fixed P107 contract")
    sealed_payloads = require_mapping(seal.get("payload_files"), "inventory seal payload_files")
    if set(sealed_payloads) != set(expected_payload_files):
        raise ValueError("inventory seal payload receipt set is incomplete or expanded")
    manifest_files = require_mapping(manifest.get("files"), "index manifest files")
    for name in expected_payload_files:
        receipt = require_mapping(sealed_payloads[name], f"inventory seal receipt {name}")
        if set(receipt) != {"sha256", "rows", "bytes"} or not is_sha256(receipt.get("sha256")):
            raise ValueError("inventory seal payload receipt is malformed")
        if receipt != manifest_files.get(name):
            raise ValueError("inventory seal payload receipt drifted from index manifest")
    return seal, actual


def parse_candidates(event_rows: Sequence[Mapping[str, Any]] | Path, source_groups: Mapping[str, SourceGroup], *,
                     expected_source_manifest_sha256: str, canonical_protocol: Any,
                     expected_task_ids: Iterable[int], expected_skill_ids: Iterable[int],
                     retained_usage_roles: set[str], max_retained_candidates: int,
                     expected_event_receipt: Mapping[str, Any],
                     boundary_gap_frames: int, long_interval_frames: int,
                     retained_event_ids: set[str] | None = None) -> tuple[list[Candidate], Counter[str], Counter[str]]:
    """Stream-validate events and retain only bounded, official-vocabulary candidates."""
    if (boundary_gap_frames < 0 or long_interval_frames < 1 or max_retained_candidates < 1 or
            not retained_usage_roles <= {"student_candidate", "annotation_calibration"}):
        raise ValueError("boundary gap and long interval policy values are invalid")
    official_task_ids, official_skill_ids = set(expected_task_ids), set(expected_skill_ids)
    if retained_event_ids is not None and any(not is_sha256(value) for value in retained_event_ids):
        raise ValueError("retained_event_ids must contain only canonical event IDs")
    parsed: list[dict[str, Any]] = []
    excluded_roles: Counter[str] = Counter()
    excluded_unrecognized_vocabulary: Counter[str] = Counter()
    seen_event_ids: set[str] = set()
    # Usage roles are immutable split reservations.  Even metadata-only
    # sampling features must not borrow occurrence counts from calibration or
    # eval groups when ranking student candidates.
    repeats: dict[tuple[str, int, str], set[tuple[str, int, int]]] = defaultdict(set)
    rows = (iter_jsonl_verified(event_rows, expected=expected_event_receipt)
            if isinstance(event_rows, Path) else event_rows)
    row_count = 0
    for event in rows:
        row_count += 1
        # Do this before role exclusion: eval is excluded from queues, not
        # exempted from canonical source/event identity validation.
        canonical_protocol.validate_event(event)
        event_id = event.get("event_id")
        if not is_sha256(event_id) or event_id in seen_event_ids:
            raise ValueError("event index has an absent or duplicate immutable event_id")
        seen_event_ids.add(str(event_id))
        source = require_mapping(event.get("source"), f"{event_id}.source")
        group_id, raw_episode_id, episode_index = source_identity(source, event_id=str(event_id))
        group = source_groups.get(group_id)
        if group is None:
            raise ValueError("event references an unknown source group")
        if (source.get("original_split") != group.original_split or
                event.get("usage_role") != group.usage_role or
                source.get("source_release_manifest_sha256") != expected_source_manifest_sha256 or
                source.get("source_release_manifest_sha256") != group.source_release_manifest_sha256 or
                source.get("task_index") != group.task_id or
                source.get("task_instance_id") != group.task_instance_id):
            raise ValueError("event/source role or immutable split identity drifted from source_groups.jsonl")
        if not is_sha256(source.get("source_annotation_sha256")):
            raise ValueError("event source has no immutable source annotation hash")
        if episode_index not in group.source_episode_ids:
            raise ValueError("event episode is absent from its immutable source-group inventory")
        if group.original_split == "eval" or group.usage_role == "evaluation_only":
            excluded_roles["evaluation_only"] += 1
            continue
        if group.original_split != "train" or group.usage_role not in {"student_candidate", "annotation_calibration"}:
            raise ValueError("a non-eval event has an invalid train usage role")
        # Even an output-zero role must not let a pre-labeled source event
        # cross this candidate-only queue boundary.
        evidence = require_mapping(event.get("evidence"), f"{event_id}.evidence")
        if evidence != {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None}:
            raise ValueError("metadata candidate index carries non-missing evidence")
        if group.task_id not in official_task_ids:
            excluded_unrecognized_vocabulary[f"TASK_ID:{group.task_id}"] += 1
            continue
        # Official source-skill identity is checked for every non-eval row,
        # including an output-zero role.  Retention may be skipped below, but
        # it must not hide an unexpected ID or turn raw text into a fallback.
        source_skills = event.get("skill_bundle")
        if not isinstance(source_skills, list) or not source_skills:
            raise ValueError("event candidate has no source skill bundle")
        source_skill_ids = []
        for skill in source_skills:
            mapping = require_mapping(skill, "event.skill_bundle member")
            skill_id = mapping.get("skill_id")
            if type(skill_id) is not int or skill_id < 0:
                raise ValueError("source skill requires a nonnegative official skill_id")
            source_skill_ids.append(skill_id)
        unknown_skill_ids = sorted(set(source_skill_ids) - official_skill_ids)
        if unknown_skill_ids:
            for skill_id in unknown_skill_ids:
                excluded_unrecognized_vocabulary[f"SKILL_ID:{skill_id}"] += 1
            continue
        # A private phase pilot may name one exact parent from a large sealed
        # student pool.  Still validate every row's immutable identity and
        # vocabulary above, but retain only the explicitly pinned parent so a
        # bounded candidate cap cannot turn the stream position into a
        # selection authority.  The final presence check below is fail-closed.
        if retained_event_ids is not None and event_id not in retained_event_ids:
            continue
        if group.usage_role not in retained_usage_roles:
            # The stream still validates every source/event identity, but a
            # zero-budget role cannot consume a full-index candidate heap.
            continue
        observation = require_mapping(event.get("observation"), f"{event_id}.observation")
        interval = require_mapping(event.get("event_interval"), f"{event_id}.event_interval")
        anchor = require_int(observation.get("frame"), f"{event_id}.observation.frame")
        start = require_int(interval.get("start_frame"), f"{event_id}.event_interval.start_frame")
        end = require_int(interval.get("end_frame"), f"{event_id}.event_interval.end_frame")
        episode_length = require_int(source.get("episode_length"), f"{event_id}.source.episode_length", minimum=1)
        if not start <= anchor < end <= episode_length:
            raise ValueError("event observation/interval is not a valid source-clock candidate")
        timestamp = observation.get("timestamp_s")
        if type(timestamp) not in (int, float) or isinstance(timestamp, bool) or not math.isfinite(float(timestamp)):
            raise ValueError("event observation timestamp is invalid")
        if abs(float(timestamp) - anchor / FRAME_RATE_HZ) > 1e-9:
            raise ValueError("event observation timestamp drifted from frozen source clock")
        skill_ids, raw_source_verbs, keys = [], [], []
        for skill in source_skills:
            mapping = require_mapping(skill, "event.skill_bundle member")
            skill_id = mapping.get("skill_id")
            if type(skill_id) is not int or skill_id < 0:
                raise ValueError("source skill requires a nonnegative official skill_id")
            skill_ids.append(skill_id)
            verb = mapping.get("verb")
            if isinstance(verb, str) and verb:
                raw_source_verbs.append(verb)
            token = skill_key(mapping)
            keys.append(token)
        for token in keys:
            repeats[group.usage_role, group.task_id, token].add((group_id, raw_episode_id, episode_index))
        if len(parsed) >= max_retained_candidates:
            raise ValueError("retained candidate cap reached; provide a narrower approved source-role/group subset")
        parsed.append({
            "event": event, "event_id": str(event_id), "source_group_id": group_id,
            "task_id": group.task_id, "task_instance_id": group.task_instance_id,
            "episode_key": (group_id, raw_episode_id, episode_index), "episode_index": episode_index,
            "episode_length": episode_length, "anchor_frame": anchor, "interval_start": start,
            "interval_end": end, "skill_ids": tuple(sorted(set(skill_ids))),
            "raw_source_verbs": tuple(sorted(set(raw_source_verbs))), "skill_keys": tuple(sorted(set(keys))),
        })
    _, expected_event_rows, _ = _expected_receipt(expected_event_receipt, name="event_candidates.jsonl")
    if row_count != expected_event_rows:
        raise ValueError("sealed event row count mismatch")
    if retained_event_ids is not None:
        retained_ids = {row["event_id"] for row in parsed}
        if retained_ids != retained_event_ids:
            raise ValueError("requested retained event ID is absent from the sealed event index")
    by_episode: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        by_episode[row["episode_key"]].append(row)
    boundaries: dict[str, tuple[bool, bool]] = {}
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: (row["interval_start"], row["interval_end"], row["event_id"]))
        for index, row in enumerate(episode_rows):
            before = index > 0 and 0 <= row["interval_start"] - episode_rows[index - 1]["interval_end"] <= boundary_gap_frames
            after = (index + 1 < len(episode_rows) and
                     0 <= episode_rows[index + 1]["interval_start"] - row["interval_end"] <= boundary_gap_frames)
            boundaries[row["event_id"]] = before, after
    candidates: list[Candidate] = []
    for row in parsed:
        repeated = max((len(repeats[row["event"]["usage_role"], row["task_id"], key])
                        for key in row["skill_keys"]), default=0)
        before, after = boundaries[row["event_id"]]
        long_interval = row["interval_end"] - row["interval_start"] >= long_interval_frames
        signals: list[str] = []
        if repeated >= 2:
            signals.append("REPEATED_SKILL_TARGET_CANDIDATE")
        if before or after:
            signals.append("ANNOTATION_BOUNDARY_CANDIDATE")
        if long_interval:
            # Duration is a review locator only, explicitly not a dwell/backtrack truth claim.
            signals.append("LONG_ANNOTATED_INTERVAL_CANDIDATE")
        if not signals:
            signals.append("NORMAL_CONTROL_CANDIDATE")
        candidates.append(Candidate(**row, repeat_attempt_count=repeated, boundary_before=before,
                                    boundary_after=after, long_interval=long_interval, candidate_signals=tuple(signals),
                                    selection_stratum=signals[0]))
    return candidates, excluded_roles, excluded_unrecognized_vocabulary


def candidate_rank(candidate: Candidate, seed: str) -> tuple[int, int, int, int, str]:
    score = {
        "REPEATED_SKILL_TARGET_CANDIDATE": 3,
        "ANNOTATION_BOUNDARY_CANDIDATE": 2,
        "LONG_ANNOTATED_INTERVAL_CANDIDATE": 1,
        "NORMAL_CONTROL_CANDIDATE": 0,
    }[candidate.selection_stratum]
    tie = hashlib.sha256(f"{seed}:{candidate.event_id}".encode("utf-8")).hexdigest()
    return (-score, -candidate.repeat_attempt_count, -(candidate.interval_end - candidate.interval_start),
            candidate.task_id, tie)


class DiverseSelector:
    """Stable, capped selection with task/skill/control coverage passes."""

    def __init__(self, candidates: Sequence[Candidate], *, budget: int, seed: str,
                 max_per_episode: int, min_separation_frames: int, max_per_source_group: int):
        if budget < 0 or max_per_episode < 1 or min_separation_frames < 0 or max_per_source_group < 1:
            raise ValueError("selection budget/caps must be nonnegative and nonzero where required")
        self.candidates = sorted(candidates, key=lambda item: candidate_rank(item, seed))
        self.budget = budget
        self.max_per_episode = max_per_episode
        self.min_separation_frames = min_separation_frames
        self.max_per_source_group = max_per_source_group
        self.selected: list[tuple[Candidate, str]] = []
        self.selected_ids: set[str] = set()
        self.episode_anchors: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        self.group_counts: Counter[str] = Counter()
        self.attempted_near_duplicates: dict[str, set[str]] = defaultdict(set)

    def _reason_blocked(self, candidate: Candidate) -> str | None:
        if candidate.event_id in self.selected_ids:
            return "ALREADY_SELECTED"
        anchors = self.episode_anchors[candidate.episode_key]
        if len(anchors) >= self.max_per_episode:
            return "SAME_EPISODE_CAP"
        if any(abs(candidate.anchor_frame - anchor) < self.min_separation_frames for anchor in anchors):
            return "ADJACENT_WINDOW_CAP"
        if self.group_counts[candidate.source_group_id] >= self.max_per_source_group:
            return "SOURCE_GROUP_CAP"
        return None

    def select(self, candidate: Candidate, phase: str) -> bool:
        if len(self.selected) >= self.budget:
            return False
        blocked = self._reason_blocked(candidate)
        if blocked:
            if blocked in {"SAME_EPISODE_CAP", "ADJACENT_WINDOW_CAP"}:
                self.attempted_near_duplicates[blocked].add(candidate.event_id)
            return False
        self.selected.append((candidate, phase))
        self.selected_ids.add(candidate.event_id)
        self.episode_anchors[candidate.episode_key].append(candidate.anchor_frame)
        self.group_counts[candidate.source_group_id] += 1
        return True

    def first_selectable(self, candidates: Iterable[Candidate], phase: str) -> None:
        for candidate in candidates:
            if self.select(candidate, phase):
                return

    def summary(self) -> dict[str, int]:
        selected = set(self.selected_ids)
        return {
            "same_episode_cap_rejected": len(self.attempted_near_duplicates["SAME_EPISODE_CAP"] - selected),
            "adjacent_window_cap_rejected": len(self.attempted_near_duplicates["ADJACENT_WINDOW_CAP"] - selected),
            "selected_episode_count": len(self.episode_anchors),
        }


def seeded_order(values: Iterable[Any], *, seed: str, namespace: str) -> list[Any]:
    """A deterministic permutation that does not privilege low task IDs."""
    unique = sorted(set(values))
    return sorted(unique, key=lambda value: (
        hashlib.sha256(f"{seed}:{namespace}:{canonical_json(value)}".encode("utf-8")).digest(),
        canonical_json(value),
    ))


def select_diverse(candidates: Sequence[Candidate], *, budget: int, seed: str, max_per_episode: int,
                   min_separation_frames: int, max_per_source_group: int,
                   normal_control_fraction: float,
                   priority_skill_ids: Iterable[int] | None = None) -> tuple[list[tuple[Candidate, str]], dict[str, Any]]:
    if not 0.0 <= normal_control_fraction <= 1.0:
        raise ValueError("normal control fraction must be in [0, 1]")
    selector = DiverseSelector(candidates, budget=budget, seed=seed, max_per_episode=max_per_episode,
                               min_separation_frames=min_separation_frames,
                               max_per_source_group=max_per_source_group)
    by_task: dict[int, list[Candidate]] = defaultdict(list)
    by_skill: dict[str, list[Candidate]] = defaultdict(list)
    by_stratum: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in selector.candidates:
        by_task[candidate.task_id].append(candidate)
        by_stratum[candidate.selection_stratum].append(candidate)
        for skill in candidate.skill_ids:
            by_skill[skill].append(candidate)
    # Reserve official skills before task coverage can exhaust a small budget.
    # This is joint: among candidates that add uncovered vocabulary, one that
    # covers more missing skills and comes from a new task is preferred, with
    # rare source skills winning deterministic ties.  It never treats a raw
    # verb as an official skill identity.
    priority_skills = set(by_skill if priority_skill_ids is None else priority_skill_ids)
    priority_skills &= set(by_skill)
    covered_priority_skills: set[int] = set()
    while len(selector.selected) < budget:
        uncovered = priority_skills - covered_priority_skills
        if not uncovered:
            break
        covered_tasks = {candidate.task_id for candidate, _ in selector.selected}
        ranked = []
        for candidate in selector.candidates:
            gain = set(candidate.skill_ids) & uncovered
            if not gain:
                continue
            rarity = min(len(by_skill[skill]) for skill in gain)
            ranked.append((candidate, gain, rarity))
        if not ranked:
            break
        ranked.sort(key=lambda item: (
            -len(item[1]), item[2], -(item[0].task_id not in covered_tasks), candidate_rank(item[0], seed),
        ))
        before = len(selector.selected)
        for candidate, _, _ in ranked:
            if selector.select(candidate, "JOINT_RARE_SKILL_RESERVE"):
                covered_priority_skills.update(set(candidate.skill_ids) & priority_skills)
                break
        if len(selector.selected) == before:
            break
    # Finish declared task coverage only after the rare-skill reserve, so a
    # 40-window calibration packet can cover up to 35 official skills.
    for task_id in seeded_order(by_task, seed=seed, namespace="task-coverage"):
        selector.first_selectable(by_task[task_id], "TASK_COVERAGE")
    normal_target = min(budget, int(round(budget * normal_control_fraction)))
    while sum(candidate.is_normal_control for candidate, _ in selector.selected) < normal_target:
        before = len(selector.selected)
        selector.first_selectable((candidate for candidate in selector.candidates if candidate.is_normal_control),
                                  "NORMAL_CONTROL_QUOTA")
        if len(selector.selected) == before:
            break
    for stratum in seeded_order(by_stratum, seed=seed, namespace="stratum-seed"):
        if stratum == "NORMAL_CONTROL_CANDIDATE":
            continue
        if not any(candidate.selection_stratum == stratum for candidate, _ in selector.selected):
            selector.first_selectable(by_stratum[stratum], "STRATUM_SEED")
    # Deterministic balanced fill: prefer an underrepresented task, then skill,
    # then stratum before using the stable event-id tie breaker.
    while len(selector.selected) < budget:
        task_counts = Counter(candidate.task_id for candidate, _ in selector.selected)
        skill_counts = Counter(skill for candidate, _ in selector.selected for skill in candidate.skill_ids)
        stratum_counts = Counter(candidate.selection_stratum for candidate, _ in selector.selected)
        remaining = [candidate for candidate in selector.candidates if candidate.event_id not in selector.selected_ids]
        if not remaining:
            break
        remaining.sort(key=lambda candidate: (
            task_counts[candidate.task_id],
            min((skill_counts[skill] for skill in candidate.skill_ids), default=0),
            stratum_counts[candidate.selection_stratum],
            candidate_rank(candidate, seed),
        ))
        before = len(selector.selected)
        for candidate in remaining:
            if selector.select(candidate, "DIVERSE_FILL"):
                break
        if len(selector.selected) == before:
            break
    selected_priority_skills = {
        skill for candidate, _ in selector.selected for skill in candidate.skill_ids if skill in priority_skills
    }
    return selector.selected, {
        **selector.summary(),
        "priority_skill_ids_present_in_pool": sorted(priority_skills),
        "priority_skill_ids_queued": sorted(selected_priority_skills),
        "priority_skill_ids_present_but_not_queued": sorted(priority_skills - selected_priority_skills),
        "priority_skill_reserve_target": min(budget, len(priority_skills)),
        "priority_skill_reserve_achieved": len(selected_priority_skills),
    }


def window_spec(candidate: Candidate, *, actor_history_frames: int, review_before_frames: int,
                review_after_frames: int, sample_stride_frames: int) -> dict[str, Any]:
    if min(actor_history_frames, review_before_frames, review_after_frames) < 0 or sample_stride_frames < 1:
        raise ValueError("temporal window policy values are invalid")
    anchor = candidate.anchor_frame
    actor_start = max(0, anchor - actor_history_frames)
    before_start = max(0, anchor - review_before_frames)
    after_end = min(candidate.episode_length, anchor + 1 + review_after_frames)

    def samples(start: int, end: int, *, include: int | None = None) -> list[int]:
        result = list(range(start, end, sample_stride_frames))
        if include is not None and start <= include < end:
            result.append(include)
        return sorted(set(result))

    actor_frames = samples(actor_start, anchor + 1, include=anchor)
    before_frames = samples(before_start, anchor, include=before_start) if before_start < anchor else []
    after_frames = samples(anchor + 1, after_end, include=after_end - 1) if anchor + 1 < after_end else []
    return {
        "source_clock": "published_local_frame_index_offset_0",
        "frame_rate_hz": FRAME_RATE_HZ,
        "anchor_frame": anchor,
        "anchor_timestamp_s": anchor / FRAME_RATE_HZ,
        "actor_available_window": {
            "start_frame": actor_start, "end_frame_exclusive": anchor + 1,
            "start_timestamp_s": actor_start / FRAME_RATE_HZ,
            "end_timestamp_s_exclusive": (anchor + 1) / FRAME_RATE_HZ,
            "sampled_frames": actor_frames,
            "causal_use": "CURRENT_DECISION_ONLY",
        },
        "offline_review_before_window": {
            "start_frame": before_start, "end_frame_exclusive": anchor,
            "start_timestamp_s": before_start / FRAME_RATE_HZ,
            "end_timestamp_s_exclusive": anchor / FRAME_RATE_HZ,
            "sampled_frames": before_frames,
            "review_use": "TEMPORAL_CONTEXT_ONLY",
        },
        "offline_review_after_window": {
            "start_frame": anchor + 1, "end_frame_exclusive": after_end,
            "start_timestamp_s": (anchor + 1) / FRAME_RATE_HZ,
            "end_timestamp_s_exclusive": after_end / FRAME_RATE_HZ,
            "sampled_frames": after_frames,
            "review_use": "OFFLINE_BC_QUALITY_ONLY_NOT_ACTOR_EVIDENCE",
        },
        "temporal_review_window": {
            "start_frame": before_start, "end_frame_exclusive": after_end,
            "start_timestamp_s": before_start / FRAME_RATE_HZ,
            "end_timestamp_s_exclusive": after_end / FRAME_RATE_HZ,
            "sampled_frames": sorted(set(before_frames + actor_frames + after_frames)),
            "review_use": "HUMAN_OR_MODEL_OFFLINE_REVIEW_ONLY",
        },
    }


def _private_phase_lineage(candidate: Candidate) -> Mapping[str, Any] | None:
    """Authenticate the narrow student phase path before deriving its clock."""
    raw_lineage = candidate.event.get("phase_lineage")
    if raw_lineage is None:
        return None
    lineage = require_mapping(raw_lineage, f"{candidate.event_id}.phase_lineage")
    is_phase_schema = lineage.get("schema_version") == PRIVATE_PHASE_DERIVATION_SCHEMA
    if candidate.event.get("usage_role") == "student_candidate" and is_phase_schema and \
            lineage.get("private_goal_state_review") is not True:
        raise ValueError("student phase event lacks the explicit private goal-state opt-in")
    if lineage.get("private_goal_state_review") is not True:
        return None
    if candidate.event.get("usage_role") != "student_candidate":
        raise ValueError("private goal-state phase must retain the student_candidate role")
    source = require_mapping(candidate.event.get("source"), f"{candidate.event_id}.source")
    if source.get("original_split") != "train":
        raise ValueError("private goal-state phase must retain the TRAIN split")
    if (not is_sha256(lineage.get("parent_event_id")) or
            lineage.get("goal_state_question") != PRIVATE_GOAL_STATE_REVIEW_QUERY or
            lineage.get("goal_state_question_sha256") != PRIVATE_GOAL_STATE_REVIEW_QUERY_SHA256):
        raise ValueError("private goal-state phase question/parent binding is not authenticated")
    for field in ("training_eligible", "outcome_supervision", "recovery_supervision",
                  "action_bc_supervision", "dart_supervision"):
        if lineage.get(field) is not False:
            raise ValueError(f"private goal-state phase {field} gate is not false")
    parent_interval = require_mapping(lineage.get("parent_event_interval"),
                                      f"{candidate.event_id}.phase_lineage.parent_event_interval")
    if set(parent_interval) != {"start_frame", "end_frame"}:
        raise ValueError("private goal-state phase parent interval schema is invalid")
    parent_start = require_int(parent_interval.get("start_frame"), "private phase parent start")
    parent_end = require_int(parent_interval.get("end_frame"), "private phase parent end", minimum=1)
    if (parent_start != candidate.interval_start or parent_end != candidate.interval_end or
            not parent_start < parent_end):
        raise ValueError("private goal-state phase parent interval drifted from the canonical event")
    anchor = candidate.anchor_frame
    phase = lineage.get("observation_phase")
    if phase == "ENTRY":
        if anchor != parent_start:
            raise ValueError("private ENTRY phase anchor is not the parent start")
    elif phase == "TERMINAL":
        if anchor != parent_end - 1 or anchor <= parent_start:
            raise ValueError("private TERMINAL phase anchor is not the parent end-1")
    else:
        raise ValueError("private goal-state phase supports only ENTRY and TERMINAL")
    return lineage


def _private_phase_window_spec(candidate: Candidate) -> dict[str, Any] | None:
    lineage = _private_phase_lineage(candidate)
    if lineage is None:
        return None
    anchor = candidate.anchor_frame
    parent_start = candidate.interval_start
    phase = lineage["observation_phase"]
    actor_frames = [anchor] if phase == "ENTRY" else [parent_start, anchor]
    actor_start = actor_frames[0]
    actor_end = anchor + 1
    return {
        "source_clock": "published_local_frame_index_offset_0",
        "frame_rate_hz": FRAME_RATE_HZ,
        "anchor_frame": anchor,
        "anchor_timestamp_s": anchor / FRAME_RATE_HZ,
        "actor_available_window": {
            "start_frame": actor_start, "end_frame_exclusive": actor_end,
            "start_timestamp_s": actor_start / FRAME_RATE_HZ,
            "end_timestamp_s_exclusive": actor_end / FRAME_RATE_HZ,
            "sampled_frames": actor_frames,
            "causal_use": "CURRENT_DECISION_ONLY",
        },
        "offline_review_before_window": {
            "start_frame": anchor, "end_frame_exclusive": anchor,
            "start_timestamp_s": anchor / FRAME_RATE_HZ,
            "end_timestamp_s_exclusive": anchor / FRAME_RATE_HZ,
            "sampled_frames": [],
            "review_use": "TEMPORAL_CONTEXT_ONLY",
        },
        "offline_review_after_window": {
            "start_frame": actor_end, "end_frame_exclusive": actor_end,
            "start_timestamp_s": actor_end / FRAME_RATE_HZ,
            "end_timestamp_s_exclusive": actor_end / FRAME_RATE_HZ,
            "sampled_frames": [],
            "review_use": "OFFLINE_BC_QUALITY_ONLY_NOT_ACTOR_EVIDENCE",
        },
        "temporal_review_window": {
            "start_frame": actor_start, "end_frame_exclusive": actor_end,
            "start_timestamp_s": actor_start / FRAME_RATE_HZ,
            "end_timestamp_s_exclusive": actor_end / FRAME_RATE_HZ,
            "sampled_frames": actor_frames,
            "review_use": "HUMAN_OR_MODEL_OFFLINE_REVIEW_ONLY",
        },
    }


def job_id(*, queue_kind: str, event_id: str, policy_sha256: str) -> str:
    return canonical_sha256({"schema_version": QUEUE_SCHEMA, "queue_kind": queue_kind,
                             "event_id": event_id, "policy_sha256": policy_sha256})


def render_request(candidate: Candidate, *, request_id: str, windows: Mapping[str, Any]) -> dict[str, Any]:
    """Ask, but do not invoke, a camera-native temporal renderer.

    The index stores anchor locators.  It cannot safely extrapolate a single
    chunk/file locator to neighbouring frames, so a renderer must resolve each
    requested frame through its own frozen metadata lookup.
    """
    source = require_mapping(candidate.event["source"], "event.source")
    temporal = require_mapping(windows["temporal_review_window"], "temporal_review_window")
    return {
        "schema_version": RENDER_REQUEST_SCHEMA,
        "request_id": request_id,
        "status": "PENDING_RENDERER_HANDSHAKE_NO_RGB_DECODED",
        "event_id": candidate.event_id,
        "source_identity": {
            "source_group_id": candidate.source_group_id,
            "task_index": candidate.task_id,
            "task_instance_id": candidate.task_instance_id,
            "raw_episode_id": source["raw_episode_id"],
            "episode_index": candidate.episode_index,
            "source_release_manifest_sha256": source["source_release_manifest_sha256"],
            "source_annotation_sha256": source["source_annotation_sha256"],
        },
        "requested_frame_indices": temporal["sampled_frames"],
        "actor_available_frame_indices": windows["actor_available_window"]["sampled_frames"],
        "offline_review_before_frame_indices": windows["offline_review_before_window"]["sampled_frames"],
        "offline_review_after_frame_indices": windows["offline_review_after_window"]["sampled_frames"],
        "anchor_camera_locators": candidate.event.get("video_locators", []),
        "frame_locator_resolution": "RENDERER_MUST_LOOK_UP_EACH_REQUESTED_FRAME_FROM_FROZEN_SOURCE_METADATA",
        "camera_delivery": {
            "camera_native_only": True,
            "decoded_by_selector": False,
            "allow_contact_sheet_in_actor_input": False,
            "include_footer": False,
            "forbidden_footer_fields": ["outcome", "success", "failure", "recovery", "label", "evidence", "review"],
        },
    }


def queue_job(candidate: Candidate, *, queue_kind: str, selection_phase: str, policy_sha256: str,
              windows: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    identifier = job_id(queue_kind=queue_kind, event_id=candidate.event_id, policy_sha256=policy_sha256)
    request = render_request(candidate, request_id=identifier, windows=windows)
    source = require_mapping(candidate.event["source"], "event.source")
    job = {
        "schema_version": QUEUE_SCHEMA,
        "job_id": identifier,
        "queue_kind": queue_kind,
        "status": STATUS,
        "training_eligible": False,
        "source_kind": "LOGGED_EXPERT_DEMONSTRATION_METADATA_CANDIDATE",
        "candidate_heuristics_are_not_truth": True,
        "event_id": candidate.event_id,
        "source_group_id": candidate.source_group_id,
        "immutable_split": "train",
        "usage_role": candidate.event["usage_role"],
        "source_identity": {
            "source_release_manifest_sha256": source["source_release_manifest_sha256"],
            "source_annotation_sha256": source["source_annotation_sha256"],
            "task_index": candidate.task_id,
            "task_instance_id": candidate.task_instance_id,
            "raw_episode_id": source["raw_episode_id"],
            "episode_index": candidate.episode_index,
        },
        "event_interval": {"start_frame": candidate.interval_start, "end_frame": candidate.interval_end},
        "candidate_signals": list(candidate.candidate_signals),
        "selection_stratum": candidate.selection_stratum,
        "selection_phase": selection_phase,
        "repeat_same_skill_target_distinct_episode_count": candidate.repeat_attempt_count,
        "annotation_boundary": {"before": candidate.boundary_before, "after": candidate.boundary_after},
        "action_heuristic_status": "DISABLED_NO_TRUSTED_EXECUTED_ACTION_PROOF_IN_METADATA_QUEUE",
        "long_annotated_interval_candidate": candidate.long_interval,
        "skill_ids": list(candidate.skill_ids),
        "temporal_windows": windows,
        "current_actor_evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "review_constraints": {
            "current_outcome_is_not_labeled": True,
            "actor_may_only_use_actor_available_window": True,
            "offline_after_frames_are_only_for_temporal_BC_quality_review": True,
            "segment_end_gripper_close_timeout_or_model_report_are_not_truth": True,
            "no_corrective_action_or_recovery_supervision_is_emitted": True,
        },
        "camera_native_render_request_id": identifier,
    }
    return job, request


def validate_queue_job(job: Any, request: Any, *, expected_job: Mapping[str, Any],
                       expected_request: Mapping[str, Any]) -> None:
    """Require complete, re-derived candidate-only job and render schemas."""
    job = require_mapping(job, "queue job")
    request = require_mapping(request, "camera-native render request")
    if set(job) != QUEUE_JOB_FIELDS:
        raise ValueError("queue job schema has missing or unrecognized fields")
    if set(request) != RENDER_REQUEST_FIELDS:
        raise ValueError("camera-native render request schema has missing or unrecognized fields")
    if (job.get("status") != STATUS or job.get("training_eligible") is not False or
            job.get("candidate_heuristics_are_not_truth") is not True or
            job.get("current_actor_evidence") != {"kind": "MISSING", "evidence_end_frame": None,
                                                   "available_frame": None}):
        raise ValueError("queue job crossed the candidate-only safety boundary")
    if canonical_json(job) != canonical_json(expected_job):
        raise ValueError("queue job differs from the canonical candidate/policy reconstruction")
    if canonical_json(request) != canonical_json(expected_request):
        raise ValueError("render request differs from the canonical candidate/policy reconstruction")


def validate_queue_payloads(student_jobs: Sequence[Any], calibration_jobs: Sequence[Any], requests: Sequence[Any], *,
                            expected_student_jobs: Sequence[Mapping[str, Any]],
                            expected_calibration_jobs: Sequence[Mapping[str, Any]],
                            expected_requests: Sequence[Mapping[str, Any]]) -> None:
    """Compare every serialized payload to a fresh canonical reconstruction."""
    if len(student_jobs) != len(expected_student_jobs) or len(calibration_jobs) != len(expected_calibration_jobs):
        raise ValueError("queue job count differs from canonical reconstruction")
    if len(requests) != len(expected_requests):
        raise ValueError("render request count differs from canonical reconstruction")
    for actual, expected in zip(requests, expected_requests):
        actual = require_mapping(actual, "render request")
        if set(actual) != RENDER_REQUEST_FIELDS or canonical_json(actual) != canonical_json(expected):
            raise ValueError("render request differs from the canonical candidate/policy reconstruction")
    actual_requests = {require_mapping(request, "render request").get("request_id"): request for request in requests}
    expected_requests_by_id = {request["request_id"]: request for request in expected_requests}
    if (len(actual_requests) != len(requests) or len(expected_requests_by_id) != len(expected_requests) or
            set(actual_requests) != set(expected_requests_by_id)):
        raise ValueError("queue render request IDs differ from canonical reconstruction")
    for actual_jobs, expected_jobs in ((student_jobs, expected_student_jobs),
                                       (calibration_jobs, expected_calibration_jobs)):
        for actual, expected in zip(actual_jobs, expected_jobs):
            actual = require_mapping(actual, "queue job")
            identifier = actual.get("job_id")
            if identifier != expected["job_id"]:
                raise ValueError("queue job order or ID differs from canonical reconstruction")
            validate_queue_job(actual, actual_requests.get(identifier), expected_job=expected,
                               expected_request=expected_requests_by_id[identifier])


def count_queue(pool: Sequence[Candidate], selected: Sequence[tuple[Candidate, str]], *, budget: int,
                near_duplicates: Mapping[str, Any], coverage_expectations: Mapping[str, Any],
                eligible_pool_retained: bool) -> dict[str, Any]:
    selected_candidates = [candidate for candidate, _ in selected]
    def per_task(items: Sequence[Candidate]) -> dict[str, int]:
        return {str(task): count for task, count in sorted(Counter(candidate.task_id for candidate in items).items())}
    def per_skill(items: Sequence[Candidate]) -> dict[str, int]:
        return {str(skill): count for skill, count in sorted(Counter(
            skill for candidate in items for skill in candidate.skill_ids).items())}
    def per_raw_source_verb(items: Sequence[Candidate]) -> dict[str, int]:
        return dict(sorted(Counter(verb for candidate in items for verb in candidate.raw_source_verbs).items()))
    def per_episode(items: Sequence[Candidate]) -> dict[str, int]:
        return dict(sorted(Counter(
            f"{candidate.source_group_id}:{candidate.episode_index}" for candidate in items).items()))
    pool_tasks = {candidate.task_id for candidate in pool}
    queued_tasks = {candidate.task_id for candidate in selected_candidates}
    pool_skills = {skill for candidate in pool for skill in candidate.skill_ids}
    queued_skills = {skill for candidate in selected_candidates for skill in candidate.skill_ids}
    expected_tasks = set(coverage_expectations["expected_task_ids"])
    expected_vocabulary = coverage_expectations["expected_skill_vocabulary"]
    expected_skills = {skill["skill_id"] for skill in expected_vocabulary}
    required_pairs = coverage_expectations["required_task_skill_pairs"]
    selected_pairs = {(candidate.task_id, skill) for candidate in selected_candidates for skill in candidate.skill_ids}
    required_pair_coverage = (None if required_pairs is None else {
        "declared_pairs": len(required_pairs),
        "queued_pairs": sum((pair["task_index"], pair["skill_id"]) in selected_pairs for pair in required_pairs),
        "missing_pairs": [pair for pair in required_pairs
                          if (pair["task_index"], pair["skill_id"]) not in selected_pairs],
    })
    return {
        "budget": budget,
        "pool_events": len(pool),
        "queued_events": len(selected_candidates),
        "per_task": per_task(selected_candidates),
        "per_skill_id": per_skill(selected_candidates),
        "per_raw_source_verb_diagnostic": per_raw_source_verb(selected_candidates),
        "per_episode": per_episode(selected_candidates),
        "per_selection_stratum": dict(sorted(Counter(
            candidate.selection_stratum for candidate in selected_candidates).items())),
        "per_selection_phase": dict(sorted(Counter(phase for _, phase in selected).items())),
        "near_duplicate": dict(near_duplicates),
        "coverage": {
            "expected_task_ids": sorted(expected_tasks),
            "expected_task_count": len(expected_tasks),
            "eligible_pool_coverage_status": (
                "EVALUATED_RETAINED_ROLE" if eligible_pool_retained else "NOT_EVALUATED_ZERO_OUTPUT_BUDGET"
            ),
            "tasks_present_in_eligible_pool": sorted(pool_tasks) if eligible_pool_retained else None,
            "tasks_queued": sorted(queued_tasks),
            "tasks_present_but_not_queued": sorted(pool_tasks - queued_tasks) if eligible_pool_retained else None,
            "missing_expected_task_ids_from_pool": sorted(expected_tasks - pool_tasks) if eligible_pool_retained else None,
            "unexpected_task_ids_in_pool": sorted(pool_tasks - expected_tasks) if eligible_pool_retained else None,
            "expected_skill_vocabulary": expected_vocabulary,
            "expected_skill_ids": sorted(expected_skills),
            "expected_skill_count": len(expected_skills),
            "skill_ids_present_in_eligible_pool": sorted(pool_skills) if eligible_pool_retained else None,
            "skill_ids_queued": sorted(queued_skills),
            "skill_ids_present_but_not_queued": sorted(pool_skills - queued_skills) if eligible_pool_retained else None,
            "skill_count_seen": len(pool_skills) if eligible_pool_retained else None,
            "missing_expected_skill_ids_from_pool": sorted(expected_skills - pool_skills) if eligible_pool_retained else None,
            "unexpected_source_skill_ids_in_pool": sorted(pool_skills - expected_skills) if eligible_pool_retained else None,
            "global_vocabulary_complete": (
                not (expected_tasks - pool_tasks) and not (expected_skills - pool_skills) if eligible_pool_retained else None
            ),
            "raw_source_verbs_diagnostic": (
                sorted({verb for candidate in pool for verb in candidate.raw_source_verbs}) if eligible_pool_retained else None
            ),
            "required_task_skill_pairs": required_pairs,
            "required_task_skill_pair_queue_coverage": required_pair_coverage,
            "required_task_skill_pair_queue_status": (
                "NOT_DECLARED_DESCRIPTIVE_ONLY" if required_pairs is None
                else ("COMPLETE" if not required_pair_coverage["missing_pairs"] else "INCOMPLETE")
            ),
        },
    }


def policy_from_args(args: argparse.Namespace, *, index_manifest_sha256: str,
                     inventory_seal_sha256: str, protocol_sha256: str,
                     coverage_expectations_sha256: str) -> dict[str, Any]:
    integer_bounds = {
        "candidate_budget": 0, "calibration_budget": 0, "max_per_episode": 1,
        "max_per_source_group": 1, "min_separation_frames": 0, "boundary_gap_frames": 0,
        "long_interval_frames": 1, "actor_history_frames": 0, "review_before_frames": 0,
        "review_after_frames": 0, "review_sample_stride_frames": 1, "max_retained_candidates": 1,
    }
    for name, minimum in integer_bounds.items():
        require_int(getattr(args, name), f"--{name.replace('_', '-')}", minimum=minimum)
    if args.max_retained_candidates > ABSOLUTE_MAX_RETAINED_CANDIDATES:
        raise ValueError("--max-retained-candidates exceeds the hard bounded-memory queue limit")
    if (not isinstance(args.seed, str) or not args.seed or type(args.normal_control_fraction) not in (int, float) or
            isinstance(args.normal_control_fraction, bool) or not math.isfinite(args.normal_control_fraction) or
            not 0.0 <= args.normal_control_fraction <= 1.0):
        raise ValueError("--seed and --normal-control-fraction are invalid")
    core = {
        "schema_version": QUEUE_SCHEMA,
        "algorithm": "p107-deterministic-metadata-candidate-selection-v1",
        "seed": args.seed,
        "frozen_source_release_manifest_sha256": args.expected_source_release_manifest_sha256,
        "sealed_index_manifest_sha256": index_manifest_sha256,
        "inventory_seal_sha256": inventory_seal_sha256,
        "canonical_protocol_sha256": protocol_sha256,
        "coverage_expectations_sha256": coverage_expectations_sha256,
        "candidate_budget": args.candidate_budget,
        "calibration_budget": args.calibration_budget,
        "max_per_episode": args.max_per_episode,
        "max_per_source_group": args.max_per_source_group,
        "min_separation_frames": args.min_separation_frames,
        "boundary_gap_frames": args.boundary_gap_frames,
        "long_interval_frames": args.long_interval_frames,
        "normal_control_fraction": args.normal_control_fraction,
        "actor_history_frames": args.actor_history_frames,
        "review_before_frames": args.review_before_frames,
        "review_after_frames": args.review_after_frames,
        "review_sample_stride_frames": args.review_sample_stride_frames,
        "max_retained_candidates": args.max_retained_candidates,
        "no_rgb_decode_by_selector": True,
        "action_sequence_heuristic": "DISABLED_NO_TRUSTED_EXECUTED_ACTION_PROOF",
        "supported_source_kinds": ["LOGGED_EXPERT_DEMONSTRATION_METADATA_CANDIDATE"],
        "all_jobs_status": STATUS,
    }
    return {**core, "policy_sha256": canonical_sha256(core)}


def build_queue_payloads(candidates: Sequence[Candidate], groups: Mapping[str, SourceGroup], *,
                         policy: Mapping[str, Any], coverage_expectations: Mapping[str, Any],
                         excluded_roles: Mapping[str, int],
                         excluded_unrecognized_vocabulary: Mapping[str, int],
                         retained_usage_roles: set[str]) -> dict[str, Any]:
    """Reconstruct every derived payload solely from validated input and policy."""
    expected_skill_ids = [skill["skill_id"] for skill in coverage_expectations["expected_skill_vocabulary"]]
    student_pool = [candidate for candidate in candidates if candidate.event["usage_role"] == "student_candidate"]
    calibration_pool = [candidate for candidate in candidates if candidate.event["usage_role"] == "annotation_calibration"]
    private_phase_student = [candidate for candidate in student_pool
                             if _private_phase_lineage(candidate) is not None]
    if private_phase_student:
        if len(private_phase_student) != len(student_pool):
            raise ValueError("private student phase events cannot mix with generic student candidates")
        if policy["candidate_budget"] < 2:
            raise ValueError("private student phase queue requires budget for ENTRY and TERMINAL")
        phase_names = {candidate.event["phase_lineage"]["observation_phase"] for candidate in private_phase_student}
        parent_ids = {candidate.event["phase_lineage"]["parent_event_id"] for candidate in private_phase_student}
        if (len(private_phase_student) != 2 or phase_names != {"ENTRY", "TERMINAL"} or len(parent_ids) != 1):
            raise ValueError("private student phase queue must contain exactly one ENTRY and one TERMINAL parent pair")
        private_phase_student.sort(key=lambda candidate: (candidate.event["phase_lineage"]["observation_phase"],
                                                           candidate.event_id))
        selected_student = [(candidate, "PRIVATE_GOAL_STATE_PHASE") for candidate in private_phase_student]
        student_diagnostics = {
            "selected": len(selected_student), "candidate_count": len(student_pool),
            "same_episode_cap_rejected": 0, "adjacent_window_cap_rejected": 0,
            "private_goal_state_phase_selection": True,
        }
    else:
        selected_student, student_diagnostics = select_diverse(
            student_pool, budget=policy["candidate_budget"], seed=policy["seed"],
            max_per_episode=policy["max_per_episode"], min_separation_frames=policy["min_separation_frames"],
            max_per_source_group=policy["max_per_source_group"], normal_control_fraction=policy["normal_control_fraction"],
            priority_skill_ids=expected_skill_ids)
    selected_calibration, calibration_diagnostics = select_diverse(
        calibration_pool, budget=policy["calibration_budget"], seed=policy["seed"] + ":calibration",
        max_per_episode=policy["max_per_episode"], min_separation_frames=policy["min_separation_frames"],
        max_per_source_group=policy["max_per_source_group"], normal_control_fraction=policy["normal_control_fraction"],
        priority_skill_ids=expected_skill_ids)

    def windows(candidate: Candidate) -> dict[str, Any]:
        return window_spec(candidate, actor_history_frames=policy["actor_history_frames"],
                           review_before_frames=policy["review_before_frames"],
                           review_after_frames=policy["review_after_frames"],
                           sample_stride_frames=policy["review_sample_stride_frames"])

    student_jobs: list[dict[str, Any]] = []
    calibration_jobs: list[dict[str, Any]] = []
    render_requests: list[dict[str, Any]] = []
    for candidate, phase in selected_student:
        phase_windows = _private_phase_window_spec(candidate)
        job, request = queue_job(candidate, queue_kind="STUDENT_CANDIDATE_REVIEW", selection_phase=phase,
                                 policy_sha256=policy["policy_sha256"],
                                 windows=phase_windows if phase_windows is not None else windows(candidate))
        student_jobs.append(job)
        render_requests.append(request)
    for candidate, phase in selected_calibration:
        job, request = queue_job(candidate, queue_kind="ANNOTATION_CALIBRATION", selection_phase=phase,
                                 policy_sha256=policy["policy_sha256"], windows=windows(candidate))
        calibration_jobs.append(job)
        render_requests.append(request)
    if len({request["request_id"] for request in render_requests}) != len(render_requests):
        raise ValueError("queue construction produced duplicate render request IDs")
    validate_queue_payloads(student_jobs, calibration_jobs, render_requests,
                            expected_student_jobs=student_jobs, expected_calibration_jobs=calibration_jobs,
                            expected_requests=render_requests)
    counts = {
        "schema_version": COUNTS_SCHEMA,
        "status": STATUS,
        "training_eligible": False,
        "source_role_inventory": dict(sorted(Counter(group.usage_role for group in groups.values()).items())),
        "events_excluded_by_role": dict(sorted(excluded_roles.items())),
        "events_excluded_by_unrecognized_official_vocabulary": dict(
            sorted(excluded_unrecognized_vocabulary.items())),
        "student_candidate_queue": count_queue(
            student_pool, selected_student, budget=policy["candidate_budget"], near_duplicates=student_diagnostics,
            coverage_expectations=coverage_expectations,
            eligible_pool_retained="student_candidate" in retained_usage_roles),
        "annotation_calibration_queue": count_queue(
            calibration_pool, selected_calibration, budget=policy["calibration_budget"],
            near_duplicates=calibration_diagnostics, coverage_expectations=coverage_expectations,
            eligible_pool_retained="annotation_calibration" in retained_usage_roles),
        "existing_local_single_frame_pilot": {
            "count": 24,
            "scope": "historical first-five-task auxiliary visual-relation calibration only",
            "parent_reviewed_after_revision": True,
            "status": "CANDIDATE_ONLY_NOT_TRAINABLE_RECOVERY",
            "included_in_this_queue": False,
        },
    }
    if set(counts) != COUNTS_FIELDS:
        raise AssertionError("internal counts schema drift")
    return {
        "student_jobs": student_jobs,
        "calibration_jobs": calibration_jobs,
        "render_requests": render_requests,
        "counts": counts,
    }


def design_handoff() -> dict[str, str]:
    return {
        "logged_successful_demonstrations": (
            "Repeated skills and annotation boundaries only locate review windows. A reviewer may inspect a "
            "temporal sequence and, only when 23D actions are from the same source clock and actually executed, "
            "propose a logged-demonstration corrective-action candidate. Repetition never proves a failure or recovery."
        ),
        "requires_human_or_model_temporal_review": (
            "Future frames can support offline BC-quality review, but labels for a current decision may use only "
            "evidence available no later than that decision anchor. Segment end, gripper closure, timeout, and model text "
            "are not result truth."
        ),
        "requires_new_simulator_or_live_branch_evidence": (
            "Genuine recovery claims require separately verified initial deviation, same-state actual execution, 23D action "
            "receipt, postcondition/stability or future-feasibility evidence, and restore/branch identity. None is available here."
        ),
        "future_source_kinds": (
            "This queue contains only logged expert-demonstration metadata candidates. A future DART/online perturbation "
            "producer must introduce its own sealed source-kind policy and evidence contract; this selector emits no DART labels."
        ),
    }


def queue_manifest(*, source_release_manifest_sha256: str, index_manifest_sha256: str,
                   inventory_seal_sha256: str, protocol_sha256: str, coverage_expectations_sha256: str,
                   index_manifest: Mapping[str, Any], policy: Mapping[str, Any], files: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "METADATA_CANDIDATES_READY_FOR_RENDERER_HANDSHAKE",
        "training_eligible": False,
        "all_jobs_status": STATUS,
        "source_release_manifest_sha256": source_release_manifest_sha256,
        "index_manifest_sha256": index_manifest_sha256,
        "inventory_seal_sha256": inventory_seal_sha256,
        "canonical_protocol_sha256": protocol_sha256,
        "coverage_expectations_sha256": coverage_expectations_sha256,
        "index_event_file_sha256": index_manifest["files"]["event_candidates.jsonl"]["sha256"],
        "index_source_group_file_sha256": index_manifest["files"]["source_groups.jsonl"]["sha256"],
        "policy": policy,
        "files": files,
        "renderer_handshake": {
            "status": "PENDING_INTERFACE_OWNER_CONFIRMATION",
            "schema_version": RENDER_REQUEST_SCHEMA,
            "selector_decoded_rgb": False,
            "temporal_sequence_required": True,
            "no_footer_truth": True,
        },
        "design_handoff": design_handoff(),
    }
    if set(result) != MANIFEST_FIELDS:
        raise AssertionError("internal manifest schema drift")
    return result


def build_queue(index: Path, output: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    """Build one immutable candidate/calibration queue set, never append."""
    index, output = Path(index), Path(output)
    if output.exists():
        raise FileExistsError("queue output already exists; use --resume only to verify it")
    if args.candidate_budget < 0 or args.calibration_budget < 0:
        raise ValueError("candidate and calibration budgets must be nonnegative")
    canonical_protocol = load_canonical_protocol(args.protocol_path, expected_sha256=args.expected_protocol_sha256)
    manifest, event_rows, groups = validate_index(
        index, expected_source_manifest_sha256=args.expected_source_release_manifest_sha256,
        canonical_protocol=canonical_protocol, expected_protocol_sha256=args.expected_protocol_sha256)
    inventory_seal, inventory_seal_sha256 = validate_inventory_seal(
        index, expected_inventory_seal_sha256=args.expected_inventory_seal_sha256)
    coverage_expectations = read_coverage_expectations(
        args.coverage_expectations,
        expected_sha256=inventory_seal["coverage_expectations_sha256"])
    if manifest.get("coverage_expectations_sha256") != canonical_sha256(coverage_expectations):
        raise ValueError("index manifest coverage expectations differ from the inventory-sealed contract")
    forbid_output_inside(output, index)
    index_manifest_sha256 = sha256_file(index / "manifest.json")
    policy = policy_from_args(args, index_manifest_sha256=index_manifest_sha256,
                              inventory_seal_sha256=inventory_seal_sha256,
                              protocol_sha256=args.expected_protocol_sha256,
                              coverage_expectations_sha256=canonical_sha256(coverage_expectations))
    retained_usage_roles = {
        role for role, budget in (("student_candidate", policy["candidate_budget"]),
                                  ("annotation_calibration", policy["calibration_budget"])) if budget > 0
    }
    candidates, excluded_roles, excluded_unrecognized_vocabulary = parse_candidates(
        event_rows, groups, expected_source_manifest_sha256=args.expected_source_release_manifest_sha256,
        canonical_protocol=canonical_protocol, expected_task_ids=coverage_expectations["expected_task_ids"],
        expected_skill_ids=[skill["skill_id"] for skill in coverage_expectations["expected_skill_vocabulary"]],
        retained_usage_roles=retained_usage_roles, max_retained_candidates=policy["max_retained_candidates"],
        expected_event_receipt=manifest["files"]["event_candidates.jsonl"],
        boundary_gap_frames=args.boundary_gap_frames,
        long_interval_frames=args.long_interval_frames)
    payloads = build_queue_payloads(candidates, groups, policy=policy, coverage_expectations=coverage_expectations,
                                    excluded_roles=excluded_roles,
                                    excluded_unrecognized_vocabulary=excluded_unrecognized_vocabulary,
                                    retained_usage_roles=retained_usage_roles)
    student_jobs = payloads["student_jobs"]
    calibration_jobs = payloads["calibration_jobs"]
    render_requests = payloads["render_requests"]
    counts = payloads["counts"]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        files = {
            "student_candidate_queue.jsonl": write_jsonl(staging / "student_candidate_queue.jsonl", student_jobs),
            "annotation_calibration_queue.jsonl": write_jsonl(staging / "annotation_calibration_queue.jsonl", calibration_jobs),
            "camera_native_render_requests.jsonl": write_jsonl(staging / "camera_native_render_requests.jsonl", render_requests),
        }
        (staging / "counts.json").write_bytes(canonical_json(counts).encode("utf-8") + b"\n")
        files["counts.json"] = {"sha256": sha256_file(staging / "counts.json"), "rows": 1,
                                "bytes": (staging / "counts.json").stat().st_size}
        result = queue_manifest(
            source_release_manifest_sha256=args.expected_source_release_manifest_sha256,
            index_manifest_sha256=index_manifest_sha256, inventory_seal_sha256=inventory_seal_sha256,
            protocol_sha256=args.expected_protocol_sha256,
            coverage_expectations_sha256=canonical_sha256(coverage_expectations), index_manifest=manifest,
            policy=policy, files=files)
        (staging / "manifest.json").write_bytes(canonical_json(result).encode("utf-8") + b"\n")
        queue_seal = {
            "schema_version": QUEUE_SEAL_SCHEMA,
            "queue_manifest_sha256": sha256_file(staging / "manifest.json"),
            "inventory_seal_sha256": inventory_seal_sha256,
            "source_release_manifest_sha256": args.expected_source_release_manifest_sha256,
            "canonical_protocol_sha256": args.expected_protocol_sha256,
            "coverage_expectations_sha256": canonical_sha256(coverage_expectations),
            "policy_sha256": policy["policy_sha256"],
            "expected_payload_files": sorted(QUEUE_PAYLOAD_FILES),
            "payload_files": files,
        }
        (staging / "queue_seal.json").write_bytes(canonical_json(queue_seal).encode("utf-8") + b"\n")
        queue_seal_sha256 = sha256_file(staging / "queue_seal.json")
        os.rename(staging, output)
        return {**result, "queue_seal_sha256": queue_seal_sha256}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_queue_seal(output: Path, manifest: Mapping[str, Any], *, expected_queue_seal_sha256: str,
                        expected_inventory_seal_sha256: str, expected_source_manifest_sha256: str,
                        expected_protocol_sha256: str, expected_coverage_expectations_sha256: str) -> str:
    """Require a separately recorded queue seal before trusting any queue receipt."""
    if not is_sha256(expected_queue_seal_sha256):
        raise ValueError("--expected-queue-seal-sha256 must be a lowercase SHA-256")
    seal_path = Path(output) / "queue_seal.json"
    seal = read_json(seal_path)
    if not isinstance(seal, Mapping) or sha256_file(seal_path) != expected_queue_seal_sha256:
        raise ValueError("queue seal does not match the externally recorded handoff")
    required = {
        "schema_version", "queue_manifest_sha256", "inventory_seal_sha256", "source_release_manifest_sha256",
        "canonical_protocol_sha256", "coverage_expectations_sha256", "policy_sha256",
        "expected_payload_files", "payload_files",
    }
    if set(seal) != required or seal.get("schema_version") != QUEUE_SEAL_SCHEMA:
        raise ValueError("queue seal schema/field set is not the fixed P107 contract")
    manifest_policy = require_mapping(manifest.get("policy"), "queue manifest policy")
    if (seal.get("queue_manifest_sha256") != sha256_file(Path(output) / "manifest.json") or
            seal.get("inventory_seal_sha256") != expected_inventory_seal_sha256 or
            seal.get("source_release_manifest_sha256") != expected_source_manifest_sha256 or
            seal.get("canonical_protocol_sha256") != expected_protocol_sha256 or
            seal.get("coverage_expectations_sha256") != expected_coverage_expectations_sha256 or
            seal.get("policy_sha256") != manifest_policy.get("policy_sha256")):
        raise ValueError("queue seal does not bind the expected external authority roots")
    expected_files = sorted(QUEUE_PAYLOAD_FILES)
    if seal.get("expected_payload_files") != expected_files:
        raise ValueError("queue seal payload file set/order changed")
    payloads = require_mapping(seal.get("payload_files"), "queue seal payload files")
    manifest_files = require_mapping(manifest.get("files"), "queue manifest files")
    if set(payloads) != QUEUE_PAYLOAD_FILES or payloads != manifest_files:
        raise ValueError("queue seal receipts do not bind the exact queue manifest payloads")
    for name, receipt in payloads.items():
        receipt = require_mapping(receipt, f"queue seal receipt {name}")
        if set(receipt) != {"sha256", "rows", "bytes"} or not is_sha256(receipt.get("sha256")):
            raise ValueError("queue seal payload receipt malformed")
        path = Path(output) / str(name)
        file_receipt(path, expected=receipt)
        if name.endswith(".jsonl") and receipt.get("rows") != len(read_jsonl(path)):
            raise ValueError(f"sealed queue row count changed: {name}")
    return expected_queue_seal_sha256


def validate_queue_policy(policy: Any, *, expected_policy: Mapping[str, Any]) -> dict[str, Any]:
    policy = dict(require_mapping(policy, "queue policy"))
    if set(policy) != POLICY_CORE_FIELDS | {"policy_sha256"}:
        raise ValueError("queue policy schema has missing or unrecognized fields")
    actual = policy.pop("policy_sha256")
    if not is_sha256(actual) or canonical_sha256(policy) != actual:
        raise ValueError("queue policy hash is not canonical")
    reconstructed = {**policy, "policy_sha256": actual}
    if canonical_json(reconstructed) != canonical_json(expected_policy):
        raise ValueError("queue policy differs from the externally invoked sealed selection configuration")
    return reconstructed


def resume_queue(index: Path, output: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    index, output = Path(index), Path(output)
    canonical_protocol = load_canonical_protocol(args.protocol_path, expected_sha256=args.expected_protocol_sha256)
    index_manifest, event_rows, groups = validate_index(
        index, expected_source_manifest_sha256=args.expected_source_release_manifest_sha256,
        canonical_protocol=canonical_protocol, expected_protocol_sha256=args.expected_protocol_sha256)
    inventory_seal, inventory_seal_sha256 = validate_inventory_seal(
        index, expected_inventory_seal_sha256=args.expected_inventory_seal_sha256)
    coverage_expectations = read_coverage_expectations(
        args.coverage_expectations, expected_sha256=inventory_seal["coverage_expectations_sha256"])
    coverage_sha256 = canonical_sha256(coverage_expectations)
    if index_manifest.get("coverage_expectations_sha256") != coverage_sha256:
        raise ValueError("index manifest coverage expectations changed")
    index_manifest_sha256 = sha256_file(index / "manifest.json")
    expected_policy = policy_from_args(
        args, index_manifest_sha256=index_manifest_sha256, inventory_seal_sha256=inventory_seal_sha256,
        protocol_sha256=args.expected_protocol_sha256, coverage_expectations_sha256=coverage_sha256)
    manifest = read_json(output / "manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("--resume requires a sealed P107 annotation queue")
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("queue manifest schema has missing or unrecognized fields")
    policy = validate_queue_policy(manifest.get("policy"), expected_policy=expected_policy)
    queue_seal_sha256 = validate_queue_seal(
        output, manifest, expected_queue_seal_sha256=args.expected_queue_seal_sha256,
        expected_inventory_seal_sha256=inventory_seal_sha256,
        expected_source_manifest_sha256=args.expected_source_release_manifest_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_coverage_expectations_sha256=coverage_sha256)
    retained_usage_roles = {
        role for role, budget in (("student_candidate", policy["candidate_budget"]),
                                  ("annotation_calibration", policy["calibration_budget"])) if budget > 0
    }
    candidates, excluded_roles, excluded_unrecognized_vocabulary = parse_candidates(
        event_rows, groups, expected_source_manifest_sha256=args.expected_source_release_manifest_sha256,
        canonical_protocol=canonical_protocol, expected_task_ids=coverage_expectations["expected_task_ids"],
        expected_skill_ids=[skill["skill_id"] for skill in coverage_expectations["expected_skill_vocabulary"]],
        retained_usage_roles=retained_usage_roles, max_retained_candidates=policy["max_retained_candidates"],
        expected_event_receipt=index_manifest["files"]["event_candidates.jsonl"],
        boundary_gap_frames=policy["boundary_gap_frames"],
        long_interval_frames=policy["long_interval_frames"])
    payloads = build_queue_payloads(candidates, groups, policy=policy, coverage_expectations=coverage_expectations,
                                    excluded_roles=excluded_roles,
                                    excluded_unrecognized_vocabulary=excluded_unrecognized_vocabulary,
                                    retained_usage_roles=retained_usage_roles)
    expected_manifest = queue_manifest(
        source_release_manifest_sha256=args.expected_source_release_manifest_sha256,
        index_manifest_sha256=index_manifest_sha256, inventory_seal_sha256=inventory_seal_sha256,
        protocol_sha256=args.expected_protocol_sha256, coverage_expectations_sha256=coverage_sha256,
        index_manifest=index_manifest, policy=policy, files=require_mapping(manifest["files"], "queue manifest files"))
    if canonical_json(manifest) != canonical_json(expected_manifest):
        raise ValueError("queue manifest differs from the canonical candidate-only reconstruction")
    validate_queue_payloads(
        read_jsonl(output / "student_candidate_queue.jsonl"), read_jsonl(output / "annotation_calibration_queue.jsonl"),
        read_jsonl(output / "camera_native_render_requests.jsonl"), expected_student_jobs=payloads["student_jobs"],
        expected_calibration_jobs=payloads["calibration_jobs"], expected_requests=payloads["render_requests"])
    actual_counts = read_json(output / "counts.json")
    if not isinstance(actual_counts, Mapping) or set(actual_counts) != COUNTS_FIELDS:
        raise ValueError("queue counts schema has missing or unrecognized fields")
    if canonical_json(actual_counts) != canonical_json(payloads["counts"]):
        raise ValueError("queue counts differ from the canonical candidate/policy reconstruction")
    return {"status": "RESUME_VALIDATED", "manifest": manifest, "queue_seal_sha256": queue_seal_sha256}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", type=Path, required=True, help="sealed P107 event index from the data owner")
    result.add_argument("--output", type=Path, required=True, help="new external queue sidecar directory")
    result.add_argument("--expected-source-release-manifest-sha256", default=DEFAULT_SOURCE_MANIFEST_SHA256)
    result.add_argument("--expected-inventory-seal-sha256", required=True,
                        help="exact externally reviewed inventory_seal.json SHA-256")
    result.add_argument("--coverage-expectations", type=Path, required=True,
                        help="exact official task/skill coverage contract sealed by inventory_seal.json")
    result.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH,
                        help="SHA-pinned data-owner protocol (defaults to the in-repository integration path)")
    result.add_argument("--expected-protocol-sha256", required=True,
                        help="exact externally reviewed canonical protocol SHA-256")
    result.add_argument("--expected-queue-seal-sha256",
                        help="required with --resume; externally recorded queue_seal.json SHA-256")
    result.add_argument("--seed", default=DEFAULT_SEED)
    result.add_argument("--candidate-budget", type=int, default=200)
    result.add_argument("--calibration-budget", type=int, default=40)
    result.add_argument("--max-per-episode", type=int, default=2)
    result.add_argument("--max-per-source-group", type=int, default=2)
    result.add_argument("--min-separation-frames", type=int, default=120)
    result.add_argument("--boundary-gap-frames", type=int, default=1)
    result.add_argument("--long-interval-frames", type=int, default=360)
    result.add_argument("--normal-control-fraction", type=float, default=0.25)
    result.add_argument("--actor-history-frames", type=int, default=60)
    result.add_argument("--review-before-frames", type=int, default=60)
    result.add_argument("--review-after-frames", type=int, default=60)
    result.add_argument("--review-sample-stride-frames", type=int, default=15)
    result.add_argument("--max-retained-candidates", type=int, default=ABSOLUTE_MAX_RETAINED_CANDIDATES,
                        help="hard memory bound after streaming validation; narrow the approved role/group scope if exceeded")
    result.add_argument("--resume", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.resume and args.expected_queue_seal_sha256 is None:
        raise ValueError("--resume requires --expected-queue-seal-sha256 from an external handoff")
    result = (resume_queue(args.index, args.output, args=args)
              if args.resume else build_queue(args.index, args.output, args=args))
    print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
