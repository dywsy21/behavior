#!/usr/bin/env python3
"""Build role-separated, answer-independent P107 coverage query candidates.

This producer is deliberately separate from the fixed phase40 producer.  It
joins a sealed coverage selector row to a sealed event-index record by
``event_id``, reuses the phase producer's *current visible relation* templates,
and emits no answer, outcome, recovery, action, RGB evidence, or post-label
view identity.  TRAIN and EVAL are different output contracts: EVAL is a
diagnostic native-packet review sidecar and is never a TRAIN registry/queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_memlite_visual_relation_queries as templates  # noqa: E402
import select_memlite_event_annotation_queue as base  # noqa: E402


COVERAGE_QUERY_SCHEMA = "p107.coverage.visual_relation_query.v1"
COVERAGE_REGISTRY_SCHEMA = "p107.coverage.visual_relation_query_registry.v1"
COVERAGE_EVAL_REVIEW_SCHEMA = "p107.coverage.evaluation_native_review.v1"
COVERAGE_SOURCE_PIN_SCHEMA = "p107.coverage.source-pin.v1"
COVERAGE_MANIFEST_SCHEMA = "p107.coverage.visual_relation_query_manifest.v1"
UNSUPPORTED_SCHEMA = "p107.coverage.unsupported_goal_query.v1"
SELECTOR_SCHEMA = "p107-diagnostic-coverage-selection-v1"
QUEUE_SCHEMA = "p107-metadata-annotation-queue-v1"
PROTOCOL_SHA256_DEFAULT = "7f4f9fbf18fb4ba6ec97a304f0d787ebadb4f84c7184dd041c99027ad84aab0c"
SOURCE_RELEASE_SHA256_DEFAULT = "90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23"
INVENTORY_SHA256_DEFAULT = "7ed1b2c5cbb50c8af042a2b9dca8529a6d133de5fc0fbe224601854235c31479"
COVERAGE_EXPECTATIONS_SHA256_DEFAULT = "39ccfb79420010bfc32e2f00d66cae255340a997b20026dc970fdffee412b3c7"

ROLE_CONFIG: dict[str, dict[str, str]] = {
    "train": {
        "usage_role": "annotation_calibration",
        "immutable_split": "train",
        "job_filename": "annotation_calibration_queue.jsonl",
        "request_filename": "camera_native_render_requests.jsonl",
        "job_kind": "ANNOTATION_CALIBRATION_COVERAGE",
    },
    "eval": {
        "usage_role": "evaluation_only",
        "immutable_split": "eval",
        "job_filename": "evaluation_only_selection.jsonl",
        "request_filename": "evaluation_only_render_requests.jsonl",
        "job_kind": "EVALUATION_ONLY_DIAGNOSTIC",
    },
}

# These are the relation branches explicitly implemented by the v5 template
# module.  A future canonical skill outside this set is quarantined rather than
# silently converted into an action-recognition question.
SUPPORTED_RELATION_VERBS = frozenset({
    "NAVIGATE", "GRASP", "PLACE_ON", "PLACE_IN", "HANDOVER", "INSERT", "RELEASE",
    "OPEN_DRAWER", "OPEN_DOOR", "CLOSE_DRAWER", "CLOSE_DOOR", "OPEN_LID", "CLOSE_LID",
    "ATTACH", "POUR", "CHOP", "WIPE_HARD", "SWEEP_SURFACE", "HANG", "PRESS",
    "TURN_ON_SWITCH", "TURN_OFF_SWITCH", "IGNITE", "PUSH", "PLACE_NEXT_TO",
    "PLACE_IN_NEXT_TO", "TURN_TO", "HOLD", "SPRAY", "PLACE_UNDER", "TIP_OVER",
    "PUSH_TRAY", "PULL_TRAY", "SWEEP_OFF", "LIFT",
})
REQUIRED_VIEWS = ("head", "left_wrist", "right_wrist")
MISSING_EVIDENCE = {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, name: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing regular file")
    return path


def _directory(path: Path, name: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be an existing regular directory")
    return path


def _sha(value: Any, name: str) -> str:
    if not base.is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return str(value)


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    return base.require_int(value, name, minimum=minimum)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    value = base.read_json(_regular(path, name))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _receipt(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "rows", "bytes"}:
        raise ValueError(f"{name} has an invalid sealed receipt")
    _sha(value.get("sha256"), f"{name}.sha256")
    if type(value.get("rows")) is not int or value["rows"] < 0:
        raise ValueError(f"{name}.rows must be a nonnegative integer")
    if type(value.get("bytes")) is not int or value["bytes"] < 0:
        raise ValueError(f"{name}.bytes must be a nonnegative integer")
    return dict(value)


def _read_verified_jsonl(path: Path, receipt: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    rows = list(base.iter_jsonl_verified(_regular(path, name), expected=_receipt(receipt, name)))
    if len(rows) != receipt["rows"]:
        raise ValueError(f"{name} row count disagrees with its sealed receipt")
    return rows


def _check_exact_missing(value: Any, name: str) -> None:
    if value != MISSING_EVIDENCE:
        raise ValueError(f"{name} must remain missing evidence")


def _validate_selector(
    selector_root: Path,
    role: str,
    *,
    expected_manifest_sha256: str,
    expected_selection_seal_sha256: str,
) -> dict[str, Any]:
    """Validate one immutable role directory before joining any event row."""

    if role not in ROLE_CONFIG:
        raise ValueError(f"unknown coverage role {role!r}")
    config = ROLE_CONFIG[role]
    root = _directory(selector_root, "selector root")
    manifest_path = _regular(root / "manifest.json", "selector manifest")
    selection_seal_path = _regular(root / "selection_seal.json", "selector selection seal")
    actual_manifest_sha256 = sha256_file(manifest_path)
    actual_selection_seal_sha256 = sha256_file(selection_seal_path)
    if actual_manifest_sha256 != _sha(expected_manifest_sha256, "expected selector manifest SHA"):
        raise ValueError("selector manifest bytes do not match the external pin")
    if actual_selection_seal_sha256 != _sha(expected_selection_seal_sha256, "expected selector selection seal SHA"):
        raise ValueError("selector selection-seal bytes do not match the external pin")

    manifest = _read_json(manifest_path, "selector manifest")
    selection_seal = _read_json(selection_seal_path, "selector selection seal")
    if manifest.get("schema_version") != SELECTOR_SCHEMA or selection_seal.get("schema_version") != SELECTOR_SCHEMA:
        raise ValueError("coverage selector uses an unsupported schema")
    if manifest.get("manifest_sha256") not in (None, actual_manifest_sha256):
        raise ValueError("selector manifest contains a conflicting self pin")
    if selection_seal.get("manifest_sha256") != actual_manifest_sha256:
        raise ValueError("selection seal does not bind the selector manifest")
    expected_usage = config["usage_role"]
    expected_split = config["immutable_split"]
    if manifest.get("selection_role") != role or manifest.get("usage_role") != expected_usage:
        raise ValueError("selector manifest has the wrong role")
    if selection_seal.get("training_eligible") is not False:
        raise ValueError("selection seal is unexpectedly training eligible")
    if manifest.get("immutable_split") != expected_split:
        raise ValueError("selector manifest immutable split does not match role")
    if manifest.get("status") != "DIAGNOSTIC_CANDIDATES_READY_FOR_INDEPENDENT_REVIEW":
        raise ValueError("selector manifest is not review-pending")
    if manifest.get("no_outcome_or_action_labels") is not True:
        raise ValueError("selector manifest does not preserve the no-label gate")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("selector manifest files are missing")
    if selection_seal.get("files") != files:
        raise ValueError("selection seal file receipts disagree with selector manifest")

    # A TRAIN queue seal is an explicit role boundary.  EVAL may not smuggle
    # one into the diagnostic path, even if its row bytes otherwise match.
    queue_seal_path = root / "queue_seal.json"
    if role == "train":
        if not queue_seal_path.is_file() or queue_seal_path.is_symlink():
            raise ValueError("TRAIN selector is missing its queue seal")
        queue_seal_sha256 = sha256_file(queue_seal_path)
        if selection_seal.get("queue_seal_sha256") != queue_seal_sha256:
            raise ValueError("TRAIN selection seal does not bind queue_seal.json")
        queue_seal = _read_json(queue_seal_path, "TRAIN queue seal")
        if queue_seal.get("schema_version") != "p107-metadata-annotation-queue-seal-v1":
            raise ValueError("TRAIN queue seal has the wrong schema")
        if queue_seal.get("queue_manifest_sha256") != actual_manifest_sha256:
            raise ValueError("TRAIN queue seal does not bind selector manifest")
    else:
        if queue_seal_path.exists():
            raise ValueError("EVAL selector must not carry a TRAIN queue seal")
        if "queue_seal_sha256" in selection_seal:
            raise ValueError("EVAL selection seal must not carry a TRAIN queue-seal pin")
        queue_seal_sha256 = None

    job_name = config["job_filename"]
    request_name = config["request_filename"]
    for filename in (job_name, request_name, "selected_rows.jsonl"):
        if filename not in files:
            raise ValueError(f"selector manifest lacks {filename}")
    job_path = _regular(root / job_name, job_name)
    request_path = _regular(root / request_name, request_name)
    selected_path = _regular(root / "selected_rows.jsonl", "selected_rows.jsonl")
    job_rows = _read_verified_jsonl(job_path, files[job_name], job_name)
    request_rows = _read_verified_jsonl(request_path, files[request_name], request_name)
    selected_rows = _read_verified_jsonl(selected_path, files["selected_rows.jsonl"], "selected_rows.jsonl")
    if not job_rows or len(job_rows) != len(request_rows) or len(job_rows) != len(selected_rows):
        raise ValueError("selector job/request/selected row counts disagree")

    expected_ids: set[str] = set()
    requests_by_event: dict[str, dict[str, Any]] = {}
    jobs_by_event: dict[str, dict[str, Any]] = {}
    selected_by_event: dict[str, dict[str, Any]] = {}
    for row in job_rows:
        event_id = _sha(row.get("event_id"), "selector job event_id")
        if event_id in jobs_by_event:
            raise ValueError("selector job contains a duplicate event_id")
        if (row.get("schema_version") != QUEUE_SCHEMA or row.get("queue_kind") != config["job_kind"] or
                row.get("usage_role") != expected_usage or row.get("immutable_split") != expected_split or
                row.get("training_eligible") is not False or row.get("status") != "CANDIDATE_MISSING_EVIDENCE"):
            raise ValueError("selector job row violates the immutable role contract")
        _check_exact_missing(row.get("current_actor_evidence"), "selector job current_actor_evidence")
        constraints = row.get("review_constraints")
        if not isinstance(constraints, Mapping) or any(value is not True for value in constraints.values()):
            raise ValueError("selector job review constraints are weakened")
        required_constraints = {
            "actor_may_only_use_actor_available_window", "current_outcome_is_not_labeled",
            "no_corrective_action_or_recovery_supervision_is_emitted",
            "offline_after_frames_are_only_for_temporal_BC_quality_review",
            "segment_end_gripper_close_timeout_or_model_report_are_not_truth",
        }
        if set(constraints) != required_constraints:
            raise ValueError("selector job review constraints are incomplete")
        source = row.get("source_identity")
        if not isinstance(source, Mapping):
            raise ValueError("selector job source_identity is missing")
        _sha(row.get("source_group_id"), "selector job source_group_id")
        for field in ("source_release_manifest_sha256", "source_annotation_sha256"):
            _sha(source.get(field), f"selector job source_identity.{field}")
        for field in ("task_index", "task_instance_id", "raw_episode_id", "episode_index"):
            _int(source.get(field), f"selector job source_identity.{field}")
        skill_ids = row.get("skill_ids")
        if (not isinstance(skill_ids, list) or not skill_ids or
                any(type(skill_id) is not int or skill_id < 0 for skill_id in skill_ids) or
                skill_ids != sorted(set(skill_ids))):
            raise ValueError("selector job skill_ids are not a sorted unique list")
        windows = row.get("temporal_windows")
        if not isinstance(windows, Mapping):
            raise ValueError("selector job lacks temporal windows")
        actor = windows.get("actor_available_window")
        after = windows.get("offline_review_after_window")
        before = windows.get("offline_review_before_window")
        if not all(isinstance(window, Mapping) for window in (actor, after, before)):
            raise ValueError("selector job has malformed temporal windows")
        anchor = _int(windows.get("anchor_frame"), "selector anchor frame")
        causal = actor.get("sampled_frames")
        if not isinstance(causal, list) or not causal or causal != sorted(set(causal)) or any(
            type(frame) is not int or frame > anchor for frame in causal
        ) or causal[-1] != anchor:
            raise ValueError("selector causal window is not a sorted anchor-bounded list")
        for window_name, window in (("offline before", before), ("offline after", after)):
            frames = window.get("sampled_frames")
            if not isinstance(frames, list) or any(type(frame) is not int for frame in frames):
                raise ValueError(f"selector {window_name} window has malformed frame indices")
        expected_ids.add(event_id)
        jobs_by_event[event_id] = row

    for row in request_rows:
        event_id = _sha(row.get("event_id"), "render request event_id")
        if event_id in requests_by_event:
            raise ValueError("render requests contain a duplicate event_id")
        if row.get("schema_version") != "p107-camera-native-temporal-request-v1":
            raise ValueError("render request has an unsupported schema")
        request_id = _sha(row.get("request_id"), "render request request_id")
        source = row.get("source_identity")
        if not isinstance(source, Mapping):
            raise ValueError("render request source_identity is missing")
        for field in ("source_release_manifest_sha256", "source_annotation_sha256", "source_group_id"):
            _sha(source.get(field), f"render request source_identity.{field}")
        for field in ("task_index", "task_instance_id", "raw_episode_id", "episode_index"):
            _int(source.get(field), f"render request source_identity.{field}")
        if row.get("status") not in {"PENDING_RENDERER_HANDSHAKE_NO_RGB_DECODED", "METADATA_ONLY_UNRESOLVED"}:
            raise ValueError("render request is not metadata-only")
        locators = row.get("anchor_camera_locators")
        if (not isinstance(locators, list) or len(locators) != len(REQUIRED_VIEWS) or
                {item.get("view") for item in locators if isinstance(item, Mapping)} != set(REQUIRED_VIEWS)):
            raise ValueError("render request must bind exactly head/left_wrist/right_wrist")
        job = jobs_by_event.get(event_id)
        if job is None or job.get("camera_native_render_request_id") != request_id:
            raise ValueError("render request does not bind the selector job")
        expected_source = dict(job["source_identity"])
        expected_source["source_group_id"] = job["source_group_id"]
        if dict(source) != expected_source:
            raise ValueError("render request source identity disagrees with selector job")
        windows = job["temporal_windows"]
        if row.get("actor_available_frame_indices") != windows["actor_available_window"]["sampled_frames"]:
            raise ValueError("render request causal frames disagree with selector job")
        if row.get("offline_review_before_frame_indices") != windows["offline_review_before_window"]["sampled_frames"]:
            raise ValueError("render request before frames disagree with selector job")
        if row.get("offline_review_after_frame_indices") != windows["offline_review_after_window"]["sampled_frames"]:
            raise ValueError("render request after frames disagree with selector job")
        requests_by_event[event_id] = row

    for row in selected_rows:
        event_id = _sha(row.get("event_id"), "selected row event_id")
        if event_id in selected_by_event:
            raise ValueError("selected_rows contains a duplicate event_id")
        if (row.get("schema_version") != SELECTOR_SCHEMA or row.get("usage_role") != expected_usage or
                row.get("immutable_split") != expected_split or row.get("training_eligible") is not False):
            raise ValueError("selected row violates the immutable role contract")
        expected_source = dict(jobs_by_event[event_id].get("source_identity") or {})
        expected_source["source_group_id"] = jobs_by_event[event_id].get("source_group_id")
        if event_id not in expected_ids or row.get("source_identity") != expected_source:
            raise ValueError("selected row does not bind its selector job")
        selected_by_event[event_id] = row
    if set(requests_by_event) != expected_ids or set(selected_by_event) != expected_ids:
        raise ValueError("selector files do not cover the same event IDs")

    if role == "train":
        student = files.get("student_candidate_queue.jsonl")
        if student is None:
            raise ValueError("TRAIN selector lacks the student-candidate receipt")
        student_rows = _read_verified_jsonl(root / "student_candidate_queue.jsonl", student, "student_candidate_queue.jsonl")
        if student_rows:
            raise ValueError("coverage selector unexpectedly contains student candidates")

    return {
        "root": root,
        "role": role,
        "usage_role": expected_usage,
        "immutable_split": expected_split,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": actual_manifest_sha256,
        "selection_seal": selection_seal,
        "selection_seal_path": selection_seal_path,
        "selection_seal_sha256": actual_selection_seal_sha256,
        "queue_seal_sha256": queue_seal_sha256,
        "job_path": job_path,
        "job_sha256": sha256_file(job_path),
        "request_path": request_path,
        "request_sha256": sha256_file(request_path),
        "jobs_by_event": jobs_by_event,
        "requests_by_event": requests_by_event,
        "selected_by_event": selected_by_event,
    }


def _load_index(
    index: Path,
    *,
    expected_manifest_sha256: str,
    expected_inventory_sha256: str,
    expected_source_release_sha256: str,
    expected_protocol_sha256: str,
    expected_coverage_sha256: str | None,
) -> dict[str, Any]:
    index = _directory(index, "sealed event index")
    manifest_path = _regular(index / "manifest.json", "index manifest")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != _sha(expected_manifest_sha256, "expected index manifest SHA"):
        raise ValueError("index manifest bytes do not match the external pin")
    return {
        "index": index,
        "manifest_path": manifest_path,
        "manifest": _read_json(manifest_path, "index manifest"),
        "manifest_sha256": actual_manifest_sha256,
        "expected_inventory_sha256": _sha(expected_inventory_sha256, "expected inventory seal SHA"),
        "expected_source_release_sha256": _sha(expected_source_release_sha256, "expected source release SHA"),
        "expected_protocol_sha256": _sha(expected_protocol_sha256, "expected protocol SHA"),
        "expected_coverage_sha256": (_sha(expected_coverage_sha256, "expected coverage SHA") if expected_coverage_sha256 else None),
    }


def _authenticate_index(index_spec: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    index = index_spec["index"]
    expected_source = index_spec["expected_source_release_sha256"]
    expected_protocol = index_spec["expected_protocol_sha256"]
    protocol = base.load_canonical_protocol(protocol_path, expected_sha256=expected_protocol)
    manifest, events_path, groups = base.validate_index(
        index, expected_source_manifest_sha256=expected_source,
        canonical_protocol=protocol, expected_protocol_sha256=expected_protocol,
    )
    if index_spec["manifest"].get("coverage_expectations_sha256") != index_spec["expected_coverage_sha256"]:
        raise ValueError("index coverage expectations do not match the external pin")
    inventory, inventory_sha256 = base.validate_inventory_seal(
        index, expected_inventory_seal_sha256=index_spec["expected_inventory_sha256"]
    )
    if inventory.get("source_release_manifest_sha256") != expected_source:
        raise ValueError("index inventory source release differs from the external pin")
    return {
        **index_spec,
        "protocol": protocol,
        "events_path": events_path,
        "groups": groups,
        "manifest": manifest,
        "inventory": inventory,
        "inventory_sha256": inventory_sha256,
        "event_file_sha256": manifest["files"]["event_candidates.jsonl"]["sha256"],
    }


def _selected_events(index_spec: dict[str, Any], event_ids: set[str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    expected = index_spec["manifest"]["files"]["event_candidates.jsonl"]
    for event in base.iter_jsonl_verified(index_spec["events_path"], expected=expected):
        event_id = event.get("event_id")
        if event_id not in event_ids:
            continue
        if event_id in selected:
            raise ValueError("sealed event index contains a duplicate selected event_id")
        index_spec["protocol"].validate_event(event)
        selected[event_id] = event
    if set(selected) != event_ids:
        missing = sorted(event_ids - set(selected))
        raise ValueError(f"selector event IDs are absent from the sealed event index: {missing[:3]}")
    return selected


def _validate_event_binding(event: Mapping[str, Any], job: Mapping[str, Any], *, role: str) -> None:
    event_id = _sha(event.get("event_id"), "event.event_id")
    source = event.get("source")
    selector_source = dict(job.get("source_identity") or {})
    selector_source["source_group_id"] = job.get("source_group_id")
    if not isinstance(source, Mapping) or not isinstance(selector_source, Mapping):
        raise ValueError(f"{event_id} is missing source identity")
    for field in ("source_release_manifest_sha256", "source_annotation_sha256", "source_group_id",
                  "task_index", "task_instance_id", "raw_episode_id", "episode_index"):
        if source.get(field) != selector_source.get(field):
            raise ValueError(f"{event_id} source binding mismatch at {field}")
    expected_usage = ROLE_CONFIG[role]["usage_role"]
    expected_split = ROLE_CONFIG[role]["immutable_split"]
    if event.get("usage_role") != expected_usage or source.get("original_split") != expected_split:
        raise ValueError(f"{event_id} event role/split disagrees with selector")
    _check_exact_missing(event.get("evidence"), f"{event_id}.evidence")
    observation = event.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError(f"{event_id} lacks observation metadata")
    anchor = _int(job.get("temporal_windows", {}).get("anchor_frame"), f"{event_id}.anchor_frame")
    if observation.get("frame") != anchor:
        raise ValueError(f"{event_id} observation frame disagrees with selector")
    skills = event.get("skill_bundle")
    selected_ids = job.get("skill_ids")
    if not isinstance(skills, list) or not isinstance(selected_ids, list):
        raise ValueError(f"{event_id} has malformed skill binding")
    actual_ids = [skill.get("skill_id") for skill in skills if isinstance(skill, Mapping)]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(selected_ids):
        raise ValueError(f"{event_id} source skills disagree with selector skill_ids")


def _source_skill(event: Mapping[str, Any], skill_id: int) -> Mapping[str, Any]:
    matches = [skill for skill in event.get("skill_bundle", []) if isinstance(skill, Mapping) and skill.get("skill_id") == skill_id]
    if len(matches) != 1:
        raise ValueError(f"event {event.get('event_id')} does not have exactly one skill {skill_id}")
    return matches[0]


def _normalise_for_templates(job: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    source_skills: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for skill_id in job["skill_ids"]:
        skill = _source_skill(event, skill_id)
        source_skills.append({
            "skill_id": skill.get("skill_id"),
            "skill_description": skill.get("raw_description"),
            "canonical_verb": skill.get("verb"),
            "skill_idx": skill.get("skill_idx"),
            "skill_start": skill.get("skill_start"),
            "skill_end": skill.get("skill_end"),
            "source": skill.get("source", ""),
            "target": skill.get("target", ""),
            "destination": skill.get("destination", ""),
            "target_part": skill.get("target_part", ""),
            "binding_confidence": skill.get("binding_confidence"),
            "raw_relation": skill.get("raw_relation", {}),
        })
        candidates.append({
            "skill_id": skill_id,
            "relation_family": "coverage_metadata_goal_relation",
            "intended_goal_relation_question": "PENDING_TEMPLATE_RENDER",
            "observability": "METADATA_ONLY_NEEDS_RGB_REVIEW",
            "metadata_target_ids": templates.unique(templates.flatten_strings([skill.get("target", "")])),
            "metadata_reference_ids": templates.unique(templates.flatten_strings([
                skill.get("source", ""), skill.get("destination", "")
            ])),
            "binding_status": "COVERAGE_METADATA_BINDING",
        })
    windows = job["temporal_windows"]
    return {
        "event_id": event["event_id"],
        "source_group_id": job["source_group_id"],
        "training_eligible": False,
        "source_skill_ids_at_anchor": list(job["skill_ids"]),
        "source_skills": source_skills,
        "event_interval": job["event_interval"],
        "question_context": {
            "question_candidates": candidates,
            "future_frames_are_offline_only": True,
        },
        "temporal_packet": {
            "anchor_frame": windows["anchor_frame"],
            "causal_frame_indices": windows["actor_available_window"]["sampled_frames"],
            "future_frame_indices": windows["offline_review_after_window"]["sampled_frames"],
        },
    }


def _template_records(normalized: dict[str, Any], ordinal: int, category_mapping: dict[str, str] | None,
                      category_mapping_metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    first = templates.generate_row(normalized, ordinal, category_mapping, category_mapping_metadata)
    for candidate, generated in zip(normalized["question_context"]["question_candidates"], first):
        candidate["intended_goal_relation_question"] = generated["current_visible_goal_relation"]["question"]
    return templates.generate_row(normalized, ordinal, category_mapping, category_mapping_metadata)


def _source_pin(selector: Mapping[str, Any], index_spec: Mapping[str, Any], event: Mapping[str, Any],
                job: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    source = event["source"]
    return {
        "schema_version": COVERAGE_SOURCE_PIN_SCHEMA,
        "binding_kind": "coverage_selector_event_query",
        "selection_role": selector["role"],
        "usage_role": selector["usage_role"],
        "immutable_split": selector["immutable_split"],
        "event_id": event["event_id"],
        "selector_manifest_sha256": selector["manifest_sha256"],
        "selector_selection_seal_sha256": selector["selection_seal_sha256"],
        "selector_job_filename": selector["job_path"].name,
        "selector_job_sha256": selector["job_sha256"],
        "selector_record_sha256": canonical_digest(job),
        "render_request_filename": selector["request_path"].name,
        "render_request_sha256": selector["request_sha256"],
        "render_request_record_sha256": canonical_digest(request),
        "index_manifest_sha256": index_spec["manifest_sha256"],
        "index_event_file_sha256": index_spec["event_file_sha256"],
        "index_event_record_sha256": canonical_digest(event),
        "inventory_seal_sha256": index_spec["inventory_sha256"],
        "source_release_manifest_sha256": source["source_release_manifest_sha256"],
        "source_annotation_sha256": source["source_annotation_sha256"],
        "source_group_id": source["source_group_id"],
        "raw_episode_id": source["raw_episode_id"],
        "episode_index": source["episode_index"],
        "observation_frame": event["observation"]["frame"],
    }


def _causal_binding(job: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    windows = job["temporal_windows"]
    locators = sorted(request["anchor_camera_locators"], key=lambda item: REQUIRED_VIEWS.index(item["view"]))
    return {
        "anchor_frame": windows["anchor_frame"],
        "actor_causal_frame_indices": list(windows["actor_available_window"]["sampled_frames"]),
        "offline_review_before_frame_indices": list(windows["offline_review_before_window"]["sampled_frames"]),
        "offline_review_after_frame_indices": list(windows["offline_review_after_window"]["sampled_frames"]),
        "causal_rgb_views": [
            {"view": item["view"], "camera_key": item["camera_key"]}
            for item in locators
        ],
        "future_actor_references_allowed": False,
        "future_use": "OFFLINE_REVIEW_ONLY_NOT_ACTOR_EVIDENCE",
    }


def _query_identity(candidate: Mapping[str, Any], pin: Mapping[str, Any], role: str) -> dict[str, Any]:
    current = candidate["current_visible_goal_relation"]
    binding = candidate["queried_skill_binding"]
    return {
        "schema_version": COVERAGE_QUERY_SCHEMA,
        "selection_role": role,
        "usage_role": candidate["usage_role"],
        "immutable_split": candidate["immutable_split"],
        "event_id": candidate["event_id"],
        "source_group_id": candidate["source_group_id"],
        "observation_frame": candidate["observation_frame"],
        "query_ordinal_within_event": candidate["query_ordinal_within_event"],
        "skill_id": binding["skill_id"],
        "canonical_verb": binding["canonical_verb"],
        "query_text": current["query_text"],
        "relation_family": current["relation_family"],
        "goal_scope": "CURRENT_VISIBLE_RELATION_AT_ANCHOR",
        "source_pin": dict(pin),
    }


def _query_content_sha256(text: str) -> str:
    return canonical_digest({
        "schema_version": "p107.coverage.visual_relation_query_content.v1",
        "kind": "goal_satisfaction_counterfactual",
        "text": text,
    })


def _candidate_record(base_record: Mapping[str, Any], *, selector: Mapping[str, Any], index_spec: Mapping[str, Any],
                      event: Mapping[str, Any], job: Mapping[str, Any], request: Mapping[str, Any],
                      pin: Mapping[str, Any]) -> dict[str, Any]:
    current = dict(base_record["current_visible_goal_relation"])
    binding = base_record["queried_skill_binding"]
    text = current.pop("question")
    current["query_text"] = text
    candidate: dict[str, Any] = {
        "schema_version": COVERAGE_QUERY_SCHEMA,
        "status": "PRELABEL_QUERY_CANDIDATE_ONLY",
        "prelabel_query_id": None,
        "query_content_sha256": _query_content_sha256(text),
        "event_id": event["event_id"],
        "source_group_id": event["source"]["source_group_id"],
        "observation_frame": event["observation"]["frame"],
        "query_ordinal_within_event": base_record["query_ordinal_within_event"],
        "queried_skill_binding": binding,
        "source_v1_query": base_record["source_v1_query"],
        "current_visible_goal_relation": current,
        "historical_change_relation": base_record.get("historical_change_relation"),
        "supporting_observation": base_record.get("supporting_observation"),
        "temporal_binding": _causal_binding(job, request),
        "source_pin": dict(pin),
        "coverage_binding": {
            "selection_role": selector["role"],
            "usage_role": selector["usage_role"],
            "immutable_split": selector["immutable_split"],
            "selector_record_is_metadata_candidate_only": True,
            "packet_role_for_review": "TRAIN_CALIBRATION" if selector["role"] == "train" else "EVAL_NATIVE_DIAGNOSTIC",
        },
        "metadata_entities_raw": base_record["metadata_entities_raw"],
        "category_grounding_audit": base_record.get("category_grounding_audit", []),
        "provenance": {
            "producer": "build_memlite_coverage_visual_relation_queries.py",
            "model": "gpt-5.6-luna/max",
            "human_reviewed": False,
            "image_inspected": False,
            "labels_created": False,
            "root_review": "PENDING",
        },
        "training_eligible": False,
        "no_outcome_or_action_labels": True,
        "no_future_actor_evidence": True,
        "usage_role": selector["usage_role"],
        "immutable_split": selector["immutable_split"],
    }
    candidate["prelabel_query_id"] = canonical_digest(_query_identity(candidate, pin, selector["role"]))
    return candidate


def _registry_projection(candidate: Mapping[str, Any]) -> dict[str, Any]:
    current = candidate["current_visible_goal_relation"]
    binding = candidate["queried_skill_binding"]
    return {
        "schema_version": COVERAGE_REGISTRY_SCHEMA,
        "status": "PRELABEL_QUERY_CANDIDATE_ONLY",
        "prelabel_query_id": candidate["prelabel_query_id"],
        "event_id": candidate["event_id"],
        "source_group_id": candidate["source_group_id"],
        "query_ordinal_within_event": candidate["query_ordinal_within_event"],
        "observation_frame": candidate["observation_frame"],
        "skill_id": binding["skill_id"],
        "canonical_verb": binding["canonical_verb"],
        "query_text": current["query_text"],
        "query_content_sha256": candidate["query_content_sha256"],
        "relation_family": current["relation_family"],
        "source_pin": candidate["source_pin"],
        "usage_role": candidate["usage_role"],
        "immutable_split": candidate["immutable_split"],
        "training_eligible": False,
        "no_outcome_or_action_labels": True,
    }


def _unsupported_record(*, selector: Mapping[str, Any], event: Mapping[str, Any], job: Mapping[str, Any],
                       request: Mapping[str, Any], pin: Mapping[str, Any], skill: Mapping[str, Any],
                       reason: str) -> dict[str, Any]:
    return {
        "schema_version": UNSUPPORTED_SCHEMA,
        "status": "UNSUPPORTED_GOAL_TEMPLATE",
        "event_id": event["event_id"],
        "source_group_id": event["source"]["source_group_id"],
        "observation_frame": event["observation"]["frame"],
        "skill_id": skill.get("skill_id"),
        "canonical_verb": skill.get("verb"),
        "skill_description": skill.get("raw_description"),
        "reason": reason,
        "query_text": None,
        "source_pin": dict(pin),
        "temporal_binding": _causal_binding(job, request),
        "training_eligible": False,
        "no_outcome_or_action_labels": True,
        "usage_role": selector["usage_role"],
        "immutable_split": selector["immutable_split"],
        "provenance": {
            "producer": "build_memlite_coverage_visual_relation_queries.py",
            "model": "gpt-5.6-luna/max",
            "human_reviewed": False,
            "image_inspected": False,
            "labels_created": False,
            "root_review": "PENDING",
        },
    }


def _build_records(selector: Mapping[str, Any], index_spec: Mapping[str, Any],
                   category_mapping: dict[str, str] | None,
                   category_mapping_metadata: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_ids = set(selector["jobs_by_event"])
    events = _selected_events(index_spec, event_ids)
    records: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    for ordinal, event_id in enumerate(sorted(event_ids)):
        job = selector["jobs_by_event"][event_id]
        request = selector["requests_by_event"][event_id]
        event = events[event_id]
        _validate_event_binding(event, job, role=selector["role"])
        pin = _source_pin(selector, index_spec, event, job, request)
        supported_skill_ids: list[int] = []
        for skill_id in job["skill_ids"]:
            skill = _source_skill(event, skill_id)
            verb = skill.get("verb")
            if verb not in SUPPORTED_RELATION_VERBS:
                unsupported.append(_unsupported_record(
                    selector=selector, event=event, job=job, request=request, pin=pin, skill=skill,
                    reason="existing v5 goal-relation template has no branch for this canonical skill; human query design required",
                ))
            else:
                supported_skill_ids.append(skill_id)
        if not supported_skill_ids:
            continue
        template_job = dict(job)
        template_job["skill_ids"] = supported_skill_ids
        normalized = _normalise_for_templates(template_job, event)
        bases = _template_records(normalized, ordinal, category_mapping, category_mapping_metadata)
        if len(bases) != len(supported_skill_ids):
            raise ValueError(f"event {event_id} template output count disagrees with selected skills")
        for base_record in bases:
            query_ordinal = base_record["query_ordinal_within_event"]
            pair = (event_id, query_ordinal)
            if pair in seen_pairs:
                raise ValueError(f"duplicate event/query ordinal {pair}")
            seen_pairs.add(pair)
            candidate = _candidate_record(
                base_record, selector=selector, index_spec=index_spec, event=event,
                job=job, request=request, pin=pin,
            )
            if candidate["prelabel_query_id"] in seen_ids:
                raise ValueError("duplicate coverage prelabel_query_id")
            seen_ids.add(candidate["prelabel_query_id"])
            records.append(candidate)
            registry.append(_registry_projection(candidate))
    records.sort(key=lambda row: (row["event_id"], row["query_ordinal_within_event"]))
    registry.sort(key=lambda row: (row["event_id"], row["query_ordinal_within_event"]))
    unsupported.sort(key=lambda row: (row["event_id"], row["skill_id"] if isinstance(row["skill_id"], int) else -1))
    return records, registry, unsupported


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    with path.open("x", encoding="utf-8") as stream:
        count = 0
        for row in rows:
            stream.write(canonical_json(row) + "\n")
            count += 1
    return {"rows": count, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_output(
    *, index: Path, selector_root: Path, output_dir: Path, role: str,
    expected_selector_manifest_sha256: str, expected_selector_selection_seal_sha256: str,
    expected_index_manifest_sha256: str, expected_inventory_sha256: str,
    expected_source_release_sha256: str, expected_protocol_sha256: str,
    expected_coverage_sha256: str | None, protocol_path: Path,
    category_mapping_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("coverage query output exists; immutable output is never overwritten")
    selector = _validate_selector(
        selector_root, role,
        expected_manifest_sha256=expected_selector_manifest_sha256,
        expected_selection_seal_sha256=expected_selector_selection_seal_sha256,
    )
    index_spec = _load_index(
        index,
        expected_manifest_sha256=expected_index_manifest_sha256,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_source_release_sha256=expected_source_release_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_coverage_sha256=expected_coverage_sha256,
    )
    index_spec = _authenticate_index(index_spec, protocol_path)
    category_mapping = None
    category_mapping_metadata = None
    if category_mapping_path is not None:
        category_mapping, category_mapping_metadata = templates.load_category_mapping(category_mapping_path)
    records, registry, unsupported = _build_records(selector, index_spec, category_mapping, category_mapping_metadata)
    if not records and not unsupported:
        raise ValueError("coverage selector produced no query records")

    stage_parent = output_dir.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=str(stage_parent)))
    try:
        query_receipt = _write_jsonl(stage / "query_candidates.jsonl", records)
        registry_name = "prelabel_registry.jsonl" if role == "train" else "evaluation_query_registry.jsonl"
        registry_receipt = _write_jsonl(stage / registry_name, registry)
        unsupported_receipt = _write_jsonl(stage / "unsupported_goal_queries.jsonl", unsupported)
        if role == "eval":
            eval_rows = []
            for record in records:
                eval_rows.append({
                    "schema_version": COVERAGE_EVAL_REVIEW_SCHEMA,
                    "status": "DIAGNOSTIC_ONLY_PACKET_PENDING",
                    "prelabel_query_id": record["prelabel_query_id"],
                    "event_id": record["event_id"],
                    "source_group_id": record["source_group_id"],
                    "observation_frame": record["observation_frame"],
                    "skill_id": record["queried_skill_binding"]["skill_id"],
                    "query_text": record["current_visible_goal_relation"]["query_text"],
                    "query_content_sha256": record["query_content_sha256"],
                    "usage_role": "evaluation_only",
                    "immutable_split": "eval",
                    "training_eligible": False,
                    "no_outcome_or_action_labels": True,
                    "source_pin": record["source_pin"],
                    "temporal_binding": record["temporal_binding"],
                    "native_packet_contract": {
                        "required_actor_field": "actor_packet.causal_temporal_rgb",
                        "required_views": list(REQUIRED_VIEWS),
                        "future_actor_field_forbidden": "audit.offline_future_rgb",
                        "packet_receipt_sha256": None,
                    },
                    "root_review": "PENDING",
                    "provenance": record["provenance"],
                })
            eval_receipt = _write_jsonl(stage / "evaluation_review.jsonl", eval_rows)
        else:
            eval_receipt = None
        metadata = {
            "schema_version": COVERAGE_MANIFEST_SCHEMA,
            "status": "PRELABEL_QUERY_CANDIDATES_ONLY_NO_ANSWERS_NO_ANNOTATIONS_NO_RELEASE",
            "selection_role": role,
            "usage_role": selector["usage_role"],
            "immutable_split": selector["immutable_split"],
            "training_eligible": False,
            "no_outcome_or_action_labels": True,
            "no_future_actor_evidence": True,
            "query_id_rule": "sha256(canonical JSON of role/event/skill/query ordinal/text/source pins; excludes answers, evidence, actions, recovery, and postlabel view IDs)",
            "source_pins": {
                "selector_manifest_sha256": selector["manifest_sha256"],
                "selector_selection_seal_sha256": selector["selection_seal_sha256"],
                "selector_queue_seal_sha256": selector["queue_seal_sha256"],
                "index_manifest_sha256": index_spec["manifest_sha256"],
                "index_event_file_sha256": index_spec["event_file_sha256"],
                "inventory_seal_sha256": index_spec["inventory_sha256"],
                "source_release_manifest_sha256": expected_source_release_sha256,
                "canonical_protocol_sha256": expected_protocol_sha256,
            },
            "counts": {
                "selector_events": len(selector["jobs_by_event"]),
                "prelabel_queries": len(records),
                "unsupported_goal_queries": len(unsupported),
                "distinct_source_groups": len({record["source_group_id"] for record in records}),
                "multiple_query_events": len({event_id for event_id, ordinal in {(row["event_id"], row["query_ordinal_within_event"]) for row in records} if sum(item["event_id"] == event_id for item in records) > 1}),
            },
            "provenance": {
                "producer": "build_memlite_coverage_visual_relation_queries.py",
                "model": "gpt-5.6-luna/max",
                "human_reviewed": False,
                "image_inspected": False,
                "labels_created": False,
                "root_review": "PENDING",
            },
            "outputs": {
                "query_candidates.jsonl": query_receipt,
                registry_name: registry_receipt,
                "unsupported_goal_queries.jsonl": unsupported_receipt,
            },
        }
        if eval_receipt is not None:
            metadata["outputs"]["evaluation_review.jsonl"] = eval_receipt
        metadata_path = stage / "build_metadata.json"
        metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
        manifest = {
            **metadata,
            "manifest_self_hash": "EXCLUDED_FROM_SELF_HASH",
            "outputs": {
                **metadata["outputs"],
                "build_metadata.json": {"rows": 1, "bytes": metadata_path.stat().st_size, "sha256": sha256_file(metadata_path)},
            },
        }
        (stage / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
        return {
            "output_dir": str(output_dir),
            "role": role,
            "query_count": len(records),
            "unsupported_count": len(unsupported),
            "manifest": str(output_dir / "manifest.json"),
            "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        }
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="sealed full-v3 or canonical subset event index")
    parser.add_argument("--selector-root", type=Path, required=True, help="one sealed coverage role directory")
    parser.add_argument("--role", choices=sorted(ROLE_CONFIG), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--expected-selector-manifest-sha256", required=True)
    parser.add_argument("--expected-selector-selection-seal-sha256", required=True)
    parser.add_argument("--expected-index-manifest-sha256", required=True)
    parser.add_argument("--expected-inventory-seal-sha256", required=True)
    parser.add_argument("--expected-source-release-manifest-sha256", default=SOURCE_RELEASE_SHA256_DEFAULT)
    parser.add_argument("--expected-protocol-sha256", default=PROTOCOL_SHA256_DEFAULT)
    parser.add_argument("--expected-coverage-expectations-sha256", default=COVERAGE_EXPECTATIONS_SHA256_DEFAULT)
    parser.add_argument("--category-mapping", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_output(
            index=args.index, selector_root=args.selector_root, output_dir=args.output_dir, role=args.role,
            expected_selector_manifest_sha256=args.expected_selector_manifest_sha256,
            expected_selector_selection_seal_sha256=args.expected_selector_selection_seal_sha256,
            expected_index_manifest_sha256=args.expected_index_manifest_sha256,
            expected_inventory_sha256=args.expected_inventory_seal_sha256,
            expected_source_release_sha256=args.expected_source_release_manifest_sha256,
            expected_protocol_sha256=args.expected_protocol_sha256,
            expected_coverage_sha256=args.expected_coverage_expectations_sha256,
            protocol_path=args.protocol_path, category_mapping_path=args.category_mapping,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"build_memlite_coverage_visual_relation_queries: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
