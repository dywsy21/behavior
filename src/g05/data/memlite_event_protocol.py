"""Immutable, view-separated contracts for P107 event/recovery data.

This module deliberately contains no simulator, video, torch, or dataset
imports.  It is the narrow interchange contract between the metadata index,
annotation/review packaging, and restore-and-branch collection.  In
particular, an annotation interval ending is *not* an outcome and a corrective
action is trainable only after an actual-execution receipt and verification.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "memlite-event-recovery-v1"
PACKET_SCHEMA_VERSION = "memlite-event-packet-v1"
SOURCE_SPLITS = frozenset(("train", "eval"))
USAGE_ROLES = frozenset(("student_candidate", "annotation_calibration", "evaluation_only"))
ATTEMPT_OUTCOMES = frozenset(("IN_PROGRESS", "SUCCEEDED", "FAILED", "UNKNOWN"))
GOAL_SATISFACTION = frozenset(("SATISFIED", "NOT_SATISFIED", "UNKNOWN"))
DECISIONS = frozenset(("EXECUTE", "RETRY", "REPLAN", "STOP"))
LABEL_KINDS = frozenset((
    "goal_satisfaction_counterfactual",
    "attempt_outcome",
    "recovery_decision",
    "corrective_action",
))
ACTION_DIM = 23
MODEL_ACTION_DIM = 27
MODEL_PADDING_INDICES = (7, 8, 17, 18)
ACTION_HORIZON = 32
EXECUTION_RECEIPT_SCHEMA = "p107-execution-receipt-v1"
VERIFICATION_RECEIPT_SCHEMA = "p107-recovery-verification-receipt-v1"
PARENT_APPROVAL_SCHEMA = "p107-parent-low-fm-approval-v1"
PUBLISHER_MANIFEST_SCHEMA = "p107-corrective-publisher-manifest-v1"
RAW_ACTION_PAYLOAD_SCHEMA = "p107-raw23-action-payload-v1"
QUALITY_TIERS = frozenset(("GOLD_PHYSICAL", "SILVER_REVIEWED_LOGGED_DEMONSTRATION"))
EVIDENCE_ORIGINS = frozenset(("logged_demonstration", "live_sim_branch"))


class ContractError(ValueError):
    """A record cannot safely cross an event/recovery data boundary."""


def canonical_json(value: Any) -> str:
    """Return the only JSON representation used in P107 identity hashes."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError(f"non-canonical JSON value: {error}") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Digest a receipt excluding its self-referential digest field."""
    receipt = dict(_mapping(receipt, "receipt"))
    receipt.pop("receipt_sha256", None)
    return canonical_sha256(receipt)


def _normalized_raw_actions_23(raw_actions_23: Sequence[Sequence[float]]) -> list[list[float]]:
    if (not isinstance(raw_actions_23, Sequence) or isinstance(raw_actions_23, (str, bytes)) or
            not 1 <= len(raw_actions_23) <= ACTION_HORIZON):
        raise ContractError("raw action payload must contain one to 32 actual 23D controls")
    normalized = []
    for index, action in enumerate(raw_actions_23):
        if not isinstance(action, Sequence) or isinstance(action, (str, bytes)) or len(action) != ACTION_DIM:
            raise ContractError(f"raw action payload row {index} must have exactly 23 controls")
        normalized.append([_finite_number(value, f"raw action payload row {index}") for value in action])
    return normalized


def action_payload_sha256(raw_actions_23: Sequence[Sequence[float]]) -> str:
    """Canonical SHA-256 of the preserved actual 23D controls (without padding)."""
    return canonical_sha256({"schema_version": RAW_ACTION_PAYLOAD_SCHEMA,
                             "raw_actions_23": _normalized_raw_actions_23(raw_actions_23)})


def raw_action_payload_sha256(raw_actions_23: Sequence[Sequence[float]], *, raw_action_artifact_sha256: str) -> str:
    """Hash the actual finite 23D payload and its sealed source artifact together."""
    _sha256(raw_action_artifact_sha256, "raw_action_artifact_sha256")
    return canonical_sha256({"schema_version": RAW_ACTION_PAYLOAD_SCHEMA,
                             "raw_action_artifact_sha256": raw_action_artifact_sha256,
                             "raw_actions_23": _normalized_raw_actions_23(raw_actions_23)})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _required(row: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ContractError(f"{name} missing required fields: {missing}")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{name} must be a bool")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _finite_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    return float(value)


def source_group_id(source: Mapping[str, Any]) -> str:
    """Hash the immutable source-instance group, never an episode/frame.

    All sibling episodes, clips, counterfactual questions, and restore branches
    from a task instance share this ID and must retain the same source split.
    """
    source = _mapping(source, "source")
    _required(source, ("source_release_manifest_sha256", "task_index", "task_instance_id"), "source")
    return canonical_sha256({
        "schema_version": SCHEMA_VERSION,
        "kind": "source_group",
        "source_release_manifest_sha256": source["source_release_manifest_sha256"],
        "task_index": source["task_index"],
        "task_instance_id": source["task_instance_id"],
    })


def validate_source_ref(source: Mapping[str, Any]) -> None:
    source = _mapping(source, "source")
    _required(source, (
        "source_release_manifest_sha256", "source_annotation_sha256", "task_index",
        "task_instance_id", "raw_episode_id", "episode_index", "original_split",
        "episode_length", "source_group_id",
    ), "source")
    _sha256(source["source_release_manifest_sha256"], "source_release_manifest_sha256")
    _sha256(source["source_annotation_sha256"], "source_annotation_sha256")
    _integer(source["task_index"], "task_index")
    _integer(source["task_instance_id"], "task_instance_id", minimum=1)
    _integer(source["raw_episode_id"], "raw_episode_id")
    _integer(source["episode_index"], "episode_index")
    _integer(source["episode_length"], "episode_length", minimum=1)
    if source["original_split"] not in SOURCE_SPLITS:
        raise ContractError("original_split must be train or eval")
    if source["source_group_id"] != source_group_id(source):
        raise ContractError("source_group_id does not match immutable release/task/instance identity")


def _interval(value: Any, name: str, *, length: int) -> tuple[int, int]:
    row = _mapping(value, name)
    _required(row, ("start_frame", "end_frame"), name)
    start = _integer(row["start_frame"], f"{name}.start_frame")
    end = _integer(row["end_frame"], f"{name}.end_frame")
    if end <= start or end > length:
        raise ContractError(f"{name} must be a nonempty half-open interval in the source clock")
    return start, end


def _bundle_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a nonempty canonical bundle identifier")
    return value


def event_id(event: Mapping[str, Any]) -> str:
    """Derive a collision-resistant, content-addressed event identity."""
    event = _mapping(event, "event")
    source = _mapping(event.get("source"), "event.source")
    interval = _mapping(event.get("event_interval"), "event.event_interval")
    observation = _mapping(event.get("observation"), "event.observation")
    return canonical_sha256({
        "schema_version": SCHEMA_VERSION,
        "kind": "event",
        "source_release_manifest_sha256": source.get("source_release_manifest_sha256"),
        "source_group_id": source.get("source_group_id"),
        "raw_episode_id": source.get("raw_episode_id"),
        "episode_index": source.get("episode_index"),
        "event_kind": event.get("event_kind"),
        "event_start_frame": interval.get("start_frame"),
        "event_end_frame": interval.get("end_frame"),
        "observation_frame": observation.get("frame"),
        "bundle_id": event.get("bundle_id"),
    })


def validate_event(event: Mapping[str, Any]) -> None:
    event = _mapping(event, "event")
    _required(event, (
        "schema_version", "record_kind", "event_id", "source", "event_kind", "event_interval",
        "observation", "action", "bundle_id", "skill_bundle", "parallel_bundle", "evidence",
        "video_locators",
    ), "event")
    if event["schema_version"] != SCHEMA_VERSION or event["record_kind"] != "event_candidate":
        raise ContractError("not a P107 event candidate")
    validate_source_ref(event["source"])
    if not isinstance(event["event_kind"], str) or not event["event_kind"]:
        raise ContractError("event_kind must be nonempty")
    start, end = _interval(event["event_interval"], "event_interval", length=event["source"]["episode_length"])
    observation = _mapping(event["observation"], "observation")
    _required(observation, ("frame", "timestamp_s"), "observation")
    frame = _integer(observation["frame"], "observation.frame")
    if not start <= frame < end:
        raise ContractError("observation must lie in its event interval")
    timestamp = _finite_number(observation["timestamp_s"], "observation.timestamp_s")
    if abs(timestamp - frame / 30.0) > 1e-9:
        raise ContractError("observation timestamp must equal the published frame/30 clock")
    action = _mapping(event["action"], "action")
    _required(action, ("start_frame", "actual_executed_length", "raw_action_dim", "model_action_dim",
                       "model_padding_indices"), "action")
    if _integer(action["start_frame"], "action.start_frame") != frame:
        raise ContractError("event observation must be the pre-action observation at action.start_frame")
    if action["actual_executed_length"] is not None:
        _integer(action["actual_executed_length"], "action.actual_executed_length", minimum=1)
    if action["raw_action_dim"] != ACTION_DIM or action["model_action_dim"] != MODEL_ACTION_DIM:
        raise ContractError("P107 action dimensions must be 23 raw and 27 model")
    if tuple(action["model_padding_indices"]) != MODEL_PADDING_INDICES:
        raise ContractError("P107 model padding indices must be [7,8,17,18]")
    _bundle_id(event["bundle_id"], "bundle_id")
    if not isinstance(event["skill_bundle"], list) or not event["skill_bundle"]:
        raise ContractError("skill_bundle must preserve at least one raw source member")
    if type(event["parallel_bundle"]) is not bool:
        raise ContractError("parallel_bundle must be boolean")
    if event["parallel_bundle"] != (len(event["skill_bundle"]) > 1):
        raise ContractError("parallel_bundle disagrees with skill_bundle membership")
    _validate_evidence(event["evidence"], observation_frame=frame, allow_missing=True)
    if not isinstance(event["video_locators"], list):
        raise ContractError("video_locators must be a list (possibly empty when metadata is unavailable)")
    if event["event_id"] != event_id(event):
        raise ContractError("event_id does not match canonical event identity")


def _validate_evidence(value: Any, *, observation_frame: int, allow_missing: bool) -> None:
    evidence = _mapping(value, "evidence")
    _required(evidence, ("kind", "evidence_end_frame", "available_frame"), "evidence")
    kind = evidence["kind"]
    if kind == "MISSING":
        if not allow_missing or evidence["evidence_end_frame"] is not None or evidence["available_frame"] is not None:
            raise ContractError("MISSING evidence must have null temporal fields and a false supervision mask")
        return
    if kind in {"SEGMENT_END", "ANNOTATION_END", "TIMEOUT", "GRIPPER_CLOSE", "MODEL_SELF_REPORT"}:
        raise ContractError("non-physical boundary/status cannot be outcome evidence")
    if not isinstance(kind, str) or not kind:
        raise ContractError("evidence.kind must name actual visual/physical/review evidence")
    evidence_end = _integer(evidence["evidence_end_frame"], "evidence.evidence_end_frame")
    available = _integer(evidence["available_frame"], "evidence.available_frame")
    if evidence_end > observation_frame or available > observation_frame or available < evidence_end:
        raise ContractError("outcome evidence/availability must be causal at the observation clock")


def view_id(view: Mapping[str, Any]) -> str:
    view = dict(_mapping(view, "view"))
    view.pop("view_id", None)
    return canonical_sha256({"schema_version": SCHEMA_VERSION, "kind": "label_view", "view": view})


def _validate_view_common(view: Mapping[str, Any], *, expected_kind: str,
                          event: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    view = _mapping(view, "view")
    _required(view, ("schema_version", "label_kind", "view_id", "event_id", "source_group_id",
                     "observation_frame", "annotation_provenance", "review_status", "actor_evidence"), "view")
    if view["schema_version"] != SCHEMA_VERSION or view["label_kind"] != expected_kind:
        raise ContractError(f"expected {expected_kind} label view")
    _sha256(view["event_id"], "event_id")
    _sha256(view["source_group_id"], "source_group_id")
    observation = _integer(view["observation_frame"], "view.observation_frame")
    if not isinstance(view["annotation_provenance"], str) or not view["annotation_provenance"]:
        raise ContractError("annotation_provenance must be explicit")
    if not isinstance(view["review_status"], str) or not view["review_status"]:
        raise ContractError("review_status must be explicit")
    _validate_actor_evidence(view["actor_evidence"], observation)
    if "privileged_evidence" in view and view["privileged_evidence"] is not None:
        _mapping(view["privileged_evidence"], "privileged_evidence")
    if view["view_id"] != view_id(view):
        raise ContractError("view_id does not match canonical label payload")
    if event is not None:
        validate_event(event)
        if (view["event_id"] != event["event_id"] or view["source_group_id"] != event["source"]["source_group_id"] or
                observation != event["observation"]["frame"]):
            raise ContractError("view does not bind the supplied event/source/observation exactly")
    return view


def _validate_actor_evidence(value: Any, observation_frame: int) -> None:
    evidence = _mapping(value, "actor_evidence")
    _required(evidence, ("kind", "evidence_end_frame", "available_frame", "references"), "actor_evidence")
    if set(evidence) != {"kind", "evidence_end_frame", "available_frame", "references"}:
        raise ContractError("actor_evidence has unallowlisted fields")
    _validate_evidence(evidence, observation_frame=observation_frame, allow_missing=True)
    if not isinstance(evidence["references"], list):
        raise ContractError("actor_evidence.references must be a list")
    if evidence["kind"] == "MISSING" and evidence["references"]:
        raise ContractError("MISSING actor evidence cannot carry hidden references")
    available = evidence["available_frame"]
    for index, reference in enumerate(evidence["references"]):
        _validate_actor_value(reference, f"actor_evidence.references[{index}]")
        _validate_actor_reference(reference, f"actor_evidence.references[{index}]",
                                  available_frame=available, observation_frame=observation_frame)


def _validate_actor_value(value: Any, name: str) -> None:
    """Recursively reject untyped/privileged payloads before model projection."""
    forbidden = ("object_pose", "robot_state", "contact_state", "simulator_state", "oracle", "privileged",
                 "physics", "ground_truth", "world_state", "segmentation", "depth")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or any(token in key.lower() for token in forbidden):
                raise ContractError(f"actor evidence contains forbidden nested key at {name}")
            _validate_actor_value(child, f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_actor_value(child, f"{name}[{index}]")
    elif value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not math.isfinite(value):
            raise ContractError(f"actor evidence contains non-finite value at {name}")
    else:
        raise ContractError(f"actor evidence has unsupported nested value at {name}")


def _validate_actor_reference(value: Any, name: str, *, available_frame: int | None,
                              observation_frame: int) -> None:
    """Allow only typed actor-visible locators, never nested simulator state."""
    reference = _mapping(value, name)
    kind = reference.get("kind")
    if kind == "rgb_frame":
        required = {"kind", "view", "frame", "artifact_sha256"}
        if set(reference) != required or reference["view"] not in {"head", "left_wrist", "right_wrist"}:
            raise ContractError("rgb actor reference must be an exact named camera/frame/artifact locator")
        frame = _integer(reference["frame"], f"{name}.frame")
        _sha256(reference["artifact_sha256"], f"{name}.artifact_sha256")
    elif kind == "proprio_observation":
        required = {"kind", "frame", "artifact_sha256", "field_names"}
        if set(reference) != required or not isinstance(reference["field_names"], list) or not reference["field_names"]:
            raise ContractError("proprio actor reference must name a public observation artifact and fields")
        frame = _integer(reference["frame"], f"{name}.frame")
        _sha256(reference["artifact_sha256"], f"{name}.artifact_sha256")
        if any(not isinstance(field, str) or not field or "state" in field.lower() or "pose" in field.lower()
               for field in reference["field_names"]):
            raise ContractError("proprio actor reference names a forbidden privileged field")
    elif kind == "command_identifier":
        required = {"kind", "issued_frame", "command_id"}
        if set(reference) != required or not isinstance(reference["command_id"], str) or not reference["command_id"]:
            raise ContractError("command actor reference must be an exact issued command identifier")
        frame = _integer(reference["issued_frame"], f"{name}.issued_frame")
    else:
        raise ContractError("actor evidence reference must be rgb_frame, proprio_observation, or command_identifier")
    if available_frame is None or frame > available_frame or frame > observation_frame:
        raise ContractError("actor evidence reference frame must be available at the current observation clock")


def actor_evidence_projection(view: Mapping[str, Any]) -> dict[str, Any]:
    """Return only causal actor-visible evidence; never privileged audit facts."""
    view = _mapping(view, "view")
    _integer(view.get("observation_frame"), "view.observation_frame")
    _validate_actor_evidence(view.get("actor_evidence"), view["observation_frame"])
    return json.loads(canonical_json(view["actor_evidence"]))


def validate_goal_satisfaction_view(view: Mapping[str, Any], event: Mapping[str, Any] | None = None) -> None:
    view = _validate_view_common(view, expected_kind="goal_satisfaction_counterfactual", event=event)
    _required(view, ("goal_relation", "goal_satisfaction", "valid_goal_mask", "attempt_outcome",
                     "evidence", "low_action_supervision_mask"), "goal_satisfaction view")
    if not isinstance(view["goal_relation"], str) or not view["goal_relation"]:
        raise ContractError("goal_relation is required")
    if view["goal_satisfaction"] not in GOAL_SATISFACTION or view["attempt_outcome"] != "NOT_APPLICABLE":
        raise ContractError("goal view must not masquerade as an attempt outcome")
    valid = _boolean(view["valid_goal_mask"], "valid_goal_mask")
    if not valid:
        if view["goal_satisfaction"] != "UNKNOWN":
            raise ContractError("unlabeled goal satisfaction must be UNKNOWN with a false mask")
        _validate_evidence(view["evidence"], observation_frame=view["observation_frame"], allow_missing=True)
        if view["evidence"]["kind"] != "MISSING":
            raise ContractError("goal mask=false must preserve MISSING evidence")
    else:
        _validate_evidence(view["evidence"], observation_frame=view["observation_frame"], allow_missing=False)
    if _boolean(view["low_action_supervision_mask"], "low_action_supervision_mask"):
        raise ContractError("counterfactual goal labels never supervise FM actions")


def validate_attempt_outcome_view(view: Mapping[str, Any], event: Mapping[str, Any] | None = None) -> None:
    view = _validate_view_common(view, expected_kind="attempt_outcome", event=event)
    _required(view, ("attempt_id", "parent_attempt_id", "evaluated_bundle_id", "attempt_outcome",
                     "valid_result_mask", "evidence", "low_action_supervision_mask"), "attempt_outcome view")
    if not isinstance(view["attempt_id"], str) or not view["attempt_id"]:
        raise ContractError("attempt_id is required")
    if view["parent_attempt_id"] is not None and not isinstance(view["parent_attempt_id"], str):
        raise ContractError("parent_attempt_id must be null or a string")
    valid = _boolean(view["valid_result_mask"], "valid_result_mask")
    if view["attempt_outcome"] not in ATTEMPT_OUTCOMES:
        raise ContractError("invalid attempt outcome")
    _boolean(view["low_action_supervision_mask"], "low_action_supervision_mask")
    if view["low_action_supervision_mask"]:
        raise ContractError("attempt outcome data is never an FM positive by itself")
    if not valid:
        if view["attempt_outcome"] != "UNKNOWN" or view["evaluated_bundle_id"] is not None:
            raise ContractError("missing outcome evidence is mask=false, UNKNOWN, and has no evaluated bundle")
        _validate_evidence(view["evidence"], observation_frame=view["observation_frame"], allow_missing=True)
        if view["evidence"]["kind"] != "MISSING":
            raise ContractError("mask=false must preserve MISSING rather than a fabricated UNKNOWN")
    else:
        _bundle_id(view["evaluated_bundle_id"], "evaluated_bundle_id")
        _validate_evidence(view["evidence"], observation_frame=view["observation_frame"], allow_missing=False)


def validate_recovery_decision_view(view: Mapping[str, Any], event: Mapping[str, Any] | None = None) -> None:
    view = _validate_view_common(view, expected_kind="recovery_decision", event=event)
    _required(view, ("decision", "target_bundle_id", "decision_supervision_mask", "evidence"), "recovery_decision view")
    valid = _boolean(view["decision_supervision_mask"], "decision_supervision_mask")
    if view["decision"] not in DECISIONS:
        raise ContractError("invalid recovery decision")
    if valid:
        _bundle_id(view["target_bundle_id"], "target_bundle_id")
        _validate_evidence(view["evidence"], observation_frame=view["observation_frame"], allow_missing=False)
    else:
        if view["target_bundle_id"] is not None:
            raise ContractError("unmasked recovery decision must not carry a target bundle")
        _validate_evidence(view["evidence"], observation_frame=view["observation_frame"], allow_missing=True)
        if view["evidence"]["kind"] != "MISSING":
            raise ContractError("unmasked recovery decision must preserve MISSING evidence")


def _receipt_registry(receipts: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    registry: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        receipt = _mapping(receipt, name)
        _required(receipt, ("receipt_sha256",), name)
        digest = _sha256(receipt["receipt_sha256"], f"{name}.receipt_sha256")
        if digest != receipt_sha256(receipt):
            raise ContractError(f"{name} receipt_sha256 does not match canonical receipt bytes")
        if digest in registry:
            raise ContractError(f"duplicate {name} receipt digest")
        registry[digest] = receipt
    return registry


def _validate_execution_receipt(receipt: Mapping[str, Any]) -> None:
    receipt = _mapping(receipt, "execution_receipt")
    required = {
        "schema_version", "receipt_sha256", "event_id", "source_group_id", "source_release_manifest_sha256",
        "raw_episode_id", "episode_index", "raw_action_sha256", "action_payload_sha256",
        "raw_action_artifact_sha256",
        "action_start_frame", "actual_end_frame", "actual_executed_length", "raw_action_dim",
        "intent_bundle_id", "executed",
    }
    if set(receipt) != required or receipt["schema_version"] != EXECUTION_RECEIPT_SCHEMA:
        raise ContractError("execution receipt must use the exact P107 receipt schema")
    _sha256(receipt["event_id"], "execution_receipt.event_id")
    _sha256(receipt["source_group_id"], "execution_receipt.source_group_id")
    _sha256(receipt["source_release_manifest_sha256"], "execution_receipt.source_release_manifest_sha256")
    for key in ("raw_action_sha256", "action_payload_sha256", "raw_action_artifact_sha256"):
        _sha256(receipt[key], f"execution_receipt.{key}")
    _integer(receipt["raw_episode_id"], "execution_receipt.raw_episode_id")
    _integer(receipt["episode_index"], "execution_receipt.episode_index")
    start = _integer(receipt["action_start_frame"], "execution_receipt.action_start_frame")
    end = _integer(receipt["actual_end_frame"], "execution_receipt.actual_end_frame")
    length = _integer(receipt["actual_executed_length"], "execution_receipt.actual_executed_length", minimum=1)
    if end != start + length or length > ACTION_HORIZON or receipt["raw_action_dim"] != ACTION_DIM:
        raise ContractError("execution receipt action clock/dimension is not an actual 23D P107 control sequence")
    _bundle_id(receipt["intent_bundle_id"], "execution_receipt.intent_bundle_id")
    if _boolean(receipt["executed"], "execution_receipt.executed") is not True:
        raise ContractError("execution receipt must attest actual execution, not a planned suffix")


def _validate_verification_receipt(receipt: Mapping[str, Any]) -> None:
    receipt = _mapping(receipt, "recovery_verification_receipt")
    required = {
        "schema_version", "receipt_sha256", "event_id", "source_group_id", "recovery_attempt_id",
        "recovery_from_attempt_id", "action_intent_bundle_id", "raw_action_sha256", "evidence_origin",
        "quality_tier", "verification_frame", "label_available_frame", "branch_or_episode_final_frame",
        "evidence", "verification_artifact_sha256", "frozen_action_window_artifact_sha256",
        "temporal_review_artifact_sha256", "live_runtime_acceptance_root_sha256",
    }
    if set(receipt) != required or receipt["schema_version"] != VERIFICATION_RECEIPT_SCHEMA:
        raise ContractError("recovery verification receipt must use the exact P107 receipt schema")
    for key in ("event_id", "source_group_id", "raw_action_sha256", "verification_artifact_sha256"):
        _sha256(receipt[key], f"recovery_verification_receipt.{key}")
    if receipt["evidence_origin"] not in EVIDENCE_ORIGINS or receipt["quality_tier"] not in QUALITY_TIERS:
        raise ContractError("recovery verification has an unknown evidence origin or quality tier")
    if ((receipt["evidence_origin"] == "live_sim_branch") != (receipt["quality_tier"] == "GOLD_PHYSICAL") or
            (receipt["evidence_origin"] == "logged_demonstration" and
             receipt["quality_tier"] != "SILVER_REVIEWED_LOGGED_DEMONSTRATION")):
        raise ContractError("recovery evidence origin and quality tier cannot be silently promoted")
    if receipt["evidence_origin"] == "live_sim_branch":
        _sha256(receipt["live_runtime_acceptance_root_sha256"],
                 "recovery_verification_receipt.live_runtime_acceptance_root_sha256")
        if (receipt["frozen_action_window_artifact_sha256"] is not None or
                receipt["temporal_review_artifact_sha256"] is not None):
            raise ContractError("live GOLD receipt cannot impersonate logged-demo SILVER artifacts")
    else:
        if receipt["live_runtime_acceptance_root_sha256"] is not None:
            raise ContractError("logged demonstration receipt cannot claim a live runtime root")
        _sha256(receipt["frozen_action_window_artifact_sha256"],
                "recovery_verification_receipt.frozen_action_window_artifact_sha256")
        _sha256(receipt["temporal_review_artifact_sha256"],
                "recovery_verification_receipt.temporal_review_artifact_sha256")
    for key in ("recovery_attempt_id", "recovery_from_attempt_id", "action_intent_bundle_id"):
        _bundle_id(receipt[key], f"recovery_verification_receipt.{key}")
    verification = _integer(receipt["verification_frame"], "recovery_verification_receipt.verification_frame")
    available = _integer(receipt["label_available_frame"], "recovery_verification_receipt.label_available_frame")
    final = _integer(receipt["branch_or_episode_final_frame"], "recovery_verification_receipt.branch_or_episode_final_frame")
    evidence = _mapping(receipt["evidence"], "recovery_verification_receipt.evidence")
    if set(evidence) != {"kind", "evidence_end_frame", "available_frame", "artifact_sha256"}:
        raise ContractError("recovery verification evidence must be an exact evidence receipt")
    if evidence["kind"] in {"MISSING", "SEGMENT_END", "ANNOTATION_END", "TIMEOUT", "GRIPPER_CLOSE", "MODEL_SELF_REPORT"}:
        raise ContractError("missing or non-physical boundary cannot verify a corrective action")
    if not isinstance(evidence["kind"], str) or not evidence["kind"]:
        raise ContractError("recovery verification evidence kind is required")
    evidence_end = _integer(evidence["evidence_end_frame"], "recovery_verification_receipt.evidence_end_frame")
    evidence_available = _integer(evidence["available_frame"], "recovery_verification_receipt.evidence.available_frame")
    _sha256(evidence["artifact_sha256"], "recovery_verification_receipt.evidence.artifact_sha256")
    if not (evidence_end <= evidence_available <= verification <= available <= final):
        raise ContractError("verification evidence/availability/final clocks are inconsistent")


def _validate_parent_approval(receipt: Mapping[str, Any]) -> None:
    receipt = _mapping(receipt, "parent_approval_receipt")
    required = {
        "schema_version", "receipt_sha256", "approval_status", "view_id", "event_id", "source_group_id",
        "source_release_manifest_sha256", "raw_episode_id", "episode_index", "original_split", "usage_role",
        "action_intent_bundle_id", "executed_intent_bundle_id", "raw_action_sha256", "action_payload_sha256",
        "execution_receipt_sha256", "recovery_verification_receipt_sha256", "parent_review_artifact_sha256",
        "evidence_origin", "quality_tier", "index_inventory_seal_sha256", "action_payload_root_sha256",
        "execution_receipt_root_sha256", "recovery_verification_receipt_root_sha256",
        "artifact_manifest_root_sha256",
    }
    if set(receipt) != required or receipt["schema_version"] != PARENT_APPROVAL_SCHEMA:
        raise ContractError("parent approval must use the exact P107 approval schema")
    if receipt["approval_status"] != "APPROVED_FOR_LOW_FM":
        raise ContractError("only explicit parent APPROVED_FOR_LOW_FM may authorize an FM action")
    for key in ("receipt_sha256", "view_id", "event_id", "source_group_id", "source_release_manifest_sha256",
                "raw_action_sha256", "action_payload_sha256", "execution_receipt_sha256",
                "recovery_verification_receipt_sha256", "parent_review_artifact_sha256",
                "index_inventory_seal_sha256", "action_payload_root_sha256", "execution_receipt_root_sha256",
                "recovery_verification_receipt_root_sha256", "artifact_manifest_root_sha256"):
        _sha256(receipt[key], f"parent_approval_receipt.{key}")
    _integer(receipt["raw_episode_id"], "parent_approval_receipt.raw_episode_id")
    _integer(receipt["episode_index"], "parent_approval_receipt.episode_index")
    if receipt["original_split"] != "train" or receipt["usage_role"] != "student_candidate":
        raise ContractError("heldout/calibration groups cannot receive a student FM approval")
    if receipt["evidence_origin"] not in EVIDENCE_ORIGINS or receipt["quality_tier"] not in QUALITY_TIERS:
        raise ContractError("parent approval has unknown evidence origin or quality tier")
    for key in ("action_intent_bundle_id", "executed_intent_bundle_id"):
        _bundle_id(receipt[key], f"parent_approval_receipt.{key}")

def _strict_json(data: bytes, *, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ContractError(f"non-finite JSON number in {name}: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_publisher_path(root: Path, relative_path: Any, name: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ContractError(f"{name}.relative_path must be a nonempty relative path")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"{name}.relative_path escapes the sealed publisher root")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or path.is_symlink() or not path.is_file():
        raise ContractError(f"{name}.relative_path is not a regular file inside publisher root")
    return path


def _read_sealed_jsonl(root: Path, receipt: Any, name: str) -> tuple[list[dict[str, Any]], str]:
    receipt = _mapping(receipt, name)
    if set(receipt) != {"relative_path", "sha256", "bytes", "rows"}:
        raise ContractError(f"{name} file receipt must have exact path/hash/bytes/rows fields")
    _sha256(receipt["sha256"], f"{name}.sha256")
    path = _safe_publisher_path(root, receipt["relative_path"], name)
    if path.stat().st_size != _integer(receipt["bytes"], f"{name}.bytes") or _file_sha256(path) != receipt["sha256"]:
        raise ContractError(f"{name} bytes or SHA-256 does not match the externally sealed publisher manifest")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_bytes().splitlines(), start=1):
        if line.strip():
            row = _strict_json(line, name=f"{path}:{line_no}")
            if not isinstance(row, dict):
                raise ContractError(f"{name} row {line_no} is not an object")
            rows.append(row)
    if len(rows) != _integer(receipt["rows"], f"{name}.rows"):
        raise ContractError(f"{name} row count does not match sealed publisher manifest")
    return rows, receipt["sha256"]


_AUTHORITY_TOKEN = object()


class CorrectiveActionAuthority:
    """Opaque, deep-immutable result of loading an externally SHA-pinned publisher root."""
    __slots__ = ("_token", "publisher_manifest_sha256", "index_inventory_seal_sha256", "_event_json",
                 "_group_json", "_execution_json", "_verification_json", "_approval_json", "_payload_json",
                 "_artifact_sha256s", "_action_payload_root_sha256", "_execution_root_sha256",
                 "_verification_root_sha256", "_artifact_root_sha256", "_live_runtime_root_sha256")

    def __init__(self, token: object, **values: Any) -> None:
        if token is not _AUTHORITY_TOKEN:
            raise ContractError("CorrectiveActionAuthority may only be loaded from a sealed publisher manifest")
        self._token = token
        for key, value in values.items():
            setattr(self, key, value)


def _frozen_json_registry(rows: Mapping[str, Mapping[str, Any]], name: str) -> Mapping[str, str]:
    return MappingProxyType({key: canonical_json(value) for key, value in rows.items()})


def _authority_value(authority: CorrectiveActionAuthority, registry: str, key: str, name: str) -> Mapping[str, Any]:
    if not isinstance(authority, CorrectiveActionAuthority) or authority._token is not _AUTHORITY_TOKEN:
        raise ContractError("FM positive requires a sealed CorrectiveActionAuthority capability")
    encoded = getattr(authority, registry).get(key)
    if encoded is None:
        raise ContractError(f"FM positive refers to an absent sealed {name}")
    row = _strict_json(encoded.encode("utf-8"), name=f"sealed {name}")
    if not isinstance(row, Mapping):
        raise ContractError(f"sealed {name} is malformed")
    return row


def _validate_indexed_group(group: Mapping[str, Any]) -> None:
    _required(group, ("source_group_id", "source_release_manifest_sha256", "task_index", "task_instance_id",
                      "original_split", "usage_role"), "indexed_source_group")
    _sha256(group["source_group_id"], "indexed_source_group.source_group_id")
    _sha256(group["source_release_manifest_sha256"], "indexed_source_group.source_release_manifest_sha256")
    _integer(group["task_index"], "indexed_source_group.task_index")
    _integer(group["task_instance_id"], "indexed_source_group.task_instance_id", minimum=1)
    if group["original_split"] not in SOURCE_SPLITS or group["usage_role"] not in USAGE_ROLES:
        raise ContractError("indexed source group has invalid split or role")


def _validate_raw_action_payload(payload: Mapping[str, Any]) -> None:
    required = {"schema_version", "action_payload_sha256", "raw_action_sha256", "raw_action_artifact_sha256",
                "raw_actions_23"}
    if set(payload) != required or payload["schema_version"] != RAW_ACTION_PAYLOAD_SCHEMA:
        raise ContractError("raw action payload must use the exact P107 raw23 payload schema")
    for key in ("action_payload_sha256", "raw_action_sha256", "raw_action_artifact_sha256"):
        _sha256(payload[key], f"raw_action_payload.{key}")
    actions = payload["raw_actions_23"]
    raw_digest = raw_action_payload_sha256(actions, raw_action_artifact_sha256=payload["raw_action_artifact_sha256"])
    plain_digest = action_payload_sha256(actions)
    if payload["raw_action_sha256"] != raw_digest or payload["action_payload_sha256"] != plain_digest:
        raise ContractError("raw action payload hashes are not recomputed from actual 23D payload bytes")


def authority_raw_action_payload(authority: CorrectiveActionAuthority, action_payload_sha256_value: str) -> dict[str, Any]:
    """Return a fresh verified 23D target payload for a publisher-authorized package only."""
    _sha256(action_payload_sha256_value, "action_payload_sha256")
    payload = _authority_value(authority, "_payload_json", action_payload_sha256_value, "raw action payload")
    _validate_raw_action_payload(payload)
    return json.loads(canonical_json(payload))


def load_corrective_action_authority(publisher_root: Path, *, expected_publisher_manifest_sha256: str,
                                     expected_index_inventory_seal_sha256: str,
                                     expected_live_runtime_acceptance_root_sha256: str | None = None) -> CorrectiveActionAuthority:
    """Load final FM authority only from an externally SHA-pinned publisher root.

    The expected roots are release-policy capabilities supplied outside candidate
    views.  This function never accepts caller-created event/receipt dictionaries.
    """
    _sha256(expected_publisher_manifest_sha256, "expected_publisher_manifest_sha256")
    _sha256(expected_index_inventory_seal_sha256, "expected_index_inventory_seal_sha256")
    if expected_live_runtime_acceptance_root_sha256 is not None:
        _sha256(expected_live_runtime_acceptance_root_sha256, "expected_live_runtime_acceptance_root_sha256")
    root = Path(publisher_root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("publisher root must be a regular directory")
    manifest_path = root / "publisher_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or _file_sha256(manifest_path) != expected_publisher_manifest_sha256:
        raise ContractError("publisher manifest does not match the externally pinned authority root SHA-256")
    manifest = _strict_json(manifest_path.read_bytes(), name=str(manifest_path))
    if not isinstance(manifest, Mapping):
        raise ContractError("publisher manifest must be an object")
    required = {"schema_version", "index_inventory_seal_sha256", "accepted_live_runtime_root_sha256", "files"}
    if set(manifest) != required or manifest["schema_version"] != PUBLISHER_MANIFEST_SCHEMA:
        raise ContractError("publisher manifest must use the exact P107 authority schema")
    if manifest["index_inventory_seal_sha256"] != expected_index_inventory_seal_sha256:
        raise ContractError("publisher manifest does not bind the expected immutable index inventory seal")
    files = _mapping(manifest["files"], "publisher_manifest.files")
    names = {"events", "source_groups", "raw_action_payloads", "execution_receipts", "verification_receipts",
             "parent_approval_receipts", "artifacts"}
    if set(files) != names:
        raise ContractError("publisher manifest must seal the exact event/group/payload/receipt/artifact inventory")
    live_root = manifest["accepted_live_runtime_root_sha256"]
    if live_root is not None:
        _sha256(live_root, "publisher_manifest.accepted_live_runtime_root_sha256")
    if live_root != expected_live_runtime_acceptance_root_sha256:
        raise ContractError("publisher live-runtime root is not the externally accepted capability")
    loaded = {name: _read_sealed_jsonl(root, files[name], name) for name in names}
    events, groups = {}, {}
    for group in loaded["source_groups"][0]:
        _validate_indexed_group(group)
        key = group["source_group_id"]
        if key in groups:
            raise ContractError("duplicate sealed indexed source group")
        groups[key] = group
    for event in loaded["events"][0]:
        validate_event(event)
        key = event["event_id"]
        group = groups.get(event["source"]["source_group_id"])
        if key in events or group is None or event.get("usage_role") != group["usage_role"]:
            raise ContractError("sealed event/group identity or role is not canonical")
        if (group["source_release_manifest_sha256"] != event["source"]["source_release_manifest_sha256"] or
                group["original_split"] != event["source"]["original_split"]):
            raise ContractError("sealed event/source split identity mismatch")
        events[key] = event
    payloads = {}
    for payload in loaded["raw_action_payloads"][0]:
        _validate_raw_action_payload(payload)
        key = payload["action_payload_sha256"]
        if key in payloads:
            raise ContractError("duplicate sealed raw action payload")
        payloads[key] = payload
    execution = _receipt_registry(loaded["execution_receipts"][0], "execution_receipt")
    for receipt in execution.values():
        _validate_execution_receipt(receipt)
    verification = _receipt_registry(loaded["verification_receipts"][0], "recovery_verification_receipt")
    for receipt in verification.values():
        _validate_verification_receipt(receipt)
    artifacts = set()
    for artifact in loaded["artifacts"][0]:
        if set(artifact) != {"artifact_sha256", "artifact_kind"} or not isinstance(artifact["artifact_kind"], str):
            raise ContractError("artifact manifest rows must name exact artifact digests and kinds")
        _sha256(artifact["artifact_sha256"], "artifact.artifact_sha256")
        if artifact["artifact_sha256"] in artifacts:
            raise ContractError("duplicate sealed artifact digest")
        artifacts.add(artifact["artifact_sha256"])
    approvals = {}
    roots = {"index_inventory_seal_sha256": expected_index_inventory_seal_sha256,
             "action_payload_root_sha256": loaded["raw_action_payloads"][1],
             "execution_receipt_root_sha256": loaded["execution_receipts"][1],
             "recovery_verification_receipt_root_sha256": loaded["verification_receipts"][1],
             "artifact_manifest_root_sha256": loaded["artifacts"][1]}
    for approval in _receipt_registry(loaded["parent_approval_receipts"][0], "parent_approval_receipt").values():
        _validate_parent_approval(approval)
        if approval["view_id"] in approvals or any(approval[key] != value for key, value in roots.items()):
            raise ContractError("parent approval has duplicate view or does not bind sealed authority roots")
        approvals[approval["view_id"]] = approval
    return CorrectiveActionAuthority(
        _AUTHORITY_TOKEN, publisher_manifest_sha256=expected_publisher_manifest_sha256,
        index_inventory_seal_sha256=expected_index_inventory_seal_sha256,
        _event_json=_frozen_json_registry(events, "event"), _group_json=_frozen_json_registry(groups, "group"),
        _execution_json=_frozen_json_registry(execution, "execution"),
        _verification_json=_frozen_json_registry(verification, "verification"),
        _approval_json=_frozen_json_registry(approvals, "approval"), _payload_json=_frozen_json_registry(payloads, "payload"),
        _artifact_sha256s=frozenset(artifacts), _action_payload_root_sha256=roots["action_payload_root_sha256"],
        _execution_root_sha256=roots["execution_receipt_root_sha256"],
        _verification_root_sha256=roots["recovery_verification_receipt_root_sha256"],
        _artifact_root_sha256=roots["artifact_manifest_root_sha256"], _live_runtime_root_sha256=live_root)


def indexed_event_eligibility(event: Mapping[str, Any], source_group: Mapping[str, Any]) -> dict[str, bool]:
    """Keep valid heldout evaluation supervision distinct from student training."""
    validate_event(event)
    group = _mapping(source_group, "indexed_source_group")
    if (event["source"]["source_group_id"] != group.get("source_group_id") or
            event["source"]["original_split"] != group.get("original_split")):
        raise ContractError("event does not match its indexed source group")
    student = event["source"]["original_split"] == "train" and group.get("usage_role") == "student_candidate"
    return {"evaluation_loss_eligible": True, "student_train_eligible": student}


def _validate_authorized_corrective_action(view: Mapping[str, Any], event: Mapping[str, Any],
                                            authority: CorrectiveActionAuthority) -> None:
    indexed = _authority_value(authority, "_event_json", view["event_id"], "event")
    if canonical_json(indexed) != canonical_json(event):
        raise ContractError("FM positive must bind an exact canonical indexed event, not a caller substitute")
    group = _authority_value(authority, "_group_json", view["source_group_id"], "source group")
    if not indexed_event_eligibility(indexed, group)["student_train_eligible"]:
        raise ContractError("only indexed TRAIN student_candidate groups may export student FM actions")
    if view["review_status"] != "PARENT_APPROVED":
        raise ContractError("PROPOSED/agent-only view cannot export a student FM action")
    approval = _authority_value(authority, "_approval_json", view["view_id"], "parent approval")
    execution = _authority_value(authority, "_execution_json", view["execution_receipt_sha256"], "execution receipt")
    verification = _authority_value(authority, "_verification_json", view["recovery_verification_receipt_sha256"], "verification receipt")
    payload = _authority_value(authority, "_payload_json", view["action_payload_sha256"], "raw action payload")
    _validate_execution_receipt(execution)
    _validate_verification_receipt(verification)
    _validate_parent_approval(approval)
    _validate_raw_action_payload(payload)
    expected = {
        "view_id": view["view_id"], "event_id": indexed["event_id"], "source_group_id": group["source_group_id"],
        "source_release_manifest_sha256": indexed["source"]["source_release_manifest_sha256"],
        "raw_episode_id": indexed["source"]["raw_episode_id"], "episode_index": indexed["source"]["episode_index"],
        "original_split": "train", "usage_role": "student_candidate",
        "action_intent_bundle_id": view["action_intent_bundle_id"],
        "executed_intent_bundle_id": view["executed_intent_bundle_id"],
        "raw_action_sha256": view["executed_action_receipt"]["raw_action_sha256"],
        "action_payload_sha256": view["action_payload_sha256"],
        "execution_receipt_sha256": view["execution_receipt_sha256"],
        "recovery_verification_receipt_sha256": view["recovery_verification_receipt_sha256"],
        "evidence_origin": verification["evidence_origin"], "quality_tier": verification["quality_tier"],
        "index_inventory_seal_sha256": authority.index_inventory_seal_sha256,
        "action_payload_root_sha256": authority._action_payload_root_sha256,
        "execution_receipt_root_sha256": authority._execution_root_sha256,
        "recovery_verification_receipt_root_sha256": authority._verification_root_sha256,
        "artifact_manifest_root_sha256": authority._artifact_root_sha256,
    }
    if any(approval[key] != value for key, value in expected.items()):
        raise ContractError("parent approval does not bind this exact indexed action/evidence payload")
    if approval["parent_review_artifact_sha256"] not in authority._artifact_sha256s:
        raise ContractError("parent approval references an unverified review artifact")
    if (execution["event_id"] != indexed["event_id"] or execution["source_group_id"] != group["source_group_id"] or
            execution["source_release_manifest_sha256"] != indexed["source"]["source_release_manifest_sha256"] or
            execution["raw_episode_id"] != indexed["source"]["raw_episode_id"] or
            execution["episode_index"] != indexed["source"]["episode_index"] or
            execution["raw_action_sha256"] != expected["raw_action_sha256"] or
            execution["action_payload_sha256"] != expected["action_payload_sha256"] or
            execution["raw_action_artifact_sha256"] != payload["raw_action_artifact_sha256"] or
            execution["intent_bundle_id"] != view["executed_intent_bundle_id"] or
            execution["action_start_frame"] != view["action_start_frame"] or
            execution["actual_end_frame"] != view["executed_action_receipt"]["actual_end_frame"] or
            execution["actual_executed_length"] != view["actual_executed_length"]):
        raise ContractError("execution receipt does not match the exact indexed corrective action")
    if (payload["raw_action_sha256"] != expected["raw_action_sha256"] or
            len(payload["raw_actions_23"]) != execution["actual_executed_length"] or
            payload["raw_action_artifact_sha256"] not in authority._artifact_sha256s):
        raise ContractError("sealed actual 23D payload does not match the execution receipt")
    if (verification["event_id"] != indexed["event_id"] or verification["source_group_id"] != group["source_group_id"] or
            verification["recovery_attempt_id"] != view["recovery_attempt_id"] or
            verification["recovery_from_attempt_id"] != view["recovery_from_attempt_id"] or
            verification["action_intent_bundle_id"] != view["action_intent_bundle_id"] or
            verification["raw_action_sha256"] != expected["raw_action_sha256"] or
            verification["verification_frame"] < execution["actual_end_frame"] or
            verification["evidence"]["evidence_end_frame"] < execution["actual_end_frame"]):
        raise ContractError("recovery verification does not match the executed corrective action")
    for key in ("verification_artifact_sha256", "evidence"):
        artifact = verification[key] if key != "evidence" else verification[key]["artifact_sha256"]
        if artifact not in authority._artifact_sha256s:
            raise ContractError("verification evidence artifact is absent from the sealed artifact manifest")
    if verification["evidence_origin"] == "live_sim_branch":
        if authority._live_runtime_root_sha256 is None or (
                verification["live_runtime_acceptance_root_sha256"] != authority._live_runtime_root_sha256):
            raise ContractError("GOLD live recovery requires an externally accepted live-runtime root")
    elif (verification["frozen_action_window_artifact_sha256"] != payload["raw_action_artifact_sha256"] or
          verification["frozen_action_window_artifact_sha256"] not in authority._artifact_sha256s or
          verification["temporal_review_artifact_sha256"] not in authority._artifact_sha256s):
        raise ContractError("SILVER logged recovery requires sealed frozen action-window and temporal-review artifacts")


def validate_corrective_action_view(view: Mapping[str, Any], event: Mapping[str, Any] | None = None,
                                    *, authority: CorrectiveActionAuthority | None = None) -> None:
    view = _validate_view_common(view, expected_kind="corrective_action", event=event)
    _required(view, (
        "recovery_attempt_id", "recovery_from_attempt_id", "action_intent_bundle_id",
        "executed_intent_bundle_id", "action_start_frame", "actual_executed_length", "raw_action_dim",
        "model_action_dim", "model_padding_indices", "action_is_pad", "executed_action_receipt",
        "action_payload_sha256", "execution_receipt_sha256", "recovery_verification_receipt_sha256",
        "recovery_verified", "low_action_supervision_mask",
    ), "corrective_action view")
    if not isinstance(view["recovery_attempt_id"], str) or not view["recovery_attempt_id"]:
        raise ContractError("recovery_attempt_id is required")
    if not isinstance(view["recovery_from_attempt_id"], str) or not view["recovery_from_attempt_id"]:
        raise ContractError("recovery_from_attempt_id is required")
    if _integer(view["action_start_frame"], "action_start_frame") != view["observation_frame"]:
        raise ContractError("corrective action observation must be the pre-action state")
    length = _integer(view["actual_executed_length"], "actual_executed_length", minimum=1)
    if length > ACTION_HORIZON:
        raise ContractError("corrective action length cannot exceed the fixed 32-action horizon")
    if view["raw_action_dim"] != ACTION_DIM or view["model_action_dim"] != MODEL_ACTION_DIM:
        raise ContractError("corrective action must use approved 23D->27D dimensions")
    if tuple(view["model_padding_indices"]) != MODEL_PADDING_INDICES:
        raise ContractError("corrective action changed approved padding dimensions")
    action_is_pad = view["action_is_pad"]
    if (not isinstance(action_is_pad, list) or len(action_is_pad) != ACTION_HORIZON or
            any(type(x) is not bool for x in action_is_pad) or
            action_is_pad != [False] * length + [True] * (ACTION_HORIZON - length)):
        raise ContractError("action_is_pad must distinguish actual controls from 32-step shape padding")
    intended = _bundle_id(view["action_intent_bundle_id"], "action_intent_bundle_id")
    executed = _bundle_id(view["executed_intent_bundle_id"], "executed_intent_bundle_id")
    receipt = _mapping(view["executed_action_receipt"], "executed_action_receipt")
    _required(receipt, ("executed", "raw_action_sha256", "intent_bundle_id", "actual_end_frame"),
              "executed_action_receipt")
    _sha256(receipt["raw_action_sha256"], "executed_action_receipt.raw_action_sha256")
    if _boolean(receipt["executed"], "executed_action_receipt.executed") is not True:
        raise ContractError("a predicted or unexecuted action cannot be a corrective transition")
    if receipt["intent_bundle_id"] != executed or _integer(receipt["actual_end_frame"], "actual_end_frame") != view["action_start_frame"] + length:
        raise ContractError("executed action receipt has an action/bundle/clock mismatch")
    for key in ("action_payload_sha256", "execution_receipt_sha256", "recovery_verification_receipt_sha256"):
        _sha256(view[key], key)
    _boolean(view["recovery_verified"], "recovery_verified")
    trainable = _boolean(view["low_action_supervision_mask"], "low_action_supervision_mask")
    if trainable and intended != executed:
        raise ContractError("FM positive requires an actually executed action for the same intent bundle")
    if trainable and (event is None or authority is None):
        raise ContractError("FM positive requires canonical indexed event plus external parent authority")
    if not trainable and "rejection_reason" not in view:
        raise ContractError("non-trainable corrective action needs an explicit rejection_reason")
    if event is not None and receipt["actual_end_frame"] > event["source"]["episode_length"]:
        raise ContractError("corrective action receipt extends beyond its source episode clock")
    if trainable:
        _validate_authorized_corrective_action(view, event, authority)


def project_action_23_to_27(raw_actions: Sequence[Sequence[float]], actual_executed_length: int) -> dict[str, Any]:
    """Project actual controls only; post-length rows are shape padding, never transitions."""
    length = _integer(actual_executed_length, "actual_executed_length", minimum=1)
    if length > ACTION_HORIZON or len(raw_actions) != length:
        raise ContractError("raw action list must contain exactly the actual (<=32) executed controls")
    projected: list[list[float]] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != ACTION_DIM:
            raise ContractError(f"raw action {index} must have exactly 23 controls")
        values = [_finite_number(value, f"raw action {index}") for value in raw]
        model = []
        cursor = 0
        for dim in range(MODEL_ACTION_DIM):
            if dim in MODEL_PADDING_INDICES:
                model.append(0.0)
            else:
                model.append(values[cursor]); cursor += 1
        projected.append(model)
    # This numerical repeat merely satisfies fixed-shape model tensors.  The
    # explicit mask makes the rows unavailable as executed action evidence.
    projected.extend([list(projected[-1]) for _ in range(ACTION_HORIZON - length)])
    return {
        "actions_27": projected,
        "action_is_pad": [False] * length + [True] * (ACTION_HORIZON - length),
        "action_dim_is_pad": [dim in MODEL_PADDING_INDICES for dim in range(MODEL_ACTION_DIM)],
    }
