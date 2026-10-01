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
    _validate_evidence(evidence, observation_frame=observation_frame, allow_missing=True)
    if not isinstance(evidence["references"], list):
        raise ContractError("actor_evidence.references must be a list")
    forbidden = {"object_pose", "robot_state", "contact_state", "simulator_state", "oracle", "privileged"}
    if any(any(token in str(key).lower() for token in forbidden) for key in evidence):
        raise ContractError("actor_evidence must not carry privileged simulator facts")


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


def validate_corrective_action_view(view: Mapping[str, Any], event: Mapping[str, Any] | None = None) -> None:
    view = _validate_view_common(view, expected_kind="corrective_action", event=event)
    _required(view, (
        "recovery_attempt_id", "recovery_from_attempt_id", "action_intent_bundle_id",
        "executed_intent_bundle_id", "action_start_frame", "actual_executed_length", "raw_action_dim",
        "model_action_dim", "model_padding_indices", "action_is_pad", "executed_action_receipt",
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
    verified = _boolean(view["recovery_verified"], "recovery_verified")
    trainable = _boolean(view["low_action_supervision_mask"], "low_action_supervision_mask")
    if trainable and (not verified or intended != executed):
        raise ContractError("FM positive requires a verified, actually executed action for the same intent bundle")
    if not trainable and "rejection_reason" not in view:
        raise ContractError("non-trainable corrective action needs an explicit rejection_reason")
    if event is not None and receipt["actual_end_frame"] > event["source"]["episode_length"]:
        raise ContractError("corrective action receipt extends beyond its source episode clock")


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
