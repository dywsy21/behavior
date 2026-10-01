"""Compact, pinned test double for the public P107 protocol surface.

It intentionally implements only the source/event authority used by the
selector fixtures.  Production must use a SHA-pinned owner protocol module.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = "memlite-event-recovery-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def source_group_id(source: Mapping[str, Any]) -> str:
    return canonical_sha256({
        "schema_version": SCHEMA_VERSION,
        "kind": "source_group",
        "source_release_manifest_sha256": source.get("source_release_manifest_sha256"),
        "task_index": source.get("task_index"),
        "task_instance_id": source.get("task_instance_id"),
    })


def validate_source_ref(source: Mapping[str, Any]) -> None:
    if not isinstance(source, Mapping):
        raise ValueError("source must be an object")
    required = {"source_release_manifest_sha256", "source_annotation_sha256", "source_group_id", "task_index",
                "task_instance_id", "raw_episode_id", "episode_index", "episode_length", "original_split"}
    if not required <= set(source):
        raise ValueError("source fields missing")
    _sha256(source["source_release_manifest_sha256"], "source_release_manifest_sha256")
    _sha256(source["source_annotation_sha256"], "source_annotation_sha256")
    _integer(source["task_index"], "task_index")
    _integer(source["task_instance_id"], "task_instance_id", minimum=1)
    _integer(source["raw_episode_id"], "raw_episode_id")
    _integer(source["episode_index"], "episode_index")
    _integer(source["episode_length"], "episode_length", minimum=1)
    if source["original_split"] not in {"train", "eval"} or source["source_group_id"] != source_group_id(source):
        raise ValueError("source identity invalid")


def event_id(event: Mapping[str, Any]) -> str:
    source = event.get("source", {})
    interval = event.get("event_interval", {})
    observation = event.get("observation", {})
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
    if not isinstance(event, Mapping):
        raise ValueError("event must be an object")
    required = {"schema_version", "record_kind", "event_id", "source", "event_kind", "event_interval",
                "observation", "action", "bundle_id", "skill_bundle", "parallel_bundle", "evidence", "video_locators"}
    if not required <= set(event) or event["schema_version"] != SCHEMA_VERSION or event["record_kind"] != "event_candidate":
        raise ValueError("event identity invalid")
    validate_source_ref(event["source"])
    start, end = event["event_interval"].get("start_frame"), event["event_interval"].get("end_frame")
    frame, timestamp = event["observation"].get("frame"), event["observation"].get("timestamp_s")
    if (type(start) is not int or type(end) is not int or not 0 <= start <= frame < end <= event["source"]["episode_length"] or
            type(frame) is not int or type(timestamp) not in (int, float) or isinstance(timestamp, bool) or
            not math.isfinite(timestamp) or abs(timestamp - frame / 30.0) > 1e-9):
        raise ValueError("event time invalid")
    action = event["action"]
    if (not isinstance(action, Mapping) or action.get("start_frame") != frame or action.get("actual_executed_length") is not None or
            action.get("raw_action_dim") != 23 or action.get("model_action_dim") != 27 or
            action.get("model_padding_indices") != [7, 8, 17, 18]):
        raise ValueError("event action metadata invalid")
    if (not isinstance(event["skill_bundle"], list) or not event["skill_bundle"] or
            any(type(member.get("skill_id")) is not int or member["skill_id"] < 0 for member in event["skill_bundle"]) or
            event["evidence"] != {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None} or
            event["event_id"] != event_id(event)):
        raise ValueError("event candidate fields invalid")
