"""Canonical schema-v6 MEM-Lite skill / outcome contract.

This module intentionally has no torch, dataset, or simulator dependency.  It
is the single serializer shared by the data builder, loader and model-side
templates.  Keeping the contract here prevents a low-level builder from
silently turning an ordered parallel skill bundle into an arbitrary free-form
string.

Schema v6 is additive.  The old :mod:`memlite_protocol` continues to own the
v5 text-generation grammar and is not reinterpreted here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import json
import math


MEMLITE_SKILL_SCHEMA_VERSION = 6

# This is deliberately the exact observed (skill_id, description) allow-list,
# rather than an NLP normalizer.  A new annotation spelling must be reviewed
# before it can become a trainable action condition.
SKILL_MAPPING: dict[tuple[int, str], str] = {
    (1, "move to"): "NAVIGATE",
    (2, "pick up from"): "GRASP",
    (3, "place on"): "PLACE_ON",
    (4, "place in"): "PLACE_IN",
    (5, "hand over"): "HANDOVER",
    (9, "open drawer"): "OPEN_DRAWER",
    (10, "open door"): "OPEN_DOOR",
    (11, "close drawer"): "CLOSE_DRAWER",
    (12, "close door"): "CLOSE_DOOR",
    (13, "open lid"): "OPEN_LID",
    (14, "close lid"): "CLOSE_LID",
    (67, "press"): "PRESS",
    (90, "push to"): "PUSH",
    (91, "place on next to"): "PLACE_NEXT_TO",
    (93, "turn to"): "TURN_TO",
}
VALID_SKILLS = frozenset({*SKILL_MAPPING.values(), "SKILL_UNKNOWN", "RECOVERY_BRAKE", "RECOVERY_ALIGN", "RECOVERY_APPROACH", "RECOVERY_SETTLE"})
VALID_OUTCOMES = frozenset({"IN_PROGRESS", "SUCCEEDED", "FAILED", "UNKNOWN"})
VALID_DECISIONS = frozenset({"EXECUTE", "RETRY", "REPLAN", "STOP"})
VALID_ARMS = frozenset({"UNSPECIFIED", "LEFT", "RIGHT", "BOTH", "LEFT_TO_RIGHT", "RIGHT_TO_LEFT"})
VALID_BINDING_CONFIDENCE = frozenset({"BOUND", "UNKNOWN"})

# These strings are intentionally explicit: UNKNOWN caused by unavailable
# original-demo physical state must not be accidentally treated as a labelled
# observer training example.
MISSING_PHYSICAL_EVIDENCE = "MISSING_PHYSICAL_EVIDENCE"
RECOVERY_GEOMETRY_EVIDENCE = "RECOVERY_GEOMETRY_TEACHER"

V6_PLANNER_INPUT_FIELDS = (
    "task_name",
    "previous_parent_goal",
    "memory",
    "previous_intent",
    "known_previous_outcome",
    "execution_feedback",
)
V6_PLANNER_TARGET_FIELDS = (
    "target_parent_goal",
    "active_skills_semantic_json",
    "next_decision",
    "memory_update",
    "task_complete",
)
V6_OUTCOME_TARGET_FIELDS = (
    "evaluated_bundle_id",
    "evaluated_skills_json",
    "outcome_target",
    "outcome_supervision_mask",
    "outcome_evidence_end_frame",
    "outcome_available_frame",
    "outcome_evidence_kind",
)

# This is the entire model-facing v6 label contract.  It intentionally does
# not contain raw active bundles, source identifiers, temporal boundaries, or
# annotation provenance.  The dataset stores this mapping as one projection
# and the processor is the only component allowed to flatten it for a model
# builder.  Keep the order stable: tests and cross-tree consumers require an
# exact field set, not a permissive subset.
V6_MODEL_PROJECTION_FIELDS = (
    "schema_version",
    "memlite_branch",
    "task_name",
    "parent_goal",
    "target_parent_goal",
    "previous_parent_goal",
    "memory",
    "previous_intent",
    "known_previous_outcome",
    "execution_feedback",
    "active_skills_semantic_json",
    "active_skills_text",
    "next_decision",
    "memory_update",
    "task_complete",
    "outcome_target",
    "outcome_supervision_mask",
    "parent_goal_supervision_mask",
    "low_action_supervision_mask",
)

# These are source/audit fields, deliberately excluded from deployed semantic
# text and the planner's AR target.  The serialized audit bundle remains in
# Parquet so a review can trace every leaf back to its annotation interval.
V6_AUDIT_ONLY_FIELDS = frozenset({
    "episode_index", "task_index", "task_instance_id", "raw_episode_id",
    "annotation_path", "frame_index", "bundle_id", "segment_start",
    "segment_end", "action_horizon_end", "active_skill_count",
    "requires_parallel", "active_skills_json", "evaluated_bundle_id",
    "evaluated_skills_json", "outcome_target", "outcome_supervision_mask",
    "outcome_evidence_end_frame", "outcome_available_frame",
    "outcome_evidence_kind", "known_previous_outcome_evidence_end_frame",
    "low_action_supervision_mask", "parent_goal_provenance",
    "parent_goal_supervision_mask", "expected_bundle_member_keys_json",
    "bundle_member_keys_json", "annotated_parent_command",
    "annotated_parent_command_provenance", "annotated_parent_command_audit_json",
})

# V5 aliases have different semantics.  A v6 row must not smuggle one into a
# generic overlay which would make templates interpret two competing labels.
V6_LEGACY_CONFLICT_FIELDS = frozenset({
    "branch", "status", "prev_memory", "updated_memory", "atomic_task",
    "future_task", "high_level_instruction", "action_hint", "intent",
    "intent_status",
})


class MemLiteSkillProtocolError(ValueError):
    """A v6 skill/outcome record is malformed or semantically unsafe."""


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8-safe JSON suitable for parquet/text equality.

    Object relations are kept as trees.  In particular, this function never
    flattens nested annotation lists into a bag of object IDs.
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MemLiteSkillProtocolError(f"Value is not canonical JSON: {exc}") from exc


def _clean_text(value: Any, *, field: str, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (dict, list, tuple)):
        raise MemLiteSkillProtocolError(f"{field} must be scalar text, not {type(value).__name__}")
    text = str(value).strip()
    if "\x00" in text:
        raise MemLiteSkillProtocolError(f"{field} contains a NUL byte")
    return text or default


def _optional_int(value: Any, *, field: str, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise MemLiteSkillProtocolError(f"{field} cannot be boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise MemLiteSkillProtocolError(f"{field} is not an integer: {value!r}") from exc
    if not math.isfinite(numeric) or numeric != int(numeric):
        raise MemLiteSkillProtocolError(f"{field} is not a finite integer: {value!r}")
    return int(numeric)


def normalize_description(value: Any) -> str:
    """Exact mapping normalizer: whitespace/case only, never synonym repair."""
    return " ".join(_clean_text(value, field="skill_description").casefold().split())


def skill_from_annotation(skill_id: Any, description: Any) -> str:
    """Map one raw leaf annotation to a reviewed v6 skill enum.

    Unknown pairs remain explicit instead of being coerced to a nearby skill.
    """
    parsed_id = _optional_int(skill_id, field="skill_id")
    if parsed_id is None:
        return "SKILL_UNKNOWN"
    return SKILL_MAPPING.get((parsed_id, normalize_description(description)), "SKILL_UNKNOWN")


def _enum(value: Any, allowed: frozenset[str], *, field: str, default: str) -> str:
    result = _clean_text(value, field=field, default=default).upper()
    if result not in allowed:
        raise MemLiteSkillProtocolError(f"{field}={result!r} is outside {sorted(allowed)!r}")
    return result


def _strict_bool(value: Any, *, field: str, default: bool) -> bool:
    """Accept the parquet boolean type only; never treat ``'false'`` as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise MemLiteSkillProtocolError(f"{field} must be a boolean, not {value!r}")


# Parent plans are actual low-level teacher conditions and high-level targets,
# not an annotation dump. These markers identify metadata which has no runtime
# counterpart (in particular primitive-to-leaf links and raw relation trees).
_PARENT_AUDIT_MARKERS = frozenset({
    "skill_idx", "primitive_idx", "raw_relation", "memory_prefix",
    "object_id", "manipulating_object_id", "annotation teacher",
    "annotated_parent", "provenance", "episode_index", "frame_index",
    "bundle_id", "annotation_path",
})
_PARENT_COMMAND_PREFIX = "Parent command: "
_TASK_GOAL_PREFIX = "Task goal: "


def validate_semantic_parent_goal(
    value: Any,
    *,
    field: str,
    allow_none: bool = False,
) -> str:
    """Validate a deployment-equivalent parent plan, never raw annotation JSON.

    The compact command grammar intentionally has only a natural-language
    description plus entity/relationship values derived from already-bound
    leaf skills. Primitive ``skill_idxes`` and raw relation trees remain in a
    separate audit field and cannot enter a model prefix or target.
    """
    text = _clean_text(value, field=field, default="none" if allow_none else "")
    if allow_none and text.casefold() == "none":
        return "none"
    if not text:
        raise MemLiteSkillProtocolError(f"{field} must be a nonempty semantic parent goal")
    folded = text.casefold()
    if "{" in text or "}" in text or any(marker in folded for marker in _PARENT_AUDIT_MARKERS):
        raise MemLiteSkillProtocolError(
            f"{field} contains annotation/audit-only parent metadata"
        )
    if text.startswith(_TASK_GOAL_PREFIX):
        if not text[len(_TASK_GOAL_PREFIX):].strip():
            raise MemLiteSkillProtocolError(f"{field} task fallback is empty")
        return text
    commands = text.split(" | ")
    required = ("description=", "targets=", "sources=", "destinations=", "target_parts=", "arms=")
    if not commands or any(
        not command.startswith(_PARENT_COMMAND_PREFIX)
        or not all(token in command for token in required)
        for command in commands
    ):
        raise MemLiteSkillProtocolError(
            f"{field} must use compact Parent command grammar or Task goal fallback"
        )
    return text


def semantic_parent_goal_text(
    description: Any,
    member_skills: Sequence[Mapping[str, Any]],
) -> str:
    """Serialize an auditable primitive-derived plan without primitive metadata.

    Entity values come only from leaf fields whose binding was already marked
    ``BOUND``. This preserves useful parent-level object/relationship context
    while refusing to guess from a primitive raw relation tree.
    """
    desc = _clean_text(description, field="parent_description")
    if not desc:
        raise MemLiteSkillProtocolError("semantic parent goal needs a scalar description")
    skills = canonicalize_active_skills(member_skills)
    values: dict[str, list[str]] = {
        "targets": [], "sources": [], "destinations": [],
        "target_parts": [], "arms": [],
    }
    for skill in skills:
        if skill["binding_confidence"] != "BOUND":
            continue
        for key, source_key in (
            ("targets", "target"),
            ("sources", "source"),
            ("destinations", "destination"),
            ("target_parts", "target_part"),
        ):
            item = skill[source_key]
            if item and item not in values[key]:
                values[key].append(item)
        if skill["arm"] != "UNSPECIFIED" and skill["arm"] not in values["arms"]:
            values["arms"].append(skill["arm"])
    # A primitive description with no safely-bound entity/relationship is not
    # a deployment-equivalent command. The builder uses a task fallback and
    # masks parent-plan supervision in that case.
    if not any(values.values()):
        raise MemLiteSkillProtocolError("semantic parent goal has no reliably bound entity/relationship")
    text = _PARENT_COMMAND_PREFIX + "; ".join((
        f"description={canonical_json(desc)}",
        f"targets={canonical_json(values['targets'])}",
        f"sources={canonical_json(values['sources'])}",
        f"destinations={canonical_json(values['destinations'])}",
        f"target_parts={canonical_json(values['target_parts'])}",
        f"arms={canonical_json(values['arms'])}",
    ))
    return validate_semantic_parent_goal(text, field="semantic_parent_goal")


def _raw_relation(value: Any) -> dict[str, Any]:
    if value is None:
        return {"object_id": None, "spatial_prefix": None, "memory_prefix": None, "manipulating_object_id": None}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MemLiteSkillProtocolError("raw_relation_json is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise MemLiteSkillProtocolError("raw_relation must be an object")
    # Preserve the complete relation/provenance tree for audit; normalized
    # aliases help the binding parser but never discard the original spelling.
    result = dict(value)
    result.setdefault("object_id", value.get("object_ids"))
    result.setdefault("spatial_prefix", None)
    result.setdefault("memory_prefix", None)
    result.setdefault("manipulating_object_id", None)
    return result


def canonicalize_active_skill(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one leaf skill without inferring missing object bindings."""
    if not isinstance(value, Mapping):
        raise MemLiteSkillProtocolError("active skill must be an object")
    skill_id = _optional_int(value.get("skill_id"), field="skill_id")
    skill_idx = _optional_int(value.get("skill_idx"), field="skill_idx")
    interval_id = _optional_int(value.get("interval_id"), field="interval_id", default=0)
    raw_description = _clean_text(value.get("raw_description", value.get("skill_description", "")), field="raw_description")
    derived = skill_from_annotation(skill_id, raw_description)
    requested = _clean_text(value.get("verb", value.get("active_skill", derived)), field="verb", default=derived).upper()
    if requested not in VALID_SKILLS:
        raise MemLiteSkillProtocolError(f"Unknown v6 skill enum {requested!r}")
    # For normal demo leaves, enum must equal the exact reviewed map.  The
    # recovery namespace has no raw BEHAVIOR skill_id and is validated by its
    # dedicated converter.
    if requested not in {"RECOVERY_BRAKE", "RECOVERY_ALIGN", "RECOVERY_APPROACH", "RECOVERY_SETTLE"} and requested != derived:
        raise MemLiteSkillProtocolError(
            f"verb {requested!r} disagrees with exact raw pair ({skill_id!r}, {raw_description!r}) -> {derived!r}"
        )
    relation = _raw_relation(value.get("raw_relation", value.get("raw_relation_json")))
    confidence = _enum(value.get("binding_confidence"), VALID_BINDING_CONFIDENCE,
                       field="binding_confidence", default="UNKNOWN")
    target = _clean_text(value.get("target"), field="target")
    source = _clean_text(value.get("source"), field="source")
    destination = _clean_text(value.get("destination"), field="destination")
    target_part = _clean_text(value.get("target_part"), field="target_part")
    arm = _enum(value.get("arm"), VALID_ARMS, field="arm", default="UNSPECIFIED")
    if confidence == "UNKNOWN" and any((target, source, destination, target_part, arm != "UNSPECIFIED")):
        raise MemLiteSkillProtocolError("unbound relation may not claim target/source/destination/part/arm")
    if confidence == "BOUND":
        if requested == "NAVIGATE" and not target:
            raise MemLiteSkillProtocolError("BOUND NAVIGATE needs target")
        if requested == "GRASP" and (not target or not source):
            raise MemLiteSkillProtocolError("BOUND GRASP needs target and source")
        if requested in {"PLACE_ON", "PLACE_IN", "PLACE_NEXT_TO", "PUSH"} and (not target or not destination):
            raise MemLiteSkillProtocolError(f"BOUND {requested} needs target and destination")
        if requested in {"OPEN_DRAWER", "OPEN_DOOR", "CLOSE_DRAWER", "CLOSE_DOOR", "OPEN_LID", "CLOSE_LID", "PRESS", "TURN_TO"} and not target:
            raise MemLiteSkillProtocolError(f"BOUND {requested} needs target")
        if requested == "HANDOVER" and (not target or arm not in {"LEFT_TO_RIGHT", "RIGHT_TO_LEFT"}):
            raise MemLiteSkillProtocolError("BOUND HANDOVER needs target and directed arms")
    start = _optional_int(value.get("skill_start"), field="skill_start")
    end = _optional_int(value.get("skill_end"), field="skill_end")
    if (start is None) != (end is None):
        raise MemLiteSkillProtocolError("skill_start and skill_end must be supplied together")
    if start is not None and end <= start:
        raise MemLiteSkillProtocolError("skill interval must be non-empty and half-open")
    return {
        "skill_id": skill_id,
        "skill_idx": skill_idx,
        "interval_id": interval_id,
        "verb": requested,
        "raw_description": raw_description,
        "target": target,
        "source": source,
        "destination": destination,
        "target_part": target_part,
        "arm": arm,
        "binding_confidence": confidence,
        "raw_relation": relation,
        "skill_start": start,
        "skill_end": end,
    }


def canonicalize_active_skills(values: Sequence[Mapping[str, Any]], *, allow_empty: bool = False) -> list[dict[str, Any]]:
    """Canonicalize an ordered, complete parallel bundle without deduplication."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MemLiteSkillProtocolError("active_skills must be a sequence of objects")
    result = [canonicalize_active_skill(value) for value in values]
    if not result and not allow_empty:
        raise MemLiteSkillProtocolError("low-level active_skills cannot be empty")
    # Skill index is the source-order authority.  Two real parallel skills may
    # share verb, interval and relation; do not deduplicate them.
    ordered = sorted(enumerate(result), key=lambda item: (
        item[1]["skill_idx"] is None,
        item[1]["skill_idx"] if item[1]["skill_idx"] is not None else 10**12,
        item[1]["interval_id"] if item[1]["interval_id"] is not None else 10**12,
        item[0],
    ))
    result = [value for _, value in ordered]
    member_keys = [(item["skill_idx"], item["interval_id"]) for item in result]
    if len(set(member_keys)) != len(member_keys):
        raise MemLiteSkillProtocolError("parallel bundle repeats a skill_idx/interval_id member")
    return result


def serialize_active_skills(values: Sequence[Mapping[str, Any]], *, allow_empty: bool = False) -> str:
    return canonical_json(canonicalize_active_skills(values, allow_empty=allow_empty))


def parse_active_skills_json(value: Any, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MemLiteSkillProtocolError("active_skills_json is invalid JSON") from exc
    result = canonicalize_active_skills(value, allow_empty=allow_empty)
    # A noncanonical payload is a source-data error.  This catches manual
    # changes such as reordering a parallel bundle after its text condition was
    # emitted.
    if isinstance(value, list) and canonical_json(value) != canonical_json(result):
        raise MemLiteSkillProtocolError("active_skills_json is not canonical source order")
    return result


def semantic_active_skills(values: Sequence[Mapping[str, Any]], *, allow_empty: bool = False) -> list[dict[str, str]]:
    """Project an audit bundle into fields available at deployment.

    Annotation IDs, frame offsets and builder-only binding confidence are
    excluded.  An unbound leaf receives a *strict* semantic relation
    projection: stable entity references plus a reviewed controlled spatial
    vocabulary.  It never receives a filtered copy of the raw relation tree:
    that tree can contain annotation history, primitive links, or future-only
    provenance which have no deployment-time counterpart.  The full tree
    remains in ``active_skills_json`` for audit only.
    """
    result = canonicalize_active_skills(values, allow_empty=allow_empty)
    projected: list[dict[str, str]] = []
    for skill in result:
        item = {field: str(skill[field]) for field in ("verb", "target", "source", "destination", "target_part", "arm")}
        item["unbound_relation"] = canonical_json(_semantic_relation(skill["raw_relation"])) if skill["binding_confidence"] == "UNKNOWN" else ""
        projected.append(item)
    return projected


_UNBOUND_RELATION_SOURCE_KEYS = frozenset({
    "manipulating_object_id", "object_id", "spatial_prefix", "memory_prefix",
})
# This is the entire vocabulary observed in the frozen five-task raw v6
# sidecar (see ``memlite_v6_unbound_relation_full_inventory_20260909.json``).
# Adding a new relation is a schema review, never an implicit text passthrough.
_UNBOUND_SPATIAL_RELATIONS = frozenset({"in_front_of", "face"})


def _relation_text_leaves(value: Any, *, field: str) -> list[str]:
    """Flatten a source relation sequence only after rejecting structure.

    ``raw_relation`` is deliberately heterogeneous annotation data.  Dicts,
    numbers and arbitrary objects have no unambiguous deployment semantics;
    accepting them and stringifying them would recreate the audit leak that
    this projection is intended to prevent.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value, field=field)
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for child in value:
            flattened.extend(_relation_text_leaves(child, field=field))
        return flattened
    raise MemLiteSkillProtocolError(
        f"unbound raw_relation {field} must contain only text sequences"
    )


def _semantic_relation(value: Any) -> dict[str, list[str]]:
    """Whitelist a raw unknown-binding relation into deployment semantics.

    ``memory_prefix`` is retained in the audit JSON but intentionally omitted
    even when it is empty.  It is annotation history, not robot-observable
    memory.  No recursive blacklist is used: any unreviewed top-level field,
    nested mapping, index, time, provenance, or arbitrary spatial relation
    fails closed rather than appearing in prompt text.
    """
    if not isinstance(value, Mapping):
        raise MemLiteSkillProtocolError("unbound raw_relation must be an object")
    if any(not isinstance(key, str) for key in value):
        raise MemLiteSkillProtocolError("unbound raw_relation keys must be text")
    keys = set(value)
    unsupported = sorted(keys - _UNBOUND_RELATION_SOURCE_KEYS)
    if unsupported:
        raise MemLiteSkillProtocolError(
            "unbound raw_relation has non-semantic/audit fields: "
            f"{unsupported!r}"
        )

    entities: list[str] = []
    for source_key in ("manipulating_object_id", "object_id"):
        for entity in _relation_text_leaves(value.get(source_key), field=source_key):
            if entity not in entities:
                entities.append(entity)

    spatial_relations: list[str] = []
    for relation in _relation_text_leaves(value.get("spatial_prefix"), field="spatial_prefix"):
        normalized = relation.casefold()
        if normalized not in _UNBOUND_SPATIAL_RELATIONS:
            raise MemLiteSkillProtocolError(
                "unbound raw_relation has unreviewed spatial relation "
                f"{relation!r}"
            )
        if normalized not in spatial_relations:
            spatial_relations.append(normalized)

    # Validate only enough to reject a malformed JSON-like value.  Its
    # contents are never returned, so raw annotation memory cannot enter the
    # model semantic projection under any spelling or nesting.
    memory = value.get("memory_prefix")
    if memory is not None and not isinstance(memory, (str, list, tuple)):
        raise MemLiteSkillProtocolError("unbound raw_relation memory_prefix must be scalar text or a sequence")
    return {"entities": entities, "spatial_relations": spatial_relations}


def serialize_semantic_active_skills(values: Sequence[Mapping[str, Any]], *, allow_empty: bool = False) -> str:
    return canonical_json(semantic_active_skills(values, allow_empty=allow_empty))


def active_skills_text(values: Sequence[Mapping[str, Any]], *, allow_empty: bool = False) -> str:
    """Stable deployment-equivalent VLM conditioning for a parallel bundle."""
    result = semantic_active_skills(values, allow_empty=allow_empty)
    if not result:
        return "Active skills: none."
    items = []
    for skill in result:
        # JSON scalar quoting makes separators in an object name unambiguous
        # (e.g. ``plate; arm=LEFT`` cannot forge a second field).
        content = "; ".join(
            f"{field}={canonical_json(skill[field] or 'NONE')}"
            for field in ("verb", "target", "source", "destination", "target_part", "arm")
        )
        if skill["unbound_relation"]:
            content += f"; unbound_relation={skill['unbound_relation']}"
        items.append(f"[{content}]")
    return "Active skills: " + "; ".join(items) + "."


def planner_input_projection(row: Mapping[str, Any]) -> dict[str, str]:
    """Only causal, deployment-available planner prompt fields."""
    result = {field: _clean_text(row.get(field), field=field, default="none") for field in V6_PLANNER_INPUT_FIELDS}
    result["previous_parent_goal"] = validate_semantic_parent_goal(
        result["previous_parent_goal"], field="previous_parent_goal", allow_none=True
    )
    return result


def planner_target_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Masked high-level target, without source indices or audit JSON."""
    values = parse_active_skills_json(row.get("active_skills_json", "[]"), allow_empty=True)
    return {
        "target_parent_goal": validate_semantic_parent_goal(
            row.get("target_parent_goal"), field="target_parent_goal"
        ),
        "active_skills_semantic_json": serialize_semantic_active_skills(values, allow_empty=True),
        "next_decision": _enum(row.get("next_decision"), VALID_DECISIONS, field="next_decision", default="STOP"),
        "memory_update": _clean_text(row.get("memory_update"), field="memory_update"),
        "task_complete": _strict_bool(row.get("task_complete"), field="task_complete", default=False),
    }


def low_condition_projection(row: Mapping[str, Any]) -> dict[str, str]:
    """Teacher condition for low-level training; same shape runtime receives."""
    values = parse_active_skills_json(row.get("active_skills_json", "[]"))
    return {
        "parent_goal": validate_semantic_parent_goal(
            row.get("parent_goal"), field="parent_goal"
        ),
        "active_skills_semantic_json": serialize_semantic_active_skills(values),
    }


def outcome_target_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Outcome loss fields only; never planner-prompt fields."""
    return {field: row.get(field) for field in V6_OUTCOME_TARGET_FIELDS}


def default_unobserved_outcome_fields() -> dict[str, Any]:
    """Outcome defaults for original demonstrations lacking physical evidence."""
    return {
        "known_previous_outcome": "UNKNOWN",
        "known_previous_outcome_evidence_end_frame": -1,
        "outcome_target": "UNKNOWN",
        "outcome_supervision_mask": False,
        "outcome_evidence_end_frame": -1,
        "outcome_available_frame": -1,
        "outcome_evidence_kind": MISSING_PHYSICAL_EVIDENCE,
        "evaluated_bundle_id": "",
        "evaluated_skills_json": "[]",
    }


def validate_outcome_fields(row: Mapping[str, Any], *, frame_index: int) -> dict[str, Any]:
    """Validate causal separation between planner-known and target outcomes."""
    defaults = default_unobserved_outcome_fields()
    known = _enum(row.get("known_previous_outcome"), VALID_OUTCOMES,
                  field="known_previous_outcome", default=defaults["known_previous_outcome"])
    known_end = _optional_int(row.get("known_previous_outcome_evidence_end_frame"),
                              field="known_previous_outcome_evidence_end_frame", default=-1)
    # ``known_previous_outcome`` is an input available before the current
    # decision.  Even explicit UNKNOWN may be evidence-backed, but no known
    # input may point to a current/future observation.
    if known_end is None or known_end < -1 or known_end >= frame_index:
        raise MemLiteSkillProtocolError("known previous outcome evidence must be absent (-1) or strictly past")
    if known != "UNKNOWN" and known_end < 0:
        raise MemLiteSkillProtocolError("known non-UNKNOWN outcome needs strictly past evidence")
    target = _enum(row.get("outcome_target"), VALID_OUTCOMES, field="outcome_target",
                   default=defaults["outcome_target"])
    mask = _strict_bool(row.get("outcome_supervision_mask"), field="outcome_supervision_mask",
                        default=defaults["outcome_supervision_mask"])
    evidence_end = _optional_int(row.get("outcome_evidence_end_frame"), field="outcome_evidence_end_frame", default=-1)
    available = _optional_int(row.get("outcome_available_frame"), field="outcome_available_frame", default=-1)
    kind = _clean_text(row.get("outcome_evidence_kind"), field="outcome_evidence_kind",
                       default=defaults["outcome_evidence_kind"])
    evaluated = parse_active_skills_json(row.get("evaluated_skills_json", defaults["evaluated_skills_json"]), allow_empty=True)
    evaluated_bundle_id = _clean_text(row.get("evaluated_bundle_id"), field="evaluated_bundle_id")
    if not mask:
        if target != "UNKNOWN" or evidence_end != -1 or available != -1 or kind != MISSING_PHYSICAL_EVIDENCE:
            raise MemLiteSkillProtocolError("unobserved demo outcome must be UNKNOWN with mask=false and no evidence")
        if evaluated or evaluated_bundle_id:
            raise MemLiteSkillProtocolError("unobserved demo outcome may not claim evaluated skills")
    else:
        if evidence_end is None or evidence_end < 0 or evidence_end > frame_index:
            raise MemLiteSkillProtocolError("outcome target evidence must end at or before current frame")
        if available is None or available < evidence_end or available > frame_index:
            raise MemLiteSkillProtocolError("outcome availability must be after evidence and no later than this row")
        if kind == MISSING_PHYSICAL_EVIDENCE:
            raise MemLiteSkillProtocolError("supervised outcome requires a real evidence kind")
        if not evaluated or not evaluated_bundle_id:
            raise MemLiteSkillProtocolError("supervised outcome needs an explicit prior evaluated bundle")
        current_bundle_id = _clean_text(row.get("bundle_id"), field="bundle_id")
        if current_bundle_id and evaluated_bundle_id == current_bundle_id:
            raise MemLiteSkillProtocolError("outcome target must evaluate a prior bundle, not the current active bundle")
    return {
        "known_previous_outcome": known,
        "known_previous_outcome_evidence_end_frame": known_end,
        "outcome_target": target,
        "outcome_supervision_mask": mask,
        "outcome_evidence_end_frame": evidence_end,
        "outcome_available_frame": available,
        "outcome_evidence_kind": kind,
        "evaluated_bundle_id": evaluated_bundle_id,
        "evaluated_skills_json": serialize_active_skills(evaluated, allow_empty=True),
    }


def validate_v6_label(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a v6 sidecar row without changing v5 semantics."""
    conflicts = sorted(V6_LEGACY_CONFLICT_FIELDS & set(row))
    if conflicts:
        raise MemLiteSkillProtocolError(f"schema-v6 row carries conflicting v5 fields: {conflicts!r}")
    version = _optional_int(row.get("schema_version"), field="schema_version")
    if version != MEMLITE_SKILL_SCHEMA_VERSION:
        raise MemLiteSkillProtocolError(f"Expected schema_version={MEMLITE_SKILL_SCHEMA_VERSION}, got {version!r}")
    frame = _optional_int(row.get("frame_index", row.get("frame")), field="frame_index")
    if frame is None or frame < 0:
        raise MemLiteSkillProtocolError("v6 row requires nonnegative frame_index")
    branch = _clean_text(row.get("memlite_branch", row.get("branch")), field="memlite_branch").lower()
    if branch not in {"high", "low"}:
        raise MemLiteSkillProtocolError("v6 row requires high or low memlite_branch")
    values = parse_active_skills_json(row.get("active_skills_json", "[]"), allow_empty=(branch == "high"))
    result = dict(row)
    result.update(
        schema_version=MEMLITE_SKILL_SCHEMA_VERSION,
        frame_index=frame,
        memlite_branch=branch,
        active_skills_json=serialize_active_skills(values, allow_empty=(branch == "high")),
        active_skills_text=active_skills_text(values, allow_empty=(branch == "high")),
        active_skill_count=len(values),
        requires_parallel=len(values) > 1,
        parent_goal=validate_semantic_parent_goal(row.get("parent_goal"), field="parent_goal"),
        previous_parent_goal=validate_semantic_parent_goal(
            row.get("previous_parent_goal"), field="previous_parent_goal", allow_none=True
        ),
        next_decision=_enum(row.get("next_decision"), VALID_DECISIONS,
                            field="next_decision", default="STOP" if not values else "EXECUTE"),
        task_complete=_strict_bool(row.get("task_complete"), field="task_complete", default=False),
    )
    bundle_id = _clean_text(row.get("bundle_id"), field="bundle_id")
    if not bundle_id:
        raise MemLiteSkillProtocolError("v6 row needs a nonempty audit bundle_id")
    result["bundle_id"] = bundle_id
    member_keys = canonical_json([
        {"skill_idx": skill["skill_idx"], "interval_id": skill["interval_id"]}
        for skill in values
    ])
    declared_member_keys = _clean_text(
        row.get("expected_bundle_member_keys_json"), field="expected_bundle_member_keys_json"
    )
    if not declared_member_keys:
        raise MemLiteSkillProtocolError("v6 row needs source-emitted expected_bundle_member_keys_json")
    if declared_member_keys != member_keys:
        raise MemLiteSkillProtocolError("expected_bundle_member_keys_json disagrees with complete active_skills bundle")
    legacy_member_keys = row.get("bundle_member_keys_json")
    if legacy_member_keys is not None and _clean_text(legacy_member_keys, field="bundle_member_keys_json") != member_keys:
        raise MemLiteSkillProtocolError("bundle_member_keys_json disagrees with complete active_skills bundle")
    result["bundle_member_keys_json"] = member_keys
    result["expected_bundle_member_keys_json"] = declared_member_keys
    if branch == "low":
        if not values or result["task_complete"] or result["next_decision"] != "EXECUTE":
            raise MemLiteSkillProtocolError("low v6 row needs nonempty active skills, EXECUTE, and task_complete=false")
    elif not values:
        if not result["task_complete"] or result["next_decision"] != "STOP":
            raise MemLiteSkillProtocolError("empty high v6 terminal row requires task_complete=true and STOP")
    elif result["task_complete"] or result["next_decision"] == "STOP":
        raise MemLiteSkillProtocolError("nonempty high v6 row cannot be terminal/STOP")
    low_action_supervision_mask = _strict_bool(
        row.get("low_action_supervision_mask"), field="low_action_supervision_mask", default=True
    )
    if branch == "low" and any(skill["verb"] == "SKILL_UNKNOWN" for skill in values) and low_action_supervision_mask:
        raise MemLiteSkillProtocolError("low SKILL_UNKNOWN bundle must have low_action_supervision_mask=false")
    result["low_action_supervision_mask"] = low_action_supervision_mask
    # A low action target must stop at the earliest active leaf boundary,
    # including true parallel bundles with different end frames.
    if values:
        ends = [skill["skill_end"] for skill in values if skill["skill_end"] is not None]
        if ends:
            earliest = min(ends)
            if frame >= earliest:
                raise MemLiteSkillProtocolError("v6 row frame must precede every active skill boundary")
            declared = _optional_int(row.get("action_horizon_end", row.get("segment_end")),
                                     field="action_horizon_end", default=earliest)
            if declared != earliest:
                raise MemLiteSkillProtocolError("v6 action_horizon_end must equal earliest active skill end")
            result["action_horizon_end"] = earliest
            result["segment_end"] = earliest
    result["active_skills_semantic_json"] = serialize_semantic_active_skills(
        values, allow_empty=(branch == "high")
    )
    result["target_parent_goal"] = validate_semantic_parent_goal(
        row.get("target_parent_goal", row.get("parent_goal")), field="target_parent_goal"
    )
    result["parent_goal_supervision_mask"] = _strict_bool(
        row.get("parent_goal_supervision_mask"), field="parent_goal_supervision_mask", default=True
    )
    result.update(validate_outcome_fields(row, frame_index=frame))
    return result
