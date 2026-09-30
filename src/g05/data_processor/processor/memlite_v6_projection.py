# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""Unchanged schema-v6 projection guards from the frozen A4 runtime.

Separated from its old sample builders so the throughput-only port does not
replace unrelated/current builders. This is the old 15-skill contract, not
a release of the 2026 35-skill dataset interface.
"""
from collections.abc import Mapping
from typing import Dict, Any, List, Optional
import json
import logging

import torch

from g05.utils.memlite_skill_protocol import (
    MEMLITE_SKILL_SCHEMA_VERSION,
    MemLiteSkillProtocolError,
    V6_MODEL_PROJECTION_FIELDS,
    planner_input_projection,
    VALID_ARMS,
    VALID_DECISIONS,
    VALID_OUTCOMES,
    VALID_SKILLS,
    canonical_json,
    validate_semantic_parent_goal,
)

logger = logging.getLogger(__name__)


# The sidecar retains ``active_skills_json`` so annotation provenance can be
# audited offline.  It is deliberately *not* a model field: it contains
# skill/frame identifiers and raw relation trees with no serving equivalent.
# Dataset code validates that source record and creates the compact semantic
# projection before it invokes a SamplesBuilder.  The builder repeats a
# narrow, fail-closed validation here rather than accepting the audit record
# and relying on a later template to omit it.  This makes an accidental
# processor/collate copy a hard error before tokenization.
_MODEL_FORBIDDEN_AUDIT_FIELDS = frozenset({
    "active_skills_json", "evaluated_skills_json",
    "expected_bundle_member_keys_json", "bundle_member_keys_json",
    "annotated_parent_command", "annotated_parent_command_provenance",
    "annotated_parent_command_audit_json", "parent_goal_provenance",
    "bundle_id", "episode_index", "raw_episode_id", "task_index",
    "task_instance_id", "annotation_path", "frame_index", "segment_start",
    "segment_end", "action_horizon_end", "active_skill_count",
    "requires_parallel", "outcome_evidence_end_frame",
    "outcome_available_frame", "outcome_evidence_kind",
    "known_previous_outcome_evidence_end_frame", "evaluated_bundle_id",
    # Dataset tuple locators are useful to an outer sampler/receipt resolver,
    # but are not a model condition and must not reach the builder either.
    "idx", "dataset_locator", "memlite_requested_index",
    "memlite_requested_branch", "memlite_actual_branch",
    "memlite_source_episode_index", "memlite_source_frame_index",
})
_SEMANTIC_SKILL_FIELDS = (
    "verb", "target", "source", "destination", "target_part", "arm",
    "unbound_relation",
)
# Imported from the paired data protocol rather than duplicated locally: a
# field addition must now break the candidate's byte-pinned integration test
# instead of silently drifting a model-only allowlist.
_MEMLITE_MODEL_PROJECTION_FIELDS = frozenset(V6_MODEL_PROJECTION_FIELDS)
_SEMANTIC_RELATION_FORBIDDEN_KEYS = frozenset({
    "skill_id", "skill_idx", "primitive_idx", "interval_id", "skill_start",
    "skill_end", "raw_description", "raw_relation", "binding_confidence",
    "episode_index", "frame_index", "bundle_id", "annotation_path",
    "raw_episode_id", "task_index", "task_instance_id", "provenance",
    "memory_prefix", "object_id", "manipulating_object_id", "evidence",
    "evidence_id", "dataset_locator", "review_id",
})


def assert_model_projection_has_no_audit(data: Mapping[str, Any]) -> None:
    """Shared policy/builder guard for flat model-visible sample mappings."""
    forbidden = sorted(_MODEL_FORBIDDEN_AUDIT_FIELDS & set(data))
    if forbidden:
        raise MemLiteSkillProtocolError(
            f"schema-v6 model projection carries audit-only fields: {forbidden!r}"
        )


def _safe_scalar_text(value: Any, *, field: str, default: str | None = None) -> str:
    """Accept only scalar deployment text; mappings cannot hide audit JSON."""
    if value is None:
        if default is None:
            raise MemLiteSkillProtocolError(f"{field} is required")
        return default
    if isinstance(value, (Mapping, list, tuple, set)):
        raise MemLiteSkillProtocolError(f"{field} must be scalar text")
    text = str(value).strip()
    if not text and default is not None:
        return default
    if not text or "\x00" in text:
        raise MemLiteSkillProtocolError(f"{field} must be nonempty scalar text")
    return text


def _strict_projection_bool(value: Any, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise MemLiteSkillProtocolError(f"{field} must be a boolean")
    return value


def _reject_audit_keys(value: Any, *, field: str) -> None:
    """Reject identity/provenance nested in the one allowed relation field."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _SEMANTIC_RELATION_FORBIDDEN_KEYS:
                raise MemLiteSkillProtocolError(
                    f"{field} carries audit-only relation key {key!r}"
                )
            _reject_audit_keys(child, field=field)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_audit_keys(child, field=field)


def _semantic_bundle_from_model_projection(
    data: Mapping[str, Any], *, allow_empty: bool
) -> tuple[list[dict[str, str]], str, str]:
    """Parse the compact v6 bundle emitted by the data-side allowlist.

    The full audit representation is purposely unavailable here.  Canonical
    JSON plus a regenerated text rendering binds the semantic JSON and the
    human-readable VLM condition to exactly the same ordered parallel bundle.
    """
    raw_json = _safe_scalar_text(data.get("active_skills_semantic_json"), field="active_skills_semantic_json")
    try:
        decoded = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise MemLiteSkillProtocolError("active_skills_semantic_json is invalid JSON") from exc
    if not isinstance(decoded, list) or (not decoded and not allow_empty):
        raise MemLiteSkillProtocolError("semantic active-skills bundle has invalid emptiness")

    semantic: list[dict[str, str]] = []
    for index, item in enumerate(decoded):
        if not isinstance(item, Mapping) or set(item) != set(_SEMANTIC_SKILL_FIELDS):
            raise MemLiteSkillProtocolError(
                f"semantic active skill {index} must contain exactly {_SEMANTIC_SKILL_FIELDS}"
            )
        parsed: dict[str, str] = {}
        for key in _SEMANTIC_SKILL_FIELDS:
            value = item[key]
            if not isinstance(value, str):
                raise MemLiteSkillProtocolError(f"semantic active skill {index}.{key} must be text")
            if "\x00" in value:
                raise MemLiteSkillProtocolError(f"semantic active skill {index}.{key} contains NUL")
            parsed[key] = value
        parsed["verb"] = parsed["verb"].upper()
        parsed["arm"] = parsed["arm"].upper()
        if parsed["verb"] not in VALID_SKILLS:
            raise MemLiteSkillProtocolError(f"semantic active skill {index} has invalid verb")
        if parsed["arm"] not in VALID_ARMS:
            raise MemLiteSkillProtocolError(f"semantic active skill {index} has invalid arm")
        relation = parsed["unbound_relation"]
        if relation:
            try:
                relation_value = json.loads(relation)
            except json.JSONDecodeError as exc:
                raise MemLiteSkillProtocolError(
                    f"semantic active skill {index}.unbound_relation is invalid JSON"
                ) from exc
            if canonical_json(relation_value) != relation:
                raise MemLiteSkillProtocolError(
                    f"semantic active skill {index}.unbound_relation is not canonical JSON"
                )
            _reject_audit_keys(relation_value, field=f"semantic active skill {index}.unbound_relation")
        semantic.append(parsed)

    semantic_json = canonical_json(semantic)
    if raw_json != semantic_json:
        raise MemLiteSkillProtocolError("active_skills_semantic_json is not canonical")
    if not semantic:
        rendered = "Active skills: none."
    else:
        rendered_items = []
        for skill in semantic:
            content = "; ".join(
                f"{field}={canonical_json(skill[field] or 'NONE')}"
                for field in ("verb", "target", "source", "destination", "target_part", "arm")
            )
            if skill["unbound_relation"]:
                content += f"; unbound_relation={skill['unbound_relation']}"
            rendered_items.append(f"[{content}]")
        rendered = "Active skills: " + "; ".join(rendered_items) + "."
    source_text = _safe_scalar_text(data.get("active_skills_text"), field="active_skills_text")
    if source_text != rendered:
        raise MemLiteSkillProtocolError(
            "active_skills_text does not exactly match active_skills_semantic_json"
        )
    return semantic, semantic_json, rendered


def _model_safe_v6_label(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the data-owned, audit-free schema-v6 projection.

    Full source-label validation is intentionally performed before this point
    by the sidecar/dataset.  This function validates only fields that are
    legal model conditions or supervision targets, and rejects every audit
    field if a processor accidentally forwards it into a builder.
    """
    assert_model_projection_has_no_audit(data)
    actual_fields = set(data)
    if actual_fields != _MEMLITE_MODEL_PROJECTION_FIELDS:
        raise MemLiteSkillProtocolError(
            "schema-v6 model projection must use the exact 19-field allowlist; "
            f"missing={sorted(_MEMLITE_MODEL_PROJECTION_FIELDS - actual_fields)!r}; "
            f"extra={sorted(actual_fields - _MEMLITE_MODEL_PROJECTION_FIELDS)!r}"
        )
    version = data.get("schema_version")
    try:
        version_ok = not isinstance(version, bool) and int(version) == MEMLITE_SKILL_SCHEMA_VERSION
    except (TypeError, ValueError):
        version_ok = False
    if not version_ok:
        raise MemLiteSkillProtocolError(
            f"expected schema_version={MEMLITE_SKILL_SCHEMA_VERSION}, got {version!r}"
        )
    branch = _safe_scalar_text(data.get("memlite_branch"), field="memlite_branch").lower()
    if branch not in {"low", "high"}:
        raise MemLiteSkillProtocolError("model projection requires low or high memlite_branch")
    active_skills, semantic_json, semantic_text = _semantic_bundle_from_model_projection(
        data, allow_empty=(branch == "high")
    )
    next_decision = _safe_scalar_text(data.get("next_decision"), field="next_decision").upper()
    if next_decision not in VALID_DECISIONS:
        raise MemLiteSkillProtocolError(f"invalid next_decision {next_decision!r}")
    task_complete = _strict_projection_bool(data.get("task_complete"), field="task_complete", default=False)
    result: dict[str, Any] = {
        "schema_version": MEMLITE_SKILL_SCHEMA_VERSION,
        "memlite_branch": branch,
        "active_skills": active_skills,
        "active_skills_semantic_json": semantic_json,
        "active_skills_text": semantic_text,
        "parent_goal": validate_semantic_parent_goal(data.get("parent_goal"), field="parent_goal"),
        "next_decision": next_decision,
        "task_complete": task_complete,
    }
    if branch == "low":
        low_mask = _strict_projection_bool(
            data.get("low_action_supervision_mask"), field="low_action_supervision_mask", default=True
        )
        if not active_skills or task_complete or next_decision != "EXECUTE":
            raise MemLiteSkillProtocolError("low model projection needs nonterminal EXECUTE active skills")
        if any(skill["verb"] == "SKILL_UNKNOWN" for skill in active_skills):
            raise MemLiteSkillProtocolError("SKILL_UNKNOWN cannot enter low FM supervision")
        if not low_mask:
            raise MemLiteSkillProtocolError("low_action_supervision_mask=false may not enter FM")
        result["low_action_supervision_mask"] = low_mask
        return result

    if not active_skills:
        if not task_complete or next_decision != "STOP":
            raise MemLiteSkillProtocolError("empty high model projection requires terminal STOP")
    elif task_complete or next_decision == "STOP":
        raise MemLiteSkillProtocolError("nonempty high model projection cannot be terminal/STOP")
    result.update({
        "task_name": _safe_scalar_text(data.get("task_name"), field="task_name"),
        "memory": _safe_scalar_text(data.get("memory"), field="memory", default="none"),
        "previous_intent": _safe_scalar_text(data.get("previous_intent"), field="previous_intent", default="none"),
        "previous_parent_goal": validate_semantic_parent_goal(
            data.get("previous_parent_goal"), field="previous_parent_goal", allow_none=True
        ),
        "known_previous_outcome": _safe_scalar_text(
            data.get("known_previous_outcome"), field="known_previous_outcome", default="UNKNOWN"
        ).upper(),
        "execution_feedback": _safe_scalar_text(
            data.get("execution_feedback"), field="execution_feedback", default="none"
        ),
        "target_parent_goal": validate_semantic_parent_goal(
            data.get("target_parent_goal"), field="target_parent_goal"
        ),
        "memory_update": _safe_scalar_text(data.get("memory_update"), field="memory_update", default="none"),
        "parent_goal_supervision_mask": _strict_projection_bool(
            data.get("parent_goal_supervision_mask"), field="parent_goal_supervision_mask", default=True
        ),
        "outcome_supervision_mask": _strict_projection_bool(
            data.get("outcome_supervision_mask"), field="outcome_supervision_mask", default=False
        ),
    })
    if result["known_previous_outcome"] not in VALID_OUTCOMES:
        raise MemLiteSkillProtocolError("invalid known_previous_outcome")
    result["outcome_target"] = _safe_scalar_text(
        data.get("outcome_target"), field="outcome_target", default="UNKNOWN"
    ).upper()
    if result["outcome_target"] not in VALID_OUTCOMES:
        raise MemLiteSkillProtocolError("invalid outcome_target")
    if not result["outcome_supervision_mask"] and result["outcome_target"] != "UNKNOWN":
        raise MemLiteSkillProtocolError("unobserved outcome model target must remain UNKNOWN")
    return result


def validate_embedded_model_projection(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact projection after a builder adds template/image keys.

    A built model sample necessarily contains additional generated values such
    as ``template``, ``proprio`` and ``image0``.  Extracting the 19 declared
    fields keeps that legitimate post-build shape distinct from data-side
    projection widening, while the shared audit guard still protects the
    whole container.
    """
    assert_model_projection_has_no_audit(sample)
    missing = sorted(_MEMLITE_MODEL_PROJECTION_FIELDS - set(sample))
    if missing:
        raise MemLiteSkillProtocolError(
            f"built MEM-Lite sample is missing model-projection fields: {missing!r}"
        )
    return _model_safe_v6_label({key: sample[key] for key in _MEMLITE_MODEL_PROJECTION_FIELDS})
