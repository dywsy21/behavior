#!/usr/bin/env python3
"""Build answer-independent, prelabel visual-relation queries for P107 phase40.

The producer joins the sealed phase queue to its exact mini-index event record,
pins every source hash, and emits one record per query (not one mutable record
per event).  It never emits RGB evidence references, answers, labels, action
payloads, training rows, or postlabel canonical view IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


# This is the official 35-skill projection recorded by
# index-validation/skill-vocabulary-map.v1.json.  The pair (ID, description)
# is the identity; canonical_verb is not inferred by a synonym model.
CANONICAL35: dict[int, tuple[str, str]] = {
    1: ("move to", "NAVIGATE"),
    2: ("pick up from", "GRASP"),
    3: ("place on", "PLACE_ON"),
    4: ("place in", "PLACE_IN"),
    5: ("hand over", "HANDOVER"),
    6: ("insert", "INSERT"),
    8: ("release", "RELEASE"),
    9: ("open drawer", "OPEN_DRAWER"),
    10: ("open door", "OPEN_DOOR"),
    11: ("close drawer", "CLOSE_DRAWER"),
    12: ("close door", "CLOSE_DOOR"),
    13: ("open lid", "OPEN_LID"),
    14: ("close lid", "CLOSE_LID"),
    19: ("attach", "ATTACH"),
    28: ("pour", "POUR"),
    34: ("chop", "CHOP"),
    46: ("wipe hard", "WIPE_HARD"),
    50: ("sweep surface", "SWEEP_SURFACE"),
    61: ("hang", "HANG"),
    67: ("press", "PRESS"),
    69: ("turn on switch", "TURN_ON_SWITCH"),
    70: ("turn off switch", "TURN_OFF_SWITCH"),
    88: ("ignite", "IGNITE"),
    90: ("push to", "PUSH"),
    91: ("place on next to", "PLACE_NEXT_TO"),
    92: ("place in next to", "PLACE_IN_NEXT_TO"),
    93: ("turn to", "TURN_TO"),
    94: ("hold", "HOLD"),
    95: ("spray", "SPRAY"),
    98: ("place under", "PLACE_UNDER"),
    99: ("tip over", "TIP_OVER"),
    100: ("push tray", "PUSH_TRAY"),
    101: ("pull tray", "PULL_TRAY"),
    102: ("sweep off", "SWEEP_OFF"),
    103: ("lift", "LIFT"),
}

GEOMETRY = {
    "OPEN_DOOR",
    "CLOSE_DOOR",
    "OPEN_DRAWER",
    "CLOSE_DRAWER",
    "OPEN_LID",
    "CLOSE_LID",
    "PULL_TRAY",
    "PUSH_TRAY",
    "TIP_OVER",
}
PLACEMENT = {
    "PLACE_ON",
    "PLACE_IN",
    "PLACE_UNDER",
    "PLACE_NEXT_TO",
    "PLACE_IN_NEXT_TO",
}
CONTACT_RELATIONS = {"ATTACH", "HANG", "HOLD", "INSERT", "PUSH", "LIFT", "RELEASE"}
EFFECTS = {
    "CHOP",
    "IGNITE",
    "POUR",
    "SPRAY",
    "SWEEP_OFF",
    "SWEEP_SURFACE",
    "WIPE_HARD",
    "TURN_ON_SWITCH",
    "TURN_OFF_SWITCH",
}
TOOL_TARGET = {"CHOP", "IGNITE", "SPRAY", "WIPE_HARD"}
STATE_CHANGE = GEOMETRY | PLACEMENT | {"GRASP", "HANDOVER", "TURN_ON_SWITCH", "TURN_OFF_SWITCH", "TURN_TO"}

# The category vocabulary is an explicit, pinned input for v4.  It is kept
# outside the repository because it is an external data asset; the build
# records these pins in its metadata and refuses a changed file.
OFFICIAL_CATEGORY_COMMIT = "bd049de3119acdcdf2334fe9e1ebe060fa20c108"
OFFICIAL_CATEGORY_MAPPING_SHA256 = "ef4636716bc1f243f89735ffe89e8e931a2d5739e87164e7146d54d11c681eab"
OFFICIAL_CATEGORY_MAPPING_URL = (
    "https://raw.githubusercontent.com/StanfordVL/BEHAVIOR-1K/"
    f"{OFFICIAL_CATEGORY_COMMIT}/bddl3/bddl/generated_data/category_mapping.csv"
)
OFFICIAL_CATEGORY_ROWS = 2424
QUERY_TEMPLATE_REVISION_V5 = "v5_official_category_grounding_exact_membership"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_category_mapping(
    path: Path,
    *,
    expected_sha256: str = OFFICIAL_CATEGORY_MAPPING_SHA256,
    expected_rows: int = OFFICIAL_CATEGORY_ROWS,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load the official category-to-synset vocabulary and pin its identity.

    This intentionally accepts the CSV only as an explicit build input.  A
    changed or incomplete vocabulary is a hard error rather than a reason to
    fall back to a guessed suffix parser.
    """

    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"category mapping SHA mismatch: {actual_sha256} != {expected_sha256}"
        )
    categories: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "category" not in reader.fieldnames or "synset" not in reader.fieldnames:
            raise ValueError("category mapping must contain category and synset columns")
        for row in reader:
            category = (row.get("category") or "").strip()
            synset = (row.get("synset") or "").strip()
            if not category or not synset:
                raise ValueError("category mapping contains an empty category or synset")
            if category in categories:
                raise ValueError(f"duplicate official category {category!r}")
            categories[category] = synset
    if expected_rows and len(categories) != expected_rows:
        raise ValueError(
            f"category mapping row count mismatch: {len(categories)} != {expected_rows}"
        )
    metadata = {
        "path": str(path),
        "sha256": actual_sha256,
        "rows": len(categories),
        "official_commit": OFFICIAL_CATEGORY_COMMIT,
        "official_url": OFFICIAL_CATEGORY_MAPPING_URL,
    }
    return categories, metadata


def resolve_category(raw_id: str, category_mapping: dict[str, str]) -> dict[str, Any]:
    """Resolve one raw metadata ID by the unique longest official prefix.

    The suffix is opaque: it is retained only in this audit result and never
    used to manufacture a prompt noun.  Unknown and tied matches are
    quarantined instead of falling back to a lossy string heuristic.
    """

    # A raw metadata token can already be the canonical category (for example
    # ``electric_switch``) rather than a category plus an opaque instance
    # suffix.  Treat that exact vocabulary membership as resolved, while
    # retaining the boundary-aware longest-prefix rule for suffixed IDs.
    matches = [category for category in category_mapping if raw_id.startswith(f"{category}_")]
    if raw_id in category_mapping:
        matches.append(raw_id)
    if not matches:
        return {
            "status": "UNKNOWN_CATEGORY",
            "raw_object_id": raw_id,
            "category": None,
            "prompt_noun": None,
            "synset": None,
            "opaque_instance_suffix": None,
            "quarantine": True,
        }
    longest_length = max(len(category) for category in matches)
    longest = [category for category in matches if len(category) == longest_length]
    if len(longest) != 1:
        return {
            "status": "UNKNOWN_CATEGORY",
            "raw_object_id": raw_id,
            "category": None,
            "prompt_noun": None,
            "synset": None,
            "opaque_instance_suffix": None,
            "quarantine": True,
            "candidate_categories": sorted(longest),
        }
    category = longest[0]
    suffix = raw_id[len(category) + 1 :] if raw_id != category else None
    return {
        "status": "RESOLVED",
        "raw_object_id": raw_id,
        "category": category,
        "prompt_noun": category.replace("_", " "),
        "synset": category_mapping[category],
        "opaque_instance_suffix": suffix,
        "quarantine": False,
    }


def category_prompt_noun(raw_id: str, category_mapping: dict[str, str] | None) -> str | None:
    if category_mapping is None:
        return None
    result = resolve_category(raw_id, category_mapping)
    if result["status"] != "RESOLVED":
        return None
    return str(result["prompt_noun"])


def flatten_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        if value:
            result.append(value)
    elif isinstance(value, list):
        for item in value:
            result.extend(flatten_strings(item))
    return result


def unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def pretty_entity(entity: str) -> str:
    """Make a question-readable label without treating it as visual identity."""

    tokens = entity.split("_")
    if tokens and re.fullmatch(r"\d+", tokens[-1] or ""):
        tokens.pop()
    return " ".join(tokens) or entity


def q_label(
    ids: list[str],
    fallback: str = "the named entity",
    category_mapping: dict[str, str] | None = None,
) -> str:
    if not ids:
        return fallback
    labels = []
    for item in ids:
        if category_mapping is None:
            labels.append(pretty_entity(item))
        else:
            noun = category_prompt_noun(item, category_mapping)
            # Unknown categories stay quarantined.  This generic wording is
            # deliberately not a guessed noun and contains no opaque ID tail.
            labels.append(noun if noun is not None else "the named metadata entity")
    # If multiple metadata entities resolve to one category noun, retain the
    # multiplicity instead of pretending they are one visually identified item.
    counts: dict[str, int] = {}
    order: list[str] = []
    for label in labels:
        if label not in counts:
            order.append(label)
            counts[label] = 0
        counts[label] += 1
    formatted_labels: list[str] = []
    for label in order:
        count = counts[label]
        if count <= 1:
            formatted_labels.append(label)
        elif label == "the named metadata entity":
            formatted_labels.append(f"{count} named metadata entities")
        else:
            formatted_labels.append(f"{count} {label} entities")
    labels = formatted_labels
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _entity_ids(value: str | list[str] | None) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if value is None:
        return []
    return [item for item in value if isinstance(item, str) and item]


def controlled_part_subject(
    verb: str,
    target: str,
    skill: dict[str, Any],
    category_mapping: dict[str, str] | None = None,
) -> str:
    """Name the controlled door/drawer/lid rather than only its parent body."""

    target_label = (
        category_prompt_noun(target, category_mapping)
        if target and category_mapping is not None
        else (pretty_entity(target) if target else None)
    ) or "named target"
    target_lower = target_label.lower()
    part = pretty_entity(str(skill.get("target_part") or ""))
    if verb in {"OPEN_DOOR", "CLOSE_DOOR"}:
        if "door" in part.lower():
            return f"the {part} of {target_label}"
        if part:
            return f"the {part} door of {target_label}"
        if "door" in target_lower:
            return f"the {target_label}"
        return f"the {target_label} door"
    if verb in {"OPEN_DRAWER", "CLOSE_DRAWER"}:
        if "drawer" in part.lower():
            return f"the {part} of {target_label}"
        if part:
            return f"the {part} drawer of {target_label}"
        if "drawer" in target_lower:
            return f"the {target_label}"
        return f"the {target_label} drawer"
    if verb in {"OPEN_LID", "CLOSE_LID"}:
        if part:
            return f"the {part} of {target_label}"
        if any(word in target_lower.split() for word in ("lid", "trunk", "hood")):
            return f"the {target_label}"
        if target_lower.split()[0] in {"car", "vehicle", "automobile"} or target_lower.endswith(" car"):
            return f"the lid or trunk of the {target_label}"
        return f"the {target_label} lid"
    return f"the {target_label}"


def effect_subject_phrase(
    target_ids: list[str],
    reference_ids: list[str],
    category_mapping: dict[str, str] | None = None,
) -> str:
    """Name effect entities without assigning unresolved source/destination roles."""

    ids = [*target_ids, *reference_ids]
    if not ids:
        return "the grounded target or material entities"
    if len(ids) == 1:
        label = q_label(ids, category_mapping=category_mapping)
        return label if label == "the named metadata entity" else f"the named entity {label}"
    return f"the named entities {q_label(ids, category_mapping=category_mapping)}"


def entity_record(
    entity_id: str,
    role: str,
    role_status: str,
    role_candidates: list[str] | None = None,
    category_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    grounding = resolve_category(entity_id, category_mapping) if category_mapping is not None else None
    display_label = (
        grounding["prompt_noun"]
        if grounding is not None and grounding["status"] == "RESOLVED"
        else ("the named metadata entity" if category_mapping is not None else pretty_entity(entity_id))
    )
    record: dict[str, Any] = {
        "metadata_id": entity_id,
        "display_label": display_label,
        "role": role,
        "role_status": role_status,
        "visual_grounding": "REQUIRED_FROM_RGB",
        "suffix_rule": "ID suffix is metadata only and is not visually guessable",
    }
    if grounding is not None:
        # Keep only non-sensitive display/audit status in the actor-facing
        # entity record.  The raw ID and opaque suffix stay in the separate
        # category_grounding_audit sidecar emitted by generate_row.
        record["category_grounding"] = {
            "status": grounding["status"],
            "category": grounding["category"],
            "prompt_noun": grounding["prompt_noun"],
            "synset": grounding["synset"],
            "quarantine": grounding["quarantine"],
            "taxonomy_rule": "official_longest_category_prefix_with_boundary",
        }
    if role_candidates:
        record["role_candidates"] = role_candidates
    return record


def source_skill(row: dict[str, Any], skill_id: int) -> dict[str, Any]:
    for skill in row.get("source_skills", []):
        if skill.get("skill_id") == skill_id:
            return skill
    raise ValueError(f"event {row.get('event_id')} has no source skill {skill_id}")


def validate_skill(skill: dict[str, Any]) -> str:
    try:
        skill_id = int(skill["skill_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid skill ID in {skill!r}") from exc
    if skill_id not in CANONICAL35:
        raise ValueError(f"skill {skill_id} is outside canonical 35-skill map")
    expected_description, expected_verb = CANONICAL35[skill_id]
    if skill.get("skill_description", "").strip().lower() != expected_description:
        raise ValueError(
            f"skill {skill_id} description mismatch: {skill.get('skill_description')!r} != {expected_description!r}"
        )
    actual_verb = skill.get("canonical_verb")
    if actual_verb != expected_verb:
        raise ValueError(f"skill {skill_id} canonical verb mismatch: {actual_verb!r} != {expected_verb!r}")
    return expected_verb


def metadata_entities(row: dict[str, Any], q: dict[str, Any], skill: dict[str, Any], verb: str) -> dict[str, Any]:
    raw = skill.get("raw_relation") or {}
    manipulating = unique(flatten_strings(raw.get("manipulating_object_id", [])))
    object_ids = unique(flatten_strings(raw.get("object_id", [])))
    q_targets = unique(flatten_strings(q.get("metadata_target_ids", [])))
    q_refs = unique(flatten_strings(q.get("metadata_reference_ids", [])))
    source = str(skill.get("source") or "")
    destination = str(skill.get("destination") or "")
    target = str(skill.get("target") or "")

    if verb == "NAVIGATE":
        target_ids = unique([target, *q_targets, *object_ids])
        references: list[str] = []
        actor = ["robot"]
        return {
            "actor": actor,
            "tools": [],
            "targets": target_ids[:1],
            "references": references,
            "recipient": None,
            "reference_role_status": "NONE",
        }

    if verb == "TURN_TO":
        reference_ids = unique([*q_targets, *object_ids, *q_refs])
        reference_ids = [item for item in reference_ids if item != "robot"]
        return {
            "actor": ["robot"],
            "tools": [],
            "targets": [],
            "references": reference_ids,
            "recipient": None,
            "reference_role_status": "REFERENCE_GROUNDING_REQUIRED",
        }

    if verb == "HANDOVER":
        target_ids = unique([target, *manipulating, *q_targets])
        recipient_candidates = unique(
            flatten_strings(raw.get("recipient", []))
            + flatten_strings(skill.get("recipient", []))
        )
        recipient_candidates = [item for item in recipient_candidates if item not in target_ids]
        recipient = recipient_candidates[0] if recipient_candidates else None
        return {
            "actor": ["robot"],
            "tools": [],
            "targets": target_ids[:1],
            "references": [],
            "recipient": recipient,
            "reference_role_status": "RECIPIENT_EXPLICIT" if recipient else "RECIPIENT_UNSPECIFIED",
        }

    if verb == "PRESS":
        target_ids = unique([target, *manipulating, *q_targets])
        return {
            "actor": ["robot"],
            # The radio is the manipulated target, not a visually identified
            # robot tool.  Never relabel it as a tool from this field alone.
            "tools": [],
            "targets": target_ids[:1],
            "references": unique(q_refs),
            "recipient": None,
            "reference_role_status": "TARGET_CONTROL_UNSPECIFIED",
        }

    if verb == "SWEEP_OFF":
        target_ids = unique([*q_targets, *manipulating])
        references = unique([*q_refs, *[item for item in object_ids if item not in target_ids]])
        return {
            "actor": ["robot"],
            "tools": [],
            "targets": target_ids,
            "references": references,
            "recipient": None,
            "reference_role_status": "EFFECT_ENTITIES_NEED_ROLE_GROUNDING",
        }

    if verb in TOOL_TARGET:
        tools = manipulating
        result_targets = [item for item in object_ids if item not in tools]
        if not result_targets:
            result_targets = [item for item in q_refs if item not in tools]
        # A result/effect relation may contain material and destination objects;
        # preserve them as unresolved rather than inventing source/destination.
        return {
            "actor": ["robot"],
            "tools": tools,
            "targets": unique(result_targets),
            "references": unique(item for item in q_refs if item not in result_targets),
            "recipient": None,
            "reference_role_status": "EFFECT_ENTITIES_NEED_ROLE_GROUNDING",
        }

    if verb in {"TURN_ON_SWITCH", "TURN_OFF_SWITCH"}:
        target_ids = unique([target, *manipulating, *q_targets, *object_ids])
        return {
            "actor": ["robot"],
            "tools": [],
            "targets": target_ids[:1],
            "references": unique(q_refs),
            "recipient": None,
            "reference_role_status": "NONE",
        }

    if verb in EFFECTS:
        tools = manipulating
        result_entities = [item for item in object_ids if item not in tools]
        if not result_entities:
            result_entities = [item for item in q_refs if item not in tools]
        return {
            "actor": ["robot"],
            "tools": tools,
            "targets": unique(result_entities),
            "references": unique(item for item in q_refs if item not in result_entities),
            "recipient": None,
            "reference_role_status": "EFFECT_ENTITIES_NEED_ROLE_GROUNDING",
        }

    if verb == "GRASP":
        target_ids = unique([target, *manipulating, *q_targets])
        references = unique([source, *q_refs, *[item for item in object_ids if item not in target_ids]])
        return {
            "actor": ["robot"],
            "tools": [],
            "targets": target_ids[:1],
            "references": references,
            "recipient": None,
            "reference_role_status": "SOURCE_SUPPORT" if references else "SOURCE_SUPPORT_UNSPECIFIED",
        }

    if verb in PLACEMENT:
        target_ids = unique([target, *manipulating, *q_targets])
        references = unique([destination, *q_refs, *[item for item in object_ids if item not in target_ids]])
        explicit_destination = bool(destination)
        return {
            "actor": ["robot"],
            "tools": [],
            "targets": target_ids[:1],
            "references": references,
            "recipient": None,
            "reference_role_status": "EXPLICIT_DESTINATION" if explicit_destination else "REFERENCE_ROLE_UNRESOLVED",
        }

    if verb in GEOMETRY:
        target_ids = unique([target, *q_targets, *manipulating, *object_ids])
        target_ids = target_ids[:1]
        references = unique(q_refs)
        return {
            "actor": ["robot"],
            "tools": [],
            "targets": target_ids,
            "references": references,
            "recipient": None,
            "reference_role_status": "TARGET_PART_OR_FRAME_REQUIRED",
        }

    # CONTACT_RELATIONS and any future canonical verb use exact metadata while
    # retaining unresolved reference roles.  This is safer than a guessed role.
    target_ids = unique([target, *manipulating, *q_targets, *object_ids])
    references = unique([*q_refs, *[item for item in object_ids if item not in target_ids[:1]]])
    return {
        "actor": ["robot"],
        "tools": [],
        "targets": target_ids[:1],
        "references": references,
        "recipient": None,
        "reference_role_status": "REFERENCE_ROLE_UNRESOLVED" if references else "NONE",
    }


def role_records(
    entities: dict[str, Any],
    skill: dict[str, Any],
    verb: str,
    category_mapping: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_status = "METADATA_CANDIDATE_REQUIRES_VISUAL_GROUNDING"
    if entities["reference_role_status"] == "EFFECT_ENTITIES_NEED_ROLE_GROUNDING":
        targets = [
            entity_record(
                item,
                "effect_entity_unresolved",
                "METADATA_ROLE_AMBIGUOUS_REQUIRES_VISUAL_GROUNDING",
                ["material_or_target", "destination_or_surface", "object_reference"],
                category_mapping,
            )
            for item in entities["targets"]
        ]
    else:
        targets = [entity_record(item, "target", target_status, category_mapping=category_mapping) for item in entities["targets"]]
    refs: list[dict[str, Any]] = []
    explicit_destination = bool(skill.get("destination"))
    for item in entities["references"]:
        if explicit_destination and item == skill.get("destination"):
            refs.append(entity_record(item, "destination", "METADATA_EXPLICIT_REQUIRES_VISUAL_GROUNDING", category_mapping=category_mapping))
        elif verb == "GRASP" and item == skill.get("source"):
            refs.append(entity_record(item, "source_support", "METADATA_EXPLICIT_REQUIRES_VISUAL_GROUNDING", category_mapping=category_mapping))
        else:
            refs.append(
                entity_record(
                    item,
                    "reference_unresolved",
                    "METADATA_ROLE_AMBIGUOUS_REQUIRES_VISUAL_GROUNDING",
                    ["destination_or_container", "object_reference", "location_or_surface"],
                    category_mapping,
                )
            )
    return targets, refs


def acting_records(
    entities: dict[str, Any],
    verb: str,
    category_mapping: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Expose manipulated metadata without overclaiming that it is a tool."""

    records: list[dict[str, Any]] = []
    for item in entities.get("tools", []):
        if verb in TOOL_TARGET:
            role = "tool_candidate"
            candidates = ["tool"]
        elif verb == "POUR":
            role = "source_container_or_tool_candidate"
            candidates = ["source_container", "tool", "target_object"]
        else:
            role = "manipulated_object_candidate"
            candidates = ["tool", "source_object", "target_object"]
        records.append(
            entity_record(
                item,
                role,
                "METADATA_ROLE_REQUIRES_VISUAL_GROUNDING",
                candidates,
                category_mapping,
            )
        )
    return records


def category_grounding_audit(
    entities: dict[str, Any],
    category_mapping: dict[str, str] | None,
    category_mapping_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return raw-ID/category resolution details for the audit sidecar only."""

    if category_mapping is None:
        return []
    roles: dict[str, list[str]] = {
        "target": list(entities.get("targets", [])),
        "reference": list(entities.get("references", [])),
        "acting": list(entities.get("tools", [])),
    }
    recipient = entities.get("recipient")
    if isinstance(recipient, str) and recipient:
        roles["recipient"] = [recipient]
    role_by_id: dict[str, list[str]] = {}
    for role, ids in roles.items():
        for raw_id in ids:
            role_by_id.setdefault(raw_id, []).append(role)
    audit: list[dict[str, Any]] = []
    for raw_id in sorted(role_by_id):
        resolved = resolve_category(raw_id, category_mapping)
        resolved["roles"] = sorted(set(role_by_id[raw_id]))
        resolved["taxonomy_rule"] = "official_longest_category_prefix_with_boundary"
        if category_mapping_metadata:
            resolved["taxonomy_commit"] = category_mapping_metadata.get("official_commit")
            resolved["taxonomy_sha256"] = category_mapping_metadata.get("sha256")
        audit.append(resolved)
    return audit


def relation_family(verb: str) -> str:
    if verb == "NAVIGATE":
        return "navigation_reach_metric"
    if verb == "PRESS":
        return "press_control_contact"
    if verb == "HANDOVER":
        return "handover_recipient_relation"
    if verb == "GRASP":
        return "grasp_current_hold"
    if verb in PLACEMENT:
        return "placement_current_relation"
    if verb in GEOMETRY:
        return "geometry_current_relation"
    if verb in EFFECTS:
        return "effect_current_state_or_contact"
    if verb == "TURN_TO":
        return "orientation_current_relation"
    return "contact_current_relation"


def relation_phrase(
    verb: str,
    target: str | list[str],
    refs: list[str],
    skill: dict[str, Any],
    category_mapping: dict[str, str] | None = None,
) -> str:
    target_ids = _entity_ids(target)
    target_label = q_label(target_ids, "the named target", category_mapping)
    ref_label = q_label(refs, "the named reference", category_mapping)
    if verb == "GRASP":
        return f"At the anchor, is {target_label} visibly held by the robot gripper or hand?"
    if verb == "PRESS":
        return f"At the anchor, is the robot tool or gripper visibly contacting an identifiable press control on {target_label}?"
    if verb == "HANDOVER":
        return f"At the anchor, is {target_label} visibly held by an explicitly identified receiving agent or other robot gripper?"
    if verb == "NAVIGATE":
        return f"Can the RGB sequence establish that the robot is at {target_label} within the required distance and pose?"
    if verb == "OPEN_DOOR":
        return f"Is {controlled_part_subject(verb, target_ids[0] if target_ids else '', skill, category_mapping)} visibly open relative to its door frame?"
    if verb == "CLOSE_DOOR":
        return f"Is {controlled_part_subject(verb, target_ids[0] if target_ids else '', skill, category_mapping)} visibly closed against its door frame?"
    if verb == "OPEN_DRAWER":
        return f"Is {controlled_part_subject(verb, target_ids[0] if target_ids else '', skill, category_mapping)} visibly open relative to its frame?"
    if verb == "CLOSE_DRAWER":
        return f"Is {controlled_part_subject(verb, target_ids[0] if target_ids else '', skill, category_mapping)} visibly closed or seated flush with its cabinet frame?"
    if verb == "OPEN_LID":
        return f"Is {controlled_part_subject(verb, target_ids[0] if target_ids else '', skill, category_mapping)} visibly open or raised relative to its body?"
    if verb == "CLOSE_LID":
        return f"Is {controlled_part_subject(verb, target_ids[0] if target_ids else '', skill, category_mapping)} visibly closed and seated on its body?"
    if verb == "PULL_TRAY":
        return f"Is the designated part of {target_label} visibly pulled out from its appliance or frame?"
    if verb == "PUSH_TRAY":
        return f"Is the designated part of {target_label} visibly pushed into its appliance or frame?"
    if verb == "TIP_OVER":
        return f"Is {target_label} visibly tipped over from its upright orientation?"
    if verb == "PLACE_ON":
        if not skill.get("destination"):
            return f"At the anchor, is {target_label} visibly in the recorded on/support relation to {ref_label}, with target/reference roles visually grounded?"
        return f"At the anchor, is {target_label} visibly resting on or supported by {ref_label}?"
    if verb == "PLACE_IN":
        if not skill.get("destination"):
            return f"At the anchor, is {target_label} visibly in the recorded containment relation to {ref_label}, with target/reference roles visually grounded?"
        return f"At the anchor, is {target_label} visibly contained inside {ref_label}?"
    if verb == "PLACE_UNDER":
        if not skill.get("destination"):
            return f"At the anchor, is {target_label} visibly in the recorded under relation to {ref_label}, with target/reference roles visually grounded?"
        return f"At the anchor, is {target_label} visibly positioned under {ref_label}?"
    if verb == "PLACE_NEXT_TO":
        return f"At the anchor, is {target_label} visibly next to {ref_label}, with target/reference roles visually grounded?"
    if verb == "PLACE_IN_NEXT_TO":
        if not skill.get("destination"):
            return f"At the anchor, is {target_label} visibly in the recorded placement relation to {ref_label}, with target/reference roles visually grounded?"
        return f"At the anchor, is {target_label} visibly inside the grounded destination and next to the grounded reference entity?"
    if verb == "ATTACH":
        return f"At the anchor, is {target_label} visibly seated on or connected to {ref_label}?"
    if verb == "HANG":
        return f"At the anchor, is {target_label} visibly hanging from or supported by {ref_label}?"
    if verb == "HOLD":
        return f"At the anchor, is {target_label} visibly held or supported by the robot gripper or hand?"
    if verb == "INSERT":
        return f"At the anchor, is {target_label} visibly inserted into an opening or slot of {ref_label}?"
    if verb == "PUSH":
        return f"At the anchor, is {target_label} visibly at the grounded pushed-to relation relative to {ref_label}?"
    if verb == "LIFT":
        return f"At the anchor, is {target_label} visibly lifted clear of its prior support?"
    if verb == "RELEASE":
        return f"At the anchor, is {target_label} visibly released from the robot gripper or hand?"
    if verb == "TURN_ON_SWITCH":
        return f"At the anchor, is {target_label} visibly in the ON state?"
    if verb == "TURN_OFF_SWITCH":
        return f"At the anchor, is {target_label} visibly in the OFF state?"
    if verb == "TURN_TO":
        return f"At the anchor, is the robot visibly oriented in the recorded relation to {ref_label}?"
    if verb in EFFECTS:
        subjects = effect_subject_phrase(target_ids, refs, category_mapping)
        return (
            f"At the anchor, is the requested {verb.lower().replace('_', ' ')} effect visibly present "
            f"for {subjects}, with target/material/reference roles visually grounded?"
        )
    return f"At the anchor, is the canonical {verb} relation visibly true for {target_label}?"


def current_gates(verb: str, entities: dict[str, Any], q: dict[str, Any], skill: dict[str, Any]) -> tuple[list[str], list[str]]:
    gates = ["VISUAL_TARGET_IDENTITY"]
    unknown_reasons = ["TARGET_ID_IS_METADATA_ONLY"]
    if not entities["targets"]:
        gates.append("TARGET_ROLE")
        unknown_reasons.append("MISSING_TARGET_GROUNDING")
    if entities["references"]:
        gates.append("VISUAL_REFERENCE_IDENTITY_AND_ROLE")
        if entities["reference_role_status"] in {"REFERENCE_ROLE_UNRESOLVED", "EFFECT_ENTITIES_NEED_ROLE_GROUNDING"}:
            unknown_reasons.append("REFERENCE_ROLE_UNRESOLVED")
    if verb == "PRESS":
        gates.append("VISUAL_TARGET_CONTROL_OR_PRESS_AFFORDANCE")
        unknown_reasons.append("MISSING_TARGET_CONTROL_GROUNDING")
    if verb == "HANDOVER":
        gates.append("VISUAL_RECIPIENT_TYPE_AND_IDENTITY")
        if not entities["recipient"]:
            unknown_reasons.append("MISSING_RECIPIENT_GROUNDING_NO_PERSON_ASSUMPTION")
    if verb == "NAVIGATE":
        gates.extend(["METRIC_DISTANCE_STATE", "ROBOT_POSE_STATE"])
        unknown_reasons.append("METRIC_DISTANCE_AND_POSE_NOT_RGB_GUARANTEED")
    if verb in EFFECTS - {"TURN_ON_SWITCH", "TURN_OFF_SWITCH"}:
        gates.append("DIRECTLY_VISIBLE_EFFECT_OR_STATE")
        unknown_reasons.append("EFFECT_NOT_PROVEN_BY_TOOL_CONTACT_OR_ENDPOINT")
    if entities["reference_role_status"] == "EFFECT_ENTITIES_NEED_ROLE_GROUNDING":
        gates.append("EFFECT_ENTITY_ROLE_GROUNDING")
        unknown_reasons.append("EFFECT_ENTITY_ROLE_UNRESOLVED")
    return unique(gates), unique(unknown_reasons)


def supporting_observation(verb: str, entities: dict[str, Any]) -> dict[str, Any] | None:
    if verb == "PRESS":
        return {
            "kind": "contact_diagnostic",
            "question": "Is the tool or gripper visibly near or contacting the named object?",
            "not_goal_substitute": True,
            "warning": "Contact with an object body is not proof of contact with its press control or of a press effect.",
        }
    if verb in EFFECTS - {"TURN_ON_SWITCH", "TURN_OFF_SWITCH"}:
        return {
            "kind": "tool_target_contact_diagnostic",
            "question": "Is the named tool visibly contacting or operating toward a grounded target/material entity?",
            "not_goal_substitute": True,
            "warning": "Contact, motion, or a segment endpoint does not prove the requested effect or state change.",
        }
    if verb == "NAVIGATE":
        return {
            "kind": "target_visibility_diagnostic",
            "question": "Is the metadata candidate category visible in RGB?",
            "not_goal_substitute": True,
            "warning": "Target visibility is not navigation reach, distance, or pose satisfaction.",
        }
    return None


def history_query(
    verb: str,
    entities: dict[str, Any],
    skill: dict[str, Any],
    anchor: int,
    causal_frames: list[int],
    category_mapping: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    if verb == "NAVIGATE":
        return {
            "status": "EXCLUDED_UNSUPPORTED_BY_RGB_STATE",
            "question": "Was the robot moved into the required metric distance and pose relative to the target?",
            "required_observations": ["robot state/pose", "metric target distance"],
            "review_gate": "STATE_EVIDENCE_REQUIRED",
            "reason": "Do not replace this with target visibility or apparent proximity.",
        }
    if verb == "PRESS":
        return {
            "status": "SEPARATE_EFFECT_NOT_REQUESTED",
            "question": "Did the grounded control visibly change state across the press interval?",
            "required_observations": ["identified control", "prestate", "poststate", "causal identity continuity"],
            "review_gate": "CONTROL_PRESTATE_POSTSTATE_REQUIRED",
            "reason": "Body contact alone is not a press or effect.",
        }
    if verb not in STATE_CHANGE and verb not in EFFECTS and verb not in {"HOLD", "ATTACH", "HANG", "INSERT", "PUSH", "LIFT", "RELEASE"}:
        return None
    start = skill.get("skill_start")
    start_in_packet = isinstance(start, int) and start in causal_frames
    if verb in GEOMETRY:
        if verb.startswith("OPEN_"):
            from_state, to_state = "closed", "open"
        elif verb.startswith("CLOSE_"):
            from_state, to_state = "open", "closed"
        elif verb == "TIP_OVER":
            from_state, to_state = "upright", "tipped_over"
        elif verb in {"PULL_TRAY", "PUSH_TRAY"}:
            from_state, to_state = "seated", "pulled_or_pushed_geometry"
        else:
            from_state, to_state = "prior_geometry", "requested_geometry"
    elif verb == "GRASP":
        from_state, to_state = "supported_or_not_held", "held_by_robot"
    elif verb == "HANDOVER":
        from_state, to_state = "held_by_robot", "held_by_explicit_recipient"
    elif verb in {"TURN_ON_SWITCH", "TURN_OFF_SWITCH"}:
        from_state, to_state = ("off", "on") if verb == "TURN_ON_SWITCH" else ("on", "off")
    elif verb == "TURN_TO":
        from_state, to_state = "prior_orientation", "recorded_orientation"
    elif verb in EFFECTS:
        from_state, to_state = "pre_effect_state", "requested_effect_state"
    else:
        from_state, to_state = "prior_relation", "requested_relation"
    target = q_label(entities["targets"], "the named target", category_mapping)
    return {
        "status": "REQUIRES_OBSERVABLE_PRESTATE",
        "question": f"Does actor-causal RGB show {target} changing from {from_state} to {to_state}?",
        "required_observations": [
            "same visually grounded entity in prestate and poststate",
            "observable prestate at or before the skill start",
            "observable poststate at the anchor",
        ],
        "skill_start_frame": start,
        "anchor_frame": anchor,
        "skill_start_exactly_in_current_packet": start_in_packet,
        "current_packet_causal_frames": causal_frames,
        "review_gate": "PRESTATE_REQUIRED",
        "not_proven_by": ["segment end", "gripper closure", "timeout", "model text", "single endpoint", "contact alone"],
        "endpoint_history_warning": "An endpoint or nominal ±2 second window does not necessarily establish the earlier wipe/chop/transfer prestate.",
        "phase_sampler_note": "Retain skill_start/skill_end for a future causal render; this frame is not automatically actor input or renderer-approved evidence.",
    }


def validate_row(row: dict[str, Any]) -> None:
    event_id = row.get("event_id")
    if not isinstance(event_id, str) or len(event_id) != 64:
        raise ValueError(f"invalid event_id {event_id!r}")
    source_ids = set(row.get("source_skill_ids_at_anchor", []))
    if not source_ids:
        raise ValueError(f"event {event_id} has no source_skill_ids_at_anchor")
    if row.get("training_eligible") is not False:
        raise ValueError(f"event {event_id} is unexpectedly training eligible")


def normalize_phase_row(queue_row: dict[str, Any], indexed_event: dict[str, Any]) -> dict[str, Any]:
    """Adapt the phase-balanced queue without joining by selection order.

    The queue is intentionally small, while the phase candidate index carries
    the exact raw relation.  Every identity check below is by event_id and the
    queried skill's metadata fields.  A mismatch is a hard error, not a best
    effort merge.
    """

    event_id = queue_row.get("event_id")
    if indexed_event.get("event_id") != event_id:
        raise ValueError(f"phase event join mismatch: {event_id}")
    indexed_source = indexed_event.get("source") or {}
    queue_source = queue_row.get("source_identity") or {}
    for key in ("source_group_id", "task_index", "task_instance_id"):
        if indexed_source.get(key) != queue_source.get(key):
            raise ValueError(f"phase {event_id} source mismatch at {key}")
    if indexed_event.get("observation", {}).get("frame") != queue_row.get("observation_frame"):
        raise ValueError(f"phase {event_id} observation-frame mismatch")
    qbinding = (queue_row.get("queried_skill") or {}).get("queried_skill") or {}
    qid = qbinding.get("skill_id")
    bundle = indexed_event.get("skill_bundle") or []
    members = {member.get("skill_id"): member for member in bundle}
    if qid not in members:
        raise ValueError(f"phase {event_id} queried skill {qid} absent from indexed skill_bundle")
    member = members[qid]
    for qkey, ikey in (("skill_id", "skill_id"), ("skill_start", "skill_start"), ("skill_end", "skill_end"), ("raw_description", "raw_description"), ("verb", "verb")):
        if qbinding.get(qkey) != member.get(ikey):
            raise ValueError(f"phase {event_id} queried-skill mismatch at {qkey}")
    source_skills = []
    for item in bundle:
        source_skills.append({
            "skill_id": item.get("skill_id"),
            "skill_description": item.get("raw_description"),
            "canonical_verb": item.get("verb"),
            "skill_idx": item.get("skill_idx"),
            "skill_start": item.get("skill_start"),
            "skill_end": item.get("skill_end"),
            "source": item.get("source", ""),
            "target": item.get("target", ""),
            "destination": item.get("destination", ""),
            "target_part": item.get("target_part", ""),
            "binding_confidence": item.get("binding_confidence"),
            "raw_relation": item.get("raw_relation", {}),
        })
    qcandidate = {
        "skill_id": qid,
        "skill_description": qbinding.get("raw_description"),
        "canonical_verb": qbinding.get("verb"),
        "relation_family": "phase_queue_metadata_goal_relation",
        "intended_goal_relation_question": "PENDING_V3_TEMPLATE_EXPANSION",
        "observability": "METADATA_ONLY_NEEDS_RGB_REVIEW",
        "metadata_target_ids": unique(flatten_strings([member.get("target", "")])),
        "metadata_reference_ids": unique(flatten_strings([member.get("source", ""), member.get("destination", "")])),
        "binding_status": (queue_row.get("queried_skill") or {}).get("binding_status"),
    }
    windows = queue_row.get("temporal_windows") or {}
    actor = windows.get("actor_available_window") or {}
    after = windows.get("offline_review_after_window") or {}
    normalized = {
        "event_id": event_id,
        "source_group_id": queue_source.get("source_group_id"),
        "training_eligible": queue_row.get("training_eligible"),
        "source_skill_ids_at_anchor": [qid],
        "source_skills": source_skills,
        "event_interval": queue_row.get("original_annotated_segment"),
        "question_context": {
            "question_candidates": [qcandidate],
            "future_frames_are_offline_only": True,
        },
        "temporal_packet": {
            "anchor_frame": queue_row.get("observation_frame"),
            "causal_frame_indices": actor.get("sampled_frames", []),
            "future_frame_indices": after.get("sampled_frames", []),
        },
        "_phase_queue_binding": {
            "phase_queue_schema": queue_row.get("schema_version"),
            "selection_order": queue_row.get("selection_order"),
            "selection_stratum": queue_row.get("selection_stratum"),
            "observation_phase": queue_row.get("observation_phase"),
            "parent_event_id": queue_row.get("parent_event_id"),
            "queried_skill_start_frame": queue_row.get("queried_skill_start_frame"),
            "queried_skill_end_frame": queue_row.get("queried_skill_end_frame"),
            "intent_start_causal_reference_frame": (queue_row.get("intent_start_causal_reference") or {}).get("frame"),
            "terminal_query_binding": queue_row.get("terminal_query_binding"),
            "phase_index_bundle_id": indexed_event.get("bundle_id"),
            "future_use": "OFFLINE_REVIEW_ONLY_NOT_ACTOR_EVIDENCE",
        },
    }
    return normalized


def generate_row(
    row: dict[str, Any],
    question_ordinal: int,
    category_mapping: dict[str, str] | None = None,
    category_mapping_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validate_row(row)
    anchor = row.get("temporal_packet", {}).get("anchor_frame")
    causal_frames = row.get("temporal_packet", {}).get("causal_frame_indices", [])
    source_id_set = set(row.get("source_skill_ids_at_anchor", []))
    candidates = row.get("question_context", {}).get("question_candidates", [])
    if not candidates:
        raise ValueError(f"event {row['event_id']} has no question candidates")
    outputs: list[dict[str, Any]] = []
    for query_ordinal, q in enumerate(candidates):
        skill_id = int(q["skill_id"])
        skill = source_skill(row, skill_id)
        verb = validate_skill(skill)
        in_anchor = skill_id in source_id_set and skill.get("skill_start", 0) <= anchor <= skill.get("skill_end", anchor)
        binding_status = "BOUND_AT_ANCHOR" if in_anchor else "MISSING_OR_AMBIGUOUS"
        entities = metadata_entities(row, q, skill, verb)
        targets, references = role_records(entities, skill, verb, category_mapping)
        target_ids = [item["metadata_id"] for item in targets]
        reference_ids = [item["metadata_id"] for item in references]
        gates, unknown_reasons = current_gates(verb, entities, q, skill)
        if verb in {"PLACE_NEXT_TO", "PLACE_IN_NEXT_TO"} and not skill.get("destination"):
            unknown_reasons.append("NO_EXPLICIT_DESTINATION_FIELD")
        historical = history_query(verb, entities, skill, anchor, causal_frames, category_mapping)
        audit = category_grounding_audit(entities, category_mapping, category_mapping_metadata)
        unknown_grounding = [item for item in audit if item["status"] != "RESOLVED"]
        current = {
            "question": relation_phrase(verb, target_ids, reference_ids, skill, category_mapping),
            "relation_family": relation_family(verb),
            "target_entities": targets,
            "reference_entities": references,
            "acting_entities": acting_records(entities, verb, category_mapping),
            "required_visual_predicates": gates,
            "grounding_gates": gates,
            "unknown_reasons": unique(unknown_reasons),
            "goal_semantics": "CURRENT_VISIBLE_RELATION_ONLY",
            "metadata_attempt_is_not_outcome": True,
        }
        if verb == "NAVIGATE":
            current["supervision_eligibility"] = "EXCLUDED_UNLESS_APPROVED_STATE_EVIDENCE"
            current["not_a_goal_substitute"] = "target_visibility_diagnostic"
        elif verb == "PRESS":
            current["supervision_eligibility"] = "CONDITIONAL_ON_CONTROL_GROUNDING"
        elif verb == "HANDOVER":
            current["supervision_eligibility"] = "CONDITIONAL_ON_RECIPIENT_GROUNDING"
        elif verb in EFFECTS - {"TURN_ON_SWITCH", "TURN_OFF_SWITCH"}:
            current["supervision_eligibility"] = "CONDITIONAL_ON_DIRECT_EFFECT_VISIBILITY"
        elif verb in {"TURN_ON_SWITCH", "TURN_OFF_SWITCH"}:
            current["supervision_eligibility"] = "CONDITIONAL_ON_DIRECT_STATE_VISIBILITY"
        else:
            current["supervision_eligibility"] = "CONDITIONAL_ON_RGB_GROUNDING"
        current["category_grounding_status"] = (
            "QUARANTINED_UNKNOWN_CATEGORY" if unknown_grounding else "RESOLVED_OFFICIAL_CATEGORY"
        )
        current["category_grounding_unknown_count"] = len(unknown_grounding)
        if unknown_grounding:
            current["supervision_eligibility"] = "QUARANTINED_UNKNOWN_CATEGORY"
        out = {
            "schema_version": "p107.visual_query_candidate.v3",
            "status": "QUERY_DESIGN_ONLY_NO_ANSWER",
            "event_id": row["event_id"],
            "source_group_id": row.get("source_group_id"),
            "source_question_file_ordinal": question_ordinal,
            "query_ordinal_within_event": query_ordinal,
            "observation_frame": anchor,
            "queried_skill_binding": {
                "skill_id": skill_id,
                "skill_description": skill.get("skill_description"),
                "canonical_verb": verb,
                "skill_idx": skill.get("skill_idx"),
                "skill_start": skill.get("skill_start"),
                "skill_end": skill.get("skill_end"),
                "binding_status": binding_status,
                "source_skill_ids_at_anchor": sorted(source_id_set),
                "metadata_binding_confidence": skill.get("binding_confidence"),
            },
            "source_v1_query": {
                "relation_family": q.get("relation_family"),
                "intended_goal_relation_question": q.get("intended_goal_relation_question"),
                "metadata_target_ids": q.get("metadata_target_ids", []),
                "metadata_reference_ids": q.get("metadata_reference_ids", []),
                "binding_status": q.get("binding_status"),
            },
            "current_visible_goal_relation": current,
            "historical_change_relation": historical,
            "supporting_observation": supporting_observation(verb, entities),
            "evidence_policy": {
                "actor_causal_latest_frame": anchor,
                "actor_causal_frame_indices": causal_frames,
                "future_use": "OFFLINE_AUDIT_ONLY_NEVER_ANCHOR_ANSWER",
            },
            "metadata_entities_raw": {
                "manipulating_object_id": (skill.get("raw_relation") or {}).get("manipulating_object_id", []),
                "object_id": (skill.get("raw_relation") or {}).get("object_id", []),
                "source": skill.get("source", ""),
                "destination": skill.get("destination", ""),
                "target": skill.get("target", ""),
                "target_part": skill.get("target_part", ""),
            },
            # This is an audit sidecar.  It is intentionally not projected
            # into the compact actor registry and includes opaque suffixes
            # only to make taxonomy resolution reproducible.
            "category_grounding_audit": audit,
            "provenance": {
                "annotator_model": "gpt-5.6-luna/max",
                "human_reviewed": False,
                "image_inspected": False,
                "labels_created": False,
                "root_review": "PENDING",
            },
        }
        if row.get("_phase_queue_binding") is not None:
            out["phase_queue_binding"] = row["_phase_queue_binding"]
        outputs.append(out)
    return outputs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def require_sha(path: Path, expected: str, description: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{description} SHA mismatch: {actual} != {expected}")
    return actual


def validate_phase_inputs(
    queue_path: Path,
    index_path: Path,
    selection_manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Validate the exact phase queue/index/selection-manifest contract."""

    selection_manifest = load_json(selection_manifest_path)
    if selection_manifest.get("schema_version") != "p107-phase-balanced-calibration-manifest-v1":
        raise ValueError("unexpected phase selection manifest schema")
    if selection_manifest.get("training_eligible") is not False:
        raise ValueError("phase selection manifest is unexpectedly training eligible")
    if selection_manifest.get("status") != "METADATA_CANDIDATES_READY_FOR_INDEPENDENT_RENDER_REVIEW":
        raise ValueError("phase selection manifest is not the sealed review-pending status")
    files = selection_manifest.get("files") or {}
    queue_file = files.get("phase_balanced_queue.jsonl") or {}
    index_file = files.get("phase_candidate_index/event_candidates.jsonl") or {}
    if not queue_file.get("sha256") or not index_file.get("sha256"):
        raise ValueError("selection manifest lacks queue/index file pins")
    hashes = {
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "queue_sha256": require_sha(queue_path, queue_file["sha256"], "phase queue"),
        "phase_index_sha256": require_sha(index_path, index_file["sha256"], "phase event index"),
    }
    queue_rows = read_jsonl(queue_path)
    index_rows = read_jsonl(index_path)
    if len(queue_rows) != queue_file.get("rows") or len(index_rows) != index_file.get("rows"):
        raise ValueError("phase queue/index row count disagrees with sealed selection manifest")
    if len(queue_rows) != 40 or len(index_rows) != 40:
        raise ValueError("phase40 producer requires exactly 40 queue/index rows")
    if selection_manifest.get("selection", {}).get("exact_budget") != 40:
        raise ValueError("selection manifest exact budget is not 40")
    return queue_rows, index_rows, selection_manifest, hashes


def validate_selection_row(queue_row: dict[str, Any], selection_manifest: dict[str, Any]) -> None:
    event_id = queue_row.get("event_id")
    summaries = selection_manifest.get("selection", {}).get("rows", [])
    matches = [item for item in summaries if item.get("event_id") == event_id]
    if len(matches) != 1:
        raise ValueError(f"selection manifest has {len(matches)} rows for event {event_id}")
    summary = matches[0]
    checks = {
        "observation_frame": queue_row.get("observation_frame"),
        "parent_event_id": queue_row.get("parent_event_id"),
        "queried_skill_id": (queue_row.get("queried_skill", {}).get("queried_skill") or {}).get("skill_id"),
        "selection_stratum": queue_row.get("selection_stratum"),
        "source_group_id": queue_row.get("source_identity", {}).get("source_group_id"),
        "task_index": queue_row.get("source_identity", {}).get("task_index"),
    }
    for key, actual in checks.items():
        if summary.get(key) != actual:
            raise ValueError(f"selection/event mismatch {event_id}.{key}: {actual!r} != {summary.get(key)!r}")


def validate_phase_binding(queue_row: dict[str, Any], indexed_event: dict[str, Any]) -> None:
    """Make the event/skill/phase/target join fail closed."""

    event_id = queue_row.get("event_id")
    normalize_phase_row(queue_row, indexed_event)  # validates identity and queried skill fields
    lineage = indexed_event.get("phase_lineage") or {}
    if lineage.get("parent_event_id") != queue_row.get("parent_event_id"):
        raise ValueError(f"phase lineage parent mismatch {event_id}")
    if lineage.get("selection_stratum") != queue_row.get("selection_stratum"):
        raise ValueError(f"phase stratum mismatch {event_id}")
    q = (queue_row.get("queried_skill") or {}).get("queried_skill") or {}
    indexed_skill = next(skill for skill in indexed_event.get("skill_bundle", []) if skill.get("skill_id") == q.get("skill_id"))
    # Target/source/destination/part are semantic query inputs.  If the queue
    # and index disagree, do not silently choose one.
    for field in ("target", "source", "destination", "target_part"):
        if q.get(field, "") != indexed_skill.get(field, ""):
            raise ValueError(f"phase target binding mismatch {event_id}.{field}")
    if indexed_event.get("event_interval") != queue_row.get("original_annotated_segment"):
        raise ValueError(f"phase event interval mismatch {event_id}")


def source_pin(
    queue_row: dict[str, Any],
    indexed_event: dict[str, Any],
    queue_path: Path,
    index_path: Path,
    selection_manifest_path: Path,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "event_id": queue_row["event_id"],
        "selection_manifest_sha256": input_hashes["selection_manifest_sha256"],
        "queue_sha256": input_hashes["queue_sha256"],
        "queue_record_sha256": canonical_digest(queue_row),
        "phase_index_sha256": input_hashes["phase_index_sha256"],
        "phase_index_record_sha256": canonical_digest(indexed_event),
        "selection_manifest_path": str(selection_manifest_path),
        "queue_path": str(queue_path),
        "phase_index_path": str(index_path),
    }


def registry_projection(
    base: dict[str, Any],
    queue_row: dict[str, Any],
    indexed_event: dict[str, Any],
    pin: dict[str, Any],
) -> dict[str, Any]:
    """Remove label-like fields and retain only the query contract."""

    current = base["current_visible_goal_relation"]
    verb = base["queried_skill_binding"]["canonical_verb"]
    current_projection = {
        "query_text": current["question"],
        "relation_family": current["relation_family"],
        "goal_scope": "CURRENT_VISIBLE_RELATION_AT_ANCHOR",
        "target_entities": current["target_entities"],
        "reference_entities": current["reference_entities"],
        "acting_entities": current.get("acting_entities", []),
        "required_visual_predicates": current["required_visual_predicates"],
        "grounding_gates": current["grounding_gates"],
        "review_unknown_reasons": current["unknown_reasons"],
        "supervision_eligibility": current["supervision_eligibility"],
        "category_grounding_status": current.get("category_grounding_status"),
        "category_grounding_unknown_count": current.get("category_grounding_unknown_count", 0),
        "metadata_attempt_is_not_outcome": True,
    }
    if verb == "PRESS":
        current_projection["press_semantics"] = "CONTACT_WITH_GROUNDED_CONTROL_ONLY; CONTACT_IS_NOT_PRESS_RESULT"
    if verb == "HANDOVER":
        current_projection["recipient_semantics"] = "EXPLICIT_RECIPIENT_OR_OTHER_ROBOT_GRIPPER_ONLY; NEVER_INVENT_PERSON"
    if verb == "NAVIGATE":
        current_projection["navigation_semantics"] = "REQUIRED_DISTANCE_AND_POSE_NEED_STATE; TARGET_VISIBILITY_IS_NOT_GOAL"
    if verb in GEOMETRY:
        current_projection["geometry_semantics"] = "CURRENT_GEOMETRY_ONLY; STATE_CHANGE_IS_SEPARATE"
    history = base.get("historical_change_relation")
    history_projection = None
    if history is not None:
        history_projection = {
            key: value
            for key, value in history.items()
            if key not in {"default_answer", "future_frames", "answer_enum"}
        }
        history_projection["scope"] = "SEPARATE_PRESTATE_TO_POSTSTATE_OR_EFFECT_RELATION"
        history_projection["review_requires_observable_prestate"] = True
    phase = base.get("phase_queue_binding") or {}
    causal_frames = base.get("evidence_policy", {}).get("actor_causal_frame_indices", [])
    anchor = base["observation_frame"]
    if not all(isinstance(frame, int) and frame <= anchor for frame in causal_frames):
        raise ValueError(f"future frame leaked into actor-causal policy for {base['event_id']}")
    record = {
        "schema_version": "p107.visual_relation_query_prelabel.v1",
        "status": "PRELABEL_QUERY_CANDIDATE_ONLY",
        "prelabel_query_id": None,
        "event_id": base["event_id"],
        "source_group_id": base["source_group_id"],
        "query_ordinal_within_event": base["query_ordinal_within_event"],
        "observation_frame": anchor,
        "queried_skill_binding": base["queried_skill_binding"],
        "current_visible_goal_relation": current_projection,
        "historical_change_relation": history_projection,
        "supporting_observation": base.get("supporting_observation"),
        "actor_evidence_policy": {
            "causal_frame_indices": causal_frames,
            "latest_allowed_frame": anchor,
            "future_actor_references_allowed": False,
            "future_use": "OFFLINE_REVIEW_ONLY_NOT_ACTOR_EVIDENCE",
        },
        "phase_binding": {
            "selection_order": phase.get("selection_order"),
            "selection_stratum": phase.get("selection_stratum"),
            "observation_phase": phase.get("observation_phase"),
            "parent_event_id": phase.get("parent_event_id"),
            "queried_skill_start_frame": phase.get("queried_skill_start_frame"),
            "queried_skill_end_frame": phase.get("queried_skill_end_frame"),
            "intent_start_causal_reference_frame": phase.get("intent_start_causal_reference_frame"),
            "terminal_query_binding": phase.get("terminal_query_binding"),
            "source_segment_end_is_not_outcome": True,
        },
        "metadata_entities_raw": base["metadata_entities_raw"],
        "category_grounding_audit": base.get("category_grounding_audit", []),
        "source_pin": pin,
        "provenance": {
            "producer": "build_memlite_visual_relation_queries.py",
            "model": "gpt-5.6-luna/max",
            "human_reviewed": False,
            "image_inspected": False,
            "labels_created": False,
        },
        "training_eligible": False,
        "usage_role": queue_row.get("usage_role"),
    }
    identity = {
        "schema_version": record["schema_version"],
        "event_id": record["event_id"],
        "source_group_id": record["source_group_id"],
        "query_ordinal_within_event": record["query_ordinal_within_event"],
        "observation_frame": record["observation_frame"],
        "queried_skill_binding": record["queried_skill_binding"],
        "current_visible_goal_relation": record["current_visible_goal_relation"],
        "historical_change_relation": record["historical_change_relation"],
        "phase_binding": record["phase_binding"],
        "source_pin": {
            key: pin[key]
            for key in (
                "selection_manifest_sha256",
                "queue_sha256",
                "queue_record_sha256",
                "phase_index_sha256",
                "phase_index_record_sha256",
            )
        },
    }
    record["prelabel_query_id"] = canonical_digest(identity)
    return record


def build_phase_registry(
    queue_path: Path,
    index_path: Path,
    selection_manifest_path: Path,
    category_mapping: dict[str, str] | None = None,
    category_mapping_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queue_rows, index_rows, selection_manifest, input_hashes = validate_phase_inputs(
        queue_path, index_path, selection_manifest_path
    )
    index_by_event: dict[str, dict[str, Any]] = {}
    for indexed in index_rows:
        event_id = indexed.get("event_id")
        if not isinstance(event_id, str) or event_id in index_by_event:
            raise ValueError(f"duplicate or invalid phase index event_id {event_id!r}")
        index_by_event[event_id] = indexed
    records: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    for ordinal, queue_row in enumerate(queue_rows):
        event_id = queue_row.get("event_id")
        if event_id not in index_by_event:
            raise ValueError(f"queue event {event_id} absent from phase index")
        validate_selection_row(queue_row, selection_manifest)
        indexed = index_by_event[event_id]
        validate_phase_binding(queue_row, indexed)
        normalized = normalize_phase_row(queue_row, indexed)
        base_records = generate_row(
            normalized,
            ordinal,
            category_mapping,
            category_mapping_metadata,
        )
        if not base_records:
            raise ValueError(f"no query candidate for phase event {event_id}")
        pin = source_pin(queue_row, indexed, queue_path, index_path, selection_manifest_path, input_hashes)
        for base in base_records:
            pair = (base["event_id"], base["query_ordinal_within_event"])
            if pair in seen_pairs:
                raise ValueError(f"duplicate event/query ordinal {pair}")
            seen_pairs.add(pair)
            record = registry_projection(base, queue_row, indexed, pin)
            query_id = record["prelabel_query_id"]
            if query_id in seen_query_ids:
                raise ValueError(f"duplicate prelabel query ID {query_id}")
            seen_query_ids.add(query_id)
            records.append(record)
    if len(records) != 40:
        raise ValueError(f"expected 40 phase queries, found {len(records)}")
    records.sort(key=lambda item: (item["phase_binding"]["selection_order"], item["query_ordinal_within_event"]))
    registry = []
    for record in records:
        registry.append({
            "schema_version": "p107.visual_relation_query_prelabel_registry.v1",
            "prelabel_query_id": record["prelabel_query_id"],
            "event_id": record["event_id"],
            "source_group_id": record["source_group_id"],
            "query_ordinal_within_event": record["query_ordinal_within_event"],
            "observation_frame": record["observation_frame"],
            "skill_id": record["queried_skill_binding"]["skill_id"],
            "canonical_verb": record["queried_skill_binding"]["canonical_verb"],
            "query_text": record["current_visible_goal_relation"]["query_text"],
            "relation_family": record["current_visible_goal_relation"]["relation_family"],
            "source_pin": record["source_pin"],
        })
    category_grounding_metadata: dict[str, Any] | None = None
    if category_mapping is not None:
        status_counts: dict[str, int] = {"RESOLVED": 0, "UNKNOWN_CATEGORY": 0}
        unique_raw: set[str] = set()
        quarantined_queries = 0
        for record in records:
            audit = record.get("category_grounding_audit", [])
            if any(item.get("status") != "RESOLVED" for item in audit):
                quarantined_queries += 1
            for item in audit:
                raw_id = item.get("raw_object_id")
                if isinstance(raw_id, str):
                    unique_raw.add(raw_id)
                status = item.get("status")
                status_counts[status] = status_counts.get(status, 0) + 1
        category_grounding_metadata = {
            **(category_mapping_metadata or {}),
            "rule": "official_longest_category_prefix_with_boundary",
            "status_counts_by_entity_occurrence": status_counts,
            "unique_raw_object_ids": len(unique_raw),
            "quarantined_queries": quarantined_queries,
        }
    metadata = {
        "schema_version": "p107.visual_relation_query_prelabel_build.v1",
        "query_template_revision": (
            QUERY_TEMPLATE_REVISION_V5 if category_mapping is not None
            else "v3_entity_grounded_parts_and_effects"
        ),
        "quality_revision": "CORRECTED_AFTER_INDEPENDENT_METADATA_REVIEW",
        "prior_revision_status": "V2_WITHDRAWN_METADATA_QUALITY_NOT_LABEL_ERRORS",
        "status": "PRELABEL_QUERY_CANDIDATES_ONLY_NO_ANSWERS_NO_ANNOTATIONS_NO_RELEASE",
        "training_eligible": False,
        "input_hashes": input_hashes,
        "selection_manifest_schema": selection_manifest.get("schema_version"),
        "query_id_rule": "sha256(canonical JSON of metadata/query identity; excludes answer, evidence, postlabel view ID, and action fields)",
        "counts": {
            "source_events": len(queue_rows),
            "prelabel_queries": len(records),
            "distinct_source_groups": len({record["source_group_id"] for record in records}),
            "multiple_query_events": len({event for event, _ in seen_pairs if sum(1 for pair in seen_pairs if pair[0] == event) > 1}),
        },
        "future_policy": "No future frame references are emitted as actor evidence; later frames remain offline review only.",
        "provenance": {
            "producer": "build_memlite_visual_relation_queries.py",
            "model": "gpt-5.6-luna/max",
            "human_reviewed": False,
            "image_inspected": False,
            "labels_created": False,
            "root_review": "PENDING",
        },
    }
    if category_grounding_metadata is not None:
        metadata["category_grounding"] = category_grounding_metadata
    return records, {"metadata": metadata, "registry": registry}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def build_output(
    queue_path: Path,
    index_path: Path,
    selection_manifest_path: Path,
    output_dir: Path,
    category_mapping_path: Path | None = None,
) -> dict[str, Any]:
    category_mapping = None
    category_mapping_metadata = None
    if category_mapping_path is not None:
        category_mapping, category_mapping_metadata = load_category_mapping(category_mapping_path)
    records, auxiliary = build_phase_registry(
        queue_path,
        index_path,
        selection_manifest_path,
        category_mapping,
        category_mapping_metadata,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / "query_candidates.jsonl"
    registry_path = output_dir / "prelabel_registry.jsonl"
    write_jsonl(query_path, records)
    write_jsonl(registry_path, auxiliary["registry"])
    metadata = dict(auxiliary["metadata"])
    metadata["outputs"] = {
        "query_candidates.jsonl": {"rows": len(records), "sha256": sha256_file(query_path)},
        "prelabel_registry.jsonl": {"rows": len(auxiliary["registry"]), "sha256": sha256_file(registry_path)},
    }
    metadata_path = output_dir / "build_metadata.json"
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    manifest = {
        **metadata,
        "schema_version": "p107.visual_relation_query_prelabel_manifest.v1",
        "outputs": {
            "query_candidates.jsonl": {"rows": len(records), "sha256": sha256_file(query_path)},
            "prelabel_registry.jsonl": {"rows": len(auxiliary["registry"]), "sha256": sha256_file(registry_path)},
            "build_metadata.json": {"rows": 1, "sha256": sha256_file(metadata_path)},
        },
        "manifest_self_hash": "EXCLUDED_FROM_SELF_HASH",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "records": len(records), "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, help="sealed phase_balanced_queue.jsonl")
    parser.add_argument("--phase-index", type=Path, required=True, help="phase_candidate_index/event_candidates.jsonl")
    parser.add_argument("--selection-manifest", type=Path, required=True, help="sealed phase_selection_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="new external prelabel-query output directory")
    parser.add_argument(
        "--category-mapping",
        type=Path,
        help="pinned official category_mapping.csv (required for v4 grounding; external input)",
    )
    args = parser.parse_args(argv)
    try:
        result = build_output(
            args.queue,
            args.phase_index,
            args.selection_manifest,
            args.output_dir,
            args.category_mapping,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"build_memlite_visual_relation_queries: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
