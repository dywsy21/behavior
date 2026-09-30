"""Compact annotation intervals and causal stage-1 targets (no image dependency).

There is no result annotation in this release. Annotation endings NEVER label
SUCCEEDED/FAILED or STOP. Every action chunk stops at any active-bundle change,
including the START of a second parallel skill and a parent-command change.
"""
from __future__ import annotations

import hashlib
from g05.utils.memlite_skill_protocol import (
    canonical_json, canonicalize_active_skills, skill_from_annotation,
    semantic_active_skills, semantic_active_skills_text, semantic_parent_goal_text,
    append_b_memory_idempotent, V6_MODEL_PROJECTION_FIELDS,
)


def fixed_phase(row, seed=17):
    identity = [seed, int(row["task_index"]), int(row["raw_episode_id"]),
                int(row["task_instance_id"])]
    return int.from_bytes(hashlib.sha256(canonical_json(identity).encode()).digest()[:8], "big") % 16


def intervals(value):
    if (isinstance(value, list) and len(value) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in value)):
        if value[0] < 0 or value[1] <= value[0]:
            raise ValueError("Malformed half-open annotation interval")
        return [tuple(value)]
    if not isinstance(value, list) or not value:
        raise ValueError("Missing annotation interval")
    return [pair for part in value for pair in intervals(part)]


def text_leaves(value):
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [x for child in value for x in text_leaves(child)]
    raise ValueError("Relation contains a non-text leaf")


def binding(verb, relation):
    result = dict(target="", source="", destination="", target_part="",
                  arm="UNSPECIFIED", binding_confidence="UNKNOWN")
    objects = relation["object_id"]
    # Flatten exactly one wrapper, never flatten groups of objects into roles.
    if len(objects) == 1 and isinstance(objects[0], list):
        objects = objects[0]
    if not all(isinstance(x, str) for x in objects):
        return result
    held = text_leaves(relation["manipulating_object_id"])
    spatial = text_leaves(relation["spatial_prefix"])
    if verb == "NAVIGATE" and len(objects) == 1 and not spatial:
        result.update(target=objects[0], binding_confidence="BOUND")
    elif verb == "GRASP" and len(objects) == 2 and held == objects[:1] and not spatial:
        result.update(target=objects[0], source=objects[1], binding_confidence="BOUND")
    elif (verb in {"PLACE_ON", "PLACE_IN", "PLACE_NEXT_TO", "PUSH", "INSERT", "HANG", "PLACE_UNDER", "ATTACH"}
          and len(objects) == 2 and held == objects[:1] and not spatial):
        result.update(target=objects[0], destination=objects[1], binding_confidence="BOUND")
    elif (verb in {"OPEN_DRAWER", "OPEN_DOOR", "CLOSE_DRAWER", "CLOSE_DOOR", "OPEN_LID",
                   "CLOSE_LID", "PRESS", "TURN_TO", "TURN_ON_SWITCH", "TURN_OFF_SWITCH",
                   "HOLD", "RELEASE", "TIP_OVER", "PUSH_TRAY", "PULL_TRAY", "LIFT"}
          and len(objects) == 1 and len(spatial) <= 1):
        result.update(target=objects[0], target_part=spatial[0] if spatial else "", binding_confidence="BOUND")
    elif verb == "HANDOVER" and len(objects) == 3 and held == objects[:1]:
        direction = {("right", "left"): "RIGHT_TO_LEFT", ("left", "right"): "LEFT_TO_RIGHT"}.get(tuple(objects[1:]))
        if direction:
            result.update(target=objects[0], arm=direction, binding_confidence="BOUND")
    return result


def compile_episode(annotation, row, task_name, *, issues=None):
    """Keep the published local frame clock; reject, do not guess, overflow.

    task_duration is NOT the LeRobot episode length. valid_duration selects a
    labeled subset on the same local clock. Ambiguous/out-of-range intervals
    are quarantined by the caller, with their reason retained in the manifest.
    """
    issues = issues if issues is not None else []
    length = int(row["length"])
    valid = intervals(annotation["meta_data"]["valid_duration"])
    if len(valid) != 1 or valid[0][1] > length:
        raise ValueError("valid_duration outside local episode clock; requires separate alignment review")
    lo, hi = valid[0]
    skills = []
    for raw in annotation["skill_annotation"]:
        if len(raw["skill_id"]) != 1 or len(raw["skill_description"]) != 1:
            raise ValueError("Unreviewed multi-valued raw skill")
        sid, desc = raw["skill_id"][0], raw["skill_description"][0]
        verb = skill_from_annotation(sid, desc)
        if verb == "SKILL_UNKNOWN":
            raise ValueError(f"Unreviewed skill pair {(sid, desc)}")
        relation = {k: raw.get(k, []) for k in
                    ("object_id", "manipulating_object_id", "spatial_prefix", "memory_prefix")}
        for interval_id, (start, end) in enumerate(intervals(raw["frame_duration"])):
            if end > length:
                raise ValueError("Skill interval outside episode; no implicit temporal warp")
            start, end = max(lo, start), min(hi, end)
            if start < end:
                skills.append(dict(skill_id=sid, skill_idx=int(raw["skill_idx"]), interval_id=interval_id,
                                   verb=verb, raw_description=desc, raw_relation=relation,
                                   skill_start=start, skill_end=end, **binding(verb, relation)))
    skills = canonicalize_active_skills(skills)
    parents = []
    for raw in annotation.get("primitive_annotation", []):
        desc = raw.get("primitive_description", [])
        if len(desc) != 1:
            continue
        members = [s for s in skills if s["skill_idx"] in raw.get("skill_idxes", [])]
        try:
            # Navigation/holding supports a primitive but does not determine
            # its argument roles. In particular, moving to a cabinet cannot
            # make that cabinet the object being placed into itself.
            primary = [s for s in members if s["raw_description"] == desc[0]]
            if not primary or any(s["binding_confidence"] != "BOUND" for s in primary):
                raise ValueError("Parent's core skill has no unambiguous argument binding")
            command = semantic_parent_goal_text(desc[0], primary)
            parent_intervals = intervals(raw["frame_duration"])
            if any(end > length for _, end in parent_intervals):
                raise ValueError("Parent interval outside episode")
        except ValueError as error:
            # A malformed OPTIONAL parent may not destroy valid leaf skills.
            # Do not infer its interval from nested scalars/member IDs. The
            # affected segment falls back to the public task, with semantic
            # parent supervision masked. The warning remains in the release.
            issues.append(dict(kind="parent_fallback", primitive_idx=raw.get("primitive_idx"), reason=str(error)))
            continue
        for start, end in parent_intervals:
            parents.append(dict(start=max(start, lo), end=min(end, hi), text=command))
    boundaries = sorted({lo, hi, *(s[k] for s in skills for k in ("skill_start", "skill_end")),
                         *(p[k] for p in parents for k in ("start", "end"))})
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [s for s in skills if s["skill_start"] <= start < s["skill_end"]]
        if not active:
            continue  # Gaps are not fake idle, recovery, terminal, or successful rows.
        commands = sorted({p["text"] for p in parents if p["start"] <= start < p["end"]})
        parent = commands[0] if len(commands) == 1 else "Task goal: " + task_name
        # Overlapping source intervals can describe the exact same semantic
        # command twice. Preserve both audit leaves, but never train repeated
        # identical commands as if they were distinct deployed instructions.
        semantic = list({canonical_json(s): s for s in semantic_active_skills(active)}.values())
        segments.append(dict(start=start, end=end, parent=parent, parent_supervised=len(commands) == 1,
                             skills=active, semantic=canonical_json(semantic),
                             text=semantic_active_skills_text(semantic)))
    if not segments:
        raise ValueError("No eligible nonterminal skills")
    return segments


def anchor_records(segments, offset):
    """Generate one candidate per fixed stride anchor, with causal prior state.

    Prior intent/parent come from the previous eligible anchor, NOT from the
    current annotation. This includes within-skill refreshes and excludes any
    future-only annotation memory. Ledger records issued commands, not success.
    """
    previous_intent, previous_parent = "None", "None"
    history = []
    records = []
    for segment_id, seg in enumerate(segments):
        start = seg["start"] + (offset - seg["start"]) % 16
        for frame in range(start, seg["end"], 16):
            records.append((frame, segment_id, previous_intent, previous_parent, list(history)))
            if previous_intent != "None" and (not history or history[-1] != previous_intent):
                history.append(previous_intent)
                history = history[-3:]
            previous_intent, previous_parent = seg["text"], seg["parent"]
    return records


def projection(segment, *, branch, task_name, previous_intent, previous_parent, history):
    memory = canonical_json(dict(task_name=task_name, issued_command_history=history, verified_world_facts=[]))
    result = dict(schema_version=6, memlite_branch=branch, task_name=task_name,
                  parent_goal=segment["parent"], target_parent_goal=segment["parent"],
                  previous_parent_goal=previous_parent, previous_intent=previous_intent,
                  memory=memory, known_previous_outcome="UNKNOWN", execution_feedback="none",
                  active_skills_semantic_json=segment["semantic"], active_skills_text=segment["text"],
                  next_decision="EXECUTE", memory_update=append_b_memory_idempotent(memory, previous_intent, task_name=task_name),
                  task_complete=False, outcome_target="UNKNOWN", outcome_supervision_mask=False,
                  parent_goal_supervision_mask=segment["parent_supervised"], low_action_supervision_mask=True)
    if set(result) != set(V6_MODEL_PROJECTION_FIELDS):
        raise AssertionError("Model projection schema drift")
    return result
