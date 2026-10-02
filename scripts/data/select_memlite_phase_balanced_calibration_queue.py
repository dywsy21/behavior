"""Build a bounded, metadata-only, phase-balanced P107 calibration mini-index.

This producer never edits the sealed full-v3 index.  It projects 40 new,
canonical event candidates from the original annotated segment bounds carried by
that index: entry, strict middle, terminal/terminal-transition, and repeated
metadata-query.  Observation phase is a sampling coordinate, not an outcome.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_memlite_event_annotation_queue as base  # noqa: E402


PHASE_QUEUE_SCHEMA = "p107-phase-balanced-calibration-queue-v1"
PHASE_MANIFEST_SCHEMA = "p107-phase-balanced-calibration-manifest-v1"
PHASE_SEAL_SCHEMA = "p107-phase-balanced-calibration-seal-v1"
PHASE_INDEX_DERIVATION_SCHEMA = "p107-phase-balanced-mini-index-v1"
STATUS = "CANDIDATE_MISSING_EVIDENCE"
GOAL_STATE_REVIEW_QUERY = "At the anchor, is the brown animal toy visibly held by the robot’s right gripper?"
GOAL_STATE_REVIEW_QUERY_SHA256 = hashlib.sha256(GOAL_STATE_REVIEW_QUERY.encode("utf-8")).hexdigest()
ROLE_ANNOTATION_CALIBRATION = "annotation_calibration"
ROLE_STUDENT_CANDIDATE = "student_candidate"
DEFAULT_SEED = "p107-phase-balanced-calibration-20261002"
DEFAULT_PHASE_QUOTA = 10
DEFAULT_TRANSITION_GAP = 1
PHASE_STRATA = ("ENTRY", "MID", "TERMINAL_OR_TRANSITION", "REPEATED_METADATA_QUERY")
REQUIRED_EVENT_FILES = ("event_candidates.jsonl", "source_groups.jsonl")
REQUIRED_OUTPUT_FILES = (
    "phase_balanced_queue.jsonl",
    "phase_selection_manifest.json",
    "phase_queue_seal.json",
    "phase_candidate_index/event_candidates.jsonl",
    "phase_candidate_index/source_groups.jsonl",
    "phase_candidate_index/manifest.json",
    "phase_candidate_index/inventory_seal.json",
)


@dataclass(frozen=True)
class PhaseCandidate:
    """One independently addressable metadata observation candidate."""

    parent: base.Candidate
    stratum: str
    observation_phase: str
    anchor_frame: int
    queried_skill: Mapping[str, Any]
    parent_skill_index: int
    repeat_distinct_episode_count: int
    successor_parent_event_id: str | None
    transition_gap_frames: int | None

    @property
    def source_group_id(self) -> str:
        return self.parent.source_group_id

    @property
    def task_id(self) -> int:
        return self.parent.task_id

    @property
    def episode_key(self) -> tuple[str, int, int]:
        return self.parent.episode_key

    @property
    def skill_id(self) -> int:
        return self.queried_skill["skill_id"]


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha(value: Any, name: str) -> str:
    if not base.is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return str(value)


def _canonical_sha256(value: Any) -> str:
    return base.canonical_sha256(value)


def _file_receipt(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result = {"sha256": base.sha256_file(path), "bytes": path.stat().st_size}
    if rows is not None:
        result["rows"] = rows
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.write_bytes(base.canonical_json(value).encode("utf-8") + b"\n")
    return _file_receipt(path, rows=1)


def _source_group_row(group: base.SourceGroup, *, protocol: Any) -> dict[str, Any]:
    return {
        "schema_version": protocol.SCHEMA_VERSION,
        "source_group_id": group.source_group_id,
        "source_release_manifest_sha256": group.source_release_manifest_sha256,
        "task_index": group.task_id,
        "task_instance_id": group.task_instance_id,
        "original_split": group.original_split,
        "usage_role": group.usage_role,
        "source_episode_ids": list(group.source_episode_ids),
        "source_episode_count": len(group.source_episode_ids),
    }


def _skill_interval(skill: Mapping[str, Any], *, parent_event_id: str) -> tuple[int, int]:
    start = _require_int(skill.get("skill_start"), f"{parent_event_id}.skill_start")
    end = _require_int(skill.get("skill_end"), f"{parent_event_id}.skill_end")
    if end <= start:
        raise ValueError("queried source skill has an invalid original interval")
    _require_int(skill.get("skill_id"), f"{parent_event_id}.skill_id")
    if not isinstance(skill.get("verb"), str) or not skill["verb"]:
        raise ValueError("queried source skill has no original verb")
    return start, end


def _skill_key(skill: Mapping[str, Any]) -> str:
    """Same task-scoped semantic key used by the parent queue's repeat logic."""
    return base.skill_key(skill)


def _phase_bundle_id(candidate: PhaseCandidate) -> str:
    fields = ("skill_id", "verb", "target", "source", "destination", "target_part", "arm",
              "skill_start", "skill_end", "raw_description")
    return _canonical_sha256({
        "kind": "p107_phase_balanced_metadata_query",
        "parent_event_id": candidate.parent.event_id,
        "parent_skill_index": candidate.parent_skill_index,
        "observation_phase": candidate.observation_phase,
        "selection_stratum": candidate.stratum,
        "observation_frame": candidate.anchor_frame,
        "queried_skill": {field: candidate.queried_skill.get(field) for field in fields},
    })


def _parent_skill_identity(candidate: PhaseCandidate) -> dict[str, Any]:
    """Expose the exact raw source-skill member used for this new anchor.

    ``skill_id`` alone is not an identity: an original segment can contain
    more than one member with the same official vocabulary ID.  The parent
    event ID, member position, raw-member digest, and original bounds make the
    selection auditable without assigning an outcome to that member.
    """
    start, end = _skill_interval(candidate.queried_skill, parent_event_id=candidate.parent.event_id)
    return {
        "parent_event_id": candidate.parent.event_id,
        "parent_skill_index": candidate.parent_skill_index,
        "parent_skill_member_sha256": _canonical_sha256(candidate.queried_skill),
        "skill_id": candidate.queried_skill["skill_id"],
        "skill_start_frame": start,
        "skill_end_frame": end,
    }


def _shifted_locators(parent_event: Mapping[str, Any], *, anchor_frame: int) -> list[dict[str, Any]]:
    locators = parent_event.get("video_locators")
    if not isinstance(locators, list) or len(locators) != 3:
        raise ValueError("phase candidate requires exactly three parent camera locators")
    result = []
    views = set()
    for locator in locators:
        if not isinstance(locator, Mapping):
            raise ValueError("parent camera locator is malformed")
        origin = locator.get("episode_start_timestamp_s")
        if type(origin) not in (int, float) or isinstance(origin, bool):
            raise ValueError("parent camera locator has no numeric episode timestamp")
        view = locator.get("view")
        if not isinstance(view, str) or not view or view in views:
            raise ValueError("parent camera locators must have unique named views")
        views.add(view)
        shifted = dict(locator)
        shifted["requested_timestamp_s"] = float(origin) + anchor_frame / 30.0
        result.append(shifted)
    if views != {"head", "left_wrist", "right_wrist"}:
        raise ValueError("phase candidate must preserve the three canonical cameras")
    return result


def _query_intent(skill: Mapping[str, Any]) -> dict[str, Any]:
    """Structured same-intent query; deliberately no outcome field exists here."""
    fields = ("skill_id", "verb", "target", "source", "destination", "target_part", "arm",
              "raw_description", "skill_start", "skill_end")
    return {
        "query_kind": "METADATA_DESCRIBED_SKILL_GOAL_RELATION",
        "query_scope": "SAME_ORIGINAL_ANNOTATED_SEGMENT_SKILL",
        "source_skill_is_attempted_instruction_not_observed_outcome": True,
        "queried_skill": {field: copy.deepcopy(skill.get(field)) for field in fields},
        "binding_status": "BOUND_METADATA_NEEDS_VISUAL_CONFIRMATION",
        "unknown_is_required_when_relation_or_entity_is_not_visually_grounded": True,
    }


def derived_event(candidate: PhaseCandidate, *, protocol: Any,
                  usage_role: str = ROLE_ANNOTATION_CALIBRATION,
                  private_diagnostic_student_candidate: bool = False) -> dict[str, Any]:
    """Create a new canonical event at the phase anchor, retaining parent lineage."""
    if usage_role not in {ROLE_ANNOTATION_CALIBRATION, ROLE_STUDENT_CANDIDATE}:
        raise ValueError("phase event role is not an approved P107 role")
    if (usage_role == ROLE_STUDENT_CANDIDATE) != private_diagnostic_student_candidate:
        raise ValueError("student_candidate phase events require the explicit private diagnostic opt-in")
    parent = candidate.parent.event
    source = copy.deepcopy(parent["source"])
    # Older sealed event fixtures carry the authenticated role on the event
    # (the source-group row is the authoritative group-level role), while
    # newer projections may also repeat it inside ``source``.  Preserve the
    # strict role/split binding without making the legacy event shape
    # impossible to derive from.
    parent_role = source.get("usage_role", parent.get("usage_role"))
    if parent_role != usage_role or source.get("original_split") != "train":
        raise ValueError("derived event role/split does not preserve the authenticated parent source")
    interval = copy.deepcopy(parent["event_interval"])
    start, end = interval["start_frame"], interval["end_frame"]
    if not start <= candidate.anchor_frame < end:
        raise ValueError("derived observation anchor escapes its original annotated segment")
    skill_start, skill_end = _skill_interval(candidate.queried_skill, parent_event_id=candidate.parent.event_id)
    parent_skill_identity = _parent_skill_identity(candidate)
    if not skill_start <= candidate.anchor_frame < skill_end:
        raise ValueError("derived anchor is outside its explicitly queried source skill")
    if candidate.observation_phase == "ENTRY" and candidate.anchor_frame != start:
        raise ValueError("entry candidate must use the original annotated segment start")
    if candidate.observation_phase == "MID" and not skill_start < candidate.anchor_frame < skill_end - 1:
        raise ValueError("mid candidate must be strictly inside the queried source skill")
    if candidate.observation_phase in {"TERMINAL", "TERMINAL_TRANSITION"}:
        if candidate.anchor_frame != end - 1 or skill_end != end:
            raise ValueError("terminal candidate must query the just-completed source skill at end-1")
    if candidate.observation_phase == "REPEATED_METADATA_QUERY" and candidate.anchor_frame != start:
        raise ValueError("repeated metadata query must use its own new entry event anchor")
    event_kind = "ANNOTATED_SKILL_SEGMENT_PHASE_" + candidate.observation_phase
    event = {
        "schema_version": protocol.SCHEMA_VERSION,
        "record_kind": "event_candidate",
        "event_id": "",
        "source": source,
        "event_kind": event_kind,
        "event_interval": interval,
        "observation": {"frame": candidate.anchor_frame, "timestamp_s": candidate.anchor_frame / 30.0},
        "action": {
            "start_frame": candidate.anchor_frame,
            "actual_executed_length": None,
            "raw_action_dim": 23,
            "model_action_dim": 27,
            "model_padding_indices": [7, 8, 17, 18],
        },
        "bundle_id": _phase_bundle_id(candidate),
        "skill_bundle": [copy.deepcopy(dict(candidate.queried_skill))],
        "parallel_bundle": False,
        "binding": [{field: candidate.queried_skill.get(field, "")
                     for field in ("verb", "target", "source", "destination", "target_part", "arm",
                                   "binding_confidence")}],
        "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "video_locators": _shifted_locators(parent, anchor_frame=candidate.anchor_frame),
        "task_name": parent.get("task_name"),
        "usage_role": usage_role,
        "phase_lineage": {
            "schema_version": PHASE_INDEX_DERIVATION_SCHEMA,
            "parent_event_id": candidate.parent.event_id,
            "parent_event_interval": {"start_frame": start, "end_frame": end},
            "queried_skill_start_frame": skill_start,
            "queried_skill_end_frame": skill_end,
            "parent_skill_identity": parent_skill_identity,
            "observation_phase": candidate.observation_phase,
            "observation_phase_is_not_outcome": True,
            "selection_stratum": candidate.stratum,
            "queried_skill_end_matches_parent_end": skill_end == end,
            "successor_parent_event_id": candidate.successor_parent_event_id,
            "transition_gap_frames": candidate.transition_gap_frames,
            "goal_query": _query_intent(candidate.queried_skill),
        },
    }
    if usage_role == ROLE_STUDENT_CANDIDATE:
        event["phase_lineage"]["private_goal_state_review"] = True
        event["phase_lineage"]["goal_state_question"] = GOAL_STATE_REVIEW_QUERY
        event["phase_lineage"]["goal_state_question_sha256"] = GOAL_STATE_REVIEW_QUERY_SHA256
        event["phase_lineage"]["training_eligible"] = False
        event["phase_lineage"]["outcome_supervision"] = False
        event["phase_lineage"]["recovery_supervision"] = False
        event["phase_lineage"]["action_bc_supervision"] = False
        event["phase_lineage"]["dart_supervision"] = False
    event["event_id"] = protocol.event_id(event)
    protocol.validate_event(event)
    if event["event_id"] == candidate.parent.event_id:
        raise ValueError("phase event must have a new canonical event identity")
    return event


def _adjacent_successors(candidates: Sequence[base.Candidate], *, max_gap_frames: int) -> dict[str, tuple[str, int]]:
    grouped: dict[tuple[str, int, int], list[base.Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.episode_key].append(candidate)
    successors: dict[str, tuple[str, int]] = {}
    for rows in grouped.values():
        rows.sort(key=lambda item: (item.interval_start, item.interval_end, item.event_id))
        for index, current in enumerate(rows[:-1]):
            following = rows[index + 1]
            gap = following.interval_start - current.interval_end
            if 0 <= gap <= max_gap_frames:
                successors[current.event_id] = (following.event_id, gap)
    return successors


def derive_candidates(candidates: Sequence[base.Candidate], *, max_transition_gap_frames: int) -> list[PhaseCandidate]:
    """Project valid anchors only from source segment/skill bounds, never result fields."""
    if max_transition_gap_frames < 0:
        raise ValueError("max transition gap must be nonnegative")
    successors = _adjacent_successors(candidates, max_gap_frames=max_transition_gap_frames)
    repeats: dict[tuple[int, str], set[tuple[str, int, int]]] = defaultdict(set)
    for candidate in candidates:
        for skill in candidate.event["skill_bundle"]:
            _skill_interval(skill, parent_event_id=candidate.event_id)
            repeats[candidate.task_id, _skill_key(skill)].add(candidate.episode_key)

    result: list[PhaseCandidate] = []
    for parent in candidates:
        start, end = parent.interval_start, parent.interval_end
        midpoint = start + (end - start - 1) // 2
        successor = successors.get(parent.event_id)
        for member_index, original_skill in enumerate(parent.event["skill_bundle"]):
            skill = copy.deepcopy(dict(original_skill))
            skill_start, skill_end = _skill_interval(skill, parent_event_id=parent.event_id)
            repeat_count = len(repeats[parent.task_id, _skill_key(skill)])
            if skill_start == start:
                result.append(PhaseCandidate(parent, "ENTRY", "ENTRY", start, skill, member_index,
                                             repeat_count, None, None))
                if repeat_count >= 2:
                    result.append(PhaseCandidate(parent, "REPEATED_METADATA_QUERY", "REPEATED_METADATA_QUERY",
                                                 start, skill, member_index, repeat_count, None, None))
            if skill_start < midpoint < skill_end - 1:
                result.append(PhaseCandidate(parent, "MID", "MID", midpoint, skill, member_index,
                                             repeat_count, None, None))
            if skill_end == end:
                if successor is None:
                    phase, successor_id, gap = "TERMINAL", None, None
                else:
                    phase, successor_id, gap = "TERMINAL_TRANSITION", successor[0], successor[1]
                result.append(PhaseCandidate(parent, "TERMINAL_OR_TRANSITION", phase, end - 1, skill,
                                             member_index, repeat_count, successor_id, gap))
    return result


def _candidate_tie(candidate: PhaseCandidate, *, seed: str) -> str:
    return _canonical_sha256({
        "seed": seed,
        "parent_event_id": candidate.parent.event_id,
        "stratum": candidate.stratum,
        "observation_phase": candidate.observation_phase,
        "anchor_frame": candidate.anchor_frame,
        "queried_skill": _skill_key(candidate.queried_skill),
        "parent_skill_index": candidate.parent_skill_index,
    })


def select_balanced(candidates: Sequence[PhaseCandidate], *, seed: str, phase_quota: int,
                    max_per_source_group: int, max_per_episode: int) -> list[PhaseCandidate]:
    """Deterministically select exact quotas with group/episode holdout protection."""
    if phase_quota != DEFAULT_PHASE_QUOTA:
        raise ValueError("phase quota is fixed at 10 for the approved bounded 40-candidate batch")
    if max_per_source_group != 1 or max_per_episode != 1:
        raise ValueError("phase-balanced calibration must preserve one whole source group and episode per selected row")
    by_stratum: dict[str, list[PhaseCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.stratum not in PHASE_STRATA:
            raise ValueError("unknown phase stratum")
        by_stratum[candidate.stratum].append(candidate)
    selected: list[PhaseCandidate] = []
    groups: Counter[str] = Counter()
    episodes: Counter[tuple[str, int, int]] = Counter()
    tasks: Counter[int] = Counter()
    skills: Counter[int] = Counter()
    # Allocate rarer terminal/mid/repeated strata before redundant entries.
    allocation_order = ("TERMINAL_OR_TRANSITION", "MID", "REPEATED_METADATA_QUERY", "ENTRY")
    for stratum in allocation_order:
        while sum(item.stratum == stratum for item in selected) < phase_quota:
            available = [item for item in by_stratum[stratum]
                         if groups[item.source_group_id] < max_per_source_group and
                         episodes[item.episode_key] < max_per_episode]
            if not available:
                raise ValueError(f"insufficient safe candidates for exact {stratum} quota")
            available.sort(key=lambda item: (
                tasks[item.task_id], skills[item.skill_id],
                0 if item.observation_phase == "TERMINAL_TRANSITION" else 1,
                -item.repeat_distinct_episode_count,
                _candidate_tie(item, seed=seed),
            ))
            chosen = available[0]
            selected.append(chosen)
            groups[chosen.source_group_id] += 1
            episodes[chosen.episode_key] += 1
            tasks[chosen.task_id] += 1
            skills[chosen.skill_id] += 1
    if len(selected) != len(PHASE_STRATA) * phase_quota:
        raise AssertionError("internal exact-budget drift")
    return selected


def _temporal_windows(anchor: int, *, episode_length: int) -> dict[str, Any]:
    """Causal/review windows are recomputed around the new derived event anchor."""
    return base.window_spec(
        base.Candidate(event={}, event_id="0" * 64, source_group_id="0" * 64, task_id=0,
                       task_instance_id=1, episode_key=("0" * 64, 0, 0), episode_index=0,
                       episode_length=episode_length, anchor_frame=anchor, interval_start=anchor,
                       interval_end=anchor + 1, skill_ids=(), raw_source_verbs=(), skill_keys=(),
                       repeat_attempt_count=0, boundary_before=False, boundary_after=False,
                       long_interval=False, candidate_signals=(), selection_stratum="NORMAL_CONTROL_CANDIDATE"),
        actor_history_frames=60, review_before_frames=60, review_after_frames=60, sample_stride_frames=15)


def queue_row(candidate: PhaseCandidate, event: Mapping[str, Any], *, selection_order: int,
              usage_role: str = ROLE_ANNOTATION_CALIBRATION) -> dict[str, Any]:
    if usage_role not in {ROLE_ANNOTATION_CALIBRATION, ROLE_STUDENT_CANDIDATE}:
        raise ValueError("phase queue role is not an approved P107 role")
    if event.get("usage_role") != usage_role:
        raise ValueError("phase queue role does not match derived event role")
    source = event["source"]
    windows = _temporal_windows(event["observation"]["frame"], episode_length=source["episode_length"])
    row = {
        "schema_version": PHASE_QUEUE_SCHEMA,
        "selection_order": selection_order,
        "event_id": event["event_id"],
        "parent_event_id": candidate.parent.event_id,
        "status": STATUS,
        "training_eligible": False,
        "usage_role": usage_role,
        "immutable_split": "train",
        "source_identity": {
            "source_release_manifest_sha256": source["source_release_manifest_sha256"],
            "source_annotation_sha256": source["source_annotation_sha256"],
            "source_group_id": source["source_group_id"],
            "task_index": source["task_index"],
            "task_instance_id": source["task_instance_id"],
            "raw_episode_id": source["raw_episode_id"],
            "episode_index": source["episode_index"],
        },
        "original_annotated_segment": copy.deepcopy(candidate.parent.event["event_interval"]),
        "observation_phase": candidate.observation_phase,
        "selection_stratum": candidate.stratum,
        "observation_frame": event["observation"]["frame"],
        "observation_phase_is_not_outcome": True,
        "actor_observation_is_new_event_anchor_not_parent_future_context": True,
        "queried_skill": _query_intent(candidate.queried_skill),
        "queried_skill_start_frame": candidate.queried_skill["skill_start"],
        "queried_skill_end_frame": candidate.queried_skill["skill_end"],
        "parent_skill_identity": _parent_skill_identity(candidate),
        "intent_start_causal_reference": {
            "frame": candidate.queried_skill["skill_start"],
            "timestamp_s": candidate.queried_skill["skill_start"] / 30.0,
            "relation_to_observation": "AT_OR_BEFORE_OBSERVATION",
            "availability": "METADATA_REFERENCE_ONLY_NO_HISTORY_IMAGES_RENDERED_BY_THIS_QUEUE",
        },
        "terminal_query_binding": "QUERIED_SKILL_END_EQUALS_ORIGINAL_SEGMENT_END" if
        candidate.observation_phase in {"TERMINAL", "TERMINAL_TRANSITION"} else None,
        "transition": {
            "successor_parent_event_id": candidate.successor_parent_event_id,
            "gap_frames": candidate.transition_gap_frames,
            "successor_never_supplies_queried_skill": True,
        },
        "repeated_metadata_query_distinct_source_episode_count": candidate.repeat_distinct_episode_count,
        "temporal_windows": windows,
        "current_actor_evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "review_constraints": {
            "source_segment_end_is_not_completion_truth": True,
            "no_success_failure_outcome_or_recovery_label": True,
            "future_frames_are_only_offline_review_for_this_new_anchor": True,
            "no_action_supervision_emitted": True,
        },
    }
    if usage_role == ROLE_STUDENT_CANDIDATE:
        row["private_goal_state_review"] = True
        row["goal_state_question"] = GOAL_STATE_REVIEW_QUERY
        row["goal_state_question_sha256"] = GOAL_STATE_REVIEW_QUERY_SHA256
        row["action_bc_supervision"] = False
        row["outcome_supervision"] = False
        row["recovery_supervision"] = False
        row["dart_supervision"] = False
    return row


def _coverage(selected: Sequence[PhaseCandidate], all_candidates: Sequence[PhaseCandidate],
              coverage_expectations: Mapping[str, Any]) -> dict[str, Any]:
    expected_tasks = set(coverage_expectations["expected_task_ids"])
    expected_skills = {row["skill_id"] for row in coverage_expectations["expected_skill_vocabulary"]}
    selected_tasks = {item.task_id for item in selected}
    selected_skills = {item.skill_id for item in selected}
    pool_tasks = {item.task_id for item in all_candidates}
    pool_skills = {item.skill_id for item in all_candidates}
    return {
        "expected_task_count": len(expected_tasks),
        "expected_skill_count": len(expected_skills),
        "pool_task_count": len(pool_tasks),
        "pool_skill_count": len(pool_skills),
        "selected_task_count": len(selected_tasks),
        "selected_skill_count": len(selected_skills),
        "missing_expected_task_ids_from_pool": sorted(expected_tasks - pool_tasks),
        "missing_expected_skill_ids_from_pool": sorted(expected_skills - pool_skills),
        "missing_expected_task_ids_from_selection": sorted(expected_tasks - selected_tasks),
        "missing_expected_skill_ids_from_selection": sorted(expected_skills - selected_skills),
    }


def _phase_counts(items: Iterable[PhaseCandidate]) -> dict[str, int]:
    return dict(sorted(Counter(item.stratum for item in items).items()))


def _payloads(index: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    usage_role = getattr(args, "usage_role", ROLE_ANNOTATION_CALIBRATION)
    private_student = bool(getattr(args, "private_diagnostic_student_candidate", False))
    parent_event_id = getattr(args, "goal_state_parent_event_id", None)
    if usage_role not in {ROLE_ANNOTATION_CALIBRATION, ROLE_STUDENT_CANDIDATE}:
        raise ValueError("--usage-role must be annotation_calibration or student_candidate")
    if (usage_role == ROLE_STUDENT_CANDIDATE) != private_student:
        raise ValueError("student_candidate requires --private-diagnostic-student-candidate")
    if usage_role == ROLE_STUDENT_CANDIDATE and not isinstance(parent_event_id, str):
        raise ValueError("student_candidate goal-state review requires --goal-state-parent-event-id")
    if usage_role == ROLE_ANNOTATION_CALIBRATION and parent_event_id is not None:
        raise ValueError("--goal-state-parent-event-id is only valid for private student_candidate review")
    protocol = base.load_canonical_protocol(args.protocol_path, expected_sha256=args.expected_protocol_sha256)
    manifest, event_rows, groups = base.validate_index(
        index, expected_source_manifest_sha256=args.expected_source_release_manifest_sha256,
        canonical_protocol=protocol, expected_protocol_sha256=args.expected_protocol_sha256)
    inventory_seal, inventory_seal_sha256 = base.validate_inventory_seal(
        index, expected_inventory_seal_sha256=args.expected_inventory_seal_sha256)
    coverage_expectations = base.read_coverage_expectations(
        args.coverage_expectations, expected_sha256=inventory_seal["coverage_expectations_sha256"])
    if manifest.get("coverage_expectations_sha256") != _canonical_sha256(coverage_expectations):
        raise ValueError("parent index coverage contract does not match the sealed expectation")
    retained, excluded_roles, excluded_vocab = base.parse_candidates(
        event_rows, groups, expected_source_manifest_sha256=args.expected_source_release_manifest_sha256,
        canonical_protocol=protocol, expected_task_ids=coverage_expectations["expected_task_ids"],
        expected_skill_ids=[row["skill_id"] for row in coverage_expectations["expected_skill_vocabulary"]],
        retained_usage_roles={usage_role}, max_retained_candidates=args.max_retained_candidates,
        expected_event_receipt=manifest["files"]["event_candidates.jsonl"], boundary_gap_frames=args.transition_gap_frames,
        long_interval_frames=1,
        retained_event_ids=({parent_event_id} if usage_role == ROLE_STUDENT_CANDIDATE else None))
    if any(groups[item.source_group_id].original_split != "train" or
           groups[item.source_group_id].usage_role != usage_role for item in retained):
        raise ValueError("phase sampler may retain only immutable TRAIN sources of the selected role")
    all_phase_candidates = derive_candidates(retained, max_transition_gap_frames=args.transition_gap_frames)
    if usage_role == ROLE_STUDENT_CANDIDATE:
        parent_matches = [item for item in all_phase_candidates if item.parent.event_id == parent_event_id]
        selected = [item for item in parent_matches if item.observation_phase in {"ENTRY", "TERMINAL"}]
        if len(selected) != 2 or {item.observation_phase for item in selected} != {"ENTRY", "TERMINAL"}:
            raise ValueError("goal-state student review parent must produce exactly ENTRY and TERMINAL candidates")
        if any(item.parent.event.get("usage_role") != usage_role for item in selected):
            raise ValueError("goal-state parent role drifted from student_candidate")
    else:
        selected = select_balanced(all_phase_candidates, seed=args.seed, phase_quota=args.phase_quota,
                                   max_per_source_group=args.max_per_source_group,
                                   max_per_episode=args.max_per_episode)
        if _phase_counts(selected) != {phase: args.phase_quota for phase in PHASE_STRATA}:
            raise AssertionError("phase selection does not meet exact four-way quota")
    events_and_rows = []
    for order, candidate in enumerate(selected):
        event = derived_event(candidate, protocol=protocol, usage_role=usage_role,
                              private_diagnostic_student_candidate=private_student)
        events_and_rows.append((candidate, event, queue_row(candidate, event, selection_order=order,
                                                            usage_role=usage_role)))
    event_ids = [event["event_id"] for _, event, _ in events_and_rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("derived phase candidates collided in canonical event identity")
    selected_group_ids = {row["source_identity"]["source_group_id"] for _, _, row in events_and_rows}
    if usage_role == ROLE_STUDENT_CANDIDATE:
        if len(selected_group_ids) != 1:
            raise ValueError("private student goal-state phases must share exactly one authenticated source group")
    elif len(selected_group_ids) != len(events_and_rows):
        raise ValueError("phase selection crossed the whole-source-group holdout cap")
    selected_groups = {candidate.source_group_id for candidate, _, _ in events_and_rows}
    source_group_rows = [_source_group_row(groups[group_id], protocol=protocol) for group_id in sorted(selected_groups)]
    phase_policy_core = {
        "schema_version": PHASE_QUEUE_SCHEMA,
        "algorithm": "p107-phase-balanced-original-segment-bounds-v1",
        "seed": args.seed,
        "phase_quota": args.phase_quota,
        "phase_strata": list(PHASE_STRATA),
        "max_per_source_group": args.max_per_source_group,
        "max_per_episode": args.max_per_episode,
        "transition_gap_frames": args.transition_gap_frames,
        "max_retained_candidates": args.max_retained_candidates,
        "metadata_only": True,
        "no_rgb_decode": True,
        "observation_phase_is_not_outcome": True,
        "repeated_query_is_not_retry_or_recovery": True,
        "parent_inventory_seal_sha256": inventory_seal_sha256,
        "parent_index_manifest_sha256": base.sha256_file(index / "manifest.json"),
        "parent_event_candidates_sha256": manifest["files"]["event_candidates.jsonl"]["sha256"],
        "parent_source_groups_sha256": manifest["files"]["source_groups.jsonl"]["sha256"],
        "source_release_manifest_sha256": args.expected_source_release_manifest_sha256,
        "protocol_sha256": args.expected_protocol_sha256,
        "coverage_expectations_sha256": inventory_seal["coverage_expectations_sha256"],
        "usage_role": usage_role,
        "private_diagnostic_student_candidate": private_student,
        "goal_state_parent_event_id": parent_event_id,
    }
    if private_student:
        phase_policy_core["goal_state_question"] = GOAL_STATE_REVIEW_QUERY
        phase_policy_core["goal_state_question_sha256"] = GOAL_STATE_REVIEW_QUERY_SHA256
    policy = {**phase_policy_core, "policy_sha256": _canonical_sha256(phase_policy_core)}
    return {
        "protocol": protocol,
        "parent_manifest": manifest,
        "parent_inventory_seal": inventory_seal,
        "parent_inventory_seal_sha256": inventory_seal_sha256,
        "coverage_expectations": coverage_expectations,
        "excluded_roles": dict(sorted(excluded_roles.items())),
        "excluded_vocabulary": dict(sorted(excluded_vocab.items())),
        "all_phase_candidates": all_phase_candidates,
        "selected": selected,
        "events": [event for _, event, _ in events_and_rows],
        "queue_rows": [row for _, _, row in events_and_rows],
        "source_group_rows": source_group_rows,
        "policy": policy,
        "coverage": _coverage(selected, all_phase_candidates, coverage_expectations),
        "usage_role": usage_role,
        "private_diagnostic_student_candidate": private_student,
        "goal_state_parent_event_id": parent_event_id,
    }


def _mini_index_manifest(payloads: Mapping[str, Any], *, files: Mapping[str, Any]) -> dict[str, Any]:
    parent = payloads["parent_manifest"]
    usage_role = payloads.get("usage_role", ROLE_ANNOTATION_CALIBRATION)
    return {
        "schema_version": base.INDEX_SCHEMA,
        "status": base.INDEX_STATUS,
        "training_eligible": False,
        "source_release_manifest_sha256": parent["source_release_manifest_sha256"],
        "protocol_sha256": payloads["policy"]["protocol_sha256"],
        "coverage_expectations_sha256": payloads["policy"]["coverage_expectations_sha256"],
        "files": files,
        "event_candidates": len(payloads["events"]),
        "source_groups": len(payloads["source_group_rows"]),
        "source_episodes": len(payloads["source_group_rows"]),
        "usage_roles": {usage_role: len(payloads["source_group_rows"])},
        "outcome_supervision": False,
        "corrective_action_supervision": False,
        "partial_source_coverage": True,
        "derivation": {
            "schema_version": PHASE_INDEX_DERIVATION_SCHEMA,
            "parent_inventory_seal_sha256": payloads["parent_inventory_seal_sha256"],
            "parent_index_manifest_sha256": payloads["policy"]["parent_index_manifest_sha256"],
            "parent_event_candidates_sha256": payloads["policy"]["parent_event_candidates_sha256"],
            "original_v3_rows_unchanged": True,
            "new_events_are_canonical_phase_candidates": True,
            "phase_observation_is_not_outcome": True,
            "renderer_compatible_with_pinned_protocol": True,
            "usage_role_preserved": usage_role,
            "private_goal_state_review": payloads.get("private_diagnostic_student_candidate", False),
        },
    }


def _selection_manifest(payloads: Mapping[str, Any], *, files: Mapping[str, Any],
                        mini_index_manifest_sha256: str, mini_index_inventory_seal_sha256: str) -> dict[str, Any]:
    selected = payloads["selected"]
    event_rows = payloads["queue_rows"]
    usage_role = payloads.get("usage_role", ROLE_ANNOTATION_CALIBRATION)
    private_student = bool(payloads.get("private_diagnostic_student_candidate", False))
    return {
        "schema_version": PHASE_MANIFEST_SCHEMA,
        "status": "METADATA_CANDIDATES_READY_FOR_INDEPENDENT_RENDER_REVIEW",
        "training_eligible": False,
        "source_release_manifest_sha256": payloads["policy"]["source_release_manifest_sha256"],
        "parent_index": {
            "inventory_seal_sha256": payloads["parent_inventory_seal_sha256"],
            "index_manifest_sha256": payloads["policy"]["parent_index_manifest_sha256"],
            "event_candidates_sha256": payloads["policy"]["parent_event_candidates_sha256"],
            "source_groups_sha256": payloads["policy"]["parent_source_groups_sha256"],
            "protocol_sha256": payloads["policy"]["protocol_sha256"],
        },
        "mini_index": {
            "relative_path": "phase_candidate_index",
            "manifest_sha256": mini_index_manifest_sha256,
            "inventory_seal_sha256": mini_index_inventory_seal_sha256,
            "renderer_compatible": True,
            "decode_requires_independent_review": True,
        },
        "policy": payloads["policy"],
        "selection": {
            "exact_budget": len(event_rows),
            "phase_stratum_counts": _phase_counts(selected),
            "observation_phase_counts": dict(sorted(Counter(item.observation_phase for item in selected).items())),
            "unique_source_groups": len({item.source_group_id for item in selected}),
            "unique_source_episodes": len({item.episode_key for item in selected}),
            "all_usage_role_annotation_calibration": usage_role == ROLE_ANNOTATION_CALIBRATION,
            "all_usage_role_student_candidate_private": usage_role == ROLE_STUDENT_CANDIDATE and
            private_student,
            "usage_role": usage_role,
            "all_immutable_split_train": True,
            "all_evidence_missing": True,
            "coverage": payloads["coverage"],
            "excluded_parent_index_events_by_role": payloads["excluded_roles"],
            "excluded_parent_index_events_by_unrecognized_vocabulary": payloads["excluded_vocabulary"],
            "rows": [{
                "event_id": row["event_id"], "parent_event_id": row["parent_event_id"],
                "observation_phase": row["observation_phase"], "selection_stratum": row["selection_stratum"],
                "observation_frame": row["observation_frame"],
                "task_index": row["source_identity"]["task_index"],
                "source_group_id": row["source_identity"]["source_group_id"],
                "queried_skill_id": row["queried_skill"]["queried_skill"]["skill_id"],
            } for row in event_rows],
        },
        "files": files,
        "interpretation_guardrails": [
            "Observation phase is a source-clock sampling coordinate, never a success, failure, attempt, or recovery result.",
            "TERMINAL and TERMINAL_TRANSITION query the source skill whose original end equals the parent segment end; a successor is lineage only.",
            "REPEATED_METADATA_QUERY means repeated metadata binding across distinct source episodes, not retry or recovery.",
            "A later anchor is a new canonical event and its later source frames are not future context for the parent event.",
            "student_candidate is private diagnostic review only and carries no training or outcome authorization.",
        ],
    }


def _queue_seal(manifest_sha256: str, *, payloads: Mapping[str, Any], files: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PHASE_SEAL_SCHEMA,
        "selection_manifest_sha256": manifest_sha256,
        "parent_inventory_seal_sha256": payloads["parent_inventory_seal_sha256"],
        "source_release_manifest_sha256": payloads["policy"]["source_release_manifest_sha256"],
        "protocol_sha256": payloads["policy"]["protocol_sha256"],
        "payload_files": files,
        "training_eligible": False,
    }


def _write_payloads(output: Path, payloads: Mapping[str, Any]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        mini = staging / "phase_candidate_index"
        mini.mkdir()
        events = sorted(payloads["events"], key=lambda item: item["event_id"])
        groups = sorted(payloads["source_group_rows"], key=lambda item: item["source_group_id"])
        mini_files = {
            "event_candidates.jsonl": base.write_jsonl(mini / "event_candidates.jsonl", events),
            "source_groups.jsonl": base.write_jsonl(mini / "source_groups.jsonl", groups),
        }
        mini_manifest = _mini_index_manifest(payloads, files=mini_files)
        _write_json(mini / "manifest.json", mini_manifest)
        mini_manifest_sha256 = base.sha256_file(mini / "manifest.json")
        mini_seal = {
            "schema_version": "memlite-event-inventory-seal-v1",
            "index_manifest_sha256": mini_manifest_sha256,
            "source_release_manifest_sha256": mini_manifest["source_release_manifest_sha256"],
            "coverage_expectations_sha256": mini_manifest["coverage_expectations_sha256"],
            "expected_payload_files": list(REQUIRED_EVENT_FILES),
            "payload_files": mini_files,
        }
        _write_json(mini / "inventory_seal.json", mini_seal)
        mini_seal_sha256 = base.sha256_file(mini / "inventory_seal.json")
        queue_receipt = base.write_jsonl(staging / "phase_balanced_queue.jsonl", payloads["queue_rows"])
        # These are the non-self-referential payload receipts.  The selection
        # manifest and seal bind them, but neither recursively claims a hash
        # of itself.
        payload_files = {
            "phase_balanced_queue.jsonl": queue_receipt,
            "phase_candidate_index/event_candidates.jsonl": mini_files["event_candidates.jsonl"],
            "phase_candidate_index/source_groups.jsonl": mini_files["source_groups.jsonl"],
            "phase_candidate_index/manifest.json": _file_receipt(mini / "manifest.json", rows=1),
            "phase_candidate_index/inventory_seal.json": _file_receipt(mini / "inventory_seal.json", rows=1),
        }
        selection_manifest = _selection_manifest(payloads, files=payload_files,
                                                 mini_index_manifest_sha256=mini_manifest_sha256,
                                                 mini_index_inventory_seal_sha256=mini_seal_sha256)
        manifest_receipt = _write_json(staging / "phase_selection_manifest.json", selection_manifest)
        selection_manifest_sha256 = manifest_receipt["sha256"]
        seal = _queue_seal(selection_manifest_sha256, payloads=payloads, files=payload_files)
        seal_receipt = _write_json(staging / "phase_queue_seal.json", seal)
        actual_files = {
            "phase_balanced_queue.jsonl",
            "phase_selection_manifest.json",
            "phase_queue_seal.json",
            "phase_candidate_index/event_candidates.jsonl",
            "phase_candidate_index/source_groups.jsonl",
            "phase_candidate_index/manifest.json",
            "phase_candidate_index/inventory_seal.json",
        }
        if actual_files != set(REQUIRED_OUTPUT_FILES):
            raise AssertionError("phase output file inventory drift")
        os.rename(staging, output)
        return {
            "status": "CREATED_CANDIDATE_ONLY",
            "selection_manifest_sha256": selection_manifest_sha256,
            "phase_queue_seal_sha256": seal_receipt["sha256"],
            "mini_index_manifest_sha256": mini_manifest_sha256,
            "mini_index_inventory_seal_sha256": mini_seal_sha256,
            "selected": len(payloads["queue_rows"]),
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_json(path: Path) -> Mapping[str, Any]:
    value = base.read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    return value


def _validate_output_layout(output: Path) -> None:
    """Reject a resumed directory with additions, links, or missing payloads."""
    actual: set[str] = set()
    for path in output.rglob("*"):
        if path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("phase output may contain only regular sealed payload files")
        actual.add(path.relative_to(output).as_posix())
    if actual != set(REQUIRED_OUTPUT_FILES):
        raise ValueError("phase output payload file layout is incomplete or expanded")


def _validate_existing(output: Path, payloads: Mapping[str, Any], *, expected_phase_queue_seal_sha256: str) -> dict[str, Any]:
    _sha(expected_phase_queue_seal_sha256, "expected phase queue seal")
    if output.is_symlink() or not output.is_dir():
        raise ValueError("phase output must be an existing regular directory for resume")
    _validate_output_layout(output)
    actual_rows = base.read_jsonl(output / "phase_balanced_queue.jsonl")
    expected_rows = payloads["queue_rows"]
    if base.canonical_json(actual_rows) != base.canonical_json(expected_rows):
        raise ValueError("phase queue rows differ from fresh canonical reconstruction")
    mini = output / "phase_candidate_index"
    actual_events = base.read_jsonl(mini / "event_candidates.jsonl")
    expected_events = sorted(payloads["events"], key=lambda item: item["event_id"])
    if base.canonical_json(actual_events) != base.canonical_json(expected_events):
        raise ValueError("phase mini-index events differ from fresh canonical reconstruction")
    actual_groups = base.read_jsonl(mini / "source_groups.jsonl")
    expected_groups = sorted(payloads["source_group_rows"], key=lambda item: item["source_group_id"])
    if base.canonical_json(actual_groups) != base.canonical_json(expected_groups):
        raise ValueError("phase mini-index source groups differ from fresh canonical reconstruction")
    mini_manifest = _read_json(mini / "manifest.json")
    mini_files = {
        "event_candidates.jsonl": _file_receipt(mini / "event_candidates.jsonl", rows=len(actual_events)),
        "source_groups.jsonl": _file_receipt(mini / "source_groups.jsonl", rows=len(actual_groups)),
    }
    if base.canonical_json(mini_manifest) != base.canonical_json(_mini_index_manifest(payloads, files=mini_files)):
        raise ValueError("phase mini-index manifest differs from canonical reconstruction")
    mini_seal = _read_json(mini / "inventory_seal.json")
    expected_mini_seal = {
        "schema_version": "memlite-event-inventory-seal-v1",
        "index_manifest_sha256": base.sha256_file(mini / "manifest.json"),
        "source_release_manifest_sha256": mini_manifest["source_release_manifest_sha256"],
        "coverage_expectations_sha256": mini_manifest["coverage_expectations_sha256"],
        "expected_payload_files": list(REQUIRED_EVENT_FILES),
        "payload_files": mini_files,
    }
    if base.canonical_json(mini_seal) != base.canonical_json(expected_mini_seal):
        raise ValueError("phase mini-index inventory seal differs from canonical reconstruction")
    payload_files = {
        "phase_balanced_queue.jsonl": _file_receipt(output / "phase_balanced_queue.jsonl", rows=len(actual_rows)),
        "phase_candidate_index/event_candidates.jsonl": mini_files["event_candidates.jsonl"],
        "phase_candidate_index/source_groups.jsonl": mini_files["source_groups.jsonl"],
        "phase_candidate_index/manifest.json": _file_receipt(mini / "manifest.json", rows=1),
        "phase_candidate_index/inventory_seal.json": _file_receipt(mini / "inventory_seal.json", rows=1),
    }
    selection_manifest = _read_json(output / "phase_selection_manifest.json")
    selection_manifest_receipt = _file_receipt(output / "phase_selection_manifest.json", rows=1)
    expected_selection = _selection_manifest(payloads, files=payload_files,
                                             mini_index_manifest_sha256=base.sha256_file(mini / "manifest.json"),
                                             mini_index_inventory_seal_sha256=base.sha256_file(mini / "inventory_seal.json"))
    if base.canonical_json(selection_manifest) != base.canonical_json(expected_selection):
        raise ValueError("phase selection manifest differs from canonical reconstruction")
    seal = _read_json(output / "phase_queue_seal.json")
    if base.sha256_file(output / "phase_queue_seal.json") != expected_phase_queue_seal_sha256:
        raise ValueError("phase queue seal does not match the externally supplied SHA-256")
    expected_seal = _queue_seal(selection_manifest_receipt["sha256"], payloads=payloads, files=payload_files)
    if base.canonical_json(seal) != base.canonical_json(expected_seal):
        raise ValueError("phase queue seal differs from canonical reconstruction")
    return {"status": "RESUME_VALIDATED", "phase_queue_seal_sha256": expected_phase_queue_seal_sha256,
            "selected": len(actual_rows)}


def build_phase_queue(index: Path, output: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    index, output = Path(index), Path(output)
    if output.exists():
        raise FileExistsError("phase output exists; use --resume with the externally recorded seal")
    base.forbid_output_inside(output, index)
    started = time.monotonic()
    payloads = _payloads(index, args=args)
    if time.monotonic() - started > args.max_seconds:
        raise TimeoutError("phase metadata selection exceeded its CPU wall budget")
    return _write_payloads(output, payloads)


def resume_phase_queue(index: Path, output: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_phase_queue_seal_sha256 is None:
        raise ValueError("--resume requires --expected-phase-queue-seal-sha256")
    started = time.monotonic()
    payloads = _payloads(Path(index), args=args)
    if time.monotonic() - started > args.max_seconds:
        raise TimeoutError("phase metadata resume exceeded its CPU wall budget")
    return _validate_existing(Path(output), payloads,
                              expected_phase_queue_seal_sha256=args.expected_phase_queue_seal_sha256)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", type=Path, required=True, help="sealed immutable full-v3 parent event index")
    result.add_argument("--output", type=Path, required=True, help="new external phase-balanced candidate directory")
    result.add_argument("--expected-source-release-manifest-sha256", default=base.DEFAULT_SOURCE_MANIFEST_SHA256)
    result.add_argument("--expected-inventory-seal-sha256", required=True)
    result.add_argument("--coverage-expectations", type=Path, required=True)
    result.add_argument("--protocol-path", type=Path, required=True,
                        help="owner protocol snapshot used to create the parent index")
    result.add_argument("--expected-protocol-sha256", required=True)
    result.add_argument("--seed", default=DEFAULT_SEED)
    result.add_argument("--phase-quota", type=int, default=DEFAULT_PHASE_QUOTA)
    result.add_argument("--max-per-source-group", type=int, default=1)
    result.add_argument("--max-per-episode", type=int, default=1)
    result.add_argument("--transition-gap-frames", type=int, default=DEFAULT_TRANSITION_GAP)
    result.add_argument("--max-retained-candidates", type=int, default=base.ABSOLUTE_MAX_RETAINED_CANDIDATES)
    result.add_argument("--usage-role", choices=(ROLE_ANNOTATION_CALIBRATION, ROLE_STUDENT_CANDIDATE),
                        default=ROLE_ANNOTATION_CALIBRATION,
                        help="preserve the authenticated source role; student_candidate is private opt-in only")
    result.add_argument("--private-diagnostic-student-candidate", action="store_true",
                        help="required explicit opt-in for the private student_candidate goal-state pilot")
    result.add_argument("--goal-state-parent-event-id",
                        help="exact sealed parent event ID for the two-anchor student goal-state pilot")
    result.add_argument("--max-seconds", type=float, default=300.0)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--expected-phase-queue-seal-sha256")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    result = (resume_phase_queue(args.index, args.output, args=args) if args.resume
              else build_phase_queue(args.index, args.output, args=args))
    print(base.canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
