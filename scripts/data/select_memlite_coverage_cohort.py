"""Select a bounded diagnostic coverage cohort from the sealed P107 index.

This producer is deliberately separate from the fixed 40-row phase sampler.
It streams the sealed event JSONL, validates every row against the canonical
protocol, and retains only deterministic min-hash winners for requested task
cells.  The selector therefore does not build the generic selector's 50,000
candidate pool and cannot silently turn a missing task/category into a
different role or an inferred outcome.

The two supported roles are intentionally independent:

* ``train`` selects only immutable ``annotation_calibration``/``train``
  groups.  It may be handed to the existing calibration renderer after the
  usual queue-seal handoff.
* ``eval`` selects only immutable ``evaluation_only``/``eval`` groups.  Its
  rows are diagnostic-only and are never emitted as a training queue.  The
  current renderer's calibration-only guard rejects these requests; callers
  must use a separately reviewed evaluation adapter.

The selection input is metadata only.  No RGB is decoded and no success,
failure, recovery, action, or label is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_memlite_event_annotation_queue as base  # noqa: E402


COVERAGE_SCHEMA = "p107-diagnostic-coverage-selection-v1"
EXCLUSION_SCHEMA = "p107-source-window-exclusion-v1"
STATUS = "CANDIDATE_MISSING_EVIDENCE"
MAX_TOTAL_PER_ROLE = 80
DEFAULT_SEED = "p107-diagnostic-coverage-20261002"
DEFAULT_ACTOR_HISTORY = 60
DEFAULT_REVIEW_BEFORE = 60
DEFAULT_REVIEW_AFTER = 60
DEFAULT_REVIEW_STRIDE = 15
ROLE_CONFIG = {
    "train": {"usage_role": "annotation_calibration", "immutable_split": "train"},
    "eval": {"usage_role": "evaluation_only", "immutable_split": "eval"},
}
STRUCTURAL_NAMES = frozenset({
    "ATOMIC", "COMPOUND", "PARALLEL", "NON_PARALLEL", "BIMANUAL", "NON_BIMANUAL",
    "BOUNDARY", "INTERIOR",
})
SOURCE_WINDOW_FIELDS = (
    "source_release_manifest_sha256", "source_group_id", "raw_episode_id", "observation_frame",
)


SourceWindow = tuple[str, str, int, int]


def _sha256(value: Any, name: str) -> str:
    if not base.is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return str(value)


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    return base.require_int(value, name, minimum=minimum)


def _regular_file(path: Path, name: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing regular file")
    return path


def _source_window_key(event: Mapping[str, Any], *, expected_release: str | None = None) -> SourceWindow:
    event_id = event.get("event_id", "event")
    source = base.require_mapping(event.get("source"), f"{event_id}.source")
    release = _sha256(source.get("source_release_manifest_sha256"), f"{event_id}.source.release")
    if expected_release is not None and release != expected_release:
        raise ValueError("event source release differs from the pinned release")
    group = _sha256(source.get("source_group_id"), f"{event_id}.source.source_group_id")
    raw_episode = _require_int(source.get("raw_episode_id"), f"{event_id}.source.raw_episode_id")
    observation = base.require_mapping(event.get("observation"), f"{event_id}.observation")
    frame = _require_int(observation.get("frame"), f"{event_id}.observation.frame")
    # episode_index remains independently validated, but is deliberately not
    # part of the exclusion key: the sealed source-window contract is release,
    # source-group, raw-episode, and observation frame.
    _require_int(source.get("episode_index"), f"{event_id}.source.episode_index")
    return release, group, raw_episode, frame


def source_window_dict(key: SourceWindow) -> dict[str, Any]:
    return dict(zip(SOURCE_WINDOW_FIELDS, key))


@dataclass(frozen=True)
class PriorWindows:
    keys: frozenset[SourceWindow]
    groups: frozenset[str]
    sha256: str | None
    rows: int


def _sidecar_key(row: Mapping[str, Any], *, expected_release: str) -> SourceWindow:
    """Read a direct key or a source-window object, never an event ID."""
    if "source_window" in row:
        value = row["source_window"]
        if not isinstance(value, Mapping):
            raise ValueError("source_window exclusion row must be an object")
        source = value
        frame = source.get("observation_frame")
    elif "source_identity" in row:
        value = row["source_identity"]
        if not isinstance(value, Mapping):
            raise ValueError("source_identity exclusion row must be an object")
        source = value
        frame = row.get("observation_frame")
    else:
        source = row
        frame = row.get("observation_frame")
    release = _sha256(source.get("source_release_manifest_sha256"), "prior release")
    group = _sha256(source.get("source_group_id"), "prior source_group_id")
    raw_episode = _require_int(source.get("raw_episode_id"), "prior raw_episode_id")
    frame = _require_int(frame, "prior observation_frame")
    if release != expected_release:
        raise ValueError("prior source-window row has a different release hash")
    if "event_id" in row:
        _sha256(row["event_id"], "prior event_id")
    if "source_annotation_sha256" in source:
        _sha256(source["source_annotation_sha256"], "prior source_annotation_sha256")
    return release, group, raw_episode, frame


def load_prior_windows(path: Path | None, *, expected_sha256: str | None,
                       expected_release: str) -> PriorWindows:
    """Load a byte-pinned exact-window sidecar.

    Event IDs are accepted only as audit metadata.  They never participate in
    exclusion, which protects against old/new derived IDs for the same source
    observation.
    """
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("--expected-prior-source-windows-sha256 requires --prior-source-windows")
        return PriorWindows(frozenset(), frozenset(), None, 0)
    path = _regular_file(path, "--prior-source-windows")
    if expected_sha256 is None:
        raise ValueError("--prior-source-windows requires --expected-prior-source-windows-sha256")
    expected_sha256 = _sha256(expected_sha256, "--expected-prior-source-windows-sha256")
    actual = base.sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("prior source-window bytes do not match the externally pinned SHA-256")
    rows = base.read_jsonl(path)
    keys: set[SourceWindow] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("prior source-window sidecar rows must be objects")
        if row.get("schema_version") != EXCLUSION_SCHEMA:
            raise ValueError("prior source-window sidecar row has an invalid schema_version")
        key = _sidecar_key(row, expected_release=expected_release)
        if key in keys:
            raise ValueError("prior source-window sidecar contains a duplicate exact key")
        keys.add(key)
    return PriorWindows(frozenset(keys), frozenset(key[1] for key in keys), actual, len(rows))


@dataclass(frozen=True)
class RoleSpec:
    name: str
    required_tasks: frozenset[int]
    total_target: int
    structural_quotas: Mapping[str, int]
    seed: str
    isolate_prior_source_groups: bool = True
    max_per_source_group: int = 1
    max_per_episode: int = 1

    @property
    def usage_role(self) -> str:
        return ROLE_CONFIG[self.name]["usage_role"]

    @property
    def immutable_split(self) -> str:
        return ROLE_CONFIG[self.name]["immutable_split"]

    def validate(self) -> None:
        if self.name not in ROLE_CONFIG:
            raise ValueError(f"unknown selection role: {self.name}")
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError("selection seed must be a nonempty string")
        if any(type(task) is not int or task < 0 for task in self.required_tasks):
            raise ValueError("required task IDs must be nonnegative integers")
        if self.total_target < len(self.required_tasks) or self.total_target > MAX_TOTAL_PER_ROLE:
            raise ValueError("total target must cover required tasks and be <= 80 per role")
        if self.max_per_source_group < 1 or self.max_per_episode < 1:
            raise ValueError("source-group and episode caps must be positive")
        for name, quota in self.structural_quotas.items():
            if name not in STRUCTURAL_NAMES:
                raise ValueError(f"unknown metadata structural stratum: {name}")
            if type(quota) is not int or quota < 0:
                raise ValueError("structural quotas must be nonnegative integers")


@dataclass(frozen=True)
class CandidateRecord:
    event: Mapping[str, Any]
    event_id: str
    usage_role: str
    immutable_split: str
    task_id: int
    source_group_id: str
    raw_episode_id: int
    episode_index: int
    episode_length: int
    observation_frame: int
    interval_start: int
    interval_end: int
    skill_ids: tuple[int, ...]
    structural_strata: tuple[str, ...]
    source_window: SourceWindow

    @property
    def episode_key(self) -> tuple[str, int, int]:
        return self.source_group_id, self.raw_episode_id, self.episode_index


@dataclass
class RoleState:
    spec: RoleSpec
    required_winners: dict[int, CandidateRecord] = field(default_factory=dict)
    structural_winners: dict[tuple[str, int], CandidateRecord] = field(default_factory=dict)
    available_tasks: set[int] = field(default_factory=set)
    eligible_events: int = 0
    excluded_prior_exact: int = 0
    excluded_prior_group: int = 0
    excluded_unknown_task: int = 0
    excluded_unknown_skill: int = 0
    excluded_wrong_role: int = 0
    refused_student_candidates: int = 0
    malformed_or_ineligible: int = 0


@dataclass(frozen=True)
class RoleSelection:
    spec: RoleSpec
    selected: tuple[CandidateRecord, ...]
    phases: Mapping[str, str]
    report: Mapping[str, Any]


def _structural_strata(event: Mapping[str, Any], *, start: int, end: int,
                       episode_length: int) -> tuple[str, ...]:
    skills = event.get("skill_bundle")
    if not isinstance(skills, list) or not skills:
        raise ValueError("event skill_bundle is missing for structural classification")
    parallel = event.get("parallel_bundle")
    if type(parallel) is not bool:
        raise ValueError("event parallel_bundle must be a boolean")
    names = ["ATOMIC" if len(skills) == 1 else "COMPOUND"]
    names.append("PARALLEL" if parallel else "NON_PARALLEL")
    # Only the explicit parallel_bundle marker is accepted as bimanual
    # evidence.  Arm text such as UNSPECIFIED is never upgraded to bimanual.
    names.append("BIMANUAL" if parallel else "NON_BIMANUAL")
    names.append("BOUNDARY" if start == 0 or end == episode_length else "INTERIOR")
    return tuple(names)


def _candidate_from_event(event: Mapping[str, Any], *, expected_release: str,
                          groups: Mapping[str, base.SourceGroup], official_tasks: set[int],
                          official_skills: set[int]) -> tuple[CandidateRecord | None, str | None]:
    """Validate one event and return (candidate, exclusion_reason)."""
    event_id = event.get("event_id")
    if not base.is_sha256(event_id):
        raise ValueError("event index contains an invalid event_id")
    source = base.require_mapping(event.get("source"), f"{event_id}.source")
    key = _source_window_key(event, expected_release=expected_release)
    group_id = key[1]
    group = groups.get(group_id)
    if group is None:
        raise ValueError("event references an unknown source group")
    if (event.get("usage_role") != group.usage_role or source.get("original_split") != group.original_split or
            source.get("task_index") != group.task_id or source.get("task_instance_id") != group.task_instance_id or
            source.get("source_release_manifest_sha256") != group.source_release_manifest_sha256):
        raise ValueError("event/source immutable role or group identity drifted")
    source_annotation = source.get("source_annotation_sha256")
    _sha256(source_annotation, f"{event_id}.source.source_annotation_sha256")
    episode_index = _require_int(source.get("episode_index"), f"{event_id}.source.episode_index")
    if episode_index not in group.source_episode_ids:
        raise ValueError("event episode index is absent from source-group inventory")
    if group.task_id not in official_tasks:
        return None, "unknown_task"
    evidence = event.get("evidence")
    if evidence != {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None}:
        raise ValueError("coverage selector accepts metadata candidates with missing evidence only")
    skills = event.get("skill_bundle")
    if not isinstance(skills, list) or not skills:
        raise ValueError("event candidate has no source skill bundle")
    skill_ids: list[int] = []
    for skill in skills:
        if not isinstance(skill, Mapping) or type(skill.get("skill_id")) is not int or skill["skill_id"] < 0:
            raise ValueError("event skill has an invalid skill_id")
        skill_ids.append(skill["skill_id"])
    if set(skill_ids) - official_skills:
        return None, "unknown_skill"
    observation = base.require_mapping(event.get("observation"), f"{event_id}.observation")
    interval = base.require_mapping(event.get("event_interval"), f"{event_id}.event_interval")
    frame = _require_int(observation.get("frame"), f"{event_id}.observation.frame")
    start = _require_int(interval.get("start_frame"), f"{event_id}.event_interval.start_frame")
    end = _require_int(interval.get("end_frame"), f"{event_id}.event_interval.end_frame")
    length = _require_int(source.get("episode_length"), f"{event_id}.source.episode_length", minimum=1)
    if not start <= frame < end <= length:
        raise ValueError("event observation/interval is outside the source episode")
    timestamp = observation.get("timestamp_s")
    if type(timestamp) not in (int, float) or isinstance(timestamp, bool) or not math.isfinite(float(timestamp)):
        raise ValueError("event observation timestamp is invalid")
    if abs(float(timestamp) - frame / base.FRAME_RATE_HZ) > 1e-9:
        raise ValueError("event observation timestamp drifted from source frame")
    return CandidateRecord(
        event=event, event_id=str(event_id), usage_role=str(event["usage_role"]),
        immutable_split=str(source["original_split"]), task_id=group.task_id, source_group_id=group_id,
        raw_episode_id=key[2], episode_index=episode_index,
        episode_length=length, observation_frame=frame, interval_start=start, interval_end=end,
        skill_ids=tuple(sorted(set(skill_ids))),
        structural_strata=_structural_strata(event, start=start, end=end, episode_length=length),
        source_window=key,
    ), None


def _rank(candidate: CandidateRecord, spec: RoleSpec) -> str:
    return base.canonical_sha256({
        "schema_version": COVERAGE_SCHEMA,
        "seed": spec.seed,
        "selection_role": spec.name,
        "task_id": candidate.task_id,
        "source_window": source_window_dict(candidate.source_window),
        "event_id": candidate.event_id,
    })


def _better(new: CandidateRecord, old: CandidateRecord, spec: RoleSpec) -> bool:
    return (_rank(new, spec), new.event_id) < (_rank(old, spec), old.event_id)


def _consider(state: RoleState, candidate: CandidateRecord, prior: PriorWindows) -> None:
    spec = state.spec
    if candidate.source_window in prior.keys:
        state.excluded_prior_exact += 1
        return
    if spec.isolate_prior_source_groups and candidate.source_group_id in prior.groups:
        state.excluded_prior_group += 1
        return
    state.eligible_events += 1
    state.available_tasks.add(candidate.task_id)
    if candidate.task_id in spec.required_tasks:
        old = state.required_winners.get(candidate.task_id)
        if old is None or _better(candidate, old, spec):
            state.required_winners[candidate.task_id] = candidate
    for name in spec.structural_quotas:
        if name not in candidate.structural_strata:
            continue
        cell = name, candidate.task_id
        old = state.structural_winners.get(cell)
        if old is None or _better(candidate, old, spec):
            state.structural_winners[cell] = candidate


def _can_add(candidate: CandidateRecord, *, groups: Counter[str], episodes: Counter[tuple[str, int, int]],
             windows: set[SourceWindow], spec: RoleSpec) -> bool:
    return (candidate.source_window not in windows and
            groups[candidate.source_group_id] < spec.max_per_source_group and
            episodes[candidate.episode_key] < spec.max_per_episode)


def _select_state(state: RoleState) -> RoleSelection:
    spec = state.spec
    selected: list[CandidateRecord] = []
    phases: dict[str, str] = {}
    groups: Counter[str] = Counter()
    episodes: Counter[tuple[str, int, int]] = Counter()
    windows: set[SourceWindow] = set()
    missing_due_collision: list[int] = []

    for task in sorted(spec.required_tasks):
        candidate = state.required_winners.get(task)
        if candidate is None:
            continue
        if not _can_add(candidate, groups=groups, episodes=episodes, windows=windows, spec=spec):
            missing_due_collision.append(task)
            continue
        selected.append(candidate)
        phases[candidate.event_id] = "REQUIRED_TASK_CELL"
        groups[candidate.source_group_id] += 1
        episodes[candidate.episode_key] += 1
        windows.add(candidate.source_window)

    structural_selected: Counter[str] = Counter(
        name for candidate in selected for name in candidate.structural_strata if name in spec.structural_quotas
    )
    structural_candidates: dict[str, list[CandidateRecord]] = defaultdict(list)
    for (name, _task), candidate in state.structural_winners.items():
        structural_candidates[name].append(candidate)
    for name in sorted(spec.structural_quotas):
        structural_candidates[name].sort(key=lambda candidate: (_rank(candidate, spec), candidate.event_id))
        for candidate in structural_candidates[name]:
            if structural_selected[name] >= spec.structural_quotas[name] or len(selected) >= spec.total_target:
                break
            if not _can_add(candidate, groups=groups, episodes=episodes, windows=windows, spec=spec):
                continue
            selected.append(candidate)
            phases[candidate.event_id] = "STRUCTURAL_SUPPLEMENT"
            structural_selected.update(name for name in candidate.structural_strata if name in spec.structural_quotas)
            groups[candidate.source_group_id] += 1
            episodes[candidate.episode_key] += 1
            windows.add(candidate.source_window)

    selected.sort(key=lambda candidate: (candidate.task_id, _rank(candidate, spec), candidate.event_id))
    structural_available = {
        name: sum(1 for candidate in state.structural_winners.values() if name in candidate.structural_strata)
        for name in sorted(spec.structural_quotas)
    }
    structural_selected_counts = {
        name: sum(name in candidate.structural_strata for candidate in selected)
        for name in sorted(spec.structural_quotas)
    }
    missing_structural = {
        name: max(0, spec.structural_quotas[name] - structural_selected_counts.get(name, 0))
        for name in sorted(spec.structural_quotas)
        if structural_selected_counts.get(name, 0) < spec.structural_quotas[name]
    }
    selected_required_tasks = sorted(
        candidate.task_id for candidate in selected
        if phases[candidate.event_id] == "REQUIRED_TASK_CELL" and candidate.task_id in spec.required_tasks
    )
    report = {
        "schema_version": COVERAGE_SCHEMA,
        "status": STATUS,
        "selection_role": spec.name,
        "usage_role": spec.usage_role,
        "immutable_split": spec.immutable_split,
        "training_eligible": False,
        "requested_required_task_ids": sorted(spec.required_tasks),
        "selected_required_task_ids": selected_required_tasks,
        "missing_required_task_ids": sorted(set(spec.required_tasks) - set(selected_required_tasks)),
        "missing_required_task_ids_due_to_source_caps": missing_due_collision,
        "available_task_ids_after_prior_exclusion": sorted(state.available_tasks),
        "eligible_events_after_prior_exclusion": state.eligible_events,
        "eligible_candidate_events_after_prior_exclusion": state.eligible_events,
        "eligible_distinct_source_windows_after_prior_exclusion": None,
        "eligible_window_cardinality": "NOT_MATERIALIZED_BY_BOUNDED_SELECTOR",
        "selected_rows": len(selected),
        "selected_distinct_source_windows": len(windows),
        "selected_distinct_source_groups": len(groups),
        "selected_distinct_source_episodes": len(episodes),
        "excluded_prior_exact_source_windows": state.excluded_prior_exact,
        "excluded_prior_source_groups": state.excluded_prior_group,
        "excluded_wrong_role_or_split_events": state.excluded_wrong_role,
        "refused_student_candidate_events": state.refused_student_candidates,
        "structural_quota_requested": dict(sorted(spec.structural_quotas.items())),
        "structural_quota_available_cells": dict(sorted(structural_available.items())),
        "structural_quota_selected": structural_selected_counts,
        "missing_structural_quota": missing_structural,
        "caps": {"max_per_source_group": spec.max_per_source_group, "max_per_episode": spec.max_per_episode},
        "prior_source_group_isolation": spec.isolate_prior_source_groups,
        "selection_phases": dict(sorted(Counter(phases.values()).items())),
        "no_outcome_or_action_labels": True,
    }
    return RoleSelection(spec=spec, selected=tuple(selected), phases=phases, report=report)


def select_records(records: Iterable[CandidateRecord], *, specs: Sequence[RoleSpec],
                   prior: PriorWindows) -> dict[str, RoleSelection]:
    """Select from already validated records; useful for unit tests and adapters."""
    if not specs:
        raise ValueError("at least one role selection spec is required")
    states = {spec.name: RoleState(spec) for spec in specs}
    if len(states) != len(specs):
        raise ValueError("selection role names must be unique")
    for spec in specs:
        spec.validate()
    for candidate in records:
        for state in states.values():
            if candidate.usage_role != state.spec.usage_role:
                if candidate.usage_role == "student_candidate" and state.spec.name == "train":
                    state.refused_student_candidates += 1
                else:
                    state.excluded_wrong_role += 1
                continue
            if candidate.immutable_split != state.spec.immutable_split:
                state.excluded_wrong_role += 1
                continue
            _consider(state, candidate, prior)
    return {name: _select_state(state) for name, state in states.items()}


def _iter_valid_records(index: Path, *, expected_release: str, expected_inventory_seal: str,
                        coverage_path: Path, expected_coverage_sha256: str, protocol_path: Path,
                        expected_protocol_sha256: str, specs: Sequence[RoleSpec],
                        prior: PriorWindows) -> tuple[dict[str, RoleSelection], dict[str, Any]]:
    protocol = base.load_canonical_protocol(protocol_path, expected_sha256=expected_protocol_sha256)
    manifest, events_path, groups = base.validate_index(
        index, expected_source_manifest_sha256=expected_release,
        canonical_protocol=protocol, expected_protocol_sha256=expected_protocol_sha256)
    inventory, inventory_sha256 = base.validate_inventory_seal(
        index, expected_inventory_seal_sha256=expected_inventory_seal)
    coverage = base.read_coverage_expectations(
        coverage_path, expected_sha256=inventory["coverage_expectations_sha256"])
    coverage_sha256 = base.canonical_sha256(coverage)
    if coverage_sha256 != expected_coverage_sha256 or manifest.get("coverage_expectations_sha256") != coverage_sha256:
        raise ValueError("coverage expectation bytes do not match both CLI and sealed index pins")
    for spec in specs:
        spec.validate()
    states = {spec.name: RoleState(spec) for spec in specs}
    if len(states) != len(specs):
        raise ValueError("selection role names must be unique")
    official_tasks = set(coverage["expected_task_ids"])
    official_skills = {item["skill_id"] for item in coverage["expected_skill_vocabulary"]}
    seen_event_ids: set[str] = set()
    role_counts: Counter[str] = Counter()
    unknown_tasks: Counter[str] = Counter()
    unknown_skills: Counter[str] = Counter()
    rows = 0
    for event in base.iter_jsonl_verified(events_path, expected=manifest["files"]["event_candidates.jsonl"]):
        rows += 1
        protocol.validate_event(event)
        event_id = event.get("event_id")
        if not base.is_sha256(event_id) or event_id in seen_event_ids:
            raise ValueError("sealed event index has an absent or duplicate event_id")
        seen_event_ids.add(str(event_id))
        role_counts[str(event.get("usage_role"))] += 1
        candidate, reason = _candidate_from_event(
            event, expected_release=expected_release, groups=groups,
            official_tasks=official_tasks, official_skills=official_skills,
        )
        if reason == "unknown_task":
            unknown_tasks[str(event["source"]["task_index"])] += 1
            continue
        if reason == "unknown_skill":
            unknown_skills[str(event_id)] += 1
            continue
        assert candidate is not None
        for state in states.values():
            if candidate.usage_role != state.spec.usage_role:
                if candidate.usage_role == "student_candidate" and state.spec.name == "train":
                    state.refused_student_candidates += 1
                else:
                    state.excluded_wrong_role += 1
                continue
            if candidate.immutable_split != state.spec.immutable_split:
                state.excluded_wrong_role += 1
                continue
            _consider(state, candidate, prior)
    expected_rows = manifest["files"]["event_candidates.jsonl"].get("rows")
    if type(expected_rows) is not int or expected_rows < 0:
        raise ValueError("sealed event receipt has an invalid row count")
    if rows != expected_rows:
        raise ValueError("sealed event row count mismatch")
    selections = {name: _select_state(state) for name, state in states.items()}
    # Keep these diagnostics outside the role reports so train and eval are
    # independently auditable while sharing one sealed source pass.
    source_report = {
        "index_manifest_path": str(index / "manifest.json"),
        "index_manifest_sha256": base.sha256_file(index / "manifest.json"),
        "inventory_seal_sha256": inventory_sha256,
        "source_release_manifest_sha256": expected_release,
        "coverage_expectations_sha256": coverage_sha256,
        "event_rows_validated": rows,
        "source_role_rows": dict(sorted(role_counts.items())),
        "unknown_task_rows": dict(sorted(unknown_tasks.items())),
        "unknown_skill_event_rows": len(unknown_skills),
        "prior_source_windows_sha256": prior.sha256,
        "prior_source_window_rows": prior.rows,
    }
    return selections, source_report


def _as_base_candidate(candidate: CandidateRecord, *, selection_phase: str) -> base.Candidate:
    skills = candidate.event["skill_bundle"]
    skill_keys = tuple(base.skill_key(skill) for skill in skills)
    verbs = tuple(sorted({skill["verb"] for skill in skills if isinstance(skill.get("verb"), str) and skill["verb"]}))
    boundary = "BOUNDARY" in candidate.structural_strata
    signals = ["COVERAGE_TASK_CELL" if selection_phase == "REQUIRED_TASK_CELL" else "COVERAGE_STRUCTURAL_SUPPLEMENT"]
    signals.extend(candidate.structural_strata)
    return base.Candidate(
        event=candidate.event, event_id=candidate.event_id, source_group_id=candidate.source_group_id,
        task_id=candidate.task_id, task_instance_id=candidate.event["source"]["task_instance_id"],
        episode_key=candidate.episode_key, episode_index=candidate.episode_index,
        episode_length=candidate.episode_length, anchor_frame=candidate.observation_frame,
        interval_start=candidate.interval_start, interval_end=candidate.interval_end,
        skill_ids=candidate.skill_ids, raw_source_verbs=verbs, skill_keys=skill_keys,
        repeat_attempt_count=0, boundary_before=boundary, boundary_after=boundary,
        long_interval=(candidate.interval_end - candidate.interval_start >= 360),
        candidate_signals=tuple(signals), selection_stratum=selection_phase,
    )


def selection_row(candidate: CandidateRecord, *, selection_order: int, selection_phase: str) -> dict[str, Any]:
    source = candidate.event["source"]
    return {
        "schema_version": COVERAGE_SCHEMA,
        "selection_order": selection_order,
        "status": STATUS,
        "training_eligible": False,
        "candidate_heuristics_are_not_truth": True,
        "usage_role": candidate.usage_role,
        "immutable_split": candidate.immutable_split,
        "event_id": candidate.event_id,
        "source_identity": {
            "source_release_manifest_sha256": source["source_release_manifest_sha256"],
            "source_annotation_sha256": source["source_annotation_sha256"],
            "source_group_id": candidate.source_group_id,
            "task_index": candidate.task_id,
            "task_instance_id": source["task_instance_id"],
            "raw_episode_id": candidate.raw_episode_id,
            "episode_index": candidate.episode_index,
        },
        "source_window": source_window_dict(candidate.source_window),
        "observation_frame": candidate.observation_frame,
        "event_interval": {"start_frame": candidate.interval_start, "end_frame": candidate.interval_end},
        "skill_ids": list(candidate.skill_ids),
        "structural_strata": list(candidate.structural_strata),
        "selection_phase": selection_phase,
        "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "outcome_supervision": False,
        "corrective_action_supervision": False,
    }


def _role_policy(spec: RoleSpec, *, index_manifest_sha256: str, inventory_sha256: str,
                 coverage_sha256: str, protocol_sha256: str, source_release_sha256: str,
                 prior: PriorWindows) -> dict[str, Any]:
    core = {
        "schema_version": COVERAGE_SCHEMA,
        "algorithm": "p107-diagnostic-coverage-minhash-v1",
        "seed": spec.seed,
        "selection_role": spec.name,
        "usage_role": spec.usage_role,
        "immutable_split": spec.immutable_split,
        "required_task_ids": sorted(spec.required_tasks),
        "total_target": spec.total_target,
        "structural_quotas": dict(sorted(spec.structural_quotas.items())),
        "max_per_source_group": spec.max_per_source_group,
        "max_per_episode": spec.max_per_episode,
        "isolate_prior_source_groups": spec.isolate_prior_source_groups,
        "source_release_manifest_sha256": source_release_sha256,
        "index_manifest_sha256": index_manifest_sha256,
        "index_inventory_seal_sha256": inventory_sha256,
        "coverage_expectations_sha256": coverage_sha256,
        "canonical_protocol_sha256": protocol_sha256,
        "prior_source_windows_sha256": prior.sha256,
        "prior_source_window_rows": prior.rows,
        "streaming_bounded_selection": True,
        "retained_pool_cap": None,
        "rgb_decoded": False,
        "outcome_labels_emitted": False,
        "training_eligible": False,
    }
    return {**core, "policy_sha256": base.canonical_sha256(core)}


def _make_jobs(selection: RoleSelection, *, policy_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for order, candidate in enumerate(selection.selected):
        phase = selection.phases[candidate.event_id]
        base_candidate = _as_base_candidate(candidate, selection_phase=phase)
        windows = base.window_spec(
            base_candidate, actor_history_frames=DEFAULT_ACTOR_HISTORY,
            review_before_frames=DEFAULT_REVIEW_BEFORE, review_after_frames=DEFAULT_REVIEW_AFTER,
            sample_stride_frames=DEFAULT_REVIEW_STRIDE,
        )
        queue_kind = "ANNOTATION_CALIBRATION_COVERAGE" if selection.spec.name == "train" else "EVALUATION_ONLY_DIAGNOSTIC"
        job, request = base.queue_job(
            base_candidate, queue_kind=queue_kind, selection_phase=phase,
            policy_sha256=policy_sha256, windows=windows,
        )
        if selection.spec.name == "eval":
            # Keep the existing job field set, but make the immutable eval
            # split explicit.  The current renderer intentionally rejects this
            # role; this row is for a separately reviewed diagnostic adapter.
            job = {**job, "immutable_split": "eval"}
        jobs.append(job)
        requests.append(request)
    return jobs, requests


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("xb") as stream:
        for row in rows:
            stream.write(base.canonical_json(row).encode("utf-8") + b"\n")
            count += 1
    return {"sha256": base.sha256_file(path), "rows": count, "bytes": path.stat().st_size}


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.write_bytes(base.canonical_json(value).encode("utf-8") + b"\n")
    return {"sha256": base.sha256_file(path), "rows": 1, "bytes": path.stat().st_size}


def _queue_counts(selection: RoleSelection, *, coverage_expectations: Mapping[str, Any],
                  jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the existing counts field set without claiming a full pool scan."""
    empty = {
        "budget": 0, "pool_events": 0, "queued_events": 0, "per_task": {}, "per_skill_id": {},
        "per_raw_source_verb_diagnostic": {}, "per_episode": {}, "per_selection_stratum": {},
        "per_selection_phase": {}, "near_duplicate": {}, "coverage": {
            "expected_task_ids": list(coverage_expectations["expected_task_ids"]),
            "expected_task_count": len(coverage_expectations["expected_task_ids"]),
            "eligible_pool_coverage_status": "NOT_EVALUATED_ZERO_OUTPUT_BUDGET",
            "tasks_present_in_eligible_pool": None, "tasks_queued": [],
            "tasks_present_but_not_queued": None, "missing_expected_task_ids_from_pool": None,
            "unexpected_task_ids_in_pool": None,
            "expected_skill_vocabulary": coverage_expectations["expected_skill_vocabulary"],
            "expected_skill_ids": [item["skill_id"] for item in coverage_expectations["expected_skill_vocabulary"]],
            "expected_skill_count": len(coverage_expectations["expected_skill_vocabulary"]),
            "skill_ids_present_in_eligible_pool": None, "skill_ids_queued": [],
            "skill_ids_present_but_not_queued": None, "skill_count_seen": None,
            "missing_expected_skill_ids_from_pool": None, "unexpected_source_skill_ids_in_pool": None,
            "global_vocabulary_complete": None, "raw_source_verbs_diagnostic": None,
            "required_task_skill_pairs": coverage_expectations["required_task_skill_pairs"],
            "required_task_skill_pair_queue_coverage": None,
            "required_task_skill_pair_queue_status": "NOT_EVALUATED_ZERO_OUTPUT_BUDGET",
        },
    }
    if selection.spec.name != "train":
        return {
            "schema_version": base.COUNTS_SCHEMA, "status": STATUS, "training_eligible": False,
            "source_role_inventory": {selection.spec.usage_role: len(jobs)},
            "events_excluded_by_role": {}, "events_excluded_by_unrecognized_official_vocabulary": {},
            "student_candidate_queue": empty, "annotation_calibration_queue": empty,
            "existing_local_single_frame_pilot": {"count": 0, "included_in_this_queue": False},
        }
    candidates = [_as_base_candidate(item, selection_phase=selection.phases[item.event_id])
                  for item in selection.selected]
    selected_pairs = [(candidate, selection.phases[candidate.event_id])
                      for candidate in candidates]
    calibration = base.count_queue(
        candidates, selected_pairs, budget=len(candidates), near_duplicates={
            "streaming_pool_not_retained": True,
            "selection_report_candidate_events": selection.report["eligible_candidate_events_after_prior_exclusion"],
        }, coverage_expectations=coverage_expectations, eligible_pool_retained=False,
    )
    return {
        "schema_version": base.COUNTS_SCHEMA, "status": STATUS, "training_eligible": False,
        "source_role_inventory": {"annotation_calibration": len(candidates)},
        "events_excluded_by_role": {}, "events_excluded_by_unrecognized_official_vocabulary": {},
        "student_candidate_queue": empty, "annotation_calibration_queue": calibration,
        "existing_local_single_frame_pilot": {"count": 0, "included_in_this_queue": False},
    }


def write_role_output(output: Path, selection: RoleSelection, *, source_report: Mapping[str, Any],
                      index_manifest: Mapping[str, Any], inventory_sha256: str,
                      coverage_sha256: str, coverage_expectations: Mapping[str, Any],
                      protocol_sha256: str, prior: PriorWindows) -> dict[str, Any]:
    """Write a role-specific candidate handoff; never overwrite an output."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"coverage output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    index_path = source_report.get("index_manifest_path")
    if isinstance(index_path, str) and Path(index_path).parent.exists():
        base.forbid_output_inside(output, Path(index_path).parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        policy = _role_policy(
            selection.spec, index_manifest_sha256=source_report["index_manifest_sha256"],
            inventory_sha256=inventory_sha256, coverage_sha256=coverage_sha256,
            protocol_sha256=protocol_sha256,
            source_release_sha256=index_manifest["source_release_manifest_sha256"], prior=prior,
        )
        rows = [selection_row(candidate, selection_order=order,
                              selection_phase=selection.phases[candidate.event_id])
                for order, candidate in enumerate(selection.selected)]
        row_receipt = _write_jsonl(staging / "selected_rows.jsonl", rows)
        report = {**dict(source_report), **dict(selection.report), "policy_sha256": policy["policy_sha256"]}
        report_receipt = _write_json(staging / "coverage_report.json", report)
        jobs, requests = _make_jobs(selection, policy_sha256=policy["policy_sha256"])
        request_name = "camera_native_render_requests.jsonl" if selection.spec.name == "train" else "evaluation_only_render_requests.jsonl"
        job_name = "annotation_calibration_queue.jsonl" if selection.spec.name == "train" else "evaluation_only_selection.jsonl"
        job_receipt = _write_jsonl(staging / job_name, jobs)
        request_receipt = _write_jsonl(staging / request_name, requests)
        extra_files: dict[str, dict[str, Any]] = {}
        if selection.spec.name == "train":
            student_receipt = _write_jsonl(staging / "student_candidate_queue.jsonl", [])
            counts = _queue_counts(selection, coverage_expectations=coverage_expectations, jobs=jobs)
            counts_path = staging / "counts.json"
            counts_path.write_bytes(base.canonical_json(counts).encode("utf-8") + b"\n")
            counts_receipt = {"sha256": base.sha256_file(counts_path), "rows": 1, "bytes": counts_path.stat().st_size}
            extra_files = {
                "student_candidate_queue.jsonl": student_receipt,
                "counts.json": counts_receipt,
            }
        manifest = {
            "schema_version": COVERAGE_SCHEMA,
            "status": "DIAGNOSTIC_CANDIDATES_READY_FOR_INDEPENDENT_REVIEW",
            "training_eligible": False,
            "selection_role": selection.spec.name,
            "usage_role": selection.spec.usage_role,
            "immutable_split": selection.spec.immutable_split,
            "source_release_manifest_sha256": index_manifest["source_release_manifest_sha256"],
            "index_manifest_sha256": source_report["index_manifest_sha256"],
            "inventory_seal_sha256": inventory_sha256,
            "coverage_expectations_sha256": coverage_sha256,
            "canonical_protocol_sha256": protocol_sha256,
            "prior_source_windows_sha256": prior.sha256,
            "policy": policy,
            "files": {
                "selected_rows.jsonl": row_receipt,
                "coverage_report.json": report_receipt,
                job_name: job_receipt,
                request_name: request_receipt,
                **extra_files,
            },
            "renderer_compatibility": (
                "EXISTING_CALIBRATION_RENDERER_COMPATIBLE" if selection.spec.name == "train"
                else "EXISTING_RENDERER_REJECTS_EVALUATION_ONLY_ROLE_USE_SEPARATE_DIAGNOSTIC_ADAPTER"
            ),
            "no_outcome_or_action_labels": True,
        }
        manifest_receipt = _write_json(staging / "manifest.json", manifest)
        seal = {
            "schema_version": COVERAGE_SCHEMA,
            "manifest_sha256": manifest_receipt["sha256"],
            "policy_sha256": policy["policy_sha256"],
            "training_eligible": False,
            "files": manifest["files"],
        }
        if selection.spec.name == "train":
            queue_payloads = {
                "annotation_calibration_queue.jsonl": job_receipt,
                "camera_native_render_requests.jsonl": request_receipt,
                "counts.json": extra_files["counts.json"],
                "student_candidate_queue.jsonl": extra_files["student_candidate_queue.jsonl"],
            }
            queue_seal = {
                "schema_version": base.QUEUE_SEAL_SCHEMA,
                "queue_manifest_sha256": manifest_receipt["sha256"],
                "inventory_seal_sha256": inventory_sha256,
                "source_release_manifest_sha256": index_manifest["source_release_manifest_sha256"],
                "canonical_protocol_sha256": protocol_sha256,
                "coverage_expectations_sha256": coverage_sha256,
                "policy_sha256": policy["policy_sha256"],
                "expected_payload_files": sorted(base.QUEUE_PAYLOAD_FILES),
                "payload_files": queue_payloads,
            }
            queue_seal_receipt = _write_json(staging / "queue_seal.json", queue_seal)
            seal["queue_seal_sha256"] = queue_seal_receipt["sha256"]
        seal_receipt = _write_json(staging / "selection_seal.json", seal)
        _ = seal_receipt
        os.rename(staging, output)
        return {
            "status": "CREATED_DIAGNOSTIC_CANDIDATE_HANDOFF",
            "output": str(output),
            "manifest_sha256": manifest_receipt["sha256"],
            "selection_seal_sha256": base.sha256_file(output / "selection_seal.json"),
            "queue_seal_sha256": (base.sha256_file(output / "queue_seal.json")
                                  if (output / "queue_seal.json").is_file() else None),
            "selected": len(selection.selected),
            "renderer_compatibility": manifest["renderer_compatibility"],
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_cohort_outputs(train_output: Path, eval_output: Path, *, train_selection: RoleSelection,
                         eval_selection: RoleSelection, source_report: Mapping[str, Any],
                         index_manifest: Mapping[str, Any], inventory_sha256: str,
                         coverage_sha256: str, coverage_expectations: Mapping[str, Any],
                         protocol_sha256: str, prior: PriorWindows) -> dict[str, Any]:
    """Publish TRAIN and EVAL handoffs as one fresh, atomic cohort root.

    The two role directories must be siblings below a non-existing parent.
    Both are built under a hidden sibling staging root first, so a failure
    while building EVAL cannot leave a visible TRAIN directory behind.
    """
    train_output = Path(train_output).absolute()
    eval_output = Path(eval_output).absolute()
    if train_output == eval_output:
        raise ValueError("TRAIN and EVAL outputs must be distinct sibling directories")
    if train_output.parent != eval_output.parent:
        raise ValueError("TRAIN and EVAL outputs must be sibling directories under one fresh cohort root")
    cohort_root = train_output.parent
    if not cohort_root.parent.is_dir() or cohort_root.parent.is_symlink():
        raise ValueError("fresh cohort root parent must be an existing regular directory")
    if cohort_root.exists() or cohort_root.is_symlink() or train_output.exists() or eval_output.exists():
        raise FileExistsError(f"fresh cohort root or role output already exists: {cohort_root}")
    index_path = source_report.get("index_manifest_path")
    if isinstance(index_path, str) and Path(index_path).parent.exists():
        base.forbid_output_inside(cohort_root, Path(index_path).parent)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{cohort_root.name}.staging-", dir=str(cohort_root.parent)))
    try:
        staged_train = staging_root / train_output.name
        staged_eval = staging_root / eval_output.name
        train_result = write_role_output(
            staged_train, train_selection, source_report=source_report, index_manifest=index_manifest,
            inventory_sha256=inventory_sha256, coverage_sha256=coverage_sha256,
            coverage_expectations=coverage_expectations, protocol_sha256=protocol_sha256, prior=prior,
        )
        eval_result = write_role_output(
            staged_eval, eval_selection, source_report=source_report, index_manifest=index_manifest,
            inventory_sha256=inventory_sha256, coverage_sha256=coverage_sha256,
            coverage_expectations=coverage_expectations, protocol_sha256=protocol_sha256, prior=prior,
        )
        os.rename(staging_root, cohort_root)
        return {
            "status": "CREATED_DIAGNOSTIC_COHORT_HANDOFF",
            "cohort_root": str(cohort_root),
            "train_output": {**train_result, "output": str(train_output)},
            "eval_output": {**eval_result, "output": str(eval_output)},
        }
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _parse_task_ids(value: str) -> frozenset[int]:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("task IDs must be a nonempty comma-separated list")
    try:
        tasks = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("task IDs must be integers") from error
    if not tasks or any(task < 0 for task in tasks) or len(set(tasks)) != len(tasks):
        raise argparse.ArgumentTypeError("task IDs must be unique nonnegative integers")
    return frozenset(tasks)


def _parse_structural_quota(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("structural quota must use NAME=COUNT")
    name, count = value.split("=", 1)
    name = name.strip().upper()
    try:
        count_int = int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("structural quota count must be an integer") from error
    if name not in STRUCTURAL_NAMES or count_int < 0:
        raise argparse.ArgumentTypeError("structural quota has an unknown name or negative count")
    return name, count_int


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", type=Path, required=True)
    result.add_argument("--train-output", type=Path, required=True)
    result.add_argument("--eval-output", type=Path, required=True)
    result.add_argument("--train-task-ids", type=_parse_task_ids, required=True,
                        help="required annotation_calibration task cells")
    result.add_argument("--eval-task-ids", type=_parse_task_ids, required=True,
                        help="required evaluation_only task cells")
    result.add_argument("--prior-source-windows", type=Path, required=True,
                        help="byte-pinned exact source-window exclusion JSONL")
    result.add_argument("--expected-prior-source-windows-sha256", required=True)
    result.add_argument("--expected-source-release-manifest-sha256", required=True)
    result.add_argument("--expected-inventory-seal-sha256", required=True)
    result.add_argument("--coverage-expectations", type=Path, required=True)
    result.add_argument("--expected-coverage-expectations-sha256", required=True)
    result.add_argument("--protocol-path", type=Path, required=True)
    result.add_argument("--expected-protocol-sha256", required=True)
    result.add_argument("--seed", default=DEFAULT_SEED)
    result.add_argument("--train-total-target", type=int)
    result.add_argument("--eval-total-target", type=int)
    result.add_argument("--train-structural-quota", action="append", type=_parse_structural_quota, default=[])
    result.add_argument("--eval-structural-quota", action="append", type=_parse_structural_quota, default=[])
    result.add_argument("--allow-prior-source-group-reuse", action="store_true",
                        help="only exact source windows are excluded; off by default")
    result.add_argument("--write", action="store_true",
                        help="write the separately sealed diagnostic handoffs after the metadata report")
    return result


def _quota_map(values: Sequence[tuple[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, count in values:
        if name in result:
            raise ValueError(f"duplicate structural quota: {name}")
        result[name] = count
    return result


def main() -> None:
    args = parser().parse_args()
    expected_release = _sha256(args.expected_source_release_manifest_sha256, "source release")
    train_tasks, eval_tasks = args.train_task_ids, args.eval_task_ids
    train_total = len(train_tasks) if args.train_total_target is None else args.train_total_target
    eval_total = len(eval_tasks) if args.eval_total_target is None else args.eval_total_target
    specs = [
        RoleSpec("train", train_tasks, train_total, _quota_map(args.train_structural_quota), args.seed,
                 isolate_prior_source_groups=not args.allow_prior_source_group_reuse),
        RoleSpec("eval", eval_tasks, eval_total, _quota_map(args.eval_structural_quota), args.seed + ":eval",
                 isolate_prior_source_groups=not args.allow_prior_source_group_reuse),
    ]
    for spec in specs:
        spec.validate()
    if train_total + eval_total > MAX_TOTAL_PER_ROLE:
        raise ValueError("combined diagnostic cohort target must be <= 80")
    prior = load_prior_windows(args.prior_source_windows,
                               expected_sha256=args.expected_prior_source_windows_sha256,
                               expected_release=expected_release)
    selections, source_report = _iter_valid_records(
        args.index, expected_release=expected_release,
        expected_inventory_seal=args.expected_inventory_seal_sha256,
        coverage_path=args.coverage_expectations,
        expected_coverage_sha256=_sha256(args.expected_coverage_expectations_sha256, "coverage expectations"),
        protocol_path=args.protocol_path,
        expected_protocol_sha256=_sha256(args.expected_protocol_sha256, "protocol"),
        specs=specs, prior=prior,
    )
    print(base.canonical_json({"source": source_report,
                               "train": selections["train"].report,
                               "eval": selections["eval"].report}), flush=True)
    if args.write:
        index_manifest = base.read_json(args.index / "manifest.json")
        cohort_result = write_cohort_outputs(
            args.train_output, args.eval_output, train_selection=selections["train"],
            eval_selection=selections["eval"], source_report=source_report,
            index_manifest=index_manifest, inventory_sha256=args.expected_inventory_seal_sha256,
            coverage_sha256=args.expected_coverage_expectations_sha256,
            coverage_expectations=base.read_json(args.coverage_expectations),
            protocol_sha256=args.expected_protocol_sha256, prior=prior,
        )
        print(base.canonical_json(cohort_result), flush=True)
    # Without --write this is a bounded metadata dry-run.  Neither path is
    # created and no RGB/rendering work is performed by this command.


if __name__ == "__main__":
    main()
