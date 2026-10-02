"""Build a bounded, metadata-only GRASP goal-state review batch.

This module is intentionally narrower than the generic candidate selector.  It
consumes an authenticated strict GRASP-pair pool, removes every source episode
named by the separately sealed exclusion input, joins the selected skills to
the immutable event index in one streaming pass, and emits two phase anchors
per selected episode.  The initial phase is ``goal_unbound``: it contains no
question and is suitable only for private target grounding.  After an
independent RGB review, :func:`seal_query_registry` adds a target-conditioned
question and registry SHA; only those sealed events may be passed to the actor
renderer.

No action, outcome, recovery, DART, BC, training, or label field is inferred by
this module.  A GRASP pair is a structural metadata candidate, not a failure
or retry truth.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_memlite_event_annotation_queue as queue  # noqa: E402
import select_memlite_phase_balanced_calibration_queue as phase  # noqa: E402


BATCH_SCHEMA = "p107-grasp-goal-state-batch-v1"
BATCH_MANIFEST_SCHEMA = "p107-grasp-goal-state-batch-manifest-v1"
GOAL_UNBOUND_MODE = "goal_unbound"
SEALED_QUERY_MODE = "sealed_query"
LEGACY_FIXED_MODE = "legacy_fixed"
EXPECTED_METADATA_SHA256 = "c62fe885143bcdc07a9dcb302a5af294afb355db98d078a838c587f9efcc16ca"
EXPECTED_SOURCE_RELEASE_SHA256 = "90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23"
EXPECTED_EXCLUSION_SCHEMA = "p107-source-episode-exclusions-v1"
EXPECTED_TRIAGE_SCHEMA = "p107-natural-retry-metadata-triage-v2"
ROLE = "student_candidate"
SPLIT = "train"
FRAME_RATE_HZ = 30
MAX_BATCH_EPISODES = 32
FORBIDDEN_QUERY_TOKENS = (
    "success", "failure", "failed", "succeeded", "recovery", "retry", "future", "phase",
    "outcome", "terminal", "entry", "label", "ground truth", "any object",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular JSON input is missing: {path}")
    return json.loads(path.read_bytes())


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular JSONL input is missing: {path}")
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{number}")
            yield row


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tuple_key(group: Any, episode: Any, *, name: str) -> tuple[str, int]:
    return _require_sha(group, f"{name}.source_group_id"), _require_int(episode, f"{name}.episode_index")


def _validate_exclusion_rows(exclusions: Mapping[str, Any], *, expected_sha256: str,
                             include_protective: bool) -> tuple[set[tuple[str, int]], dict[str, int]]:
    if exclusions.get("schema_version") != EXPECTED_EXCLUSION_SCHEMA:
        raise ValueError("exclusion input schema is not the sealed p107 source-episode contract")
    mandatory_names = (
        "mandatory_root_reviewed_source_episode_tuples",
        "mandatory_coverage_eligible_unreviewed_train_query_tuples",
    )
    selected: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for name in mandatory_names:
        rows = exclusions.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"exclusion field {name} is not a list")
        counts[name] = len(rows)
        for index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) < 2:
                raise ValueError(f"malformed exclusion tuple {name}[{index}]")
            key = _tuple_key(row[0], row[1], name=f"{name}[{index}]")
            selected.add(key)
    optional = {
        "protective_category_quarantine_query_tuples_not_e2": include_protective,
        "supplemental_private_review_source_episodes_not_e2": True,
        "supplemental_runtime_structural_source_episode_not_e2": True,
    }
    for name, enabled in optional.items():
        rows = exclusions.get(name)
        if not isinstance(rows, list):
            # The runtime field is a single tuple list in the sealed file.
            if name == "supplemental_runtime_structural_source_episode_not_e2" and isinstance(rows, list):
                pass
            else:
                raise ValueError(f"exclusion field {name} is not a list")
        if not enabled:
            counts[name] = 0
            continue
        counts[name] = 0
        if name == "supplemental_runtime_structural_source_episode_not_e2":
            rows_to_check = [rows]
        else:
            rows_to_check = rows
        for index, row in enumerate(rows_to_check):
            if not isinstance(row, list) or len(row) < 2:
                raise ValueError(f"malformed exclusion tuple {name}[{index}]")
            selected.add(_tuple_key(row[0], row[1], name=f"{name}[{index}]"))
            counts[name] += 1
    actual = exclusions.get("counts")
    if not isinstance(actual, Mapping):
        raise ValueError("exclusion input has no count receipt")
    if int(actual.get("mandatory_union_source_episodes", -1)) != len({
            _tuple_key(row[0], row[1], name="mandatory")
            for field in mandatory_names for row in exclusions[field]
    }):
        raise ValueError("exclusion mandatory union count is inconsistent")
    # expected_sha256 is checked by the caller against the bytes.  Retain it
    # here to make accidental unpinned helper use obvious in stack traces.
    _require_sha(expected_sha256, "expected exclusion SHA")
    return selected, counts


def load_exclusions(path: Path, *, expected_sha256: str, include_protective: bool = True) -> tuple[set[tuple[str, int]], dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"exclusion SHA mismatch: {actual}")
    raw = _read_json(path)
    selected, counts = _validate_exclusion_rows(raw, expected_sha256=expected_sha256,
                                                include_protective=include_protective)
    return selected, {"sha256": actual, "schema_version": raw["schema_version"],
                      "applied_counts": counts, "applied_tuple_count": len(selected),
                      "protective_quarantine_applied": include_protective,
                      # Keep these authenticated episode hints for a later
                      # source-group cross-check.  They are not an alternate
                      # exclusion key: a group mismatch must fail closed.
                      "supplemental_private_rows": copy.deepcopy(
                          raw["supplemental_private_review_source_episodes_not_e2"]),
                      "supplemental_runtime_row": copy.deepcopy(
                          raw["supplemental_runtime_structural_source_episode_not_e2"])}


def _candidate_interval(row: Mapping[str, Any], name: str) -> tuple[int, int]:
    skill = row.get(name)
    if not isinstance(skill, Mapping):
        raise ValueError(f"candidate lacks {name}")
    start = _require_int(skill.get("skill_start"), f"{name}.skill_start")
    end = _require_int(skill.get("skill_end"), f"{name}.skill_end")
    if not start < end:
        raise ValueError(f"candidate {name} interval is empty")
    if skill.get("verb") != "GRASP" or not isinstance(skill.get("target"), str) or not skill.get("target"):
        raise ValueError(f"candidate {name} is not an explicit GRASP target")
    return start, end


def _validate_candidate(row: Mapping[str, Any], *, number: int, excluded: set[tuple[str, int]]) -> tuple[str, int]:
    if row.get("record_kind") == "summary":
        raise ValueError("summary row passed to candidate validator")
    if row.get("split") != SPLIT or row.get("usage_role") != ROLE or row.get("training_eligible") is not False:
        raise ValueError(f"candidate {number} is not a TRAIN student_candidate")
    if any(row.get(field) != "NOT_APPLICABLE" for field in ("attempt_status", "outcome_status", "recovery_status")):
        raise ValueError(f"candidate {number} contains an outcome/recovery status")
    for field in ("action_supervision", "outcome_supervision", "recovery_supervision"):
        if row.get(field) is not False:
            raise ValueError(f"candidate {number} enables forbidden supervision: {field}")
    group, episode = _tuple_key(row.get("source_group_id"), row.get("episode_index"), name=f"candidate[{number}]")
    if (group, episode) in excluded:
        raise ValueError("excluded candidate reached post-filter validation")
    first_start, first_end = _candidate_interval(row, "first_grasp")
    second_start, second_end = _candidate_interval(row, "second_grasp")
    if not first_start < first_end <= second_start < second_end:
        raise ValueError(f"candidate {number} does not have strict half-open GRASP order")
    if row.get("target") != row["first_grasp"].get("target") or row.get("target") != row["second_grasp"].get("target"):
        raise ValueError(f"candidate {number} target binding is inconsistent")
    return group, episode


def load_candidate_pool(path: Path, *, expected_sha256: str, expected_metadata_sha256: str,
                        exclusions: set[tuple[str, int]], exclusion_receipt: Mapping[str, Any],
                        expected_source_groups_sha256: str | None = None,
                        target_count: int = MAX_BATCH_EPISODES) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"candidate pool SHA mismatch: {actual}")
    rows = list(_iter_jsonl(path))
    if not rows or rows[0].get("record_kind") != "summary":
        raise ValueError("candidate pool must start with its authenticated summary row")
    summary = rows[0]
    if (summary.get("schema_version") != EXPECTED_TRIAGE_SCHEMA or
            summary.get("metadata_sha256") != expected_metadata_sha256 or
            (expected_source_groups_sha256 is not None and
             summary.get("source_groups_sha256") != expected_source_groups_sha256)):
        raise ValueError("candidate pool metadata/schema pin mismatch")
    if summary.get("status_fields", {}).get("training_eligible") is not False:
        raise ValueError("candidate pool summary is not candidate-only")
    eligible: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for number, row in enumerate(rows[1:], 2):
        raw_key = _tuple_key(row.get("source_group_id"), row.get("episode_index"), name=f"candidate[{number}]")
        # The sealed exclusion contract is keyed by (group, episode).  If a
        # supplemental row names this episode with a different group, do not
        # silently fall back to episode-only matching; that would hide an
        # input identity drift.  Stop before producing a seemingly valid batch.
        for supplemental in exclusion_receipt.get("supplemental_private_rows", []):
            if int(supplemental[1]) == raw_key[1] and raw_key != (supplemental[0], raw_key[1]):
                raise ValueError(
                    f"supplemental exclusion source_group mismatch for episode {raw_key[1]}: "
                    f"sealed={supplemental[0]} candidate={raw_key[0]}")
        runtime = exclusion_receipt.get("supplemental_runtime_row")
        if isinstance(runtime, list) and len(runtime) >= 2 and int(runtime[1]) == raw_key[1] and \
                raw_key != (runtime[0], raw_key[1]):
            raise ValueError(
                f"runtime exclusion source_group mismatch for episode {raw_key[1]}: "
                f"sealed={runtime[0]} candidate={raw_key[0]}")
        key = _validate_candidate(row, number=number, excluded=exclusions) if raw_key not in exclusions else None
        if key is None:
            continue
        if key in seen:
            raise ValueError("candidate pool has duplicate source episode after exclusion")
        seen.add(key)
        eligible.append(row)
    if len(eligible) < target_count:
        raise ValueError(f"strict candidate pool has only {len(eligible)} eligible episodes; need {target_count}")
    # Preserve the triage algorithm's shortest-gap ordering within each task,
    # then round-robin tasks.  This gives deterministic task spread without
    # pretending that task coverage is visual evidence.
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_task[_require_int(row.get("task_index"), "candidate.task_index")].append(row)
    for values in by_task.values():
        values.sort(key=lambda row: (
            _require_int(row["inter_grasp_end_to_start_gap_frames"], "candidate.gap"),
            _candidate_interval(row, "first_grasp")[0],
            _candidate_interval(row, "second_grasp")[0],
            _require_int(row["episode_index"], "candidate.episode_index"),
            row["candidate_id"],
        ))
    selected: list[dict[str, Any]] = []
    task_order = sorted(by_task)
    round_index = 0
    while len(selected) < target_count:
        added = False
        for task in task_order:
            values = by_task[task]
            if round_index < len(values):
                selected.append(values[round_index])
                added = True
                if len(selected) == target_count:
                    break
        if not added:
            break
        round_index += 1
    if len({(row["source_group_id"], int(row["episode_index"])) for row in selected}) != target_count:
        raise ValueError("selected batch has duplicate source episodes")
    if len({row["task_index"] for row in selected}) < 16:
        raise ValueError("selected batch has fewer than 16 task indices")
    return selected, {"sha256": actual, "summary": summary,
                      "pool_rows": len(rows) - 1, "eligible_after_exclusion": len(eligible),
                      "selected_rows": len(selected), "selected_task_count": len({row["task_index"] for row in selected}),
                      "exclusion_applied_before_selection": True}


def _index_receipt(index: Path, filename: str, expected_sha256: str | None) -> tuple[dict[str, Any], str]:
    manifest = _read_json(index / "manifest.json")
    if manifest.get("schema_version") != "memlite-event-index-v1" or manifest.get("training_eligible") is not False:
        raise ValueError("input is not the sealed candidate index")
    receipt = manifest.get("files", {}).get(filename)
    if not isinstance(receipt, Mapping):
        raise ValueError(f"index manifest lacks {filename} receipt")
    actual = sha256_file(index / filename)
    if actual != receipt.get("sha256") or (expected_sha256 is not None and actual != expected_sha256):
        raise ValueError(f"{filename} SHA does not match the sealed index receipt")
    return manifest, actual


def _load_groups(path: Path, expected_sha256: str) -> dict[str, Mapping[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("source group file SHA mismatch")
    result: dict[str, Mapping[str, Any]] = {}
    for row in _iter_jsonl(path):
        group = _require_sha(row.get("source_group_id"), "source_group_id")
        if group in result:
            raise ValueError("duplicate source_group_id in sealed source groups")
        if row.get("original_split") != SPLIT or row.get("usage_role") != ROLE or \
                row.get("source_release_manifest_sha256") != EXPECTED_SOURCE_RELEASE_SHA256:
            continue
        result[group] = row
    return result


def _event_key(event: Mapping[str, Any]) -> tuple[Any, ...] | None:
    source = event.get("source")
    skills = event.get("skill_bundle")
    interval = event.get("event_interval")
    if not isinstance(source, Mapping) or not isinstance(skills, list) or len(skills) != 1 or not isinstance(interval, Mapping):
        return None
    skill = skills[0]
    if not isinstance(skill, Mapping):
        return None
    try:
        return (
            source.get("source_group_id"), source.get("episode_index"), skill.get("skill_idx"),
            skill.get("skill_start"), skill.get("skill_end"), skill.get("verb"), skill.get("target"),
            source.get("source_annotation_sha256"),
        )
    except AttributeError:
        return None


def _candidate_event_key(candidate: Mapping[str, Any], skill_name: str) -> tuple[Any, ...]:
    skill = candidate[skill_name]
    return (
        candidate["source_group_id"], candidate["episode_index"], skill.get("skill_idx"),
        skill.get("skill_start"), skill.get("skill_end"), skill.get("verb"), skill.get("target"),
        candidate.get("source_annotation_sha256"),
    )


def join_events(index: Path, selected: Sequence[Mapping[str, Any]], *, expected_event_sha256: str,
                groups: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    wanted: dict[tuple[Any, ...], tuple[str, int, str]] = {}
    for row in selected:
        for name in ("first_grasp", "second_grasp"):
            key = _candidate_event_key(row, name)
            identity = (row["candidate_id"], int(row["episode_index"]), name)
            if key in wanted:
                raise ValueError("selected candidates share an ambiguous source skill identity")
            wanted[key] = identity
    found: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    event_path = index / "event_candidates.jsonl"
    if sha256_file(event_path) != expected_event_sha256:
        raise ValueError("event candidate payload SHA mismatch")
    for event in _iter_jsonl(event_path):
        key = _event_key(event)
        identity = wanted.get(key)
        if identity is None:
            continue
        source = event.get("source")
        skill = event.get("skill_bundle", [None])[0]
        group = groups.get(source.get("source_group_id")) if isinstance(source, Mapping) else None
        if group is None or event.get("usage_role") != ROLE or source.get("original_split") != SPLIT:
            raise ValueError("selected event does not retain TRAIN student_candidate identity")
        if (source.get("source_release_manifest_sha256") != EXPECTED_SOURCE_RELEASE_SHA256 or
                source.get("task_index") != group.get("task_index") or
                source.get("task_instance_id") != group.get("task_instance_id") or
                source.get("episode_index") not in group.get("source_episode_ids", [])):
            raise ValueError("selected event source/group identity drifted")
        if not isinstance(skill, Mapping) or event.get("event_interval") != {
                "start_frame": skill.get("skill_start"), "end_frame": skill.get("skill_end") }:
            raise ValueError("event interval does not exactly match its named GRASP skill")
        locators = event.get("video_locators")
        if not isinstance(locators, list) or len(locators) != 3 or {x.get("view") for x in locators if isinstance(x, Mapping)} != {
                "head", "left_wrist", "right_wrist"}:
            raise ValueError("selected event lacks the exact three camera locators")
        if identity in found or not isinstance(event.get("event_id"), str):
            raise ValueError("duplicate/malformed selected event join")
        found[identity] = event
    if len(found) != len(wanted):
        missing = sorted(set(wanted.values()) - set(found))
        raise ValueError(f"sealed event index is missing selected GRASP events: {missing[:3]}")
    return found


def _as_base_candidate(event: Mapping[str, Any], *, signal: str) -> queue.Candidate:
    source = event["source"]
    skill = event["skill_bundle"][0]
    group = source["source_group_id"]
    start = event["event_interval"]["start_frame"]
    end = event["event_interval"]["end_frame"]
    return queue.Candidate(
        event=event, event_id=event["event_id"], source_group_id=group,
        task_id=source["task_index"], task_instance_id=source["task_instance_id"],
        episode_key=(group, source["raw_episode_id"], source["episode_index"]),
        episode_index=source["episode_index"], episode_length=source["episode_length"],
        anchor_frame=event["observation"]["frame"], interval_start=start, interval_end=end,
        skill_ids=(skill["skill_id"],), raw_source_verbs=(skill["verb"],),
        skill_keys=(queue.skill_key(skill),), repeat_attempt_count=0,
        boundary_before=False, boundary_after=False, long_interval=(end - start >= 1),
        candidate_signals=(signal,), selection_stratum="GRASP_GOAL_STATE_BATCH",
    )


def _windows(event: Mapping[str, Any], phase_name: str) -> dict[str, Any]:
    anchor = event["observation"]["frame"]
    start = event["event_interval"]["start_frame"]
    frames = [anchor] if phase_name == "ENTRY" else [start, anchor]
    if phase_name == "ENTRY" and anchor != start:
        raise ValueError("ENTRY anchor must be the GRASP start")
    if phase_name == "TERMINAL" and anchor != event["event_interval"]["end_frame"] - 1:
        raise ValueError("TERMINAL anchor must be GRASP end-1")
    return {
        "source_clock": "published_local_frame_index_offset_0", "frame_rate_hz": FRAME_RATE_HZ,
        "anchor_frame": anchor, "anchor_timestamp_s": anchor / FRAME_RATE_HZ,
        "actor_available_window": {"start_frame": frames[0], "end_frame_exclusive": anchor + 1,
                                    "start_timestamp_s": frames[0] / FRAME_RATE_HZ,
                                    "end_timestamp_s_exclusive": (anchor + 1) / FRAME_RATE_HZ,
                                    "sampled_frames": frames, "causal_use": "CURRENT_DECISION_ONLY"},
        "offline_review_before_window": {"start_frame": anchor, "end_frame_exclusive": anchor,
                                          "start_timestamp_s": anchor / FRAME_RATE_HZ,
                                          "end_timestamp_s_exclusive": anchor / FRAME_RATE_HZ,
                                          "sampled_frames": [], "review_use": "TEMPORAL_CONTEXT_ONLY"},
        "offline_review_after_window": {"start_frame": anchor + 1, "end_frame_exclusive": anchor + 1,
                                         "start_timestamp_s": (anchor + 1) / FRAME_RATE_HZ,
                                         "end_timestamp_s_exclusive": (anchor + 1) / FRAME_RATE_HZ,
                                         "sampled_frames": [], "review_use": "OFFLINE_BC_QUALITY_ONLY_NOT_ACTOR_EVIDENCE"},
        "temporal_review_window": {"start_frame": frames[0], "end_frame_exclusive": anchor + 1,
                                    "start_timestamp_s": frames[0] / FRAME_RATE_HZ,
                                    "end_timestamp_s_exclusive": (anchor + 1) / FRAME_RATE_HZ,
                                    "sampled_frames": frames, "review_use": "HUMAN_OR_MODEL_OFFLINE_REVIEW_ONLY"},
    }


def build_goal_unbound_events(selected: Sequence[Mapping[str, Any]], joined: Mapping[tuple[str, int, str], Mapping[str, Any]],
                              *, protocol: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    policy_sha = hashlib.sha256(canonical({"schema": BATCH_SCHEMA, "mode": GOAL_UNBOUND_MODE,
                                           "entry_terminal": True, "future_frames": False}).encode()).hexdigest()
    for row in selected:
        first = joined[(row["candidate_id"], int(row["episode_index"]), "first_grasp")]
        parent = _as_base_candidate(first, signal="STRICT_SAME_TARGET_GRASP_PAIR_FIRST_MEMBER")
        for phase_name, anchor in (("ENTRY", first["event_interval"]["start_frame"]),
                                   ("TERMINAL", first["event_interval"]["end_frame"] - 1)):
            item = phase.PhaseCandidate(
                parent=parent, stratum="GRASP_GOAL_STATE_BATCH", observation_phase=phase_name,
                anchor_frame=anchor, queried_skill=copy.deepcopy(first["skill_bundle"][0]),
                parent_skill_index=0, repeat_distinct_episode_count=1,
                successor_parent_event_id=None, transition_gap_frames=None,
            )
            event = phase.derived_event(
                item, protocol=protocol, usage_role=ROLE,
                private_diagnostic_student_candidate=True,
                private_goal_state_mode=GOAL_UNBOUND_MODE,
            )
            if event["event_id"] in {existing["event_id"] for existing in events}:
                raise ValueError("derived phase event identity collision")
            derived = _as_base_candidate(event, signal="GRASP_GOAL_STATE_PHASE_ANCHOR")
            windows = _windows(event, phase_name)
            job, request = queue.queue_job(
                derived, queue_kind="STUDENT_CANDIDATE_GOAL_STATE_GROUNDING",
                selection_phase=f"GOAL_UNBOUND_{phase_name}", policy_sha256=policy_sha, windows=windows)
            events.append(event)
            jobs.append(job)
            requests.append(request)
    if len(events) != 2 * len(selected) or len({event["source"]["episode_index"] for event in events}) != len(selected):
        raise ValueError("goal-state batch must emit exactly two anchors per source episode")
    queue.validate_queue_payloads(jobs, [], requests, expected_student_jobs=jobs,
                                 expected_calibration_jobs=[], expected_requests=requests)
    return events, jobs, requests


def validate_goal_unbound_event(event: Mapping[str, Any], *, protocol: Any) -> None:
    protocol.validate_event(event)
    lineage = event.get("phase_lineage")
    if (event.get("usage_role") != ROLE or not isinstance(lineage, Mapping) or
            lineage.get("private_goal_state_review") is not True or lineage.get("goal_state_mode") != GOAL_UNBOUND_MODE):
        raise ValueError("event is not an authenticated goal_unbound student event")
    if any(key in lineage for key in ("goal_state_question", "goal_state_question_sha256",
                                      "goal_state_query_registry_sha256")):
        raise ValueError("goal_unbound event contains question material")
    for field in ("training_eligible", "outcome_supervision", "recovery_supervision", "action_bc_supervision", "dart_supervision"):
        if lineage.get(field) is not False:
            raise ValueError(f"goal_unbound event gate is not false: {field}")


def validate_goal_unbound_queue(events: Sequence[Mapping[str, Any]], jobs: Sequence[Mapping[str, Any]],
                                requests: Sequence[Mapping[str, Any]], *, protocol: Any) -> None:
    """Validate the batch-specific event→queue→camera request binding."""
    by_event = {event["event_id"]: event for event in events}
    if len(by_event) != len(events) or len(jobs) != len(events) or len(requests) != len(events):
        raise ValueError("goal-state queue must have one job and request per derived event")
    seen_jobs: set[str] = set()
    seen_requests: set[str] = set()
    for job, request in zip(jobs, requests):
        event_id = job.get("event_id")
        event = by_event.get(event_id)
        if event is None or job.get("job_id") != request.get("request_id") or request.get("event_id") != event_id:
            raise ValueError("goal-state queue event/job/request identity drifted")
        if job["job_id"] in seen_jobs or request["request_id"] in seen_requests:
            raise ValueError("goal-state queue contains duplicate job/request IDs")
        seen_jobs.add(job["job_id"])
        seen_requests.add(request["request_id"])
        validate_goal_unbound_event(event, protocol=protocol)
        phase_name = event["phase_lineage"]["observation_phase"]
        expected = [event["observation"]["frame"]] if phase_name == "ENTRY" else [
            event["event_interval"]["start_frame"], event["observation"]["frame"]]
        if (job.get("usage_role") != ROLE or job.get("immutable_split") != SPLIT or
                job.get("training_eligible") is not False or request.get("requested_frame_indices") != expected or
                request.get("actor_available_frame_indices") != expected or
                request.get("offline_review_before_frame_indices") != [] or
                request.get("offline_review_after_frame_indices") != []):
            raise ValueError("goal-state queue carries non-causal/future or non-private fields")
    if seen_jobs != {job["job_id"] for job in jobs} or seen_requests != {request["request_id"] for request in requests}:
        raise ValueError("goal-state queue identity accounting is inconsistent")


def _validate_query_row(row: Mapping[str, Any], *, event: Mapping[str, Any]) -> tuple[str, str, str]:
    required = {"event_id", "query_id", "question", "question_sha256", "target_raw", "target_display_phrase",
                "relation_family"}
    if set(row) != required:
        raise ValueError("query registry row has missing or unrecognized fields")
    if row.get("event_id") != event["event_id"] or not isinstance(row.get("query_id"), str) or not row["query_id"]:
        raise ValueError("query registry row is not bound to the selected event")
    question = row.get("question")
    phrase = row.get("target_display_phrase")
    target_raw = row.get("target_raw")
    if (not isinstance(question, str) or not question.strip() or not isinstance(phrase, str) or not phrase.strip() or
            not isinstance(target_raw, str) or not target_raw.strip() or row.get("relation_family") != "grasp_current_hold"):
        raise ValueError("query registry row has incomplete target-conditioned grounding")
    qsha = row.get("question_sha256")
    if not isinstance(qsha, str) or sha256_bytes(question.encode("utf-8")) != qsha:
        raise ValueError("query question SHA mismatch")
    lower = question.casefold()
    if any(token in lower for token in FORBIDDEN_QUERY_TOKENS):
        raise ValueError("query question contains phase/outcome/future language")
    if target_raw.casefold() in lower or target_raw.casefold() == phrase.casefold():
        raise ValueError("query must use a reviewed visual target phrase, not a raw metadata ID")
    if phrase.casefold() not in lower:
        raise ValueError("query does not contain its reviewed visual target phrase")
    if not lower.startswith("at the anchor,"):
        raise ValueError("query must be an anchor-local current-state question")
    return question, qsha, phrase


def seal_query_registry(events: Sequence[Mapping[str, Any]], registry_rows: Sequence[Mapping[str, Any]], *,
                        registry_sha256: str, protocol: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    _require_sha(registry_sha256, "query registry SHA")
    if len(events) != len(registry_rows) or len({event["event_id"] for event in events}) != len(events):
        raise ValueError("query registry must cover each selected event exactly once")
    by_id = {event["event_id"]: event for event in events}
    sealed: list[dict[str, Any]] = []
    questions: dict[str, dict[str, str]] = {}
    seen = set()
    for row in registry_rows:
        event = by_id.get(row.get("event_id"))
        if event is None or row["event_id"] in seen:
            raise ValueError("query registry has duplicate or unknown event_id")
        validate_goal_unbound_event(event, protocol=protocol)
        question, question_sha, _phrase = _validate_query_row(row, event=event)
        copy_event = copy.deepcopy(event)
        lineage = copy_event["phase_lineage"]
        lineage["goal_state_mode"] = SEALED_QUERY_MODE
        lineage["goal_state_question"] = question
        lineage["goal_state_question_sha256"] = question_sha
        lineage["goal_state_query_registry_sha256"] = registry_sha256
        protocol.validate_event(copy_event)
        sealed.append(copy_event)
        questions[event["event_id"]] = {"question": question}
        seen.add(row["event_id"])
    if seen != set(by_id):
        raise ValueError("query registry does not cover the complete event set")
    return sealed, questions


def validate_blind_contexts(events: Sequence[Mapping[str, Any]], assignments: Mapping[str, Sequence[str]]) -> None:
    event_by_id = {event["event_id"]: event for event in events}
    seen: set[str] = set()
    for context, event_ids in assignments.items():
        if not isinstance(context, str) or not context or not isinstance(event_ids, Sequence):
            raise ValueError("blind context assignment is malformed")
        episodes: set[tuple[str, int]] = set()
        for event_id in event_ids:
            if event_id in seen or event_id not in event_by_id:
                raise ValueError("blind context assignment duplicates or references an unknown event")
            event = event_by_id[event_id]
            source = event["source"]
            key = (source["source_group_id"], source["episode_index"])
            if key in episodes:
                raise ValueError("one fresh blind context contains multiple anchors from one episode")
            episodes.add(key)
            seen.add(event_id)
    if seen != set(event_by_id):
        raise ValueError("blind context assignment does not cover all events")


def build_batch(*, index: Path, candidate_pool: Path, exclusions_path: Path, output: Path | None,
                expected_candidate_pool_sha256: str, expected_exclusions_sha256: str,
                expected_metadata_sha256: str = EXPECTED_METADATA_SHA256,
                expected_index_manifest_sha256: str | None = None,
                expected_event_sha256: str | None = None,
                expected_source_groups_sha256: str | None = None,
                protocol_path: Path | None = None, expected_protocol_sha256: str | None = None,
                include_protective: bool = True, target_count: int = MAX_BATCH_EPISODES) -> dict[str, Any]:
    if target_count != MAX_BATCH_EPISODES:
        raise ValueError("this bounded pilot is fixed at exactly 32 source episodes")
    if expected_metadata_sha256 != EXPECTED_METADATA_SHA256:
        raise ValueError("metadata pin is not the frozen p107 episodes SHA")
    index = Path(index)
    if index.is_symlink() or not index.is_dir():
        raise ValueError("index must be a regular sealed directory")
    manifest = _read_json(index / "manifest.json")
    manifest_sha = sha256_file(index / "manifest.json")
    if expected_index_manifest_sha256 is not None and manifest_sha != expected_index_manifest_sha256:
        raise ValueError("index manifest SHA mismatch")
    if manifest.get("source_release_manifest_sha256") != EXPECTED_SOURCE_RELEASE_SHA256:
        raise ValueError("source release manifest is not the frozen p107 release")
    source_file = index / "source_groups.jsonl"
    source_sha = manifest["files"]["source_groups.jsonl"]["sha256"]
    event_sha = manifest["files"]["event_candidates.jsonl"]["sha256"]
    if expected_source_groups_sha256 is not None and expected_source_groups_sha256 != source_sha:
        raise ValueError("source group expected SHA disagrees with sealed index manifest")
    if expected_event_sha256 is not None and expected_event_sha256 != event_sha:
        raise ValueError("event expected SHA disagrees with sealed index manifest")
    groups = _load_groups(source_file, source_sha)
    excluded, exclusion_receipt = load_exclusions(exclusions_path, expected_sha256=expected_exclusions_sha256,
                                                   include_protective=include_protective)
    selected, pool_receipt = load_candidate_pool(candidate_pool, expected_sha256=expected_candidate_pool_sha256,
                                                 expected_metadata_sha256=expected_metadata_sha256,
                                                 exclusions=excluded, exclusion_receipt=exclusion_receipt,
                                                 expected_source_groups_sha256=source_sha,
                                                 target_count=target_count)
    joined = join_events(index, selected, expected_event_sha256=event_sha, groups=groups)
    if protocol_path is None:
        protocol_path = REPO_ROOT / "src" / "g05" / "data" / "memlite_event_protocol.py"
    if expected_protocol_sha256 is None:
        expected_protocol_sha256 = manifest.get("protocol_sha256")
    protocol = queue.load_canonical_protocol(protocol_path, expected_sha256=expected_protocol_sha256)
    events, jobs, requests = build_goal_unbound_events(selected, joined, protocol=protocol)
    for event in events:
        validate_goal_unbound_event(event, protocol=protocol)
    validate_goal_unbound_queue(events, jobs, requests, protocol=protocol)
    candidate_rows = []
    for row in selected:
        first = joined[(row["candidate_id"], int(row["episode_index"]), "first_grasp")]
        second = joined[(row["candidate_id"], int(row["episode_index"]), "second_grasp")]
        candidate_rows.append({
            "candidate_id": row["candidate_id"], "source_group_id": row["source_group_id"],
            "episode_index": row["episode_index"], "task_index": row["task_index"],
            "task_instance_id": row["task_instance_id"], "target_raw": row["target"],
            "first_event_id": first["event_id"], "second_event_id": second["event_id"],
            "first_interval": first["event_interval"], "second_interval": second["event_interval"],
            "intervening_skills": row.get("intervening_skills", []),
            "structural_candidate_only": True, "attempt_status": "NOT_APPLICABLE",
            "outcome_status": "NOT_APPLICABLE", "recovery_status": "NOT_APPLICABLE",
            "training_eligible": False,
        })
    result = {
        "schema_version": BATCH_MANIFEST_SCHEMA, "status": "GOAL_UNBOUND_GROUNDING_REQUIRED",
        "training_eligible": False, "usage_role": ROLE, "split": SPLIT,
        "private_goal_state_mode": GOAL_UNBOUND_MODE,
        "private_review_may_include_future_or_cross_skill": False,
        "actor_event_count": 0, "label_count": 0, "outcome_or_recovery_count": 0,
        "source_release_manifest_sha256": EXPECTED_SOURCE_RELEASE_SHA256,
        "input_pins": {"candidate_pool": pool_receipt, "exclusions": exclusion_receipt,
                        "index_manifest_sha256": manifest_sha, "event_candidates_sha256": event_sha,
                        "source_groups_sha256": source_sha, "metadata_sha256": expected_metadata_sha256,
                        "protocol_sha256": expected_protocol_sha256},
        "selection": {"source_episode_count": len(selected), "anchor_count": len(events),
                      "task_count": len({row["task_index"] for row in selected}),
                      "candidate_ids": [row["candidate_id"] for row in selected],
                      "selection_rule": "strict GRASP pair pool -> exact sealed exclusions -> task-index round-robin",
                      "grasp_only": True, "open_toggle_count": 0, "navigation_metric_count": 0},
        "files": {},
        "non_authorization": {"canonical_labels_created": False, "student_release": False,
                              "attempt_outcome": False, "recovery_decision": False,
                              "corrective_action": False, "bc_or_dart_supervision": False},
        "candidate_rows": candidate_rows,
        "query_stage": {"status": "UNBOUND_NO_QUERY_OR_PLACEHOLDER", "registry_required_after_private_rgb_grounding": True},
        "events": events, "queue_jobs": jobs, "render_requests": requests,
    }
    if output is not None:
        write_batch(Path(output), result, protocol=protocol)
    return result


def write_batch(output: Path, result: Mapping[str, Any], *, protocol: Any) -> None:
    if output.exists():
        raise FileExistsError("batch output exists; refusing overwrite")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        def write_jsonl(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            path = staging / name
            path.write_bytes(b"".join(protocol.canonical_json(row).encode() + b"\n" for row in rows))
            return {"sha256": sha256_file(path), "bytes": path.stat().st_size, "rows": len(rows)}
        files = {
            "candidate_rows.jsonl": write_jsonl("candidate_rows.jsonl", result["candidate_rows"]),
            "goal_unbound_events.jsonl": write_jsonl("goal_unbound_events.jsonl", result["events"]),
            "student_candidate_queue.jsonl": write_jsonl("student_candidate_queue.jsonl", result["queue_jobs"]),
            "camera_native_render_requests.jsonl": write_jsonl("camera_native_render_requests.jsonl", result["render_requests"]),
        }
        manifest = dict(result)
        for key in ("candidate_rows", "events", "queue_jobs", "render_requests"):
            manifest.pop(key, None)
        manifest["files"] = files
        (staging / "manifest.json").write_bytes(protocol.canonical_json(manifest).encode() + b"\n")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--expected-candidate-pool-sha256", required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--expected-exclusions-sha256", required=True)
    parser.add_argument("--expected-index-manifest-sha256", required=True)
    parser.add_argument("--expected-event-sha256", required=True)
    parser.add_argument("--expected-source-groups-sha256", required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-protective-exclusions", action="store_true")
    args = parser.parse_args()
    result = build_batch(
        index=args.index, candidate_pool=args.candidate_pool, exclusions_path=args.exclusions, output=args.output,
        expected_candidate_pool_sha256=args.expected_candidate_pool_sha256,
        expected_exclusions_sha256=args.expected_exclusions_sha256,
        expected_index_manifest_sha256=args.expected_index_manifest_sha256,
        expected_event_sha256=args.expected_event_sha256,
        expected_source_groups_sha256=args.expected_source_groups_sha256,
        protocol_path=args.protocol_path,
        expected_protocol_sha256=args.expected_protocol_sha256,
        include_protective=not args.allow_protective_exclusions,
    )
    print(canonical({"status": result["status"], "output": str(args.output),
                     "source_episode_count": result["selection"]["source_episode_count"],
                     "anchor_count": result["selection"]["anchor_count"],
                     "task_count": result["selection"]["task_count"]}), flush=True)


if __name__ == "__main__":
    main()
