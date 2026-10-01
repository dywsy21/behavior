"""Publish a sealed, calibration-only diagnostic derivative of a phase mini-index.

The input phase index remains immutable.  This tool copies its compact event and
source-group payloads into a new index only after binding the phase queue and
selection manifest to a sealed full-v3 vocabulary contract.  It never turns a
partial 40-event sample into a complete source index, and it never emits a
student, action, outcome, recovery, or training authorization.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_memlite_event_annotation_queue as base  # noqa: E402


SCHEMA = "p107-memlite-diagnostic-subset-index-v1"
SELECTION_SCHEMA = "p107-phase40-diagnostic-subset-calibration-selection-v1"
EXPECTED_EVENT_COUNT = 40
PAYLOAD_FILES = ("event_candidates.jsonl", "source_groups.jsonl")


def _sha256(path: Path) -> str:
    return base.sha256_file(path)


def _require_sha(value: Any, name: str) -> str:
    if not base.is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return str(value)


def _read_pinned_json(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    path = Path(path)
    _require_sha(expected_sha256, f"expected {name} SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing regular file")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{name} SHA-256 does not match its external pin")
    value = base.read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _read_pinned_jsonl(path: Path, expected_sha256: str, *, name: str) -> list[dict[str, Any]]:
    path = Path(path)
    _require_sha(expected_sha256, f"expected {name} SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing regular file")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{name} SHA-256 does not match its external pin")
    return list(base.iter_jsonl(path))


def _expect_exact_receipt(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "rows", "bytes"}:
        raise ValueError(f"{name} receipt is malformed")
    digest, rows, byte_count = value.get("sha256"), value.get("rows"), value.get("bytes")
    if not base.is_sha256(digest) or type(rows) is not int or rows < 0 or type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{name} receipt is malformed")
    return {"sha256": digest, "rows": rows, "bytes": byte_count}


def _verify_inventory(index: Path, expected_inventory_seal_sha256: str, *, name: str) -> tuple[dict[str, Any], str]:
    index = Path(index)
    if index.is_symlink() or not index.is_dir():
        raise ValueError(f"{name} must be an existing regular index directory")
    _, actual = base.validate_inventory_seal(index, expected_inventory_seal_sha256=expected_inventory_seal_sha256)
    manifest = base.read_json(index / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != base.INDEX_SCHEMA:
        raise ValueError(f"{name} is not a P107 event index")
    return manifest, actual


def _coverage_contract(full_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    coverage = full_manifest.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("full-v3 index has no explicit coverage report")
    expectations = coverage.get("expectations")
    if not isinstance(expectations, Mapping):
        raise ValueError("full-v3 coverage report has no official expectations")
    required = {
        "schema_version", "official_task_metadata_sha256", "official_skill_vocabulary_sha256",
        "expected_task_ids", "expected_skill_vocabulary", "required_task_skill_pairs",
    }
    if set(expectations) != required or expectations.get("schema_version") != "p107-official-coverage-expectations-v3":
        raise ValueError("full index must supply the official v3 coverage contract")
    expected_sha = _require_sha(full_manifest.get("coverage_expectations_sha256"), "full-v3 coverage pin")
    if coverage.get("expectations_sha256") != expected_sha or base.canonical_sha256(expectations) != expected_sha:
        raise ValueError("full-v3 coverage expectations do not match their pinned digest")
    if expectations.get("required_task_skill_pairs") is not None:
        raise ValueError("diagnostic subset publisher only supports the official NOT_DECLARED pair contract")
    tasks = expectations.get("expected_task_ids")
    vocabulary = expectations.get("expected_skill_vocabulary")
    if (not isinstance(tasks, list) or not tasks or tasks != sorted(set(tasks)) or
            any(type(item) is not int or item < 0 for item in tasks)):
        raise ValueError("full-v3 expected task vocabulary is malformed")
    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError("full-v3 expected skill vocabulary is malformed")
    previous = -1
    for row in vocabulary:
        if (not isinstance(row, Mapping) or set(row) != {"skill_id", "skill_description"} or
                type(row.get("skill_id")) is not int or row["skill_id"] < 0 or row["skill_id"] <= previous or
                not isinstance(row.get("skill_description"), str) or not row["skill_description"]):
            raise ValueError("full-v3 expected skill vocabulary is malformed")
        previous = row["skill_id"]
    if expectations.get("official_task_metadata_sha256") != full_manifest.get("source_release_manifest_sha256"):
        raise ValueError("full-v3 official task metadata pin conflicts with its immutable source")
    return dict(coverage), dict(expectations)


def _coverage_for_subset(events: Sequence[Mapping[str, Any]], groups: Mapping[str, Any],
                         expectations: Mapping[str, Any]) -> dict[str, Any]:
    expected_tasks = set(expectations["expected_task_ids"])
    vocabulary = list(expectations["expected_skill_vocabulary"])
    expected_skills = {row["skill_id"] for row in vocabulary}
    seen_events: set[str] = set()
    episode_sets: dict[tuple[int, int], set[tuple[str, int]]] = defaultdict(set)
    found_tasks: set[int] = set()
    found_skills: set[int] = set()
    for event in events:
        event_id = event.get("event_id")
        if not base.is_sha256(event_id) or event_id in seen_events:
            raise ValueError("phase index has an absent or duplicate event ID")
        seen_events.add(event_id)
        source = event.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("phase event has no immutable source identity")
        group_id = source.get("source_group_id")
        group = groups.get(group_id)
        if group is None:
            raise ValueError("phase event references an absent source group")
        task = source.get("task_index")
        episode = source.get("episode_index")
        if type(task) is not int or type(episode) is not int:
            raise ValueError("phase event has malformed source task or episode identity")
        if task not in expected_tasks:
            raise ValueError("phase event task is outside the official full-v3 vocabulary")
        if (source.get("source_release_manifest_sha256") != group["source_release_manifest_sha256"] or
                task != group["task_index"] or event.get("usage_role") != group["usage_role"] or
                source.get("original_split") != group["original_split"] or
                group["usage_role"] != "annotation_calibration" or group["original_split"] != "train"):
            raise ValueError("phase event/source-group immutable identity drifted")
        skills = event.get("skill_bundle")
        if not isinstance(skills, list) or not skills:
            raise ValueError("phase event has no source skill bundle")
        found_tasks.add(task)
        for skill in skills:
            if not isinstance(skill, Mapping) or type(skill.get("skill_id")) is not int:
                raise ValueError("phase event has malformed official skill identity")
            skill_id = skill["skill_id"]
            if skill_id not in expected_skills:
                raise ValueError("phase event skill is outside the official full-v3 vocabulary")
            found_skills.add(skill_id)
            episode_sets[(task, skill_id)].add((group_id, episode))
    observed_grid = {
        str(task): {
            str(row["skill_id"]): {
                "skill_id": row["skill_id"], "skill_description": row["skill_description"],
                "event_candidates": sum(
                    1 for event in events
                    if event["source"]["task_index"] == task and any(
                        skill.get("skill_id") == row["skill_id"] for skill in event["skill_bundle"])),
                "unique_source_episodes": len(episode_sets[task, row["skill_id"]]),
            }
            for row in vocabulary
        }
        for task in expectations["expected_task_ids"]
    }
    found_episodes = {(event["source"]["source_group_id"], event["source"]["episode_index"]) for event in events}
    source_episodes = {
        (group_id, episode) for group_id, group in groups.items() for episode in group["source_episode_ids"]
    }
    if found_episodes != source_episodes:
        raise ValueError("phase events do not exactly account for the source-group episode inventory")
    return {
        "expectations": dict(expectations),
        "expectations_sha256": base.canonical_sha256(expectations),
        "found_task_ids": sorted(found_tasks),
        "missing_task_ids": sorted(expected_tasks - found_tasks),
        "unexpected_task_ids": [],
        "found_global_skill_ids": sorted(found_skills),
        "missing_global_skill_ids": sorted(expected_skills - found_skills),
        "unmapped_source_skill_ids": [],
        "source_skill_members_missing_skill_id": 0,
        "global_vocabulary_complete": False,
        "required_task_skill_pairs": None,
        "missing_required_task_skill_pairs": None,
        "required_pair_coverage_status": "NOT_DECLARED",
        "unique_input_source_episodes": len(source_episodes),
        "unique_candidate_source_episodes": len(found_episodes),
        "unique_event_candidates": len(events),
        "observed_task_skill_grid": observed_grid,
    }


def _validate_phase_binding(*, mini_manifest: Mapping[str, Any], mini_inventory_sha256: str,
                            full_manifest_sha256: str, full_inventory_sha256: str,
                            selection_manifest: Mapping[str, Any], queue_rows: Sequence[Mapping[str, Any]],
                            events: Sequence[Mapping[str, Any]], groups: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    if selection_manifest.get("schema_version") != "p107-phase-balanced-calibration-manifest-v1":
        raise ValueError("phase selection manifest has an unsupported schema")
    mini = selection_manifest.get("mini_index")
    selection = selection_manifest.get("selection")
    policy = selection_manifest.get("policy")
    if not isinstance(mini, Mapping) or not isinstance(selection, Mapping) or not isinstance(policy, Mapping):
        raise ValueError("phase selection manifest lacks mini-index, selection, or policy binding")
    if (mini.get("manifest_sha256") != base.sha256_file(Path(mini_manifest["_path"])) or
            mini.get("inventory_seal_sha256") != mini_inventory_sha256):
        raise ValueError("phase selection manifest does not bind the exact mini index")
    required_guardrails = {
        "source_segment_end_is_not_completion_truth",
        "no_success_failure_outcome_or_recovery_label",
        "future_frames_are_only_offline_review_for_this_new_anchor",
        "no_action_supervision_emitted",
    }
    if (policy.get("schema_version") != "p107-phase-balanced-calibration-queue-v1" or
            policy.get("parent_index_manifest_sha256") != full_manifest_sha256 or
            policy.get("parent_inventory_seal_sha256") != full_inventory_sha256 or
            policy.get("source_release_manifest_sha256") != mini_manifest.get("source_release_manifest_sha256") or
            policy.get("protocol_sha256") != mini_manifest.get("protocol_sha256") or
            policy.get("coverage_expectations_sha256") != mini_manifest.get("coverage_expectations_sha256") or
            policy.get("policy_sha256") != base.canonical_sha256({key: value for key, value in policy.items() if key != "policy_sha256"})):
        raise ValueError("phase selection policy does not preserve the sealed parent lineage")
    if (selection.get("exact_budget") != EXPECTED_EVENT_COUNT or
            selection.get("unique_source_groups") != EXPECTED_EVENT_COUNT or
            selection.get("unique_source_episodes") != EXPECTED_EVENT_COUNT or
            selection.get("all_usage_role_annotation_calibration") is not True or
            selection.get("all_immutable_split_train") is not True):
        raise ValueError("phase selection manifest is not the fixed 40-row calibration-only subset")
    if len(queue_rows) != EXPECTED_EVENT_COUNT or len(events) != EXPECTED_EVENT_COUNT or len(groups) != EXPECTED_EVENT_COUNT:
        raise ValueError("phase subset must contain exactly 40 queue rows, events, and source groups")
    event_by_id = {event.get("event_id"): event for event in events}
    if len(event_by_id) != len(events) or any(not base.is_sha256(key) for key in event_by_id):
        raise ValueError("phase mini index has absent or duplicate event IDs")
    queue_by_event: dict[str, Mapping[str, Any]] = {}
    selected_groups: set[str] = set()
    for row in queue_rows:
        if not isinstance(row, Mapping) or row.get("schema_version") != "p107-phase-balanced-calibration-queue-v1":
            raise ValueError("phase queue row has an unsupported schema")
        event_id, order = row.get("event_id"), row.get("selection_order")
        if (not base.is_sha256(event_id) or event_id in queue_by_event or type(order) is not int or
                order < 0 or row.get("usage_role") != "annotation_calibration" or row.get("immutable_split") != "train" or
                row.get("training_eligible") is not False or row.get("observation_phase_is_not_outcome") is not True or
                row.get("current_actor_evidence") != {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None}):
            raise ValueError("phase queue row violates the calibration-only candidate contract")
        constraints = row.get("review_constraints")
        if (not isinstance(constraints, Mapping) or set(constraints) != required_guardrails or
                any(value is not True for value in constraints.values())):
            raise ValueError("phase queue row has weakened review guardrails")
        event = event_by_id.get(event_id)
        source = row.get("source_identity")
        event_source = event.get("source") if isinstance(event, Mapping) else None
        source_fields = ("source_release_manifest_sha256", "source_annotation_sha256", "source_group_id",
                         "task_index", "task_instance_id", "raw_episode_id", "episode_index")
        if (event is None or not isinstance(source, Mapping) or not isinstance(event_source, Mapping) or
                set(source) != set(source_fields) or
                any(source.get(field) != event_source.get(field) for field in source_fields)):
            raise ValueError("phase queue event/source binding disagrees with the mini index")
        group_id = source.get("source_group_id")
        group = groups.get(group_id)
        if (group is None or group.get("usage_role") != "annotation_calibration" or
                group.get("original_split") != "train"):
            raise ValueError("phase queue references a non-calibration source group")
        queue_by_event[event_id] = row
        selected_groups.add(group_id)
    if sorted(row["selection_order"] for row in queue_rows) != list(range(EXPECTED_EVENT_COUNT)):
        raise ValueError("phase queue selection orders are not exactly 0..39")
    if set(event_by_id) != set(queue_by_event) or selected_groups != set(groups):
        raise ValueError("phase queue does not cover each mini-index event and group exactly once")
    rows = selection.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_EVENT_COUNT:
        raise ValueError("phase selection manifest does not enumerate the 40 selected rows")
    selection_ids = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("phase selection row is malformed")
        event_id, group_id, task = row.get("event_id"), row.get("source_group_id"), row.get("task_index")
        event = event_by_id.get(event_id)
        if (event is None or event_id in selection_ids or group_id != event["source"]["source_group_id"] or
                task != event["source"]["task_index"]):
            raise ValueError("phase selection row disagrees with the mini-index event identity")
        selection_ids.add(event_id)
    if selection_ids != set(event_by_id):
        raise ValueError("phase selection manifest does not cover all mini-index events")
    return sorted(event_by_id), sorted(groups)


def _copy_verified(source: Path, destination: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    expected = _expect_exact_receipt(receipt, name=source.name)
    if _sha256(source) != expected["sha256"] or source.stat().st_size != expected["bytes"]:
        raise ValueError(f"source payload receipt mismatch before publishing: {source.name}")
    shutil.copyfile(source, destination)
    if _sha256(destination) != expected["sha256"] or destination.stat().st_size != expected["bytes"]:
        raise ValueError(f"copied payload receipt mismatch: {source.name}")
    return expected


def publish(*, phase_index: Path, expected_phase_index_inventory_seal_sha256: str,
            full_index: Path, expected_full_index_inventory_seal_sha256: str,
            expected_full_index_manifest_sha256: str, phase_selection_manifest: Path,
            expected_phase_selection_manifest_sha256: str, phase_queue: Path,
            expected_phase_queue_sha256: str, output: Path) -> dict[str, Any]:
    """Validate all pinned metadata first, then atomically publish one derivative."""
    phase_index, full_index, output = Path(phase_index), Path(full_index), Path(output)
    if output.exists():
        raise FileExistsError("output exists; immutable diagnostic indexes are never overwritten")
    base.forbid_output_inside(output, phase_index, full_index, Path(phase_selection_manifest), Path(phase_queue))
    mini_manifest, mini_inventory_sha256 = _verify_inventory(
        phase_index, expected_phase_index_inventory_seal_sha256, name="phase mini index")
    full_manifest, full_inventory_sha256 = _verify_inventory(
        full_index, expected_full_index_inventory_seal_sha256, name="full-v3 index")
    if _sha256(full_index / "manifest.json") != _require_sha(expected_full_index_manifest_sha256, "expected full-v3 manifest SHA-256"):
        raise ValueError("full-v3 manifest does not match its external pin")
    _, expectations = _coverage_contract(full_manifest)
    if (mini_manifest.get("schema_version") != base.INDEX_SCHEMA or mini_manifest.get("status") != base.INDEX_STATUS or
            mini_manifest.get("training_eligible") is not False or mini_manifest.get("partial_source_coverage") is not True or
            mini_manifest.get("source_release_manifest_sha256") != full_manifest.get("source_release_manifest_sha256") or
            mini_manifest.get("coverage_expectations_sha256") != full_manifest.get("coverage_expectations_sha256")):
        raise ValueError("phase mini index is not a compatible non-training partial derivative of the full-v3 index")
    derivation = mini_manifest.get("derivation")
    full_files = full_manifest.get("files")
    if not isinstance(full_files, Mapping):
        raise ValueError("full-v3 index lacks source payload receipts")
    full_event_receipt = _expect_exact_receipt(full_files.get("event_candidates.jsonl"),
                                               name="full-v3 event_candidates.jsonl")
    full_group_receipt = _expect_exact_receipt(full_files.get("source_groups.jsonl"),
                                               name="full-v3 source_groups.jsonl")
    if (not isinstance(derivation, Mapping) or derivation.get("schema_version") != "p107-phase-balanced-mini-index-v1" or
            derivation.get("parent_index_manifest_sha256") != expected_full_index_manifest_sha256 or
            derivation.get("parent_inventory_seal_sha256") != full_inventory_sha256 or
            derivation.get("parent_event_candidates_sha256") != full_event_receipt["sha256"] or
            derivation.get("parent_source_groups_sha256") != full_group_receipt["sha256"] or
            derivation.get("phase_observation_is_not_outcome") is not True):
        raise ValueError("phase mini index does not preserve the full-v3 lineage")
    selection = _read_pinned_json(
        phase_selection_manifest, expected_phase_selection_manifest_sha256, name="phase selection manifest")
    queue_rows = _read_pinned_jsonl(phase_queue, expected_phase_queue_sha256, name="phase queue")
    protocol = base.load_canonical_protocol(
        REPO / "src" / "g05" / "data" / "memlite_event_protocol.py",
        expected_sha256=_require_sha(mini_manifest.get("protocol_sha256"), "phase protocol SHA-256"))
    verified_mini, events_path, source_groups = base.validate_index(
        phase_index, expected_source_manifest_sha256=mini_manifest["source_release_manifest_sha256"],
        canonical_protocol=protocol, expected_protocol_sha256=mini_manifest["protocol_sha256"])
    if verified_mini != mini_manifest:
        raise ValueError("phase mini index changed while being validated")
    event_receipt = _expect_exact_receipt(mini_manifest.get("files", {}).get("event_candidates.jsonl"),
                                          name="phase event_candidates.jsonl")
    group_receipt = _expect_exact_receipt(mini_manifest.get("files", {}).get("source_groups.jsonl"),
                                          name="phase source_groups.jsonl")
    events = list(base.iter_jsonl_verified(events_path, expected=event_receipt))
    for event in events:
        protocol.validate_event(event)
    groups = {
        group_id: {
            "source_release_manifest_sha256": group.source_release_manifest_sha256,
            "task_index": group.task_id,
            "task_instance_id": group.task_instance_id,
            "original_split": group.original_split,
            "usage_role": group.usage_role,
            "source_episode_ids": list(group.source_episode_ids),
        }
        for group_id, group in source_groups.items()
    }
    event_ids, group_ids = _validate_phase_binding(
        mini_manifest={**mini_manifest, "_path": str(phase_index / "manifest.json")},
        mini_inventory_sha256=mini_inventory_sha256, full_manifest_sha256=expected_full_index_manifest_sha256,
        full_inventory_sha256=full_inventory_sha256, selection_manifest=selection, queue_rows=queue_rows,
        events=events, groups=groups)
    coverage = _coverage_for_subset(events, groups, expectations)
    if coverage["expectations_sha256"] != mini_manifest["coverage_expectations_sha256"]:
        raise ValueError("subset coverage does not retain the sealed full-v3 official expectation pin")
    policy_core = {
        "schema_version": SELECTION_SCHEMA,
        "mode": "EXPLICIT_PHASE40_EVENT_AND_SOURCE_GROUP_SET",
        "diagnostic_subset": True,
        "partial_source_coverage": True,
        "release_roles": ["annotation_calibration"],
        "training_eligible": False,
        "formal_training_release": False,
        "outcome_supervision": False,
        "recovery_supervision": False,
        "corrective_action_supervision": False,
        "student_supervision": False,
        "event_count": EXPECTED_EVENT_COUNT,
        "source_group_count": EXPECTED_EVENT_COUNT,
        "event_ids": event_ids,
        "source_group_ids": group_ids,
        "phase_mini_index_manifest_sha256": _sha256(phase_index / "manifest.json"),
        "phase_mini_index_inventory_seal_sha256": mini_inventory_sha256,
        "phase_selection_manifest_sha256": expected_phase_selection_manifest_sha256,
        "phase_queue_sha256": expected_phase_queue_sha256,
        "full_v3_index_manifest_sha256": expected_full_index_manifest_sha256,
        "full_v3_inventory_seal_sha256": full_inventory_sha256,
        "source_release_manifest_sha256": mini_manifest["source_release_manifest_sha256"],
        "coverage_expectations_sha256": mini_manifest["coverage_expectations_sha256"],
        "protocol_sha256": mini_manifest["protocol_sha256"],
    }
    policy = {**policy_core, "policy_sha256": base.canonical_sha256(policy_core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        files = {
            "event_candidates.jsonl": _copy_verified(phase_index / "event_candidates.jsonl",
                                                       staging / "event_candidates.jsonl", event_receipt),
            "source_groups.jsonl": _copy_verified(phase_index / "source_groups.jsonl",
                                                    staging / "source_groups.jsonl", group_receipt),
        }
        manifest = {
            "schema_version": base.INDEX_SCHEMA,
            "status": base.INDEX_STATUS,
            "diagnostic_subset_schema_version": SCHEMA,
            "diagnostic_subset": True,
            "partial_source_coverage": True,
            "training_eligible": False,
            "formal_training_release": False,
            "outcome_supervision": False,
            "recovery_supervision": False,
            "corrective_action_supervision": False,
            "student_supervision": False,
            "release_roles": ["annotation_calibration"],
            "source_release_manifest_sha256": mini_manifest["source_release_manifest_sha256"],
            "protocol_sha256": mini_manifest["protocol_sha256"],
            "coverage_expectations_sha256": mini_manifest["coverage_expectations_sha256"],
            "files": files,
            "source_episodes": EXPECTED_EVENT_COUNT,
            "source_groups": EXPECTED_EVENT_COUNT,
            "event_candidates": EXPECTED_EVENT_COUNT,
            "usage_roles": {"annotation_calibration": EXPECTED_EVENT_COUNT},
            "candidate_counts": {"input_source_episodes": EXPECTED_EVENT_COUNT,
                                 "eligible_source_episodes": EXPECTED_EVENT_COUNT,
                                 "event_candidates": EXPECTED_EVENT_COUNT},
            "accepted_label_counts": {"goal_satisfaction_counterfactual": 0, "attempt_outcome": 0,
                                      "recovery_decision": 0, "corrective_action": 0,
                                      "accepted_temporal_windows": 0,
                                      "accepted_corrective_action_windows": 0},
            "coverage": coverage,
            "calibration_selection": policy,
            "derivation": {
                "schema_version": SCHEMA,
                "phase_mini_index_manifest_sha256": _sha256(phase_index / "manifest.json"),
                "phase_mini_index_inventory_seal_sha256": mini_inventory_sha256,
                "phase_selection_manifest_sha256": expected_phase_selection_manifest_sha256,
                "phase_queue_sha256": expected_phase_queue_sha256,
                "full_v3_index_manifest_sha256": expected_full_index_manifest_sha256,
                "full_v3_inventory_seal_sha256": full_inventory_sha256,
                "original_phase_payload_bytes_preserved": True,
                "diagnostic_subset_is_not_complete_source_coverage": True,
                "phase_observation_is_not_outcome": True,
                "no_student_or_action_or_outcome_or_recovery_release": True,
            },
            "publisher_sha256": _sha256(Path(__file__)),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(base.canonical_json(manifest).encode("utf-8") + b"\n")
        seal = {
            "schema_version": "memlite-event-inventory-seal-v1",
            "index_manifest_sha256": _sha256(manifest_path),
            "source_release_manifest_sha256": manifest["source_release_manifest_sha256"],
            "coverage_expectations_sha256": manifest["coverage_expectations_sha256"],
            "expected_payload_files": list(PAYLOAD_FILES),
            "payload_files": files,
        }
        seal_path = staging / "inventory_seal.json"
        seal_path.write_bytes(base.canonical_json(seal).encode("utf-8") + b"\n")
        for item in staging.iterdir():
            item.chmod(0o444)
        staging.chmod(0o555)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"manifest_sha256": _sha256(output / "manifest.json"),
            "inventory_seal_sha256": _sha256(output / "inventory_seal.json"),
            "event_candidates": EXPECTED_EVENT_COUNT,
            "source_groups": EXPECTED_EVENT_COUNT,
            "coverage_complete": False,
            "release_role": "annotation_calibration",
            "ready_for_training": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-index", type=Path, required=True)
    parser.add_argument("--expected-phase-index-inventory-seal-sha256", required=True)
    parser.add_argument("--full-index", type=Path, required=True)
    parser.add_argument("--expected-full-index-inventory-seal-sha256", required=True)
    parser.add_argument("--expected-full-index-manifest-sha256", required=True)
    parser.add_argument("--phase-selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-phase-selection-manifest-sha256", required=True)
    parser.add_argument("--phase-queue", type=Path, required=True)
    parser.add_argument("--expected-phase-queue-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = publish(
        phase_index=args.phase_index,
        expected_phase_index_inventory_seal_sha256=args.expected_phase_index_inventory_seal_sha256,
        full_index=args.full_index,
        expected_full_index_inventory_seal_sha256=args.expected_full_index_inventory_seal_sha256,
        expected_full_index_manifest_sha256=args.expected_full_index_manifest_sha256,
        phase_selection_manifest=args.phase_selection_manifest,
        expected_phase_selection_manifest_sha256=args.expected_phase_selection_manifest_sha256,
        phase_queue=args.phase_queue,
        expected_phase_queue_sha256=args.expected_phase_queue_sha256,
        output=args.output)
    print(base.canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
