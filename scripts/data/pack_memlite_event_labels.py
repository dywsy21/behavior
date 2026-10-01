"""Seal P107 event labels without turning candidates into a training release.

This publisher deliberately sits after the immutable event index and before any
future training integration.  It copies only compact metadata/actual controls
into a new directory, verifies every canonical id again, and always publishes
``ready_for_training=false``.  A later, independently reviewed authorization
is intentionally outside this tool's scope.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[2]
RELEASE_SCHEMA = "memlite-event-label-release-v1"
ANNOTATION_SCHEMA = "memlite-event-label-annotations-v1"
USAGE_ROLES = frozenset(("student_candidate", "annotation_calibration", "evaluation_only"))
PRIVATE_TOKENS = frozenset((
    "private", "pose", "predicate", "snapshot", "teacher", "future", "truth", "oracle",
    "simulator", "provider", "contact_state", "robot_state",
))


def _load_protocol() -> Any:
    path = REPO / "src/g05/data/memlite_event_protocol.py"
    spec = importlib.util.spec_from_file_location("p107_memlite_event_protocol_pack", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load P107 protocol: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_protocol = _load_protocol()
SCHEMA_VERSION = _protocol.SCHEMA_VERSION
ACTION_HORIZON = _protocol.ACTION_HORIZON
LABEL_KINDS = _protocol.LABEL_KINDS
canonical_json = _protocol.canonical_json
canonical_sha256 = _protocol.canonical_sha256
validate_event = _protocol.validate_event
validate_goal_satisfaction_view = _protocol.validate_goal_satisfaction_view
validate_attempt_outcome_view = _protocol.validate_attempt_outcome_view
validate_recovery_decision_view = _protocol.validate_recovery_decision_view
validate_corrective_action_view = _protocol.validate_corrective_action_view
actor_evidence_projection = _protocol.actor_evidence_projection
project_action_23_to_27 = _protocol.project_action_23_to_27
action_payload_sha256 = _protocol.action_payload_sha256
raw_action_payload_sha256 = _protocol.raw_action_payload_sha256
load_corrective_action_authority = _protocol.load_corrective_action_authority
authority_raw_action_payload = _protocol.authority_raw_action_payload


def strict_json_bytes(data: bytes, *, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number in {name}: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def read_json(path: Path) -> Any:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular input file is missing: {path}")
    return strict_json_bytes(path.read_bytes(), name=str(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular input file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        value = strict_json_bytes(line, name=f"{path}:{number}")
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {path}:{number} is not an object")
        rows.append(value)
    return rows


def _stream_sealed_jsonl(path: Path, receipt: Mapping[str, Any], *, name: str,
                         consume: Any) -> int:
    """Strictly parse a sealed JSONL file in bounded memory.

    The digest and byte count cover every byte read, including blank rows.  A
    caller may retain only a selected subset in ``consume``; this helper still
    parses every row so duplicate keys, malformed JSON, and malformed
    unselected index rows cannot evade the immutable-index validation.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular input file is missing: {path}")
    expected_sha, expected_bytes, expected_rows = receipt.get("sha256"), receipt.get("bytes"), receipt.get("rows")
    if not _is_sha(expected_sha) or type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError(f"event index receipt is malformed: {name}")
    if expected_rows is not None and (type(expected_rows) is not int or expected_rows < 0):
        raise ValueError(f"event index receipt is malformed: {name}")
    digest, byte_count, row_count = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            digest.update(line)
            byte_count += len(line)
            if not line.strip():
                continue
            value = strict_json_bytes(line, name=f"{path}:{number}")
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {path}:{number} is not an object")
            consume(value)
            row_count += 1
    if (digest.hexdigest() != expected_sha or byte_count != expected_bytes or
            (expected_rows is not None and row_count != expected_rows)):
        raise ValueError(f"event index receipt mismatch: {name}")
    return row_count


def _write_sealed_jsonl_copy(output: Path, source: Path, receipt: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Re-encode a fully sealed source JSONL without retaining its rows.

    This second pass closes the interval between index validation and package
    publication: a changed source is rejected before staging can be renamed.
    """
    count = 0
    with output.open("xb") as destination:
        def write_row(row: Mapping[str, Any]) -> None:
            nonlocal count
            destination.write(canonical_json(row).encode("utf-8") + b"\n")
            count += 1

        _stream_sealed_jsonl(source, receipt, name=name, consume=write_row)
    return {"sha256": sha256_file(output), "rows": count, "bytes": output.stat().st_size}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_json(row).encode("utf-8") + b"\n")
            count += 1
    return {"sha256": sha256_file(path), "rows": count, "bytes": path.stat().st_size}


def _assert_no_sidecar(value: Any, path: str = "annotations") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "sidecar" in str(key).casefold():
                raise ValueError(f"old MemLiteSidecar field is forbidden in P107 package: {path}.{key}")
            _assert_no_sidecar(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_sidecar(child, f"{path}[{index}]")


def _assert_actor_projection_safe(view: Mapping[str, Any]) -> dict[str, Any]:
    """Reject provider/private facts even when nested in an evidence reference."""
    evidence = actor_evidence_projection(view)

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                token = str(key).casefold()
                if any(bad in token for bad in PRIVATE_TOKENS):
                    raise ValueError(f"actor evidence leaks private/provider field at {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            token = value.casefold()
            if any(bad in token for bad in ("private pose", "oracle", "future truth", "simulator snapshot", "teacher action")):
                raise ValueError(f"actor evidence contains private/provider text at {path}")

    walk(evidence, "actor_evidence")
    return evidence


def _validate_view(view: Mapping[str, Any], event: Mapping[str, Any], *, authority: Any | None = None) -> None:
    kind = view.get("label_kind")
    validators = {
        "goal_satisfaction_counterfactual": validate_goal_satisfaction_view,
        "attempt_outcome": validate_attempt_outcome_view,
        "recovery_decision": validate_recovery_decision_view,
        "corrective_action": validate_corrective_action_view,
    }
    if kind not in validators:
        raise ValueError(f"unknown P107 label view: {kind}")
    if kind == "corrective_action":
        validators[kind](view, event, authority=authority)
    else:
        validators[kind](view, event)
    _assert_actor_projection_safe(view)


def _validate_source_groups(groups: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        group_id, role, split = group.get("source_group_id"), group.get("usage_role"), group.get("original_split")
        if not _is_sha(group_id) or role not in USAGE_ROLES or split not in {"train", "eval"}:
            raise ValueError("malformed immutable source-group inventory")
        if group_id in by_id:
            raise ValueError("duplicate source-group inventory row")
        if (split == "eval") != (role == "evaluation_only"):
            raise ValueError("source group attempts to change its inherited original split")
        if int(group.get("task_instance_id", 0)) >= 301 and role != "evaluation_only":
            raise ValueError("public_test source group cannot enter train or calibration")
        by_id[group_id] = dict(group)
    return by_id


def _validate_index_event(event: Mapping[str, Any], *, group_by_id: Mapping[str, Mapping[str, Any]],
                          seen_events: set[str], published_episode_identity: dict[int, tuple[Any, ...]]) -> None:
    validate_event(event)
    event_id, source = event["event_id"], event["source"]
    if event_id in seen_events:
        raise ValueError("duplicate event_id in package source")
    seen_events.add(event_id)
    episode_identity = (source["source_release_manifest_sha256"], source["task_index"],
                        source["task_instance_id"], source["raw_episode_id"])
    prior = published_episode_identity.setdefault(source["episode_index"], episode_identity)
    if prior != episode_identity:
        raise ValueError("published episode_index maps to conflicting immutable source episodes")
    group = group_by_id.get(source["source_group_id"])
    if group is None:
        raise ValueError("event has no immutable source-group inventory row")
    if event.get("usage_role") != group["usage_role"]:
        raise ValueError("event usage role differs from immutable source-group policy")
    for key in ("source_release_manifest_sha256", "task_index", "task_instance_id", "original_split"):
        if source[key] != group[key]:
            raise ValueError(f"event source {key} conflicts with inherited source-group provenance")


def _read_sealed_index(index_dir: Path, *, expected_inventory_seal_sha256: str,
                       selected_event_ids: set[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    index_dir = Path(index_dir)
    if index_dir.is_symlink() or not index_dir.is_dir():
        raise ValueError("event index must be a regular sealed directory")
    manifest = read_json(index_dir / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "memlite-event-index-v1":
        raise ValueError("not a P107 immutable event index")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("event index lacks sealed file receipts")
    paths = {"source_groups.jsonl": index_dir / "source_groups.jsonl",
             "event_candidates.jsonl": index_dir / "event_candidates.jsonl"}
    for name in paths:
        receipt = files.get(name)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"event index receipt mismatch: {name}")
    seal_path = index_dir / "inventory_seal.json"
    if not _is_sha(expected_inventory_seal_sha256):
        raise ValueError("publisher needs an externally recorded immutable index inventory seal SHA-256")
    actual_inventory_seal_sha256 = sha256_file(seal_path)
    if actual_inventory_seal_sha256 != expected_inventory_seal_sha256:
        raise ValueError("externally recorded immutable index inventory seal SHA-256 does not match input")
    seal = read_json(seal_path)
    expected_seal = {"schema_version": "memlite-event-inventory-seal-v1",
                     "index_manifest_sha256": sha256_file(index_dir / "manifest.json"),
                     "source_release_manifest_sha256": manifest.get("source_release_manifest_sha256"),
                     "coverage_expectations_sha256": manifest.get("coverage_expectations_sha256"),
                     "expected_payload_files": sorted(paths), "payload_files": files}
    if seal != expected_seal:
        raise ValueError("event index inventory seal does not bind exact manifest/payload files")
    groups: list[dict[str, Any]] = []
    _stream_sealed_jsonl(paths["source_groups.jsonl"], files["source_groups.jsonl"], name="source_groups.jsonl",
                         consume=groups.append)
    group_by_id = _validate_source_groups(groups)
    selected = None if selected_event_ids is None else set(selected_event_ids)
    if selected is not None and any(not _is_sha(event_id) for event_id in selected):
        raise ValueError("selected event IDs must be canonical SHA-256 values")
    events: list[dict[str, Any]] = []
    found_selected: set[str] = set()
    seen_events: set[str] = set()
    published_episode_identity: dict[int, tuple[Any, ...]] = {}

    def consume_event(event: dict[str, Any]) -> None:
        _validate_index_event(event, group_by_id=group_by_id, seen_events=seen_events,
                              published_episode_identity=published_episode_identity)
        if selected is None or event["event_id"] in selected:
            events.append(event)
            found_selected.add(event["event_id"])

    event_count = _stream_sealed_jsonl(paths["event_candidates.jsonl"], files["event_candidates.jsonl"],
                                        name="event_candidates.jsonl", consume=consume_event)
    if selected is not None and selected - found_selected:
        raise ValueError("selected event IDs are missing from immutable source index")
    if not groups or not event_count:
        raise ValueError("event index must contain source groups and event candidates")
    return groups, events, manifest, actual_inventory_seal_sha256


def _validate_group_inventory(groups: list[Mapping[str, Any]], events: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = _validate_source_groups(groups)
    seen_events: set[str] = set()
    published_episode_identity: dict[int, tuple[Any, ...]] = {}
    for event in events:
        _validate_index_event(event, group_by_id=by_id, seen_events=seen_events,
                              published_episode_identity=published_episode_identity)
    return by_id


def _sorted_unique_ints(value: Any, *, name: str, allow_empty: bool = True) -> list[int]:
    if (not isinstance(value, list) or (not allow_empty and not value) or
            any(type(item) is not int or item < 0 for item in value) or value != sorted(set(value))):
        raise ValueError(f"{name} must be a sorted unique nonnegative integer list")
    return list(value)


def _sorted_unique_strings(value: Any, *, name: str, allow_empty: bool = True) -> list[str]:
    if (not isinstance(value, list) or (not allow_empty and not value) or
            any(not isinstance(item, str) or not item for item in value) or value != sorted(set(value))):
        raise ValueError(f"{name} must be a sorted unique nonempty string list")
    return list(value)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _coverage_expectation_pin(coverage: Mapping[str, Any], expectations: Mapping[str, Any],
                              index_manifest: Mapping[str, Any]) -> str:
    expected_sha = index_manifest.get("coverage_expectations_sha256")
    if not _is_sha(expected_sha) or coverage.get("expectations_sha256") != expected_sha:
        raise ValueError("sealed event index coverage expectations pin is invalid")
    if canonical_sha256(expectations) != expected_sha:
        raise ValueError("sealed event index coverage expectations differ from their pin")
    return expected_sha


def _coverage_counts(coverage: Mapping[str, Any], index_manifest: Mapping[str, Any]) -> None:
    for report_field, manifest_field in (("unique_input_source_episodes", "source_episodes"),
                                         ("unique_event_candidates", "event_candidates")):
        report_count = _nonnegative_int(coverage.get(report_field), name=f"coverage.{report_field}")
        manifest_count = _nonnegative_int(index_manifest.get(manifest_field), name=f"index.{manifest_field}")
        if report_count != manifest_count:
            raise ValueError(f"sealed event index coverage {report_field} conflicts with immutable {manifest_field}")
    candidate_episodes = _nonnegative_int(
        coverage.get("unique_candidate_source_episodes"), name="coverage.unique_candidate_source_episodes")
    if candidate_episodes > coverage["unique_input_source_episodes"]:
        raise ValueError("coverage candidate source episodes exceed immutable input source episodes")


def _v3_pairs(value: Any, *, tasks: set[int], skill_ids: set[int], name: str,
              allow_none: bool) -> list[dict[str, int]] | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a declared task/skill pair list")
    if not isinstance(value, list):
        raise ValueError(f"{name} must be null or a task/skill pair list")
    result: list[dict[str, int]] = []
    previous: tuple[int, int] | None = None
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"task_index", "skill_id"}:
            raise ValueError(f"{name} has a malformed task/skill pair")
        task, skill = row["task_index"], row["skill_id"]
        if type(task) is not int or type(skill) is not int or task not in tasks or skill not in skill_ids:
            raise ValueError(f"{name} pair is outside the official task/skill contract")
        key = task, skill
        if previous is not None and key <= previous:
            raise ValueError(f"{name} pairs must be sorted unique")
        previous = key
        result.append({"task_index": task, "skill_id": skill})
    return result


def _normalize_legacy_coverage(coverage: Mapping[str, Any], expectations: Mapping[str, Any],
                               index_manifest: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "official_metadata_sha256", "expected_task_ids", "expected_skill_verbs"}
    if set(expectations) != required or expectations.get("schema_version") != "p107-official-coverage-expectations-v1":
        raise ValueError("sealed event index coverage does not use a supported legacy official contract")
    if not _is_sha(expectations.get("official_metadata_sha256")):
        raise ValueError("legacy coverage official metadata pin is invalid")
    tasks = _sorted_unique_ints(expectations["expected_task_ids"], name="legacy expected_task_ids", allow_empty=False)
    skills = _sorted_unique_strings(expectations["expected_skill_verbs"], name="legacy expected_skill_verbs",
                                    allow_empty=False)
    _coverage_expectation_pin(coverage, expectations, index_manifest)
    _coverage_counts(coverage, index_manifest)
    grid = coverage.get("found_grid")
    if not isinstance(grid, Mapping) or set(grid) != {str(task) for task in tasks}:
        raise ValueError("legacy coverage grid does not bind the official task contract")
    derived_missing: list[dict[str, Any]] = []
    for task in tasks:
        task_grid = grid.get(str(task))
        if not isinstance(task_grid, Mapping) or set(task_grid) != set(skills):
            raise ValueError("legacy coverage grid does not bind the official skill contract")
        for skill in skills:
            cell = task_grid[skill]
            if not isinstance(cell, Mapping) or set(cell) != {"event_candidates", "unique_source_episodes"}:
                raise ValueError("legacy coverage grid has an invalid count cell")
            _nonnegative_int(cell["event_candidates"], name="legacy coverage event_candidates")
            _nonnegative_int(cell["unique_source_episodes"], name="legacy coverage unique_source_episodes")
            if cell["event_candidates"] == 0:
                derived_missing.append({"task_index": task, "skill_verb": skill})
    missing_grid = coverage.get("missing_grid")
    if missing_grid != derived_missing:
        raise ValueError("legacy coverage missing grid conflicts with its sealed count grid")
    found_tasks = _sorted_unique_ints(coverage.get("found_task_ids"), name="legacy found_task_ids")
    unexpected_tasks = _sorted_unique_ints(coverage.get("unexpected_task_ids"), name="legacy unexpected_task_ids")
    found_skills = _sorted_unique_strings(coverage.get("found_skill_verbs"), name="legacy found_skill_verbs")
    unexpected_skills = _sorted_unique_strings(coverage.get("unexpected_skill_verbs"), name="legacy unexpected_skill_verbs")
    if (not set(found_tasks) <= set(tasks) or not set(found_skills) <= set(skills) or
            set(unexpected_tasks) & set(tasks) or set(unexpected_skills) & set(skills)):
        raise ValueError("legacy coverage marks official task or skill vocabulary as unexpected")
    complete = not derived_missing and not unexpected_tasks and not unexpected_skills
    if type(coverage.get("coverage_complete")) is not bool or coverage["coverage_complete"] != complete:
        raise ValueError("legacy coverage completeness conflicts with its sealed report")
    return {"schema": "p107-official-coverage-expectations-v1", "coverage_complete": complete,
            "diagnostic_only": False, "missing_pairs": derived_missing}


def _normalize_v3_coverage(coverage: Mapping[str, Any], expectations: Mapping[str, Any],
                           index_manifest: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "official_task_metadata_sha256", "official_skill_vocabulary_sha256",
                "expected_task_ids", "expected_skill_vocabulary", "required_task_skill_pairs"}
    if set(expectations) != required or expectations.get("schema_version") != "p107-official-coverage-expectations-v3":
        raise ValueError("sealed event index coverage does not use a supported v3 official contract")
    if (not _is_sha(expectations.get("official_task_metadata_sha256")) or
            not _is_sha(expectations.get("official_skill_vocabulary_sha256"))):
        raise ValueError("v3 coverage official source pins are invalid")
    if expectations["official_task_metadata_sha256"] != index_manifest.get("source_release_manifest_sha256"):
        raise ValueError("v3 coverage task metadata pin conflicts with immutable source release")
    tasks = _sorted_unique_ints(expectations["expected_task_ids"], name="v3 expected_task_ids", allow_empty=False)
    vocabulary = expectations["expected_skill_vocabulary"]
    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError("v3 coverage expected_skill_vocabulary must be nonempty")
    skill_ids: list[int] = []
    descriptions: set[str] = set()
    for skill in vocabulary:
        if not isinstance(skill, Mapping) or set(skill) != {"skill_id", "skill_description"}:
            raise ValueError("v3 coverage official skill vocabulary is malformed")
        skill_id, description = skill["skill_id"], skill["skill_description"]
        if (type(skill_id) is not int or skill_id < 0 or (skill_ids and skill_id <= skill_ids[-1]) or
                not isinstance(description, str) or not description or description in descriptions):
            raise ValueError("v3 coverage official skill vocabulary is not sorted unique")
        skill_ids.append(skill_id)
        descriptions.add(description)
    _coverage_expectation_pin(coverage, expectations, index_manifest)
    _coverage_counts(coverage, index_manifest)
    expected_tasks, expected_skills = set(tasks), set(skill_ids)
    found_tasks = _sorted_unique_ints(coverage.get("found_task_ids"), name="v3 found_task_ids")
    missing_tasks = _sorted_unique_ints(coverage.get("missing_task_ids"), name="v3 missing_task_ids")
    unexpected_tasks = _sorted_unique_ints(coverage.get("unexpected_task_ids"), name="v3 unexpected_task_ids")
    found_skills = _sorted_unique_ints(coverage.get("found_global_skill_ids"), name="v3 found_global_skill_ids")
    missing_skills = _sorted_unique_ints(coverage.get("missing_global_skill_ids"), name="v3 missing_global_skill_ids")
    unmapped_skills = _sorted_unique_ints(coverage.get("unmapped_source_skill_ids"), name="v3 unmapped_source_skill_ids")
    if (not set(missing_tasks) <= expected_tasks or not set(missing_skills) <= expected_skills or
            set(unexpected_tasks) & expected_tasks or set(unmapped_skills) & expected_skills or
            set(found_tasks) != expected_tasks - set(missing_tasks) or
            set(found_skills) != expected_skills - set(missing_skills)):
        raise ValueError("v3 coverage task/skill vocabulary report is inconsistent")
    source_members_missing = _nonnegative_int(
        coverage.get("source_skill_members_missing_skill_id"), name="v3 source_skill_members_missing_skill_id")
    derived_global_complete = not (missing_tasks or missing_skills or unexpected_tasks or unmapped_skills or
                                   source_members_missing)
    if type(coverage.get("global_vocabulary_complete")) is not bool or coverage["global_vocabulary_complete"] != derived_global_complete:
        raise ValueError("v3 coverage global vocabulary status is inconsistent")
    declared_pairs = _v3_pairs(coverage.get("required_task_skill_pairs"), tasks=expected_tasks,
                               skill_ids=expected_skills, name="v3 required_task_skill_pairs", allow_none=True)
    if declared_pairs != _v3_pairs(expectations["required_task_skill_pairs"], tasks=expected_tasks,
                                   skill_ids=expected_skills, name="v3 expectations.required_task_skill_pairs",
                                   allow_none=True):
        raise ValueError("v3 coverage required task/skill pairs differ from the official contract")
    missing_pairs = coverage.get("missing_required_task_skill_pairs")
    status = coverage.get("required_pair_coverage_status")
    if declared_pairs is None:
        if missing_pairs is not None or status != "NOT_DECLARED":
            raise ValueError("v3 coverage must retain null missing pairs and NOT_DECLARED status when pairs are absent")
        return {"schema": "p107-official-coverage-expectations-v3", "coverage_complete": False,
                "diagnostic_only": True, "missing_pairs": None}
    normalized_missing = _v3_pairs(missing_pairs, tasks=expected_tasks, skill_ids=expected_skills,
                                   name="v3 missing_required_task_skill_pairs", allow_none=False)
    if not set((row["task_index"], row["skill_id"]) for row in normalized_missing) <= set(
            (row["task_index"], row["skill_id"]) for row in declared_pairs):
        raise ValueError("v3 coverage missing task/skill pairs are not declared official pairs")
    expected_status = "COMPLETE" if not normalized_missing else "INCOMPLETE"
    if status != expected_status:
        raise ValueError("v3 coverage required pair status conflicts with its missing pair report")
    return {"schema": "p107-official-coverage-expectations-v3",
            "coverage_complete": bool(derived_global_complete and not normalized_missing),
            "diagnostic_only": False, "missing_pairs": normalized_missing}


def _normalize_index_coverage(index_manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not _is_sha(index_manifest.get("source_release_manifest_sha256")):
        raise ValueError("sealed event index source release pin is invalid")
    if type(index_manifest.get("partial_source_coverage")) is not bool:
        raise ValueError("sealed event index partial source coverage status is invalid")
    coverage = index_manifest.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("sealed event index lacks an explicit coverage/exclusion report")
    expectations = coverage.get("expectations")
    if not isinstance(expectations, Mapping):
        raise ValueError("sealed event index coverage lacks official expectations")
    schema = expectations.get("schema_version")
    if schema == "p107-official-coverage-expectations-v1":
        return _normalize_legacy_coverage(coverage, expectations, index_manifest)
    if schema == "p107-official-coverage-expectations-v3":
        return _normalize_v3_coverage(coverage, expectations, index_manifest)
    raise ValueError("sealed event index coverage uses an unsupported official report schema")


def _validate_review(review: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, Any]:
    required = ("view_id", "candidate_annotated", "parent_reviewed", "accepted_auxiliary",
                "outcome_validated", "reviewer")
    if any(key not in review for key in required):
        raise ValueError("each P107 view needs an explicit review/provenance record")
    if review["view_id"] != view["view_id"]:
        raise ValueError("review is bound to a different view_id")
    if any(type(review[key]) is not bool for key in required[1:5]):
        raise ValueError("review state booleans must be explicit")
    reviewer = review["reviewer"]
    if not isinstance(reviewer, Mapping) or reviewer.get("kind") not in {"agent", "human", "root_model"}:
        raise ValueError("reviewer provenance must identify agent, human, or root_model")
    provenance = view["annotation_provenance"].casefold()
    if provenance == "agent":
        if reviewer.get("kind") != "agent" or not isinstance(reviewer.get("model"), str) or not reviewer["model"]:
            raise ValueError("agent annotation must record its actual agent model, never human provenance")
    if provenance == "human" and reviewer.get("kind") != "human":
        raise ValueError("human annotation provenance must not be substituted by an agent")
    if review["outcome_validated"] and view["label_kind"] != "attempt_outcome":
        raise ValueError("only attempt_outcome can be marked outcome-validated")
    if view["review_status"] == "PARENT_APPROVED" and not review["parent_reviewed"]:
        raise ValueError("PARENT_APPROVED view needs a matching parent-review status record")
    return dict(review)


def _validate_action_payload(payload: Mapping[str, Any], view: Mapping[str, Any], *, authority: Any | None,
                             accept_packaged_positive_payload: bool = False) -> dict[str, Any]:
    """Copy controls from a pinned parent root for positives, never annotation input."""
    supplied_fields = {"view_id", "raw_actions_23", "raw_action_sha256", "raw_action_artifact_sha256"}
    stored_fields = supplied_fields | {"action_payload_sha256", "actions_27", "action_is_pad",
                                       "action_dim_is_pad"}
    if view["low_action_supervision_mask"]:
        if payload and not accept_packaged_positive_payload:
            raise ValueError("positive corrective controls must come only from the sealed parent publisher root")
        if authority is None:
            raise ValueError("positive corrective controls require a sealed parent publisher capability")
        sealed_payload = authority_raw_action_payload(authority, view["action_payload_sha256"])
        raw = sealed_payload["raw_actions_23"]
        raw_sha = sealed_payload["raw_action_sha256"]
        raw_artifact_sha = sealed_payload["raw_action_artifact_sha256"]
        if payload and ({"view_id": view["view_id"], "raw_actions_23": raw, "raw_action_sha256": raw_sha,
                         "raw_action_artifact_sha256": raw_artifact_sha,
                         "action_payload_sha256": sealed_payload["action_payload_sha256"]} !=
                         {key: payload.get(key) for key in ("view_id", "raw_actions_23", "raw_action_sha256",
                                                           "raw_action_artifact_sha256", "action_payload_sha256")}):
            raise ValueError("packaged positive action differs from the sealed parent publisher payload")
    else:
        expected_fields = stored_fields if accept_packaged_positive_payload else supplied_fields
        if set(payload) != expected_fields:
            raise ValueError("non-qualifying corrective payload must be an exact actual-23D schema; DART intended/noise controls are unsupported")
        if payload.get("view_id") != view["view_id"] or not isinstance(payload.get("raw_actions_23"), list):
            raise ValueError("non-qualifying corrective action payload must bind a view_id and actual raw 23D actions")
        raw = payload["raw_actions_23"]
        raw_artifact_sha = payload.get("raw_action_artifact_sha256")
        if not _is_sha(raw_artifact_sha):
            raise ValueError("non-qualifying corrective action payload must preserve its raw action artifact digest")
        raw_sha = raw_action_payload_sha256(raw, raw_action_artifact_sha256=raw_artifact_sha)
        if payload.get("raw_action_sha256") != raw_sha:
            raise ValueError("non-qualifying actual action payload hash is not bound to its source artifact")
    if len(raw) != view["actual_executed_length"]:
        raise ValueError("actual 23D payload length disagrees with corrective action receipt")
    # This owns no action mapping: it asks the protocol to enforce the R1Pro
    # 23D->27D placement and fixed 32-step shape/padding contract.
    projection = project_action_23_to_27(raw, view["actual_executed_length"])
    if not _is_sha(raw_sha) or raw_sha != view["executed_action_receipt"]["raw_action_sha256"]:
        raise ValueError("actual action payload hash disagrees with executed-action receipt")
    payload_sha = action_payload_sha256(raw)
    if view["action_payload_sha256"] != payload_sha:
        raise ValueError("actual action payload digest disagrees with corrective action view")
    result = {"view_id": view["view_id"], "raw_actions_23": raw, "raw_action_sha256": raw_sha,
              "raw_action_artifact_sha256": raw_artifact_sha, "action_payload_sha256": payload_sha,
              "actions_27": projection["actions_27"],
              "action_is_pad": projection["action_is_pad"],
              "action_dim_is_pad": projection["action_dim_is_pad"]}
    if accept_packaged_positive_payload and payload and payload != result:
        raise ValueError("stored corrective action projection/padding receipt changed")
    if len(result["actions_27"]) != ACTION_HORIZON or any(result["action_is_pad"][:len(raw)]):
        raise ValueError("projected corrective action lost its actual-vs-pad proof")
    return result


def _root_review_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"completed": False, "reason": "root visual-review receipt missing"}
    completed = value.get("completed") is True
    if completed and (value.get("reviewer_kind") != "root_model" or not isinstance(value.get("artifact"), str) or not value["artifact"]):
        raise ValueError("a completed root visual review needs root_model provenance and an artifact")
    return {"completed": completed, "artifact": value.get("artifact"), "reviewer_kind": value.get("reviewer_kind"),
            "model": value.get("model")}


def _authority_capability(value: Any, *, index_inventory_seal_sha256: str) -> dict[str, Any] | None:
    """Persist only externally pinned authority roots, never receipt rows."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("corrective authority capability must be a separately supplied publisher-root descriptor")
    required = {"publisher_manifest_sha256", "index_inventory_seal_sha256",
                "accepted_live_runtime_acceptance_root_sha256"}
    if set(value) != required or not _is_sha(value.get("publisher_manifest_sha256")):
        raise ValueError("corrective authority capability has an invalid externally pinned publisher manifest SHA")
    if value["index_inventory_seal_sha256"] != index_inventory_seal_sha256:
        raise ValueError("corrective authority capability must bind this immutable index inventory seal")
    live_root = value["accepted_live_runtime_acceptance_root_sha256"]
    if live_root is not None and not _is_sha(live_root):
        raise ValueError("corrective authority capability has an invalid live-runtime acceptance root SHA")
    return dict(value)


def _pilot_receipt(path: Path) -> dict[str, Any]:
    """Preserve the local 24-row visual pilot as calibration-only, never labels."""
    pilot = read_json(path)
    if not isinstance(pilot, Mapping) or pilot.get("schema_version") != "p107.visual_label.v1":
        raise ValueError("auxiliary pilot must be p107.visual_label.v1")
    if pilot.get("accepted_scope") != "visual_relation_calibration_only":
        raise ValueError("only the explicitly auxiliary visual-relation pilot may be attached")
    if pilot.get("human_reviewed") is not False or not isinstance(pilot.get("annotator_model"), str):
        raise ValueError("pilot must retain agent model provenance and human_reviewed=false")
    records = pilot.get("records")
    if not isinstance(records, list):
        raise ValueError("pilot records missing")
    for record in records:
        if record.get("split") != "train":
            raise ValueError("pilot calibration must not contain eval/public sources")
        for query in record.get("queries", []):
            if query.get("outcome_loss_mask") is not False or query.get("action_loss_mask") is not False:
                raise ValueError("visual calibration pilot must preserve zero outcome/action masks")
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "records": len(records),
            "annotator_model": pilot["annotator_model"], "accepted_scope": pilot["accepted_scope"],
            "release_eligibility": "AUXILIARY_CALIBRATION_ONLY", "stage3_training_authorized": False,
            "ambiguous_legacy_record_review_fields": sum(record.get("parent_review") == "PENDING" for record in records)}


def build_release(groups: list[Mapping[str, Any]], events: list[Mapping[str, Any]], annotations: Mapping[str, Any], *,
                  index_manifest: Mapping[str, Any], release_role: str, minimum_scale: Mapping[str, int] | None,
                  request_dataset_quality_eligibility: bool, auxiliary_pilots: Iterable[Path] = (),
                  require_complete_source_index: bool = False,
                  index_inventory_seal_sha256: str | None = None,
                  corrective_action_authority: Any | None = None,
                  corrective_authority_capability: Mapping[str, Any] | None = None,
                  source_indexed_count: int | None = None) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Validate a sealed package in memory; callers write it atomically afterwards."""
    if release_role not in USAGE_ROLES:
        raise ValueError("release role must be explicit")
    if not isinstance(annotations, Mapping) or annotations.get("schema_version") != ANNOTATION_SCHEMA:
        raise ValueError(f"annotations must declare {ANNOTATION_SCHEMA}")
    if set(annotations) != {"schema_version", "views", "view_reviews", "action_payloads", "root_visual_review"}:
        raise ValueError("candidate annotations must use the exact P107 annotation schema; queue/approval authority is external")
    _assert_no_sidecar(annotations)
    if "corrective_action_authority" in annotations:
        raise ValueError("parent corrective-action authority must be supplied separately from agent annotations")
    group_by_id = _validate_group_inventory(list(groups), list(events))
    if source_indexed_count is None:
        source_indexed_count = len(events)
    elif type(source_indexed_count) is not int or source_indexed_count < len(events):
        raise ValueError("source indexed count must cover every supplied immutable event")
    if not _is_sha(index_inventory_seal_sha256):
        raise ValueError("publisher must receive the sealed immutable index inventory digest")
    authority_capability = _authority_capability(corrective_authority_capability,
                                                 index_inventory_seal_sha256=index_inventory_seal_sha256)
    if (corrective_action_authority is None) != (authority_capability is None):
        raise ValueError("sealed parent authority and its externally pinned capability must be supplied together")
    views = annotations.get("views")
    reviews = annotations.get("view_reviews")
    payloads = annotations.get("action_payloads", [])
    if not isinstance(views, list) or not isinstance(reviews, list) or not isinstance(payloads, list):
        raise ValueError("annotations must have views, view_reviews, and action_payloads lists")
    event_by_id = {event["event_id"]: event for event in events}
    review_by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, Mapping) or review.get("view_id") in review_by_id:
            raise ValueError("duplicate or malformed view review")
        review_by_id[review["view_id"]] = dict(review)
    payload_by_id: dict[str, Mapping[str, Any]] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping) or payload.get("view_id") in payload_by_id:
            raise ValueError("duplicate or malformed corrective action payload")
        payload_by_id[payload["view_id"]] = payload
    sealed_views, sealed_actions, status = [], [], Counter()
    accepted_goal_outcome_windows: set[str] = set()
    accepted_corrective_windows: set[tuple[str, str]] = set()
    accepted_goal_outcome_source_episodes: set[tuple[str, int]] = set()
    accepted_decision_source_episodes: set[tuple[str, int]] = set()
    accepted_corrective_source_episodes: set[tuple[str, int]] = set()
    seen_views: set[str] = set()
    coverage: dict[str, dict[str, Any]] = {}
    for view in sorted(views, key=lambda row: str(row.get("view_id", ""))):
        if not isinstance(view, Mapping):
            raise ValueError("label view must be an object")
        event = event_by_id.get(view.get("event_id"))
        if event is None or view.get("view_id") in seen_views:
            raise ValueError("view must have one unique event-bound view_id")
        seen_views.add(view["view_id"])
        _validate_view(view, event, authority=corrective_action_authority)
        group = group_by_id[view["source_group_id"]]
        if group["usage_role"] != event["usage_role"]:
            raise ValueError("view cannot override its source-group usage role")
        review = _validate_review(review_by_id.pop(view["view_id"], {}), view)
        kind = view["label_kind"]
        quality_gates = {"goal_calibration": False, "outcome": False, "decision": False, "corrective_fm": False}
        student_train_group = group["usage_role"] == "student_candidate" and group["original_split"] == "train"
        if kind == "goal_satisfaction_counterfactual":
            quality_gates["goal_calibration"] = bool(
                student_train_group and view["valid_goal_mask"] and review["accepted_auxiliary"])
        elif kind == "attempt_outcome":
            quality_gates["outcome"] = bool(
                student_train_group and view["valid_result_mask"] and review["outcome_validated"])
        elif kind == "recovery_decision":
            quality_gates["decision"] = bool(
                student_train_group and view["decision_supervision_mask"] and review["accepted_auxiliary"])
        else:
            payload = _validate_action_payload(payload_by_id.pop(view["view_id"], {}), view,
                                               authority=corrective_action_authority)
            sealed_actions.append(payload)
            # A failed correction is retained in the package, but protocol
            # validation and this gate make it impossible to become FM BC.
            quality_gates["corrective_fm"] = bool(view["low_action_supervision_mask"] and student_train_group)
            status["non_qualifying_corrective_actions"] += int(not quality_gates["corrective_fm"])
        status["candidate_annotated"] += int(review["candidate_annotated"])
        status["parent_reviewed"] += int(review["parent_reviewed"])
        status["accepted_auxiliary"] += int(review["accepted_auxiliary"])
        status["outcome_validated"] += int(review["outcome_validated"])
        # Recovery-action verification is a protocol/authority conclusion,
        # never an annotation review checkbox.
        if quality_gates["goal_calibration"] or quality_gates["outcome"]:
            # A repeated counterfactual label at the same immutable event is
            # one observation window, never additional scale.
            accepted_goal_outcome_windows.add(event["event_id"])
            accepted_goal_outcome_source_episodes.add(
                (event["source"]["source_group_id"], event["source"]["episode_index"]))
        if quality_gates["decision"]:
            accepted_decision_source_episodes.add(
                (event["source"]["source_group_id"], event["source"]["episode_index"]))
        if quality_gates["corrective_fm"]:
            # Distinguish an actual executed window from copied/reworded views.
            accepted_corrective_windows.add((event["event_id"], view["executed_action_receipt"]["raw_action_sha256"]))
            accepted_corrective_source_episodes.add(
                (event["source"]["source_group_id"], event["source"]["episode_index"]))
        task = str(event["source"]["task_index"])
        local = coverage.setdefault(task, {"skills": set(), "instances": set(), "episodes": set(),
                                           "source_groups": set(), "candidate_views": 0,
                                           "quality_accepted": Counter()})
        local["skills"].update(skill.get("verb") for skill in event["skill_bundle"])
        local["instances"].add(event["source"]["task_instance_id"])
        local["episodes"].add(event["source"]["episode_index"])
        local["source_groups"].add(event["source"]["source_group_id"])
        local["candidate_views"] += 1
        for gate_name, enabled in quality_gates.items():
            local["quality_accepted"][gate_name] += int(enabled)
        sealed_views.append({"view": dict(view), "review": review, "usage_role": group["usage_role"],
                             "original_split": group["original_split"], "dataset_quality_gates": quality_gates})
    if review_by_id:
        raise ValueError("review refers to no packaged view")
    if payload_by_id:
        raise ValueError("action payload refers to no packaged corrective view")
    root_gate = _root_review_gate(annotations.get("root_visual_review"))
    minimum_scale = dict(minimum_scale or {})
    if request_dataset_quality_eligibility:
        required_minima = {"source_episodes", "goal_outcome_windows", "corrective_action_windows"}
        if set(minimum_scale) != required_minima or any(type(v) is not int or v < 1 for v in minimum_scale.values()):
            raise ValueError("dataset-quality eligibility requires explicit positive --minimum-source-episodes, --minimum-goal-outcome-windows, and --minimum-corrective-action-windows")
    status["verified_recovery_actions"] = len(accepted_corrective_windows)
    actual = {"source_episodes": len(accepted_goal_outcome_source_episodes),
              "goal_outcome_windows": len(accepted_goal_outcome_windows),
              "corrective_action_windows": len(accepted_corrective_windows),
              # These diagnostic counts intentionally do not satisfy the
              # goal/outcome source-episode minimum: decision-only and action-
              # only rows cannot make a sparse outcome set look broad.
              "decision_source_episodes": len(accepted_decision_source_episodes),
              "corrective_action_source_episodes": len(accepted_corrective_source_episodes)}
    scale_ok = bool(minimum_scale) and all(actual[name] >= value for name, value in minimum_scale.items())
    index_coverage = index_manifest.get("coverage")
    normalized_coverage = _normalize_index_coverage(index_manifest)
    if normalized_coverage["diagnostic_only"] and release_role != "annotation_calibration":
        raise ValueError("official coverage without declared task/skill pairs supports annotation_calibration only")
    missing_grid = normalized_coverage["missing_pairs"]
    source_index_complete = bool(require_complete_source_index and normalized_coverage["coverage_complete"] and
                                 not index_manifest.get("partial_source_coverage", True))
    # A source group cannot try to smuggle an otherwise valid action positive
    # through a later quality gate.  The attempt remains retained as
    # non-student evidence, never a corrective FM target.
    forbidden_student_positive = any(
        row["view"]["label_kind"] == "corrective_action" and row["view"]["low_action_supervision_mask"] and
        (row["usage_role"] != "student_candidate" or row["original_split"] != "train") for row in sealed_views)
    no_protected_student = not forbidden_student_positive and all(
        row["usage_role"] == "student_candidate" and row["original_split"] == "train"
        for row in sealed_views if row["dataset_quality_gates"].get("corrective_fm") or
        row["dataset_quality_gates"].get("outcome") or row["dataset_quality_gates"].get("decision"))
    dataset_quality_eligibility = bool(request_dataset_quality_eligibility and require_complete_source_index and
                                       source_index_complete and root_gate["completed"] and scale_ok and no_protected_student)
    pilots = [_pilot_receipt(Path(path)) for path in auxiliary_pilots]
    policy = index_manifest.get("calibration_selection")
    if not isinstance(policy, Mapping):
        raise ValueError("sealed event index has no calibration selection policy")
    policy_core = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy.get("policy_sha256") != canonical_sha256(policy_core):
        raise ValueError("sealed event index calibration policy hash is invalid")
    manifest = {
        "schema_version": RELEASE_SCHEMA, "protocol_schema_version": SCHEMA_VERSION,
        "release_role": release_role, "immutable_index_manifest_sha256": canonical_sha256(index_manifest),
        "immutable_index_inventory_seal_sha256": index_inventory_seal_sha256,
        "calibration_policy": policy, "calibration_policy_sha256": policy["policy_sha256"],
        "source_group_policy_sha256": canonical_sha256(sorted(groups, key=lambda row: row["source_group_id"])),
        "root_visual_review": root_gate, "minimum_scale_requested": minimum_scale,
        "actual_scale": actual, "require_complete_source_index": require_complete_source_index,
        "source_index_complete": source_index_complete, "source_index_coverage": index_coverage,
        "corrective_authority_capability": authority_capability,
        "dataset_quality_eligibility": dataset_quality_eligibility,
        "release_eligibility": ("DATASET_QUALITY_QUALIFIED_PENDING_INDEPENDENT_REVIEW"
                                if dataset_quality_eligibility else "CANDIDATE_ONLY"),
        # No caller can turn preparation into stage3 authorization by copying files.
        "ready_for_training": False, "training_run_authorized": False, "stage3_training_authorized": False,
        "status_report": {"source_indexed": source_indexed_count, "candidate_annotated": int(status["candidate_annotated"]),
                          "parent_reviewed": int(status["parent_reviewed"]), "accepted_auxiliary": int(status["accepted_auxiliary"]),
                          "outcome_validated": int(status["outcome_validated"]),
                          "verified_recovery_actions": int(status["verified_recovery_actions"]),
                          "non_qualifying_corrective_actions": int(status["non_qualifying_corrective_actions"]),
                          "auxiliary_visual_calibration_pilots": pilots},
        "coverage": {task: {"skills": sorted(values["skills"]), "source_instances": sorted(values["instances"]),
                             "source_episodes": sorted(values["episodes"]), "source_episode_count": len(values["episodes"]),
                             "source_group_count": len(values["source_groups"]), "candidate_view_count": values["candidate_views"],
                             "quality_accepted_view_counts": dict(sorted(values["quality_accepted"].items()))}
                     for task, values in sorted(coverage.items(), key=lambda item: int(item[0]))},
        "exclusions": {"protected_roles": ["annotation_calibration", "evaluation_only"],
                       "public_test": "never student", "source_group_atomic": True,
                       "old_memlite_sidecar": "not imported or accepted", "pilot": "auxiliary calibration only",
                       "partial_index": bool(index_manifest.get("partial_source_coverage", True)),
                       "official_coverage_missing_grid": missing_grid},
    }
    return manifest, {"source_groups": [dict(row) for row in groups], "events": [dict(row) for row in events],
                      "views": sealed_views, "actions": sealed_actions}


def publish(index_dir: Path, annotations_path: Path, output: Path, *, release_role: str,
            minimum_scale: Mapping[str, int] | None, request_dataset_quality_eligibility: bool,
            expected_index_inventory_seal_sha256: str,
            corrective_publisher_root: Path | None = None,
            expected_corrective_publisher_manifest_sha256: str | None = None,
            expected_live_runtime_acceptance_root_sha256: str | None = None,
            auxiliary_pilots: Iterable[Path] = (), require_complete_source_index: bool = False) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError("output exists; immutable packages are never appended or overwritten")
    if Path(index_dir).resolve() in output.resolve(strict=False).parents:
        raise ValueError("package output may not be inside its immutable source index")
    annotations = read_json(annotations_path)
    annotation_views = annotations.get("views") if isinstance(annotations, Mapping) else None
    if not isinstance(annotation_views, list):
        raise ValueError("annotations must have views, view_reviews, and action_payloads lists")
    selected_event_ids: set[str] = set()
    for view in annotation_views:
        if not isinstance(view, Mapping) or not _is_sha(view.get("event_id")):
            raise ValueError("annotations must contain event-bound canonical label views")
        selected_event_ids.add(view["event_id"])
    groups, events, index_manifest, index_inventory_seal_sha256 = _read_sealed_index(
        index_dir, expected_inventory_seal_sha256=expected_index_inventory_seal_sha256,
        selected_event_ids=selected_event_ids)
    if (corrective_publisher_root is None) != (expected_corrective_publisher_manifest_sha256 is None):
        raise ValueError("corrective publisher root and its externally pinned manifest SHA must be supplied together")
    corrective_action_authority = None
    corrective_authority_capability = None
    if corrective_publisher_root is not None:
        corrective_action_authority = load_corrective_action_authority(
            corrective_publisher_root,
            index_root=Path(index_dir),
            expected_publisher_manifest_sha256=expected_corrective_publisher_manifest_sha256,
            expected_index_inventory_seal_sha256=index_inventory_seal_sha256,
            expected_live_runtime_acceptance_root_sha256=expected_live_runtime_acceptance_root_sha256)
        corrective_authority_capability = {
            "publisher_manifest_sha256": expected_corrective_publisher_manifest_sha256,
            "index_inventory_seal_sha256": index_inventory_seal_sha256,
            "accepted_live_runtime_acceptance_root_sha256": expected_live_runtime_acceptance_root_sha256,
        }
    event_receipt = index_manifest["files"]["event_candidates.jsonl"]
    event_count = event_receipt.get("rows") if isinstance(event_receipt, Mapping) else None
    if type(event_count) is not int or event_count < 1:
        raise ValueError("event index receipt must state a positive event row count")
    manifest, records = build_release(groups, events, annotations, index_manifest=index_manifest,
                                      release_role=release_role, minimum_scale=minimum_scale,
                                      request_dataset_quality_eligibility=request_dataset_quality_eligibility,
                                      auxiliary_pilots=auxiliary_pilots,
                                      require_complete_source_index=require_complete_source_index,
                                      index_inventory_seal_sha256=index_inventory_seal_sha256,
                                      corrective_action_authority=corrective_action_authority,
                                      corrective_authority_capability=corrective_authority_capability,
                                      source_indexed_count=event_count)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        files = {"source_groups.jsonl": _write_jsonl(staging / "source_groups.jsonl", records["source_groups"]),
                 "events.jsonl": _write_sealed_jsonl_copy(
                     staging / "events.jsonl", Path(index_dir) / "event_candidates.jsonl", event_receipt,
                     name="event_candidates.jsonl"),
                 "views.jsonl": _write_jsonl(staging / "views.jsonl", records["views"]),
                 "actions.jsonl": _write_jsonl(staging / "actions.jsonl", records["actions"])}
        manifest["files"] = files
        manifest["publisher_sha256"] = sha256_file(Path(__file__))
        manifest_path = staging / "release_manifest.json"
        manifest_path.write_bytes(canonical_json(manifest).encode("utf-8") + b"\n")
        seal = {"schema_version": "memlite-event-label-release-seal-v1",
                "release_manifest_sha256": sha256_file(manifest_path), "files": files,
                "publisher_sha256": manifest["publisher_sha256"]}
        seal_path = staging / "release_seal.json"
        seal_path.write_bytes(canonical_json(seal).encode("utf-8") + b"\n")
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**manifest, "release_seal_sha256": sha256_file(output / "release_seal.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-index", type=Path, required=True, help="sealed build_memlite_event_index output")
    parser.add_argument("--annotations", type=Path, required=True, help="small P107 label/review JSON")
    parser.add_argument("--output", type=Path, required=True, help="new external immutable package directory")
    parser.add_argument("--release-role", choices=sorted(USAGE_ROLES), required=True)
    parser.add_argument("--expected-index-inventory-seal-sha256", required=True,
                        help="externally recorded inventory seal SHA-256; self-consistent input is insufficient")
    parser.add_argument("--corrective-publisher-root", type=Path,
                        help="separate parent-owned sealed publisher root; required for a positive corrective FM mask")
    parser.add_argument("--expected-corrective-publisher-manifest-sha256",
                        help="externally pinned parent publisher-manifest SHA-256")
    parser.add_argument("--expected-live-runtime-acceptance-root-sha256",
                        help="externally accepted live-runtime root for GOLD, omitted for SILVER-only authority")
    parser.add_argument("--request-dataset-quality-eligibility", action="store_true",
                        help="request an evidence-quality gate only; it never authorizes a training run")
    parser.add_argument("--require-complete-source-index", action="store_true",
                        help="require the index's explicit official coverage contract to have no gaps")
    parser.add_argument("--minimum-source-episodes", type=int)
    parser.add_argument("--minimum-goal-outcome-windows", type=int)
    parser.add_argument("--minimum-corrective-action-windows", type=int)
    parser.add_argument("--auxiliary-pilot", type=Path, action="append", default=[],
                        help="agent visual pilot, retained only as auxiliary calibration provenance")
    args = parser.parse_args()
    supplied = (args.minimum_source_episodes, args.minimum_goal_outcome_windows, args.minimum_corrective_action_windows)
    if any(value is not None for value in supplied) and any(value is None for value in supplied):
        parser.error("all three --minimum-* arguments must be supplied together")
    minima = None if supplied[0] is None else {"source_episodes": supplied[0], "goal_outcome_windows": supplied[1],
                                                "corrective_action_windows": supplied[2]}
    result = publish(args.event_index, args.annotations, args.output, release_role=args.release_role,
                     minimum_scale=minima, request_dataset_quality_eligibility=args.request_dataset_quality_eligibility,
                     expected_index_inventory_seal_sha256=args.expected_index_inventory_seal_sha256,
                     corrective_publisher_root=args.corrective_publisher_root,
                     expected_corrective_publisher_manifest_sha256=args.expected_corrective_publisher_manifest_sha256,
                     expected_live_runtime_acceptance_root_sha256=args.expected_live_runtime_acceptance_root_sha256,
                     auxiliary_pilots=args.auxiliary_pilot,
                     require_complete_source_index=args.require_complete_source_index)
    print(canonical_json({"release_eligibility": result["release_eligibility"], "ready_for_training": False,
                          "release_seal_sha256": result["release_seal_sha256"],
                          "status_report": result["status_report"]}), flush=True)


if __name__ == "__main__":
    main()
