"""Build an immutable P107 metadata-only event index from a frozen v4 release.

This builder reads only ``manifest.json``, ``episodes.jsonl`` and optional
split provenance.  It neither opens RGB/video/action arrays nor writes into
the release.  Segment boundaries become review candidates only: no outcome,
recovery, or action supervision is inferred from an annotation end.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[2]


def _load_protocol() -> Any:
    """Avoid importing g05.data.__init__, which intentionally has ML deps."""
    path = REPO / "src/g05/data/memlite_event_protocol.py"
    spec = importlib.util.spec_from_file_location("p107_memlite_event_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load P107 protocol: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_protocol = _load_protocol()
ACTION_DIM = _protocol.ACTION_DIM
MODEL_ACTION_DIM = _protocol.MODEL_ACTION_DIM
MODEL_PADDING_INDICES = _protocol.MODEL_PADDING_INDICES
SCHEMA_VERSION = _protocol.SCHEMA_VERSION
canonical_json = _protocol.canonical_json
canonical_sha256 = _protocol.canonical_sha256
event_id = _protocol.event_id
source_group_id = _protocol.source_group_id
validate_event = _protocol.validate_event
validate_source_ref = _protocol.validate_source_ref


CAMERAS = (
    ("head", "observation.rgb.zed_link_camera_0"),
    ("left_wrist", "observation.rgb.left_realsense_link_camera_0"),
    ("right_wrist", "observation.rgb.right_realsense_link_camera_0"),
)
INDEX_SCHEMA = "memlite-event-index-v1"
INVENTORY_SEAL_SCHEMA = "memlite-event-inventory-seal-v1"
INDEX_PAYLOAD_FILES = frozenset(("source_groups.jsonl", "event_candidates.jsonl"))


def strict_json_bytes(data: bytes, *, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON number in {name}: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular input file is missing: {path}")
    return strict_json_bytes(path.read_bytes(), name=str(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular input file is missing: {path}")
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if line.strip():
            value = strict_json_bytes(line, name=f"{path}:{number}")
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {path}:{number} is not an object")
            rows.append(value)
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream a frozen JSONL source after its full-file SHA was pinned."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular input file is missing: {path}")
    with path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            if line.strip():
                value = strict_json_bytes(line, name=f"{path}:{number}")
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {path}:{number} is not an object")
                yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forbid_source_output(output: Path, source: Path) -> None:
    resolved_output = output.resolve(strict=False)
    resolved_source = source.resolve(strict=True)
    if resolved_output == resolved_source or resolved_source in resolved_output.parents:
        raise ValueError("Output must be outside the immutable source release")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def source_from_episode(record: Mapping[str, Any], release_manifest_sha256: str) -> dict[str, Any]:
    row = record.get("row")
    if not isinstance(row, Mapping):
        raise ValueError("v4 episode record has no raw metadata row")
    source = {
        "source_release_manifest_sha256": release_manifest_sha256,
        "source_annotation_sha256": record.get("annotation_sha256"),
        "task_index": row.get("task_index"),
        "task_instance_id": row.get("task_instance_id"),
        "raw_episode_id": row.get("raw_episode_id"),
        "episode_index": row.get("episode_index"),
        "original_split": record.get("split"),
        "episode_length": row.get("length"),
    }
    source["source_group_id"] = source_group_id(source)
    validate_source_ref(source)
    if source["task_instance_id"] >= 301:
        raise ValueError("public-test source instance cannot enter P107 inventory")
    return source


def _video_locators(row: Mapping[str, Any], frame: int) -> list[dict[str, Any]]:
    """Emit metadata locators only; a renderer resolves and pins real files later."""
    locators = []
    for view, camera in CAMERAS:
        prefix = f"videos/{camera}"
        chunk, file, origin = (row.get(prefix + "/chunk_index"), row.get(prefix + "/file_index"),
                               row.get(prefix + "/from_timestamp"))
        if type(chunk) is not int or type(file) is not int or type(origin) not in (int, float):
            continue
        if isinstance(origin, bool):
            continue
        locators.append({
            "view": view,
            "camera_key": camera,
            "relative_path": f"videos/{camera}/chunk-{chunk:03d}/file-{file:03d}.mp4",
            "episode_start_timestamp_s": float(origin),
            "requested_timestamp_s": float(origin) + frame / 30.0,
            "expected_fps": 30,
            "locator_status": "METADATA_ONLY_UNRESOLVED",
        })
    return locators


def _bundle_id(segment: Mapping[str, Any]) -> str:
    # Raw members remain audit data.  The identity also includes source-facing
    # semantic text so different source bundles at identical boundaries cannot
    # silently collide.
    return canonical_sha256({
        "semantic": segment.get("semantic"),
        "text": segment.get("text"),
        "skills": segment.get("skills"),
    })


def event_from_segment(record: Mapping[str, Any], segment: Mapping[str, Any],
                       release_manifest_sha256: str) -> dict[str, Any]:
    source = source_from_episode(record, release_manifest_sha256)
    start = _integer(segment.get("start"), "segment.start")
    end = _integer(segment.get("end"), "segment.end")
    if end <= start or end > source["episode_length"]:
        raise ValueError("v4 segment is outside source episode clock")
    skills = segment.get("skills")
    if not isinstance(skills, list) or not skills:
        raise ValueError("eligible v4 segment has no source skill bundle")
    bundle_id = _bundle_id(segment)
    event = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "event_candidate",
        "event_id": "",
        "source": source,
        "event_kind": "ANNOTATED_SKILL_SEGMENT",
        "event_interval": {"start_frame": start, "end_frame": end},
        "observation": {"frame": start, "timestamp_s": start / 30.0},
        # No action sequence was loaded.  The dimensions/mapping are recorded
        # now so later action export cannot change embodiment silently.
        "action": {"start_frame": start, "actual_executed_length": None,
                   "raw_action_dim": ACTION_DIM, "model_action_dim": MODEL_ACTION_DIM,
                   "model_padding_indices": list(MODEL_PADDING_INDICES)},
        "bundle_id": bundle_id,
        "skill_bundle": skills,
        "parallel_bundle": len(skills) > 1,
        "binding": [{key: skill.get(key) for key in ("verb", "target", "source", "destination",
                    "target_part", "arm", "binding_confidence")} for skill in skills],
        # This is intentionally MISSING: segment end and annotation duration
        # are not physical outcome evidence.
        "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "video_locators": _video_locators(record["row"], start),
        "task_name": record.get("task_name"),
    }
    event["event_id"] = event_id(event)
    validate_event(event)
    return event


def _calibration_policy(release_manifest_sha256: str, calibration_per_task: int,
                        explicit_group_ids: set[str]) -> dict[str, Any]:
    core = {
        "algorithm": "p107-calibration-v1",
        "source_release_manifest_sha256": release_manifest_sha256,
        "deterministic_per_task": calibration_per_task,
        "explicit_group_ids": sorted(explicit_group_ids),
    }
    return {**core, "policy_sha256": canonical_sha256(core)}


def _role_for_groups(sources: list[dict[str, Any]], calibration_per_task: int,
                     explicit_group_ids: set[str]) -> dict[str, str]:
    by_task: dict[int, list[str]] = defaultdict(list)
    by_id = {source["source_group_id"]: source for source in sources}
    unknown = explicit_group_ids - set(by_id)
    if unknown:
        raise ValueError("explicit calibration source_group_id is absent from this frozen release")
    if any(by_id[group]["original_split"] != "train" for group in explicit_group_ids):
        raise ValueError("explicit calibration source_group_id must be an immutable TRAIN group")
    for source in sources:
        if source["original_split"] == "train":
            by_task[source["task_index"]].append(source["source_group_id"])
    roles: dict[str, str] = {}
    for source in sources:
        roles[source["source_group_id"]] = "evaluation_only" if source["original_split"] == "eval" else "student_candidate"
    for task, groups in by_task.items():
        # Never turn a task's sole train source into calibration.  The output
        # records the resulting coverage gap rather than inventing a split.
        selected = sorted(set(groups), key=lambda group: hashlib.sha256(
            f"p107-calibration-v1:{task}:{group}".encode()).digest())[:max(0, min(calibration_per_task, len(set(groups)) - 1))]
        for group in selected:
            roles[group] = "annotation_calibration"
    for group in explicit_group_ids:
        roles[group] = "annotation_calibration"
    return roles


def _source_group_rows(sources: list[dict[str, Any]], roles: Mapping[str, str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        grouped[source["source_group_id"]].append(source)
    rows = []
    for group_id, members in sorted(grouped.items()):
        first = members[0]
        if any((member["original_split"] != first["original_split"] or
                member["task_index"] != first["task_index"] or
                member["task_instance_id"] != first["task_instance_id"]) for member in members):
            raise ValueError("source-group split/task/instance collision")
        role = roles[group_id]
        if role not in {"student_candidate", "annotation_calibration", "evaluation_only"}:
            raise ValueError("unknown usage role")
        if (first["original_split"] == "eval") != (role == "evaluation_only"):
            raise ValueError("evaluation group cannot be relabeled train/calibration")
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "source_group_id": group_id,
            "source_release_manifest_sha256": first["source_release_manifest_sha256"],
            "task_index": first["task_index"], "task_instance_id": first["task_instance_id"],
            "original_split": first["original_split"], "usage_role": role,
            "source_episode_ids": sorted(member["episode_index"] for member in members),
            "source_episode_count": len(members),
        })
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_json(row).encode("utf-8") + b"\n")
            count += 1
    return {"sha256": sha256_file(path), "rows": count, "bytes": path.stat().st_size}


def validate_coverage_expectations(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate global official task/verb vocabulary without inventing task×verb requirements."""
    if not isinstance(value, Mapping):
        raise ValueError("coverage_expectations must be an object")
    required = {"schema_version", "official_task_metadata_sha256", "official_skill_vocabulary_sha256",
                "expected_task_ids", "expected_skill_verbs", "required_task_skill_pairs"}
    if set(value) != required or value["schema_version"] != "p107-official-coverage-expectations-v2":
        raise ValueError("coverage expectations must use the exact P107 official-metadata schema")
    _sha(value["official_task_metadata_sha256"], "coverage_expectations.official_task_metadata_sha256")
    _sha(value["official_skill_vocabulary_sha256"], "coverage_expectations.official_skill_vocabulary_sha256")
    tasks, skills = value["expected_task_ids"], value["expected_skill_verbs"]
    if (not isinstance(tasks, list) or not tasks or any(type(task) is not int or task < 0 for task in tasks) or
            len(set(tasks)) != len(tasks) or tasks != sorted(tasks)):
        raise ValueError("expected_task_ids must be a nonempty sorted unique integer list")
    if (not isinstance(skills, list) or not skills or any(not isinstance(skill, str) or not skill for skill in skills) or
            len(set(skills)) != len(skills) or skills != sorted(skills)):
        raise ValueError("expected_skill_verbs must be a nonempty sorted unique string list")
    required_pairs = value["required_task_skill_pairs"]
    if required_pairs is not None:
        if not isinstance(required_pairs, list):
            raise ValueError("required_task_skill_pairs must be null or a sorted unique official pair list")
        previous: tuple[int, str] | None = None
        for pair in required_pairs:
            if not isinstance(pair, Mapping) or set(pair) != {"task_index", "skill_verb"}:
                raise ValueError("required task/skill pairs must be exact objects")
            task, skill = pair["task_index"], pair["skill_verb"]
            if task not in tasks or skill not in skills:
                raise ValueError("required task/skill pair is outside the official global vocabulary")
            key = (task, skill)
            if previous is not None and key <= previous:
                raise ValueError("required task/skill pairs must be sorted and unique")
            previous = key
    return dict(value)


def read_coverage_expectations(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("coverage expectations must be a regular official-metadata contract file")
    return validate_coverage_expectations(read_json(path))


def _coverage(events: list[Mapping[str, Any]], sources: list[Mapping[str, Any]],
              expectations: Mapping[str, Any]) -> dict[str, Any]:
    expectations = validate_coverage_expectations(expectations)
    tasks, skills = expectations["expected_task_ids"], expectations["expected_skill_verbs"]
    grid: dict[str, dict[str, dict[str, int]]] = {
        str(task): {skill: {"event_candidates": 0, "unique_source_episodes": 0} for skill in skills}
        for task in tasks
    }
    episode_sets: dict[tuple[int, str], set[int]] = defaultdict(set)
    observed_tasks, observed_skills = set(), set()
    for event in events:
        task = event["source"]["task_index"]
        observed_tasks.add(task)
        for skill in event["skill_bundle"]:
            verb = str(skill.get("verb", "MISSING_VERB"))
            observed_skills.add(verb)
            if task in tasks and verb in skills:
                grid[str(task)][verb]["event_candidates"] += 1
                episode_sets[(task, verb)].add(event["source"]["episode_index"])
    for (task, verb), episodes in episode_sets.items():
        grid[str(task)][verb]["unique_source_episodes"] = len(episodes)
    required_pairs = expectations["required_task_skill_pairs"]
    missing_required_pairs = (None if required_pairs is None else [
        pair for pair in required_pairs
        if grid[str(pair["task_index"])][pair["skill_verb"]]["event_candidates"] == 0
    ])
    expected_task_set, expected_skill_set = set(tasks), set(skills)
    return {
        "expectations": expectations,
        "expectations_sha256": canonical_sha256(expectations),
        # This matrix is observed source coverage, not an assertion that every
        # verb must occur in every task.  An absent pair only becomes a failed
        # requirement when it was explicitly supplied by official metadata.
        "observed_task_skill_grid": grid,
        "required_task_skill_pairs": required_pairs,
        "missing_required_task_skill_pairs": missing_required_pairs,
        "required_pair_coverage_status": (
            "NOT_DECLARED" if required_pairs is None
            else ("COMPLETE" if not missing_required_pairs else "INCOMPLETE")
        ),
        "found_task_ids": sorted(observed_tasks & expected_task_set),
        "missing_task_ids": sorted(expected_task_set - observed_tasks),
        "unexpected_task_ids": sorted(observed_tasks - expected_task_set),
        "found_skill_verbs": sorted(observed_skills & expected_skill_set),
        "missing_global_skill_verbs": sorted(expected_skill_set - observed_skills),
        "unexpected_skill_verbs": sorted(observed_skills - expected_skill_set),
        # Never call global presence a completed task×skill coverage matrix.
        "global_vocabulary_complete": not (expected_task_set - observed_tasks) and not (expected_skill_set - observed_skills),
        "unique_input_source_episodes": len({source["episode_index"] for source in sources}),
        "unique_candidate_source_episodes": len({event["source"]["episode_index"] for event in events}),
        "unique_event_candidates": len({event["event_id"] for event in events}),
    }


def _input_files(release: Path) -> dict[str, str]:
    paths = {"manifest.json": release / "manifest.json", "episodes.jsonl": release / "episodes.jsonl",
             "split_provenance.json": release / "split_provenance.json"}
    provenance = release / "split_provenance.json"
    if provenance.is_symlink() or not provenance.is_file():
        raise ValueError("split_provenance.json must be a regular immutable v4 input")
    return {name: sha256_file(path) for name, path in paths.items()}


def _validate_release_split_provenance(release: Path, sources: list[Mapping[str, Any]]) -> None:
    provenance = read_json(release / "split_provenance.json")
    assignments = provenance.get("assignments") if isinstance(provenance, Mapping) else None
    if not isinstance(assignments, list):
        raise ValueError("v4 split_provenance.json must contain source-instance assignments")
    assigned: dict[tuple[int, int], str] = {}
    for row in assignments:
        if not isinstance(row, Mapping):
            raise ValueError("split assignment must be an object")
        task, instance, split = row.get("task"), row.get("instance"), row.get("split")
        if type(task) is not int or type(instance) is not int or split not in {"train", "eval"}:
            raise ValueError("malformed v4 split assignment")
        key = task, instance
        if key in assigned and assigned[key] != split:
            raise ValueError("v4 split provenance has a source-instance collision")
        assigned[key] = split
    for source in sources:
        key = source["task_index"], source["task_instance_id"]
        if assigned.get(key) != source["original_split"]:
            raise ValueError("episode source split does not match immutable v4 source-instance assignment")


def _check_timeout(started: float, max_seconds: float) -> None:
    if time.monotonic() - started > max_seconds:
        raise TimeoutError("metadata index CPU budget reached; no release was written")


def _selection_policy(max_source_episodes: int | None) -> dict[str, Any]:
    if max_source_episodes is not None and (type(max_source_episodes) is not int or max_source_episodes < 1):
        raise ValueError("max_source_episodes must be a positive integer when supplied")
    core = {"schema_version": "p107-index-selection-v1", "mode": "PUBLISHED_EPISODE_PREFIX",
            "max_source_episodes": max_source_episodes, "partial": max_source_episodes is not None}
    return {**core, "policy_sha256": canonical_sha256(core)}


def read_calibration_group_ids(path: Path | None, release_manifest_sha256: str) -> set[str]:
    """Read predeclared group IDs or task/instance reservations without labels.

    A line is either a 64-char ``source_group_id`` or an exact JSON object
    ``{"task_index": N, "task_instance_id": N}``.  The latter is converted
    under the frozen release hash before the policy itself is sealed.
    """
    if path is None:
        return set()
    if path.is_symlink() or not path.is_file():
        raise ValueError("calibration group reservation file must be a regular file")
    selected: set[str] = set()
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        text = line.decode("utf-8")
        if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
            group_id = text
        else:
            row = strict_json_bytes(line, name=f"{path}:{number}")
            if not isinstance(row, dict) or set(row) != {"task_index", "task_instance_id"}:
                raise ValueError("calibration reservation must be a source_group_id or exact task/instance object")
            if type(row["task_index"]) is not int or type(row["task_instance_id"]) is not int:
                raise ValueError("calibration task/instance reservation must use integers")
            group_id = source_group_id({"source_release_manifest_sha256": release_manifest_sha256,
                                        "task_index": row["task_index"],
                                        "task_instance_id": row["task_instance_id"]})
        if group_id in selected:
            raise ValueError("duplicate explicit calibration source group")
        selected.add(group_id)
    return selected


def build_index(release: Path, output: Path, *, calibration_per_task: int = 1,
                calibration_group_ids: set[str] | None = None,
                coverage_expectations: Mapping[str, Any] | None = None,
                max_source_episodes: int | None = None,
                max_seconds: float = 1800.0) -> dict[str, Any]:
    """Create a sealed index.  Inputs must be a v4 compact release layout."""
    release, output = Path(release), Path(output)
    if calibration_per_task < 0 or type(calibration_per_task) is not int:
        raise ValueError("calibration_per_task must be a nonnegative integer")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if output.exists():
        raise FileExistsError("output already exists; use --resume only to verify a sealed identical index")
    if release.is_symlink() or not release.is_dir():
        raise ValueError("release must be an existing non-symlink directory")
    forbid_source_output(output, release)
    started = time.monotonic()
    manifest_path = release / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("release manifest must be an object")
    release_manifest_sha256 = sha256_file(manifest_path)
    calibration_group_ids = set() if calibration_group_ids is None else set(calibration_group_ids)
    calibration_selection = _calibration_policy(release_manifest_sha256, calibration_per_task, calibration_group_ids)
    selection_policy = _selection_policy(max_source_episodes)
    if coverage_expectations is None:
        raise ValueError("explicit official coverage expectations are required; no hardcoded 100-task fallback")
    coverage_expectations = validate_coverage_expectations(coverage_expectations)
    inputs = _input_files(release)
    sources, events, seen_raw_episode, seen_published_episode, seen_event = [], [], set(), set(), set()
    selected_records = 0
    previous_published_episode_index = -1
    for record in iter_jsonl(release / "episodes.jsonl"):
        _check_timeout(started, max_seconds)
        if max_source_episodes is not None and selected_records >= max_source_episodes:
            break
        source = source_from_episode(record, release_manifest_sha256)
        raw_episode_key = (source["task_index"], source["raw_episode_id"], source["task_instance_id"])
        if raw_episode_key in seen_raw_episode:
            raise ValueError("duplicate raw source episode identity in release")
        if source["episode_index"] in seen_published_episode:
            raise ValueError("duplicate published episode_index in release")
        if source["episode_index"] <= previous_published_episode_index:
            raise ValueError("episodes.jsonl must be strictly ordered by published episode_index for prefix selection")
        previous_published_episode_index = source["episode_index"]
        seen_raw_episode.add(raw_episode_key)
        seen_published_episode.add(source["episode_index"])
        sources.append(source)
        selected_records += 1
        segments = record.get("segments")
        if not isinstance(segments, list):
            raise ValueError("episode segments must be a list")
        for segment in segments:
            event = event_from_segment(record, segment, release_manifest_sha256)
            if event["event_id"] in seen_event:
                raise ValueError("duplicate canonical event_id; source event collision")
            seen_event.add(event["event_id"])
            events.append(event)
    if not sources:
        raise ValueError("release selection contains no source episode metadata records")
    _validate_release_split_provenance(release, sources)
    roles = _role_for_groups(sources, calibration_per_task, calibration_group_ids)
    group_rows = _source_group_rows(sources, roles)
    for event in events:
        event["usage_role"] = roles[event["source"]["source_group_id"]]
    # Sorting makes byte-identical builds independent of harmless input order.
    events.sort(key=lambda row: ({"train": 0, "eval": 1}[row["source"]["original_split"]], row["source"]["task_index"],
                                 row["source"]["task_instance_id"], row["source"]["episode_index"],
                                 row["event_interval"]["start_frame"], row["event_id"]))
    coverage = _coverage(events, sources, coverage_expectations)
    eligible_episode_ids = {event["source"]["episode_index"] for event in events}
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        files = {
            "source_groups.jsonl": _write_jsonl(staging / "source_groups.jsonl", group_rows),
            "event_candidates.jsonl": _write_jsonl(staging / "event_candidates.jsonl", events),
        }
        _check_timeout(started, max_seconds)
        result = {
            "schema_version": INDEX_SCHEMA,
            "status": "METADATA_ONLY_READY_FOR_PACKET_RENDER",
            "training_eligible": False,
            "outcome_supervision": False,
            "corrective_action_supervision": False,
            "source_release_manifest_sha256": release_manifest_sha256,
            "source_release_files": inputs,
            "source_release_path": str(release.resolve()),
            "coverage_expectations_sha256": canonical_sha256(coverage_expectations),
            "selection_policy": selection_policy,
            "calibration_selection": calibration_selection,
            "files": files,
            "source_episodes": len(sources), "unique_raw_source_episodes": len(seen_raw_episode),
            "unique_published_episode_indices": len(seen_published_episode),
            "source_groups": len(group_rows), "event_candidates": len(events),
            # Candidate inventory is intentionally separate from accepted
            # labels/windows.  Metadata cannot manufacture review or physics
            # evidence, so every accepted view starts at zero here.
            "candidate_counts": {"input_source_episodes": len(sources),
                                 "eligible_source_episodes": len(eligible_episode_ids),
                                 "event_candidates": len(events)},
            "accepted_label_counts": {"goal_satisfaction_counterfactual": 0, "attempt_outcome": 0,
                                      "recovery_decision": 0, "corrective_action": 0,
                                      "accepted_temporal_windows": 0,
                                  "accepted_corrective_action_windows": 0},
            "partial_source_coverage": selection_policy["partial"],
            "usage_roles": dict(sorted(Counter(row["usage_role"] for row in group_rows).items())),
            "coverage": coverage,
            "builder_sha256": sha256_file(Path(__file__)),
            "protocol_sha256": sha256_file(REPO / "src/g05/data/memlite_event_protocol.py"),
            "wall_seconds": time.monotonic() - started,
        }
        (staging / "manifest.json").write_bytes(canonical_json(result).encode("utf-8") + b"\n")
        inventory_seal = {
            "schema_version": INVENTORY_SEAL_SCHEMA,
            "index_manifest_sha256": sha256_file(staging / "manifest.json"),
            "source_release_manifest_sha256": release_manifest_sha256,
            "coverage_expectations_sha256": canonical_sha256(coverage_expectations),
            "expected_payload_files": sorted(INDEX_PAYLOAD_FILES),
            "payload_files": files,
        }
        (staging / "inventory_seal.json").write_bytes(canonical_json(inventory_seal).encode("utf-8") + b"\n")
        inventory_seal_sha256 = sha256_file(staging / "inventory_seal.json")
        os.rename(staging, output)
        return {**result, "inventory_seal_sha256": inventory_seal_sha256}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def resume_index(release: Path, output: Path, *, calibration_per_task: int,
                 calibration_group_ids: set[str] | None = None,
                 coverage_expectations: Mapping[str, Any] | None = None,
                 max_source_episodes: int | None = None,
                 expected_inventory_seal_sha256: str | None = None) -> dict[str, Any]:
    release, output = Path(release), Path(output)
    if not output.is_dir() or output.is_symlink():
        raise ValueError("--resume requires a sealed regular output directory")
    if coverage_expectations is None:
        raise ValueError("--resume requires the exact official coverage expectations")
    coverage_expectations = validate_coverage_expectations(coverage_expectations)
    if expected_inventory_seal_sha256 is None:
        raise ValueError("--resume requires an externally recorded inventory seal SHA-256")
    _sha(expected_inventory_seal_sha256, "expected_inventory_seal_sha256")
    result = read_json(output / "manifest.json")
    if result.get("schema_version") != INDEX_SCHEMA or result.get("status") != "METADATA_ONLY_READY_FOR_PACKET_RENDER":
        raise ValueError("output is not a sealed P107 metadata index")
    if result.get("source_release_manifest_sha256") != sha256_file(release / "manifest.json"):
        raise ValueError("release manifest changed; refusing to resume mismatched index")
    if result.get("source_release_files") != _input_files(release):
        raise ValueError("release metadata changed; refusing to resume mismatched index")
    if result.get("coverage_expectations_sha256") != canonical_sha256(coverage_expectations):
        raise ValueError("official coverage expectations changed; refusing to resume mismatched index")
    if result.get("selection_policy") != _selection_policy(max_source_episodes):
        raise ValueError("source selection policy changed; refusing to resume mismatched index")
    expected_policy = _calibration_policy(sha256_file(release / "manifest.json"), calibration_per_task,
                                          set() if calibration_group_ids is None else set(calibration_group_ids))
    if result.get("calibration_selection") != expected_policy:
        raise ValueError("calibration policy changed; refusing to resume mismatched index")
    if set(result.get("files", {})) != INDEX_PAYLOAD_FILES:
        raise ValueError("sealed index manifest must contain the exact expected payload inventory")
    inventory_seal_path = output / "inventory_seal.json"
    if sha256_file(inventory_seal_path) != expected_inventory_seal_sha256:
        raise ValueError("externally recorded inventory seal SHA-256 does not match output")
    inventory_seal = read_json(inventory_seal_path)
    expected_seal = {
        "schema_version": INVENTORY_SEAL_SCHEMA,
        "index_manifest_sha256": sha256_file(output / "manifest.json"),
        "source_release_manifest_sha256": result["source_release_manifest_sha256"],
        "coverage_expectations_sha256": result["coverage_expectations_sha256"],
        "expected_payload_files": sorted(INDEX_PAYLOAD_FILES),
        "payload_files": result["files"],
    }
    if inventory_seal != expected_seal:
        raise ValueError("inventory seal does not bind the exact manifest/payload inventory")
    for name, receipt in result["files"].items():
        path = output / name
        if sha256_file(path) != receipt.get("sha256") or path.stat().st_size != receipt.get("bytes"):
            raise ValueError(f"sealed output file changed: {name}")
        if len(read_jsonl(path)) != receipt.get("rows"):
            raise ValueError(f"sealed output row count changed: {name}")
    return {"status": "RESUME_VALIDATED", "manifest": result,
            "inventory_seal_sha256": expected_inventory_seal_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="frozen v4 compact release root")
    parser.add_argument("--output", type=Path, required=True, help="new external index directory")
    parser.add_argument("--calibration-per-task", type=int, default=1)
    parser.add_argument("--calibration-group-ids", type=Path,
                        help="predeclared one-per-line source group hashes or task/instance JSON objects")
    parser.add_argument("--coverage-expectations", type=Path, required=True,
                        help="official global task/verb contract; required task×verb pairs may be undeclared")
    parser.add_argument("--max-source-episodes", type=int,
                        help="explicit partial pilot prefix; seals partial coverage and never changes source SHA")
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true", help="verify a complete sealed index; never append")
    parser.add_argument("--expected-inventory-seal-sha256",
                        help="externally recorded inventory seal required only by --resume")
    args = parser.parse_args()
    release_manifest_sha256 = sha256_file(args.release / "manifest.json")
    calibration_group_ids = read_calibration_group_ids(args.calibration_group_ids, release_manifest_sha256)
    coverage_expectations = read_coverage_expectations(args.coverage_expectations)
    result = (resume_index(args.release, args.output, calibration_per_task=args.calibration_per_task,
                           calibration_group_ids=calibration_group_ids, coverage_expectations=coverage_expectations,
                           max_source_episodes=args.max_source_episodes,
                           expected_inventory_seal_sha256=args.expected_inventory_seal_sha256)
              if args.resume else build_index(args.release, args.output, calibration_per_task=args.calibration_per_task,
                                               calibration_group_ids=calibration_group_ids,
                                               coverage_expectations=coverage_expectations,
                                               max_source_episodes=args.max_source_episodes, max_seconds=args.max_seconds))
    print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
