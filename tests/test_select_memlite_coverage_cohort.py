import hashlib
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import select_memlite_coverage_cohort as cohort  # noqa: E402


RELEASE = "a" * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def record(*, task: int, role: str = "annotation_calibration", frame: int = 10,
           group: str | None = None, raw_episode: int = 1, event_id: str | None = None,
           compound: bool = False, parallel: bool = False) -> cohort.CandidateRecord:
    group = group or digest(f"group:{task}:{raw_episode}:{frame}")
    event_id = event_id or digest(f"event:{task}:{raw_episode}:{frame}:{group}")
    skill = {"skill_id": 10, "verb": "GRASP", "target": "cup", "source": "", "destination": "",
             "target_part": "", "arm": "UNSPECIFIED", "skill_start": 0, "skill_end": 100}
    skills = [skill, {**skill, "skill_id": 11, "verb": "PLACE"}] if compound else [skill]
    source = {
        "source_release_manifest_sha256": RELEASE,
        "source_annotation_sha256": digest(f"annotation:{event_id}"),
        "source_group_id": group,
        "task_index": task,
        "task_instance_id": task + 1,
        "raw_episode_id": raw_episode,
        "episode_index": raw_episode,
        "episode_length": 1000,
        "original_split": "eval" if role == "evaluation_only" else "train",
    }
    event = {
        "event_id": event_id,
        "source": source,
        "observation": {"frame": frame, "timestamp_s": frame / 30.0},
        "event_interval": {"start_frame": 0, "end_frame": 100},
        "skill_bundle": skills,
        "parallel_bundle": parallel,
        "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "usage_role": role,
    }
    strata = cohort._structural_strata(event, start=0, end=100, episode_length=1000)
    return cohort.CandidateRecord(
        event=event, event_id=event_id, usage_role=role, immutable_split=source["original_split"],
        task_id=task, source_group_id=group, raw_episode_id=raw_episode, episode_index=raw_episode,
        episode_length=1000, observation_frame=frame, interval_start=0, interval_end=100,
        skill_ids=tuple(sorted({skill["skill_id"] for skill in skills})), structural_strata=strata,
        source_window=(RELEASE, group, raw_episode, frame),
    )


def spec(name: str, tasks: set[int], *, total: int | None = None,
         structural: dict[str, int] | None = None,
         isolate: bool = True, max_group: int = 1) -> cohort.RoleSpec:
    return cohort.RoleSpec(
        name=name, required_tasks=frozenset(tasks), total_target=len(tasks) if total is None else total,
        structural_quotas={} if structural is None else structural, seed="fixture-seed",
        isolate_prior_source_groups=isolate, max_per_source_group=max_group,
    )


def no_prior() -> cohort.PriorWindows:
    return cohort.PriorWindows(frozenset(), frozenset(), None, 0)


def test_exact_prior_window_excludes_different_event_id_and_group_policy_controls_distinct_frame():
    first = record(task=0, frame=10, event_id=digest("old-id"), group=digest("same-group"))
    later = record(task=0, frame=11, event_id=digest("new-id"), group=first.source_group_id)
    prior = cohort.PriorWindows(frozenset({first.source_window}), frozenset({first.source_group_id}), "b" * 64, 1)

    isolated = cohort.select_records([first, later], specs=[spec("train", {0})], prior=prior)["train"]
    assert not isolated.selected
    assert isolated.report["excluded_prior_exact_source_windows"] == 1
    assert isolated.report["excluded_prior_source_groups"] == 1

    reusable = cohort.select_records(
        [later], specs=[spec("train", {0}, isolate=False)], prior=prior
    )["train"]
    assert [item.event_id for item in reusable.selected] == [later.event_id]


def test_train_and_eval_roles_are_strict_and_student_groups_are_refused():
    student = record(task=0, role="student_candidate")
    evaluation = record(task=0, role="evaluation_only", group=digest("eval"))
    train = cohort.select_records([student], specs=[spec("train", {0})], prior=no_prior())["train"]
    assert not train.selected
    assert train.report["refused_student_candidate_events"] == 1

    eval_result = cohort.select_records([evaluation], specs=[spec("eval", {0})], prior=no_prior())["eval"]
    assert [item.event_id for item in eval_result.selected] == [evaluation.event_id]
    assert eval_result.selected[0].immutable_split == "eval"


def test_late_candidate_wins_without_retained_pool_ceiling():
    records = [record(task=0, frame=index, raw_episode=index, group=digest(f"g:{index}"))
               for index in range(50001)]
    expected = min(records, key=lambda item: (cohort._rank(item, spec("train", {0})), item.event_id))
    result = cohort.select_records(records, specs=[spec("train", {0})], prior=no_prior())["train"]
    assert len(result.selected) == 1
    assert result.selected[0].event_id == expected.event_id


def test_deterministic_selection_is_input_order_independent_and_reports_caps():
    shared_group = digest("shared-group")
    rows = [
        record(task=0, raw_episode=1, frame=10, group=shared_group),
        record(task=1, raw_episode=2, frame=20, group=shared_group),
        record(task=2, raw_episode=3, frame=30, group=digest("independent"), compound=True),
    ]
    first = cohort.select_records(rows, specs=[spec("train", {0, 1}, total=2)], prior=no_prior())["train"]
    second = cohort.select_records(list(reversed(rows)), specs=[spec("train", {0, 1}, total=2)], prior=no_prior())["train"]
    assert [item.event_id for item in first.selected] == [item.event_id for item in second.selected]
    assert first.report["missing_required_task_ids_due_to_source_caps"]


def test_structural_quota_reports_genuine_missing_category_without_forcing_rows():
    rows = [record(task=0), record(task=1, compound=True, group=digest("compound"))]
    result = cohort.select_records(
        rows, specs=[spec("train", {0}, total=3, structural={"COMPOUND": 2})], prior=no_prior()
    )["train"]
    assert result.report["structural_quota_available_cells"]["COMPOUND"] == 1
    assert result.report["missing_structural_quota"]["COMPOUND"] == 1
    assert len(result.selected) == 2


def test_prior_sidecar_hash_and_release_are_fail_closed(tmp_path):
    sidecar = tmp_path / "prior.jsonl"
    row = {
        "schema_version": cohort.EXCLUSION_SCHEMA,
        "source_release_manifest_sha256": RELEASE,
        "source_group_id": digest("group"),
        "raw_episode_id": 4,
        "observation_frame": 12,
        "event_id": digest("audit-only-id"),
    }
    sidecar.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    expected = cohort.base.sha256_file(sidecar)
    loaded = cohort.load_prior_windows(sidecar, expected_sha256=expected, expected_release=RELEASE)
    assert loaded.rows == 1
    assert loaded.keys == frozenset({(RELEASE, row["source_group_id"], 4, 12)})

    bad = dict(row)
    bad["source_release_manifest_sha256"] = "c" * 64
    sidecar.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="different release"):
        cohort.load_prior_windows(sidecar, expected_sha256=cohort.base.sha256_file(sidecar), expected_release=RELEASE)


def test_candidate_source_window_hash_is_fail_closed():
    candidate = record(task=0)
    malformed = json.loads(json.dumps(candidate.event))
    malformed["source"]["source_group_id"] = "not-a-sha"
    with pytest.raises(ValueError, match="source_group_id"):
        cohort._source_window_key(malformed, expected_release=RELEASE)


def test_writer_keeps_train_queue_seal_and_eval_outside_train_payload(tmp_path):
    index = tmp_path / "sealed-index"
    index.mkdir()
    (index / "manifest.json").write_text("{}\n")
    source_report = {
        "index_manifest_path": str(index / "manifest.json"),
        "index_manifest_sha256": digest("index-manifest"),
        "source_release_manifest_sha256": RELEASE,
        "coverage_expectations_sha256": digest("coverage"),
    }
    index_manifest = {"source_release_manifest_sha256": RELEASE}
    coverage = {
        "expected_task_ids": [0],
        "expected_skill_vocabulary": [{"skill_id": 10, "skill_description": "grasp"}],
        "required_task_skill_pairs": None,
    }
    train_selection = cohort.select_records([record(task=0)], specs=[spec("train", {0})], prior=no_prior())["train"]
    train_result = cohort.write_role_output(
        tmp_path / "train-output", train_selection, source_report=source_report,
        index_manifest=index_manifest, inventory_sha256=digest("inventory"),
        coverage_sha256=source_report["coverage_expectations_sha256"], coverage_expectations=coverage,
        protocol_sha256=digest("protocol"), prior=no_prior(),
    )
    assert train_result["queue_seal_sha256"]
    assert (tmp_path / "train-output" / "queue_seal.json").is_file()
    queue_seal = json.loads((tmp_path / "train-output" / "queue_seal.json").read_text())
    assert queue_seal["schema_version"] == cohort.base.QUEUE_SEAL_SCHEMA
    assert queue_seal["payload_files"]["student_candidate_queue.jsonl"]["rows"] == 0

    eval_record = record(task=0, role="evaluation_only", group=digest("eval-writer"))
    eval_selection = cohort.select_records([eval_record], specs=[spec("eval", {0})], prior=no_prior())["eval"]
    eval_result = cohort.write_role_output(
        tmp_path / "eval-output", eval_selection, source_report=source_report,
        index_manifest=index_manifest, inventory_sha256=digest("inventory"),
        coverage_sha256=source_report["coverage_expectations_sha256"], coverage_expectations=coverage,
        protocol_sha256=digest("protocol"), prior=no_prior(),
    )
    assert eval_result["queue_seal_sha256"] is None
    assert not (tmp_path / "eval-output" / "queue_seal.json").exists()
    assert (tmp_path / "eval-output" / "evaluation_only_render_requests.jsonl").is_file()
