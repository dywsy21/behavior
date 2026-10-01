"""Focused contract tests for the role-separated coverage query producer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
sys.path.insert(0, str(REPO / "tests"))

import build_memlite_coverage_visual_relation_queries as producer  # noqa: E402
import select_memlite_event_annotation_queue as queue  # noqa: E402
import test_memlite_coverage_render_handoffs as handoff_fixture  # noqa: E402
import test_select_memlite_event_annotation_queue as selector_fixture  # noqa: E402


def _selected(tmp_path: Path) -> dict[str, object]:
    data = handoff_fixture._fixture(tmp_path)
    # The handoff fixture intentionally uses arbitrary verbs/IDs for renderer
    # tests.  Replace only the immutable event skill metadata with canonical-35
    # examples so this producer test exercises real goal templates.
    events_path = data["index"] / "event_candidates.jsonl"
    events = queue.read_jsonl(events_path)
    canonical_by_role = {
        "annotation_calibration": [(67, "press", "PRESS"), (10, "open door", "OPEN_DOOR")],
        "evaluation_only": [(2, "pick up from", "GRASP")],
    }
    counters = {role: 0 for role in canonical_by_role}
    for event in events:
        role = event["usage_role"]
        if role not in canonical_by_role:
            continue
        skill_id, description, verb = canonical_by_role[role][counters[role] % len(canonical_by_role[role])]
        counters[role] += 1
        skill = event["skill_bundle"][0]
        skill.update({
            "skill_id": skill_id,
            "raw_description": description,
            "verb": verb,
            "skill_start": event["event_interval"]["start_frame"],
            "skill_end": event["event_interval"]["end_frame"],
            "skill_idx": 0,
        })
        if verb == "PRESS":
            skill.update({"target": "radio_89", "source": "", "destination": ""})
        elif verb == "OPEN_DOOR":
            skill.update({"target": "fridge", "source": "", "destination": ""})
        else:
            skill.update({"target": "cup", "source": "table", "destination": ""})
    events_path.write_text("".join(queue.canonical_json(row) + "\n" for row in events))
    manifest_path = data["index"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["event_candidates.jsonl"] = selector_fixture.receipt(events_path)
    manifest_path.write_text(queue.canonical_json(manifest) + "\n")
    selector_fixture.reseal_index(data["index"], manifest)
    data["inventory_sha"] = queue.sha256_file(data["index"] / "inventory_seal.json")
    return handoff_fixture._selectors(tmp_path, data)


def _build(tmp_path: Path, selected: dict[str, object], role: str) -> dict[str, object]:
    role_result = selected["selectors"][role]
    selector_root = tmp_path / f"{role}-selector"
    return producer.build_output(
        index=selected["index"],
        selector_root=selector_root,
        output_dir=tmp_path / f"{role}-queries",
        role=role,
        expected_selector_manifest_sha256=role_result["manifest_sha256"],
        expected_selector_selection_seal_sha256=role_result["selection_seal_sha256"],
        expected_index_manifest_sha256=queue.sha256_file(selected["index"] / "manifest.json"),
        expected_inventory_sha256=selected["inventory_sha"],
        expected_source_release_sha256=selected["source_release_sha"],
        expected_protocol_sha256=selected["protocol_sha"],
        expected_coverage_sha256=selected["coverage_sha"],
        protocol_path=selected["protocol_path"],
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_train_output_reuses_goal_templates_without_phase_pins(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    result = _build(tmp_path, selected, "train")
    output = Path(result["output_dir"])
    queries = _rows(output / "query_candidates.jsonl")
    registry = _rows(output / "prelabel_registry.jsonl")
    assert len(queries) == len(registry) == 2
    assert all(row["schema_version"] == producer.COVERAGE_QUERY_SCHEMA for row in queries)
    assert all(row["source_pin"]["schema_version"] == producer.COVERAGE_SOURCE_PIN_SCHEMA for row in queries)
    assert all("phase_index_path" not in row["source_pin"] for row in queries)
    assert all(row["usage_role"] == "annotation_calibration" for row in queries)
    assert all(row["immutable_split"] == "train" and row["training_eligible"] is False for row in queries)
    assert len({row["prelabel_query_id"] for row in queries}) == len(queries)


def test_eval_is_separate_native_review_sidecar_and_never_train_registry(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    result = _build(tmp_path, selected, "eval")
    output = Path(result["output_dir"])
    assert not (output / "prelabel_registry.jsonl").exists()
    assert (output / "evaluation_query_registry.jsonl").is_file()
    review = _rows(output / "evaluation_review.jsonl")
    assert review
    assert all(row["schema_version"] == producer.COVERAGE_EVAL_REVIEW_SCHEMA for row in review)
    assert all(row["usage_role"] == "evaluation_only" for row in review)
    assert all(row["immutable_split"] == "eval" and row["training_eligible"] is False for row in review)
    assert all(row["native_packet_contract"]["required_actor_field"] == "actor_packet.causal_temporal_rgb" for row in review)
    assert all(row["native_packet_contract"]["future_actor_field_forbidden"] == "audit.offline_future_rgb" for row in review)


def test_selector_manifest_pin_tamper_fails_before_index_join(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    role_result = selected["selectors"]["train"]
    with pytest.raises(ValueError, match="manifest bytes"):
        producer.build_output(
            index=selected["index"], selector_root=tmp_path / "train-selector", output_dir=tmp_path / "bad",
            role="train", expected_selector_manifest_sha256="0" * 64,
            expected_selector_selection_seal_sha256=role_result["selection_seal_sha256"],
            expected_index_manifest_sha256=queue.sha256_file(selected["index"] / "manifest.json"),
            expected_inventory_sha256=selected["inventory_sha"], expected_source_release_sha256=selected["source_release_sha"],
            expected_protocol_sha256=selected["protocol_sha"], expected_coverage_sha256=selected["coverage_sha"],
            protocol_path=selected["protocol_path"],
        )


def test_eval_role_rejects_train_selector_and_queue_seal_leak(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    train_result = selected["selectors"]["train"]
    with pytest.raises(ValueError, match="wrong role"):
        producer.build_output(
            index=selected["index"], selector_root=tmp_path / "train-selector", output_dir=tmp_path / "bad-role",
            role="eval", expected_selector_manifest_sha256=train_result["manifest_sha256"],
            expected_selector_selection_seal_sha256=train_result["selection_seal_sha256"],
            expected_index_manifest_sha256=queue.sha256_file(selected["index"] / "manifest.json"),
            expected_inventory_sha256=selected["inventory_sha"], expected_source_release_sha256=selected["source_release_sha"],
            expected_protocol_sha256=selected["protocol_sha"], expected_coverage_sha256=selected["coverage_sha"],
            protocol_path=selected["protocol_path"],
        )

    eval_root = tmp_path / "eval-selector"
    (eval_root / "queue_seal.json").write_bytes((tmp_path / "train-selector" / "queue_seal.json").read_bytes())
    eval_result = selected["selectors"]["eval"]
    with pytest.raises(ValueError, match="TRAIN queue seal"):
        producer.build_output(
            index=selected["index"], selector_root=eval_root, output_dir=tmp_path / "bad-eval-leak",
            role="eval", expected_selector_manifest_sha256=eval_result["manifest_sha256"],
            expected_selector_selection_seal_sha256=eval_result["selection_seal_sha256"],
            expected_index_manifest_sha256=queue.sha256_file(selected["index"] / "manifest.json"),
            expected_inventory_sha256=selected["inventory_sha"], expected_source_release_sha256=selected["source_release_sha"],
            expected_protocol_sha256=selected["protocol_sha"], expected_coverage_sha256=selected["coverage_sha"],
            protocol_path=selected["protocol_path"],
        )


def test_causal_frames_are_anchor_bounded_and_future_is_audit_only(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    result = _build(tmp_path, selected, "train")
    for row in _rows(Path(result["output_dir"]) / "query_candidates.jsonl"):
        temporal = row["temporal_binding"]
        assert all(frame <= temporal["anchor_frame"] for frame in temporal["actor_causal_frame_indices"])
        assert temporal["actor_causal_frame_indices"][-1] == temporal["anchor_frame"]
        assert temporal["future_actor_references_allowed"] is False
        assert set(item["view"] for item in temporal["causal_rgb_views"]) == set(producer.REQUIRED_VIEWS)


def test_query_id_changes_for_ordinal_or_source_pin_without_answer_fields(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    result = _build(tmp_path, selected, "train")
    row = _rows(Path(result["output_dir"]) / "query_candidates.jsonl")[0]
    identity = producer._query_identity(row, row["source_pin"], "train")
    changed_ordinal = copy.deepcopy(row)
    changed_ordinal["query_ordinal_within_event"] += 1
    changed_ordinal["current_visible_goal_relation"]["query_text"] += " (second atomic query)"
    changed_pin = copy.deepcopy(row)
    changed_pin["source_pin"]["selector_record_sha256"] = "f" * 64
    assert producer.canonical_digest(identity) != producer.canonical_digest(
        producer._query_identity(changed_ordinal, changed_ordinal["source_pin"], "train")
    )
    assert producer.canonical_digest(identity) != producer.canonical_digest(
        producer._query_identity(changed_pin, changed_pin["source_pin"], "train")
    )
    encoded = json.dumps(row, sort_keys=True)
    for forbidden in ("goal_satisfaction", "attempt_outcome", "recovery_decision", "action_payload", "canonical_goal_view_id"):
        assert forbidden not in encoded


def test_unsupported_goal_template_is_quarantined_not_rewritten_as_action(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    original = producer.SUPPORTED_RELATION_VERBS
    producer.SUPPORTED_RELATION_VERBS = frozenset()
    try:
        selector = producer._validate_selector(
            tmp_path / "train-selector", "train",
            expected_manifest_sha256=selected["selectors"]["train"]["manifest_sha256"],
            expected_selection_seal_sha256=selected["selectors"]["train"]["selection_seal_sha256"],
        )
        index_spec = producer._authenticate_index(
            producer._load_index(
                selected["index"], expected_manifest_sha256=queue.sha256_file(selected["index"] / "manifest.json"),
                expected_inventory_sha256=selected["inventory_sha"], expected_source_release_sha256=selected["source_release_sha"],
                expected_protocol_sha256=selected["protocol_sha"], expected_coverage_sha256=selected["coverage_sha"],
            ), selected["protocol_path"]
        )
        records, registry, unsupported = producer._build_records(selector, index_spec, None, None)
        assert records == [] and registry == [] and unsupported
        assert all(row["status"] == "UNSUPPORTED_GOAL_TEMPLATE" for row in unsupported)
        assert all(row["query_text"] is None for row in unsupported)
    finally:
        producer.SUPPORTED_RELATION_VERBS = original
