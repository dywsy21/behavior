"""Contract tests for the metadata-only coverage render handoffs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
sys.path.insert(0, str(REPO / "tests"))
import publish_memlite_coverage_render_handoff as handoff
import render_memlite_evaluation_packets as evaluation_renderer
import render_memlite_event_packets as renderer
import select_memlite_coverage_cohort as cohort
import select_memlite_event_annotation_queue as queue
import test_select_memlite_event_annotation_queue as selector_fixture


def _fixture(tmp_path: Path) -> dict[str, object]:
    """Build a sealed compact index with the production packet constant.

    The selector fixture intentionally has a minimal protocol double.  The
    adapter needs the production packet schema constant as well, so the test
    pins a byte-identical copy with that one public constant added.  Event IDs
    remain unchanged because the protocol's event identity excludes camera
    locator metadata.
    """
    index = tmp_path / "index"
    coverage_path = selector_fixture.write_index(index)
    protocol_path = tmp_path / "protocol.py"
    protocol_path.write_bytes(
        selector_fixture.OWNER_PROTOCOL.read_bytes()
        + b'\nPACKET_SCHEMA_VERSION = "memlite-event-packet-v1"\n'
    )
    protocol_sha = queue.sha256_file(protocol_path)
    protocol = queue.load_canonical_protocol(protocol_path, expected_sha256=protocol_sha)

    events = []
    for event in queue.read_jsonl(index / "event_candidates.jsonl"):
        event["video_locators"] = [
            {
                "view": view,
                "camera_key": f"observation.rgb.{view}",
                "relative_path": f"videos/{view}.mp4",
                "episode_start_timestamp_s": 0.0,
                "requested_timestamp_s": event["observation"]["frame"] / 30.0,
                "expected_fps": 30,
                "locator_status": "METADATA_ONLY_UNRESOLVED",
            }
            for view in ("head", "left_wrist", "right_wrist")
        ]
        protocol.validate_event(event)
        events.append(event)
    events_path = index / "event_candidates.jsonl"
    events_path.write_text("".join(queue.canonical_json(row) + "\n" for row in events))
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_sha256"] = protocol_sha
    manifest["files"]["event_candidates.jsonl"] = selector_fixture.receipt(events_path)
    manifest_path.write_text(queue.canonical_json(manifest) + "\n")
    selector_fixture.reseal_index(index, manifest)

    coverage = queue.read_json(coverage_path)
    return {
        "index": index,
        "coverage_path": coverage_path,
        "coverage": coverage,
        "coverage_sha": queue.canonical_sha256(coverage),
        "protocol_path": protocol_path,
        "protocol_sha": protocol_sha,
        "source_release_sha": selector_fixture.SOURCE_SHA,
        "inventory_sha": queue.sha256_file(index / "inventory_seal.json"),
    }


def _selectors(tmp_path: Path, data: dict[str, object]) -> dict[str, object]:
    index = data["index"]
    protocol = queue.load_canonical_protocol(data["protocol_path"], expected_sha256=data["protocol_sha"])
    manifest, events_path, groups = queue.validate_index(
        index,
        expected_source_manifest_sha256=data["source_release_sha"],
        canonical_protocol=protocol,
        expected_protocol_sha256=data["protocol_sha"],
    )
    coverage = data["coverage"]
    official_tasks = set(coverage["expected_task_ids"])
    official_skills = {
        skill["skill_id"]
        for event in queue.read_jsonl(events_path)
        for skill in event["skill_bundle"]
    }
    candidates = []
    for event in queue.read_jsonl(events_path):
        candidate, reason = cohort._candidate_from_event(
            event,
            expected_release=data["source_release_sha"],
            groups=groups,
            official_tasks=official_tasks,
            official_skills=official_skills,
        )
        assert reason is None, (event.get("event_id"), reason)
        candidates.append(candidate)
    prior = cohort.PriorWindows(frozenset(), frozenset(), None, 0)
    selections = cohort.select_records(
        candidates,
        specs=[
            cohort.RoleSpec("train", frozenset({0, 1}), 2, {}, "fixture-train"),
            cohort.RoleSpec("eval", frozenset({2}), 1, {}, "fixture-eval"),
        ],
        prior=prior,
    )
    source_report = {
        "index_manifest_path": str(index / "manifest.json"),
        "index_manifest_sha256": queue.sha256_file(index / "manifest.json"),
        "source_release_manifest_sha256": data["source_release_sha"],
        "coverage_expectations_sha256": data["coverage_sha"],
    }
    outputs = {}
    for role in ("train", "eval"):
        outputs[role] = cohort.write_role_output(
            tmp_path / f"{role}-selector",
            selections[role],
            source_report=source_report,
            index_manifest=manifest,
            inventory_sha256=data["inventory_sha"],
            coverage_sha256=data["coverage_sha"],
            coverage_expectations=coverage,
            protocol_sha256=data["protocol_sha"],
            prior=prior,
        )
    return {**data, "manifest": manifest, "selectors": outputs}


def _train_handoff(tmp_path: Path, selected: dict[str, object]) -> dict[str, object]:
    train = selected["selectors"]["train"]
    return handoff.publish_canonical_train_handoff(
        index=selected["index"],
        train_selector=tmp_path / "train-selector",
        output_root=tmp_path / "train-handoff",
        coverage_path=selected["coverage_path"],
        expected_source_release_sha256=selected["source_release_sha"],
        expected_inventory_sha256=selected["inventory_sha"],
        expected_coverage_sha256=selected["coverage_sha"],
        protocol_path=selected["protocol_path"],
        expected_protocol_sha256=selected["protocol_sha"],
        expected_selector_manifest_sha256=train["manifest_sha256"],
        expected_selector_selection_seal_sha256=train["selection_seal_sha256"],
        expected_selector_queue_seal_sha256=train["queue_seal_sha256"],
    )


def test_train_handoff_is_canonical_and_resume_validated(tmp_path: Path) -> None:
    selected = _selectors(tmp_path, _fixture(tmp_path))
    result = _train_handoff(tmp_path, selected)
    train = Path(result["train_output"])
    manifest = queue.read_json(train / "manifest.json")
    assert manifest["schema_version"] == queue.MANIFEST_SCHEMA
    assert set(manifest["files"]) == queue.QUEUE_PAYLOAD_FILES
    assert not (train / "train_coverage_provenance.json").exists()
    provenance = queue.read_json(Path(result["output_root"]) / "train_coverage_provenance.json")
    assert provenance["selector_manifest_sha256"] == selected["selectors"]["train"]["manifest_sha256"]
    assert provenance["no_outcome_or_action_labels"] is True
    seal_sha = handoff.base.validate_queue_seal(
        train,
        manifest,
        expected_queue_seal_sha256=result["queue_seal_sha256"],
        expected_inventory_seal_sha256=queue.sha256_file(Path(result["train_source_index"]) / "inventory_seal.json"),
        expected_source_manifest_sha256=selected["source_release_sha"],
        expected_protocol_sha256=selected["protocol_sha"],
        expected_coverage_expectations_sha256=selected["coverage_sha"],
    )
    assert seal_sha == result["queue_seal_sha256"]
    args = handoff._canonical_train_args(
        Path(result["train_source_index"]), train, selected["coverage_path"],
        inventory_sha256=queue.sha256_file(Path(result["train_source_index"]) / "inventory_seal.json"),
        source_release_sha256=selected["source_release_sha"], protocol_path=selected["protocol_path"],
        protocol_sha256=selected["protocol_sha"], calibration_budget=2,
    )
    args.expected_queue_seal_sha256 = seal_sha
    assert queue.resume_queue(Path(result["train_source_index"]), train, args=args)["status"] == "RESUME_VALIDATED"


def test_train_packet_transport_uses_canonical_queue_and_pinned_protocol(tmp_path: Path) -> None:
    selected = _selectors(tmp_path, _fixture(tmp_path))
    result = _train_handoff(tmp_path, selected)
    train = Path(result["train_output"])
    source_index = Path(result["train_source_index"])
    binding = renderer._load_protocol(selected["protocol_path"], expected_sha256=selected["protocol_sha"])
    requests, request_sha = renderer._read_render_requests(
        train / "camera_native_render_requests.jsonl", result["request_sha256"])
    queue_sha, _ = renderer._read_queue_seal(
        train / "queue_seal.json", result["queue_seal_sha256"],
        render_requests_path=train / "camera_native_render_requests.jsonl",
        render_requests_sha256=request_sha,
    )
    packet_dir = tmp_path / "train-packets"
    packet = renderer.create_packets(
        source_index, packet_dir, event_ids=set(requests), limit=None, questions={},
        include_source_annotation_context=False, decode=False, raw_root=None, contact_sheets=False,
        temporal_offsets_frames=evaluation_renderer.DEFAULT_EVAL_OFFSETS,
        render_requests=requests, render_requests_sha256=request_sha, queue_seal_sha256=queue_sha,
        queue_seal_source_release_manifest_sha256=selected["source_release_sha"],
        protocol_binding=binding, max_seconds=30,
    )
    assert packet["training_eligible"] is False
    assert renderer.resume_packets(
        source_index, packet_dir, expected_packet_manifest_sha256=packet["packet_manifest_sha256"],
        protocol_binding=binding,
    )["status"] == "RESUME_VALIDATED"


def test_eval_renderer_emits_only_eval_receipt_and_no_train_seal(tmp_path: Path) -> None:
    selected = _selectors(tmp_path, _fixture(tmp_path))
    evaluation = selected["selectors"]["eval"]
    result = evaluation_renderer.render_evaluation_packets(
        index=selected["index"], selector_output=tmp_path / "eval-selector", output=tmp_path / "eval-packets",
        coverage_path=selected["coverage_path"], protocol_path=selected["protocol_path"],
        expected_protocol_sha256=selected["protocol_sha"],
        expected_source_release_sha256=selected["source_release_sha"],
        expected_inventory_sha256=selected["inventory_sha"], expected_coverage_sha256=selected["coverage_sha"],
        expected_selector_manifest_sha256=evaluation["manifest_sha256"],
        expected_selector_selection_seal_sha256=evaluation["selection_seal_sha256"], decode=False, max_seconds=30,
    )
    packets = Path(result["output"])
    receipt = queue.read_json(packets / "evaluation_render_receipt.json")
    assert set(receipt) == handoff.EVAL_RECEIPT_FIELDS
    assert receipt["usage_role"] == "evaluation_only"
    assert receipt["immutable_split"] == "eval"
    assert receipt["training_eligible"] is False
    assert not (packets / "queue_seal.json").exists()
    assert receipt["packet_manifest_queue_authority_sha256"] == evaluation["selection_seal_sha256"]
    assert receipt["no_outcome_or_action_labels"] is True


def test_train_renderer_keeps_rejecting_eval_request_role(tmp_path: Path) -> None:
    selected = _selectors(tmp_path, _fixture(tmp_path))
    request_path = tmp_path / "eval-selector" / "evaluation_only_render_requests.jsonl"
    requests, _ = renderer._read_render_requests(
        request_path, queue.sha256_file(request_path),
    )
    event_id = next(iter(requests))
    event = next(row for row in queue.read_jsonl(selected["index"] / "event_candidates.jsonl")
                 if row["event_id"] == event_id)
    binding = renderer._load_protocol(selected["protocol_path"], expected_sha256=selected["protocol_sha"])
    with pytest.raises(ValueError, match="role"):
        renderer._request_temporal_samples(event, requests[event_id], protocol_binding=binding)


def test_eval_selector_hash_tamper_fails_before_packet_output(tmp_path: Path) -> None:
    selected = _selectors(tmp_path, _fixture(tmp_path))
    evaluation = selected["selectors"]["eval"]
    output = tmp_path / "tampered-eval-packets"
    with pytest.raises(ValueError, match="selector manifest bytes"):
        evaluation_renderer.render_evaluation_packets(
            index=selected["index"], selector_output=tmp_path / "eval-selector", output=output,
            coverage_path=selected["coverage_path"], protocol_path=selected["protocol_path"],
            expected_protocol_sha256=selected["protocol_sha"],
            expected_source_release_sha256=selected["source_release_sha"],
            expected_inventory_sha256=selected["inventory_sha"], expected_coverage_sha256=selected["coverage_sha"],
            expected_selector_manifest_sha256="0" * 64,
            expected_selector_selection_seal_sha256=evaluation["selection_seal_sha256"], decode=False, max_seconds=30,
        )
    assert not output.exists()


def test_eval_authority_rejects_any_train_queue_seal_argument(tmp_path: Path) -> None:
    selected = _selectors(tmp_path, _fixture(tmp_path))
    evaluation = selected["selectors"]["eval"]
    with pytest.raises(ValueError, match="must not carry a TRAIN queue seal"):
        handoff.read_selector_handoff(
            tmp_path / "eval-selector",
            role="eval",
            expected_manifest_sha256=evaluation["manifest_sha256"],
            expected_selection_seal_sha256=evaluation["selection_seal_sha256"],
            expected_queue_seal_sha256="0" * 64,
        )


def test_eval_protocol_pin_fails_before_staging_or_decode(tmp_path: Path) -> None:
    selected = _selectors(tmp_path, _fixture(tmp_path))
    evaluation = selected["selectors"]["eval"]
    output = tmp_path / "bad-protocol-packets"
    with pytest.raises(ValueError, match="protocol"):
        evaluation_renderer.render_evaluation_packets(
            index=selected["index"], selector_output=tmp_path / "eval-selector", output=output,
            coverage_path=selected["coverage_path"], protocol_path=selected["protocol_path"],
            expected_protocol_sha256="0" * 64,
            expected_source_release_sha256=selected["source_release_sha"],
            expected_inventory_sha256=selected["inventory_sha"], expected_coverage_sha256=selected["coverage_sha"],
            expected_selector_manifest_sha256=evaluation["manifest_sha256"],
            expected_selector_selection_seal_sha256=evaluation["selection_seal_sha256"], decode=False, max_seconds=30,
        )
    assert not output.exists()


def test_train_publisher_tamper_fails_before_fresh_root_publish(tmp_path: Path) -> None:
    selected = _selectors(tmp_path, _fixture(tmp_path))
    train = selected["selectors"]["train"]
    output = tmp_path / "tampered-train-handoff"
    with pytest.raises(ValueError, match="selector manifest bytes"):
        handoff.publish_canonical_train_handoff(
            index=selected["index"], train_selector=tmp_path / "train-selector", output_root=output,
            coverage_path=selected["coverage_path"], expected_source_release_sha256=selected["source_release_sha"],
            expected_inventory_sha256=selected["inventory_sha"], expected_coverage_sha256=selected["coverage_sha"],
            protocol_path=selected["protocol_path"], expected_protocol_sha256=selected["protocol_sha"],
            expected_selector_manifest_sha256="0" * 64,
            expected_selector_selection_seal_sha256=train["selection_seal_sha256"],
            expected_selector_queue_seal_sha256=train["queue_seal_sha256"],
        )
    assert not output.exists()
