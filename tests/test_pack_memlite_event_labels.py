"""Adversarial P107 package tests; only synthetic metadata is published."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import tracemalloc
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import build_memlite_event_index as indexer  # noqa: E402
import pack_memlite_event_labels as pack  # noqa: E402
import audit_memlite_event_release as release_audit  # noqa: E402


FIXTURE = REPO / "tests" / "fixtures" / "p107_v4_compact_fixture.json"


def write_release(root: Path) -> None:
    fixture = json.loads(FIXTURE.read_text())
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(fixture["release_manifest"], sort_keys=True))
    (root / "episodes.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in fixture["episodes"]))
    (root / "split_provenance.json").write_text(json.dumps(fixture["split_provenance"], sort_keys=True))


def indexed(root: Path) -> tuple[Path, list[dict], list[dict], dict]:
    release, output = root / "v4", root / "index"
    fixture = json.loads(FIXTURE.read_text())
    write_release(release)
    indexer.build_index(release, output, calibration_per_task=1,
                        coverage_expectations=fixture["coverage_expectations"], max_seconds=10)
    groups = pack.read_jsonl(output / "source_groups.jsonl")
    events = pack.read_jsonl(output / "event_candidates.jsonl")
    return output, groups, events, pack.read_json(output / "manifest.json")


def reseal_index(index_path: Path) -> str:
    """Bind the current test manifest with the exact external inventory seal."""
    manifest = pack.read_json(index_path / "manifest.json")
    seal = {"schema_version": "memlite-event-inventory-seal-v1",
            "index_manifest_sha256": pack.sha256_file(index_path / "manifest.json"),
            "source_release_manifest_sha256": manifest["source_release_manifest_sha256"],
            "coverage_expectations_sha256": manifest["coverage_expectations_sha256"],
            "expected_payload_files": ["event_candidates.jsonl", "source_groups.jsonl"],
            "payload_files": manifest["files"]}
    (index_path / "inventory_seal.json").write_bytes(pack.canonical_json(seal).encode("utf-8") + b"\n")
    return pack.sha256_file(index_path / "inventory_seal.json")


def update_payload_receipt(index_path: Path, name: str) -> str:
    manifest_path = index_path / "manifest.json"
    manifest = pack.read_json(manifest_path)
    payload = index_path / name
    receipt = dict(manifest["files"][name])
    receipt.update(sha256=pack.sha256_file(payload), bytes=payload.stat().st_size,
                   rows=len(payload.read_bytes().splitlines()))
    manifest["files"][name] = receipt
    manifest_path.write_bytes(pack.canonical_json(manifest).encode("utf-8") + b"\n")
    return reseal_index(index_path)


def write_streaming_fixture(index_path: Path, event: dict, *, row_count: int) -> str:
    """Make a modest valid JSONL whose retained subset is one event."""
    payload = index_path / "event_candidates.jsonl"
    selected = ""
    with payload.open("wb") as stream:
        for number in range(row_count):
            row = deepcopy(event)
            row["event_kind"] = f"streamed-event-{number}"
            row["bundle_id"] = f"streamed-bundle-{number}"
            row["event_id"] = pack._protocol.event_id(row)
            stream.write(pack.canonical_json(row).encode("utf-8") + b"\n")
            selected = row["event_id"]
    update_payload_receipt(index_path, "event_candidates.jsonl")
    return selected


REAL_V3_SKILLS = (
    (1, "move to"), (2, "pick up from"), (3, "place on"), (4, "place in"), (5, "hand over"),
    (6, "insert"), (8, "release"), (9, "open drawer"), (10, "open door"), (11, "close drawer"),
    (12, "close door"), (13, "open lid"), (14, "close lid"), (19, "attach"), (28, "pour"),
    (34, "chop"), (46, "wipe hard"), (50, "sweep surface"), (61, "hang"), (67, "press"),
    (69, "turn on switch"), (70, "turn off switch"), (88, "ignite"), (90, "push to"),
    (91, "place on next to"), (92, "place in next to"), (93, "turn to"), (94, "hold"),
    (95, "spray"), (98, "place under"), (99, "tip over"), (100, "push tray"),
    (101, "pull tray"), (102, "sweep off"), (103, "lift"),
)


def real_full_v3_coverage_metadata() -> dict:
    """Small pinned copy of the sealed full-v3 coverage metadata, not its event JSONL."""
    expectations = {
        "schema_version": "p107-official-coverage-expectations-v3",
        "official_task_metadata_sha256": "90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23",
        "official_skill_vocabulary_sha256": "811375498518c84ca671b18109e7ff959cd7d2d0e4037cdded305bc1334f0e11",
        "expected_task_ids": list(range(100)),
        "expected_skill_vocabulary": [
            {"skill_id": skill_id, "skill_description": description} for skill_id, description in REAL_V3_SKILLS],
        "required_task_skill_pairs": None,
    }
    expected_sha = "39ccfb79420010bfc32e2f00d66cae255340a997b20026dc970fdffee412b3c7"
    assert pack.canonical_sha256(expectations) == expected_sha
    return {
        "source_release_manifest_sha256": expectations["official_task_metadata_sha256"],
        "coverage_expectations_sha256": expected_sha,
        "source_episodes": 20000,
        "event_candidates": 403257,
        "partial_source_coverage": False,
        "coverage": {
            "expectations": expectations,
            "expectations_sha256": expected_sha,
            "found_task_ids": list(range(100)),
            "missing_task_ids": [],
            "unexpected_task_ids": [],
            "found_global_skill_ids": [skill_id for skill_id, _ in REAL_V3_SKILLS],
            "missing_global_skill_ids": [],
            "unmapped_source_skill_ids": [],
            "source_skill_members_missing_skill_id": 0,
            "global_vocabulary_complete": True,
            "required_task_skill_pairs": None,
            "missing_required_task_skill_pairs": None,
            "required_pair_coverage_status": "NOT_DECLARED",
            "unique_input_source_episodes": 20000,
            "unique_candidate_source_episodes": 19889,
            "unique_event_candidates": 403257,
        },
    }


def v3_diagnostic_coverage(index_manifest: dict) -> dict:
    expectations = {
        "schema_version": "p107-official-coverage-expectations-v3",
        "official_task_metadata_sha256": index_manifest["source_release_manifest_sha256"],
        "official_skill_vocabulary_sha256": "b" * 64,
        "expected_task_ids": [0],
        "expected_skill_vocabulary": [{"skill_id": 1, "skill_description": "move to"}],
        "required_task_skill_pairs": None,
    }
    index_manifest = deepcopy(index_manifest)
    index_manifest["coverage_expectations_sha256"] = pack.canonical_sha256(expectations)
    index_manifest["coverage"] = {
        "expectations": expectations,
        "expectations_sha256": index_manifest["coverage_expectations_sha256"],
        "found_task_ids": [0], "missing_task_ids": [], "unexpected_task_ids": [],
        "found_global_skill_ids": [1], "missing_global_skill_ids": [], "unmapped_source_skill_ids": [],
        "source_skill_members_missing_skill_id": 0, "global_vocabulary_complete": True,
        "required_task_skill_pairs": None, "missing_required_task_skill_pairs": None,
        "required_pair_coverage_status": "NOT_DECLARED",
        "unique_input_source_episodes": index_manifest["source_episodes"],
        "unique_candidate_source_episodes": index_manifest["source_episodes"],
        "unique_event_candidates": index_manifest["event_candidates"],
    }
    return index_manifest


def view(event: dict, kind: str) -> dict:
    return {"schema_version": pack.SCHEMA_VERSION, "label_kind": kind, "view_id": "",
            "event_id": event["event_id"], "source_group_id": event["source"]["source_group_id"],
            "observation_frame": event["observation"]["frame"], "annotation_provenance": "agent",
            "review_status": "PROPOSED", "actor_evidence": {"kind": "MISSING", "evidence_end_frame": None,
                                                               "available_frame": None, "references": []}}


def with_id(row: dict) -> dict:
    row["view_id"] = pack._protocol.view_id(row)
    return row


def review(row: dict, **changes: object) -> dict:
    value = {"view_id": row["view_id"], "candidate_annotated": True, "parent_reviewed": False,
             "accepted_auxiliary": False, "outcome_validated": False,
             "reviewer": {"kind": "agent", "id": "annotator-1", "model": "gpt-5.6-luna"}}
    value.update(changes)
    return value


def annotations(event: dict, *, include_nontrainable_action: bool = True) -> dict:
    goal = with_id(view(event, "goal_satisfaction_counterfactual"))
    goal.update(goal_relation="radio is on", goal_satisfaction="UNKNOWN", valid_goal_mask=False,
                attempt_outcome="NOT_APPLICABLE", evidence={"kind": "MISSING", "evidence_end_frame": None,
                                                              "available_frame": None}, low_action_supervision_mask=False)
    with_id(goal)
    outcome = with_id(view(event, "attempt_outcome"))
    outcome.update(attempt_id="attempt-1", parent_attempt_id=None, evaluated_bundle_id=None, attempt_outcome="UNKNOWN",
                   valid_result_mask=False, evidence={"kind": "MISSING", "evidence_end_frame": None,
                                                       "available_frame": None}, low_action_supervision_mask=False)
    with_id(outcome)
    decision = with_id(view(event, "recovery_decision"))
    decision.update(decision="RETRY", target_bundle_id=None, decision_supervision_mask=False,
                    evidence={"kind": "MISSING", "evidence_end_frame": None, "available_frame": None})
    with_id(decision)
    views = [goal, outcome, decision]
    actions: list[dict] = []
    if include_nontrainable_action:
        raw = [[float(i) for i in range(23)], [float(i + 1) for i in range(23)]]
        raw_artifact_sha = "f" * 64
        raw_sha = pack.raw_action_payload_sha256(raw, raw_action_artifact_sha256=raw_artifact_sha)
        payload_sha = pack.action_payload_sha256(raw)
        corrective = with_id(view(event, "corrective_action"))
        corrective.update(recovery_attempt_id="recovery-2", recovery_from_attempt_id="attempt-1",
                          action_intent_bundle_id="bundle-recovery", executed_intent_bundle_id="bundle-recovery",
                          action_start_frame=event["observation"]["frame"], actual_executed_length=2,
                          raw_action_dim=23, model_action_dim=27, model_padding_indices=[7, 8, 17, 18],
                          action_is_pad=[False, False] + [True] * 30,
                          executed_action_receipt={"executed": True, "raw_action_sha256": raw_sha,
                                                    "intent_bundle_id": "bundle-recovery",
                                                    "actual_end_frame": event["observation"]["frame"] + 2},
                          action_payload_sha256=payload_sha, execution_receipt_sha256="d" * 64,
                          recovery_verification_receipt_sha256="e" * 64,
                          low_action_supervision_mask=False,
                          rejection_reason="failed correction is retained as outcome evidence, never FM BC")
        # Updating a canonical field requires a new identity.
        with_id(corrective)
        views.append(corrective)
        actions.append({"view_id": corrective["view_id"], "raw_actions_23": raw, "raw_action_sha256": raw_sha,
                        "raw_action_artifact_sha256": raw_artifact_sha})
    return {"schema_version": pack.ANNOTATION_SCHEMA, "views": views, "view_reviews": [review(row) for row in views],
            "action_payloads": actions,
            "root_visual_review": {"completed": True, "reviewer_kind": "root_model", "artifact": "synthetic-only-root-review", "model": "gpt-5.6-luna"}}


def prelabel_registry_row(event: dict, goal: dict, *, query_id: str = "producer-query-1",
                          text: str | None = None) -> dict:
    skill = event["skill_bundle"][0]
    return {
        "schema_version": "p107.visual_relation_query_prelabel_registry.v1",
        "prelabel_query_id": query_id,
        "event_id": event["event_id"],
        "source_group_id": event["source"]["source_group_id"],
        "observation_frame": event["observation"]["frame"],
        "query_ordinal_within_event": 0,
        "query_text": goal["goal_relation"] if text is None else text,
        "relation_family": "synthetic_goal_relation",
        "canonical_verb": skill["verb"],
        "skill_id": skill.get("skill_id", skill["skill_idx"]),
        "source_pin": {
            "event_id": event["event_id"],
            "phase_index_path": "synthetic/phase-index.jsonl",
            "phase_index_record_sha256": "1" * 64,
            "phase_index_sha256": "2" * 64,
            "queue_path": "synthetic/queue.jsonl",
            "queue_record_sha256": "3" * 64,
            "queue_sha256": "4" * 64,
            "selection_manifest_path": "synthetic/selection-manifest.json",
            "selection_manifest_sha256": "5" * 64,
        },
    }


def write_actor_query_inputs(root: Path, event: dict, goal: dict, *, registry_rows: list[dict] | None = None,
                             binding_query_id: str = "producer-query-1",
                             binding_rows: list[dict] | None = None) -> tuple[Path, str, Path]:
    registry_path, bindings_path = root / "prelabel-registry.jsonl", root / "view-bindings.jsonl"
    rows = registry_rows if registry_rows is not None else [prelabel_registry_row(event, goal)]
    pack._write_jsonl(registry_path, rows)
    rows = binding_rows if binding_rows is not None else [
        {"view_id": goal["view_id"], "prelabel_query_id": binding_query_id}]
    pack._write_jsonl(bindings_path, rows)
    return registry_path, pack.sha256_file(registry_path), bindings_path


def _write_parent_jsonl(path: Path, rows: list[dict]) -> dict:
    path.write_bytes(b"".join(pack.canonical_json(row).encode("utf-8") + b"\n" for row in rows))
    return {"relative_path": path.name, "sha256": pack.sha256_file(path), "bytes": path.stat().st_size, "rows": len(rows)}


def _write_parent_artifact(root: Path, *, name: str, kind: str, event: dict, group: dict,
                           payload: dict) -> dict:
    path = root / "artifact_blobs" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(pack.canonical_json(payload).encode("utf-8"))
    return {"schema_version": "p107-sealed-artifact-v1", "artifact_sha256": pack.sha256_file(path),
            "artifact_kind": kind, "relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "event_id": event["event_id"], "source_group_id": group["source_group_id"]}


def approve_logged_corrective(labels: dict, event: dict, groups: list[dict], publisher_root: Path,
                              *, index_root: Path, index_inventory_seal_sha256: str) -> tuple[dict, str]:
    """Construct one typed, external parent authority over a compact fixture row.

    This is intentionally test-only and mirrors the published protocol fixture:
    every receipt refers to bytes in ``publisher_root`` while index membership is
    checked against the separately sealed ``index_root``.
    """
    publisher_root.mkdir()
    corrective = next(row for row in labels["views"] if row["label_kind"] == "corrective_action")
    raw = labels["action_payloads"][0]["raw_actions_23"]
    group = next(row for row in groups if row["source_group_id"] == event["source"]["source_group_id"])
    raw_artifact = _write_parent_artifact(
        publisher_root, name="raw23.json", kind="raw_action_23_artifact", event=event, group=group,
        payload={"schema_version": "p107-raw23-action-artifact-v1", "raw_actions_23": raw})
    raw_artifact_sha = raw_artifact["artifact_sha256"]
    raw_sha = pack.raw_action_payload_sha256(raw, raw_action_artifact_sha256=raw_artifact_sha)
    payload_sha = pack.action_payload_sha256(raw)
    corrective.update(review_status="PARENT_APPROVED", low_action_supervision_mask=True,
                      action_payload_sha256=payload_sha,
                      executed_action_receipt={**corrective["executed_action_receipt"], "raw_action_sha256": raw_sha})
    def seal(receipt: dict) -> dict:
        receipt["receipt_sha256"] = pack._protocol.receipt_sha256(receipt)
        return receipt
    raw_payload = {"schema_version": "p107-raw23-action-payload-v1", "action_payload_sha256": payload_sha,
                   "raw_action_sha256": raw_sha, "raw_action_artifact_sha256": raw_artifact_sha,
                   "raw_actions_23": raw}
    execution = seal({"schema_version": "p107-execution-receipt-v1", "receipt_sha256": "", "event_id": event["event_id"],
                      "source_group_id": event["source"]["source_group_id"],
                      "source_release_manifest_sha256": event["source"]["source_release_manifest_sha256"],
                      "raw_episode_id": event["source"]["raw_episode_id"], "episode_index": event["source"]["episode_index"],
                      "raw_action_sha256": raw_sha, "action_payload_sha256": payload_sha,
                      "raw_action_artifact_sha256": raw_artifact_sha,
                      "action_start_frame": corrective["action_start_frame"],
                      "actual_end_frame": corrective["executed_action_receipt"]["actual_end_frame"],
                      "actual_executed_length": corrective["actual_executed_length"], "raw_action_dim": 23,
                      "intent_bundle_id": corrective["executed_intent_bundle_id"], "executed": True})
    action_end = corrective["executed_action_receipt"]["actual_end_frame"]
    evidence_artifact = _write_parent_artifact(
        publisher_root, name="temporal-evidence.json", kind="temporal_review_evidence", event=event, group=group,
        payload={"schema_version": "p107-temporal-review-evidence-v1", "event_id": event["event_id"],
                 "source_group_id": group["source_group_id"], "raw_action_sha256": raw_sha,
                 "evidence_end_frame": action_end + 1, "available_frame": action_end + 1,
                 "verification_frame": action_end + 1})
    verification_artifact = _write_parent_artifact(
        publisher_root, name="verification-receipt.json", kind="verification_receipt_artifact", event=event, group=group,
        payload={"schema_version": "p107-verification-receipt-artifact-v1", "event_id": event["event_id"],
                 "source_group_id": group["source_group_id"], "raw_action_sha256": raw_sha,
                 "verification_frame": action_end + 1})
    temporal_artifact = _write_parent_artifact(
        publisher_root, name="temporal-review.json", kind="temporal_review_artifact", event=event, group=group,
        payload={"schema_version": "p107-temporal-review-artifact-v1", "event_id": event["event_id"],
                 "source_group_id": group["source_group_id"], "raw_action_sha256": raw_sha,
                 "verification_frame": action_end + 1})
    verification = seal({"schema_version": "p107-recovery-verification-receipt-v1", "receipt_sha256": "",
                         "event_id": event["event_id"], "source_group_id": event["source"]["source_group_id"],
                         "recovery_attempt_id": corrective["recovery_attempt_id"],
                         "recovery_from_attempt_id": corrective["recovery_from_attempt_id"],
                         "action_intent_bundle_id": corrective["action_intent_bundle_id"], "raw_action_sha256": raw_sha,
                         "evidence_origin": "logged_demonstration", "quality_tier": "SILVER_REVIEWED_LOGGED_DEMONSTRATION",
                         "verification_frame": action_end + 1,
                         "label_available_frame": action_end + 2,
                         "branch_or_episode_final_frame": action_end + 3,
                         "evidence": {"kind": "TEMPORAL_REVIEW_FROZEN_PARQUET",
                                      "evidence_end_frame": action_end + 1, "available_frame": action_end + 1,
                                      "artifact_sha256": evidence_artifact["artifact_sha256"]},
                         "verification_artifact_sha256": verification_artifact["artifact_sha256"],
                         "frozen_action_window_artifact_sha256": raw_artifact_sha,
                         "temporal_review_artifact_sha256": temporal_artifact["artifact_sha256"],
                         "live_runtime_acceptance_root_sha256": None})
    corrective["execution_receipt_sha256"] = execution["receipt_sha256"]
    corrective["recovery_verification_receipt_sha256"] = verification["receipt_sha256"]
    with_id(corrective)
    parent_artifact = _write_parent_artifact(
        publisher_root, name="parent-review.json", kind="parent_review_artifact", event=event, group=group,
        payload={"schema_version": "p107-parent-review-artifact-v1", "view_id": corrective["view_id"],
                 "event_id": event["event_id"], "source_group_id": group["source_group_id"],
                 "raw_action_sha256": raw_sha})
    files = {
        "events": _write_parent_jsonl(publisher_root / "events.jsonl", [event]),
        "source_groups": _write_parent_jsonl(publisher_root / "source_groups.jsonl", [group]),
        "raw_action_payloads": _write_parent_jsonl(publisher_root / "raw_action_payloads.jsonl", [raw_payload]),
        "execution_receipts": _write_parent_jsonl(publisher_root / "execution_receipts.jsonl", [execution]),
        "verification_receipts": _write_parent_jsonl(publisher_root / "verification_receipts.jsonl", [verification]),
        "artifacts": _write_parent_jsonl(publisher_root / "artifacts.jsonl", [
            raw_artifact, evidence_artifact, verification_artifact, temporal_artifact, parent_artifact]),
    }
    approval = seal({"schema_version": "p107-parent-low-fm-approval-v1", "receipt_sha256": "",
                     "approval_status": "APPROVED_FOR_LOW_FM", "view_id": corrective["view_id"], "event_id": event["event_id"],
                     "source_group_id": group["source_group_id"],
                     "source_release_manifest_sha256": group["source_release_manifest_sha256"],
                     "raw_episode_id": event["source"]["raw_episode_id"], "episode_index": event["source"]["episode_index"],
                     "original_split": group["original_split"], "usage_role": group["usage_role"],
                     "action_intent_bundle_id": corrective["action_intent_bundle_id"],
                     "executed_intent_bundle_id": corrective["executed_intent_bundle_id"], "raw_action_sha256": raw_sha,
                     "action_payload_sha256": payload_sha, "execution_receipt_sha256": execution["receipt_sha256"],
                     "recovery_verification_receipt_sha256": verification["receipt_sha256"],
                     "parent_review_artifact_sha256": parent_artifact["artifact_sha256"],
                     "evidence_origin": "logged_demonstration",
                     "quality_tier": "SILVER_REVIEWED_LOGGED_DEMONSTRATION",
                     "index_inventory_seal_sha256": index_inventory_seal_sha256,
                     "action_payload_root_sha256": files["raw_action_payloads"]["sha256"],
                     "execution_receipt_root_sha256": files["execution_receipts"]["sha256"],
                     "recovery_verification_receipt_root_sha256": files["verification_receipts"]["sha256"],
                     "artifact_manifest_root_sha256": files["artifacts"]["sha256"]})
    files["parent_approval_receipts"] = _write_parent_jsonl(
        publisher_root / "parent_approval_receipts.jsonl", [approval])
    publisher_manifest = {"schema_version": "p107-corrective-publisher-manifest-v1",
                          "index_inventory_seal_sha256": index_inventory_seal_sha256,
                          "accepted_live_runtime_root_sha256": None, "files": files}
    manifest_path = publisher_root / "publisher_manifest.json"
    manifest_path.write_bytes(pack.canonical_json(publisher_manifest).encode("utf-8"))
    # Agent annotations intentionally carry no positive raw action payload.
    labels["action_payloads"] = []
    labels["view_reviews"] = [review(row, parent_reviewed=(row["view_id"] == corrective["view_id"]))
                              for row in labels["views"]]
    # Loading now proves the parent rows match this exact *external* index.
    pack.load_corrective_action_authority(
        publisher_root, index_root=index_root, expected_publisher_manifest_sha256=pack.sha256_file(manifest_path),
        expected_index_inventory_seal_sha256=index_inventory_seal_sha256)
    return labels, pack.sha256_file(manifest_path)


class PackMemLiteEventLabelsTests(unittest.TestCase):
    def test_candidate_package_preserves_unknown_masks_actions_and_never_self_authorizes_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, manifest = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            labels = annotations(student)
            label_path, output = root / "labels.json", root / "package"
            label_path.write_text(json.dumps(labels))
            pilot_path = root / "aux-pilot.json"
            pilot_path.write_text(json.dumps({"schema_version": "p107.visual_label.v1", "annotator_model": "gpt-5.6-luna",
                                               "human_reviewed": False, "accepted_scope": "visual_relation_calibration_only",
                                               "records": [{"split": "train", "queries": [{"outcome_loss_mask": False,
                                                                                                  "action_loss_mask": False}]}]}))
            result = pack.publish(index_path, label_path, output, release_role="student_candidate", minimum_scale=None,
                                  request_dataset_quality_eligibility=False,
                                  expected_index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"),
                                  auxiliary_pilots=[pilot_path])
            self.assertEqual(result["release_eligibility"], "CANDIDATE_ONLY")
            self.assertFalse(result["ready_for_training"])
            self.assertFalse(result["stage3_training_authorized"])
            self.assertEqual(result["status_report"]["auxiliary_visual_calibration_pilots"][0]["release_eligibility"],
                             "AUXILIARY_CALIBRATION_ONLY")
            saved = pack.read_jsonl(output / "views.jsonl")
            goal = next(row for row in saved if row["view"]["label_kind"] == "goal_satisfaction_counterfactual")
            attempt = next(row for row in saved if row["view"]["label_kind"] == "attempt_outcome")
            action = next(row for row in saved if row["view"]["label_kind"] == "corrective_action")
            self.assertFalse(goal["view"]["valid_goal_mask"])
            self.assertEqual(goal["view"]["goal_satisfaction"], "UNKNOWN")
            self.assertFalse(attempt["view"]["valid_result_mask"])
            self.assertEqual(attempt["view"]["attempt_outcome"], "UNKNOWN")
            self.assertFalse(action["dataset_quality_gates"]["corrective_fm"])
            stored = pack.read_jsonl(output / "actions.jsonl")[0]
            self.assertEqual(len(stored["actions_27"]), 32)
            self.assertEqual(stored["action_is_pad"], [False, False] + [True] * 30)
            self.assertEqual([stored["actions_27"][0][i] for i in (7, 8, 17, 18)], [0.0] * 4)

    def test_publisher_requires_external_index_seal_and_separate_parent_authority(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, index_manifest = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            labels = annotations(student)
            labels["corrective_action_authority"] = {"execution_receipts": [], "verification_receipts": [],
                                                      "parent_approval_receipts": [], "parent_review_artifact_sha256s": []}
            with self.assertRaisesRegex(ValueError, "queue/approval authority is external"):
                pack.build_release(groups, events, labels, index_manifest=index_manifest,
                                   release_role="student_candidate", minimum_scale=None,
                                   request_dataset_quality_eligibility=False,
                                   index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"))
            labels.pop("corrective_action_authority")
            labels_path = root / "labels.json"
            labels_path.write_text(pack.canonical_json(labels))
            with self.assertRaisesRegex(ValueError, "externally recorded immutable index inventory seal"):
                pack.publish(index_path, labels_path, root / "rejected-package", release_role="student_candidate",
                             minimum_scale=None, request_dataset_quality_eligibility=False,
                             expected_index_inventory_seal_sha256="0" * 64)
            self.assertFalse((root / "rejected-package").exists())

    def test_wrong_intent_old_expert_and_recursive_private_actor_reference_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, groups, events, index_manifest = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            bad = annotations(student)
            corrective = next(row for row in bad["views"] if row["label_kind"] == "corrective_action")
            corrective["low_action_supervision_mask"] = True
            corrective["executed_intent_bundle_id"] = "old-expert-bundle"
            corrective["executed_action_receipt"]["intent_bundle_id"] = "old-expert-bundle"
            with_id(corrective)
            bad["view_reviews"] = [review(row) for row in bad["views"]]
            with self.assertRaisesRegex(ValueError, "same intent bundle"):
                pack.build_release(groups, events, bad, index_manifest=index_manifest, release_role="student_candidate",
                                   minimum_scale=None, request_dataset_quality_eligibility=False,
                                   index_inventory_seal_sha256=pack.sha256_file(root / "index" / "inventory_seal.json"))
            leaked = annotations(student, include_nontrainable_action=False)
            goal = leaked["views"][0]
            goal.update(goal_satisfaction="NOT_SATISFIED", valid_goal_mask=True,
                        evidence={"kind": "VISUAL", "evidence_end_frame": 0, "available_frame": 0},
                        actor_evidence={"kind": "VISUAL", "evidence_end_frame": 0, "available_frame": 0,
                                        "references": [{"object_pose": "hidden provider fact"}]})
            with_id(goal)
            leaked["view_reviews"] = [review(row) for row in leaked["views"]]
            with self.assertRaisesRegex(ValueError, "forbidden nested key"):
                pack.build_release(groups, events, leaked, index_manifest=index_manifest, release_role="student_candidate",
                                   minimum_scale=None, request_dataset_quality_eligibility=False,
                                   index_inventory_seal_sha256=pack.sha256_file(root / "index" / "inventory_seal.json"))

    def test_calibration_group_cannot_become_student_fm_and_pilot_masks_must_stay_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, groups, events, index_manifest = indexed(root)
            calibration = next(event for event in events if event["usage_role"] == "annotation_calibration")
            labels = annotations(calibration)
            corrective = next(row for row in labels["views"] if row["label_kind"] == "corrective_action")
            corrective["low_action_supervision_mask"] = True
            with_id(corrective)
            labels["action_payloads"][0]["view_id"] = corrective["view_id"]
            labels["view_reviews"] = [review(row) for row in labels["views"]]
            with self.assertRaisesRegex(ValueError, "canonical indexed event plus external parent authority"):
                pack.build_release(groups, events, labels, index_manifest=index_manifest,
                                   release_role="student_candidate", minimum_scale={"source_episodes": 1,
                                   "goal_outcome_windows": 1, "corrective_action_windows": 1}, request_dataset_quality_eligibility=True,
                                   index_inventory_seal_sha256=pack.sha256_file(root / "index" / "inventory_seal.json"))
            pilot = {"schema_version": "p107.visual_label.v1", "annotator_model": "gpt-5.6-luna", "human_reviewed": False,
                     "accepted_scope": "visual_relation_calibration_only", "records": [{"split": "train", "queries": [
                         {"outcome_loss_mask": False, "action_loss_mask": False}]}]}
            pilot_path = root / "pilot.json"
            pilot_path.write_text(json.dumps(pilot))
            receipt = pack._pilot_receipt(pilot_path)
            self.assertEqual(receipt["release_eligibility"], "AUXILIARY_CALIBRATION_ONLY")
            pilot["records"][0]["queries"][0]["action_loss_mask"] = True
            pilot_path.write_text(json.dumps(pilot))
            with self.assertRaisesRegex(ValueError, "zero outcome/action masks"):
                pack._pilot_receipt(pilot_path)

    def test_calibration40_candidate_receipt_cannot_supply_labels_authority_or_scale(self):
        """A queue receipt is provenance, not a SILVER/GOLD or FM authority."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, index_manifest = indexed(root)
            calibration = next(event for event in events if event["usage_role"] == "annotation_calibration")
            labels = annotations(calibration)
            labels["candidate_queue_receipt"] = {
                "schema_version": "p107-annotation-queue-v1", "queue_status": "CANDIDATE_MISSING_EVIDENCE",
                "training_eligible": False, "quality_tier": "SILVER_REVIEWED_LOGGED_DEMONSTRATION",
            }
            with self.assertRaisesRegex(ValueError, "queue/approval authority is external"):
                pack.build_release(groups, events, labels, index_manifest=index_manifest,
                                   release_role="annotation_calibration",
                                   minimum_scale={"source_episodes": 1, "goal_outcome_windows": 1,
                                                  "corrective_action_windows": 1},
                                   request_dataset_quality_eligibility=True,
                                   require_complete_source_index=True,
                                   index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"))
            labels.pop("candidate_queue_receipt")
            label_path, output = root / "labels.json", root / "calibration-candidate-package"
            label_path.write_text(pack.canonical_json(labels))
            result = pack.publish(index_path, label_path, output, release_role="annotation_calibration",
                                  minimum_scale={"source_episodes": 1, "goal_outcome_windows": 1,
                                                 "corrective_action_windows": 1},
                                  request_dataset_quality_eligibility=True, require_complete_source_index=True,
                                  expected_index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"))
            self.assertEqual(result["release_eligibility"], "CANDIDATE_ONLY")
            self.assertEqual(result["actual_scale"]["source_episodes"], 0)
            self.assertEqual(result["actual_scale"]["goal_outcome_windows"], 0)
            self.assertEqual(result["actual_scale"]["corrective_action_windows"], 0)
            self.assertEqual(result["status_report"]["verified_recovery_actions"], 0)
            self.assertFalse(result["dataset_quality_eligibility"])
            # Even a typed parent root cannot turn a calibration group into a
            # student FM positive: its approval role is rejected at authority load.
            with self.assertRaisesRegex(ValueError, "heldout/calibration"):
                approve_logged_corrective(annotations(calibration), calibration, groups, root / "calibration-parent",
                                         index_root=index_path,
                                         index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"))

    def test_dart_intended_or_noise_controls_are_not_accepted_as_actual_raw23_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, index_manifest = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            labels = annotations(student)
            labels["action_payloads"][0]["a_intended"] = [0.0] * 23
            with self.assertRaisesRegex(ValueError, "DART intended/noise controls are unsupported"):
                pack.build_release(groups, events, labels, index_manifest=index_manifest,
                                   release_role="student_candidate", minimum_scale=None,
                                   request_dataset_quality_eligibility=False,
                                   index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"))

    def test_scale_counts_unique_event_windows_not_counterfactual_copies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, _ = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            labels = annotations(student)
            outcome = next(row for row in labels["views"] if row["label_kind"] == "attempt_outcome")
            outcome.update(evaluated_bundle_id="bundle-before", attempt_outcome="FAILED", valid_result_mask=True,
                           evidence={"kind": "VISUAL", "evidence_end_frame": student["observation"]["frame"],
                                     "available_frame": student["observation"]["frame"]})
            with_id(outcome)
            copied = deepcopy(outcome)
            copied["attempt_id"] = "attempt-copy-same-window"
            with_id(copied)
            labels["views"].append(copied)
            index_seal = pack.sha256_file(index_path / "inventory_seal.json")
            labels, authority_manifest_sha = approve_logged_corrective(
                labels, student, groups, root / "parent-publisher", index_root=index_path,
                index_inventory_seal_sha256=index_seal)
            labels["view_reviews"] = [review(row, parent_reviewed=(row["label_kind"] == "corrective_action"),
                                                outcome_validated=(row["label_kind"] == "attempt_outcome"))
                                      for row in labels["views"]]
            label_path, output = root / "labels.json", root / "deduplicated-scale-package"
            label_path.write_text(pack.canonical_json(labels))
            result = pack.publish(index_path, label_path, output, release_role="student_candidate",
                                  minimum_scale={"source_episodes": 1, "goal_outcome_windows": 2,
                                                 "corrective_action_windows": 1},
                                  request_dataset_quality_eligibility=True, require_complete_source_index=True,
                                  expected_index_inventory_seal_sha256=index_seal,
                                  corrective_publisher_root=root / "parent-publisher",
                                  expected_corrective_publisher_manifest_sha256=authority_manifest_sha)
            self.assertEqual(result["actual_scale"], {"source_episodes": 1, "goal_outcome_windows": 1,
                                                       "corrective_action_windows": 1,
                                                       "decision_source_episodes": 0,
                                                       "corrective_action_source_episodes": 1})
            self.assertEqual(result["release_eligibility"], "CANDIDATE_ONLY")

    def test_ten_thousand_goal_outcome_windows_need_goal_outcome_source_episodes(self):
        """Decision/action-only episodes must not pad the goal/outcome source minimum."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, index_manifest = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            base_group = next(group for group in groups if group["source_group_id"] == student["source"]["source_group_id"])
            labels = annotations(student)
            base_action = next(row for row in labels["views"] if row["label_kind"] == "corrective_action")
            labels["views"] = [base_action]
            index_seal = pack.sha256_file(index_path / "inventory_seal.json")
            labels, authority_manifest_sha = approve_logged_corrective(
                labels, student, groups, root / "parent-publisher", index_root=index_path,
                index_inventory_seal_sha256=index_seal)

            # Synthetic accounting only: this loop reuses the fixture frame/
            # interval while varying event kind and bundle identity. It proves
            # the source-episode partition gate, not real-window uniqueness.
            # Its accepted-window count reaches 10k while goal/outcome source
            # episodes remain one.
            outcome_events: list[dict] = []
            for number in range(10_000):
                event = deepcopy(student)
                event["event_kind"] = f"ANNOTATED_SKILL_SEGMENT_{number}"
                event["bundle_id"] = f"bundle-{number}"
                event["event_id"] = pack._protocol.event_id(event)
                outcome_events.append(event)
                outcome = with_id(view(event, "attempt_outcome"))
                outcome.update(attempt_id=f"attempt-{number}", parent_attempt_id=None,
                               evaluated_bundle_id="bundle-before", attempt_outcome="FAILED",
                               valid_result_mask=True,
                               evidence={"kind": "VISUAL", "evidence_end_frame": event["observation"]["frame"],
                                         "available_frame": event["observation"]["frame"]},
                               low_action_supervision_mask=False)
                labels["views"].append(with_id(outcome))

            def extra_source(seed: int) -> tuple[dict, dict]:
                event, group = deepcopy(student), deepcopy(base_group)
                event["source"].update(task_instance_id=100 + seed, raw_episode_id=5000 + seed,
                                       episode_index=8000 + seed)
                event["source"]["source_group_id"] = pack._protocol.source_group_id(event["source"])
                event["event_id"] = pack._protocol.event_id(event)
                group.update(source_group_id=event["source"]["source_group_id"],
                             task_instance_id=event["source"]["task_instance_id"],
                             source_episode_ids=[event["source"]["episode_index"]], source_episode_count=1)
                return event, group

            decision_event, decision_group = extra_source(1)
            decision = with_id(view(decision_event, "recovery_decision"))
            decision.update(decision="RETRY", target_bundle_id="bundle-recovery", decision_supervision_mask=True,
                            evidence={"kind": "VISUAL", "evidence_end_frame": decision_event["observation"]["frame"],
                                      "available_frame": decision_event["observation"]["frame"]})
            labels["views"].append(with_id(decision))
            action_event, action_group = extra_source(2)
            action_labels = annotations(action_event)
            action_only = next(row for row in action_labels["views"] if row["label_kind"] == "corrective_action")
            labels["views"].append(action_only)
            labels["action_payloads"].append(action_labels["action_payloads"][0])
            labels["view_reviews"] = [review(
                row, parent_reviewed=(row["label_kind"] == "corrective_action" and
                                       row["event_id"] == student["event_id"]),
                outcome_validated=(row["label_kind"] == "attempt_outcome"),
                accepted_auxiliary=(row["label_kind"] == "recovery_decision"))
                for row in labels["views"]]

            result, _ = pack.build_release(
                groups + [decision_group, action_group], events + outcome_events + [decision_event, action_event], labels,
                index_manifest=index_manifest, release_role="student_candidate",
                minimum_scale={"source_episodes": 2, "goal_outcome_windows": 10_000,
                               "corrective_action_windows": 1}, request_dataset_quality_eligibility=True,
                require_complete_source_index=True, index_inventory_seal_sha256=index_seal,
                corrective_action_authority=pack.load_corrective_action_authority(
                    root / "parent-publisher", index_root=index_path,
                    expected_publisher_manifest_sha256=authority_manifest_sha,
                    expected_index_inventory_seal_sha256=index_seal),
                corrective_authority_capability={
                    "publisher_manifest_sha256": authority_manifest_sha,
                    "index_inventory_seal_sha256": index_seal,
                    "accepted_live_runtime_acceptance_root_sha256": None})
            self.assertEqual(result["actual_scale"], {
                "source_episodes": 1, "goal_outcome_windows": 10_000, "corrective_action_windows": 1,
                "decision_source_episodes": 1, "corrective_action_source_episodes": 1})
            self.assertFalse(result["dataset_quality_eligibility"])
            self.assertEqual(result["release_eligibility"], "CANDIDATE_ONLY")

    def test_conflicting_published_episode_index_is_rejected_even_if_event_hash_is_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, groups, events, index_manifest = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            collision = deepcopy(student)
            collision["source"]["raw_episode_id"] += 999
            # Source-group identity intentionally excludes raw episode IDs, so
            # this models a malformed index that otherwise has valid hashes.
            collision["source"]["episode_index"] = next(event for event in events if event["usage_role"] == "annotation_calibration")["source"]["episode_index"]
            collision["event_id"] = pack._protocol.event_id(collision)
            with self.assertRaisesRegex(ValueError, "published episode_index"):
                pack.build_release(groups, events + [collision], annotations(student), index_manifest=index_manifest,
                                   release_role="student_candidate", minimum_scale=None, request_dataset_quality_eligibility=False,
                                   index_inventory_seal_sha256=pack.sha256_file(root / "index" / "inventory_seal.json"))

    def test_parent_authority_can_quality_qualify_logged_actions_without_authorizing_stage3_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, _ = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            candidate = annotations(student)
            outcome = next(row for row in candidate["views"] if row["label_kind"] == "attempt_outcome")
            outcome.update(evaluated_bundle_id="bundle-before", attempt_outcome="FAILED", valid_result_mask=True,
                           evidence={"kind": "VISUAL", "evidence_end_frame": student["observation"]["frame"],
                                     "available_frame": student["observation"]["frame"]})
            with_id(outcome)
            index_seal = pack.sha256_file(index_path / "inventory_seal.json")
            labels, authority_manifest_sha = approve_logged_corrective(
                candidate, student, groups, root / "parent-publisher",
                index_root=index_path, index_inventory_seal_sha256=index_seal)
            corrective = next(row for row in labels["views"] if row["label_kind"] == "corrective_action")
            labels["view_reviews"] = [review(row, parent_reviewed=(row is corrective),
                                                outcome_validated=(row is outcome)) for row in labels["views"]]
            path, output = root / "labels.json", root / "quality-package"
            path.write_text(pack.canonical_json(labels))
            result = pack.publish(index_path, path, output, release_role="student_candidate",
                                  minimum_scale={"source_episodes": 1, "goal_outcome_windows": 1,
                                                 "corrective_action_windows": 1},
                                  request_dataset_quality_eligibility=True, require_complete_source_index=True,
                                  expected_index_inventory_seal_sha256=index_seal,
                                  corrective_publisher_root=root / "parent-publisher",
                                  expected_corrective_publisher_manifest_sha256=authority_manifest_sha)
            action = next(row for row in pack.read_jsonl(output / "views.jsonl") if row["view"]["label_kind"] == "corrective_action")
            self.assertTrue(action["dataset_quality_gates"]["corrective_fm"])
            self.assertTrue(result["dataset_quality_eligibility"])
            self.assertEqual(result["release_eligibility"], "DATASET_QUALITY_QUALIFIED_PENDING_INDEPENDENT_REVIEW")
            self.assertFalse(result["ready_for_training"])
            self.assertFalse(result["training_run_authorized"])
            self.assertFalse(result["stage3_training_authorized"])

    def test_streamed_selected_loader_preserves_full_index_duplicate_and_lineage_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, _ = indexed(root)
            seal = pack.sha256_file(index_path / "inventory_seal.json")
            selected_id = events[0]["event_id"]
            loaded_groups, loaded_events, _, _ = pack._read_sealed_index(
                index_path, expected_inventory_seal_sha256=seal, selected_event_ids={selected_id})
            self.assertEqual(len(loaded_groups), len(groups))
            self.assertEqual([row["event_id"] for row in loaded_events], [selected_id])

            # The duplicate is deliberately not selected.  The old complete
            # index invariant must still reject it.
            with (index_path / "event_candidates.jsonl").open("ab") as stream:
                stream.write(pack.canonical_json(events[1]).encode("utf-8") + b"\n")
            with self.assertRaisesRegex(ValueError, "duplicate event_id"):
                pack._read_sealed_index(index_path, expected_inventory_seal_sha256=seal,
                                        selected_event_ids={selected_id})

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, _, events, _ = indexed(root)
            seal = pack.sha256_file(index_path / "inventory_seal.json")
            unselected = deepcopy(events[1])
            unselected["source"]["task_instance_id"] += 1000
            unselected["source"]["episode_index"] += 100000
            unselected["source"]["source_group_id"] = pack._protocol.source_group_id(unselected["source"])
            unselected["event_id"] = pack._protocol.event_id(unselected)
            with (index_path / "event_candidates.jsonl").open("ab") as stream:
                stream.write(pack.canonical_json(unselected).encode("utf-8") + b"\n")
            with self.assertRaisesRegex(ValueError, "no immutable source-group"):
                pack._read_sealed_index(index_path, expected_inventory_seal_sha256=seal,
                                        selected_event_ids={events[0]["event_id"]})

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, _, _, _ = indexed(root)
            seal = pack.sha256_file(index_path / "inventory_seal.json")
            with self.assertRaisesRegex(ValueError, "selected event IDs are missing"):
                pack._read_sealed_index(index_path, expected_inventory_seal_sha256=seal,
                                        selected_event_ids={"f" * 64})

    def test_streamed_loader_rejects_tampered_hash_or_byte_receipt(self):
        for receipt_field, value in (("sha256", "0" * 64), ("bytes", None)):
            with self.subTest(receipt_field=receipt_field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                index_path, _, events, _ = indexed(root)
                manifest_path = index_path / "manifest.json"
                manifest = pack.read_json(manifest_path)
                if receipt_field == "bytes":
                    value = manifest["files"]["event_candidates.jsonl"]["bytes"] + 1
                manifest["files"]["event_candidates.jsonl"][receipt_field] = value
                manifest_path.write_bytes(pack.canonical_json(manifest).encode("utf-8") + b"\n")
                seal = reseal_index(index_path)
                with self.assertRaisesRegex(ValueError, "event index receipt mismatch: event_candidates.jsonl"):
                    pack._read_sealed_index(index_path, expected_inventory_seal_sha256=seal,
                                            selected_event_ids={events[0]["event_id"]})

    def test_streamed_loader_keeps_strict_json_for_every_row(self):
        for malformed, error in ((b'{"event_id":"x","event_id":"y"}\n', "duplicate JSON key"),
                                 (b'{"event_id":\n', "Expecting value")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                index_path, _, events, _ = indexed(root)
                with (index_path / "event_candidates.jsonl").open("ab") as stream:
                    stream.write(malformed)
                with self.assertRaisesRegex(ValueError, error):
                    pack._read_sealed_index(
                        index_path, expected_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"),
                        selected_event_ids={events[0]["event_id"]})

    def test_streamed_publish_matches_legacy_small_fixture_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, index_manifest = indexed(root)
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            labels = annotations(student)
            label_path = root / "labels.json"
            label_path.write_bytes(pack.canonical_json(labels).encode("utf-8") + b"\n")
            expected, output = root / "expected", root / "streamed-package"
            expected.mkdir()
            seal = pack.sha256_file(index_path / "inventory_seal.json")
            expected_manifest, expected_records = pack.build_release(
                groups, events, labels, index_manifest=index_manifest, release_role="student_candidate",
                minimum_scale=None, request_dataset_quality_eligibility=False,
                index_inventory_seal_sha256=seal)
            expected_files = {
                "source_groups.jsonl": pack._write_jsonl(expected / "source_groups.jsonl", expected_records["source_groups"]),
                "events.jsonl": pack._write_jsonl(expected / "events.jsonl", expected_records["events"]),
                "views.jsonl": pack._write_jsonl(expected / "views.jsonl", expected_records["views"]),
                "actions.jsonl": pack._write_jsonl(expected / "actions.jsonl", expected_records["actions"]),
            }
            expected_manifest["files"] = expected_files
            expected_manifest["publisher_sha256"] = pack.sha256_file(REPO / "scripts/data/pack_memlite_event_labels.py")
            (expected / "release_manifest.json").write_bytes(
                pack.canonical_json(expected_manifest).encode("utf-8") + b"\n")
            expected_seal = {"schema_version": "memlite-event-label-release-seal-v1",
                             "release_manifest_sha256": pack.sha256_file(expected / "release_manifest.json"),
                             "files": expected_files, "publisher_sha256": expected_manifest["publisher_sha256"]}
            (expected / "release_seal.json").write_bytes(pack.canonical_json(expected_seal).encode("utf-8") + b"\n")

            result = pack.publish(index_path, label_path, output, release_role="student_candidate",
                                  minimum_scale=None, request_dataset_quality_eligibility=False,
                                  expected_index_inventory_seal_sha256=seal)
            self.assertEqual(result["status_report"]["source_indexed"], len(events))
            for name in ("source_groups.jsonl", "events.jsonl", "views.jsonl", "actions.jsonl",
                         "release_manifest.json", "release_seal.json"):
                self.assertEqual((output / name).read_bytes(), (expected / name).read_bytes(), name)

    def test_selected_stream_peak_is_bounded_by_selected_state_not_jsonl_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, _, events, _ = indexed(root)
            selected_id = write_streaming_fixture(index_path, events[0], row_count=4096)
            payload_bytes = (index_path / "event_candidates.jsonl").stat().st_size
            seal = pack.sha256_file(index_path / "inventory_seal.json")
            tracemalloc.start()
            try:
                _, selected, _, _ = pack._read_sealed_index(
                    index_path, expected_inventory_seal_sha256=seal, selected_event_ids={selected_id})
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertEqual([row["event_id"] for row in selected], [selected_id])
            self.assertLess(peak, payload_bytes // 2)
            self.assertLess(peak, 8 * 1024 * 1024)

    def test_real_full_v3_coverage_metadata_is_diagnostic_not_cartesian_complete(self):
        manifest = real_full_v3_coverage_metadata()
        normalized = pack._normalize_index_coverage(manifest)
        self.assertEqual(normalized, {
            "schema": "p107-official-coverage-expectations-v3",
            "coverage_complete": False,
            "diagnostic_only": True,
            "missing_pairs": None,
        })

        for path, value, error in (
                (("coverage", "required_pair_coverage_status"), "COMPLETE", "null missing pairs"),
                (("coverage", "missing_required_task_skill_pairs"), [], "null missing pairs"),
                (("coverage", "global_vocabulary_complete"), False, "global vocabulary status"),
                (("source_release_manifest_sha256",), "0" * 64, "task metadata pin")):
            with self.subTest(path=path):
                malformed = deepcopy(manifest)
                target = malformed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(ValueError, error):
                    pack._normalize_index_coverage(malformed)
        malformed = deepcopy(manifest)
        malformed["coverage"].pop("found_global_skill_ids")
        with self.assertRaisesRegex(ValueError, "found_global_skill_ids"):
            pack._normalize_index_coverage(malformed)

    def test_v3_diagnostic_coverage_is_calibration_only_and_never_formally_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, groups, events, index_manifest = indexed(root)
            v3_manifest = v3_diagnostic_coverage(index_manifest)
            calibration = next(event for event in events if event["usage_role"] == "annotation_calibration")
            result, _ = pack.build_release(
                groups, events, annotations(calibration), index_manifest=v3_manifest,
                release_role="annotation_calibration", minimum_scale=None,
                request_dataset_quality_eligibility=False,
                index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"))
            self.assertFalse(result["source_index_complete"])
            self.assertFalse(result["dataset_quality_eligibility"])
            self.assertEqual(result["release_eligibility"], "CANDIDATE_ONLY")
            self.assertEqual(result["source_index_coverage"]["required_pair_coverage_status"], "NOT_DECLARED")
            self.assertIsNone(result["exclusions"]["official_coverage_missing_grid"])
            student = next(event for event in events if event["usage_role"] == "student_candidate")
            with self.assertRaisesRegex(ValueError, "annotation_calibration only"):
                pack.build_release(
                    groups, events, annotations(student), index_manifest=v3_manifest,
                    release_role="student_candidate", minimum_scale=None,
                    request_dataset_quality_eligibility=False,
                    index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"))

    def test_optional_actor_query_sidecar_preserves_frozen_registry_and_calibration_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, _, events, _ = indexed(root)
            calibration = next(event for event in events if event["usage_role"] == "annotation_calibration")
            labels = annotations(calibration)
            goal = next(row for row in labels["views"] if row["label_kind"] == "goal_satisfaction_counterfactual")
            registry_path, registry_sha, bindings_path = write_actor_query_inputs(root, calibration, goal)
            labels_path, output = root / "labels.json", root / "sidecar-package"
            labels_path.write_bytes(pack.canonical_json(labels).encode("utf-8") + b"\n")
            with self.assertRaisesRegex(ValueError, "requires registry, expected registry SHA-256, and view/query bindings together"):
                pack.publish(
                    index_path, labels_path, root / "incomplete-sidecar-package", release_role="annotation_calibration",
                    minimum_scale=None, request_dataset_quality_eligibility=False,
                    expected_index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"),
                    prelabel_query_registry=registry_path,
                    expected_prelabel_query_registry_sha256=registry_sha)
            self.assertFalse((root / "incomplete-sidecar-package").exists())
            result = pack.publish(
                index_path, labels_path, output, release_role="annotation_calibration", minimum_scale=None,
                request_dataset_quality_eligibility=False,
                expected_index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"),
                prelabel_query_registry=registry_path,
                expected_prelabel_query_registry_sha256=registry_sha,
                postlabel_view_prelabel_query_bindings=bindings_path)
            manifest = pack.read_json(output / "release_manifest.json")
            sidecar = manifest["actor_query_sidecar"]
            self.assertEqual(sidecar["prelabel_query_registry"]["sha256"], registry_sha)
            self.assertEqual((output / pack.PRELABEL_QUERY_REGISTRY_PATH).read_bytes(), registry_path.read_bytes())
            self.assertEqual(pack.read_jsonl(output / pack.ACTOR_QUERY_SIDECAR_PATH)[0]["actor_query"]["text"],
                             goal["goal_relation"])
            self.assertEqual(result["release_eligibility"], "CANDIDATE_ONLY")
            self.assertFalse(result["ready_for_training"])
            self.assertFalse(result["dataset_quality_eligibility"])
            self.assertEqual(result["actual_scale"], {
                "source_episodes": 0, "goal_outcome_windows": 0, "corrective_action_windows": 0,
                "decision_source_episodes": 0, "corrective_action_source_episodes": 0,
            })
            self.assertTrue(all(not value for value in pack.read_jsonl(output / "views.jsonl")[0]["dataset_quality_gates"].values()))
            audited = release_audit.audit(
                output, expected_release_seal_sha256=result["release_seal_sha256"])
            self.assertEqual(audited["actor_query_sidecar_rows"], 1)
            self.assertEqual(audited["prelabel_query_registry_rows"], 1)
            self.assertFalse(audited["stage3_training_authorized"])

    def test_actor_query_sidecar_requires_complete_goal_bindings_and_allows_two_queries_for_one_event(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, _, events, _ = indexed(root)
            calibration = next(event for event in events if event["usage_role"] == "annotation_calibration")
            labels = annotations(calibration)
            first = next(row for row in labels["views"] if row["label_kind"] == "goal_satisfaction_counterfactual")
            labels_path = root / "labels.json"
            labels_path.write_bytes(pack.canonical_json(labels).encode("utf-8") + b"\n")
            index_seal = pack.sha256_file(index_path / "inventory_seal.json")
            registry_path, registry_sha, empty_bindings = write_actor_query_inputs(
                root, calibration, first, binding_rows=[])
            with self.assertRaisesRegex(ValueError, "cover exactly every packaged goal view"):
                pack.publish(
                    index_path, labels_path, root / "empty-sidecar-package", release_role="annotation_calibration",
                    minimum_scale=None, request_dataset_quality_eligibility=False,
                    expected_index_inventory_seal_sha256=index_seal,
                    prelabel_query_registry=registry_path,
                    expected_prelabel_query_registry_sha256=registry_sha,
                    postlabel_view_prelabel_query_bindings=empty_bindings)
            self.assertFalse((root / "empty-sidecar-package").exists())
            self.assertFalse(any(root.glob(".empty-sidecar-package.staging-*")))

            second = deepcopy(first)
            second["goal_relation"] = "radio is in a distinct frozen relation"
            with_id(second)
            labels["views"].append(second)
            labels["view_reviews"].append(review(second))
            labels_path.write_bytes(pack.canonical_json(labels).encode("utf-8") + b"\n")
            first_registry = prelabel_registry_row(calibration, first, query_id="producer-query-first")
            second_registry = prelabel_registry_row(calibration, second, query_id="producer-query-second")
            registry_path.unlink()
            empty_bindings.unlink()
            registry_path, registry_sha, partial_bindings = write_actor_query_inputs(
                root, calibration, first, registry_rows=[first_registry, second_registry],
                binding_query_id="producer-query-first")
            with self.assertRaisesRegex(ValueError, "cover exactly every packaged goal view"):
                pack.publish(
                    index_path, labels_path, root / "partial-sidecar-package", release_role="annotation_calibration",
                    minimum_scale=None, request_dataset_quality_eligibility=False,
                    expected_index_inventory_seal_sha256=index_seal,
                    prelabel_query_registry=registry_path,
                    expected_prelabel_query_registry_sha256=registry_sha,
                    postlabel_view_prelabel_query_bindings=partial_bindings)
            self.assertFalse((root / "partial-sidecar-package").exists())
            self.assertFalse(any(root.glob(".partial-sidecar-package.staging-*")))

            partial_bindings.unlink()
            bindings_path = root / "complete-view-bindings.jsonl"
            pack._write_jsonl(bindings_path, [
                {"view_id": first["view_id"], "prelabel_query_id": "producer-query-first"},
                {"view_id": second["view_id"], "prelabel_query_id": "producer-query-second"},
            ])
            result = pack.publish(
                index_path, labels_path, root / "complete-sidecar-package", release_role="annotation_calibration",
                minimum_scale=None, request_dataset_quality_eligibility=False,
                expected_index_inventory_seal_sha256=index_seal,
                prelabel_query_registry=registry_path,
                expected_prelabel_query_registry_sha256=registry_sha,
                postlabel_view_prelabel_query_bindings=bindings_path)
            self.assertEqual(result["status_report"]["candidate_annotated"], 5)
            output = root / "complete-sidecar-package"
            self.assertEqual(pack.read_json(output / "release_manifest.json")["actor_query_sidecar"]["row_count"], 2)
            self.assertTrue(all(not value for wrapper in pack.read_jsonl(output / "views.jsonl")
                                for value in wrapper["dataset_quality_gates"].values()))
            self.assertFalse(result["training_run_authorized"])

    def test_actor_query_sidecar_rejects_same_event_different_query_and_registry_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path, _, events, _ = indexed(root)
            calibration = next(event for event in events if event["usage_role"] == "annotation_calibration")
            labels = annotations(calibration)
            goal = next(row for row in labels["views"] if row["label_kind"] == "goal_satisfaction_counterfactual")
            labels_path = root / "labels.json"
            labels_path.write_bytes(pack.canonical_json(labels).encode("utf-8") + b"\n")
            index_seal = pack.sha256_file(index_path / "inventory_seal.json")
            wrong = prelabel_registry_row(calibration, goal, query_id="same-event-wrong-query",
                                          text="A distinct frozen question for the same event.")
            registry_path, registry_sha, bindings_path = write_actor_query_inputs(
                root, calibration, goal, registry_rows=[wrong], binding_query_id="same-event-wrong-query")
            with self.assertRaisesRegex(ValueError, "text/semantic family does not bind"):
                pack.publish(
                    index_path, labels_path, root / "wrong-query-package", release_role="annotation_calibration",
                    minimum_scale=None, request_dataset_quality_eligibility=False,
                    expected_index_inventory_seal_sha256=index_seal,
                    prelabel_query_registry=registry_path,
                    expected_prelabel_query_registry_sha256=registry_sha,
                    postlabel_view_prelabel_query_bindings=bindings_path)
            self.assertFalse((root / "wrong-query-package").exists())

            valid = prelabel_registry_row(calibration, goal)
            registry_path.unlink()
            bindings_path.unlink()
            registry_path, _, bindings_path = write_actor_query_inputs(root, calibration, goal, registry_rows=[valid])
            original_sha = pack.sha256_file(registry_path)
            valid["source_pin"]["queue_sha256"] = "not-a-sha"
            pack._write_jsonl(root / "tampered-registry.jsonl", [valid])
            with self.assertRaisesRegex(ValueError, "does not match its externally pinned SHA-256"):
                pack.publish(
                    index_path, labels_path, root / "hash-tampered-package", release_role="annotation_calibration",
                    minimum_scale=None, request_dataset_quality_eligibility=False,
                    expected_index_inventory_seal_sha256=index_seal,
                    prelabel_query_registry=root / "tampered-registry.jsonl",
                    expected_prelabel_query_registry_sha256=original_sha,
                    postlabel_view_prelabel_query_bindings=bindings_path)
            self.assertFalse((root / "hash-tampered-package").exists())

            tampered_sha = pack.sha256_file(root / "tampered-registry.jsonl")
            with self.assertRaisesRegex(ValueError, "prelabel query registry is invalid"):
                pack.publish(
                    index_path, labels_path, root / "source-tampered-package", release_role="annotation_calibration",
                    minimum_scale=None, request_dataset_quality_eligibility=False,
                    expected_index_inventory_seal_sha256=index_seal,
                    prelabel_query_registry=root / "tampered-registry.jsonl",
                    expected_prelabel_query_registry_sha256=tampered_sha,
                    postlabel_view_prelabel_query_bindings=bindings_path)
            self.assertFalse((root / "source-tampered-package").exists())


if __name__ == "__main__":
    unittest.main()
