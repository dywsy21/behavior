from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "src/g05/data/memlite_event_protocol.py"
_SPEC = importlib.util.spec_from_file_location("p107_protocol_test", _PROTOCOL_PATH)
_PROTOCOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROTOCOL
_SPEC.loader.exec_module(_PROTOCOL)
ACTION_HORIZON = _PROTOCOL.ACTION_HORIZON
MODEL_PADDING_INDICES = _PROTOCOL.MODEL_PADDING_INDICES
ContractError = _PROTOCOL.ContractError
actor_evidence_projection = _PROTOCOL.actor_evidence_projection
authority_raw_action_payload = _PROTOCOL.authority_raw_action_payload
canonical_json = _PROTOCOL.canonical_json
canonical_sha256 = _PROTOCOL.canonical_sha256
action_payload_sha256 = _PROTOCOL.action_payload_sha256
event_id = _PROTOCOL.event_id
indexed_event_eligibility = _PROTOCOL.indexed_event_eligibility
load_corrective_action_authority = _PROTOCOL.load_corrective_action_authority
project_action_23_to_27 = _PROTOCOL.project_action_23_to_27
raw_action_payload_sha256 = _PROTOCOL.raw_action_payload_sha256
receipt_sha256 = _PROTOCOL.receipt_sha256
source_group_id = _PROTOCOL.source_group_id
validate_attempt_outcome_view = _PROTOCOL.validate_attempt_outcome_view
validate_corrective_action_view = _PROTOCOL.validate_corrective_action_view
validate_event = _PROTOCOL.validate_event
validate_goal_satisfaction_view = _PROTOCOL.validate_goal_satisfaction_view
view_id = _PROTOCOL.view_id


def source():
    result = {
        "source_release_manifest_sha256": "a" * 64,
        "source_annotation_sha256": "b" * 64,
        "task_index": 2,
        "task_instance_id": 11,
        "raw_episode_id": 2100,
        "episode_index": 201,
        "original_split": "train",
        "episode_length": 100,
    }
    result["source_group_id"] = source_group_id(result)
    return result


def event():
    result = {
        "schema_version": "memlite-event-recovery-v1", "record_kind": "event_candidate", "event_id": "",
        "source": source(), "event_kind": "ANNOTATED_SKILL_SEGMENT",
        "event_interval": {"start_frame": 10, "end_frame": 50},
        "observation": {"frame": 10, "timestamp_s": 10 / 30.0},
        "action": {"start_frame": 10, "actual_executed_length": None, "raw_action_dim": 23,
                   "model_action_dim": 27, "model_padding_indices": [7, 8, 17, 18]},
        "bundle_id": "bundle-before", "skill_bundle": [{"verb": "GRASP"}], "parallel_bundle": False,
        "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "video_locators": [],
    }
    result["event_id"] = event_id(result)
    return result


def common(kind, *, review_status="PROPOSED"):
    e = event()
    result = {
        "schema_version": "memlite-event-recovery-v1", "label_kind": kind, "view_id": "",
        "event_id": e["event_id"], "source_group_id": e["source"]["source_group_id"],
        "observation_frame": 10, "annotation_provenance": "human", "review_status": review_status,
        "actor_evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None,
                           "references": []},
    }
    return result, e


def with_id(row):
    row["view_id"] = view_id(row)
    return row


def self_digest(receipt):
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    return receipt


def _write_jsonl(path, rows):
    path.write_bytes(b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in rows))
    return {"relative_path": path.name, "sha256": _PROTOCOL._file_sha256(path),
            "bytes": path.stat().st_size, "rows": len(rows)}


def authorized_corrective_row(root, *, split="train", role="student_candidate", origin="logged_demonstration",
                              evidence_end=13):
    e = event()
    e["source"]["original_split"] = split
    e["source"]["source_group_id"] = source_group_id(e["source"])
    e["usage_role"] = role
    e["event_id"] = event_id(e)
    row, _ = common("corrective_action", review_status="PARENT_APPROVED")
    row.update(event_id=e["event_id"], source_group_id=e["source"]["source_group_id"],
               recovery_attempt_id="recovery-2", recovery_from_attempt_id="attempt-1",
               action_intent_bundle_id="bundle-recovery", executed_intent_bundle_id="bundle-recovery",
               action_start_frame=10, actual_executed_length=2, raw_action_dim=23, model_action_dim=27,
               model_padding_indices=list(MODEL_PADDING_INDICES),
               action_is_pad=[False, False] + [True] * (ACTION_HORIZON - 2),
               executed_action_receipt={"executed": True, "raw_action_sha256": "",
                                        "intent_bundle_id": "bundle-recovery", "actual_end_frame": 12},
               action_payload_sha256="", execution_receipt_sha256="", recovery_verification_receipt_sha256="",
               recovery_verified=False, low_action_supervision_mask=True)
    raw_artifact = "a" * 64
    raw_actions = [[float(offset + dim) for dim in range(23)] for offset in (0, 100)]
    payload_digest = action_payload_sha256(raw_actions)
    raw_digest = raw_action_payload_sha256(raw_actions, raw_action_artifact_sha256=raw_artifact)
    payload = {"schema_version": "p107-raw23-action-payload-v1", "action_payload_sha256": payload_digest,
               "raw_action_sha256": raw_digest, "raw_action_artifact_sha256": raw_artifact,
               "raw_actions_23": raw_actions}
    row["executed_action_receipt"]["raw_action_sha256"] = raw_digest
    row["action_payload_sha256"] = payload_digest
    execution = self_digest({
        "schema_version": "p107-execution-receipt-v1", "receipt_sha256": "", "event_id": e["event_id"],
        "source_group_id": e["source"]["source_group_id"],
        "source_release_manifest_sha256": e["source"]["source_release_manifest_sha256"],
        "raw_episode_id": e["source"]["raw_episode_id"], "episode_index": e["source"]["episode_index"],
        "raw_action_sha256": raw_digest, "action_payload_sha256": payload_digest,
        "raw_action_artifact_sha256": raw_artifact,
        "action_start_frame": 10, "actual_end_frame": 12, "actual_executed_length": 2,
        "raw_action_dim": 23, "intent_bundle_id": "bundle-recovery", "executed": True,
    })
    tier = "GOLD_PHYSICAL" if origin == "live_sim_branch" else "SILVER_REVIEWED_LOGGED_DEMONSTRATION"
    live_root = "6" * 64 if origin == "live_sim_branch" else None
    frozen_window = None if origin == "live_sim_branch" else raw_artifact
    temporal_review = None if origin == "live_sim_branch" else "e" * 64
    verification = self_digest({
        "schema_version": "p107-recovery-verification-receipt-v1", "receipt_sha256": "", "event_id": e["event_id"],
        "source_group_id": e["source"]["source_group_id"], "recovery_attempt_id": "recovery-2",
        "recovery_from_attempt_id": "attempt-1", "action_intent_bundle_id": "bundle-recovery",
        "raw_action_sha256": raw_digest, "evidence_origin": origin, "quality_tier": tier,
        "verification_frame": 14, "label_available_frame": 15, "branch_or_episode_final_frame": 20,
        "evidence": {"kind": "PHYSICAL_FACT" if origin == "live_sim_branch" else "TEMPORAL_VISUAL_REVIEW",
                     "evidence_end_frame": evidence_end, "available_frame": 14, "artifact_sha256": "b" * 64},
        "verification_artifact_sha256": "c" * 64,
        "frozen_action_window_artifact_sha256": frozen_window,
        "temporal_review_artifact_sha256": temporal_review,
        "live_runtime_acceptance_root_sha256": live_root,
    })
    row["execution_receipt_sha256"] = execution["receipt_sha256"]
    row["recovery_verification_receipt_sha256"] = verification["receipt_sha256"]
    with_id(row)
    group = {"source_group_id": e["source"]["source_group_id"],
             "source_release_manifest_sha256": e["source"]["source_release_manifest_sha256"],
             "task_index": e["source"]["task_index"], "task_instance_id": e["source"]["task_instance_id"],
             "original_split": split, "usage_role": role}
    files = {
        "events": _write_jsonl(root / "events.jsonl", [e]),
        "source_groups": _write_jsonl(root / "source_groups.jsonl", [group]),
        "raw_action_payloads": _write_jsonl(root / "raw_action_payloads.jsonl", [payload]),
        "execution_receipts": _write_jsonl(root / "execution_receipts.jsonl", [execution]),
        "verification_receipts": _write_jsonl(root / "verification_receipts.jsonl", [verification]),
        "artifacts": _write_jsonl(root / "artifacts.jsonl", [
            {"artifact_sha256": digest, "artifact_kind": "sealed_test_artifact"}
            for digest in sorted({raw_artifact, "b" * 64, "c" * 64, "e" * 64, "9" * 64})
        ]),
    }
    approval = self_digest({
        "schema_version": "p107-parent-low-fm-approval-v1", "receipt_sha256": "",
        "approval_status": "APPROVED_FOR_LOW_FM", "view_id": row["view_id"], "event_id": e["event_id"],
        "source_group_id": group["source_group_id"],
        "source_release_manifest_sha256": group["source_release_manifest_sha256"],
        "raw_episode_id": e["source"]["raw_episode_id"], "episode_index": e["source"]["episode_index"],
        "original_split": split, "usage_role": role,
        "action_intent_bundle_id": "bundle-recovery", "executed_intent_bundle_id": "bundle-recovery",
        "raw_action_sha256": raw_digest, "action_payload_sha256": payload_digest,
        "execution_receipt_sha256": execution["receipt_sha256"],
        "recovery_verification_receipt_sha256": verification["receipt_sha256"],
        "parent_review_artifact_sha256": "9" * 64, "evidence_origin": origin, "quality_tier": tier,
        "index_inventory_seal_sha256": "8" * 64,
        "action_payload_root_sha256": files["raw_action_payloads"]["sha256"],
        "execution_receipt_root_sha256": files["execution_receipts"]["sha256"],
        "recovery_verification_receipt_root_sha256": files["verification_receipts"]["sha256"],
        "artifact_manifest_root_sha256": files["artifacts"]["sha256"],
    })
    files["parent_approval_receipts"] = _write_jsonl(root / "parent_approval_receipts.jsonl", [approval])
    manifest = {"schema_version": "p107-corrective-publisher-manifest-v1", "index_inventory_seal_sha256": "8" * 64,
                "accepted_live_runtime_root_sha256": live_root, "files": files}
    manifest_path = root / "publisher_manifest.json"
    manifest_path.write_bytes(canonical_json(manifest).encode("utf-8"))
    authority = load_corrective_action_authority(
        root, expected_publisher_manifest_sha256=_PROTOCOL._file_sha256(manifest_path),
        expected_index_inventory_seal_sha256="8" * 64,
        expected_live_runtime_acceptance_root_sha256=live_root)
    return row, e, authority, verification


class EventProtocolTests(unittest.TestCase):
    def test_event_identity_is_content_addressed_and_annotation_end_is_not_outcome(self):
        row = event()
        validate_event(row)
        self.assertEqual(row["action"]["actual_executed_length"], None)
        self.assertEqual(row["evidence"]["kind"], "MISSING")
        changed = deepcopy(row)
        changed["event_interval"]["end_frame"] = 51
        self.assertNotEqual(event_id(changed), row["event_id"])
        changed["event_id"] = event_id(changed)
        validate_event(changed)

    def test_goal_counterfactual_cannot_be_outcome_or_fm(self):
        row, e = common("goal_satisfaction_counterfactual")
        row.update(goal_relation="radio is on", goal_satisfaction="UNKNOWN", valid_goal_mask=False,
                   attempt_outcome="NOT_APPLICABLE",
                   evidence={"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
                   low_action_supervision_mask=False)
        validate_goal_satisfaction_view(with_id(row), e)
        bad = deepcopy(row)
        bad["attempt_outcome"] = "FAILED"
        with self.assertRaises(ContractError):
            validate_goal_satisfaction_view(with_id(bad), e)

    def test_missing_outcome_is_not_real_unknown_and_future_evidence_is_rejected(self):
        row, e = common("attempt_outcome")
        row.update(attempt_id="attempt-1", parent_attempt_id=None, evaluated_bundle_id=None,
                   attempt_outcome="UNKNOWN", valid_result_mask=False,
                   evidence={"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
                   low_action_supervision_mask=False)
        validate_attempt_outcome_view(with_id(row), e)
        bad = deepcopy(row)
        bad.update(valid_result_mask=True, evaluated_bundle_id="bundle-before", attempt_outcome="FAILED",
                   evidence={"kind": "VISUAL", "evidence_end_frame": 11, "available_frame": 11})
        with self.assertRaises(ContractError):
            validate_attempt_outcome_view(with_id(bad), e)

    def test_corrective_fm_requires_same_intent_actual_receipt_and_pre_action_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            row, e, authority, _ = authorized_corrective_row(Path(folder))
            validate_corrective_action_view(row, e, authority=authority)
            payload = authority_raw_action_payload(authority, row["action_payload_sha256"])
            self.assertEqual(len(payload["raw_actions_23"]), row["actual_executed_length"])
            payload["raw_actions_23"][0][0] = -1.0
            self.assertNotEqual(authority_raw_action_payload(authority, row["action_payload_sha256"])["raw_actions_23"][0][0], -1.0)
            for mutation in ("wrong_bundle", "future_observation", "no_authority", "proposed"):
                bad = deepcopy(row)
                if mutation == "wrong_bundle":
                    bad["executed_intent_bundle_id"] = "wrong"
                    with_id(bad)
                    with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                        validate_corrective_action_view(bad, e, authority=authority)
                elif mutation == "future_observation":
                    bad["action_start_frame"] = 11
                    with_id(bad)
                    with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                        validate_corrective_action_view(bad, e, authority=authority)
                elif mutation == "no_authority":
                    with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                        validate_corrective_action_view(bad, e)
                else:
                    bad["review_status"] = "PROPOSED"
                    with_id(bad)
                    with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                        validate_corrective_action_view(bad, e, authority=authority)

    def test_only_indexed_student_group_can_be_trainable_but_eval_remains_evaluation_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            row, e, authority, _ = authorized_corrective_row(Path(folder))
            validate_corrective_action_view(row, e, authority=authority)
        eval_event = event()
        eval_event["source"]["original_split"] = "eval"
        eval_event["source"]["source_group_id"] = source_group_id(eval_event["source"])
        eval_event["event_id"] = event_id(eval_event)
        eval_group = {"source_group_id": eval_event["source"]["source_group_id"],
                      "original_split": "eval", "usage_role": "evaluation_only"}
        self.assertEqual(indexed_event_eligibility(eval_event, eval_group),
                         {"evaluation_loss_eligible": True, "student_train_eligible": False})

    def test_eval_agent_proposed_boolean_cannot_bypass_external_authority(self):
        with tempfile.TemporaryDirectory() as folder:
            row, e, _, _ = authorized_corrective_row(Path(folder))
            e["source"]["original_split"] = "eval"
            e["source"]["source_group_id"] = source_group_id(e["source"])
            e["event_id"] = event_id(e)
            row.update(event_id=e["event_id"], source_group_id=e["source"]["source_group_id"],
                       annotation_provenance="agent-unreviewed", review_status="PROPOSED", recovery_verified=True)
            with_id(row)
            with self.assertRaises(ContractError):
                validate_corrective_action_view(row, e)

    def test_verification_after_action_is_allowed_but_not_after_final_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            row, e, authority, verification = authorized_corrective_row(Path(folder), origin="logged_demonstration")
            self.assertGreater(verification["verification_frame"], row["action_start_frame"])
            validate_corrective_action_view(row, e, authority=authority)
            bad_verification = deepcopy(verification)
            bad_verification["verification_frame"] = 999
            bad_verification["label_available_frame"] = 999
            bad_verification["branch_or_episode_final_frame"] = 20
            self_digest(bad_verification)
            with self.assertRaises(ContractError):
                _PROTOCOL._validate_verification_receipt(bad_verification)

    def test_authority_requires_pinned_files_and_post_action_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, event_row, authority, _ = authorized_corrective_row(root)
            manifest_sha = authority.publisher_manifest_sha256
            with self.assertRaises(ContractError):
                load_corrective_action_authority(
                    root, expected_publisher_manifest_sha256="0" * 64,
                    expected_index_inventory_seal_sha256="8" * 64)
            with self.assertRaises(TypeError):
                authority._approval_json[row["view_id"]] = "forged"
            (root / "raw_action_payloads.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ContractError, "bytes or SHA-256"):
                load_corrective_action_authority(
                    root, expected_publisher_manifest_sha256=manifest_sha,
                    expected_index_inventory_seal_sha256="8" * 64)
        with tempfile.TemporaryDirectory() as folder:
            row, event_row, authority, _ = authorized_corrective_row(Path(folder), evidence_end=11)
            with self.assertRaisesRegex(ContractError, "recovery verification"):
                validate_corrective_action_view(row, event_row, authority=authority)

    def test_gold_requires_externally_accepted_live_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, authority, _ = authorized_corrective_row(root, origin="live_sim_branch")
            with self.assertRaisesRegex(ContractError, "live-runtime root"):
                load_corrective_action_authority(
                    root, expected_publisher_manifest_sha256=authority.publisher_manifest_sha256,
                    expected_index_inventory_seal_sha256="8" * 64,
                    expected_live_runtime_acceptance_root_sha256=None)

    def test_live_gold_and_logged_silver_are_distinct_positive_paths(self):
        for origin in ("live_sim_branch", "logged_demonstration"):
            with tempfile.TemporaryDirectory() as folder:
                row, e, authority, verification = authorized_corrective_row(Path(folder), origin=origin)
                with self.subTest(origin=origin):
                    validate_corrective_action_view(row, e, authority=authority)
                    self.assertEqual(verification["quality_tier"],
                                     "GOLD_PHYSICAL" if origin == "live_sim_branch" else "SILVER_REVIEWED_LOGGED_DEMONSTRATION")

    def test_actor_projection_excludes_privileged_audit_and_action_projection_preserves_pad_contract(self):
        row, _ = common("attempt_outcome")
        row["privileged_evidence"] = {"object_pose": "audit-only"}
        safe = actor_evidence_projection(row)
        self.assertEqual(safe["kind"], "MISSING")
        self.assertNotIn("privileged_evidence", safe)
        row["actor_evidence"] = {"kind": "VISUAL", "evidence_end_frame": 0, "available_frame": 0,
                                 "references": [{"kind": "rgb_frame", "view": "head", "frame": 0,
                                                 "artifact_sha256": "a" * 64, "nested": {"object_pose": [1, 2, 3]}}]}
        with self.assertRaises(ContractError):
            actor_evidence_projection(row)
        row["actor_evidence"] = {"kind": "VISUAL", "evidence_end_frame": 0, "available_frame": 0,
                                 "references": [], "object_pose": [1, 2, 3]}
        with self.assertRaises(ContractError):
            actor_evidence_projection(row)
        row["actor_evidence"] = {"kind": "VISUAL", "evidence_end_frame": 0, "available_frame": 0,
                                 "references": [{"kind": "rgb_frame", "view": "head", "frame": 1,
                                                 "artifact_sha256": "a" * 64}]}
        with self.assertRaisesRegex(ContractError, "available at the current observation clock"):
            actor_evidence_projection(row)
        converted = project_action_23_to_27([[float(i) for i in range(23)]], 1)
        self.assertEqual(len(converted["actions_27"]), 32)
        self.assertEqual(converted["action_dim_is_pad"], [i in MODEL_PADDING_INDICES for i in range(27)])
        self.assertEqual(converted["action_is_pad"], [False] + [True] * 31)
        self.assertEqual([converted["actions_27"][0][i] for i in MODEL_PADDING_INDICES], [0.0] * 4)


if __name__ == "__main__":
    unittest.main()
