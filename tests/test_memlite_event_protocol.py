from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
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
event_id = _PROTOCOL.event_id
project_action_23_to_27 = _PROTOCOL.project_action_23_to_27
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


def common(kind):
    e = event()
    result = {
        "schema_version": "memlite-event-recovery-v1", "label_kind": kind, "view_id": "",
        "event_id": e["event_id"], "source_group_id": e["source"]["source_group_id"],
        "observation_frame": 10, "annotation_provenance": "human", "review_status": "PROPOSED",
        "actor_evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None,
                           "references": []},
    }
    return result, e


def with_id(row):
    row["view_id"] = view_id(row)
    return row


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
        row, e = common("corrective_action")
        row.update(recovery_attempt_id="recovery-2", recovery_from_attempt_id="attempt-1",
                   action_intent_bundle_id="bundle-recovery", executed_intent_bundle_id="bundle-recovery",
                   action_start_frame=10, actual_executed_length=2, raw_action_dim=23, model_action_dim=27,
                   model_padding_indices=list(MODEL_PADDING_INDICES),
                   action_is_pad=[False, False] + [True] * (ACTION_HORIZON - 2),
                   executed_action_receipt={"executed": True, "raw_action_sha256": "c" * 64,
                                            "intent_bundle_id": "bundle-recovery", "actual_end_frame": 12},
                   recovery_verified=True, low_action_supervision_mask=True)
        validate_corrective_action_view(with_id(row), e)
        for mutation in ("wrong_bundle", "future_observation"):
            bad = deepcopy(row)
            if mutation == "wrong_bundle":
                bad["executed_intent_bundle_id"] = "wrong"
            else:
                bad["action_start_frame"] = 11
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_corrective_action_view(with_id(bad), e)

    def test_actor_projection_excludes_privileged_audit_and_action_projection_preserves_pad_contract(self):
        row, _ = common("attempt_outcome")
        row["privileged_evidence"] = {"object_pose": "audit-only"}
        safe = actor_evidence_projection(row)
        self.assertEqual(safe["kind"], "MISSING")
        self.assertNotIn("privileged_evidence", safe)
        converted = project_action_23_to_27([[float(i) for i in range(23)]], 1)
        self.assertEqual(len(converted["actions_27"]), 32)
        self.assertEqual(converted["action_dim_is_pad"], [i in MODEL_PADDING_INDICES for i in range(27)])
        self.assertEqual(converted["action_is_pad"], [False] + [True] * 31)
        self.assertEqual([converted["actions_27"][0][i] for i in MODEL_PADDING_INDICES], [0.0] * 4)


if __name__ == "__main__":
    unittest.main()
