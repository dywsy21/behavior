#!/usr/bin/env python3
"""Focused, data-free tests for the P107 prelabel query producer."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))
import build_memlite_visual_relation_queries as producer  # noqa: E402


def phase_fixture(verb: str = "PRESS", skill_id: int = 67) -> tuple[dict, dict, dict]:
    event_id = "1" * 64
    group_id = "2" * 64
    descriptions = producer.CANONICAL35[skill_id]
    target = "radio_89" if verb == "PRESS" else "brisket"
    source = "" if verb == "PRESS" else "fridge"
    queue = {
        "event_id": event_id,
        "schema_version": "p107-phase-balanced-calibration-queue-v1",
        "source_identity": {"source_group_id": group_id, "task_index": 0, "task_instance_id": 1},
        "observation_frame": 200,
        "observation_phase": "MID",
        "parent_event_id": "3" * 64,
        "selection_order": 4,
        "selection_stratum": "MID",
        "queried_skill_start_frame": 100,
        "queried_skill_end_frame": 200,
        "terminal_query_binding": "NOT_TERMINAL",
        "training_eligible": False,
        "usage_role": "annotation_calibration",
        "original_annotated_segment": {"start_frame": 100, "end_frame": 201},
        "intent_start_causal_reference": {"frame": 100},
        "queried_skill": {
            "binding_status": "BOUND_METADATA_NEEDS_VISUAL_CONFIRMATION",
            "queried_skill": {
                "raw_description": descriptions[0],
                "skill_id": skill_id,
                "skill_start": 100,
                "skill_end": 200,
                "source": source,
                "target": target,
                "destination": "",
                "target_part": "",
                "verb": verb,
            },
        },
        "temporal_windows": {
            "actor_available_window": {"sampled_frames": [140, 155, 170, 185, 200]},
            "offline_review_after_window": {"sampled_frames": [201, 216]},
        },
    }
    indexed = {
        "event_id": event_id,
        "source": {"source_group_id": group_id, "task_index": 0, "task_instance_id": 1},
        "observation": {"frame": 200},
        "event_interval": {"start_frame": 100, "end_frame": 201},
        "bundle_id": "4" * 64,
        "phase_lineage": {
            "parent_event_id": queue["parent_event_id"],
            "selection_stratum": "MID",
        },
        "skill_bundle": [{
            "skill_id": skill_id,
            "raw_description": descriptions[0],
            "verb": verb,
            "skill_idx": 4,
            "skill_start": 100,
            "skill_end": 200,
            "source": source,
            "target": target,
            "destination": "",
            "target_part": "",
            "binding_confidence": "BOUND",
            "raw_relation": {
                "manipulating_object_id": [target],
                "object_id": [[target] if verb == "PRESS" else [target, source]],
            },
        }],
    }
    selection = {
        "schema_version": "p107-phase-balanced-calibration-manifest-v1",
        "status": "METADATA_CANDIDATES_READY_FOR_INDEPENDENT_RENDER_REVIEW",
        "training_eligible": False,
        "selection": {"rows": [{
            "event_id": event_id,
            "observation_frame": 200,
            "parent_event_id": queue["parent_event_id"],
            "queried_skill_id": skill_id,
            "selection_stratum": "MID",
            "source_group_id": group_id,
            "task_index": 0,
        }]},
    }
    return queue, indexed, selection


def make_record(queue: dict, indexed: dict, query_ordinal: int = 0) -> dict:
    normalized = producer.normalize_phase_row(queue, indexed)
    base = producer.generate_row(normalized, 0)[query_ordinal]
    pin = {
        "event_id": queue["event_id"],
        "selection_manifest_sha256": "a" * 64,
        "queue_sha256": "b" * 64,
        "queue_record_sha256": "c" * 64,
        "phase_index_sha256": "d" * 64,
        "phase_index_record_sha256": "e" * 64,
        "selection_manifest_path": "/sealed/selection.json",
        "queue_path": "/sealed/queue.jsonl",
        "phase_index_path": "/sealed/events.jsonl",
    }
    return producer.registry_projection(base, queue, indexed, pin)


class PrelabelProducerTest(unittest.TestCase):
    def test_exact_event_and_target_binding(self) -> None:
        queue, indexed, _ = phase_fixture()
        producer.validate_phase_binding(queue, indexed)
        broken = copy.deepcopy(indexed)
        broken["skill_bundle"][0]["target"] = "other_radio"
        with self.assertRaises(ValueError):
            producer.validate_phase_binding(queue, broken)

    def test_press_is_contact_query_not_result(self) -> None:
        queue, indexed, _ = phase_fixture()
        record = make_record(queue, indexed)
        current = record["current_visible_goal_relation"]
        self.assertIn("identifiable press control", current["query_text"])
        self.assertIn("CONTACT_IS_NOT_PRESS_RESULT", current["press_semantics"])
        self.assertNotIn("answer_enum", record)
        self.assertNotIn("goal_satisfaction", json.dumps(record))

    def test_no_future_actor_references_or_label_fields(self) -> None:
        queue, indexed, _ = phase_fixture()
        record = make_record(queue, indexed)
        policy = record["actor_evidence_policy"]
        self.assertEqual(policy["causal_frame_indices"][-1], 200)
        self.assertTrue(all(frame <= 200 for frame in policy["causal_frame_indices"]))
        self.assertFalse(policy["future_actor_references_allowed"])
        encoded = json.dumps(record, sort_keys=True)
        for forbidden in ("goal_satisfaction", "attempt_outcome", "action_payload", "canonical_goal_view_id", "future_frames"):
            self.assertNotIn(forbidden, encoded)

    def test_multiple_queries_same_event_get_distinct_prelabel_ids(self) -> None:
        queue, indexed, _ = phase_fixture()
        normalized = producer.normalize_phase_row(queue, indexed)
        second = copy.deepcopy(normalized["question_context"]["question_candidates"][0])
        second["intended_goal_relation_question"] = "A second atomic query over the same event"
        normalized["question_context"]["question_candidates"].append(second)
        bases = producer.generate_row(normalized, 0)
        self.assertEqual(len(bases), 2)
        pin = {
            "event_id": queue["event_id"],
            "selection_manifest_sha256": "a" * 64,
            "queue_sha256": "b" * 64,
            "queue_record_sha256": "c" * 64,
            "phase_index_sha256": "d" * 64,
            "phase_index_record_sha256": "e" * 64,
            "selection_manifest_path": "/sealed/selection.json",
            "queue_path": "/sealed/queue.jsonl",
            "phase_index_path": "/sealed/events.jsonl",
        }
        records = [producer.registry_projection(base, queue, indexed, pin) for base in bases]
        self.assertEqual([r["query_ordinal_within_event"] for r in records], [0, 1])
        self.assertNotEqual(records[0]["prelabel_query_id"], records[1]["prelabel_query_id"])

    def test_navigation_and_handover_policies_are_explicit(self) -> None:
        queue, indexed, _ = phase_fixture("HANDOVER", 5)
        record = make_record(queue, indexed)
        current = record["current_visible_goal_relation"]
        self.assertIn("other robot gripper", current["query_text"])
        self.assertIn("NEVER_INVENT_PERSON", current["recipient_semantics"])
        queue, indexed, _ = phase_fixture("NAVIGATE", 1)
        record = make_record(queue, indexed)
        current = record["current_visible_goal_relation"]
        self.assertIn("required distance and pose", current["query_text"])
        self.assertEqual(current["supervision_eligibility"], "EXCLUDED_UNLESS_APPROVED_STATE_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
