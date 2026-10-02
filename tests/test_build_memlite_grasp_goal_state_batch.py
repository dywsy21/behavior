import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import build_memlite_grasp_goal_state_batch as batch  # noqa: E402
import render_memlite_event_packets as renderer  # noqa: E402
import select_memlite_phase_balanced_calibration_queue as phase  # noqa: E402
from test_select_memlite_phase_balanced_calibration_queue import parent_candidate, protocol  # noqa: E402


class GraspGoalStateBatchTests(unittest.TestCase):
    def _parent(self):
        owner = protocol()
        return owner, parent_candidate(owner, ordinal=901, start=10, end=20,
                                      usage_role="student_candidate")

    def _unbound_pair(self):
        owner, parent = self._parent()
        item = phase.PhaseCandidate(
            parent=parent, stratum="GRASP_GOAL_STATE_BATCH", observation_phase="ENTRY",
            anchor_frame=10, queried_skill=copy.deepcopy(parent.event["skill_bundle"][0]),
            parent_skill_index=0, repeat_distinct_episode_count=1,
            successor_parent_event_id=None, transition_gap_frames=None,
        )
        event = phase.derived_event(item, protocol=owner, usage_role="student_candidate",
                                    private_diagnostic_student_candidate=True,
                                    private_goal_state_mode=batch.GOAL_UNBOUND_MODE)
        return owner, parent, event

    def test_goal_unbound_has_no_query_and_false_gates(self):
        owner, _parent, event = self._unbound_pair()
        batch.validate_goal_unbound_event(event, protocol=owner)
        lineage = event["phase_lineage"]
        self.assertEqual(lineage["goal_state_mode"], "goal_unbound")
        self.assertNotIn("goal_state_question", lineage)
        self.assertFalse(lineage["training_eligible"])
        self.assertFalse(lineage["outcome_supervision"])
        self.assertEqual(renderer_mode(event), "goal_unbound")

    def test_sealed_query_requires_visual_phrase_and_binds_registry(self):
        owner, _parent, event = self._unbound_pair()
        question = "At the anchor, is the brown animal toy visibly held by the robot gripper?"
        row = {
            "event_id": event["event_id"], "query_id": "q-901",
            "question": question, "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "target_raw": "cup", "target_display_phrase": "brown animal toy",
            "relation_family": "grasp_current_hold",
        }
        registry_sha = hashlib.sha256(batch.canonical([row]).encode()).hexdigest()
        sealed, questions = batch.seal_query_registry([event], [row],
                                                       registry_sha256=registry_sha, protocol=owner)
        self.assertEqual(set(questions), {event["event_id"]})
        self.assertEqual(questions[event["event_id"]], {"question": question})
        self.assertEqual(renderer_mode(sealed[0]), "sealed_query")
        self.assertEqual(sealed[0]["event_id"], event["event_id"])

    def test_query_rejects_raw_id_and_future_or_outcome_language(self):
        owner, _parent, event = self._unbound_pair()
        bad = {
            "event_id": event["event_id"], "query_id": "q-bad",
            "question": "At the anchor, is cup held after recovery?", "question_sha256": "x" * 64,
            "target_raw": "cup", "target_display_phrase": "brown animal toy",
            "relation_family": "grasp_current_hold",
        }
        with self.assertRaises(ValueError):
            batch.seal_query_registry([event], [bad], registry_sha256="a" * 64, protocol=owner)

    def test_sealed_query_is_accepted_by_private_locator_renderer(self):
        owner, _parent, event = self._unbound_pair()
        question = "At the anchor, is the brown animal toy visibly held by the robot gripper?"
        row = {
            "event_id": event["event_id"], "query_id": "q-render",
            "question": question, "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "target_raw": "cup", "target_display_phrase": "brown animal toy",
            "relation_family": "grasp_current_hold",
        }
        registry_sha = hashlib.sha256(batch.canonical([row]).encode()).hexdigest()
        sealed, questions = batch.seal_query_registry([event], [row], registry_sha256=registry_sha, protocol=owner)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index"
            index.mkdir()
            event_path = index / "event_candidates.jsonl"
            event_path.write_bytes((owner.canonical_json(sealed[0]) + "\n").encode())
            index_manifest = {
                "schema_version": "memlite-event-index-v1",
                "status": "METADATA_ONLY_READY_FOR_PACKET_RENDER",
                "training_eligible": False,
                "source_release_manifest_sha256": event["source"]["source_release_manifest_sha256"],
                "protocol_sha256": renderer._protocol_binding.sha256,
                "files": {"event_candidates.jsonl": {"sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
                                                          "rows": 1, "bytes": event_path.stat().st_size}},
            }
            (index / "manifest.json").write_text(owner.canonical_json(index_manifest) + "\n")
            output = root / "packets"
            result = renderer.create_packets(
                index, output, event_ids={event["event_id"]}, limit=None, questions=questions,
                include_source_annotation_context=False, decode=False, raw_root=None, contact_sheets=False,
                expected_usage_role="student_candidate", private_goal_state_review=True,
                protocol_binding=renderer._protocol_binding,
            )
            renderer.resume_packets(index, output,
                                    expected_packet_manifest_sha256=result["packet_manifest_sha256"],
                                    protocol_binding=renderer._protocol_binding)
            packet = json.loads((output / "packets.jsonl").read_text())
            self.assertEqual(packet["actor_packet"]["question_context"], {"question": question})
            self.assertFalse(packet["training_eligible"])

    def test_batch_events_and_requests_are_exactly_two_anchors(self):
        owner, parent, _event = self._unbound_pair()
        row = {
            "candidate_id": "synthetic-pair", "source_group_id": parent.source_group_id,
            "episode_index": parent.episode_index, "task_index": parent.task_id,
            "task_instance_id": parent.task_instance_id,
        }
        joined = {
            ("synthetic-pair", parent.episode_index, "first_grasp"): parent.event,
            ("synthetic-pair", parent.episode_index, "second_grasp"): parent.event,
        }
        events, jobs, requests = batch.build_goal_unbound_events([row], joined, protocol=owner)
        self.assertEqual(len(events), len(jobs),)
        self.assertEqual(len(events), 2)
        self.assertEqual({e["phase_lineage"]["observation_phase"] for e in events}, {"ENTRY", "TERMINAL"})
        self.assertEqual({tuple(r["requested_frame_indices"]) for r in requests}, {(10,), (10, 19)})
        self.assertTrue(all(j["training_eligible"] is False for j in jobs))

    def test_blind_context_rejects_two_anchors_from_one_episode(self):
        owner, parent, _event = self._unbound_pair()
        row = {"candidate_id": "synthetic-pair", "source_group_id": parent.source_group_id,
               "episode_index": parent.episode_index, "task_index": parent.task_id,
               "task_instance_id": parent.task_instance_id}
        joined = {("synthetic-pair", parent.episode_index, "first_grasp"): parent.event,
                  ("synthetic-pair", parent.episode_index, "second_grasp"): parent.event}
        events, _jobs, _requests = batch.build_goal_unbound_events([row], joined, protocol=owner)
        with self.assertRaises(ValueError):
            batch.validate_blind_contexts(events, {"fresh-leaf": [e["event_id"] for e in events]})
        batch.validate_blind_contexts(events, {"entry-leaf": [events[0]["event_id"]],
                                               "terminal-leaf": [events[1]["event_id"]]})

    def test_exclusion_sha_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exclusions.json"
            value = {
                "schema_version": batch.EXPECTED_EXCLUSION_SCHEMA,
                "mandatory_root_reviewed_source_episode_tuples": [["a" * 64, 1]],
                "mandatory_coverage_eligible_unreviewed_train_query_tuples": [],
                "protective_category_quarantine_query_tuples_not_e2": [],
                "supplemental_private_review_source_episodes_not_e2": [],
                "supplemental_runtime_structural_source_episode_not_e2": [],
                "counts": {"mandatory_union_source_episodes": 1},
            }
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                batch.load_exclusions(path, expected_sha256="b" * 64)


def renderer_mode(event):
    # Importing the renderer lazily keeps this test's helper names explicit.
    import render_memlite_event_packets as renderer
    return renderer._private_goal_state_mode(event)
