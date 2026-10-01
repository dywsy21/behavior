import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import render_memlite_event_packets as renderer  # noqa: E402
import select_memlite_event_annotation_queue as base  # noqa: E402
import select_memlite_phase_balanced_calibration_queue as phase  # noqa: E402


SOURCE_SHA = "a" * 64
PROTOCOL_PATH = REPO / "src" / "g05" / "data" / "memlite_event_protocol.py"
PROTOCOL_SHA = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def protocol():
    return base.load_canonical_protocol(PROTOCOL_PATH, expected_sha256=PROTOCOL_SHA)


def parent_candidate(owner_protocol, *, ordinal, task=0, start=100, end=300,
                     group_id=None, raw_episode=None, episode_index=None, instance=None):
    """A canonical source event whose raw skill retains original bounds."""
    instance = ordinal + 1 if instance is None else instance
    group_id = owner_protocol.source_group_id({
        "source_release_manifest_sha256": SOURCE_SHA,
        "task_index": task,
        "task_instance_id": instance,
    }) if group_id is None else group_id
    raw_episode = ordinal + 1000 if raw_episode is None else raw_episode
    episode_index = raw_episode if episode_index is None else episode_index
    skill = {
        "skill_id": 10,
        "verb": "GRASP",
        "target": "cup",
        "source": "table",
        "destination": "",
        "target_part": "",
        "arm": "UNSPECIFIED",
        "binding_confidence": "BOUND",
        "skill_start": start,
        "skill_end": end,
        "skill_idx": ordinal,
        "interval_id": ordinal,
        "raw_description": "pick up cup",
        "raw_relation": {"object_id": [["cup"]], "memory_prefix": [],
                         "manipulating_object_id": ["cup"], "spatial_prefix": []},
    }
    event = {
        "schema_version": owner_protocol.SCHEMA_VERSION,
        "record_kind": "event_candidate",
        "event_id": "",
        "source": {
            "source_release_manifest_sha256": SOURCE_SHA,
            "source_annotation_sha256": hashlib.sha256(f"annotation-{ordinal}".encode()).hexdigest(),
            "source_group_id": group_id,
            "task_index": task,
            "task_instance_id": instance,
            "raw_episode_id": raw_episode,
            "episode_index": episode_index,
            "episode_length": 600,
            "original_split": "train",
        },
        "event_kind": "ANNOTATED_SKILL_SEGMENT",
        "event_interval": {"start_frame": start, "end_frame": end},
        "observation": {"frame": start, "timestamp_s": start / 30.0},
        "action": {"start_frame": start, "actual_executed_length": None, "raw_action_dim": 23,
                   "model_action_dim": 27, "model_padding_indices": [7, 8, 17, 18]},
        "bundle_id": hashlib.sha256(f"bundle-{ordinal}".encode()).hexdigest(),
        "skill_bundle": [skill],
        "parallel_bundle": False,
        "binding": [{key: skill[key] for key in ("verb", "target", "source", "destination", "target_part", "arm",
                                                    "binding_confidence")}],
        "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "video_locators": [{
            "view": view,
            "camera_key": f"observation.rgb.{view}",
            "relative_path": f"videos/{view}.mp4",
            "episode_start_timestamp_s": 0.0,
            "requested_timestamp_s": start / 30.0,
            "expected_fps": 30,
            "locator_status": "METADATA_ONLY_UNRESOLVED",
        } for view in ("head", "left_wrist", "right_wrist")],
        "task_name": f"task-{task}",
        "usage_role": "annotation_calibration",
    }
    event["event_id"] = owner_protocol.event_id(event)
    owner_protocol.validate_event(event)
    return base.Candidate(
        event=event, event_id=event["event_id"], source_group_id=group_id, task_id=task,
        task_instance_id=instance, episode_key=(group_id, raw_episode, episode_index),
        episode_index=episode_index, episode_length=600, anchor_frame=start, interval_start=start,
        interval_end=end, skill_ids=(10,), raw_source_verbs=("GRASP",),
        skill_keys=(base.skill_key(skill),), repeat_attempt_count=0, boundary_before=False,
        boundary_after=False, long_interval=True, candidate_signals=("NORMAL_CONTROL_CANDIDATE",),
        selection_stratum="NORMAL_CONTROL_CANDIDATE",
    )


def source_group(candidate):
    source = candidate.event["source"]
    return base.SourceGroup(candidate.source_group_id, SOURCE_SHA, candidate.task_id,
                            candidate.task_instance_id, "train", "annotation_calibration",
                            (source["episode_index"],))


class PhaseBalancedCalibrationQueueTests(unittest.TestCase):
    def test_original_skill_bounds_keep_terminal_query_on_completed_skill_and_repeat_is_cross_episode(self):
        owner_protocol = protocol()
        first = parent_candidate(owner_protocol, ordinal=0, start=100, end=300)
        # Adjacent source segment shares the original episode, so it supplies
        # transition lineage but must never replace the queried first skill.
        following = parent_candidate(owner_protocol, ordinal=1, task=0, start=300, end=450,
                                     group_id=first.source_group_id, raw_episode=1000, episode_index=1000, instance=1)
        repeated_elsewhere = parent_candidate(owner_protocol, ordinal=2, task=0, start=100, end=300)
        derived = phase.derive_candidates([first, following, repeated_elsewhere], max_transition_gap_frames=1)
        terminal = next(item for item in derived if item.parent.event_id == first.event_id and
                        item.stratum == "TERMINAL_OR_TRANSITION")
        self.assertEqual(terminal.observation_phase, "TERMINAL_TRANSITION")
        self.assertEqual(terminal.successor_parent_event_id, following.event_id)
        self.assertEqual((terminal.queried_skill["skill_start"], terminal.queried_skill["skill_end"]), (100, 300))
        event = phase.derived_event(terminal, protocol=owner_protocol)
        self.assertNotEqual(event["event_id"], first.event_id)
        self.assertEqual(event["observation"]["frame"], 299)
        self.assertEqual(event["phase_lineage"]["queried_skill_end_frame"], 300)
        self.assertEqual(event["phase_lineage"]["parent_skill_identity"]["parent_event_id"], first.event_id)
        row = phase.queue_row(terminal, event, selection_order=0)
        self.assertEqual(row["terminal_query_binding"], "QUERIED_SKILL_END_EQUALS_ORIGINAL_SEGMENT_END")
        self.assertEqual(row["intent_start_causal_reference"]["frame"], 100)
        self.assertEqual(row["intent_start_causal_reference"]["availability"],
                         "METADATA_REFERENCE_ONLY_NO_HISTORY_IMAGES_RENDERED_BY_THIS_QUEUE")
        repeated = [item for item in derived if item.stratum == "REPEATED_METADATA_QUERY"]
        self.assertTrue(repeated)
        self.assertTrue(all(item.repeat_distinct_episode_count >= 2 for item in repeated))
        self.assertTrue(all(item.observation_phase == "REPEATED_METADATA_QUERY" for item in repeated))

    def test_exact_40_mini_index_resumes_and_renderer_reads_it_without_adapter_or_decode(self):
        owner_protocol = protocol()
        parents = [parent_candidate(owner_protocol, ordinal=ordinal, task=ordinal % 5)
                   for ordinal in range(40)]
        all_candidates = phase.derive_candidates(parents, max_transition_gap_frames=1)
        selected = phase.select_balanced(all_candidates, seed="fixture-phase-seed", phase_quota=10,
                                         max_per_source_group=1, max_per_episode=1)
        self.assertEqual(phase._phase_counts(selected), {name: 10 for name in phase.PHASE_STRATA})
        self.assertEqual(len({item.source_group_id for item in selected}), 40)
        events = [phase.derived_event(item, protocol=owner_protocol) for item in selected]
        rows = [phase.queue_row(item, event, selection_order=order)
                for order, (item, event) in enumerate(zip(selected, events))]
        groups_by_id = {candidate.source_group_id: source_group(candidate) for candidate in parents}
        selected_groups = [phase._source_group_row(groups_by_id[item.source_group_id], protocol=owner_protocol)
                           for item in selected]
        expectations = {
            "expected_task_ids": list(range(5)),
            "expected_skill_vocabulary": [{"skill_id": 10, "skill_description": "grasp an object"}],
        }
        policy = {
            "source_release_manifest_sha256": SOURCE_SHA,
            "protocol_sha256": PROTOCOL_SHA,
            "coverage_expectations_sha256": "b" * 64,
            "parent_index_manifest_sha256": "c" * 64,
            "parent_event_candidates_sha256": "d" * 64,
            "parent_source_groups_sha256": "e" * 64,
        }
        payloads = {
            "parent_manifest": {"source_release_manifest_sha256": SOURCE_SHA},
            "parent_inventory_seal_sha256": "f" * 64,
            "policy": policy,
            "events": events,
            "queue_rows": rows,
            "source_group_rows": selected_groups,
            "selected": selected,
            "all_phase_candidates": all_candidates,
            "coverage": phase._coverage(selected, all_candidates, expectations),
            "excluded_roles": {},
            "excluded_vocabulary": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "phase-output"
            result = phase._write_payloads(output, payloads)
            self.assertEqual(result["selected"], 40)
            self.assertEqual(set(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()),
                             set(phase.REQUIRED_OUTPUT_FILES))
            self.assertEqual(phase._validate_existing(
                output, payloads, expected_phase_queue_seal_sha256=result["phase_queue_seal_sha256"])["status"],
                "RESUME_VALIDATED")
            mini = output / "phase_candidate_index"
            manifest, event_path, groups = base.validate_index(
                mini, expected_source_manifest_sha256=SOURCE_SHA, canonical_protocol=owner_protocol,
                expected_protocol_sha256=PROTOCOL_SHA)
            self.assertEqual(manifest["event_candidates"], 40)
            self.assertEqual(len(base.parse_candidates(
                event_path, groups, expected_source_manifest_sha256=SOURCE_SHA, canonical_protocol=owner_protocol,
                expected_task_ids=range(5), expected_skill_ids=[10], retained_usage_roles={"annotation_calibration"},
                max_retained_candidates=40, expected_event_receipt=manifest["files"]["event_candidates.jsonl"],
                boundary_gap_frames=1, long_interval_frames=1)[0]), 40)
            binding = renderer._load_protocol(PROTOCOL_PATH, expected_sha256=PROTOCOL_SHA)
            packets = renderer.create_packets(
                mini, root / "locator-packets", event_ids={events[0]["event_id"]}, limit=None, questions={},
                include_source_annotation_context=False, decode=False, raw_root=None, contact_sheets=False,
                protocol_binding=binding, max_seconds=10)
            self.assertEqual(packets["status"], "LOCATORS_READY_RENDER_PENDING")
            packet = json.loads((root / "locator-packets" / "packets.jsonl").read_text())
            self.assertEqual(packet["status"], "LOCATOR_READY_NO_RGB")
            self.assertEqual(packet["actor_packet"]["images"], {})


if __name__ == "__main__":
    unittest.main()
