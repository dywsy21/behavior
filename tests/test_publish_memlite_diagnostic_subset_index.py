"""Focused tests for the sealed, non-training phase40 subset publisher."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import build_memlite_event_index as indexer  # noqa: E402
import pack_memlite_event_labels as pack  # noqa: E402
import publish_memlite_diagnostic_subset_index as publisher  # noqa: E402
import select_memlite_event_annotation_queue as base  # noqa: E402


FIXTURE = REPO / "tests" / "fixtures" / "p107_v4_compact_fixture.json"


def _write_release(root: Path) -> None:
    fixture = json.loads(FIXTURE.read_text())
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(fixture["release_manifest"], sort_keys=True))
    (root / "episodes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fixture["episodes"]))
    (root / "split_provenance.json").write_text(json.dumps(fixture["split_provenance"], sort_keys=True))


def _receipt(path: Path, rows: int) -> dict:
    return {"sha256": base.sha256_file(path), "rows": rows, "bytes": path.stat().st_size}


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(base.canonical_json(value).encode("utf-8") + b"\n")


def _reseal(index: Path) -> str:
    manifest = base.read_json(index / "manifest.json")
    seal = {
        "schema_version": "memlite-event-inventory-seal-v1",
        "index_manifest_sha256": base.sha256_file(index / "manifest.json"),
        "source_release_manifest_sha256": manifest["source_release_manifest_sha256"],
        "coverage_expectations_sha256": manifest["coverage_expectations_sha256"],
        "expected_payload_files": ["event_candidates.jsonl", "source_groups.jsonl"],
        "payload_files": manifest["files"],
    }
    _write_json(index / "inventory_seal.json", seal)
    return base.sha256_file(index / "inventory_seal.json")


def _full_v3_contract(source_release_sha: str) -> dict:
    return {
        "schema_version": "p107-official-coverage-expectations-v3",
        "official_task_metadata_sha256": source_release_sha,
        "official_skill_vocabulary_sha256": "b" * 64,
        "expected_task_ids": [0, 1],
        "expected_skill_vocabulary": [
            {"skill_id": 1, "skill_description": "synthetic skill"},
            {"skill_id": 2, "skill_description": "missing synthetic skill"},
        ],
        "required_task_skill_pairs": None,
    }


def _make_fixture(root: Path) -> dict:
    """Create a fully sealed synthetic full/mini/phase chain with 40 IDs."""
    fixture = json.loads(FIXTURE.read_text())
    release, original = root / "release", root / "original-index"
    _write_release(release)
    indexer.build_index(release, original, calibration_per_task=1,
                        coverage_expectations=fixture["coverage_expectations"], max_seconds=10)
    full = root / "full-v3"
    shutil.copytree(original, full)
    full_manifest = base.read_json(full / "manifest.json")
    expectations = _full_v3_contract(full_manifest["source_release_manifest_sha256"])
    coverage_sha = base.canonical_sha256(expectations)
    full_manifest.update({
        "coverage_expectations_sha256": coverage_sha,
        "coverage": {
            "expectations": expectations, "expectations_sha256": coverage_sha,
            "found_task_ids": [0, 1], "missing_task_ids": [], "unexpected_task_ids": [],
            "found_global_skill_ids": [1, 2], "missing_global_skill_ids": [],
            "unmapped_source_skill_ids": [], "source_skill_members_missing_skill_id": 0,
            "global_vocabulary_complete": True, "required_task_skill_pairs": None,
            "missing_required_task_skill_pairs": None, "required_pair_coverage_status": "NOT_DECLARED",
            "unique_input_source_episodes": full_manifest["source_episodes"],
            "unique_candidate_source_episodes": full_manifest["source_episodes"],
            "unique_event_candidates": full_manifest["event_candidates"],
        },
    })
    original_event = base.read_jsonl(original / "event_candidates.jsonl")[0]
    protocol_sha = hashlib.sha256((REPO / "src" / "g05" / "data" / "memlite_event_protocol.py").read_bytes()).hexdigest()
    mini = root / "mini"
    mini.mkdir()
    events, parent_events, groups, queue_rows = [], [], [], []
    for order in range(publisher.EXPECTED_EVENT_COUNT):
        event = deepcopy(original_event)
        task_instance, episode = order + 1, order + 1000
        group_id = pack._protocol.source_group_id({
            "source_release_manifest_sha256": full_manifest["source_release_manifest_sha256"],
            "task_index": 0, "task_instance_id": task_instance,
        })
        source = dict(event["source"])
        source.update({
            "source_release_manifest_sha256": full_manifest["source_release_manifest_sha256"],
            "source_annotation_sha256": hashlib.sha256(f"annotation-{order}".encode()).hexdigest(),
            "source_group_id": group_id, "task_index": 0, "task_instance_id": task_instance,
            "raw_episode_id": episode, "episode_index": episode,
        })
        event["source"] = source
        event["event_kind"] = "ANNOTATED_SKILL_SEGMENT"
        event["bundle_id"] = hashlib.sha256(f"parent-bundle-{order}".encode()).hexdigest()
        event["usage_role"] = "annotation_calibration"
        event["evidence"] = {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None}
        event["skill_bundle"][0].update({
            "skill_id": 1, "skill_start": event["event_interval"]["start_frame"],
            "skill_end": event["event_interval"]["end_frame"], "raw_description": "synthetic skill",
        })
        event["event_id"] = pack._protocol.event_id(event)
        pack.validate_event(event)
        parent_events.append(deepcopy(event))
        skill = event["skill_bundle"][0]
        event["event_kind"] = "ANNOTATED_SKILL_SEGMENT_PHASE_ENTRY"
        event["bundle_id"] = hashlib.sha256(f"phase-bundle-{order}".encode()).hexdigest()
        event["phase_lineage"] = {
            "schema_version": "p107-phase-balanced-mini-index-v1", "parent_event_id": parent_events[-1]["event_id"],
            "parent_event_interval": deepcopy(parent_events[-1]["event_interval"]), "observation_phase": "ENTRY",
            "selection_stratum": "ENTRY", "parent_skill_identity": {
                "parent_event_id": parent_events[-1]["event_id"], "parent_skill_index": 0,
                "parent_skill_member_sha256": base.canonical_sha256(skill), "skill_id": skill["skill_id"],
                "skill_start_frame": skill["skill_start"], "skill_end_frame": skill["skill_end"],
            },
            "queried_skill_start_frame": skill["skill_start"], "queried_skill_end_frame": skill["skill_end"],
            "goal_query": {"query_kind": "METADATA_DESCRIBED_SKILL_GOAL_RELATION",
                           "query_scope": "SAME_ORIGINAL_ANNOTATED_SEGMENT_SKILL",
                           "source_skill_is_attempted_instruction_not_observed_outcome": True,
                           "binding_status": "BOUND_METADATA_NEEDS_VISUAL_CONFIRMATION",
                           "unknown_is_required_when_relation_or_entity_is_not_visually_grounded": True,
                           "queried_skill": {field: skill.get(field) for field in (
                               "skill_id", "verb", "target", "source", "destination", "target_part", "arm",
                               "raw_description", "skill_start", "skill_end")}},
        }
        event["event_id"] = pack._protocol.event_id(event)
        pack.validate_event(event)
        events.append(event)
        groups.append({
            "schema_version": pack.SCHEMA_VERSION, "source_group_id": group_id,
            "source_release_manifest_sha256": source["source_release_manifest_sha256"],
            "task_index": 0, "task_instance_id": task_instance, "original_split": "train",
            "usage_role": "annotation_calibration", "source_episode_ids": [episode],
            "source_episode_count": 1,
        })
        queue_rows.append({
            "schema_version": "p107-phase-balanced-calibration-queue-v1", "event_id": event["event_id"],
            "selection_order": order, "usage_role": "annotation_calibration", "immutable_split": "train",
            "training_eligible": False, "observation_phase_is_not_outcome": True,
            "current_actor_evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
            "review_constraints": {
                "source_segment_end_is_not_completion_truth": True,
                "no_success_failure_outcome_or_recovery_label": True,
                "future_frames_are_only_offline_review_for_this_new_anchor": True,
                "no_action_supervision_emitted": True,
            },
            "source_identity": {key: source[key] for key in (
                "source_release_manifest_sha256", "source_annotation_sha256", "source_group_id", "task_index",
                "task_instance_id", "raw_episode_id", "episode_index")},
            "observation_frame": event["observation"]["frame"], "observation_phase": "ENTRY",
            "selection_stratum": "ENTRY", "parent_event_id": parent_events[-1]["event_id"],
            "parent_skill_identity": deepcopy(event["phase_lineage"]["parent_skill_identity"]),
            "queried_skill": deepcopy(event["phase_lineage"]["goal_query"]),
            "queried_skill_start_frame": skill["skill_start"], "queried_skill_end_frame": skill["skill_end"],
            "original_annotated_segment": deepcopy(parent_events[-1]["event_interval"]),
            "temporal_windows": {"anchor_frame": event["observation"]["frame"],
                                 "actor_available_window": {"causal_use": "CURRENT_DECISION_ONLY",
                                                            "sampled_frames": [event["observation"]["frame"]],
                                                            "end_frame_exclusive": event["observation"]["frame"] + 1}},
        })
    (full / "event_candidates.jsonl").write_bytes(
        b"".join(base.canonical_json(row).encode("utf-8") + b"\n" for row in parent_events))
    (full / "source_groups.jsonl").write_bytes(
        b"".join(base.canonical_json(row).encode("utf-8") + b"\n" for row in groups))
    full_manifest.update({
        "source_episodes": 40, "source_groups": 40, "event_candidates": 40,
        "usage_roles": {"annotation_calibration": 40},
        "files": {"event_candidates.jsonl": _receipt(full / "event_candidates.jsonl", 40),
                  "source_groups.jsonl": _receipt(full / "source_groups.jsonl", 40)},
    })
    full_manifest["coverage"].update({"unique_input_source_episodes": 40,
                                      "unique_candidate_source_episodes": 40,
                                      "unique_event_candidates": 40})
    _write_json(full / "manifest.json", full_manifest)
    full_inventory = _reseal(full)
    full_manifest_sha = base.sha256_file(full / "manifest.json")
    base.write_jsonl(mini / "event_candidates.jsonl", events)
    base.write_jsonl(mini / "source_groups.jsonl", groups)
    files = {"event_candidates.jsonl": _receipt(mini / "event_candidates.jsonl", 40),
             "source_groups.jsonl": _receipt(mini / "source_groups.jsonl", 40)}
    mini_manifest = {
        "schema_version": "memlite-event-index-v1", "status": "METADATA_ONLY_READY_FOR_PACKET_RENDER",
        "training_eligible": False, "source_release_manifest_sha256": full_manifest["source_release_manifest_sha256"],
        "protocol_sha256": protocol_sha, "coverage_expectations_sha256": coverage_sha, "files": files,
        "event_candidates": 40, "source_groups": 40, "source_episodes": 40,
        "usage_roles": {"annotation_calibration": 40}, "outcome_supervision": False,
        "corrective_action_supervision": False, "partial_source_coverage": True,
        "derivation": {"schema_version": "p107-phase-balanced-mini-index-v1",
                       "parent_index_manifest_sha256": full_manifest_sha,
                       "parent_inventory_seal_sha256": full_inventory,
                       "parent_event_candidates_sha256": full_manifest["files"]["event_candidates.jsonl"]["sha256"],
                       "phase_observation_is_not_outcome": True},
    }
    _write_json(mini / "manifest.json", mini_manifest)
    mini_inventory = _reseal(mini)
    queue_path = root / "phase-queue.jsonl"
    base.write_jsonl(queue_path, queue_rows)
    queue_sha = base.sha256_file(queue_path)
    policy = {
        "schema_version": "p107-phase-balanced-calibration-queue-v1",
        "parent_index_manifest_sha256": full_manifest_sha, "parent_inventory_seal_sha256": full_inventory,
        "source_release_manifest_sha256": full_manifest["source_release_manifest_sha256"],
        "protocol_sha256": protocol_sha, "coverage_expectations_sha256": coverage_sha,
    }
    policy["policy_sha256"] = base.canonical_sha256(policy)
    selection = {
        "schema_version": "p107-phase-balanced-calibration-manifest-v1",
        "mini_index": {"manifest_sha256": base.sha256_file(mini / "manifest.json"),
                       "inventory_seal_sha256": mini_inventory},
        "policy": policy,
        "selection": {"exact_budget": 40, "unique_source_groups": 40, "unique_source_episodes": 40,
                      "all_usage_role_annotation_calibration": True, "all_immutable_split_train": True,
                      "rows": [{"event_id": event["event_id"],
                                "source_group_id": event["source"]["source_group_id"],
                                "task_index": event["source"]["task_index"]} for event in events]},
    }
    selection_path = root / "phase-selection.json"
    _write_json(selection_path, selection)
    return {
        "mini": mini, "mini_inventory": mini_inventory, "full": full, "full_inventory": full_inventory,
        "full_manifest_sha": full_manifest_sha, "selection": selection_path,
        "selection_sha": base.sha256_file(selection_path), "queue": queue_path, "queue_sha": queue_sha,
        "events": events,
    }


def _publish(inputs: dict, output: Path) -> dict:
    return publisher.publish(
        phase_index=inputs["mini"], expected_phase_index_inventory_seal_sha256=inputs["mini_inventory"],
        full_index=inputs["full"], expected_full_index_inventory_seal_sha256=inputs["full_inventory"],
        expected_full_index_manifest_sha256=inputs["full_manifest_sha"],
        phase_selection_manifest=inputs["selection"],
        expected_phase_selection_manifest_sha256=inputs["selection_sha"],
        phase_queue=inputs["queue"], expected_phase_queue_sha256=inputs["queue_sha"], output=output)


def _unknown_goal_annotations(event: dict) -> dict:
    view = {
        "schema_version": pack.SCHEMA_VERSION, "label_kind": "goal_satisfaction_counterfactual", "view_id": "",
        "event_id": event["event_id"], "source_group_id": event["source"]["source_group_id"],
        "observation_frame": event["observation"]["frame"], "annotation_provenance": "agent",
        "review_status": "PROPOSED", "actor_evidence": {
            "kind": "MISSING", "evidence_end_frame": None, "available_frame": None, "references": []},
        "goal_relation": "synthetic relation", "goal_satisfaction": "UNKNOWN", "valid_goal_mask": False,
        "attempt_outcome": "NOT_APPLICABLE", "evidence": {
            "kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "low_action_supervision_mask": False,
    }
    view["view_id"] = pack._protocol.view_id(view)
    return {
        "schema_version": pack.ANNOTATION_SCHEMA, "views": [view], "action_payloads": [],
        "view_reviews": [{
            "view_id": view["view_id"], "candidate_annotated": True, "parent_reviewed": False,
            "accepted_auxiliary": False, "outcome_validated": False,
            "reviewer": {"kind": "agent", "id": "synthetic", "model": "gpt-5.6-luna"},
        }],
        "root_visual_review": {"completed": False},
    }


class DiagnosticSubsetPublisherTests(unittest.TestCase):
    def test_publishes_truthful_partial_index_that_generic_packer_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = _make_fixture(Path(directory))
            output = Path(directory) / "diagnostic-subset"
            result = _publish(inputs, output)
            manifest = pack.read_json(output / "manifest.json")
            self.assertEqual(result["event_candidates"], 40)
            self.assertFalse(result["coverage_complete"])
            self.assertFalse(result["ready_for_training"])
            self.assertEqual(manifest["release_roles"], ["annotation_calibration"])
            self.assertIsNone(manifest["derivation"]["legacy_mini_derivation_parent_source_groups_sha256"])
            self.assertEqual(manifest["derivation"]["full_v3_parent_source_groups_receipt"]["rows"], 40)
            self.assertFalse(manifest["coverage"]["global_vocabulary_complete"])
            self.assertEqual(manifest["coverage"]["required_pair_coverage_status"], "NOT_DECLARED")
            self.assertIsNone(manifest["coverage"]["missing_required_task_skill_pairs"])
            normalized = pack._normalize_index_coverage(manifest)
            self.assertTrue(normalized["diagnostic_only"])
            self.assertFalse(normalized["coverage_complete"])
            _, sealed_events, _, _ = pack._read_sealed_index(
                output, expected_inventory_seal_sha256=result["inventory_seal_sha256"])
            self.assertEqual(len(sealed_events), 40)
            labels = Path(directory) / "labels.json"
            _write_json(labels, _unknown_goal_annotations(inputs["events"][0]))
            package = pack.publish(
                output, labels, Path(directory) / "generic-calibration-package",
                release_role="annotation_calibration", minimum_scale=None,
                request_dataset_quality_eligibility=False,
                expected_index_inventory_seal_sha256=result["inventory_seal_sha256"])
            self.assertEqual(package["release_eligibility"], "CANDIDATE_ONLY")
            self.assertFalse(package["ready_for_training"])

    def test_rejects_tamper_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = _make_fixture(Path(directory))
            queue = inputs["queue"]
            queue.write_bytes(queue.read_bytes() + b"\n")
            output = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "phase queue SHA-256"):
                _publish(inputs, output)
            self.assertFalse(output.exists())
            inputs = _make_fixture(Path(directory) / "second")
            inputs["full_manifest_sha"] = "0" * 64
            output = Path(directory) / "must-not-exist-source"
            with self.assertRaisesRegex(ValueError, "full-v3 manifest does not match"):
                _publish(inputs, output)
            self.assertFalse(output.exists())
            inputs = _make_fixture(Path(directory) / "third")
            payload = inputs["mini"] / "event_candidates.jsonl"
            payload.write_bytes(payload.read_bytes() + b"\n")
            output = Path(directory) / "must-not-exist-payload"
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                _publish(inputs, output)
            self.assertFalse(output.exists())
            inputs = _make_fixture(Path(directory) / "fourth")
            incompatible_protocol = Path(directory) / "incompatible_protocol.py"
            incompatible_protocol.write_text("# not the sealed protocol\n")
            output = Path(directory) / "must-not-exist-protocol"
            with self.assertRaisesRegex(ValueError, "canonical protocol module does not match"):
                publisher.publish(
                    phase_index=inputs["mini"], expected_phase_index_inventory_seal_sha256=inputs["mini_inventory"],
                    full_index=inputs["full"], expected_full_index_inventory_seal_sha256=inputs["full_inventory"],
                    expected_full_index_manifest_sha256=inputs["full_manifest_sha"],
                    phase_selection_manifest=inputs["selection"],
                    expected_phase_selection_manifest_sha256=inputs["selection_sha"],
                    phase_queue=inputs["queue"], expected_phase_queue_sha256=inputs["queue_sha"],
                    output=output, protocol_path=incompatible_protocol)
            self.assertFalse(output.exists())
            inputs = _make_fixture(Path(directory) / "fifth")
            full_manifest = base.read_json(inputs["full"] / "manifest.json")
            full_manifest["protocol_sha256"] = "0" * 64
            _write_json(inputs["full"] / "manifest.json", full_manifest)
            inputs["full_inventory"] = _reseal(inputs["full"])
            inputs["full_manifest_sha"] = base.sha256_file(inputs["full"] / "manifest.json")
            output = Path(directory) / "must-not-exist-conflicting-protocol"
            with self.assertRaisesRegex(ValueError, "do not pin the same canonical protocol"):
                _publish(inputs, output)
            self.assertFalse(output.exists())

    def test_rejects_queue_source_disagreement_and_wrong_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _make_fixture(root)
            rows = base.read_jsonl(inputs["queue"])
            rows[0]["source_identity"]["task_instance_id"] = 999
            base.write_jsonl(root / "bad-queue.jsonl", rows)
            inputs["queue"] = root / "bad-queue.jsonl"
            inputs["queue_sha"] = base.sha256_file(inputs["queue"])
            output = root / "must-not-exist-source"
            with self.assertRaisesRegex(ValueError, "event/source binding"):
                _publish(inputs, output)
            self.assertFalse(output.exists())
            inputs = _make_fixture(root / "second")
            selection = base.read_json(inputs["selection"])
            selection["selection"]["exact_budget"] = 39
            _write_json(inputs["selection"], selection)
            inputs["selection_sha"] = base.sha256_file(inputs["selection"])
            output = root / "must-not-exist-count"
            with self.assertRaisesRegex(ValueError, "fixed 40-row"):
                _publish(inputs, output)
            self.assertFalse(output.exists())

    def test_rejects_actual_phase_queue_semantic_tampering(self):
        mutations = (
            ("observation_frame", lambda row: row.update(observation_frame=999999), "observation frame"),
            ("parent_event", lambda row: row.update(parent_event_id="0" * 64), "skill/parent/segment"),
            ("queried_skill", lambda row: row["queried_skill"]["queried_skill"].update(skill_id=999),
             "queried-skill object"),
            ("phase", lambda row: row.update(observation_phase="MID"), "observation phase"),
            ("actor_domain", lambda row: row["temporal_windows"]["actor_available_window"].update(
                sampled_frames=[999999]), "actor-causal temporal window"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (_, mutate, message) in enumerate(mutations):
                inputs = _make_fixture(root / str(index))
                rows = base.read_jsonl(inputs["queue"])
                mutate(rows[0])
                changed = root / f"changed-{index}.jsonl"
                base.write_jsonl(changed, rows)
                inputs["queue"], inputs["queue_sha"] = changed, base.sha256_file(changed)
                with self.assertRaisesRegex(ValueError, message):
                    _publish(inputs, root / f"must-not-exist-{index}")

    def test_rejects_noncalibration_group_and_duplicate_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _make_fixture(root)
            groups = base.read_jsonl(inputs["mini"] / "source_groups.jsonl")
            groups[0]["usage_role"] = "student_candidate"
            (inputs["mini"] / "source_groups.jsonl").write_bytes(
                b"".join(base.canonical_json(row).encode("utf-8") + b"\n" for row in groups))
            manifest = base.read_json(inputs["mini"] / "manifest.json")
            manifest["files"]["source_groups.jsonl"] = _receipt(inputs["mini"] / "source_groups.jsonl", 40)
            _write_json(inputs["mini"] / "manifest.json", manifest)
            inputs["mini_inventory"] = _reseal(inputs["mini"])
            selection = base.read_json(inputs["selection"])
            selection["mini_index"]["manifest_sha256"] = base.sha256_file(inputs["mini"] / "manifest.json")
            selection["mini_index"]["inventory_seal_sha256"] = inputs["mini_inventory"]
            _write_json(inputs["selection"], selection)
            inputs["selection_sha"] = base.sha256_file(inputs["selection"])
            with self.assertRaisesRegex(ValueError, "non-calibration"):
                _publish(inputs, root / "must-not-exist-role")
            inputs = _make_fixture(root / "eval")
            groups = base.read_jsonl(inputs["mini"] / "source_groups.jsonl")
            groups[0].update(original_split="eval", usage_role="evaluation_only")
            (inputs["mini"] / "source_groups.jsonl").write_bytes(
                b"".join(base.canonical_json(row).encode("utf-8") + b"\n" for row in groups))
            manifest = base.read_json(inputs["mini"] / "manifest.json")
            manifest["files"]["source_groups.jsonl"] = _receipt(inputs["mini"] / "source_groups.jsonl", 40)
            _write_json(inputs["mini"] / "manifest.json", manifest)
            inputs["mini_inventory"] = _reseal(inputs["mini"])
            selection = base.read_json(inputs["selection"])
            selection["mini_index"]["manifest_sha256"] = base.sha256_file(inputs["mini"] / "manifest.json")
            selection["mini_index"]["inventory_seal_sha256"] = inputs["mini_inventory"]
            _write_json(inputs["selection"], selection)
            inputs["selection_sha"] = base.sha256_file(inputs["selection"])
            with self.assertRaisesRegex(ValueError, "non-calibration"):
                _publish(inputs, root / "must-not-exist-eval")
            inputs = _make_fixture(root / "second")
            events = base.read_jsonl(inputs["mini"] / "event_candidates.jsonl")
            events[-1] = deepcopy(events[0])
            (inputs["mini"] / "event_candidates.jsonl").write_bytes(
                b"".join(base.canonical_json(row).encode("utf-8") + b"\n" for row in events))
            manifest = base.read_json(inputs["mini"] / "manifest.json")
            manifest["files"]["event_candidates.jsonl"] = _receipt(inputs["mini"] / "event_candidates.jsonl", 40)
            _write_json(inputs["mini"] / "manifest.json", manifest)
            inputs["mini_inventory"] = _reseal(inputs["mini"])
            selection = base.read_json(inputs["selection"])
            selection["mini_index"]["manifest_sha256"] = base.sha256_file(inputs["mini"] / "manifest.json")
            selection["mini_index"]["inventory_seal_sha256"] = inputs["mini_inventory"]
            _write_json(inputs["selection"], selection)
            inputs["selection_sha"] = base.sha256_file(inputs["selection"])
            with self.assertRaisesRegex(ValueError, "absent or duplicate event IDs"):
                _publish(inputs, root / "must-not-exist-duplicate")


if __name__ == "__main__":
    unittest.main()
