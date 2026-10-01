import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import build_memlite_event_index as builder  # noqa: E402


FIXTURE = REPO / "tests" / "fixtures" / "p107_v4_compact_fixture.json"


def strict_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_release(root):
    fixture = json.loads(FIXTURE.read_text())
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(fixture["release_manifest"], sort_keys=True))
    (root / "episodes.jsonl").write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in fixture["episodes"]))
    (root / "split_provenance.json").write_text(json.dumps(fixture["split_provenance"], sort_keys=True))
    return fixture


def coverage(fixture):
    return fixture["coverage_expectations"]


class BuildEventIndexTests(unittest.TestCase):
    def test_builds_deterministic_source_group_index_without_outcome_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            output = root / "external-index"
            result = builder.build_index(release, output, calibration_per_task=1,
                                         coverage_expectations=coverage(fixture), max_seconds=10)
            self.assertEqual(result["status"], "METADATA_ONLY_READY_FOR_PACKET_RENDER")
            self.assertFalse(result["outcome_supervision"])
            self.assertFalse(result["corrective_action_supervision"])
            self.assertEqual(result["candidate_counts"]["event_candidates"], 4)
            self.assertEqual(result["accepted_label_counts"]["accepted_temporal_windows"], 0)
            self.assertEqual(result["accepted_label_counts"]["accepted_corrective_action_windows"], 0)
            groups = strict_lines(output / "source_groups.jsonl")
            events = strict_lines(output / "event_candidates.jsonl")
            self.assertEqual(len(events), 4)
            self.assertEqual(len({event["event_id"] for event in events}), 4)
            train_group = [row for row in groups if row["task_instance_id"] == 1][0]
            self.assertEqual(train_group["source_episode_ids"], [10, 11])
            self.assertEqual(sum(row["usage_role"] == "annotation_calibration" for row in groups), 1)
            self.assertEqual(sum(row["usage_role"] == "student_candidate" for row in groups), 1)
            self.assertEqual([row["usage_role"] for row in groups if row["original_split"] == "eval"], ["evaluation_only"])
            roles = {row["source_group_id"]: row["usage_role"] for row in groups}
            for event in events:
                self.assertEqual(event["evidence"]["kind"], "MISSING")
                self.assertIsNone(event["action"]["actual_executed_length"])
                self.assertEqual(event["action"]["start_frame"], event["observation"]["frame"])
                self.assertEqual(event["usage_role"], roles[event["source"]["source_group_id"]])
            self.assertEqual(result["coverage"]["missing_required_task_skill_pairs"], None)
            self.assertEqual(result["coverage"]["required_pair_coverage_status"], "NOT_DECLARED")
            self.assertTrue(result["coverage"]["global_vocabulary_complete"])
            resume = builder.resume_index(release, output, calibration_per_task=1,
                                          coverage_expectations=coverage(fixture),
                                          expected_inventory_seal_sha256=result["inventory_seal_sha256"])
            self.assertEqual(resume["status"], "RESUME_VALIDATED")
            with self.assertRaises(ValueError):
                builder.resume_index(release, output, calibration_per_task=0,
                                     coverage_expectations=coverage(fixture),
                                     expected_inventory_seal_sha256=result["inventory_seal_sha256"])
            with self.assertRaises(FileExistsError):
                builder.build_index(release, output, coverage_expectations=coverage(fixture), max_seconds=10)

    def test_duplicate_source_identity_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            duplicate = fixture["episodes"][0]
            with (release / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(duplicate, sort_keys=True) + "\n")
            output = root / "external-index"
            with self.assertRaisesRegex(ValueError, "duplicate raw source episode"):
                builder.build_index(release, output, coverage_expectations=coverage(fixture), max_seconds=10)
            self.assertFalse(output.exists())

    def test_duplicate_published_episode_index_fails_even_with_distinct_raw_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            duplicate = json.loads(json.dumps(fixture["episodes"][0]))
            duplicate["row"]["raw_episode_id"] = 999
            duplicate["annotation_sha256"] = "f" * 64
            with (release / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(duplicate, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate published episode_index"):
                builder.build_index(release, root / "output", coverage_expectations=coverage(fixture), max_seconds=10)

    def test_generic_coverage_grid_reports_missing_without_hardcoding_full_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            expected = json.loads(json.dumps(coverage(fixture)))
            expected["expected_skill_verbs"] = ["GRASP", "HOLD", "NAVIGATE", "PLACE", "PRESS"]
            expected["required_task_skill_pairs"] = [{"task_index": 0, "skill_verb": "PLACE"}]
            result = builder.build_index(release, root / "output", coverage_expectations=expected, max_seconds=10)
            self.assertEqual(result["coverage"]["required_pair_coverage_status"], "INCOMPLETE")
            self.assertIn({"task_index": 0, "skill_verb": "PLACE"},
                          result["coverage"]["missing_required_task_skill_pairs"])

    def test_explicit_prefix_pilot_is_sealed_partial_and_cannot_resume_as_full(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            output = root / "partial-index"
            result = builder.build_index(release, output, calibration_per_task=0,
                                         coverage_expectations=coverage(fixture),
                                         max_source_episodes=2, max_seconds=10)
            self.assertTrue(result["partial_source_coverage"])
            self.assertEqual(result["source_episodes"], 2)
            self.assertEqual(result["selection_policy"]["mode"], "PUBLISHED_EPISODE_PREFIX")
            self.assertEqual(result["selection_policy"]["max_source_episodes"], 2)
            self.assertEqual(result["source_release_files"]["episodes.jsonl"],
                             builder.sha256_file(release / "episodes.jsonl"))
            self.assertEqual(builder.resume_index(
                release, output, calibration_per_task=0, coverage_expectations=coverage(fixture),
                max_source_episodes=2,
                expected_inventory_seal_sha256=result["inventory_seal_sha256"])["status"], "RESUME_VALIDATED")
            with self.assertRaisesRegex(ValueError, "source selection policy changed"):
                builder.resume_index(
                    release, output, calibration_per_task=0, coverage_expectations=coverage(fixture),
                    expected_inventory_seal_sha256=result["inventory_seal_sha256"])

    def test_resume_requires_external_seal_and_rejects_empty_payload_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            output = root / "output"
            result = builder.build_index(release, output, coverage_expectations=coverage(fixture), max_seconds=10)
            with self.assertRaisesRegex(ValueError, "externally recorded"):
                builder.resume_index(release, output, calibration_per_task=1, coverage_expectations=coverage(fixture))
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["files"] = {}
            (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
            seal = json.loads((output / "inventory_seal.json").read_text())
            seal["index_manifest_sha256"] = builder.sha256_file(output / "manifest.json")
            seal["payload_files"] = {}
            (output / "inventory_seal.json").write_text(json.dumps(seal, sort_keys=True, separators=(",", ":")))
            with self.assertRaisesRegex(ValueError, "exact expected payload"):
                builder.resume_index(release, output, calibration_per_task=1, coverage_expectations=coverage(fixture),
                                     expected_inventory_seal_sha256=builder.sha256_file(output / "inventory_seal.json"))

    def test_provenance_mismatch_and_public_test_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            fixture["split_provenance"]["assignments"][0]["split"] = "eval"
            (release / "split_provenance.json").write_text(json.dumps(fixture["split_provenance"], sort_keys=True))
            with self.assertRaisesRegex(ValueError, "does not match immutable"):
                builder.build_index(release, root / "bad-split", coverage_expectations=coverage(fixture), max_seconds=10)
            fixture = write_release(root / "second-release")
            fixture["episodes"][0]["row"]["task_instance_id"] = 301
            (root / "second-release" / "episodes.jsonl").write_text("".join(
                json.dumps(row, sort_keys=True) + "\n" for row in fixture["episodes"]))
            with self.assertRaisesRegex(ValueError, "public-test"):
                builder.build_index(root / "second-release", root / "bad-public",
                                    coverage_expectations=coverage(fixture), max_seconds=10)

    def test_explicit_calibration_group_is_union_sealed_and_group_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            reservations = root / "reserved-groups.jsonl"
            reservations.write_text(json.dumps({"task_index": 0, "task_instance_id": 1}) + "\n")
            reserved = builder.read_calibration_group_ids(reservations, builder.sha256_file(release / "manifest.json"))
            output = root / "index"
            result = builder.build_index(release, output, calibration_per_task=0,
                                         calibration_group_ids=reserved, coverage_expectations=coverage(fixture), max_seconds=10)
            self.assertEqual(result["calibration_selection"]["explicit_group_ids"], sorted(reserved))
            self.assertEqual(len(result["calibration_selection"]["policy_sha256"]), 64)
            groups = strict_lines(output / "source_groups.jsonl")
            group = next(row for row in groups if row["task_instance_id"] == 1)
            self.assertEqual(group["usage_role"], "annotation_calibration")
            events = strict_lines(output / "event_candidates.jsonl")
            self.assertTrue(all(row["usage_role"] == "annotation_calibration"
                                for row in events if row["source"]["source_group_id"] == group["source_group_id"]))
            self.assertEqual(builder.resume_index(release, output, calibration_per_task=0,
                                                  calibration_group_ids=reserved, coverage_expectations=coverage(fixture),
                                                  expected_inventory_seal_sha256=result["inventory_seal_sha256"])["status"], "RESUME_VALIDATED")
            with self.assertRaisesRegex(ValueError, "calibration policy changed"):
                builder.resume_index(release, output, calibration_per_task=0, coverage_expectations=coverage(fixture),
                                     expected_inventory_seal_sha256=result["inventory_seal_sha256"])

    def test_source_output_and_tiny_cpu_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            fixture = write_release(release)
            with self.assertRaises(ValueError):
                builder.build_index(release, release / "bad", coverage_expectations=coverage(fixture), max_seconds=10)
            with self.assertRaises(TimeoutError):
                builder.build_index(release, root / "timeout", coverage_expectations=coverage(fixture), max_seconds=1e-12)


if __name__ == "__main__":
    unittest.main()
