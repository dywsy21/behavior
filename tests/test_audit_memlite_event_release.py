"""Release audit and view-specific dataset checks for P107 preparation data."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
sys.path.insert(0, str(REPO / "tests"))
import audit_memlite_event_release as audit  # noqa: E402
import pack_memlite_event_labels as pack  # noqa: E402
import test_pack_memlite_event_labels as fixture  # noqa: E402


def load_dataset_module():
    import importlib.util

    path = REPO / "src/g05/data/memlite_event_dataset.py"
    spec = importlib.util.spec_from_file_location("p107_dataset_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATASET = load_dataset_module()


class AuditMemLiteEventReleaseTests(unittest.TestCase):
    def _package(self, root: Path) -> tuple[Path, str]:
        index_path, _, events, _ = fixture.indexed(root)
        student = next(event for event in events if event["usage_role"] == "student_candidate")
        labels = fixture.annotations(student)
        labels_path, package = root / "labels.json", root / "sealed-package"
        labels_path.write_text(pack.canonical_json(labels))
        pack.publish(index_path, labels_path, package, release_role="student_candidate", minimum_scale=None,
                     request_dataset_quality_eligibility=False,
                     expected_index_inventory_seal_sha256=pack.sha256_file(index_path / "inventory_seal.json"))
        return package, pack.sha256_file(package / "release_seal.json")

    def _quality_action_package(self, root: Path) -> tuple[Path, str, Path]:
        index_path, groups, events, _ = fixture.indexed(root)
        student = next(event for event in events if event["usage_role"] == "student_candidate")
        index_seal = pack.sha256_file(index_path / "inventory_seal.json")
        publisher_root = root / "parent-publisher"
        labels, authority_manifest_sha = fixture.approve_logged_corrective(
            fixture.annotations(student), student, groups, publisher_root,
            index_inventory_seal_sha256=index_seal)
        labels_path, package = root / "labels.json", root / "quality-package"
        labels_path.write_text(pack.canonical_json(labels))
        pack.publish(index_path, labels_path, package, release_role="student_candidate", minimum_scale=None,
                     request_dataset_quality_eligibility=False,
                     expected_index_inventory_seal_sha256=index_seal,
                     corrective_publisher_root=publisher_root,
                     expected_corrective_publisher_manifest_sha256=authority_manifest_sha)
        return package, pack.sha256_file(package / "release_seal.json"), publisher_root

    def test_audit_reports_all_statuses_but_blocks_stage3_training(self):
        with tempfile.TemporaryDirectory() as folder:
            package, seal = self._package(Path(folder))
            result = audit.audit(package, expected_release_seal_sha256=seal)
            self.assertEqual(result["status"], "PASS_WITH_STAGE3_TRAINING_BLOCKED")
            self.assertFalse(result["ready_for_training"])
            self.assertFalse(result["stage3_training_authorized"])
            self.assertEqual(set(result["status_report"]), {"source_indexed", "candidate_annotated", "parent_reviewed",
                                                               "accepted_auxiliary", "outcome_validated",
                                                               "verified_recovery_actions", "non_qualifying_corrective_actions"})
            self.assertGreater(result["status_report"]["source_indexed"], 0)
            self.assertEqual(result["status_report"]["non_qualifying_corrective_actions"], 1)
            self.assertFalse((package / "views.jsonl").stat().st_mode & 0o222)

    def test_explicit_dataset_views_project_no_private_inputs_and_training_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            package, seal = self._package(Path(folder))
            goal = DATASET.MemLiteEventDataset(package, "goal_satisfaction_counterfactual", expected_release_seal_sha256=seal)
            attempt = DATASET.MemLiteEventDataset(package, "attempt_outcome", expected_release_seal_sha256=seal)
            action = DATASET.MemLiteEventDataset(package, "corrective_action", expected_release_seal_sha256=seal)
            self.assertEqual(len(goal), len(attempt), 1)
            self.assertEqual(goal[0]["low_action_supervision_mask"], False)
            self.assertEqual(attempt[0]["low_action_supervision_mask"], False)
            self.assertEqual(goal[0]["actor_evidence"]["kind"], "MISSING")
            self.assertNotIn("privileged_evidence", goal[0])
            self.assertEqual(len(action[0]["raw_actions_23"][0]), 23)
            self.assertEqual(len(action[0]["actions_27"][0]), 27)
            self.assertEqual(action[0]["action_dim_is_pad"], [i in (7, 8, 17, 18) for i in range(27)])
            self.assertEqual(action[0]["action_is_pad"], [False, False] + [True] * 30)
            with self.assertRaisesRegex(ValueError, "not a stage3 training authorization"):
                DATASET.MemLiteEventDataset(package, "corrective_action", expected_release_seal_sha256=seal,
                                             for_training=True)

    def test_audit_and_dataset_accept_only_externally_sealed_parent_approved_action_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            package, seal, publisher_root = self._quality_action_package(Path(folder))
            result = audit.audit(package, expected_release_seal_sha256=seal,
                                 corrective_publisher_root=publisher_root)
            self.assertEqual(result["status_report"]["verified_recovery_actions"], 1)
            action = DATASET.MemLiteEventDataset(package, "corrective_action", expected_release_seal_sha256=seal,
                                                  corrective_publisher_root=publisher_root)
            self.assertTrue(action[0]["dataset_quality_gates"]["corrective_fm"])
            with self.assertRaisesRegex(ValueError, "externally recorded release seal"):
                audit.audit(package, expected_release_seal_sha256="0" * 64,
                            corrective_publisher_root=publisher_root)
            with self.assertRaisesRegex(ValueError, "externally recorded release seal"):
                DATASET.MemLiteEventDataset(package, "corrective_action", expected_release_seal_sha256="0" * 64,
                                             corrective_publisher_root=publisher_root)


if __name__ == "__main__":
    unittest.main()
