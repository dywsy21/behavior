"""Release audit and view-specific dataset checks for P107 preparation data."""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
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


def load_actor_module():
    path = REPO / "src/g05/data/memlite_event_actor_query.py"
    spec = importlib.util.spec_from_file_location("p107_actor_query_audit_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACTOR = load_actor_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_valid_actor_sidecar(package: Path) -> str:
    """Install one producer-shaped, source-bound goal query in a fixture."""
    package.chmod(0o755)
    for path in package.iterdir():
        path.chmod(0o644)
    views = [json.loads(line) for line in (package / "views.jsonl").read_text().splitlines() if line.strip()]
    events = [json.loads(line) for line in (package / "events.jsonl").read_text().splitlines() if line.strip()]
    view = next(row["view"] for row in views if row["view"]["label_kind"] == "goal_satisfaction_counterfactual")
    event = next(row for row in events if row["event_id"] == view["event_id"])
    skill = event["skill_bundle"][0]
    skill_id = skill.get("skill_id", skill.get("skill_idx"))
    text = view["goal_relation"]
    query_digest = ACTOR.query_content_sha256("goal_satisfaction_counterfactual", text)
    query_id = "producer-query-opaque"
    query = {"kind": "goal_satisfaction_counterfactual", "prelabel_query_id": query_id,
             "query_content_sha256": query_digest, "text": text}
    source_group_id = event["source"]["source_group_id"]
    registry = {
        "schema_version": ACTOR.PRELABEL_QUERY_SCHEMA, "canonical_verb": skill["verb"],
        "event_id": event["event_id"], "observation_frame": event["observation"]["frame"],
        "prelabel_query_id": query_id, "query_ordinal_within_event": 0,
        "query_text": text, "relation_family": "goal_relation", "skill_id": skill_id,
        "source_group_id": source_group_id,
        "source_pin": {"event_id": event["event_id"], "phase_index_path": "phase.jsonl",
                        "phase_index_record_sha256": "a" * 64, "phase_index_sha256": "b" * 64,
                        "queue_path": "queue.jsonl", "queue_record_sha256": "c" * 64,
                        "queue_sha256": "d" * 64, "selection_manifest_path": "selection.json",
                        "selection_manifest_sha256": "e" * 64},
    }
    sidecar = package / "actor_queries.jsonl"
    registry_path = package / "prelabel_queries.jsonl"
    sidecar.write_bytes(pack.canonical_json({"schema_version": ACTOR.ACTOR_QUERY_SCHEMA,
                                             "view_id": view["view_id"], "actor_query": query}).encode() + b"\n")
    registry_path.write_bytes(pack.canonical_json(registry).encode() + b"\n")
    manifest_path, seal_path = package / "release_manifest.json", package / "release_seal.json"
    manifest, seal = json.loads(manifest_path.read_text()), json.loads(seal_path.read_text())
    files = dict(manifest["files"])
    files[sidecar.name] = {"bytes": sidecar.stat().st_size, "rows": 1, "sha256": _sha(sidecar)}
    files[registry_path.name] = {"bytes": registry_path.stat().st_size, "rows": 1, "sha256": _sha(registry_path)}
    manifest["files"] = files
    manifest[ACTOR.ACTOR_QUERY_MANIFEST_KEY] = {
        "schema_version": ACTOR.ACTOR_QUERY_SCHEMA, "path": sidecar.name,
        "sha256": _sha(sidecar), "row_count": 1,
        "prelabel_query_registry": {"schema_version": ACTOR.PRELABEL_QUERY_SCHEMA,
                                     "adapter_schema_version": ACTOR.PRELABEL_ADAPTER_SCHEMA,
                                     "path": registry_path.name, "sha256": _sha(registry_path),
                                     "row_count": 1},
    }
    manifest_path.write_bytes(pack.canonical_json(manifest).encode() + b"\n")
    seal["files"] = files
    seal["release_manifest_sha256"] = _sha(manifest_path)
    seal_path.write_bytes(pack.canonical_json(seal).encode() + b"\n")
    for path in package.iterdir():
        path.chmod(0o444)
    package.chmod(0o555)
    return _sha(seal_path)


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

    def _quality_action_package(self, root: Path) -> tuple[Path, str, Path, Path]:
        index_path, groups, events, _ = fixture.indexed(root)
        student = next(event for event in events if event["usage_role"] == "student_candidate")
        index_seal = pack.sha256_file(index_path / "inventory_seal.json")
        publisher_root = root / "parent-publisher"
        labels, authority_manifest_sha = fixture.approve_logged_corrective(
            fixture.annotations(student), student, groups, publisher_root,
            index_root=index_path, index_inventory_seal_sha256=index_seal)
        labels_path, package = root / "labels.json", root / "quality-package"
        labels_path.write_text(pack.canonical_json(labels))
        pack.publish(index_path, labels_path, package, release_role="student_candidate", minimum_scale=None,
                     request_dataset_quality_eligibility=False,
                     expected_index_inventory_seal_sha256=index_seal,
                     corrective_publisher_root=publisher_root,
                     expected_corrective_publisher_manifest_sha256=authority_manifest_sha)
        return package, pack.sha256_file(package / "release_seal.json"), publisher_root, index_path

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

    def test_audit_checks_declared_actor_sidecar_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            package, _ = self._package(Path(folder))
            seal = _install_valid_actor_sidecar(package)
            result = audit.audit(package, expected_release_seal_sha256=seal)
            self.assertEqual(result["status"], "PASS_WITH_STAGE3_TRAINING_BLOCKED")

            sidecar = package / "actor_queries.jsonl"
            package.chmod(0o755)
            sidecar.chmod(0o644)
            sidecar.write_bytes(sidecar.read_bytes() + b" ")
            sidecar.chmod(0o444)
            package.chmod(0o555)
            with self.assertRaisesRegex(ValueError, "actor-query sidecar audit failed|receipt mismatch"):
                audit.audit(package, expected_release_seal_sha256=seal)

    def test_audit_rejects_sidecar_files_without_manifest_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            package, _ = self._package(Path(folder))
            _install_valid_actor_sidecar(package)
            package.chmod(0o755)
            manifest_path, seal_path = package / "release_manifest.json", package / "release_seal.json"
            for path in package.iterdir():
                path.chmod(0o644)
            manifest = json.loads(manifest_path.read_text())
            manifest.pop(ACTOR.ACTOR_QUERY_MANIFEST_KEY)
            manifest_path.write_bytes(pack.canonical_json(manifest).encode() + b"\n")
            seal_doc = json.loads(seal_path.read_text())
            seal_doc["release_manifest_sha256"] = _sha(manifest_path)
            seal_path.write_bytes(pack.canonical_json(seal_doc).encode() + b"\n")
            for path in package.iterdir():
                path.chmod(0o444)
            package.chmod(0o555)
            with self.assertRaisesRegex(ValueError, "without an actor_query_sidecar registration"):
                audit.audit(package, expected_release_seal_sha256=_sha(seal_path))

    def test_audit_and_dataset_accept_only_externally_sealed_parent_approved_action_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            package, seal, publisher_root, index_path = self._quality_action_package(Path(folder))
            result = audit.audit(package, expected_release_seal_sha256=seal,
                                 corrective_publisher_root=publisher_root, index_root=index_path)
            self.assertEqual(result["status_report"]["verified_recovery_actions"], 1)
            self.assertEqual(result["actual_scale"], {
                "source_episodes": 0, "goal_outcome_windows": 0, "corrective_action_windows": 1,
                "decision_source_episodes": 0, "corrective_action_source_episodes": 1})
            action = DATASET.MemLiteEventDataset(package, "corrective_action", expected_release_seal_sha256=seal,
                                                  corrective_publisher_root=publisher_root, index_root=index_path)
            self.assertTrue(action[0]["dataset_quality_gates"]["corrective_fm"])
            with self.assertRaisesRegex(ValueError, "publisher and sealed index roots"):
                audit.audit(package, expected_release_seal_sha256=seal,
                            corrective_publisher_root=publisher_root)
            with self.assertRaisesRegex(ValueError, "publisher and sealed index roots"):
                DATASET.MemLiteEventDataset(package, "corrective_action", expected_release_seal_sha256=seal,
                                             corrective_publisher_root=publisher_root)
            with self.assertRaisesRegex(ValueError, "externally recorded release seal"):
                audit.audit(package, expected_release_seal_sha256="0" * 64,
                            corrective_publisher_root=publisher_root, index_root=index_path)
            with self.assertRaisesRegex(ValueError, "externally recorded release seal"):
                DATASET.MemLiteEventDataset(package, "corrective_action", expected_release_seal_sha256="0" * 64,
                                             corrective_publisher_root=publisher_root, index_root=index_path)


if __name__ == "__main__":
    unittest.main()
