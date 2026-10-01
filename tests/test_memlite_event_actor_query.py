"""Focused tests for the additive P107 actor-query data-prep contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACTOR = load_module("p107_actor_query_test", REPO / "src/g05/data/memlite_event_actor_query.py")
DATASET = load_module("p107_actor_query_dataset_test", REPO / "src/g05/data/memlite_event_dataset.py")

sys.path.insert(0, str(REPO / "tests"))
import test_audit_memlite_event_release as release_fixture  # noqa: E402


def _goal_query(text: str, *, source_event_id: str = "e" * 64, source_group_id: str = "f" * 64,
                source_skill_id: int = 7, canonical_verb: str = "NAVIGATE",
                query_id: str = "query-id-1") -> tuple[dict, dict[str, dict]]:
    digest = ACTOR.query_content_sha256("goal_satisfaction_counterfactual", text)
    registry_row = {
        "schema_version": ACTOR.PRELABEL_QUERY_SCHEMA,
        "prelabel_query_id": query_id, "query_text": text,
        "event_id": source_event_id, "source_group_id": source_group_id,
        "skill_id": source_skill_id, "canonical_verb": canonical_verb,
        "relation_family": "goal_relation",
        "observation_frame": 0, "query_ordinal_within_event": 0,
        "source_pin": {
            "event_id": source_event_id,
            "phase_index_path": "phase-index.jsonl",
            "phase_index_record_sha256": "a" * 64,
            "phase_index_sha256": "b" * 64,
            "queue_path": "queue.jsonl", "queue_record_sha256": "c" * 64,
            "queue_sha256": "d" * 64,
            "selection_manifest_path": "selection.json",
            "selection_manifest_sha256": "e" * 64,
        },
    }
    query = {
        "kind": "goal_satisfaction_counterfactual",
        "prelabel_query_id": query_id,
        "query_content_sha256": digest,
        "text": text,
    }
    return query, {query_id: registry_row}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


def _install_sidecar(package: Path, view_id: str, query: dict, registry_row: dict) -> str:
    """Add the proposed sealed files to a synthetic package in a test tempdir."""
    package.chmod(0o755)
    for path in package.iterdir():
        path.chmod(0o644)
    sidecar_path = package / "actor_queries.jsonl"
    registry_path = package / "prelabel_queries.jsonl"
    sidecar_path.write_bytes(_canonical_bytes({
        "schema_version": ACTOR.ACTOR_QUERY_SCHEMA, "view_id": view_id, "actor_query": query,
    }))
    registry_path.write_bytes(_canonical_bytes(registry_row))
    manifest_path = package / "release_manifest.json"
    seal_path = package / "release_seal.json"
    manifest = json.loads(manifest_path.read_bytes())
    files = dict(manifest["files"])
    files[sidecar_path.name] = {"bytes": sidecar_path.stat().st_size, "rows": 1,
                                "sha256": _file_sha(sidecar_path)}
    files[registry_path.name] = {"bytes": registry_path.stat().st_size, "rows": 1,
                                 "sha256": _file_sha(registry_path)}
    manifest["files"] = files
    manifest[ACTOR.ACTOR_QUERY_MANIFEST_KEY] = {
        "schema_version": ACTOR.ACTOR_QUERY_SCHEMA, "path": sidecar_path.name,
        "sha256": _file_sha(sidecar_path), "row_count": 1,
        "prelabel_query_registry": {
            "schema_version": ACTOR.PRELABEL_QUERY_SCHEMA,
            "adapter_schema_version": ACTOR.PRELABEL_ADAPTER_SCHEMA,
            "path": registry_path.name,
            "sha256": _file_sha(registry_path), "row_count": 1,
        },
    }
    manifest_path.write_bytes(_canonical_bytes(manifest))
    seal = json.loads(seal_path.read_bytes())
    seal["files"] = files
    seal["release_manifest_sha256"] = _file_sha(manifest_path)
    seal_path.write_bytes(_canonical_bytes(seal))
    for path in package.iterdir():
        path.chmod(0o444)
    package.chmod(0o555)
    return _file_sha(seal_path)


class ActorQueryContractTests(unittest.TestCase):
    def test_standalone_python_s_import_has_no_heavy_dependency(self):
        source = (
            "import importlib.util; "
            f"s=importlib.util.spec_from_file_location('aq', {str(REPO / 'src/g05/data/memlite_event_actor_query.py')!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.ACTOR_QUERY_SCHEMA)"
        )
        result = subprocess.run([sys.executable, "-S", "-c", source], check=True,
                                capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), ACTOR.ACTOR_QUERY_SCHEMA)

    def test_goal_query_is_prelabel_bound_and_completed_is_valid_text(self):
        query, registry = _goal_query("Ask whether the just-completed skill reached the desired goal.")
        projected = ACTOR.actor_query_projection(
            query, observation_frame=12, view_kind="goal_satisfaction_counterfactual",
            prelabel_registry=registry)
        self.assertEqual(projected, query)
        bad = dict(query)
        bad["goal_satisfaction"] = "SATISFIED"
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "invalid fields"):
            ACTOR.validate_actor_query(bad, observation_frame=12,
                                       view_kind="goal_satisfaction_counterfactual",
                                       prelabel_registry=registry)
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "approved variant"):
            ACTOR.validate_actor_query(
                dict(query, kind="unsupported_query"), observation_frame=12,
                view_kind="goal_satisfaction_counterfactual", prelabel_registry=registry)

    def test_attempt_and_runtime_queries_obey_anchor_and_binding(self):
        attempt = {
            "kind": "attempt_outcome_intent", "attempt_id": "attempt-actual",
            "issued_frame": 9, "query_content_sha256": ACTOR.query_content_sha256(
                "attempt_outcome_intent", "Evaluate the issued attempt."),
            "text": "Evaluate the issued attempt.",
        }
        self.assertEqual(ACTOR.validate_actor_query(
            attempt, observation_frame=10, view_kind="attempt_outcome",
            view={"attempt_id": "attempt-actual"})["attempt_id"], "attempt-actual")
        future = dict(attempt, issued_frame=11)
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "after the observation"):
            ACTOR.validate_actor_query(future, observation_frame=10,
                                       view_kind="attempt_outcome",
                                       view={"attempt_id": "attempt-actual"})
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "does not bind"):
            ACTOR.validate_actor_query(attempt, observation_frame=10,
                                       view_kind="attempt_outcome",
                                       view={"attempt_id": "different"})

        runtime_text = "Continue the issued high-level intent."
        runtime = {
            "kind": "runtime_high_level_intent", "intent_id": "bundle-executed",
            "issued_frame": 10, "query_content_sha256": ACTOR.query_content_sha256(
                "runtime_high_level_intent", runtime_text), "text": runtime_text,
        }
        corrective = {"low_action_supervision_mask": True,
                      "action_intent_bundle_id": "bundle-executed"}
        self.assertEqual(ACTOR.validate_actor_query(
            runtime, observation_frame=10, view_kind="corrective_action", view=corrective)["intent_id"],
                         "bundle-executed")
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "runtime intent bundle"):
            ACTOR.validate_actor_query(
                dict(runtime, intent_id="wrong-bundle"), observation_frame=10,
                view_kind="corrective_action", view=corrective)
        goal_query, goal_registry = _goal_query("Counterfactually ask for the desired goal.")
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "non-goal view"):
            ACTOR.validate_actor_query(
                goal_query, observation_frame=10, view_kind="corrective_action",
                view=corrective, prelabel_registry=goal_registry)

    def test_same_rgb_different_queries_project_differently(self):
        first, first_registry = _goal_query("Place the cup on the tray.")
        second, second_registry = _goal_query("Place the cup beside the tray.", query_id="query-id-2")
        first_out = ACTOR.actor_query_projection(
            first, observation_frame=20, view_kind="goal_satisfaction_counterfactual",
            prelabel_registry=first_registry)
        second_out = ACTOR.actor_query_projection(
            second, observation_frame=20, view_kind="goal_satisfaction_counterfactual",
            prelabel_registry=second_registry)
        self.assertNotEqual(first_out, second_out)
        self.assertNotEqual(first_out["prelabel_query_id"], second_out["prelabel_query_id"])
        self.assertNotIn("goal_satisfaction", first_out)
        self.assertNotIn("evidence", first_out)

    def test_same_text_on_different_source_event_or_skill_has_distinct_prelabel_id(self):
        first, _ = _goal_query("Same desired query.", source_event_id="a" * 64,
                               source_skill_id=1, query_id="opaque-query-a")
        second, _ = _goal_query("Same desired query.", source_event_id="b" * 64,
                                source_skill_id=1, query_id="opaque-query-b")
        third, _ = _goal_query("Same desired query.", source_event_id="a" * 64,
                               source_skill_id=2, query_id="opaque-query-c")
        # Producer IDs are opaque receipts: the adapter preserves them and
        # never invents a hash-derived identity from source fields.
        self.assertNotEqual(first["prelabel_query_id"], second["prelabel_query_id"])
        self.assertNotEqual(first["prelabel_query_id"], third["prelabel_query_id"])
        self.assertEqual(first["query_content_sha256"], second["query_content_sha256"])

    def test_adapter_preserves_upstream_opaque_identity_and_source_pin(self):
        query, registry = _goal_query("A desired question.", query_id="producer-opaque-id")
        adapted = ACTOR.adapt_prelabel_registry_row(next(iter(registry.values())))
        self.assertEqual(adapted["schema_version"], ACTOR.PRELABEL_QUERY_SCHEMA)
        self.assertEqual(adapted["adapter_schema_version"], ACTOR.PRELABEL_ADAPTER_SCHEMA)
        self.assertEqual(adapted["prelabel_query_id"], "producer-opaque-id")
        self.assertEqual(adapted["event_id"], adapted["source_event_id"])
        self.assertEqual(adapted["skill_id"], adapted["source_skill_id"])
        self.assertEqual(adapted["query_text"], adapted["text"])
        self.assertEqual(adapted["source_pin"]["event_id"], adapted["event_id"])
        self.assertEqual(adapted["query_content_sha256"], query["query_content_sha256"])

    def test_goal_query_binds_specific_skill_not_only_event_id(self):
        event_id, group_id = "f" * 64, "b" * 64
        query, registry = _goal_query(
            "Is the door open?", source_event_id=event_id, source_group_id=group_id,
            source_skill_id=10, canonical_verb="OPEN_DOOR", query_id="open-door")
        event = {"event_id": event_id, "source": {"source_group_id": group_id},
                 "skill_bundle": [{"skill_id": 10, "verb": "OPEN_DOOR"},
                                  {"skill_id": 2, "verb": "GRASP"}]}
        view = {"event_id": event_id, "source_group_id": group_id,
                "label_kind": "goal_satisfaction_counterfactual",
                "goal_relation": "Is the door open?"}
        self.assertEqual(ACTOR.validate_actor_query(
            query, observation_frame=0, view_kind="goal_satisfaction_counterfactual",
            view=view, event=event, prelabel_registry=registry), query)
        wrong = dict(next(iter(registry.values())), skill_id=2)
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "canonical_verb"):
            ACTOR.validate_actor_query(
                query, observation_frame=0, view_kind="goal_satisfaction_counterfactual",
                view=view, event=event, prelabel_registry={"open-door": wrong})
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "text/semantic"):
            ACTOR.validate_actor_query(
                query, observation_frame=0, view_kind="goal_satisfaction_counterfactual",
                view=dict(view, goal_relation="Is the gripper holding it?"), event=event,
                prelabel_registry=registry)

    def test_goal_query_registry_event_must_bind_postlabel_view_event(self):
        query, registry = _goal_query("Bind this desired question.", source_event_id="a" * 64)
        with self.assertRaisesRegex(ACTOR.ActorQueryError, "packaged event"):
            ACTOR.validate_actor_query(
                query, observation_frame=4, view_kind="goal_satisfaction_counterfactual",
                view={"event_id": "b" * 64}, prelabel_registry=registry)

    def test_sidecar_requires_sealed_receipts_and_rejects_duplicate_or_missing_binding(self):
        query, registry = _goal_query("Is the desired placement completed?")
        view_id = "a" * 64
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sidecar_path = root / "actor_queries.jsonl"
            registry_path = root / "prelabel_queries.jsonl"
            sidecar_path.write_bytes(_canonical_bytes({
                "schema_version": ACTOR.ACTOR_QUERY_SCHEMA,
                "view_id": view_id, "actor_query": query,
            }))
            registry_path.write_bytes(_canonical_bytes(next(iter(registry.values()))))
            files = {
                sidecar_path.name: {"bytes": sidecar_path.stat().st_size, "rows": 1,
                                     "sha256": _file_sha(sidecar_path)},
                registry_path.name: {"bytes": registry_path.stat().st_size, "rows": 1,
                                     "sha256": _file_sha(registry_path)},
            }
            metadata = {
                "schema_version": ACTOR.ACTOR_QUERY_SCHEMA,
                "path": sidecar_path.name, "sha256": _file_sha(sidecar_path), "row_count": 1,
                "prelabel_query_registry": {
                    "schema_version": ACTOR.PRELABEL_QUERY_SCHEMA,
                    "adapter_schema_version": ACTOR.PRELABEL_ADAPTER_SCHEMA,
                    "path": registry_path.name, "sha256": _file_sha(registry_path), "row_count": 1,
                },
            }
            sidecar_path.chmod(0o444)
            registry_path.chmod(0o444)
            root.chmod(0o555)
            by_view, registry_out = ACTOR.read_actor_query_sidecar(
                root, metadata, files=files, expected_view_ids={view_id})
            self.assertEqual(by_view[view_id], query)
            self.assertEqual(set(registry_out), set(registry))

            with self.assertRaisesRegex(ACTOR.ActorQueryError, "unpackaged view"):
                ACTOR.read_actor_query_sidecar(root, metadata, files=files, expected_view_ids={"b" * 64})

            root.chmod(0o755)
            sidecar_path.chmod(0o644)
            sidecar_path.write_bytes(sidecar_path.read_bytes() + b" ")
            sidecar_path.chmod(0o444)
            root.chmod(0o555)
            with self.assertRaisesRegex(ACTOR.ActorQueryError, "receipt mismatch"):
                ACTOR.read_actor_query_sidecar(root, metadata, files=files, expected_view_ids={view_id})

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sidecar_path = root / "actor_queries.jsonl"
            sidecar_path.write_bytes(b"{}\n")
            metadata = {
                "schema_version": ACTOR.ACTOR_QUERY_SCHEMA,
                "path": sidecar_path.name, "sha256": _file_sha(sidecar_path), "row_count": 1,
                "prelabel_query_registry": None,
            }
            files = {sidecar_path.name: {"bytes": sidecar_path.stat().st_size, "rows": 1,
                                         "sha256": metadata["sha256"]}}
            # A writable sidecar is never accepted, even with a matching digest.
            with self.assertRaisesRegex(ACTOR.ActorQueryError, "writable"):
                ACTOR.read_actor_query_sidecar(root, metadata, files=files)

    def test_sidecar_duplicate_binding_and_missing_registry_are_rejected(self):
        query, registry = _goal_query("Check the desired goal.")
        row = {"schema_version": ACTOR.ACTOR_QUERY_SCHEMA,
               "view_id": "c" * 64, "actor_query": query}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sidecar_path = root / "actor_queries.jsonl"
            registry_path = root / "prelabel_queries.jsonl"
            sidecar_path.write_bytes(_canonical_bytes(row) + _canonical_bytes(row))
            registry_path.write_bytes(_canonical_bytes(next(iter(registry.values()))))
            sidecar_path.chmod(0o444)
            registry_path.chmod(0o444)
            root.chmod(0o555)
            metadata = {"schema_version": ACTOR.ACTOR_QUERY_SCHEMA, "path": sidecar_path.name,
                        "sha256": _file_sha(sidecar_path), "row_count": 2,
                        "prelabel_query_registry": None}
            files = {sidecar_path.name: {"bytes": sidecar_path.stat().st_size, "rows": 2,
                                         "sha256": metadata["sha256"]}}
            with self.assertRaisesRegex(ACTOR.ActorQueryError, "prelabel registry"):
                ACTOR.read_actor_query_sidecar(root, metadata, files=files)

            # Registering the registry but duplicating the view binding is a
            # distinct failure and must not silently select the first query.
            metadata["prelabel_query_registry"] = {
                "schema_version": ACTOR.PRELABEL_QUERY_SCHEMA,
                "adapter_schema_version": ACTOR.PRELABEL_ADAPTER_SCHEMA,
                "path": registry_path.name,
                "sha256": _file_sha(registry_path), "row_count": 1,
            }
            files[registry_path.name] = {
                "bytes": registry_path.stat().st_size, "rows": 1,
                "sha256": metadata["prelabel_query_registry"]["sha256"],
            }
            with self.assertRaisesRegex(ACTOR.ActorQueryError, "duplicate actor-query"):
                ACTOR.read_actor_query_sidecar(root, metadata, files=files)


class EventDatasetActorQueryTests(unittest.TestCase):
    def test_old_package_row_is_unchanged_and_projection_is_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            package, seal = release_fixture.AuditMemLiteEventReleaseTests()._package(Path(folder))
            dataset = DATASET.MemLiteEventDataset(
                package, "goal_satisfaction_counterfactual", expected_release_seal_sha256=seal)
            self.assertNotIn("actor_query", dataset[0])
            with self.assertRaisesRegex(ValueError, "not a stage3 training authorization"):
                DATASET.MemLiteEventDataset(
                    package, "goal_satisfaction_counterfactual", expected_release_seal_sha256=seal,
                    for_training=True)

            first, first_registry = _goal_query("Place the radio on the left.")
            second, second_registry = _goal_query("Place the radio on the right.", query_id="query-right")
            first_out = ACTOR.actor_query_projection(
                first, observation_frame=20, view_kind="goal_satisfaction_counterfactual",
                prelabel_registry=first_registry)
            second_out = ACTOR.actor_query_projection(
                second, observation_frame=20, view_kind="goal_satisfaction_counterfactual",
                prelabel_registry=second_registry)
            self.assertNotEqual(first_out, second_out)
            self.assertNotIn("goal_satisfaction", first_out)

    def test_registered_sealed_sidecar_is_loaded_without_goal_relation_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            package, _ = release_fixture.AuditMemLiteEventReleaseTests()._package(Path(folder))
            view = next(row for row in DATASET._read_jsonl(package / "views.jsonl")
                        if row["view"]["label_kind"] == "goal_satisfaction_counterfactual")["view"]
            event = next(row for row in DATASET._read_jsonl(package / "events.jsonl")
                         if row["event_id"] == view["event_id"])
            skill = event["skill_bundle"][0]
            query, registry = _goal_query(
                view["goal_relation"], source_event_id=view["event_id"],
                source_group_id=view["source_group_id"], source_skill_id=skill["skill_idx"],
                canonical_verb=skill["verb"])
            seal = _install_sidecar(package, view["view_id"], query, next(iter(registry.values())))
            dataset = DATASET.MemLiteEventDataset(
                package, "goal_satisfaction_counterfactual", expected_release_seal_sha256=seal)
            self.assertEqual(dataset[0]["actor_query"], query)
            # The view's postlabel answer is independently UNKNOWN and is not
            # used to synthesize or replace the prelabel query text.
            self.assertEqual(dataset[0]["actor_query"]["text"], view["goal_relation"])
            self.assertNotIn("goal_satisfaction", dataset[0]["actor_query"])

    def test_dataset_validates_other_kind_sidecar_rows_before_filtering(self):
        with tempfile.TemporaryDirectory() as folder:
            package, _ = release_fixture.AuditMemLiteEventReleaseTests()._package(Path(folder))
            attempt_view = next(row for row in DATASET._read_jsonl(package / "views.jsonl")
                                if row["view"]["label_kind"] == "attempt_outcome")["view"]
            query, registry = _goal_query("A query incorrectly bound to an attempt view.")
            seal = _install_sidecar(package, attempt_view["view_id"], query,
                                    next(iter(registry.values())))
            with self.assertRaisesRegex(ValueError, "non-goal view"):
                DATASET.MemLiteEventDataset(
                    package, "goal_satisfaction_counterfactual", expected_release_seal_sha256=seal)


if __name__ == "__main__":
    unittest.main()
