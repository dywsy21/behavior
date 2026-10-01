#!/usr/bin/env python3
"""CPU-only contract tests for P107 simulator-recovery seams.

These fakes exercise receipt invariants only.  They are intentionally marked
``fake`` / ``unverified`` and must never be read as an OmniGibson readiness
result.
"""

from __future__ import annotations

from copy import deepcopy
import unittest

from g05.recovery.branching import (
    Observation,
    OmniGibsonEvaluatorActionBackend,
    PairedRecoveryCollector,
    executed_prefix_from_prediction,
)
from g05.recovery.common import RecoveryContractError, canonical_sha256
from g05.recovery.evidence import (
    FaultEvidenceProvider,
    FaultEvidenceSample,
    PhysicalEvidenceProvider,
    PredicateSample,
)
from g05.recovery.protocol_bridge import (
    actor_evidence_projection,
    load_protocol,
    project_action_23_to_27,
    validate_corrective_action_view,
    validate_transport_receipt,
)
from g05.recovery.snapshot import (
    REQUIRED_INVENTORY,
    OmniGibsonPublicSnapshotBackend,
    SnapshotAdapter,
)
from g05.recovery.runtime import RuntimeCapability


ZERO23 = [0.0] * 23
ONE23 = [1.0] * 23
TEST_RUNTIME = RuntimeCapability("cpu-fake-runtime")


class FakeSnapshotBackend:
    """Explicit fake: complete inventory does not make it a live backend."""

    backend_kind = "fake"

    def __init__(
        self,
        *,
        missing_component: str | None = None,
        runtime_capability: RuntimeCapability | None = TEST_RUNTIME,
        adapter_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.state = {"counter": 0, "objects": {name: 0 for name in REQUIRED_INVENTORY}}
        self.missing_component = missing_component
        self.runtime_capability = runtime_capability
        if adapter_id is not None:
            self.adapter_id = adapter_id
        if session_id is not None:
            self.session_id = session_id

    def dump_state(self, *, serialized: bool):
        self.last_dump_serialized = serialized
        return deepcopy(self.state)

    def load_state(self, state, *, serialized: bool) -> None:
        self.last_load_serialized = serialized
        self.state = deepcopy(state)

    def inventory_component(self, component: str):
        if component == self.missing_component:
            return None
        return {"component": component, "state": self.state["objects"][component], "counter": self.state["counter"]}

    def state_fingerprint(self, state) -> str:
        return canonical_sha256(state)


class BranchRestoreInventoryDriftBackend(FakeSnapshotBackend):
    """A fake whose branch restores leave a non-robot inventory residue.

    The 61-D fake observation remains the same, reproducing why a robot-state
    equality check cannot stand in for a full restored-snapshot equality check.
    """

    def __init__(self) -> None:
        super().__init__()
        self.restore_calls = 0

    def load_state(self, state, *, serialized: bool) -> None:
        super().load_state(state, serialized=serialized)
        self.restore_calls += 1
        # The first restore is the fault evaluation.  Both branch restores
        # retain a hidden world residue while the public 61-D clock stays 0.
        if self.restore_calls > 1:
            self.state["objects"]["world"] += 1


class FakeActionBackend:
    def __init__(self, snapshots: FakeSnapshotBackend, *, fresh: bool = True,
                 runtime_capability: RuntimeCapability | None = None) -> None:
        self.snapshots = snapshots
        self.fresh = fresh
        self.executed: list[list[float]] = []
        self.runtime_capability = snapshots.runtime_capability if runtime_capability is None else runtime_capability
        self.observation_runtime_capability = self.runtime_capability

    def observe(self) -> Observation:
        clock = self.snapshots.state["counter"]
        return Observation(clock, [float(clock)] * 61, self.fresh, f"rgb://step/{clock}")

    def step(self, action23: list[float]):
        self.executed.append(list(action23))
        self.snapshots.state["counter"] += 1
        return {"fake_step": len(self.executed)}

    def official_status(self):
        return {"success": True, "terminal": True}


class FakePredicateBackend:
    def __init__(self, sample: PredicateSample) -> None:
        self.sample_value = sample
        self.calls = []

    def sample(self, *, skill_binding, context):
        self.calls.append((dict(skill_binding), dict(context)))
        return self.sample_value


class FakeFaultBackend:
    """Return a sample bound to the exact post-fault state passed by the collector."""

    def __init__(self, *, current_intent_bundle_id: str = "intent-p107-grasp-0001") -> None:
        self.current_intent_bundle_id = current_intent_bundle_id
        self.calls = []

    def sample(self, *, skill_binding, context):
        self.calls.append((dict(skill_binding), dict(context)))
        return FaultEvidenceSample(
            family=skill_binding["family"],
            object_id=skill_binding["object_id"],
            arm=skill_binding["arm"],
            prior_intent_bundle_id=context["prior_intent_bundle_id"],
            current_intent_bundle_id=self.current_intent_bundle_id,
            fault_observed=True,
            source_binding_verified=True,
            evidence_kind="documented_object_pose_deviation",
            evidence_start_frame=context["post_fault_clock"],
            evidence_end_frame=context["post_fault_clock"],
            evidence_available_time=context["post_fault_clock"],
            snapshot_state_fingerprint=context["post_fault_snapshot_fingerprint"],
            observation_state61_sha256=context["post_fault_observation_state61_sha256"],
            detail="fake test-only deviation sample",
        )


def physical_sample(**overrides) -> PredicateSample:
    values = {
        "family": "grasp",
        "sensor_supported": True,
        "source_binding_verified": True,
        "predicate": True,
        "causal_transition": True,
        "failure_evidence": False,
        "stable_frames": 3,
        "required_stable_frames": 3,
        "evidence_kind": "documented_grasp_transition",
        "evidence_start_frame": 0,
        "evidence_end_frame": 1,
        "evidence_available_time": 1,
        "object_id": "cup",
        "arm": "right",
        "prior_intent_bundle_id": "intent-before",
        "current_intent_bundle_id": "intent-p107-grasp-0001",
        "detail": "fake evaluator-only sample",
    }
    values.update(overrides)
    return PredicateSample(**values)


def grasp_binding():
    return {"family": "grasp", "object_id": "cup", "arm": "right"}


def evidence_context(**overrides):
    values = {
        "action_start_frame": 0,
        "actual_end_frame": 1,
        "prior_intent_bundle_id": "intent-before",
        "current_intent_bundle_id": "intent-p107-grasp-0001",
    }
    values.update(overrides)
    return values


def physical_provider(backend, runtime: RuntimeCapability | None = TEST_RUNTIME) -> PhysicalEvidenceProvider:
    return PhysicalEvidenceProvider(backend, runtime_capability=runtime)


def fault_provider(backend, runtime: RuntimeCapability | None = TEST_RUNTIME) -> FaultEvidenceProvider:
    return FaultEvidenceProvider(backend, runtime_capability=runtime)


class SnapshotReceiptTest(unittest.TestCase):
    def test_fake_roundtrip_is_auditable_but_never_live_ready(self) -> None:
        backend = FakeSnapshotBackend()
        adapter = SnapshotAdapter(backend)
        actions = FakeActionBackend(backend)

        def execute(chunk):
            for action in chunk:
                actions.step(action)
            return len(chunk)

        result = adapter.roundtrip([ZERO23, ONE23], execute, label="cpu_fake")
        self.assertEqual(result.actions_first_executed, 2)
        self.assertEqual(result.actions_second_executed, 2)
        self.assertTrue(all(item.status == "match" for item in result.restore_comparison))
        self.assertTrue(all(item.status == "match" for item in result.replay_comparison))
        self.assertFalse(result.passed)
        self.assertFalse(result.public()["trainable"])
        self.assertEqual(backend.last_dump_serialized, False)
        self.assertEqual(backend.last_load_serialized, False)

    def test_missing_inventory_component_blocks_receipt(self) -> None:
        adapter = SnapshotAdapter(FakeSnapshotBackend(missing_component="particles"))
        capture = adapter.capture("missing_particles")
        self.assertFalse(capture.complete_inventory)
        self.assertEqual(capture.inventory["particles"].status, "unavailable")
        self.assertTrue(capture.runtime_bound)

    def test_public_og_adapter_does_not_invent_a_restore_api(self) -> None:
        class PublicSim:
            def dump_state(self, *, serialized):
                self.serialized = serialized
                return {"documented": "dump_state_only"}

        sim = PublicSim()
        adapter = SnapshotAdapter(OmniGibsonPublicSnapshotBackend(sim))
        capture = adapter.capture("unverified_og")
        self.assertEqual(capture.backend_kind, "unverified")
        self.assertFalse(capture.complete_inventory)
        self.assertEqual(sim.serialized, False)
        with self.assertRaisesRegex(RecoveryContractError, "restore API"):
            adapter.restore(capture)

    def test_no_attestation_constructor_or_live_shortcut_exists(self) -> None:
        class PublicSim:
            def dump_state(self, *, serialized):
                return {"documented": "dump_state_only"}

        with self.assertRaises(TypeError):
            OmniGibsonPublicSnapshotBackend(PublicSim(), readiness_attestation=object())

    def test_relabelled_same_body_fake_cannot_present_as_live(self) -> None:
        backend = FakeSnapshotBackend()
        # A caller-controlled string is rejected; raw collection has no live
        # shortcut at all.
        backend.backend_kind = "live"
        with self.assertRaisesRegex(RecoveryContractError, "fake or unverified"):
            SnapshotAdapter(backend)

    def test_cross_adapter_snapshot_restore_is_rejected(self) -> None:
        first = SnapshotAdapter(FakeSnapshotBackend(adapter_id="same", session_id="same"))
        second = SnapshotAdapter(FakeSnapshotBackend(adapter_id="same", session_id="same"))
        capture = first.capture("first")
        with self.assertRaisesRegex(RecoveryContractError, "Cross-adapter"):
            second.restore(capture)


class EvidenceTest(unittest.TestCase):
    def test_success_needs_transition_and_stable_frames(self) -> None:
        provider = PhysicalEvidenceProvider(FakePredicateBackend(physical_sample(stable_frames=2)))
        result = provider.evaluate(
            skill_binding=grasp_binding(), context=evidence_context(), official_task_success=True, official_terminal=True
        )
        self.assertEqual(result.outcome, "IN_PROGRESS")
        self.assertTrue(result.valid_result_mask)
        self.assertTrue(result.official_task_success)
        self.assertTrue(result.official_terminal)

    def test_unknown_never_becomes_timeout_failure(self) -> None:
        provider = PhysicalEvidenceProvider(FakePredicateBackend(physical_sample(sensor_supported=False)))
        result = provider.evaluate(
            skill_binding=grasp_binding(), context=evidence_context(), official_task_success=False, official_terminal=True
        )
        self.assertEqual(result.outcome, "UNKNOWN")
        self.assertFalse(result.valid_result_mask)

    def test_affirmative_physical_failure_is_retained(self) -> None:
        provider = PhysicalEvidenceProvider(FakePredicateBackend(physical_sample(predicate=False, failure_evidence=True)))
        result = provider.evaluate(skill_binding=grasp_binding(), context=evidence_context())
        self.assertEqual(result.outcome, "FAILED")
        self.assertTrue(result.valid_result_mask)

    def test_future_physical_evidence_cannot_certify_current_branch(self) -> None:
        provider = PhysicalEvidenceProvider(FakePredicateBackend(physical_sample(
            evidence_start_frame=1,
            evidence_end_frame=999,
            evidence_available_time=999,
        )))
        result = provider.evaluate(skill_binding=grasp_binding(), context=evidence_context())
        self.assertEqual(result.outcome, "UNKNOWN")
        self.assertFalse(result.valid_result_mask)

    def test_mismatched_object_arm_or_intent_cannot_certify_branch(self) -> None:
        provider = PhysicalEvidenceProvider(FakePredicateBackend(physical_sample(
            object_id="different-cup", current_intent_bundle_id="another-intent"
        )))
        result = provider.evaluate(skill_binding=grasp_binding(), context=evidence_context())
        self.assertEqual(result.outcome, "UNKNOWN")
        self.assertFalse(result.valid_result_mask)

    def test_timeout_and_annotation_evidence_kinds_are_rejected(self) -> None:
        with self.assertRaisesRegex(RecoveryContractError, "closed physical taxonomy"):
            physical_sample(evidence_kind="TIMEOUT")
        with self.assertRaisesRegex(RecoveryContractError, "closed physical taxonomy"):
            FaultEvidenceSample(
                family="grasp",
                object_id="cup",
                arm="right",
                prior_intent_bundle_id="intent-before",
                current_intent_bundle_id="intent-p107-grasp-0001",
                fault_observed=True,
                source_binding_verified=True,
                evidence_kind="annotation_boundary",
                evidence_start_frame=0,
                evidence_end_frame=0,
                evidence_available_time=0,
                snapshot_state_fingerprint="a" * 64,
                observation_state61_sha256="b" * 64,
            )


class BranchCollectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_backend = FakeSnapshotBackend()
        self.snapshots = SnapshotAdapter(self.raw_backend)
        self.actions = FakeActionBackend(self.raw_backend)
        self.evidence_backend = FakePredicateBackend(physical_sample())
        self.collector = PairedRecoveryCollector(
            self.snapshots, self.actions, physical_provider(self.evidence_backend)
        )
        self.source = {"original_split": "train", "source_group_id": "train-task0-instance7"}

    def test_branches_restore_same_post_fault_state_and_export_only_actual_actions(self) -> None:
        post_fault = self.snapshots.capture("post_fault")
        receipt = self.collector.collect(
            source_ref=self.source,
            source_group_id="train-task0-instance7",
            event_ref={"event": "p107"},
            post_fault_snapshot=post_fault,
            no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23],
            skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001",
            prior_intent_bundle_id="intent-before",
            branch_seed=17,
            actor_evidence={"rgb_ref": "rgb://step/0"},
        )
        self.assertTrue(receipt.no_intervention_same_state)
        self.assertTrue(receipt.corrective_same_state)
        self.assertTrue(receipt.no_intervention_observation_matches)
        self.assertTrue(receipt.corrective_observation_matches)
        self.assertEqual(self.actions.executed, [ZERO23, ONE23])
        self.assertEqual(receipt.corrective.actual_executed_length, 1)
        self.assertEqual(len(receipt.corrective.final_observation.state61), 61)
        self.assertEqual(receipt.corrective.evidence.outcome, "SUCCEEDED")
        self.assertFalse(receipt.recovery_eligible)
        self.assertEqual(receipt.public()["recovery_kind"], "normal_or_unproven")
        self.assertFalse(receipt.corrective_positive_action_mask)  # CPU fake cannot produce positives.
        public = receipt.public()
        self.assertNotIn("step_receipts", public["privileged_evidence"]["corrective"])
        self.assertFalse(public["ready_for_training"])
        self.assertEqual(self.evidence_backend.calls[1][1]["actual_executed_length"], 1)
        execution = public["privileged_evidence"]["corrective"]["execution"]
        self.assertEqual(execution["raw_action_dim"], 23)
        self.assertEqual(execution["actual_end_frame"], execution["action_start_frame"] + 1)
        self.assertEqual(execution["executed_intent_bundle_id"], "intent-p107-grasp-0001")
        self.assertTrue(public["candidate_only"])

    def test_affirmative_prebranch_fault_allows_recovery_even_if_baseline_recovers(self) -> None:
        # Both branches have successful physical evidence.  The independent,
        # state-bound fault receipt—not a baseline failure—is what makes this a
        # genuine recovery candidate.  The fake backend still cannot emit a
        # trainable positive action label.
        self.collector.fault_evidence = fault_provider(FakeFaultBackend())
        receipt = self.collector.collect(
            source_ref=self.source,
            source_group_id="train-task0-instance7",
            event_ref={"event": "p107"},
            post_fault_snapshot=self.snapshots.capture("post_fault"),
            no_intervention_actions23=[ZERO23],
            # Identical branch actions are not a reason to erase a physics-
            # proven recovery candidate.  Comparative benefit remains unset.
            corrective_actions23=[ZERO23],
            skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001",
            prior_intent_bundle_id="intent-before",
            branch_seed=18,
            actor_evidence={"rgb_ref": "rgb://step/0"},
        )
        self.assertEqual(receipt.no_intervention.evidence.outcome, "SUCCEEDED")
        self.assertEqual(receipt.corrective.evidence.outcome, "SUCCEEDED")
        self.assertEqual(receipt.no_intervention.action_sha256, receipt.corrective.action_sha256)
        self.assertTrue(receipt.recovery_eligible)
        self.assertEqual(receipt.public()["recovery_kind"], "genuine_recovery")
        self.assertEqual(receipt.public()["comparative_benefit"], "NOT_ESTABLISHED")
        self.assertFalse(receipt.corrective_positive_action_mask)

    def test_failed_or_unknown_correction_is_never_genuine_after_valid_deviation(self) -> None:
        for name, sample, expected in (
            ("failed", physical_sample(predicate=False, failure_evidence=True), "FAILED"),
            ("unknown", physical_sample(sensor_supported=False), "UNKNOWN"),
        ):
            with self.subTest(name=name):
                raw = FakeSnapshotBackend()
                snapshots = SnapshotAdapter(raw)
                collector = PairedRecoveryCollector(
                    snapshots,
                    FakeActionBackend(raw),
                    physical_provider(FakePredicateBackend(sample)),
                    fault_provider(FakeFaultBackend()),
                )
                receipt = collector.collect(
                    source_ref=self.source,
                    source_group_id="train-task0-instance7",
                    event_ref={"event": f"p107-{name}"},
                    post_fault_snapshot=snapshots.capture("post_fault"),
                    no_intervention_actions23=[ZERO23],
                    corrective_actions23=[ONE23],
                    skill_binding=grasp_binding(),
                    intent_bundle_id="intent-p107-grasp-0001",
                    prior_intent_bundle_id="intent-before",
                    branch_seed=180,
                    actor_evidence={"rgb_ref": "rgb://step/0"},
                )
                self.assertTrue(receipt.initial_deviation_proven)
                self.assertEqual(receipt.corrective.evidence.outcome, expected)
                self.assertFalse(receipt.corrective_postcondition_proven)
                self.assertFalse(receipt.recovery_eligible)
                self.assertEqual(receipt.public()["recovery_kind"], "normal_or_unproven")

    def test_snapshot_inventory_mismatch_blocks_recovery_even_when_61d_matches(self) -> None:
        raw = BranchRestoreInventoryDriftBackend()
        snapshots = SnapshotAdapter(raw)
        collector = PairedRecoveryCollector(
            snapshots,
            FakeActionBackend(raw),
            physical_provider(FakePredicateBackend(physical_sample())),
            fault_provider(FakeFaultBackend()),
        )
        receipt = collector.collect(
            source_ref=self.source,
            source_group_id="train-task0-instance7",
            event_ref={"event": "p107-inventory-drift"},
            post_fault_snapshot=snapshots.capture("post_fault"),
            no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23],
            skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001",
            prior_intent_bundle_id="intent-before",
            branch_seed=181,
            actor_evidence={"rgb_ref": "rgb://step/0"},
        )
        self.assertTrue(receipt.no_intervention_observation_matches)
        self.assertTrue(receipt.corrective_observation_matches)
        self.assertFalse(receipt.no_intervention_same_state)
        self.assertFalse(receipt.corrective_same_state)
        self.assertFalse(receipt.recovery_eligible)

    def test_stale_corrective_final_observation_blocks_genuine_recovery(self) -> None:
        class StaleFinalObservationAction(FakeActionBackend):
            def observe(self):
                clock = self.snapshots.state["counter"]
                # Each branch starts freshly at 0, but its post-action
                # observation is stale.  The physical provider alone is not
                # enough to certify a usable corrective postcondition.
                return Observation(clock, [float(clock)] * 61, clock == 0, f"rgb://step/{clock}")

        collector = PairedRecoveryCollector(
            self.snapshots,
            StaleFinalObservationAction(self.raw_backend),
            physical_provider(FakePredicateBackend(physical_sample())),
            fault_provider(FakeFaultBackend()),
        )
        receipt = collector.collect(
            source_ref=self.source,
            source_group_id="train-task0-instance7",
            event_ref={"event": "p107-stale-final"},
            post_fault_snapshot=self.snapshots.capture("post_fault"),
            no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23],
            skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001",
            prior_intent_bundle_id="intent-before",
            branch_seed=182,
            actor_evidence={"rgb_ref": "rgb://step/0"},
        )
        self.assertEqual(receipt.corrective.evidence.outcome, "SUCCEEDED")
        self.assertFalse(receipt.corrective.observations_fresh)
        self.assertFalse(receipt.corrective_postcondition_proven)
        self.assertFalse(receipt.recovery_eligible)

    def test_fault_must_bind_the_current_corrective_intent(self) -> None:
        self.collector.fault_evidence = fault_provider(
            FakeFaultBackend(current_intent_bundle_id="wrong-recovery-intent")
        )
        receipt = self.collector.collect(
            source_ref=self.source,
            source_group_id="train-task0-instance7",
            event_ref={"event": "p107"},
            post_fault_snapshot=self.snapshots.capture("post_fault"),
            no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23],
            skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001",
            prior_intent_bundle_id="intent-before",
            branch_seed=19,
            actor_evidence={"rgb_ref": "rgb://step/0"},
        )
        self.assertFalse(receipt.recovery_eligible)
        self.assertFalse(receipt.post_fault_evidence.valid_fault_mask)

    def test_fault_is_observed_only_after_restoring_post_fault_snapshot(self) -> None:
        fault_backend = FakeFaultBackend()
        self.collector.fault_evidence = fault_provider(fault_backend)
        post_fault = self.snapshots.capture("post_fault")
        # Simulate an unrelated current evaluator state before collection.
        self.raw_backend.state["counter"] = 7
        receipt = self.collector.collect(
            source_ref=self.source,
            source_group_id="train-task0-instance7",
            event_ref={"event": "p107"},
            post_fault_snapshot=post_fault,
            no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23],
            skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001",
            prior_intent_bundle_id="intent-before",
            branch_seed=20,
            actor_evidence={"rgb_ref": "rgb://step/0"},
        )
        self.assertEqual(fault_backend.calls[0][1]["post_fault_clock"], 0)
        self.assertEqual(receipt.post_fault_observation.policy_clock, 0)
        self.assertTrue(receipt.no_intervention_observation_matches)
        self.assertTrue(receipt.corrective_observation_matches)

    def test_mixed_runtime_action_and_snapshot_are_rejected(self) -> None:
        unrelated_runtime = RuntimeCapability("other-evaluator")
        collector = PairedRecoveryCollector(
            self.snapshots,
            FakeActionBackend(self.raw_backend, runtime_capability=unrelated_runtime),
            physical_provider(FakePredicateBackend(physical_sample()), unrelated_runtime),
        )
        with self.assertRaisesRegex(RecoveryContractError, "share one runtime capability"):
            collector.collect(
                source_ref=self.source,
                source_group_id="train-task0-instance7",
                event_ref={"event": "p107"},
                post_fault_snapshot=self.snapshots.capture("post_fault"),
                no_intervention_actions23=[ZERO23],
                corrective_actions23=[ONE23],
                skill_binding=grasp_binding(),
                intent_bundle_id="intent-p107-grasp-0001",
                prior_intent_bundle_id="intent-before",
                branch_seed=21,
                actor_evidence={"rgb_ref": "rgb://step/0"},
            )

    def test_restored_branch_observation_mismatch_is_rejected(self) -> None:
        class DriftedObservationAction(FakeActionBackend):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.observe_count = 0

            def observe(self):
                self.observe_count += 1
                observation = super().observe()
                if self.observe_count == 1:
                    return observation
                shifted = observation.policy_clock + 1
                return Observation(shifted, [float(shifted)] * 61, True, observation.public_observation_ref)

        collector = PairedRecoveryCollector(
            self.snapshots,
            DriftedObservationAction(self.raw_backend),
            physical_provider(FakePredicateBackend(physical_sample())),
        )
        with self.assertRaisesRegex(RecoveryContractError, "initial observation clock/state digest"):
            collector.collect(
                source_ref=self.source,
                source_group_id="train-task0-instance7",
                event_ref={"event": "p107"},
                post_fault_snapshot=self.snapshots.capture("post_fault"),
                no_intervention_actions23=[ZERO23],
                corrective_actions23=[ONE23],
                skill_binding=grasp_binding(),
                intent_bundle_id="intent-p107-grasp-0001",
                prior_intent_bundle_id="intent-before",
                branch_seed=22,
                actor_evidence={"rgb_ref": "rgb://step/0"},
            )

    def test_bad_split_and_non_allowlisted_actor_inputs_are_rejected(self) -> None:
        post_fault = self.snapshots.capture("post_fault")
        args = dict(
            source_group_id="train-task0-instance7", event_ref={"event": "p107"}, post_fault_snapshot=post_fault,
            no_intervention_actions23=[ZERO23], corrective_actions23=[ONE23], skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001", prior_intent_bundle_id="intent-before",
            branch_seed=1, actor_evidence={"rgb_ref": "ok"},
        )
        with self.assertRaisesRegex(RecoveryContractError, "TRAIN"):
            self.collector.collect(source_ref={"original_split": "public_test"}, **args)
        for leaking_actor_input in (
            {"world_state": {"objects": ["cup"]}},
            {"physics_state": {"contact": True}},
            {"objectPose": [0, 0, 0]},
            {"ground_truth": "SUCCEEDED"},
            {"rgb_ref": {"nested": "not-a-reference"}},
            {"rgb_ref": "rgb://step/0", "unknown_nested": {"anything": True}},
        ):
            with self.subTest(leaking_actor_input=leaking_actor_input):
                with self.assertRaisesRegex(RecoveryContractError, "closed raw RGB|nonempty text"):
                    self.collector.collect(
                        source_ref=self.source,
                        **{**args, "actor_evidence": leaking_actor_input},
                    )

    def test_receipt_revalidates_actor_evidence_before_public_projection(self) -> None:
        receipt = self.collector.collect(
            source_ref=self.source,
            source_group_id="train-task0-instance7",
            event_ref={"event": "p107-actor-revalidation"},
            post_fault_snapshot=self.snapshots.capture("post_fault"),
            no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23],
            skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001",
            prior_intent_bundle_id="intent-before",
            branch_seed=183,
            actor_evidence={"rgb_ref": "rgb://step/0"},
        )
        # BranchPairReceipt remains mutable for diagnostic construction, so
        # public projection must not trust a field after construction either.
        receipt.actor_evidence = {"ground_truth": "SUCCEEDED"}
        with self.assertRaisesRegex(RecoveryContractError, "closed raw RGB"):
            receipt.public()

    def test_execution_tail_is_never_admitted(self) -> None:
        prediction = [[float(step)] * 23 for step in range(32)]
        self.assertEqual(executed_prefix_from_prediction(prediction, execution_start=0, executed_length=16), prediction[:16])
        with self.assertRaisesRegex(RecoveryContractError, "1:16"):
            executed_prefix_from_prediction(prediction, execution_start=0, executed_length=17)
        with self.assertRaisesRegex(RecoveryContractError, "index 0"):
            executed_prefix_from_prediction(prediction, execution_start=6, executed_length=16)
        with self.assertRaisesRegex(RecoveryContractError, "1:16"):
            self.collector.collect(
                source_ref=self.source, source_group_id="train-task0-instance7", event_ref={"event": "p107"},
                post_fault_snapshot=self.snapshots.capture("post_fault"), no_intervention_actions23=[ZERO23],
                corrective_actions23=prediction[:17], skill_binding=grasp_binding(),
                intent_bundle_id="intent-p107-grasp-0001", branch_seed=1, actor_evidence={"rgb_ref": "ok"},
                prior_intent_bundle_id="intent-before",
            )

    def test_failed_correction_has_no_positive_action_mask(self) -> None:
        self.collector.evidence = physical_provider(
            FakePredicateBackend(physical_sample(predicate=False, failure_evidence=True))
        )
        receipt = self.collector.collect(
            source_ref=self.source, source_group_id="train-task0-instance7", event_ref={"event": "p107"},
            post_fault_snapshot=self.snapshots.capture("post_fault"), no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23], skill_binding=grasp_binding(), branch_seed=2,
            intent_bundle_id="intent-p107-grasp-0001", actor_evidence={"rgb_ref": "rgb://step/0"},
            prior_intent_bundle_id="intent-before",
        )
        self.assertEqual(receipt.corrective.evidence.outcome, "FAILED")
        self.assertFalse(receipt.corrective_positive_action_mask)


class PublicActionAdapterTest(unittest.TestCase):
    def test_public_step_is_one_call_and_refresh_is_explicit(self) -> None:
        class Env:
            def __init__(self):
                self.calls = []

            def step(self, action, *, n_render_iterations):
                self.calls.append((action, n_render_iterations))
                return ("obs", 0.0, False, False, {"private": "not exported"})

        env = Env()
        refreshed = []
        backend = OmniGibsonEvaluatorActionBackend(
            env,
            observe_current=lambda: Observation(0, [0.0] * 61, True),
            official_status_current=lambda: {"success": False, "terminal": False},
            encode_action=lambda action: tuple(action),
            refresh_after_step=lambda result: refreshed.append(result),
        )
        self.assertEqual(backend.step(ONE23), {"env_step_returned": True})
        self.assertEqual(env.calls, [(tuple(ONE23), 1)])
        self.assertEqual(len(refreshed), 1)


class ProtocolBridgeTest(unittest.TestCase):
    @staticmethod
    def _canonical_source_and_event():
        protocol = load_protocol()
        source = {
            "source_release_manifest_sha256": "a" * 64,
            "source_annotation_sha256": "b" * 64,
            "task_index": 2,
            "task_instance_id": 11,
            "raw_episode_id": 2100,
            "episode_index": 201,
            "original_split": "train",
            "episode_length": 100,
        }
        source["source_group_id"] = protocol.source_group_id(source)
        event = {
            "schema_version": protocol.SCHEMA_VERSION,
            "record_kind": "event_candidate",
            "event_id": "",
            "source": source,
            "event_kind": "ANNOTATED_SKILL_SEGMENT",
            "event_interval": {"start_frame": 0, "end_frame": 10},
            "observation": {"frame": 0, "timestamp_s": 0.0},
            "action": {
                "start_frame": 0,
                "actual_executed_length": None,
                "raw_action_dim": 23,
                "model_action_dim": 27,
                "model_padding_indices": [7, 8, 17, 18],
            },
            "bundle_id": "intent-p107-grasp-0001",
            "skill_bundle": [{"verb": "GRASP"}],
            "parallel_bundle": False,
            "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
            "video_locators": [],
        }
        event["event_id"] = protocol.event_id(event)
        return source, event

    def test_direct_load_uses_the_stdlib_contract_without_g05_data_import(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol.SCHEMA_VERSION, "memlite-event-recovery-v1")
        projection = project_action_23_to_27([ZERO23], 1)
        self.assertEqual(len(projection["actions_27"]), 32)
        self.assertEqual(projection["action_is_pad"], [False] + [True] * 31)
        view = {
            "observation_frame": 0,
            "actor_evidence": {
                "kind": "MISSING", "evidence_end_frame": None, "available_frame": None, "references": []
            },
            "privileged_evidence": {"object_pose": "must not escape"},
        }
        self.assertEqual(actor_evidence_projection(view)["references"], [])

    def test_obsolete_bridge_never_accepts_caller_proposed_positive_mask(self) -> None:
        with self.assertRaisesRegex(RecoveryContractError, "no corrective-action publication authority"):
            validate_corrective_action_view({
                "label_kind": "corrective_action",
                "low_action_supervision_mask": True,
                "recovery_verified": "PROPOSED",
            })

    def test_identity_validation_delegates_to_data_protocol(self) -> None:
        class Protocol:
            @staticmethod
            def validate_source_ref(value):
                if value["source_group_id"] != "train-task0-instance7":
                    raise ValueError("bad source")

            @staticmethod
            def validate_event(value):
                if "event" not in value:
                    raise ValueError("bad event")

            @staticmethod
            def source_group_id(value):
                return value["source_group_id"]

            @staticmethod
            def event_id(value):
                return "event-id-1"

        raw = FakeSnapshotBackend()
        snapshots = SnapshotAdapter(raw)
        source = {"original_split": "train", "source_group_id": "train-task0-instance7"}
        event = {"event": "p107", "event_id": "event-id-1", "source": source}
        receipt = PairedRecoveryCollector(
            snapshots, FakeActionBackend(raw), physical_provider(FakePredicateBackend(physical_sample()))
        ).collect(
            source_ref=source, source_group_id="train-task0-instance7", event_ref=event,
            post_fault_snapshot=snapshots.capture("post_fault"), no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23], skill_binding=grasp_binding(), branch_seed=3,
            intent_bundle_id="intent-p107-grasp-0001", actor_evidence={"rgb_ref": "rgb://step/0"},
            prior_intent_bundle_id="intent-before",
        )
        bridge = validate_transport_receipt(
            receipt, source_ref=source, event=event, protocol=Protocol
        )
        self.assertEqual(bridge["schema_id"], "memlite-event-recovery-v1")
        self.assertEqual(bridge["event_id"], "event-id-1")

    def test_fake_branch_binds_to_a_real_canonical_event_but_remains_nontrainable(self) -> None:
        source, event = self._canonical_source_and_event()
        raw = FakeSnapshotBackend()
        snapshots = SnapshotAdapter(raw)
        receipt = PairedRecoveryCollector(
            snapshots, FakeActionBackend(raw), physical_provider(FakePredicateBackend(physical_sample()))
        ).collect(
            source_ref=source, source_group_id=source["source_group_id"], event_ref=event,
            post_fault_snapshot=snapshots.capture("post_fault"), no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23], skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001", branch_seed=4,
            actor_evidence={"rgb_ref": "rgb://step/0"},
            prior_intent_bundle_id="intent-before",
        )
        envelope = validate_transport_receipt(receipt, source_ref=source, event=event)
        self.assertTrue(envelope["canonical_event_validated"])
        self.assertEqual(envelope["event_id"], event["event_id"])
        self.assertFalse(envelope["receipt"]["ready_for_training"])
        self.assertFalse(envelope["receipt"]["corrective_positive_action_mask"])

    def test_same_group_sibling_source_or_event_cannot_envelope_receipt(self) -> None:
        protocol = load_protocol()
        source, event = self._canonical_source_and_event()
        raw = FakeSnapshotBackend()
        snapshots = SnapshotAdapter(raw)
        receipt = PairedRecoveryCollector(
            snapshots, FakeActionBackend(raw), physical_provider(FakePredicateBackend(physical_sample()))
        ).collect(
            source_ref=source, source_group_id=source["source_group_id"], event_ref=event,
            post_fault_snapshot=snapshots.capture("post_fault"), no_intervention_actions23=[ZERO23],
            corrective_actions23=[ONE23], skill_binding=grasp_binding(),
            intent_bundle_id="intent-p107-grasp-0001", branch_seed=5,
            actor_evidence={"rgb_ref": "rgb://step/0"}, prior_intent_bundle_id="intent-before",
        )
        sibling_source = deepcopy(source)
        sibling_source["raw_episode_id"] = 2101
        sibling_source["episode_index"] = 202
        sibling_source["source_group_id"] = protocol.source_group_id(sibling_source)
        self.assertEqual(sibling_source["source_group_id"], source["source_group_id"])
        sibling_event = deepcopy(event)
        sibling_event["source"] = sibling_source
        sibling_event["event_id"] = ""
        sibling_event["event_id"] = protocol.event_id(sibling_event)
        with self.assertRaisesRegex(RecoveryContractError, "Receipt source"):
            validate_transport_receipt(receipt, source_ref=sibling_source, event=sibling_event)


if __name__ == "__main__":
    unittest.main(verbosity=2)
