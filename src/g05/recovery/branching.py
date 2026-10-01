"""Paired recovery branches that execute only official 23-D controls.

The collector accepts a *post-fault* snapshot and restores it before each
branch.  It never creates a fault action, reuses an unexecuted prediction tail,
or turns a failed correction into positive action supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .common import (
    RecoveryContractError,
    canonical_sha256,
    validate_raw23_actions,
    validate_state61,
)
from .evidence import EvidenceResult, FaultEvidenceProvider, FaultEvidenceResult, PhysicalEvidenceProvider
from .snapshot import SnapshotAdapter, SnapshotCapture


_PRIVILEGED_ACTOR_KEYS = {
    "privileged_evidence", "object_pose", "object_poses", "contact", "contacts",
    "grasp", "grasp_state", "particle", "particles", "rng", "snapshot",
    "restore_identity", "controller_state", "official_task_success", "target_uid",
}


@dataclass(frozen=True)
class Observation:
    """One current, pre-action R1Pro observation for an execution receipt."""

    policy_clock: int
    state61: list[float]
    observation_fresh: bool
    public_observation_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy_clock, int) or self.policy_clock < 0:
            raise RecoveryContractError("Observation policy clock must be nonnegative")
        object.__setattr__(self, "state61", validate_state61(self.state61))
        if type(self.observation_fresh) is not bool:
            raise RecoveryContractError("Observation freshness must be explicit")
        if self.public_observation_ref is not None and not isinstance(self.public_observation_ref, str):
            raise RecoveryContractError("Public observation references must be text or None")

    def public(self) -> dict[str, Any]:
        return {
            "policy_clock": self.policy_clock,
            "state61": self.state61,
            "observation_fresh": self.observation_fresh,
            "public_observation_ref": self.public_observation_ref,
        }


class ActionBackend(Protocol):
    """A tiny wrapper whose ``step`` must call the real evaluator exactly once."""

    def observe(self) -> Observation: ...

    def step(self, action23: list[float]) -> Mapping[str, Any] | None: ...

    def official_status(self) -> Mapping[str, Any] | None: ...


class OmniGibsonEvaluatorActionBackend:
    """Explicit public-API bridge for an already-owned evaluator session.

    The existing historical collectors call
    ``evaluator.env.step(action, n_render_iterations=1)``.  This adapter makes
    exactly that call after the action owner has encoded the official raw 23-D
    vector (for example to the evaluator's torch tensor).  Observation refresh
    is intentionally an injected callback because the known collector refreshes
    evaluator observations using evaluator-specific preprocessing.  The adapter
    never imports OmniGibson, opens a session, or assumes that an observation is
    fresh merely because ``step`` returned.

    This is an integration seam, not proof that the current installed simulator
    can restore snapshots.  Its default ``integration_status`` is unverified.
    """

    integration_status = "unverified"

    def __init__(
        self,
        env: Any,
        *,
        observe_current: Callable[[], Observation],
        official_status_current: Callable[[], Mapping[str, Any] | None],
        encode_action: Callable[[list[float]], Any],
        refresh_after_step: Callable[[Any], None] | None = None,
        n_render_iterations: int = 1,
    ) -> None:
        if not callable(getattr(env, "step", None)):
            raise RecoveryContractError("Evaluator env must expose public step(action, ...)" )
        if not all(callable(value) for value in (observe_current, official_status_current, encode_action)):
            raise RecoveryContractError("Evaluator adapter hooks must be explicit callables")
        if refresh_after_step is not None and not callable(refresh_after_step):
            raise RecoveryContractError("refresh_after_step must be callable or None")
        if not isinstance(n_render_iterations, int) or n_render_iterations < 1:
            raise RecoveryContractError("n_render_iterations must be a positive integer")
        self._env = env
        self._observe_current = observe_current
        self._official_status_current = official_status_current
        self._encode_action = encode_action
        self._refresh_after_step = refresh_after_step
        self._n_render_iterations = n_render_iterations

    def observe(self) -> Observation:
        return self._observe_current()

    def step(self, action23: list[float]) -> Mapping[str, Any]:
        action23 = validate_raw23_actions([action23])[0]
        result = self._env.step(
            self._encode_action(action23), n_render_iterations=self._n_render_iterations
        )
        if self._refresh_after_step is not None:
            self._refresh_after_step(result)
        # Do not export arbitrary evaluator info: it may expose privileged state.
        return {"env_step_returned": True}

    def official_status(self) -> Mapping[str, Any] | None:
        return self._official_status_current()


@dataclass
class BranchExecution:
    branch: str
    intent_bundle_id: str
    actions23: list[list[float]]
    pre_action_observations: list[Observation]
    final_observation: Observation
    step_receipts: list[Mapping[str, Any] | None]
    official_task_success: bool | None
    official_terminal: bool | None
    evidence: EvidenceResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.branch, str) or not self.branch:
            raise RecoveryContractError("Branch name is required")
        if not isinstance(self.intent_bundle_id, str) or not self.intent_bundle_id:
            raise RecoveryContractError("Branch execution requires an intent bundle identity")
        self.actions23 = validate_raw23_actions(self.actions23)
        if len(self.pre_action_observations) != len(self.actions23):
            raise RecoveryContractError("Every exported action needs its own pre-action 61-D observation")
        if not all(isinstance(obs, Observation) for obs in self.pre_action_observations):
            raise RecoveryContractError("Pre-action observations must be Observation records")
        if not isinstance(self.final_observation, Observation):
            raise RecoveryContractError("Final observation must be an Observation record")
        start = self.pre_action_observations[0].policy_clock
        if any(obs.policy_clock != start + index for index, obs in enumerate(self.pre_action_observations)):
            raise RecoveryContractError("Action clocks must be contiguous actual execution clocks")
        if self.final_observation.policy_clock != start + len(self.actions23):
            raise RecoveryContractError("actual_end_frame must equal action_start_frame plus actual execution length")

    @property
    def actual_executed_length(self) -> int:
        return len(self.actions23)

    @property
    def action_sha256(self) -> str:
        return canonical_sha256(self.actions23)

    @property
    def observations_fresh(self) -> bool:
        return all(obs.observation_fresh for obs in [*self.pre_action_observations, self.final_observation])

    def public(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            # This nested execution receipt is intentionally pre-protocol.  It
            # is the auditable input for the data owner's canonical action
            # projection, not a competing training-row schema.
            "execution": {
                "executed": True,
                "raw_action_sha256": self.action_sha256,
                "intent_bundle_id": self.intent_bundle_id,
                "action_intent_bundle_id": self.intent_bundle_id,
                "executed_intent_bundle_id": self.intent_bundle_id,
                "action_start_frame": self.pre_action_observations[0].policy_clock,
                "actual_end_frame": self.final_observation.policy_clock,
                "actual_executed_length": self.actual_executed_length,
                "raw_action_dim": 23,
            },
            "actions23": self.actions23,
            "raw_action_sha256": self.action_sha256,
            "actual_executed_length": self.actual_executed_length,
            "pre_action_observations": [obs.public() for obs in self.pre_action_observations],
            "final_observation": self.final_observation.public(),
            # Arbitrary env.step info can contain private simulator values.  It
            # is retained by the owned runtime, never exported to actor input.
            "step_receipt_count": len(self.step_receipts),
            "official_task_success": self.official_task_success,
            "official_terminal": self.official_terminal,
            "evidence": None if self.evidence is None else self.evidence.public(),
            "observations_fresh": self.observations_fresh,
        }


@dataclass
class BranchPairReceipt:
    source_ref: Mapping[str, Any]
    source_group_id: str
    event_ref: Mapping[str, Any]
    restore_identity: str
    branch_seed: int
    no_intervention: BranchExecution
    corrective: BranchExecution
    no_intervention_same_state: bool
    corrective_same_state: bool
    post_fault_evidence: FaultEvidenceResult | None
    post_fault_live_ready: bool
    physical_provider_trusted: bool
    fault_provider_trusted: bool
    actor_evidence: Mapping[str, Any]
    backend_kind: str

    @property
    def recovery_eligible(self) -> bool:
        return bool(
            self.post_fault_evidence is not None
            and self.post_fault_evidence.fault_observed
            and self.post_fault_evidence.valid_fault_mask
        )

    @property
    def corrective_positive_action_mask(self) -> bool:
        evidence = self.corrective.evidence
        return bool(
            self.post_fault_live_ready
            and self.physical_provider_trusted
            and self.fault_provider_trusted
            and self.recovery_eligible
            and self.no_intervention_same_state
            and self.corrective_same_state
            and self.corrective.observations_fresh
            and self.corrective.actual_executed_length > 0
            and evidence is not None
            and evidence.outcome == "SUCCEEDED"
            and evidence.valid_result_mask
            and evidence.current_intent_bundle_id == self.corrective.intent_bundle_id
        )

    def public(self) -> dict[str, Any]:
        """An auditable transport receipt, deliberately not a training view."""
        return {
            "kind": "paired_event_recovery_receipt",
            "source_ref": dict(self.source_ref),
            "source_group_id": self.source_group_id,
            "event_ref": dict(self.event_ref),
            "restore_identity": self.restore_identity,
            "branch_seed": self.branch_seed,
            "backend_kind": self.backend_kind,
            "same_post_fault_state": {
                "no_intervention": self.no_intervention_same_state,
                "corrective": self.corrective_same_state,
            },
            "recovery_kind": "genuine_recovery" if self.recovery_eligible else "normal_or_unproven",
            "actor_evidence": dict(self.actor_evidence),
            "privileged_evidence": {
                "post_fault_deviation": (
                    None if self.post_fault_evidence is None else self.post_fault_evidence.public()
                ),
                "no_intervention": self.no_intervention.public(),
                "corrective": self.corrective.public(),
            },
            "live_readiness": {
                "post_fault_capture_attested": self.post_fault_live_ready,
                "physical_provider_trusted": self.physical_provider_trusted,
                "fault_provider_trusted": self.fault_provider_trusted,
            },
            "corrective_positive_action_mask": self.corrective_positive_action_mask,
            # The data owner's canonical protocol must separately validate and
            # project this receipt.  A fake/unknown snapshot can never train.
            "ready_for_training": False,
        }


def executed_prefix_from_prediction(
    predicted_actions23: Sequence[Any], *, execution_start: int, executed_length: int
) -> list[list[float]]:
    """Admit only the actually executed prefix of a 32-step prediction.

    The R1Pro serving contract executes indices ``0:16``.  This rejects the
    common but invalid inference that image history changes the action start,
    and prevents the unexecuted 16:32 tail becoming a simulator transition.
    """
    if not isinstance(execution_start, int) or execution_start != 0:
        raise RecoveryContractError("Recovery execution must begin at predicted action index 0")
    if not isinstance(executed_length, int) or not 1 <= executed_length <= 16:
        raise RecoveryContractError("Only one actually executed 1:16 prefix may be collected")
    predicted = validate_raw23_actions(predicted_actions23)
    if len(predicted) != 32:
        raise RecoveryContractError("Predicted chunks must retain the explicit 32-step horizon")
    return predicted[:executed_length]


def _validate_train_source(source_ref: Mapping[str, Any], source_group_id: str) -> None:
    if not isinstance(source_ref, Mapping):
        raise RecoveryContractError("A canonical source reference is required")
    if source_ref.get("original_split") != "train":
        raise RecoveryContractError("Only immutable TRAIN source groups may feed paired recovery collection")
    if not isinstance(source_group_id, str) or not source_group_id:
        raise RecoveryContractError("Source-group identity is required for split isolation")
    declared = source_ref.get("source_group_id")
    if declared is not None and declared != source_group_id:
        raise RecoveryContractError("Source reference and branch source-group identity disagree")


def _validate_actor_evidence(actor_evidence: Mapping[str, Any]) -> None:
    if not isinstance(actor_evidence, Mapping):
        raise RecoveryContractError("Actor evidence must be an explicit public mapping")
    seen = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise RecoveryContractError("Actor evidence keys must be strings")
                if key.casefold() in _PRIVILEGED_ACTOR_KEYS:
                    raise RecoveryContractError("Privileged simulator evidence cannot enter actor evidence")
                seen.add(key.casefold())
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(actor_evidence)
    if "privileged_evidence" in seen:
        raise RecoveryContractError("Actor evidence must not contain privileged evidence")


class PairedRecoveryCollector:
    """Run no-intervention and corrective actions from one post-fault snapshot."""

    def __init__(
        self,
        snapshots: SnapshotAdapter,
        actions: ActionBackend,
        evidence: PhysicalEvidenceProvider,
        fault_evidence: FaultEvidenceProvider | None = None,
    ):
        self.snapshots = snapshots
        self.actions = actions
        self.evidence = evidence
        self.fault_evidence = fault_evidence

    def _official_status(self) -> tuple[bool | None, bool | None]:
        status = self.actions.official_status()
        if status is None:
            return None, None
        if not isinstance(status, Mapping):
            raise RecoveryContractError("Official evaluator status must be a mapping or None")
        success, terminal = status.get("success"), status.get("terminal")
        if success is not None and type(success) is not bool:
            raise RecoveryContractError("Official success must be bool or None")
        if terminal is not None and type(terminal) is not bool:
            raise RecoveryContractError("Official terminal must be bool or None")
        return success, terminal

    def _execute(
        self,
        *,
        branch: str,
        actions23: Sequence[Any],
        skill_binding: Mapping[str, Any],
        context: Mapping[str, Any],
        intent_bundle_id: str,
        prior_intent_bundle_id: str,
    ) -> BranchExecution:
        actions23 = validate_raw23_actions(actions23)
        if len(actions23) > 16:
            raise RecoveryContractError("Collector admits only the actually executed 1:16 action prefix")
        before, step_receipts = [], []
        for action in actions23:
            observation = self.actions.observe()
            if not isinstance(observation, Observation):
                raise RecoveryContractError("Action backend observe() must return Observation")
            before.append(observation)
            # Appending only after no exception means this receipt contains
            # actual successful env.step calls, never planned actions.
            receipt = self.actions.step(action)
            if receipt is not None and not isinstance(receipt, Mapping):
                raise RecoveryContractError("Action backend step receipt must be mapping or None")
            step_receipts.append(receipt)
        final = self.actions.observe()
        if not isinstance(final, Observation):
            raise RecoveryContractError("Action backend observe() must return Observation")
        success, terminal = self._official_status()
        branch_result = BranchExecution(
            branch=branch,
            intent_bundle_id=intent_bundle_id,
            actions23=actions23,
            pre_action_observations=before,
            final_observation=final,
            step_receipts=step_receipts,
            official_task_success=success,
            official_terminal=terminal,
        )
        branch_context = {
            **dict(context),
            "branch": branch,
            "actual_executed_length": len(actions23),
            "action_start_frame": branch_result.pre_action_observations[0].policy_clock,
            "actual_end_frame": branch_result.final_observation.policy_clock,
            "prior_intent_bundle_id": prior_intent_bundle_id,
            "current_intent_bundle_id": intent_bundle_id,
        }
        branch_result.evidence = self.evidence.evaluate(
            skill_binding=skill_binding,
            context=branch_context,
            official_task_success=success,
            official_terminal=terminal,
        )
        return branch_result

    def collect(
        self,
        *,
        source_ref: Mapping[str, Any],
        source_group_id: str,
        event_ref: Mapping[str, Any],
        post_fault_snapshot: SnapshotCapture,
        no_intervention_actions23: Sequence[Any],
        corrective_actions23: Sequence[Any],
        skill_binding: Mapping[str, Any],
        intent_bundle_id: str,
        prior_intent_bundle_id: str,
        branch_seed: int,
        actor_evidence: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> BranchPairReceipt:
        _validate_train_source(source_ref, source_group_id)
        _validate_actor_evidence(actor_evidence)
        if not isinstance(event_ref, Mapping) or not event_ref:
            raise RecoveryContractError("Canonical event reference is required")
        if not isinstance(branch_seed, int) or branch_seed < 0:
            raise RecoveryContractError("A deterministic nonnegative branch seed is required")
        if not isinstance(skill_binding, Mapping) or not skill_binding:
            raise RecoveryContractError("Recovery branches require a bound physical skill")
        if not isinstance(intent_bundle_id, str) or not intent_bundle_id:
            raise RecoveryContractError("Executed recovery actions require a deterministic intent bundle identity")
        if not isinstance(prior_intent_bundle_id, str) or not prior_intent_bundle_id:
            raise RecoveryContractError("Recovery branches require an immutable prior intent identity")
        context = {} if context is None else dict(context)

        # Capture an observation before either branch executes.  A fault provider
        # must bind its affirmative deviation evidence to this snapshot digest
        # and this clock; an ordinary successful state is therefore not silently
        # reclassified as a recovery event.
        post_fault_observation = self.actions.observe()
        if not isinstance(post_fault_observation, Observation):
            raise RecoveryContractError("Action backend observe() must return Observation")
        post_fault_evidence = None
        if self.fault_evidence is not None:
            if post_fault_snapshot.state_fingerprint is None:
                raise RecoveryContractError("Fault evidence cannot bind an opaque snapshot without a fingerprint")
            post_fault_evidence = self.fault_evidence.evaluate(
                skill_binding=skill_binding,
                context={
                    **context,
                    "post_fault_clock": post_fault_observation.policy_clock,
                    "post_fault_snapshot_fingerprint": post_fault_snapshot.state_fingerprint,
                    "prior_intent_bundle_id": prior_intent_bundle_id,
                    "current_intent_bundle_id": intent_bundle_id,
                },
            )

        no_start = self.snapshots.restore_and_capture(post_fault_snapshot, label="no_intervention:restored")
        no_same = self.snapshots.same_state(post_fault_snapshot, no_start)
        no_branch = self._execute(
            branch="no_intervention", actions23=no_intervention_actions23,
            skill_binding=skill_binding, context=context, intent_bundle_id=intent_bundle_id,
            prior_intent_bundle_id=prior_intent_bundle_id,
        )

        correction_start = self.snapshots.restore_and_capture(post_fault_snapshot, label="corrective:restored")
        correction_same = self.snapshots.same_state(post_fault_snapshot, correction_start)
        correction_branch = self._execute(
            branch="corrective", actions23=corrective_actions23,
            skill_binding=skill_binding, context=context, intent_bundle_id=intent_bundle_id,
            prior_intent_bundle_id=prior_intent_bundle_id,
        )
        return BranchPairReceipt(
            source_ref=source_ref,
            source_group_id=source_group_id,
            event_ref=event_ref,
            restore_identity=post_fault_snapshot.restore_identity,
            branch_seed=branch_seed,
            no_intervention=no_branch,
            corrective=correction_branch,
            no_intervention_same_state=no_same,
            corrective_same_state=correction_same,
            post_fault_evidence=post_fault_evidence,
            post_fault_live_ready=self.snapshots.is_live_ready_capture(post_fault_snapshot),
            physical_provider_trusted=self.snapshots.accepts_evidence_provider(
                self.evidence.provider_id, kind="physical"
            ),
            fault_provider_trusted=(
                self.fault_evidence is not None
                and self.snapshots.accepts_evidence_provider(self.fault_evidence.provider_id, kind="fault")
            ),
            actor_evidence=actor_evidence,
            backend_kind=post_fault_snapshot.backend_kind,
        )
