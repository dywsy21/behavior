"""Candidate-only, closed-loop DART collection contracts.

The original DART path records a clean supervisor target at each state reached
under noisy execution.  The safety-bounded path is intentionally named
``dart_inspired``: its noisy controls are never low-level positives, while
later clean controls are actual execution receipts and still remain candidates
until the external data authority accepts them.

This is an additive adapter seam.  It does not open a simulator, create an
authority, or alter the paired-recovery collector owned by the simulator work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from math import isfinite
from time import monotonic
from typing import Any, Callable, Mapping, Protocol, Sequence

from .common import (
    RecoveryContractError,
    canonical_sha256,
    require_sha256,
    validate_raw23_action,
    validate_state61,
)
from .dart_noise import (
    BoundedNoiseProfile,
    DartInspiredBoundedNoise,
    OriginalGaussianNoise,
)


_CANDIDATE_FLAGS = {
    "candidate_only": True,
    "training_eligible": False,
    "ready_for_training": False,
    "low_action_supervision_positive": False,
    "authority_minted": False,
}
_SHA_FIELDS = (
    "source_release_sha256",
    "source_episode_sha256",
    "teacher_code_sha256",
    "teacher_weights_sha256",
    "teacher_config_sha256",
    "runtime_code_sha256",
)


def _action_tuple(action: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        return tuple(validate_raw23_action(action))
    except RecoveryContractError as exc:
        raise RecoveryContractError(f"{name}: {exc}") from exc


def _json_action(action: Sequence[float]) -> list[float]:
    return list(_action_tuple(action, "action"))


def _require_nonempty_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecoveryContractError(f"{field} must be nonempty text")
    return value


@dataclass(frozen=True)
class CandidateSourceReceipt:
    """Immutable source-group provenance; candidates may only derive from train."""

    source_group_id: str
    original_split: str
    source_release_sha256: str
    source_episode_sha256: str
    task_id: str
    task_instance_id: str
    task_seed: int

    def __post_init__(self) -> None:
        _require_nonempty_text(self.source_group_id, "source_group_id")
        if self.original_split != "train":
            raise RecoveryContractError("DART candidates may only use original train source groups")
        _require_nonempty_text(self.task_id, "task_id")
        _require_nonempty_text(self.task_instance_id, "task_instance_id")
        if not isinstance(self.task_seed, int):
            raise RecoveryContractError("task_seed must be an integer")
        for field in ("source_release_sha256", "source_episode_sha256"):
            require_sha256(getattr(self, field), field=field)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationReceipt:
    """A held-out subset of train groups reserved solely for covariance fitting."""

    source_group_ids: tuple[str, ...]
    calibration_trajectory_sha256: str
    learner_checkpoint_sha256: str
    teacher_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not self.source_group_ids or any(not isinstance(group, str) or not group for group in self.source_group_ids):
            raise RecoveryContractError("calibration receipt needs nonempty held-out train group IDs")
        if len(set(self.source_group_ids)) != len(self.source_group_ids):
            raise RecoveryContractError("calibration group IDs must be unique")
        for field in (
            "calibration_trajectory_sha256",
            "learner_checkpoint_sha256",
            "teacher_checkpoint_sha256",
        ):
            require_sha256(getattr(self, field), field=field)

    def reject_candidate_source(self, source: CandidateSourceReceipt) -> None:
        if source.source_group_id in self.source_group_ids:
            raise RecoveryContractError("held-out calibration groups cannot also produce DART candidates")

    def public(self) -> dict[str, Any]:
        return {**asdict(self), "purpose": "train_only_covariance_calibration"}


@dataclass(frozen=True)
class TeacherReceipt:
    teacher_id: str
    teacher_kind: str
    label_source: str
    feedback_mode: str
    postcondition_spec_sha256: str
    teacher_code_sha256: str
    teacher_weights_sha256: str
    teacher_config_sha256: str
    runtime_code_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_text(self.teacher_id, "teacher_id")
        _require_nonempty_text(self.teacher_kind, "teacher_kind")
        if self.label_source not in {"human", "planner", "privileged_oracle", "frozen_policy"}:
            raise RecoveryContractError("teacher label_source must be explicit")
        if self.feedback_mode != "closed_loop_fresh_observation":
            raise RecoveryContractError("DART feedback must be freshly queried, not a replayed source action")
        require_sha256(self.postcondition_spec_sha256, field="postcondition_spec_sha256")
        for field in _SHA_FIELDS[2:]:
            require_sha256(getattr(self, field), field=field)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSessionReceipt:
    """Proof that live DART starts from a fresh official task-instance reset.

    Frozen v4 metadata can identify a TRAIN source/task instance, but it does
    not contain a simulator snapshot.  A demo action/video replay is therefore
    never a valid substitute for this exact-session receipt.
    """

    runtime_session_id: str
    task_instance_id: str
    runtime_build_sha256: str
    asset_config_sha256: str
    reset_load_task_instance_receipt_sha256: str
    snapshot_mode: str = "no_restore_fresh_rollout"
    same_session_snapshot_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.runtime_session_id, "runtime_session_id")
        _require_nonempty_text(self.task_instance_id, "task_instance_id")
        for field in (
            "runtime_build_sha256",
            "asset_config_sha256",
            "reset_load_task_instance_receipt_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        if self.snapshot_mode not in {"no_restore_fresh_rollout", "same_session_verified_snapshot"}:
            raise RecoveryContractError("runtime session needs explicit fresh-rollout or same-session snapshot mode")
        if self.snapshot_mode == "same_session_verified_snapshot":
            require_sha256(self.same_session_snapshot_receipt_sha256, field="same_session_snapshot_receipt_sha256")
        elif self.same_session_snapshot_receipt_sha256 is not None:
            raise RecoveryContractError("fresh rollout may not claim a snapshot restore receipt")

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DartObservation:
    """A current state/observation receipt; raw state is never exported as actor input."""

    policy_clock: int
    state61: Sequence[float]
    observation_sha256: str
    observation_ref: str
    observation_fresh: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.policy_clock, int) or self.policy_clock < 0:
            raise RecoveryContractError("policy_clock must be a nonnegative integer")
        object.__setattr__(self, "state61", tuple(validate_state61(self.state61)))
        require_sha256(self.observation_sha256, field="observation_sha256")
        _require_nonempty_text(self.observation_ref, "observation_ref")
        if self.observation_fresh is not True:
            raise RecoveryContractError("DART teacher may only receive a fresh current RGB/proprio observation")

    @property
    def state61_sha256(self) -> str:
        return canonical_sha256(self.state61)

    def public(self) -> dict[str, Any]:
        return {
            "policy_clock": self.policy_clock,
            "state61_sha256": self.state61_sha256,
            "observation_sha256": self.observation_sha256,
            "observation_ref": self.observation_ref,
            "observation_fresh": True,
            "actor_visible": True,
        }


@dataclass(frozen=True)
class TeacherCommand:
    """A one-step clean label returned after observing the current state."""

    clean_intended23: Sequence[float]
    observed_policy_clock: int
    observed_state61_sha256: str
    observed_observation_sha256: str
    intent_bundle_id: str
    fresh_query_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "clean_intended23", _action_tuple(self.clean_intended23, "clean_intended23"))
        if not isinstance(self.observed_policy_clock, int) or self.observed_policy_clock < 0:
            raise RecoveryContractError("teacher command observed_policy_clock must be nonnegative")
        require_sha256(self.observed_state61_sha256, field="observed_state61_sha256")
        require_sha256(self.observed_observation_sha256, field="observed_observation_sha256")
        _require_nonempty_text(self.intent_bundle_id, "intent_bundle_id")
        require_sha256(self.fresh_query_receipt_sha256, field="fresh_query_receipt_sha256")


@dataclass(frozen=True)
class AppliedActionReceipt:
    """What the backend actually applied, including its own byte-level receipt."""

    status: str
    applied23: Sequence[float]
    applied_action_bytes_sha256: str
    runtime_step_receipt_sha256: str
    pre_action_policy_clock: int
    pre_action_state61_sha256: str
    pre_action_observation_sha256: str

    def __post_init__(self) -> None:
        if self.status not in {"APPLIED", "ABORTED"}:
            raise RecoveryContractError("action receipt status must be APPLIED or ABORTED")
        object.__setattr__(self, "applied23", _action_tuple(self.applied23, "applied23"))
        require_sha256(self.applied_action_bytes_sha256, field="applied_action_bytes_sha256")
        require_sha256(self.runtime_step_receipt_sha256, field="runtime_step_receipt_sha256")
        if not isinstance(self.pre_action_policy_clock, int) or self.pre_action_policy_clock < 0:
            raise RecoveryContractError("applied action pre_action_policy_clock must be nonnegative")
        require_sha256(self.pre_action_state61_sha256, field="pre_action_state61_sha256")
        require_sha256(self.pre_action_observation_sha256, field="pre_action_observation_sha256")


class DartRuntime(Protocol):
    """Closed-loop callback interface for the simulator owner's later adapter."""

    def observe(self) -> DartObservation: ...

    def apply_raw23(self, requested23: Sequence[float]) -> AppliedActionReceipt: ...


class DartTeacher(Protocol):
    def clean_action(self, observation: DartObservation, *, intent_bundle_id: str) -> TeacherCommand: ...


@dataclass(frozen=True)
class OutcomeEvidenceReceipt:
    """A delayed physical outcome label, never an actor feature or fabricated success."""

    label: str
    observed_policy_clock: int
    available_policy_clock: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.label not in {"ACTUAL_FAULT", "SURVIVAL_NONFAILURE", "OUTCOME_UNKNOWN"}:
            raise RecoveryContractError("outcome evidence must be actual fault, survival/nonfailure, or unknown")
        if not isinstance(self.observed_policy_clock, int) or not isinstance(self.available_policy_clock, int):
            raise RecoveryContractError("outcome evidence clocks must be integers")
        if self.available_policy_clock < self.observed_policy_clock:
            raise RecoveryContractError("outcome evidence cannot be available before it was observed")
        require_sha256(self.evidence_sha256, field="evidence_sha256")

    def public(self) -> dict[str, Any]:
        return {**asdict(self), "actor_visible": False}


@dataclass(frozen=True)
class CollectionLimits:
    """Bounds for an engineering round trip, not a semantic recovery horizon."""

    max_total_steps: int
    max_wall_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.max_total_steps, int) or self.max_total_steps <= 0:
            raise RecoveryContractError("max_total_steps must be positive")
        if not isinstance(self.max_wall_seconds, (int, float)) or not isfinite(float(self.max_wall_seconds)) or self.max_wall_seconds <= 0:
            raise RecoveryContractError("max_wall_seconds must be finite and positive")


@dataclass(frozen=True)
class DartCollectionResult:
    records: tuple[Mapping[str, Any], ...]
    outcome_label: str
    stop_reason: str | None

    def public(self) -> dict[str, Any]:
        return {
            "schema": "p107_dart_candidate_collection_result_v1",
            "records": list(self.records),
            "outcome_label": self.outcome_label,
            "stop_reason": self.stop_reason,
            **_CANDIDATE_FLAGS,
        }


def _assert_candidate_source(source: CandidateSourceReceipt, calibration: CalibrationReceipt) -> None:
    calibration.reject_candidate_source(source)


def _assert_teacher_matches(observation: DartObservation, command: TeacherCommand, intent_bundle_id: str) -> None:
    if command.observed_policy_clock != observation.policy_clock:
        raise RecoveryContractError("teacher target is not from the current reached policy clock")
    if command.observed_state61_sha256 != observation.state61_sha256:
        raise RecoveryContractError("teacher target state fingerprint does not match current reached state")
    if command.observed_observation_sha256 != observation.observation_sha256:
        raise RecoveryContractError("teacher target observation fingerprint does not match current observation")
    if command.intent_bundle_id != intent_bundle_id:
        raise RecoveryContractError("teacher target intent does not match collection intent")


def _profile_receipt_for_original(noise: OriginalGaussianNoise, *, covariance_update_mode: str) -> dict[str, Any]:
    if covariance_update_mode not in {"frozen_one_pass_partial_dart", "full_iterative_covariance_update"}:
        raise RecoveryContractError("original DART covariance update mode must be explicit")
    covariance = [list(row) for row in noise.covariance]
    return {
        "algorithm_id": noise.algorithm_id,
        "paper_equations": ["Eq.2", "Eq.3", "Eq.4"],
        "covariance_sha256": canonical_sha256(covariance),
        "sampling_seed": noise.seed,
        "covariance_update_mode": covariance_update_mode,
        # A frozen one-pass collection exercises exact sampling but cannot by
        # itself substantiate the paper's iterative empirical claim.
        "empirical_faithfulness": covariance_update_mode == "full_iterative_covariance_update",
    }


def _profile_receipt_for_bounded(noise: DartInspiredBoundedNoise) -> dict[str, Any]:
    profile = noise.profile
    serialized_profile = {
        "profile_id": profile.profile_id,
        "temporal_rho": profile.temporal_rho,
        "parts": [asdict(part) for part in profile.layout.parts],
        "rules_by_part": {
            key: asdict(value) if is_dataclass(value) else value
            for key, value in sorted(profile.rules_by_part.items())
        },
    }
    return {
        "algorithm_id": profile.algorithm_id,
        "profile_id": profile.profile_id,
        "profile_sha256": canonical_sha256(serialized_profile),
        "sampling_seed": noise.seed,
        "safety_bounded": True,
        "original_dart_equivalence": False,
    }


def _applied_receipt(
    runtime: DartRuntime, requested23: Sequence[float], observation: DartObservation, *, require_exact: bool
) -> AppliedActionReceipt:
    applied = runtime.apply_raw23(_action_tuple(requested23, "requested23"))
    if not isinstance(applied, AppliedActionReceipt):
        raise RecoveryContractError("DART runtime must return AppliedActionReceipt")
    if applied.status != "APPLIED":
        raise RecoveryContractError("DART collection aborted before an action was applied")
    if (
        applied.pre_action_policy_clock != observation.policy_clock
        or applied.pre_action_state61_sha256 != observation.state61_sha256
        or applied.pre_action_observation_sha256 != observation.observation_sha256
    ):
        raise RecoveryContractError("applied action receipt is not bound to the current observed clock/state/observation")
    requested = _action_tuple(requested23, "requested23")
    if require_exact and applied.applied23 != requested:
        raise RecoveryContractError("original DART Gaussian mode requires applied action equal requested sample")
    return applied


def _candidate_record(
    *,
    label_kind: str,
    collection_mode: str,
    source: CandidateSourceReceipt,
    runtime_session: RuntimeSessionReceipt,
    calibration: CalibrationReceipt,
    teacher: TeacherReceipt,
    observation: DartObservation,
    command: TeacherCommand,
    requested_noisy23: Sequence[float],
    sampled_noise23: Sequence[float],
    applied: AppliedActionReceipt,
    noise_profile: Mapping[str, Any],
    execution_role: str,
    chunk_index: int | None = None,
    chunk_step: int | None = None,
    outcome_evidence: OutcomeEvidenceReceipt | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": "p107_dart_candidate_v1",
        "label_kind": label_kind,
        "collection_mode": collection_mode,
        "source": source.public(),
        "runtime_session": runtime_session.public(),
        "calibration_receipt": calibration.public(),
        "teacher": teacher.public(),
        "teacher_query_receipt_sha256": command.fresh_query_receipt_sha256,
        "observation": observation.public(),
        # This clean target is *one* teacher response at this observation.  It
        # is expressly not a fabricated 32-step expert future.
        "clean_intended23": _json_action(command.clean_intended23),
        "clean_target_steps": 1,
        "requested_noisy23": _json_action(requested_noisy23),
        "sampled_noise23": _json_action(sampled_noise23),
        "applied23": _json_action(applied.applied23),
        "applied_action_bytes_sha256": applied.applied_action_bytes_sha256,
        "runtime_step_receipt_sha256": applied.runtime_step_receipt_sha256,
        "applied_pre_action": {
            "policy_clock": applied.pre_action_policy_clock,
            "state61_sha256": applied.pre_action_state61_sha256,
            "observation_sha256": applied.pre_action_observation_sha256,
        },
        "intent_bundle_id": command.intent_bundle_id,
        "noise_profile": dict(noise_profile),
        "execution_role": execution_role,
        "prediction_contract": {
            "future_prediction_steps": 32,
            "execution_start": 0,
            "max_executed_per_replan_chunk": 16,
            "stored_clean_future_tail": False,
        },
        **_CANDIDATE_FLAGS,
    }
    if chunk_index is not None:
        record["recovery_chunk_index"] = chunk_index
        record["recovery_chunk_step"] = chunk_step
    if outcome_evidence is not None:
        record["outcome_evidence"] = outcome_evidence.public()
    record["candidate_sha256"] = canonical_sha256(record)
    return record


def collect_original_gaussian_clean_feedback(
    *,
    runtime: DartRuntime,
    teacher_callback: DartTeacher,
    teacher_receipt: TeacherReceipt,
    source: CandidateSourceReceipt,
    runtime_session: RuntimeSessionReceipt,
    calibration: CalibrationReceipt,
    noise: OriginalGaussianNoise,
    intent_bundle_id: str,
    steps: int,
    covariance_update_mode: str,
) -> DartCollectionResult:
    """Collect original-DART clean intended labels at states reached by Gaussian control.

    Each loop queries the teacher at the current reached state before sampling
    one noisy execution.  The clean label is not marked as actual execution.
    """

    _assert_candidate_source(source, calibration)
    if runtime_session.task_instance_id != source.task_instance_id:
        raise RecoveryContractError("runtime fresh-reset task instance must match candidate source instance")
    _require_nonempty_text(intent_bundle_id, "intent_bundle_id")
    if not isinstance(steps, int) or steps <= 0:
        raise RecoveryContractError("original DART steps must be positive")
    profile = _profile_receipt_for_original(noise, covariance_update_mode=covariance_update_mode)
    records: list[Mapping[str, Any]] = []
    for _ in range(steps):
        observation = runtime.observe()
        if not isinstance(observation, DartObservation):
            raise RecoveryContractError("DART runtime must return DartObservation")
        command = teacher_callback.clean_action(observation, intent_bundle_id=intent_bundle_id)
        if not isinstance(command, TeacherCommand):
            raise RecoveryContractError("DART teacher must return TeacherCommand")
        _assert_teacher_matches(observation, command, intent_bundle_id)
        sampled = noise.sample(command.clean_intended23)
        applied = _applied_receipt(runtime, sampled.requested_noisy23, observation, require_exact=True)
        records.append(
            _candidate_record(
                label_kind="dart_clean_supervisor_feedback",
                collection_mode="original_gaussian_clean_intended_feedback",
                source=source,
                runtime_session=runtime_session,
                calibration=calibration,
                teacher=teacher_receipt,
                observation=observation,
                command=command,
                requested_noisy23=sampled.requested_noisy23,
                sampled_noise23=sampled.sampled_noise23,
                applied=applied,
                noise_profile=profile,
                execution_role="noisy_dart_execution_clean_target_not_executed",
            )
        )
    return DartCollectionResult(tuple(records), outcome_label="OUTCOME_UNKNOWN", stop_reason=None)


def collect_dart_inspired_actual_clean_recovery(
    *,
    runtime: DartRuntime,
    teacher_callback: DartTeacher,
    teacher_receipt: TeacherReceipt,
    source: CandidateSourceReceipt,
    runtime_session: RuntimeSessionReceipt,
    calibration: CalibrationReceipt,
    noise: DartInspiredBoundedNoise,
    intent_bundle_id: str,
    noisy_injection_steps: int,
    recovery_chunk_lengths: Sequence[int],
    limits: CollectionLimits,
    outcome_evidence: OutcomeEvidenceReceipt | None = None,
    clock: Callable[[], float] = monotonic,
) -> DartCollectionResult:
    """Collect bounded noisy injections followed by actual clean closed-loop recovery.

    A recovery chunk is at most 16 controls, but every control is re-planned at
    its own observed state.  The result stores no 32-action clean tail and marks
    an uncompleted budget as unknown rather than a success/failure label.
    """

    _assert_candidate_source(source, calibration)
    if runtime_session.task_instance_id != source.task_instance_id:
        raise RecoveryContractError("runtime fresh-reset task instance must match candidate source instance")
    _require_nonempty_text(intent_bundle_id, "intent_bundle_id")
    if not isinstance(noisy_injection_steps, int) or noisy_injection_steps <= 0:
        raise RecoveryContractError("noisy_injection_steps must be positive")
    if not recovery_chunk_lengths:
        raise RecoveryContractError("at least one bounded recovery chunk is required")
    if any(not isinstance(length, int) or length <= 0 or length > 16 for length in recovery_chunk_lengths):
        raise RecoveryContractError("every recovery chunk must have 1..16 actual controls")
    planned_steps = noisy_injection_steps + sum(recovery_chunk_lengths)
    if planned_steps > limits.max_total_steps:
        raise RecoveryContractError("planned DART-inspired controls exceed max_total_steps")
    if not callable(clock):
        raise RecoveryContractError("clock must be callable")
    start_time = float(clock())
    if not isfinite(start_time):
        raise RecoveryContractError("clock must return finite seconds")
    profile = _profile_receipt_for_bounded(noise)
    records: list[Mapping[str, Any]] = []
    executed = 0

    def budget_remaining() -> bool:
        now = float(clock())
        if not isfinite(now):
            raise RecoveryContractError("clock must return finite seconds")
        return executed < limits.max_total_steps and now - start_time <= limits.max_wall_seconds

    def clean_command(observation: DartObservation) -> TeacherCommand:
        command = teacher_callback.clean_action(observation, intent_bundle_id=intent_bundle_id)
        if not isinstance(command, TeacherCommand):
            raise RecoveryContractError("DART teacher must return TeacherCommand")
        _assert_teacher_matches(observation, command, intent_bundle_id)
        return command

    for _ in range(noisy_injection_steps):
        if not budget_remaining():
            return DartCollectionResult(tuple(records), "OUTCOME_UNKNOWN", "collection_budget_exhausted")
        observation = runtime.observe()
        if not isinstance(observation, DartObservation):
            raise RecoveryContractError("DART runtime must return DartObservation")
        command = clean_command(observation)
        sampled = noise.sample(command.clean_intended23)
        applied = _applied_receipt(runtime, sampled.requested_noisy23, observation, require_exact=False)
        if applied.applied23 == command.clean_intended23:
            raise RecoveryContractError("DART-inspired noisy injection did not actually perturb the applied action")
        executed += 1
        records.append(
            _candidate_record(
                label_kind="dart_inspired_noisy_injection",
                collection_mode="dart_inspired_bounded_actual_clean_recovery",
                source=source,
                runtime_session=runtime_session,
                calibration=calibration,
                teacher=teacher_receipt,
                observation=observation,
                command=command,
                requested_noisy23=sampled.requested_noisy23,
                sampled_noise23=sampled.sampled_noise23,
                applied=applied,
                noise_profile=profile,
                execution_role="noisy_injection_excluded_from_fm_supervision",
            )
        )

    for chunk_index, chunk_length in enumerate(recovery_chunk_lengths):
        for chunk_step in range(chunk_length):
            if not budget_remaining():
                return DartCollectionResult(tuple(records), "OUTCOME_UNKNOWN", "collection_budget_exhausted")
            observation = runtime.observe()
            if not isinstance(observation, DartObservation):
                raise RecoveryContractError("DART runtime must return DartObservation")
            command = clean_command(observation)
            # This is a genuine actual-clean execution receipt: controller
            # substitution is rejected instead of being silently relabelled.
            applied = _applied_receipt(runtime, command.clean_intended23, observation, require_exact=True)
            executed += 1
            records.append(
                _candidate_record(
                    label_kind="dart_inspired_actual_clean_recovery",
                    collection_mode="dart_inspired_bounded_actual_clean_recovery",
                    source=source,
                    runtime_session=runtime_session,
                    calibration=calibration,
                    teacher=teacher_receipt,
                    observation=observation,
                    command=command,
                    requested_noisy23=command.clean_intended23,
                    sampled_noise23=(0.0,) * 23,
                    applied=applied,
                    noise_profile=profile,
                    execution_role="actual_clean_closed_loop_recovery_candidate",
                    chunk_index=chunk_index,
                    chunk_step=chunk_step,
                    outcome_evidence=outcome_evidence,
                )
            )

    outcome = "OUTCOME_UNKNOWN" if outcome_evidence is None else outcome_evidence.label
    return DartCollectionResult(tuple(records), outcome_label=outcome, stop_reason=None)
