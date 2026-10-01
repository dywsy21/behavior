"""Skill-family physical-evidence interface.

This is intentionally a thin semantic layer over evaluator-side providers.  It
does not use episode end, a gripper command, a policy declaration, or a timeout
as a physical outcome.  A live OmniGibson binding can supply IsGrasping /
Inside / OnTop / ToggledOn and related facts through this interface once each
binding and version is independently checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .common import RecoveryContractError


OUTCOMES = {"IN_PROGRESS", "SUCCEEDED", "FAILED", "UNKNOWN"}


def _required_binding(binding: Mapping[str, Any], key: str) -> Any:
    if key not in binding:
        raise RecoveryContractError(f"Physical evidence requires explicit skill binding {key}")
    value = binding[key]
    if value is not None and (not isinstance(value, str) or not value):
        raise RecoveryContractError(f"Skill binding {key} must be nonempty text or None")
    return value


def _execution_context(context: Mapping[str, Any]) -> tuple[int, int, str, str]:
    required = ("action_start_frame", "actual_end_frame", "prior_intent_bundle_id", "current_intent_bundle_id")
    if any(key not in context for key in required):
        raise RecoveryContractError("Physical evidence requires execution clocks and prior/current intent identities")
    start, end = context["action_start_frame"], context["actual_end_frame"]
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise RecoveryContractError("Execution evidence requires a nonempty nonnegative action clock interval")
    prior, current = context["prior_intent_bundle_id"], context["current_intent_bundle_id"]
    if not all(isinstance(value, str) and value for value in (prior, current)):
        raise RecoveryContractError("Execution evidence requires immutable prior/current intent identities")
    return start, end, prior, current


@dataclass(frozen=True)
class PredicateSample:
    """A provider's evaluator-only observation of one skill-family predicate.

    ``predicate`` means the sought postcondition is currently true/false/unknown;
    it becomes success only with a causal observed transition and a stability
    streak.  ``failure_evidence`` is an affirmative physical fact (for example,
    an identified object remained behind after a lift), never a timeout.
    """

    family: str
    sensor_supported: bool
    source_binding_verified: bool
    predicate: bool | None
    causal_transition: bool
    failure_evidence: bool
    stable_frames: int
    required_stable_frames: int
    evidence_kind: str
    evidence_start_frame: int | None
    evidence_end_frame: int | None
    evidence_available_time: int | None
    object_id: str | None
    arm: str | None
    prior_intent_bundle_id: str | None
    current_intent_bundle_id: str | None
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise RecoveryContractError("Predicate family must be named")
        if type(self.sensor_supported) is not bool or type(self.source_binding_verified) is not bool:
            raise RecoveryContractError("Predicate support and source binding must be booleans")
        if self.predicate not in {True, False, None}:
            raise RecoveryContractError("Predicate must be true, false, or unknown")
        if type(self.causal_transition) is not bool or type(self.failure_evidence) is not bool:
            raise RecoveryContractError("Causality and failure evidence must be booleans")
        if (not isinstance(self.stable_frames, int) or not isinstance(self.required_stable_frames, int)
                or self.stable_frames < 0 or self.required_stable_frames < 1):
            raise RecoveryContractError("Stable-frame counts must be nonnegative and required>=1")
        if not isinstance(self.evidence_kind, str) or not self.evidence_kind:
            raise RecoveryContractError("Physical evidence requires an evidence kind")
        for field in ("evidence_start_frame", "evidence_end_frame", "evidence_available_time"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise RecoveryContractError(f"{field} must be a nonnegative clock or None")
        for field in ("object_id", "arm", "prior_intent_bundle_id", "current_intent_bundle_id"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise RecoveryContractError(f"{field} must be nonempty text or None")


@dataclass(frozen=True)
class EvidenceResult:
    family: str
    outcome: str
    valid_result_mask: bool
    evidence_kind: str
    evidence_start_frame: int | None
    evidence_end_frame: int | None
    evidence_available_time: int | None
    stable_frames: int
    required_stable_frames: int
    source_binding_verified: bool
    sensor_supported: bool
    causal_transition: bool
    object_id: str | None
    arm: str | None
    prior_intent_bundle_id: str | None
    current_intent_bundle_id: str | None
    # The evaluator's task termination is independent information.  It may be
    # recorded for audit but does not alter the local physical result above.
    official_task_success: bool | None
    official_terminal: bool | None
    detail: str

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise RecoveryContractError("Unexpected evidence outcome")
        if type(self.valid_result_mask) is not bool:
            raise RecoveryContractError("valid_result_mask must be a bool")
        if self.outcome == "UNKNOWN" and self.valid_result_mask:
            raise RecoveryContractError("UNKNOWN is not an affirmative labelled result")
        if not self.valid_result_mask and self.outcome != "UNKNOWN":
            raise RecoveryContractError("Missing evidence must remain UNKNOWN with mask false")
        for field in ("official_task_success", "official_terminal"):
            if getattr(self, field) is not None and type(getattr(self, field)) is not bool:
                raise RecoveryContractError(f"{field} must be bool or None")

    def public(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "outcome": self.outcome,
            "valid_result_mask": self.valid_result_mask,
            "evidence_kind": self.evidence_kind,
            "evidence_start_frame": self.evidence_start_frame,
            "evidence_end_frame": self.evidence_end_frame,
            "evidence_available_time": self.evidence_available_time,
            "stable_frames": self.stable_frames,
            "required_stable_frames": self.required_stable_frames,
            "source_binding_verified": self.source_binding_verified,
            "sensor_supported": self.sensor_supported,
            "causal_transition": self.causal_transition,
            "object_id": self.object_id,
            "arm": self.arm,
            "prior_intent_bundle_id": self.prior_intent_bundle_id,
            "current_intent_bundle_id": self.current_intent_bundle_id,
            "official_task_success": self.official_task_success,
            "official_terminal": self.official_terminal,
            "detail": self.detail,
        }


class PredicateBackend(Protocol):
    """A live owner supplies these facts from documented evaluator APIs only."""

    def sample(self, *, skill_binding: Mapping[str, Any], context: Mapping[str, Any]) -> PredicateSample: ...


class PhysicalEvidenceProvider:
    """Convert a binding-aware physical sample into a conservative label.

    The provider owns all privileged world facts.  Callers must place the
    returned result in ``privileged_evidence``; no method here projects it into
    actor input.
    """

    def __init__(self, backend: PredicateBackend, *, provider_id: str = "unverified-physical-provider"):
        if not isinstance(provider_id, str) or not provider_id:
            raise RecoveryContractError("Physical evidence provider requires a stable provider identity")
        self.backend = backend
        self.provider_id = provider_id

    def evaluate(
        self,
        *,
        skill_binding: Mapping[str, Any],
        context: Mapping[str, Any],
        official_task_success: bool | None = None,
        official_terminal: bool | None = None,
    ) -> EvidenceResult:
        if not isinstance(skill_binding, Mapping) or not skill_binding:
            raise RecoveryContractError("An explicit skill/object/arm binding is required")
        if not isinstance(context, Mapping):
            raise RecoveryContractError("Evidence context must be a mapping")
        family = _required_binding(skill_binding, "family")
        object_id = _required_binding(skill_binding, "object_id")
        arm = _required_binding(skill_binding, "arm")
        start, end, prior_intent, current_intent = _execution_context(context)
        sample = self.backend.sample(skill_binding=skill_binding, context=context)
        if not isinstance(sample, PredicateSample):
            raise RecoveryContractError("Predicate backend must return PredicateSample")
        available = (
            sample.sensor_supported
            and sample.source_binding_verified
            and sample.family == family
            and sample.object_id == object_id
            and sample.arm == arm
            and sample.prior_intent_bundle_id == prior_intent
            and sample.current_intent_bundle_id == current_intent
            and sample.evidence_start_frame is not None
            and sample.evidence_end_frame is not None
            and sample.evidence_available_time is not None
            and start <= sample.evidence_start_frame <= sample.evidence_end_frame <= sample.evidence_available_time <= end
        )
        stable = sample.stable_frames >= sample.required_stable_frames
        if not available:
            outcome, mask = "UNKNOWN", False
        elif sample.predicate is True and sample.causal_transition and stable:
            outcome, mask = "SUCCEEDED", True
        elif sample.failure_evidence:
            # This is an observed physical failure, not a clock or endpoint.
            outcome, mask = "FAILED", True
        elif sample.predicate is None:
            outcome, mask = "UNKNOWN", False
        else:
            # A known-but-not-yet-complete state is progress, not failure.
            outcome, mask = "IN_PROGRESS", True
        return EvidenceResult(
            family=sample.family,
            outcome=outcome,
            valid_result_mask=mask,
            evidence_kind=sample.evidence_kind,
            evidence_start_frame=sample.evidence_start_frame,
            evidence_end_frame=sample.evidence_end_frame,
            evidence_available_time=sample.evidence_available_time,
            stable_frames=sample.stable_frames,
            required_stable_frames=sample.required_stable_frames,
            source_binding_verified=sample.source_binding_verified,
            sensor_supported=sample.sensor_supported,
            causal_transition=sample.causal_transition,
            object_id=sample.object_id,
            arm=sample.arm,
            prior_intent_bundle_id=sample.prior_intent_bundle_id,
            current_intent_bundle_id=sample.current_intent_bundle_id,
            official_task_success=official_task_success,
            official_terminal=official_terminal,
            detail=sample.detail,
        )


@dataclass(frozen=True)
class FaultEvidenceSample:
    """Affirmative pre-branch deviation evidence for a genuine recovery event.

    A pair of successful branches is not itself a recovery.  The sample must
    bind the prior skill/object/arm/intent and the exact post-fault snapshot
    before any branch action is executed.
    """

    family: str
    object_id: str | None
    arm: str | None
    prior_intent_bundle_id: str
    current_intent_bundle_id: str | None
    fault_observed: bool
    source_binding_verified: bool
    evidence_kind: str
    evidence_start_frame: int | None
    evidence_end_frame: int | None
    evidence_available_time: int | None
    snapshot_state_fingerprint: str | None
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise RecoveryContractError("Fault evidence family must be named")
        for field in ("object_id", "arm"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise RecoveryContractError(f"Fault evidence {field} must be nonempty text or None")
        if not isinstance(self.prior_intent_bundle_id, str) or not self.prior_intent_bundle_id:
            raise RecoveryContractError("Fault evidence requires the prior intent identity")
        if self.current_intent_bundle_id is not None and (
            not isinstance(self.current_intent_bundle_id, str) or not self.current_intent_bundle_id
        ):
            raise RecoveryContractError("Fault evidence current intent must be nonempty text or None")
        if type(self.fault_observed) is not bool or type(self.source_binding_verified) is not bool:
            raise RecoveryContractError("Fault observed and source binding fields must be booleans")
        if not isinstance(self.evidence_kind, str) or not self.evidence_kind:
            raise RecoveryContractError("Fault evidence kind is required")
        for field in ("evidence_start_frame", "evidence_end_frame", "evidence_available_time"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise RecoveryContractError(f"Fault {field} must be a nonnegative clock or None")
        if self.snapshot_state_fingerprint is not None:
            from .common import require_sha256
            require_sha256(self.snapshot_state_fingerprint, field="fault snapshot_state_fingerprint")


@dataclass(frozen=True)
class FaultEvidenceResult:
    family: str
    object_id: str | None
    arm: str | None
    prior_intent_bundle_id: str
    current_intent_bundle_id: str | None
    fault_observed: bool
    valid_fault_mask: bool
    evidence_kind: str
    evidence_start_frame: int | None
    evidence_end_frame: int | None
    evidence_available_time: int | None
    snapshot_state_fingerprint: str | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise RecoveryContractError("Fault result family must be named")
        for field in ("object_id", "arm", "current_intent_bundle_id"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise RecoveryContractError(f"Fault result {field} must be nonempty text or None")
        if not isinstance(self.prior_intent_bundle_id, str) or not self.prior_intent_bundle_id:
            raise RecoveryContractError("Fault result requires the prior intent identity")
        if type(self.valid_fault_mask) is not bool:
            raise RecoveryContractError("Fault evidence mask must be boolean")
        if self.valid_fault_mask and not self.fault_observed:
            raise RecoveryContractError("Fault mask cannot be true without an affirmative observed deviation")

    def public(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "object_id": self.object_id,
            "arm": self.arm,
            "prior_intent_bundle_id": self.prior_intent_bundle_id,
            "current_intent_bundle_id": self.current_intent_bundle_id,
            "fault_observed": self.fault_observed,
            "valid_fault_mask": self.valid_fault_mask,
            "evidence_kind": self.evidence_kind,
            "evidence_start_frame": self.evidence_start_frame,
            "evidence_end_frame": self.evidence_end_frame,
            "evidence_available_time": self.evidence_available_time,
            "snapshot_state_fingerprint": self.snapshot_state_fingerprint,
            "detail": self.detail,
        }


class FaultEvidenceBackend(Protocol):
    def sample(self, *, skill_binding: Mapping[str, Any], context: Mapping[str, Any]) -> FaultEvidenceSample: ...


class FaultEvidenceProvider:
    """Fail-closed provider for evidence that the pre-branch state is faulty."""

    def __init__(self, backend: FaultEvidenceBackend, *, provider_id: str = "unverified-fault-provider"):
        if not isinstance(provider_id, str) or not provider_id:
            raise RecoveryContractError("Fault evidence provider requires a stable provider identity")
        self.backend = backend
        self.provider_id = provider_id

    def evaluate(
        self,
        *,
        skill_binding: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FaultEvidenceResult:
        if not isinstance(skill_binding, Mapping) or not isinstance(context, Mapping):
            raise RecoveryContractError("Fault evidence requires explicit binding and context mappings")
        family = _required_binding(skill_binding, "family")
        object_id = _required_binding(skill_binding, "object_id")
        arm = _required_binding(skill_binding, "arm")
        post_clock = context.get("post_fault_clock")
        prior_intent = context.get("prior_intent_bundle_id")
        current_intent = context.get("current_intent_bundle_id")
        snapshot_fingerprint = context.get("post_fault_snapshot_fingerprint")
        if type(post_clock) is not int or post_clock < 0:
            raise RecoveryContractError("Fault evidence requires an explicit post-fault observation clock")
        if not isinstance(prior_intent, str) or not prior_intent:
            raise RecoveryContractError("Fault evidence requires an immutable prior intent identity")
        if not isinstance(current_intent, str) or not current_intent:
            raise RecoveryContractError("Fault evidence requires an immutable current intent identity")
        if not isinstance(snapshot_fingerprint, str):
            raise RecoveryContractError("Fault evidence requires a captured post-fault snapshot fingerprint")
        sample = self.backend.sample(skill_binding=skill_binding, context=context)
        if not isinstance(sample, FaultEvidenceSample):
            raise RecoveryContractError("Fault backend must return FaultEvidenceSample")
        valid = bool(
            sample.fault_observed
            and sample.source_binding_verified
            and sample.family == family
            and sample.object_id == object_id
            and sample.arm == arm
            and sample.prior_intent_bundle_id == prior_intent
            and sample.current_intent_bundle_id == current_intent
            and sample.snapshot_state_fingerprint == snapshot_fingerprint
            and sample.evidence_start_frame is not None
            and sample.evidence_end_frame is not None
            and sample.evidence_available_time is not None
            and 0 <= sample.evidence_start_frame <= sample.evidence_end_frame <= sample.evidence_available_time <= post_clock
        )
        return FaultEvidenceResult(
            family=sample.family,
            object_id=sample.object_id,
            arm=sample.arm,
            prior_intent_bundle_id=sample.prior_intent_bundle_id,
            current_intent_bundle_id=sample.current_intent_bundle_id,
            fault_observed=sample.fault_observed,
            valid_fault_mask=valid,
            evidence_kind=sample.evidence_kind,
            evidence_start_frame=sample.evidence_start_frame,
            evidence_end_frame=sample.evidence_end_frame,
            evidence_available_time=sample.evidence_available_time,
            snapshot_state_fingerprint=sample.snapshot_state_fingerprint,
            detail=sample.detail,
        )
