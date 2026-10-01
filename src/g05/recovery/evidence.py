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
    evidence_available_time: int | None
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
        if self.evidence_available_time is not None and (
                not isinstance(self.evidence_available_time, int) or self.evidence_available_time < 0):
            raise RecoveryContractError("Evidence availability must be a nonnegative clock or None")


@dataclass(frozen=True)
class EvidenceResult:
    family: str
    outcome: str
    valid_result_mask: bool
    evidence_kind: str
    evidence_available_time: int | None
    stable_frames: int
    required_stable_frames: int
    source_binding_verified: bool
    sensor_supported: bool
    causal_transition: bool
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
            "evidence_available_time": self.evidence_available_time,
            "stable_frames": self.stable_frames,
            "required_stable_frames": self.required_stable_frames,
            "source_binding_verified": self.source_binding_verified,
            "sensor_supported": self.sensor_supported,
            "causal_transition": self.causal_transition,
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

    def __init__(self, backend: PredicateBackend):
        self.backend = backend

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
        sample = self.backend.sample(skill_binding=skill_binding, context=context)
        if not isinstance(sample, PredicateSample):
            raise RecoveryContractError("Predicate backend must return PredicateSample")
        available = (
            sample.sensor_supported
            and sample.source_binding_verified
            and sample.evidence_available_time is not None
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
            evidence_available_time=sample.evidence_available_time,
            stable_frames=sample.stable_frames,
            required_stable_frames=sample.required_stable_frames,
            source_binding_verified=sample.source_binding_verified,
            sensor_supported=sample.sensor_supported,
            causal_transition=sample.causal_transition,
            official_task_success=official_task_success,
            official_terminal=official_terminal,
            detail=sample.detail,
        )
