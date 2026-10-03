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
from hashlib import sha256
from math import isfinite
from pathlib import Path
import struct
import sys
from time import monotonic
from typing import Any, Callable, Mapping, Protocol, Sequence
from types import ModuleType

from .common import (
    RecoveryContractError,
    canonical_sha256,
    require_sha256,
    validate_raw23_action,
    validate_state61,
)
from .dart_noise import (
    DartInspiredBoundedNoise,
    OriginalGaussianNoise,
    scale_dart_covariance,
)


CANDIDATE_ONLY_FLAGS = {
    "candidate_only": True,
    "training_eligible": False,
    "ready_for_training": False,
    "low_action_supervision_positive": False,
    "authority_minted": False,
    "authority_status": "NO_AUTHORITY",
}
_TEACHER_SHA_FIELDS = (
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


def native_raw23_float32_bytes(action: Sequence[float]) -> bytes:
    """Return the only native R1Pro action representation this collector accepts.

    The DART sampler is deliberately mathematical: its clean target, sampled
    noise, and ``requested_noisy23`` remain Python float / float64 provenance.
    OmniGibson receives a float32 action, however.  This helper makes that
    one permitted representation conversion explicit and rejects finite
    float64 values which cannot be encoded on the native wire.  It is *not*
    a clipping, masking, or controller-substitution allowance.
    """

    values = _action_tuple(action, "raw23 native request")
    try:
        return struct.pack("<23f", *values)
    except OverflowError as exc:
        raise RecoveryContractError("raw23 native request must be representable as float32") from exc


def canonical_native_raw23_float32(action: Sequence[float]) -> tuple[float, ...]:
    """Decode the exact float32 values represented by the native raw23 wire."""

    return tuple(float(value) for value in struct.unpack("<23f", native_raw23_float32_bytes(action)))


def native_raw23_float32_sha256(action: Sequence[float]) -> str:
    """Hash the canonical little-endian float32 native action bytes."""

    return sha256(native_raw23_float32_bytes(action)).hexdigest()


def _require_nonempty_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecoveryContractError(f"{field} must be nonempty text")
    return value


@dataclass(frozen=True)
class CandidateSourceReceipt:
    """TRAIN parent provenance for a *new* DART rollout, never a demo-state ID."""

    source_group_id: str
    original_split: str
    source_release_sha256: str
    parent_task_id: str
    parent_task_index: int
    parent_task_instance_id: int
    parent_task_seed: int
    trajectory_source_kind: str
    collection_run_id: str

    def __post_init__(self) -> None:
        require_sha256(self.source_group_id, field="source_group_id")
        if self.original_split != "train":
            raise RecoveryContractError("DART candidates may only use original train source groups")
        _require_nonempty_text(self.parent_task_id, "parent_task_id")
        if not isinstance(self.parent_task_index, int) or self.parent_task_index < 0:
            raise RecoveryContractError("parent_task_index must be a nonnegative integer")
        if type(self.parent_task_instance_id) is not int or self.parent_task_instance_id < 1:
            raise RecoveryContractError("parent_task_instance_id must be an integer >= 1")
        if not isinstance(self.parent_task_seed, int):
            raise RecoveryContractError("parent_task_seed must be an integer")
        if self.trajectory_source_kind != "fresh_dart_trajectory":
            raise RecoveryContractError("DART candidates must declare trajectory_source_kind=fresh_dart_trajectory")
        _require_nonempty_text(self.collection_run_id, "collection_run_id")
        require_sha256(self.source_release_sha256, field="source_release_sha256")

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
class CanonicalSourceGroupMembership:
    """One immutable source-group fact supplied by the DATA-owned index reader."""

    source_group_id: str
    source_release_sha256: str
    task_index: int
    task_instance_id: int
    original_split: str
    usage_role: str
    source_group_member_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.source_group_id, field="source_group_id")
        if type(self.task_instance_id) is not int or self.task_instance_id < 1:
            raise RecoveryContractError("canonical source membership task_instance_id must be an integer >= 1")
        if not isinstance(self.task_index, int) or self.task_index < 0:
            raise RecoveryContractError("canonical source membership task_index must be nonnegative")
        if self.original_split not in {"train", "eval", "dev", "test", "protected"}:
            raise RecoveryContractError("canonical source membership has an unknown original split")
        if self.usage_role not in {
            "student_candidate", "annotation_calibration", "evaluation_only", "protected_holdout", "quarantined",
        }:
            raise RecoveryContractError("canonical source membership has an unknown usage role")
        for field in ("source_release_sha256", "source_group_member_sha256"):
            require_sha256(getattr(self, field), field=field)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifiedDartSourceMembership:
    """A once-read, sealed membership result returned by the DATA index adapter."""

    index_inventory_seal_sha256: str
    source_group_index_manifest_sha256: str
    source_membership_protocol_sha256: str
    candidate: CanonicalSourceGroupMembership
    calibration_groups: tuple[CanonicalSourceGroupMembership, ...]

    def __post_init__(self) -> None:
        for field in (
            "index_inventory_seal_sha256",
            "source_group_index_manifest_sha256",
            "source_membership_protocol_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        if not isinstance(self.candidate, CanonicalSourceGroupMembership):
            raise RecoveryContractError("verified DART source membership needs a canonical candidate group")
        if not self.calibration_groups or not all(
            isinstance(group, CanonicalSourceGroupMembership) for group in self.calibration_groups
        ):
            raise RecoveryContractError("verified DART source membership needs canonical calibration groups")

    def public(self) -> dict[str, Any]:
        return {
            "index_inventory_seal_sha256": self.index_inventory_seal_sha256,
            "source_group_index_manifest_sha256": self.source_group_index_manifest_sha256,
            "source_membership_protocol_sha256": self.source_membership_protocol_sha256,
            "candidate": self.candidate.public(),
            "calibration_groups": [group.public() for group in self.calibration_groups],
        }


class CanonicalSourceGroupIndexReader(Protocol):
    """DATA-owned sealed-index loader; DART intentionally defines no duplicate format."""

    def verify_dart_membership(
        self, source: CandidateSourceReceipt, calibration: CalibrationReceipt
    ) -> VerifiedDartSourceMembership: ...


def _load_dependency_free_data_protocol(
    protocol_path: Path, *, expected_protocol_source_sha256: str | None
) -> tuple[ModuleType, str]:
    """Execute one SHA-stable DATA protocol source file without importing ``g05.data``.

    ``g05.data.__init__`` is allowed to depend on training datasets and optional
    packages. Candidate provenance must remain usable in a minimal CPU
    environment, so this loader compiles precisely the bytes it hashed rather
    than normal-importing that package. The module is retained in
    ``sys.modules`` under a content-addressed private name so classes created
    during execution have a stable ``__module__``/``__file__`` identity.
    """

    if expected_protocol_source_sha256 is not None:
        require_sha256(expected_protocol_source_sha256, field="expected_protocol_source_sha256")
    raw_path = Path(protocol_path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise RecoveryContractError("DATA source-membership protocol path must be a regular file")
    resolved_path = raw_path.resolve(strict=True)
    try:
        source_bytes = resolved_path.read_bytes()
    except OSError as exc:
        raise RecoveryContractError("could not read DATA source-membership protocol source") from exc
    source_sha256 = sha256(source_bytes).hexdigest()
    if expected_protocol_source_sha256 is not None and source_sha256 != expected_protocol_source_sha256:
        raise RecoveryContractError("DATA source-membership protocol bytes do not match the supplied SHA-256")
    path_identity = sha256(str(resolved_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_p107_dart_data_protocol_{source_sha256}_{path_identity}"
    cached = sys.modules.get(module_name)
    if isinstance(cached, ModuleType) and getattr(cached, "__file__", None) == str(resolved_path):
        return cached, source_sha256
    try:
        compiled = compile(source_bytes, str(resolved_path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise RecoveryContractError("DATA source-membership protocol source cannot be compiled") from exc
    module = ModuleType(module_name, "Dependency-free DART DATA source-membership protocol")
    module.__file__ = str(resolved_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compiled, module.__dict__)
    except Exception as exc:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise RecoveryContractError("DATA source-membership protocol could not be initialized") from exc
    return module, source_sha256


class DataSealedSourceGroupIndexReader:
    """Thin adapter over DATA's externally SHA-pinned, read-only membership API.

    The DATA module owns the index format and performs the seal/manifest/file
    verification exactly once here.  This adapter neither loads corrective
    publisher authority nor invents a parallel source-index format.
    """

    def __init__(
        self,
        index_root: Path,
        *,
        expected_inventory_seal_sha256: str,
        protocol_path: Path | None = None,
        expected_protocol_source_sha256: str | None = None,
    ) -> None:
        require_sha256(expected_inventory_seal_sha256, field="expected_inventory_seal_sha256")
        if protocol_path is not None and expected_protocol_source_sha256 is None:
            raise RecoveryContractError("an external DATA protocol path requires expected_protocol_source_sha256")
        default_protocol_path = Path(__file__).resolve().parent.parent / "data" / "memlite_event_protocol.py"
        protocol, protocol_source_sha256 = _load_dependency_free_data_protocol(
            default_protocol_path if protocol_path is None else Path(protocol_path),
            expected_protocol_source_sha256=expected_protocol_source_sha256,
        )
        load_source_index_membership = getattr(protocol, "load_source_index_membership", None)
        sealed_source_group = getattr(protocol, "sealed_source_group", None)
        if not callable(load_source_index_membership) or not callable(sealed_source_group):
            raise RecoveryContractError("DATA protocol source does not expose the sealed source-membership API")
        try:
            membership = load_source_index_membership(
                Path(index_root), expected_inventory_seal_sha256=expected_inventory_seal_sha256
            )
        except Exception as exc:  # DATA ContractError is intentionally not redefined here.
            raise RecoveryContractError("could not load DATA's externally sealed source-group membership") from exc
        self._source_group_lookup = sealed_source_group
        self._membership = membership
        self._inventory_seal_sha256 = expected_inventory_seal_sha256
        self._protocol_source_sha256 = protocol_source_sha256

    def _group(self, source_group_id: str) -> CanonicalSourceGroupMembership:
        try:
            group = self._source_group_lookup(self._membership, source_group_id)
        except Exception as exc:  # Preserve the DART candidate fail-closed boundary.
            raise RecoveryContractError("source group is absent from DATA's sealed membership index") from exc
        return CanonicalSourceGroupMembership(
            source_group_id=group["source_group_id"],
            source_release_sha256=group["source_release_manifest_sha256"],
            task_index=group["task_index"],
            task_instance_id=group["task_instance_id"],
            original_split=group["original_split"],
            usage_role=group["usage_role"],
            source_group_member_sha256=canonical_sha256(group),
        )

    def verify_dart_membership(
        self, source: CandidateSourceReceipt, calibration: CalibrationReceipt
    ) -> VerifiedDartSourceMembership:
        return VerifiedDartSourceMembership(
            index_inventory_seal_sha256=self._inventory_seal_sha256,
            source_group_index_manifest_sha256=self._membership.index_manifest_sha256,
            source_membership_protocol_sha256=self._protocol_source_sha256,
            candidate=self._group(source.source_group_id),
            calibration_groups=tuple(self._group(group_id) for group_id in calibration.source_group_ids),
        )


@dataclass(frozen=True)
class OriginalGaussianCalibrationBinding:
    """Evidence that this exact sampled covariance came from held-out train calibration."""

    calibration_trajectory_sha256: str
    learner_checkpoint_sha256: str
    teacher_checkpoint_sha256: str
    covariance_estimator_code_sha256: str
    estimated_covariance23: Sequence[Sequence[float]]
    alpha: float
    horizon: int
    source_group_index_manifest_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "calibration_trajectory_sha256",
            "learner_checkpoint_sha256",
            "teacher_checkpoint_sha256",
            "covariance_estimator_code_sha256",
            "source_group_index_manifest_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        # Reuse the exact Eq.4 validation instead of accepting an opaque
        # covariance digest from a caller.  This also rejects nonfinite/PSD
        # failures before any collection is attempted.
        scaled = scale_dart_covariance(self.estimated_covariance23, alpha=self.alpha, horizon=self.horizon)
        object.__setattr__(self, "estimated_covariance23", tuple(tuple(row) for row in self.estimated_covariance23))
        object.__setattr__(self, "_scaled_covariance23", scaled)

    @property
    def scaled_covariance23(self) -> tuple[tuple[float, ...], ...]:
        return self._scaled_covariance23

    def validate(
        self,
        *,
        noise: OriginalGaussianNoise,
        calibration: CalibrationReceipt,
        membership: VerifiedDartSourceMembership,
    ) -> None:
        if (
            self.calibration_trajectory_sha256 != calibration.calibration_trajectory_sha256
            or self.learner_checkpoint_sha256 != calibration.learner_checkpoint_sha256
            or self.teacher_checkpoint_sha256 != calibration.teacher_checkpoint_sha256
        ):
            raise RecoveryContractError("original DART covariance binding does not match calibration artifacts")
        if self.source_group_index_manifest_sha256 != membership.source_group_index_manifest_sha256:
            raise RecoveryContractError("original DART covariance binding does not match verified source-group index")
        if noise.covariance != self.scaled_covariance23:
            raise RecoveryContractError("original DART sampled covariance does not match calibration alpha/horizon binding")

    def public(self) -> dict[str, Any]:
        return {
            "calibration_trajectory_sha256": self.calibration_trajectory_sha256,
            "learner_checkpoint_sha256": self.learner_checkpoint_sha256,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "covariance_estimator_code_sha256": self.covariance_estimator_code_sha256,
            "estimated_covariance_sha256": canonical_sha256(self.estimated_covariance23),
            "scaled_covariance_sha256": canonical_sha256(self.scaled_covariance23),
            "alpha": self.alpha,
            "horizon": self.horizon,
            "source_group_index_manifest_sha256": self.source_group_index_manifest_sha256,
        }


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
        for field in _TEACHER_SHA_FIELDS:
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
    task_instance_id: int
    runtime_build_sha256: str
    asset_config_sha256: str
    reset_load_task_instance_receipt_sha256: str
    snapshot_mode: str = "no_restore_fresh_rollout"
    same_session_snapshot_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.runtime_session_id, "runtime_session_id")
        if type(self.task_instance_id) is not int or self.task_instance_id < 1:
            raise RecoveryContractError("task_instance_id must be an integer >= 1")
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
            "schema": "p107_dart_candidate_collection_result_v2",
            "records": list(self.records),
            "outcome_label": self.outcome_label,
            "stop_reason": self.stop_reason,
            **CANDIDATE_ONLY_FLAGS,
        }


def _assert_candidate_source(
    source: CandidateSourceReceipt,
    calibration: CalibrationReceipt,
    membership_reader: CanonicalSourceGroupIndexReader,
) -> VerifiedDartSourceMembership:
    """Bind caller receipts once to the DATA-owned sealed source-group index."""

    if not callable(getattr(membership_reader, "verify_dart_membership", None)):
        raise RecoveryContractError("DART collection requires a DATA-owned canonical source-group index reader")
    membership = membership_reader.verify_dart_membership(source, calibration)
    if not isinstance(membership, VerifiedDartSourceMembership):
        raise RecoveryContractError("canonical source-group reader must return VerifiedDartSourceMembership")
    candidate = membership.candidate
    expected_candidate = {
        "source_group_id": source.source_group_id,
        "source_release_sha256": source.source_release_sha256,
        "task_index": source.parent_task_index,
        "task_instance_id": source.parent_task_instance_id,
        "original_split": source.original_split,
    }
    if any(getattr(candidate, field) != value for field, value in expected_candidate.items()):
        raise RecoveryContractError("candidate source receipt does not match canonical sealed source-group membership")
    if candidate.original_split != "train" or candidate.usage_role != "student_candidate":
        raise RecoveryContractError("candidate source is not a canonical train student-candidate group")
    calibration_ids = tuple(group.source_group_id for group in membership.calibration_groups)
    if set(calibration_ids) != set(calibration.source_group_ids) or len(calibration_ids) != len(calibration.source_group_ids):
        raise RecoveryContractError("calibration receipt groups do not match canonical sealed source-group membership")
    if source.source_group_id in calibration_ids:
        raise RecoveryContractError("held-out calibration groups cannot also produce DART candidates")
    for group in membership.calibration_groups:
        if group.original_split != "train" or group.usage_role != "annotation_calibration":
            raise RecoveryContractError("DART calibration must use canonical held-out train calibration groups")
        if group.source_release_sha256 != source.source_release_sha256:
            raise RecoveryContractError("candidate and calibration groups must bind the same sealed source release")
    return membership


def _assert_teacher_matches(observation: DartObservation, command: TeacherCommand, intent_bundle_id: str) -> None:
    if command.observed_policy_clock != observation.policy_clock:
        raise RecoveryContractError("teacher target is not from the current reached policy clock")
    if command.observed_state61_sha256 != observation.state61_sha256:
        raise RecoveryContractError("teacher target state fingerprint does not match current reached state")
    if command.observed_observation_sha256 != observation.observation_sha256:
        raise RecoveryContractError("teacher target observation fingerprint does not match current observation")
    if command.intent_bundle_id != intent_bundle_id:
        raise RecoveryContractError("teacher target intent does not match collection intent")


def _profile_receipt_for_original(
    noise: OriginalGaussianNoise, *, calibration_binding: OriginalGaussianCalibrationBinding
) -> dict[str, Any]:
    covariance = [list(row) for row in noise.covariance]
    return {
        "algorithm_id": noise.algorithm_id,
        "paper_equations": ["Eq.2", "Eq.3", "Eq.4"],
        "covariance_sha256": canonical_sha256(covariance),
        "sampling_seed": noise.seed,
        "covariance_update_mode": "frozen_one_pass_partial_dart",
        "covariance_calibration_binding": calibration_binding.public(),
        # This implementation has no iterative collection/covariance loop.
        # No caller-provided string may turn that fact into an empirical claim.
        "empirical_faithfulness": False,
    }


def _profile_receipt_for_bounded(noise: DartInspiredBoundedNoise) -> dict[str, Any]:
    profile = noise.profile
    serialized_profile = {
        "profile_id": profile.profile_id,
        "temporal_rho": profile.temporal_rho,
        "parts": [asdict(part) for part in profile.layout.parts],
        "embodiment_metadata_sha256": profile.layout.embodiment_metadata_sha256,
        "model_projection_manifest_sha256": profile.layout.model_projection_manifest_sha256,
        "rules_by_part": {
            key: asdict(value) if is_dataclass(value) else value
            for key, value in sorted(profile.rules_by_part.items())
        },
    }
    return {
        "algorithm_id": profile.algorithm_id,
        "profile_id": profile.profile_id,
        "profile_sha256": canonical_sha256(serialized_profile),
        # Expose these immutable receipts in addition to hashing the full
        # serialized profile.  A downstream reviewer must not have to trust
        # a caller-supplied profile identifier to locate the native layout
        # and the later 23->27 padding projection.
        "embodiment_metadata_sha256": profile.layout.embodiment_metadata_sha256,
        "model_projection_manifest_sha256": profile.layout.model_projection_manifest_sha256,
        "sampling_seed": noise.seed,
        "safety_bounded": True,
        "original_dart_equivalence": False,
    }


def _applied_receipt(runtime: DartRuntime, requested23: Sequence[float], observation: DartObservation) -> AppliedActionReceipt:
    """Apply one mathematical request after canonicalizing exactly once to native f32.

    ``requested23`` remains unchanged in the caller's DART provenance.  The
    runtime is intentionally given the decoded float32 wire values and must
    echo those exact values plus the hash of the same bytes.  Thus the only
    tolerated difference from the mathematical request is IEEE-754 f64->f32
    representation; clipping, masking, stale replay, and substitution all
    fail closed.
    """

    native_requested = canonical_native_raw23_float32(requested23)
    native_wire_sha256 = native_raw23_float32_sha256(native_requested)
    applied = runtime.apply_raw23(native_requested)
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
    if tuple(applied.applied23) != native_requested:
        raise RecoveryContractError("native raw23 runtime substituted or misrepresented the canonical float32 request")
    if applied.applied_action_bytes_sha256 != native_wire_sha256:
        raise RecoveryContractError("native raw23 runtime receipt hash does not match the canonical float32 request")
    return applied


def _candidate_record(
    *,
    label_kind: str,
    collection_mode: str,
    source: CandidateSourceReceipt,
    verified_membership: VerifiedDartSourceMembership,
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
        "schema": "p107_dart_candidate_v2",
        "label_kind": label_kind,
        "collection_mode": collection_mode,
        "source": source.public(),
        "verified_source_membership": verified_membership.public(),
        "runtime_session": runtime_session.public(),
        "calibration_receipt": calibration.public(),
        "teacher": teacher.public(),
        "teacher_query_receipt_sha256": command.fresh_query_receipt_sha256,
        "observation": observation.public(),
        # This clean target is *one* teacher response at this observation.  It
        # is expressly not a fabricated 32-step expert future.
        "clean_intended23": _json_action(command.clean_intended23),
        "clean_target_steps": 1,
        # Keep the mathematical DART request separate from the float32 values
        # actually submitted to the simulator.  The latter are derived only
        # through canonical IEEE-754 conversion at the native wire boundary.
        "requested_noisy23": _json_action(requested_noisy23),
        "sampled_noise23": _json_action(sampled_noise23),
        "requested_native23": _json_action(canonical_native_raw23_float32(requested_noisy23)),
        "requested_native_action_bytes_sha256": native_raw23_float32_sha256(requested_noisy23),
        "native_action_wire_dtype": "float32_le",
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
        **CANDIDATE_ONLY_FLAGS,
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
    source_membership_reader: CanonicalSourceGroupIndexReader,
    runtime_session: RuntimeSessionReceipt,
    calibration: CalibrationReceipt,
    noise: OriginalGaussianNoise,
    intent_bundle_id: str,
    steps: int,
    covariance_update_mode: str,
    original_calibration_binding: OriginalGaussianCalibrationBinding,
) -> DartCollectionResult:
    """Collect original-DART clean intended labels at states reached by Gaussian control.

    Each loop queries the teacher at the current reached state before sampling
    one noisy execution.  The clean label is not marked as actual execution.
    """

    membership = _assert_candidate_source(source, calibration, source_membership_reader)
    if runtime_session.task_instance_id != source.parent_task_instance_id:
        raise RecoveryContractError("runtime fresh-reset task instance must match candidate source instance")
    if runtime_session.runtime_session_id != source.collection_run_id:
        raise RecoveryContractError("DART fresh rollout collection_run_id must match its runtime session")
    _require_nonempty_text(intent_bundle_id, "intent_bundle_id")
    if not isinstance(steps, int) or steps <= 0:
        raise RecoveryContractError("original DART steps must be positive")
    if covariance_update_mode != "frozen_one_pass_partial_dart":
        raise RecoveryContractError("this candidate collector only records frozen_one_pass_partial_dart")
    if not isinstance(original_calibration_binding, OriginalGaussianCalibrationBinding):
        raise RecoveryContractError("original DART requires an explicit covariance calibration binding")
    original_calibration_binding.validate(noise=noise, calibration=calibration, membership=membership)
    profile = _profile_receipt_for_original(noise, calibration_binding=original_calibration_binding)
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
        applied = _applied_receipt(runtime, sampled.requested_noisy23, observation)
        records.append(
            _candidate_record(
                label_kind="dart_clean_supervisor_feedback",
                collection_mode="original_gaussian_clean_intended_feedback",
                source=source,
                verified_membership=membership,
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
    source_membership_reader: CanonicalSourceGroupIndexReader,
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

    membership = _assert_candidate_source(source, calibration, source_membership_reader)
    if runtime_session.task_instance_id != source.parent_task_instance_id:
        raise RecoveryContractError("runtime fresh-reset task instance must match candidate source instance")
    if runtime_session.runtime_session_id != source.collection_run_id:
        raise RecoveryContractError("DART fresh rollout collection_run_id must match its runtime session")
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
        native_clean = canonical_native_raw23_float32(command.clean_intended23)
        native_requested = canonical_native_raw23_float32(sampled.requested_noisy23)
        if native_requested == native_clean:
            raise RecoveryContractError("DART-inspired noisy injection has no effect after native float32 canonicalization")
        applied = _applied_receipt(runtime, sampled.requested_noisy23, observation)
        if tuple(applied.applied23) == native_clean:
            raise RecoveryContractError("DART-inspired noisy injection did not actually perturb the native applied action")
        executed += 1
        records.append(
            _candidate_record(
                label_kind="dart_inspired_noisy_injection",
                collection_mode="dart_inspired_bounded_actual_clean_recovery",
                source=source,
                verified_membership=membership,
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
            applied = _applied_receipt(runtime, command.clean_intended23, observation)
            executed += 1
            records.append(
                _candidate_record(
                    label_kind="dart_inspired_actual_clean_recovery",
                    collection_mode="dart_inspired_bounded_actual_clean_recovery",
                    source=source,
                    verified_membership=membership,
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
