"""One-environment, candidate-only DART factory for a live R1Pro GRASP rollout.

This is intentionally a *production wiring* module, rather than another
simulator protocol.  It binds the reviewed candidate-only DART contracts to
the pinned BEHAVIOR evaluator's ``BatchedEvaluator(num_envs=1)`` API.  All
OmniGibson / Isaac imports are inside :meth:`DartOgGraspFactory.create`, so a
CPU-only import or contract test cannot initialise a simulator by accident.

The factory does not restore a demonstration state.  A source group selects a
permitted TRAIN task/instance/seed and every collection begins from a fresh
official ``load_batch`` reset.  It remains UNQUALIFIED / candidate-only until
a separately authorised live run supplies the runtime, teacher, RGB and
post-run review evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from .common import RecoveryContractError, canonical_json, canonical_sha256, require_sha256, validate_state61
from .dart_collection import (
    AppliedActionReceipt,
    CalibrationReceipt,
    CANDIDATE_ONLY_FLAGS,
    CandidateSourceReceipt,
    canonical_native_raw23_float32,
    CollectionLimits,
    DataSealedSourceGroupIndexReader,
    DartCollectionResult,
    DartObservation,
    OutcomeEvidenceReceipt,
    OriginalGaussianCalibrationBinding,
    OriginalGaussianNoise,
    native_raw23_float32_sha256,
    DartInspiredBoundedNoise,
    RuntimeSessionReceipt,
    TeacherCommand,
    TeacherReceipt,
    VerifiedDartSourceMembership,
    collect_dart_inspired_actual_clean_recovery,
    collect_original_gaussian_clean_feedback,
)
from .dart_noise import ActionPart, BoundedNoiseProfile, BoundedPartRule, Raw23ActionLayout
from .privileged_pose_grasp import (
    ExistingPoseServoEngine,
    ExistingPrivilegedReaderWorld,
    FreshRolloutRaw23Runtime,
    PrivilegedPoseGraspTeacher,
    RootModelPoseReview,
    SealedGraspBinding,
    raw23_wire_sha256,
)


_SCHEMA = "p107-dart-live-grasp-factory-v1"
_PRIVATE_PUBLICATION_SCHEMA = "p107-dart-live-private-publication-v1"
_PRIVATE_COLLECTION_SCHEMA = "p107-dart-live-collection-provenance-v1"
_PRIVATE_EXECUTION_SCHEMA = "p107-dart-live-execution-provenance-v1"
_CAMERAS = ("left_wrist", "right_wrist", "head")
_EXPECTED_R1PRO_CONTROLLER_INDICES = {
    "base": [0, 1, 2],
    "trunk": [3, 4, 5, 6],
    "arm_left": [7, 8, 9, 10, 11, 12, 13],
    "gripper_left": [14],
    "arm_right": [15, 16, 17, 18, 19, 20, 21],
    "gripper_right": [22],
}
_EXPECTED_CONTROLLER_CLASSES = {
    "base": "HolonomicBaseJointController",
    "trunk": "JointController",
    "arm_left": "JointController",
    "gripper_left": "MultiFingerGripperController",
    "arm_right": "JointController",
    "gripper_right": "MultiFingerGripperController",
}


def _sha256_file(path: Path, *, field: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise RecoveryContractError(f"{field} must be a regular file")
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RecoveryContractError(f"could not read {field}") from exc


def _require_git_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise RecoveryContractError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise RecoveryContractError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryContractError(f"{field} must be a finite number") from exc
    if not isfinite(result):
        raise RecoveryContractError(f"{field} must be a finite number")
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryContractError(f"{field} must be a mapping")
    return value


@dataclass(frozen=True)
class DartOgFactoryPins:
    """Exact source/config facts needed before constructing a live session."""

    og_source_commit: str
    evaluator_config_path: Path
    evaluator_config_sha256: str
    r1pro_config_path: Path
    r1pro_config_sha256: str
    runtime_build_sha256: str
    asset_config_sha256: str

    def __post_init__(self) -> None:
        _require_git_sha(self.og_source_commit, field="og_source_commit")
        for field in ("evaluator_config_sha256", "r1pro_config_sha256", "runtime_build_sha256", "asset_config_sha256"):
            require_sha256(getattr(self, field), field=field)
        if _sha256_file(Path(self.evaluator_config_path), field="evaluator_config_path") != self.evaluator_config_sha256:
            raise RecoveryContractError("evaluator config bytes do not match evaluator_config_sha256")
        if _sha256_file(Path(self.r1pro_config_path), field="r1pro_config_path") != self.r1pro_config_sha256:
            raise RecoveryContractError("r1pro config bytes do not match r1pro_config_sha256")


@dataclass(frozen=True)
class FreshRuntimeSessionPlan:
    """Pre-reset runtime facts; the reset receipt can only be minted live."""

    runtime_session_id: str
    task_instance_id: int
    runtime_build_sha256: str
    asset_config_sha256: str
    snapshot_mode: str = "no_restore_fresh_rollout"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_session_id, str) or not self.runtime_session_id:
            raise RecoveryContractError("fresh runtime session plan needs a nonempty session ID")
        if type(self.task_instance_id) is not int or self.task_instance_id < 1:
            raise RecoveryContractError("fresh runtime session plan needs a positive task instance ID")
        for field in ("runtime_build_sha256", "asset_config_sha256"):
            require_sha256(getattr(self, field), field=field)
        if self.snapshot_mode != "no_restore_fresh_rollout":
            raise RecoveryContractError("live DART factory supports only no_restore_fresh_rollout")


@dataclass(frozen=True)
class TrainSourceSelection:
    """The explicit TRAIN instance a fresh official evaluator must load.

    ``CandidateSourceReceipt`` deliberately contains immutable source-group
    identifiers rather than simulator presentation names.  This narrow
    runtime binding supplies the source-index-resolved task and scene names
    while making every opaque receipt field agree before an evaluator is
    constructed.  In particular, a caller cannot set ``train=True`` while
    leaving ``BatchedEvaluator`` on its ``public_test`` default.
    """

    source_group_id: str
    source_release_sha256: str
    parent_task_id: str
    task_name: str
    scene_name: str
    task_instance_id: int
    task_seed: int
    source_selection_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "source_group_id",
            "source_release_sha256",
            "source_selection_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        for field in ("parent_task_id", "task_name", "scene_name"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise RecoveryContractError(f"{field} must be nonempty text")
        if type(self.task_instance_id) is not int or self.task_instance_id < 1:
            raise RecoveryContractError("task_instance_id must be an integer >= 1")
        if type(self.task_seed) is not int:
            raise RecoveryContractError("task_seed must be an integer")

    def assert_source(self, source: CandidateSourceReceipt) -> None:
        if not isinstance(source, CandidateSourceReceipt):
            raise RecoveryContractError("TRAIN selection needs CandidateSourceReceipt")
        if (
            self.source_group_id != source.source_group_id
            or self.source_release_sha256 != source.source_release_sha256
            or self.parent_task_id != source.parent_task_id
            or self.task_instance_id != source.parent_task_instance_id
            or self.task_seed != source.parent_task_seed
        ):
            raise RecoveryContractError("TRAIN selection differs from the sealed DART candidate source")


class PinnedSingleEnvNativeReaderView:
    """Exact scalar view required by the legacy native privileged reader.

    The pinned evaluator owns a vectorized wrapper.  Its current
    ``InstanceEnvAccessor`` provides the selected scene, robot and object
    scope, while the existing reader takes scalar ``env.robots``, ``env.scene``
    and ``env.task.object_scope`` fields.  This is an explicit num-envs-one
    projection of those exact objects -- not a permissive wrapper fallback.
    """

    def __init__(self, *, robot: Any, scene: Any, object_scope: Any) -> None:
        if robot is None or scene is None or not isinstance(object_scope, Mapping):
            raise RecoveryContractError("pinned evaluator lacks selected robot/scene/object scope for native reader")
        self.robots = [robot]
        self.scene = scene
        self.task = SimpleNamespace(object_scope=object_scope)

    @classmethod
    def from_evaluator(cls, evaluator: Any) -> "PinnedSingleEnvNativeReaderView":
        if getattr(evaluator, "num_envs", None) != 1:
            raise RecoveryContractError("native privileged reader only supports pinned evaluator num_envs=1")
        states = getattr(evaluator, "instance_eval_states", None)
        if not isinstance(states, Sequence) or len(states) != 1:
            raise RecoveryContractError("native privileged reader requires exactly one evaluator instance state")
        accessor = getattr(states[0], "env_accessor", None)
        robot, scene = getattr(accessor, "robot", None), getattr(accessor, "scene", None)
        shared_env = getattr(evaluator, "env", None)
        task = getattr(shared_env, "task", None)
        scopes = getattr(task, "object_scopes", None)
        try:
            scope = scopes[0]
        except (KeyError, TypeError, IndexError) as exc:
            raise RecoveryContractError("pinned evaluator task lacks object_scopes[0]") from exc
        scenes = getattr(shared_env, "scenes", None)
        try:
            selected_scene = scenes[0]
            selected_robot = selected_scene.robots[0]
        except (AttributeError, KeyError, TypeError, IndexError) as exc:
            raise RecoveryContractError("pinned evaluator lacks scenes[0].robots[0]") from exc
        if scene is not selected_scene or robot is not selected_robot:
            raise RecoveryContractError("evaluator accessor is not bound to selected env0 scene/robot")
        return cls(robot=robot, scene=scene, object_scope=scope)


@dataclass(frozen=True)
class CapturedRgbView:
    """Private byte payload for one current RGB view.

    Bytes are intentionally retained only for the supplied episode writer.
    ``public`` returns hashes and dimensions, never pixels or a path.
    """

    camera_id: str
    camera_key: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes

    def __post_init__(self) -> None:
        if self.camera_id not in _CAMERAS:
            raise RecoveryContractError("unexpected R1Pro camera role")
        if not isinstance(self.camera_key, str) or not self.camera_key:
            raise RecoveryContractError("captured RGB camera_key must be nonempty text")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise RecoveryContractError("RGB dtype must be explicit")
        if not self.shape or any(type(item) is not int or item <= 0 for item in self.shape):
            raise RecoveryContractError("RGB shape must be positive integer dimensions")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise RecoveryContractError("RGB payload must be nonempty bytes")

    @property
    def sha256(self) -> str:
        return sha256(self.payload).hexdigest()

    def public(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "camera_key": self.camera_key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "bytes": len(self.payload),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CapturedLiveObservation:
    """One observed RGB/proprio state plus private writer-only image bytes."""

    dart: DartObservation
    sim_step: int
    sim_time_seconds: float
    rgb: tuple[CapturedRgbView, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dart, DartObservation):
            raise RecoveryContractError("captured observation needs a DartObservation")
        if type(self.sim_step) is not int or self.sim_step < 0:
            raise RecoveryContractError("sim_step must be a nonnegative integer")
        _as_float(self.sim_time_seconds, field="sim_time_seconds")
        if tuple(item.camera_id for item in self.rgb) != _CAMERAS:
            raise RecoveryContractError("captured observation needs each R1Pro camera exactly once")

    def writer_payload(self) -> dict[str, object]:
        """Private writer input, deliberately separate from actor/public records."""

        return {
            "policy_clock": self.dart.policy_clock,
            "state61": list(self.dart.state61),
            "state61_sha256": self.dart.state61_sha256,
            "observation_sha256": self.dart.observation_sha256,
            "sim_step": self.sim_step,
            "sim_time_seconds": self.sim_time_seconds,
            "rgb": [
                {**item.public(), "payload": item.payload}
                for item in self.rgb
            ],
            "private_writer_only": True,
            "actor_visible": False,
        }


@dataclass(frozen=True)
class CapturedLiveTransition:
    """Private media/clock bridge for one exact raw23 simulator transition."""

    pre: CapturedLiveObservation
    post: CapturedLiveObservation
    applied: AppliedActionReceipt

    def __post_init__(self) -> None:
        if self.post.dart.policy_clock != self.pre.dart.policy_clock + 1:
            raise RecoveryContractError("live transition must advance policy clock by exactly one")
        if self.post.sim_step <= self.pre.sim_step:
            raise RecoveryContractError("live transition must advance simulator step")
        if self.post.sim_time_seconds <= self.pre.sim_time_seconds:
            raise RecoveryContractError("live transition must advance simulator time")
        if self.applied.pre_action_policy_clock != self.pre.dart.policy_clock:
            raise RecoveryContractError("action receipt must bind pre-action policy clock")
        if self.applied.pre_action_state61_sha256 != self.pre.dart.state61_sha256:
            raise RecoveryContractError("action receipt must bind pre-action state")
        if self.applied.pre_action_observation_sha256 != self.pre.dart.observation_sha256:
            raise RecoveryContractError("action receipt must bind pre-action RGB/proprio")

    def writer_payload(self) -> dict[str, object]:
        return {
            "pre": self.pre.writer_payload(),
            "post": self.post.writer_payload(),
            "a_executed_raw23": list(self.applied.applied23),
            "a_executed_raw23_wire_sha256": self.applied.applied_action_bytes_sha256,
            "runtime_step_receipt_sha256": self.applied.runtime_step_receipt_sha256,
            "private_writer_only": True,
            "actor_visible": False,
            "training_eligible": False,
        }


def _preflight_private_publication_output(value: Path | str) -> Path:
    """Reject an unsafe or occupied destination before touching live runtime."""

    if not isinstance(value, (str, Path)):
        raise RecoveryContractError("private DART bundle output must be a path")
    output = Path(value)
    if output.is_symlink() or not output.name or output.parent.is_symlink() or not output.parent.is_dir():
        raise RecoveryContractError("private DART bundle output must name a non-symlink path")
    # Keep the equivalent publication-time guard as protection against a race,
    # but an already-occupied path is a pure CPU/configuration error and must
    # not cause evaluator construction, reset, or a live control step.
    if os.path.lexists(output):
        raise RecoveryContractError("refusing to overwrite an existing private DART publication path")
    return output


@dataclass(frozen=True)
class PrivateEpisodeWriterConfig:
    """Pins for the DATA-owned private episode bundle writer.

    Actor-safe proprioception is intentionally ``None`` here.  The evaluator's
    61-vector is retained only in an in-process :class:`DartObservation` and
    has no reviewed actor projection in this ticket.  Supplying it as a handy
    generic vector would let private simulator state slip into an actor file.
    """

    # This is the fresh outer publication root, not the inner episode bundle.
    # The factory atomically publishes both that bundle and collection
    # provenance under this one root.
    output: Path
    source_episode_id: str
    source: CandidateSourceReceipt
    verified_membership: VerifiedDartSourceMembership
    runtime_session: RuntimeSessionReceipt
    teacher: TeacherReceipt
    collector_code_sha256: str
    capture_adapter_sha256: str
    actor_observation_schema_sha256: str
    outcome_evidence: OutcomeEvidenceReceipt | None = None
    fresh_reset_payload: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        output = _preflight_private_publication_output(self.output)
        object.__setattr__(self, "output", output)
        if not isinstance(self.source_episode_id, str) or not self.source_episode_id:
            raise RecoveryContractError("private DART source_episode_id must be nonempty text")
        if not isinstance(self.source, CandidateSourceReceipt):
            raise RecoveryContractError("private DART writer needs CandidateSourceReceipt")
        if not isinstance(self.verified_membership, VerifiedDartSourceMembership):
            raise RecoveryContractError("private DART writer needs verified source membership")
        if not isinstance(self.runtime_session, RuntimeSessionReceipt):
            raise RecoveryContractError("private DART writer needs runtime session receipt")
        if not isinstance(self.teacher, TeacherReceipt):
            raise RecoveryContractError("private DART writer needs teacher receipt")
        if self.teacher.teacher_kind != "privileged_pose_grasp_unqualified":
            raise RecoveryContractError("private live DART writer only accepts the explicitly unqualified GRASP teacher")
        for field in ("collector_code_sha256", "capture_adapter_sha256", "actor_observation_schema_sha256"):
            require_sha256(getattr(self, field), field=field)
        if self.runtime_session.runtime_session_id != self.source.collection_run_id:
            raise RecoveryContractError("private DART writer runtime run_id differs from candidate source")
        if self.runtime_session.task_instance_id != self.source.parent_task_instance_id:
            raise RecoveryContractError("private DART writer runtime instance differs from candidate source")
        candidate = self.verified_membership.candidate
        if (
            candidate.source_group_id != self.source.source_group_id
            or candidate.source_release_sha256 != self.source.source_release_sha256
            or candidate.task_index != self.source.parent_task_index
            or candidate.task_instance_id != self.source.parent_task_instance_id
            or candidate.original_split != "train"
            or candidate.usage_role != "student_candidate"
        ):
            raise RecoveryContractError("private DART writer source is not sealed TRAIN student_candidate membership")
        if self.outcome_evidence is not None:
            if not isinstance(self.outcome_evidence, OutcomeEvidenceReceipt):
                raise RecoveryContractError("private DART writer outcome evidence must be explicit")
            if self.outcome_evidence.label != "OUTCOME_UNKNOWN":
                raise RecoveryContractError("live factory has no official/local outcome verifier; only explicit UNKNOWN is allowed")
        if self.fresh_reset_payload is not None and not isinstance(self.fresh_reset_payload, Mapping):
            raise RecoveryContractError("private DART fresh reset payload must be an object when supplied")


@dataclass(frozen=True)
class PrivateEpisodePublicationReceipt:
    """Receipt for one immutable outer private DART publication root."""

    path: Path
    manifest_sha256: str
    episode_bundle_path: Path
    episode_manifest_sha256: str
    collection_result_sha256: str
    transition_count: int
    image_count: int
    training_eligible: bool = False
    release_eligible: bool = False


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _file_receipt(path: Path, *, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryContractError(f"{field} must be a regular file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RecoveryContractError(f"could not read {field}") from exc
    return {"sha256": sha256(payload).hexdigest(), "bytes": len(payload)}


def _require_private_candidate_gates(value: Mapping[str, object], *, field: str) -> None:
    for name, expected in CANDIDATE_ONLY_FLAGS.items():
        if value.get(name) != expected:
            raise RecoveryContractError(f"{field} lost a candidate-only release gate")


def _collection_execution_provenance(
    *,
    mode: str,
    original_noise: OriginalGaussianNoise | None = None,
    original_steps: int | None = None,
    original_binding: OriginalGaussianCalibrationBinding | None = None,
    bounded_noise: DartInspiredBoundedNoise | None = None,
    noisy_injection_steps: int | None = None,
    recovery_chunk_lengths: Sequence[int] | None = None,
    limits: CollectionLimits | None = None,
) -> dict[str, object]:
    """Seal complete collection controls outside the actor-visible bundle.

    Per-step records intentionally retain only the compact noise-profile
    receipt used by the collector.  The factory-owned private sidecar also
    needs the complete bounded profile (including every inactive rule), or
    the full original covariance and calibration binding, to make a later
    candidate review reproducible without treating a mode string as proof.
    """

    if mode == "original_gaussian_one_pass_partial_dart":
        if original_noise is None or original_steps is None or original_binding is None:
            raise RecoveryContractError("original DART execution provenance is incomplete")
        return {
            "schema": _PRIVATE_EXECUTION_SCHEMA,
            "factory_mode": mode,
            "record_collection_mode": "original_gaussian_clean_intended_feedback",
            "original_gaussian": {
                "algorithm_id": original_noise.algorithm_id,
                "sampling_seed": original_noise.seed,
                "covariance23": [list(row) for row in original_noise.covariance],
                "covariance23_sha256": canonical_sha256(original_noise.covariance),
                "steps": original_steps,
                "covariance_update_mode": "frozen_one_pass_partial_dart",
                "calibration_binding": original_binding.public(),
            },
        }
    if mode == "dart_inspired_bounded_actual_clean_recovery":
        if (
            bounded_noise is None
            or noisy_injection_steps is None
            or recovery_chunk_lengths is None
            or limits is None
        ):
            raise RecoveryContractError("bounded DART execution provenance is incomplete")
        profile = bounded_noise.profile
        return {
            "schema": _PRIVATE_EXECUTION_SCHEMA,
            "factory_mode": mode,
            "record_collection_mode": "dart_inspired_bounded_actual_clean_recovery",
            "bounded_dart": {
                "algorithm_id": profile.algorithm_id,
                "sampling_seed": bounded_noise.seed,
                "profile": {
                    "profile_id": profile.profile_id,
                    "temporal_rho": profile.temporal_rho,
                    "layout": asdict(profile.layout),
                    "rules_by_part": {
                        part.name: asdict(profile.rules_by_part[part.name]) for part in profile.layout.parts
                    },
                },
                "noisy_injection_steps": noisy_injection_steps,
                "recovery_chunk_lengths": list(recovery_chunk_lengths),
                "limits": asdict(limits),
            },
        }
    raise RecoveryContractError("unknown live DART collection mode for private execution provenance")


def _validate_collection_execution_provenance(
    value: object, *, records: Sequence[Mapping[str, object]]
) -> Mapping[str, object]:
    """Reject sidecars that omit a collection's real control parameters."""

    execution = _mapping(value, field="private DART execution provenance")
    if execution.get("schema") != _PRIVATE_EXECUTION_SCHEMA:
        raise RecoveryContractError("private DART execution provenance has an unknown schema")
    record_modes = {record.get("collection_mode") for record in records}
    if len(record_modes) != 1 or not all(isinstance(item, str) for item in record_modes):
        raise RecoveryContractError("private DART execution provenance needs one concrete record collection mode")
    record_mode = next(iter(record_modes))
    if execution.get("record_collection_mode") != record_mode:
        raise RecoveryContractError("private DART execution provenance does not match collection records")
    factory_mode = execution.get("factory_mode")
    if factory_mode == "original_gaussian_one_pass_partial_dart":
        original = _mapping(execution.get("original_gaussian"), field="private original Gaussian provenance")
        if record_mode != "original_gaussian_clean_intended_feedback":
            raise RecoveryContractError("original DART execution provenance has the wrong record mode")
        if original.get("algorithm_id") != "dart_original_gaussian_unbounded":
            raise RecoveryContractError("original DART execution provenance has the wrong algorithm")
        covariance = original.get("covariance23")
        if (
            not isinstance(covariance, list)
            or len(covariance) != 23
            or any(not isinstance(row, list) or len(row) != 23 for row in covariance)
            or original.get("covariance23_sha256") != canonical_sha256(covariance)
        ):
            raise RecoveryContractError("original DART execution provenance covariance is invalid")
        if type(original.get("sampling_seed")) is not int or type(original.get("steps")) is not int or original["steps"] < 1:
            raise RecoveryContractError("original DART execution provenance seed or step bound is invalid")
        try:
            # Constructor validation covers finite, symmetric, PSD 23x23
            # covariance values rather than merely accepting a matching hash.
            OriginalGaussianNoise(covariance, seed=original["sampling_seed"])
        except RecoveryContractError as exc:
            raise RecoveryContractError("original DART execution provenance covariance is not a valid sampler input") from exc
        if original.get("covariance_update_mode") != "frozen_one_pass_partial_dart":
            raise RecoveryContractError("original DART execution provenance lost its bounded classification")
        binding = _mapping(original.get("calibration_binding"), field="private original Gaussian calibration binding")
        for field in (
            "calibration_trajectory_sha256",
            "learner_checkpoint_sha256",
            "teacher_checkpoint_sha256",
            "covariance_estimator_code_sha256",
            "estimated_covariance_sha256",
            "scaled_covariance_sha256",
            "source_group_index_manifest_sha256",
        ):
            require_sha256(binding.get(field), field=f"private original Gaussian calibration binding.{field}")
        if not isinstance(binding.get("alpha"), (int, float)) or type(binding.get("horizon")) is not int:
            raise RecoveryContractError("private original Gaussian calibration binding alpha/horizon is invalid")
        for record in records:
            profile = _mapping(record.get("noise_profile"), field="private original Gaussian record noise profile")
            if (
                profile.get("algorithm_id") != original["algorithm_id"]
                or profile.get("covariance_sha256") != canonical_sha256(covariance)
                or profile.get("sampling_seed") != original["sampling_seed"]
                or profile.get("covariance_calibration_binding") != binding
            ):
                raise RecoveryContractError("original DART execution provenance differs from collection record")
        return execution
    if factory_mode == "dart_inspired_bounded_actual_clean_recovery":
        bounded = _mapping(execution.get("bounded_dart"), field="private bounded DART provenance")
        if record_mode != "dart_inspired_bounded_actual_clean_recovery":
            raise RecoveryContractError("bounded DART execution provenance has the wrong record mode")
        if bounded.get("algorithm_id") != "dart_inspired_bounded_correlated":
            raise RecoveryContractError("bounded DART execution provenance has the wrong algorithm")
        profile = _mapping(bounded.get("profile"), field="private bounded DART profile")
        layout = _mapping(profile.get("layout"), field="private bounded DART action layout")
        parts = layout.get("parts")
        rules = _mapping(profile.get("rules_by_part"), field="private bounded DART part rules")
        if (
            not isinstance(parts, (list, tuple))
            or not parts
            or any(not isinstance(part, Mapping) or not isinstance(part.get("name"), str) for part in parts)
            or set(rules) != {str(part["name"]) for part in parts}
        ):
            raise RecoveryContractError("bounded DART execution provenance lacks complete layout/rules")
        if not isinstance(profile.get("temporal_rho"), (int, float)) or not 0 <= float(profile["temporal_rho"]) < 1:
            raise RecoveryContractError("bounded DART execution provenance temporal rho is invalid")
        chunks = bounded.get("recovery_chunk_lengths")
        limits = _mapping(bounded.get("limits"), field="private bounded DART limits")
        if (
            type(bounded.get("sampling_seed")) is not int
            or type(bounded.get("noisy_injection_steps")) is not int
            or bounded["noisy_injection_steps"] < 1
            or not isinstance(chunks, list)
            or not chunks
            or any(type(chunk) is not int or not 1 <= chunk <= 16 for chunk in chunks)
            or type(limits.get("max_total_steps")) is not int
            or limits["max_total_steps"] < bounded["noisy_injection_steps"] + sum(chunks)
            or not isinstance(limits.get("max_wall_seconds"), (int, float))
        ):
            raise RecoveryContractError("bounded DART execution provenance injection/chunk/limits are invalid")
        try:
            layout_obj = Raw23ActionLayout(
                parts=tuple(ActionPart(**dict(part)) for part in parts),
                embodiment_metadata_sha256=layout.get("embodiment_metadata_sha256"),
                model_projection_manifest_sha256=layout.get("model_projection_manifest_sha256"),
            )
            profile_obj = BoundedNoiseProfile(
                layout=layout_obj,
                rules_by_part={name: BoundedPartRule(**dict(rule)) for name, rule in rules.items()},
                temporal_rho=float(profile["temporal_rho"]),
                profile_id=profile.get("profile_id"),
            )
            DartInspiredBoundedNoise(profile_obj, seed=bounded["sampling_seed"])
            CollectionLimits(**dict(limits))
        except (RecoveryContractError, TypeError, ValueError) as exc:
            raise RecoveryContractError("bounded DART execution provenance cannot reconstruct its sampler") from exc
        for record in records:
            receipt = _mapping(record.get("noise_profile"), field="private bounded DART record noise profile")
            if (
                receipt.get("algorithm_id") != profile_obj.algorithm_id
                or receipt.get("profile_id") != profile_obj.profile_id
                or receipt.get("sampling_seed") != bounded["sampling_seed"]
                or receipt.get("embodiment_metadata_sha256") != layout_obj.embodiment_metadata_sha256
                or receipt.get("model_projection_manifest_sha256") != layout_obj.model_projection_manifest_sha256
            ):
                raise RecoveryContractError("bounded DART execution provenance differs from collection record")
        return execution
    raise RecoveryContractError("private DART execution provenance has an unsupported factory mode")


def _validated_fresh_reset_payload(
    value: Mapping[str, object] | None,
    *,
    records: Sequence[Mapping[str, object]],
    transitions: Sequence[CapturedLiveTransition],
) -> Mapping[str, object] | None:
    """Bind a minted post-reset capture to the first published transition."""

    if value is None:
        return None
    payload = _mapping(value, field="private DART fresh reset payload")
    if payload.get("schema") != "p107-dart-live-fresh-reset-v1":
        raise RecoveryContractError("private DART fresh reset payload has an unknown schema")
    if not records or not transitions:
        raise RecoveryContractError("private DART fresh reset payload needs a first captured transition")
    runtime = _mapping(records[0].get("runtime_session"), field="private DART fresh reset runtime session")
    if payload.get("runtime_session_id") != runtime.get("runtime_session_id"):
        raise RecoveryContractError("private DART fresh reset payload has the wrong runtime session")
    if runtime.get("reset_load_task_instance_receipt_sha256") != canonical_sha256(payload):
        raise RecoveryContractError("private DART fresh reset payload hash differs from runtime session receipt")
    initial = _mapping(payload.get("initial_observation"), field="private DART fresh reset initial observation")
    pre = transitions[0].pre
    expected_initial = {
        "policy_clock": pre.dart.policy_clock,
        "state61_sha256": pre.dart.state61_sha256,
        "observation_sha256": pre.dart.observation_sha256,
        "simulation_tick": pre.sim_step,
        "simulation_time_seconds": pre.sim_time_seconds,
        "views": [view.public() for view in pre.rgb],
    }
    if dict(initial) != expected_initial:
        raise RecoveryContractError("private DART fresh reset payload is not the first captured observation")
    return payload


def _private_collection_payload(
    result: DartCollectionResult,
    transitions: tuple[CapturedLiveTransition, ...],
    *,
    collection_execution_provenance: Mapping[str, object] | None,
    fresh_reset_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    """Preserve collection math/provenance without exporting it to actor rows."""

    public = result.public()
    if not isinstance(public, Mapping) or public.get("outcome_label") != "OUTCOME_UNKNOWN":
        raise RecoveryContractError("private collection sidecar permits only UNKNOWN outcome evidence")
    _require_private_candidate_gates(public, field="private collection result")
    records = public.get("records")
    if not isinstance(records, list) or len(records) != len(transitions) or not records:
        raise RecoveryContractError("private collection sidecar needs cardinality-matched nonempty DART records")
    for record in records:
        if not isinstance(record, Mapping):
            raise RecoveryContractError("private collection sidecar record must be an object")
        _require_private_candidate_gates(record, field="private collection record")
        # These are the immutable receipts needed to reproduce whether the
        # run was original Gaussian or bounded DART-inspired.  A mode string
        # alone cannot substitute for either record.
        if not isinstance(record.get("noise_profile"), Mapping) or not isinstance(record.get("calibration_receipt"), Mapping):
            raise RecoveryContractError("private collection sidecar requires noise and calibration provenance on every record")
    execution = _validate_collection_execution_provenance(collection_execution_provenance, records=records)
    fresh_reset = _validated_fresh_reset_payload(
        fresh_reset_payload, records=records, transitions=transitions
    )
    transition_receipts = []
    for transition in transitions:
        transition_receipts.append(
            {
                "runtime_step_receipt_sha256": transition.applied.runtime_step_receipt_sha256,
                "applied_action_bytes_sha256": transition.applied.applied_action_bytes_sha256,
                "pre_observation_sha256": transition.pre.dart.observation_sha256,
                "post_observation_sha256": transition.post.dart.observation_sha256,
                "pre_policy_clock": transition.pre.dart.policy_clock,
                "post_policy_clock": transition.post.dart.policy_clock,
                "pre_simulation_tick": transition.pre.sim_step,
                "post_simulation_tick": transition.post.sim_step,
            }
        )
    payload: dict[str, object] = {
        "schema": _PRIVATE_COLLECTION_SCHEMA,
        "status": "PRIVATE_DART_CANDIDATE_ONLY",
        **CANDIDATE_ONLY_FLAGS,
        "collection_result": public,
        "collection_result_sha256": canonical_sha256(public),
        "collection_execution_provenance": dict(execution),
        "collection_execution_provenance_sha256": canonical_sha256(execution),
        "captured_transition_receipts": transition_receipts,
    }
    if fresh_reset is not None:
        payload["fresh_reset_payload"] = dict(fresh_reset)
        payload["fresh_reset_payload_sha256"] = canonical_sha256(fresh_reset)
    return payload


def load_private_episode_publication(
    path: Path, *, expected_manifest_sha256: str | None = None
) -> PrivateEpisodePublicationReceipt:
    """Fail-closed reader for the factory-owned outer publication manifest."""

    try:
        from .dart_dataset import load_bundle
    except ImportError as exc:
        raise RecoveryContractError("private DART dataset reader module is unavailable") from exc
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise RecoveryContractError("private DART publication root must be a regular directory")
    manifest_path = root / "publication_manifest.json"
    receipt = _file_receipt(manifest_path, field="private DART publication manifest")
    actual_sha = str(receipt["sha256"])
    if expected_manifest_sha256 is not None and actual_sha != require_sha256(
        expected_manifest_sha256, field="expected private publication manifest SHA-256"
    ):
        raise RecoveryContractError("private DART publication manifest SHA-256 does not match expected pin")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryContractError("private DART publication manifest is not readable canonical JSON") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != _PRIVATE_PUBLICATION_SCHEMA:
        raise RecoveryContractError("private DART publication manifest has an unknown schema")
    if manifest.get("status") != "PRIVATE_DART_CANDIDATE_ONLY":
        raise RecoveryContractError("private DART publication has an invalid release status")
    _require_private_candidate_gates(manifest, field="private DART publication manifest")
    if manifest.get("training_eligible") is not False or manifest.get("release_eligible") is not False:
        raise RecoveryContractError("private DART publication cannot be training or release eligible")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {"episode/manifest.json", "private_collection.json"}:
        raise RecoveryContractError("private DART publication must seal exactly episode manifest and collection provenance")
    for relative, expected in files.items():
        if not isinstance(relative, str) or "/../" in f"/{relative}" or relative.startswith("/"):
            raise RecoveryContractError("private DART publication manifest has an unsafe relative path")
        if not isinstance(expected, Mapping):
            raise RecoveryContractError("private DART publication file receipt must be an object")
        actual = _file_receipt(root / relative, field=f"private DART publication file {relative}")
        if actual != dict(expected):
            raise RecoveryContractError("private DART publication file receipt does not match manifest")
    episode = manifest.get("episode_bundle")
    collection = manifest.get("private_collection")
    if not isinstance(episode, Mapping) or not isinstance(collection, Mapping):
        raise RecoveryContractError("private DART publication is missing episode or collection provenance binding")
    if episode.get("relative_path") != "episode" or episode.get("manifest_sha256") != files["episode/manifest.json"]["sha256"]:
        raise RecoveryContractError("private DART publication episode binding disagrees with sealed manifest")
    loaded = load_bundle(root / "episode", expected_manifest_sha256=episode.get("manifest_sha256"))
    if collection.get("relative_path") != "private_collection.json" or collection.get("sha256") != files["private_collection.json"]["sha256"]:
        raise RecoveryContractError("private DART publication collection binding disagrees with sealed manifest")
    try:
        sidecar = json.loads((root / "private_collection.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryContractError("private DART collection provenance sidecar is not readable canonical JSON") from exc
    if not isinstance(sidecar, Mapping) or sidecar.get("schema") != _PRIVATE_COLLECTION_SCHEMA:
        raise RecoveryContractError("private DART collection provenance sidecar has an unknown schema")
    if sidecar.get("status") != "PRIVATE_DART_CANDIDATE_ONLY":
        raise RecoveryContractError("private DART collection provenance has an invalid release status")
    _require_private_candidate_gates(sidecar, field="private DART collection provenance")
    result = sidecar.get("collection_result")
    if not isinstance(result, Mapping) or sidecar.get("collection_result_sha256") != canonical_sha256(result):
        raise RecoveryContractError("private DART collection provenance result hash is invalid")
    if result.get("outcome_label") != "OUTCOME_UNKNOWN":
        raise RecoveryContractError("private DART collection provenance cannot claim a physical outcome")
    _require_private_candidate_gates(result, field="private DART collection result")
    if collection.get("collection_result_sha256") != sidecar["collection_result_sha256"]:
        raise RecoveryContractError("private DART collection provenance hash disagrees with outer manifest")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != len(loaded.rows):
        raise RecoveryContractError("private DART collection provenance record count differs from episode bundle")
    typed_records = [_mapping(record, field="private DART collection provenance record") for record in records]
    execution = _validate_collection_execution_provenance(
        sidecar.get("collection_execution_provenance"), records=typed_records
    )
    if sidecar.get("collection_execution_provenance_sha256") != canonical_sha256(execution):
        raise RecoveryContractError("private DART execution provenance hash is invalid")
    if collection.get("collection_execution_provenance_sha256") != sidecar["collection_execution_provenance_sha256"]:
        raise RecoveryContractError("private DART execution provenance hash disagrees with outer manifest")
    captured_receipts = sidecar.get("captured_transition_receipts")
    if not isinstance(captured_receipts, list) or len(captured_receipts) != len(loaded.rows):
        raise RecoveryContractError("private DART collection provenance lacks one receipt per captured transition")
    for row, captured in zip(loaded.rows, captured_receipts):
        if not isinstance(captured, Mapping):
            raise RecoveryContractError("private DART captured transition receipt must be an object")
        pre = _mapping(row.get("pre_observation"), field="private DART episode pre observation")
        post = _mapping(row.get("post_observation"), field="private DART episode post observation")
        applied = _mapping(row.get("applied"), field="private DART episode applied action")
        clock = _mapping(row.get("clock"), field="private DART episode clock")
        expected_captured = {
            "runtime_step_receipt_sha256": applied.get("runtime_step_receipt_sha256"),
            "applied_action_bytes_sha256": applied.get("applied_action_bytes_sha256"),
            "pre_observation_sha256": pre.get("observation_sha256"),
            "post_observation_sha256": post.get("observation_sha256"),
            "pre_policy_clock": pre.get("policy_clock"),
            "post_policy_clock": post.get("policy_clock"),
            "pre_simulation_tick": clock.get("simulation_tick_start"),
            "post_simulation_tick": clock.get("simulation_tick_end"),
        }
        if dict(captured) != expected_captured:
            raise RecoveryContractError("private DART captured transition receipt differs from sealed episode row")
    fresh_reset = sidecar.get("fresh_reset_payload")
    if fresh_reset is not None:
        fresh_reset_mapping = _mapping(fresh_reset, field="private DART fresh reset payload")
        if sidecar.get("fresh_reset_payload_sha256") != canonical_sha256(fresh_reset_mapping):
            raise RecoveryContractError("private DART fresh reset payload hash is invalid")
        first_row = loaded.rows[0]
        first_pre = _mapping(first_row.get("pre_observation"), field="private DART first pre observation")
        first_clock = _mapping(first_row.get("clock"), field="private DART first clock")
        expected_initial = {
            "policy_clock": first_pre.get("policy_clock"),
            "state61_sha256": first_pre.get("state61_sha256"),
            "observation_sha256": first_pre.get("observation_sha256"),
            "simulation_tick": first_clock.get("simulation_tick_start"),
            "simulation_time_seconds": first_clock.get("action_start_time_s"),
        }
        initial = _mapping(fresh_reset_mapping.get("initial_observation"), field="private DART sealed reset observation")
        if any(initial.get(field) != expected for field, expected in expected_initial.items()):
            raise RecoveryContractError("private DART fresh reset payload differs from first sealed episode observation")
        views = initial.get("views")
        if (
            not isinstance(views, list)
            or [view.get("camera_id") if isinstance(view, Mapping) else None for view in views] != list(_CAMERAS)
        ):
            raise RecoveryContractError("private DART fresh reset payload lacks all raw camera receipts")
        runtime = _mapping(loaded.manifest.get("runtime_session"), field="private DART episode runtime session")
        if runtime.get("reset_load_task_instance_receipt_sha256") != canonical_sha256(fresh_reset_mapping):
            raise RecoveryContractError("private DART fresh reset payload differs from sealed runtime session")
    episode_receipts = [row.get("applied", {}).get("runtime_step_receipt_sha256") for row in loaded.rows]
    sidecar_receipts = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RecoveryContractError("private DART collection provenance record must be an object")
        _require_private_candidate_gates(record, field="private DART collection provenance record")
        if (
            record.get("source") != loaded.manifest.get("source")
            or record.get("verified_source_membership") != loaded.manifest.get("verified_source_membership")
            or record.get("runtime_session") != loaded.manifest.get("runtime_session")
            or record.get("teacher") != loaded.manifest.get("teacher")
        ):
            raise RecoveryContractError("private DART collection provenance differs from sealed episode lineage")
        if not isinstance(record.get("noise_profile"), Mapping) or not isinstance(record.get("calibration_receipt"), Mapping):
            raise RecoveryContractError("private DART collection provenance lacks noise or calibration receipts")
        sidecar_receipts.append(record.get("runtime_step_receipt_sha256"))
    if sidecar_receipts != episode_receipts:
        raise RecoveryContractError("private DART collection provenance runtime receipts differ from episode bundle")
    return PrivateEpisodePublicationReceipt(
        path=root,
        manifest_sha256=actual_sha,
        episode_bundle_path=root / "episode",
        episode_manifest_sha256=str(episode["manifest_sha256"]),
        collection_result_sha256=str(sidecar["collection_result_sha256"]),
        transition_count=len(loaded.rows),
        image_count=int(loaded.manifest["image_count"]),
    )


def _publish_private_episode_bundle(
    *, config: PrivateEpisodeWriterConfig, bundle: Any, collection: Mapping[str, object]
) -> PrivateEpisodePublicationReceipt:
    """Atomically publish a bundle and full collection provenance as one root."""

    try:
        from .dart_dataset import write_episode_bundle
    except ImportError as exc:
        raise RecoveryContractError("private DART dataset writer module is unavailable") from exc
    output = config.output
    if os.path.lexists(output):
        raise RecoveryContractError("refusing to overwrite an existing private DART publication path")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        episode_path = staging / "episode"
        episode_receipt = write_episode_bundle(episode_path, bundle)
        sidecar_path = staging / "private_collection.json"
        sidecar_path.write_bytes(_json_bytes(collection))
        files = {
            "episode/manifest.json": _file_receipt(episode_path / "manifest.json", field="private episode manifest"),
            "private_collection.json": _file_receipt(sidecar_path, field="private collection provenance"),
        }
        collection_result_sha = collection.get("collection_result_sha256")
        require_sha256(collection_result_sha, field="private collection result SHA-256")
        manifest: dict[str, object] = {
            "schema": _PRIVATE_PUBLICATION_SCHEMA,
            "status": "PRIVATE_DART_CANDIDATE_ONLY",
            **CANDIDATE_ONLY_FLAGS,
            "release_eligible": False,
            "training_eligible": False,
            "episode_bundle": {
                "relative_path": "episode",
                "manifest_sha256": episode_receipt.manifest_sha256,
                "transition_count": episode_receipt.transition_count,
                "image_count": episode_receipt.image_count,
            },
            "private_collection": {
                "relative_path": "private_collection.json",
                "sha256": files["private_collection.json"]["sha256"],
                "bytes": files["private_collection.json"]["bytes"],
                "collection_result_sha256": collection_result_sha,
                "collection_execution_provenance_sha256": collection.get("collection_execution_provenance_sha256"),
            },
            "files": files,
        }
        manifest_path = staging / "publication_manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        publication = load_private_episode_publication(staging, expected_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest())
        if os.path.lexists(output):
            raise RecoveryContractError("refusing to overwrite an existing private DART publication path")
        os.replace(staging, output)
        return replace(publication, path=output, episode_bundle_path=output / "episode")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _captured_rgb_asset(*, captured: CapturedLiveObservation, view: CapturedRgbView) -> Any:
    """Convert an already-captured native uint8 frame for s1's PNG writer."""

    try:
        import numpy as np
        from .dart_dataset import RGBAsset
    except ImportError as exc:
        raise RecoveryContractError("private DART dataset writer dependencies are unavailable") from exc
    if view.dtype != "uint8" or len(view.shape) != 3 or view.shape[-1] != 3:
        raise RecoveryContractError("private DART writer only accepts native uint8 HxWx3 captured RGB")
    try:
        expected_bytes = int(np.prod(view.shape))
        if len(view.payload) != expected_bytes:
            raise RecoveryContractError("captured RGB byte length does not match native uint8 shape")
        array = np.frombuffer(view.payload, dtype=np.uint8).reshape(view.shape)
    except (TypeError, ValueError) as exc:
        raise RecoveryContractError("captured RGB bytes cannot be reconstructed as native uint8 HxWx3") from exc
    return RGBAsset.from_array(
        view=view.camera_id,
        camera_key=view.camera_key,
        array=array,
        policy_clock=captured.dart.policy_clock,
        source_locator=(
            f"{captured.dart.observation_ref}#sim-step={captured.sim_step};camera={view.camera_id}"
        ),
        capture_time_s=captured.sim_time_seconds,
        pts=None,
    )


def build_private_episode_bundle_writer(config: PrivateEpisodeWriterConfig) -> Callable[..., Any]:
    """Return the concrete bridge to s1's atomic private episode writer.

    It joins each public DART record to its exact in-memory live transition by
    runtime receipt, serializes only pre-action RGB into the actor projection,
    and delegates fresh/no-overwrite/atomic publication to ``dart_dataset``.
    No target/contact/held state or raw state61 is handed to that writer.
    """

    if not isinstance(config, PrivateEpisodeWriterConfig):
        raise RecoveryContractError("build_private_episode_bundle_writer needs PrivateEpisodeWriterConfig")

    def writer(
        result: DartCollectionResult,
        transitions: tuple[CapturedLiveTransition, ...],
        *,
        collection_execution_provenance: Mapping[str, object] | None = None,
    ) -> Any:
        try:
            from .dart_dataset import ActorObservation, DartEpisodeBundle, DartEpisodeStep, RunProvenance, StepClock
        except ImportError as exc:
            raise RecoveryContractError("private DART dataset writer module is unavailable") from exc
        if not isinstance(result, DartCollectionResult) or not isinstance(transitions, tuple) or not transitions:
            raise RecoveryContractError("private DART bundle requires nonempty result and exact captured transition tuple")
        public = result.public()
        records = public.get("records")
        if public.get("outcome_label") != "OUTCOME_UNKNOWN" or not isinstance(records, list) or len(records) != len(transitions):
            raise RecoveryContractError("live factory can publish only cardinality-matched UNKNOWN-outcome DART candidates")
        for field, expected in CANDIDATE_ONLY_FLAGS.items():
            if public.get(field) != expected:
                raise RecoveryContractError("private DART bundle result lost a candidate-only release gate")
        by_receipt = {transition.applied.runtime_step_receipt_sha256: transition for transition in transitions}
        if len(by_receipt) != len(transitions):
            raise RecoveryContractError("captured live transitions have duplicate runtime receipt identity")
        steps: list[Any] = []
        modes: set[str] = set()
        kinds: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise RecoveryContractError("DART result records must be mappings")
            for field, expected in CANDIDATE_ONLY_FLAGS.items():
                if record.get(field) != expected:
                    raise RecoveryContractError("private DART bundle record lost a candidate-only release gate")
            receipt = record.get("runtime_step_receipt_sha256")
            transition = by_receipt.get(receipt) if isinstance(receipt, str) else None
            if transition is None:
                raise RecoveryContractError("DART record is missing its exact captured live transition")
            if (
                list(transition.applied.applied23) != record.get("applied23")
                or transition.applied.applied_action_bytes_sha256 != record.get("applied_action_bytes_sha256")
                or transition.applied.pre_action_policy_clock != transition.pre.dart.policy_clock
                or transition.applied.pre_action_observation_sha256 != transition.pre.dart.observation_sha256
            ):
                raise RecoveryContractError("DART record/action receipt disagrees with captured live transition")
            expected_step_receipt = canonical_sha256(
                {
                    "schema": "p107-dart-live-step-v1",
                    "pre_observation_sha256": transition.pre.dart.observation_sha256,
                    "post_observation_sha256": transition.post.dart.observation_sha256,
                    "pre_sim_step": transition.pre.sim_step,
                    "post_sim_step": transition.post.sim_step,
                    "raw23_wire_sha256": transition.applied.applied_action_bytes_sha256,
                }
            )
            if transition.applied.runtime_step_receipt_sha256 != expected_step_receipt:
                raise RecoveryContractError("runtime step receipt is not bound to the captured post-observation digest")
            if record.get("source") != config.source.public() or record.get("runtime_session") != config.runtime_session.public():
                raise RecoveryContractError("DART record source/runtime provenance differs from private bundle config")
            if record.get("verified_source_membership") != config.verified_membership.public():
                raise RecoveryContractError("DART record membership differs from private bundle config")
            if record.get("teacher") != config.teacher.public():
                raise RecoveryContractError("DART record teacher receipt differs from private bundle config")
            if not isinstance(record.get("intent_bundle_id"), str) or not record["intent_bundle_id"]:
                raise RecoveryContractError("DART record intent bundle ID is missing")
            try:
                teacher_query = record["teacher_query_receipt_sha256"]
                clean_intended = record["clean_intended23"]
                requested = record["requested_noisy23"]
                sampled = record["sampled_noise23"]
                requested_native = record["requested_native23"]
                requested_native_wire_sha256 = record["requested_native_action_bytes_sha256"]
                wire_dtype = record["native_action_wire_dtype"]
                label_kind = record["label_kind"]
                collection_mode = record["collection_mode"]
            except KeyError as exc:
                raise RecoveryContractError("DART record is missing required candidate fields") from exc
            if not isinstance(teacher_query, str) or not isinstance(label_kind, str) or not isinstance(collection_mode, str):
                raise RecoveryContractError("DART record teacher/mode/label fields are invalid")
            native_requested = canonical_native_raw23_float32(requested)
            if (
                wire_dtype != "float32_le"
                or tuple(requested_native) != native_requested
                or requested_native_wire_sha256 != native_raw23_float32_sha256(requested)
                or tuple(transition.applied.applied23) != native_requested
            ):
                raise RecoveryContractError("DART record native raw23 wire provenance disagrees with its captured transition")
            pre_assets = {view.camera_id: _captured_rgb_asset(captured=transition.pre, view=view) for view in transition.pre.rgb}
            post_assets = {view.camera_id: _captured_rgb_asset(captured=transition.post, view=view) for view in transition.post.rgb}
            command = TeacherCommand(
                clean_intended23=clean_intended,
                observed_policy_clock=transition.pre.dart.policy_clock,
                observed_state61_sha256=transition.pre.dart.state61_sha256,
                observed_observation_sha256=transition.pre.dart.observation_sha256,
                intent_bundle_id=str(record.get("intent_bundle_id", "")),
                fresh_query_receipt_sha256=teacher_query,
            )
            steps.append(
                DartEpisodeStep(
                    step_index=index,
                    pre_observation=transition.pre.dart,
                    pre_actor_observation=ActorObservation(
                        policy_clock=transition.pre.dart.policy_clock,
                        assets=pre_assets,
                        proprioception=None,
                    ),
                    teacher_command=command,
                    requested_noisy23=requested,
                    sampled_noise23=sampled,
                    requested_native23=requested_native,
                    requested_native_action_bytes_sha256=requested_native_wire_sha256,
                    applied=transition.applied,
                    post_observation=transition.post.dart,
                    post_actor_observation=ActorObservation(
                        policy_clock=transition.post.dart.policy_clock,
                        assets=post_assets,
                        proprioception=None,
                    ),
                    clock=StepClock(
                        simulation_tick_start=transition.pre.sim_step,
                        simulation_tick_end=transition.post.sim_step,
                        action_start_time_s=transition.pre.sim_time_seconds,
                        action_end_time_s=transition.post.sim_time_seconds,
                    ),
                    label_kind=label_kind,
                    native_action_wire_dtype=wire_dtype,
                )
            )
            modes.add(collection_mode)
            kinds.add(label_kind)
        if len(modes) != 1:
            raise RecoveryContractError("one private DART bundle may not mix collection modes")
        intent_bundle_id = steps[0].teacher_command.intent_bundle_id
        if not intent_bundle_id or any(step.teacher_command.intent_bundle_id != intent_bundle_id for step in steps):
            raise RecoveryContractError("private DART bundle steps need one explicit sealed intent_bundle_id")
        bundle = DartEpisodeBundle(
            source=config.source,
            verified_membership=config.verified_membership,
            runtime_session=config.runtime_session,
            teacher=config.teacher,
            intent_bundle_id=intent_bundle_id,
            collection_mode=next(iter(modes)),
            label_kind=next(iter(kinds)) if len(kinds) == 1 else "mixed",
            run_provenance=RunProvenance(
                run_id=config.source.collection_run_id,
                source_episode_id=config.source_episode_id,
                run_origin="live_runtime",
                collector_code_sha256=config.collector_code_sha256,
                capture_adapter_sha256=config.capture_adapter_sha256,
                actor_observation_schema_sha256=config.actor_observation_schema_sha256,
            ),
            steps=tuple(steps),
            outcome_evidence=config.outcome_evidence,
            # The available binding carries only a hash for public task/intent
            # text.  Never substitute its private opaque intent ID into the
            # actor view merely to populate this optional field.
            public_causal_instruction=None,
        )
        return _publish_private_episode_bundle(
            config=config,
            bundle=bundle,
            collection=_private_collection_payload(
                result,
                transitions,
                collection_execution_provenance=collection_execution_provenance,
                fresh_reset_payload=config.fresh_reset_payload,
            ),
        )

    return writer


def validate_live_r1pro_controller(robot: Any) -> dict[str, object]:
    """Validate the actual action order and controller classes before any step."""

    if getattr(robot, "model", None) != "r1pro" or getattr(robot, "action_dim", None) != 23:
        raise RecoveryContractError("live factory requires an R1Pro with native action_dim=23")
    raw_indices = getattr(robot, "controller_action_idx", None)
    if not isinstance(raw_indices, Mapping):
        raise RecoveryContractError("R1Pro lacks controller_action_idx")
    actual = {str(name): [int(value) for value in indices] for name, indices in raw_indices.items() if len(indices)}
    if actual != _EXPECTED_R1PRO_CONTROLLER_INDICES:
        raise RecoveryContractError("live R1Pro controller action layout differs from reviewed native23 layout")
    controllers = getattr(robot, "controllers", None)
    if not isinstance(controllers, Mapping):
        raise RecoveryContractError("R1Pro lacks instantiated controller mapping")
    result: dict[str, object] = {"action_dim": 23, "controller_action_idx": actual, "controllers": {}}
    for name, expected_class in _EXPECTED_CONTROLLER_CLASSES.items():
        controller = controllers.get(name)
        if controller is None or type(controller).__name__ != expected_class:
            raise RecoveryContractError(f"R1Pro controller {name!r} is not {expected_class}")
        item: dict[str, object] = {"class": expected_class}
        if name.startswith("gripper_"):
            mode = getattr(controller, "mode", getattr(controller, "_mode", None))
            inverted = getattr(controller, "inverted", getattr(controller, "_inverted", None))
            if mode != "smooth" or type(inverted) is not bool:
                raise RecoveryContractError("live R1Pro gripper must expose smooth mode and explicit inversion state")
            # This records rather than interprets sign: command direction is a
            # controller property, never a substitute for hold/contact truth.
            item.update({"mode": mode, "inverted": inverted})
        result["controllers"][name] = item
    return result


class DartOgRuntimeHooks:
    """Concrete current-observation / exact-step bridge over a pinned evaluator.

    The evaluator is deliberately restricted to one logical environment.  The
    factory captures RGB and proprio before every raw23 action, calls the
    evaluator's actual step path, and captures fresh post-step media before a
    later DART record may refer to it.
    """

    def __init__(self, *, evaluator: Any, torch_module: Any, og_module: Any, run_id: str, max_control_steps: int) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise RecoveryContractError("live factory run_id must be nonempty")
        if type(max_control_steps) is not int or max_control_steps < 1:
            raise RecoveryContractError("live factory max_control_steps must be positive")
        if getattr(evaluator, "num_envs", None) != 1:
            raise RecoveryContractError("live DART factory requires BatchedEvaluator(num_envs=1)")
        states = getattr(evaluator, "instance_eval_states", None)
        if not isinstance(states, Sequence) or len(states) != 1:
            raise RecoveryContractError("live evaluator must expose exactly one instance state")
        accessor = getattr(states[0], "env_accessor", None)
        robot = getattr(accessor, "robot", None)
        self.controller_receipt = validate_live_r1pro_controller(robot)
        self._evaluator, self._torch, self._og = evaluator, torch_module, og_module
        self._state, self._robot = states[0], robot
        self._run_id, self._max = run_id, max_control_steps
        self._clock = 0
        self._last: CapturedLiveObservation | None = None
        self._primed_fresh_reset = False
        self._transitions: list[CapturedLiveTransition] = []
        self._closed = False

    @property
    def transitions(self) -> tuple[CapturedLiveTransition, ...]:
        return tuple(self._transitions)

    def _sim_clock(self) -> tuple[int, float]:
        sim = getattr(self._og, "sim", None)
        step = getattr(sim, "current_time_step_index", None)
        seconds = getattr(sim, "current_time", None)
        if type(step) is not int or step < 0:
            raise RecoveryContractError("pinned simulator must expose nonnegative current_time_step_index")
        return step, _as_float(seconds, field="sim.current_time")

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if callable(getattr(value, "detach", None)):
            value = value.detach()
        if callable(getattr(value, "cpu", None)):
            value = value.cpu()
        if callable(getattr(value, "numpy", None)):
            value = value.numpy()
        if not hasattr(value, "shape") or not hasattr(value, "dtype") or not hasattr(value, "tobytes"):
            raise RecoveryContractError("evaluator observation values must expose array shape/dtype/bytes")
        return value

    def _capture_current(self, *, policy_clock: int) -> CapturedLiveObservation:
        obs = getattr(self._state, "obs", None)
        if not isinstance(obs, Mapping):
            raise RecoveryContractError("evaluator has no current preprocessed observation")
        proprio_values = [value for key, value in obs.items() if str(key).endswith("::proprio")]
        if len(proprio_values) != 1:
            raise RecoveryContractError("live evaluator must expose exactly one R1Pro proprio observation")
        proprio = self._to_numpy(proprio_values[0])
        try:
            state61 = tuple(float(item) for item in proprio.reshape(-1).tolist())
        except (AttributeError, TypeError, ValueError) as exc:
            raise RecoveryContractError("R1Pro proprio must be flattenable numeric state61") from exc
        state61 = tuple(validate_state61(state61))
        names = getattr(self._evaluator, "robot_camera_names", None)
        if not isinstance(names, Mapping) or set(names) != set(_CAMERAS):
            raise RecoveryContractError("pinned evaluator must resolve exactly left/right/head R1Pro cameras")
        rgb: list[CapturedRgbView] = []
        for camera_id in _CAMERAS:
            key = str(names[camera_id]) + "::rgb"
            if key not in obs:
                raise RecoveryContractError(f"current observation is missing {camera_id} RGB")
            array = self._to_numpy(obs[key])
            shape = tuple(int(item) for item in array.shape)
            # The writer owns native PNG encoding.  Do not accidentally treat
            # model-normalized float/CHW tensors as pixels and then serialize
            # them as purported causal RGB evidence.
            if str(array.dtype) != "uint8" or len(shape) != 3 or shape[-1] != 3:
                raise RecoveryContractError("live R1Pro RGB must be native uint8 HxWx3, not a preprocessed tensor")
            rgb.append(
                CapturedRgbView(
                    camera_id=camera_id,
                    camera_key=str(names[camera_id]),
                    dtype=str(array.dtype),
                    shape=shape,
                    payload=bytes(array.tobytes()),
                )
            )
        sim_step, sim_seconds = self._sim_clock()
        observation_sha256 = canonical_sha256(
            {
                "schema": "p107-dart-live-observation-v1",
                "policy_clock": policy_clock,
                "state61": list(state61),
                "sim_step": sim_step,
                "sim_time_seconds": sim_seconds,
                "views": [item.public() for item in rgb],
            }
        )
        dart = DartObservation(
            policy_clock=policy_clock,
            state61=state61,
            observation_sha256=observation_sha256,
            observation_ref=f"dart-live://{self._run_id}/policy/{policy_clock}",
        )
        return CapturedLiveObservation(dart=dart, sim_step=sim_step, sim_time_seconds=sim_seconds, rgb=tuple(rgb))

    def observe_current(self) -> DartObservation:
        if self._closed:
            raise RecoveryContractError("live DART runtime is closed")
        if self._last is not None:
            if self._primed_fresh_reset:
                self._primed_fresh_reset = False
                return self._last.dart
            raise RecoveryContractError("live runtime needs its preceding observation consumed before another observe")
        self._last = self._capture_current(policy_clock=self._clock)
        return self._last.dart

    def prime_fresh_reset_observation(self) -> CapturedLiveObservation:
        """Capture the actual post-``load_batch`` state used by the first action.

        The following ``observe`` returns this exact object rather than taking
        a second, potentially different, pre-action capture.  This makes the
        reset receipt evidence about the action sequence being published.
        """

        if self._closed or self._last is not None or self._transitions or self._clock != 0:
            raise RecoveryContractError("fresh-reset observation can only be primed once before any live control")
        self._last = self._capture_current(policy_clock=0)
        self._primed_fresh_reset = True
        return self._last

    def apply_current_raw23(self, requested23: Sequence[float], observation: DartObservation) -> AppliedActionReceipt:
        if self._closed:
            raise RecoveryContractError("live DART runtime is closed")
        # The sampler's float64 request is preserved by the collector.  This
        # runtime owns only the exact float32 values actually passed to OG.
        requested = canonical_native_raw23_float32(requested23)
        # ``validate_raw23_action`` accepts finite Python floats.  Native
        # controller transport is float32, so prove exact wire
        # representability *before* constructing a tensor or stepping; a
        # Gaussian outlier must abort rather than be clipped/cast to infinity.
        requested_wire_sha256 = raw23_wire_sha256(requested)
        pre = self._last
        if pre is None:
            raise RecoveryContractError("live raw23 step requires an unconsumed preceding current observation")
        if observation != pre.dart:
            raise RecoveryContractError("live raw23 step is not bound to its current captured observation")
        if self._clock >= self._max:
            raise RecoveryContractError("live DART control budget exhausted before simulator step")
        try:
            action = self._torch.tensor([list(requested)], dtype=self._torch.float32)
            submitted = self._to_numpy(action)
            if tuple(int(value) for value in submitted.shape) != (1, 23) or str(submitted.dtype) != "float32":
                raise RecoveryContractError("pinned evaluator action tensor must remain native float32 [1,23]")
            if sha256(bytes(submitted.tobytes())).hexdigest() != requested_wire_sha256:
                raise RecoveryContractError("torch action tensor bytes differ from requested native raw23 wire action")
            terminated, truncated, _info = self._evaluator._apply_actions(action, [0])
        except RecoveryContractError:
            self._last = None
            self._primed_fresh_reset = False
            raise
        except Exception as exc:
            # The action may have partially executed.  Drop the preceding
            # observation so callers cannot resend it after a runtime failure.
            self._last = None
            self._primed_fresh_reset = False
            raise RecoveryContractError("pinned evaluator failed while applying raw23; no retry on stale state") from exc
        self._clock += 1
        post = self._capture_current(policy_clock=self._clock)
        receipt = AppliedActionReceipt(
            status="APPLIED",
            applied23=requested,
            applied_action_bytes_sha256=requested_wire_sha256,
            runtime_step_receipt_sha256=canonical_sha256(
                {
                    "schema": "p107-dart-live-step-v1",
                    "pre_observation_sha256": pre.dart.observation_sha256,
                    "post_observation_sha256": post.dart.observation_sha256,
                    "pre_sim_step": pre.sim_step,
                    "post_sim_step": post.sim_step,
                    "raw23_wire_sha256": requested_wire_sha256,
                }
            ),
            pre_action_policy_clock=pre.dart.policy_clock,
            pre_action_state61_sha256=pre.dart.state61_sha256,
            pre_action_observation_sha256=pre.dart.observation_sha256,
        )
        transition = CapturedLiveTransition(pre=pre, post=post, applied=receipt)
        self._last = None
        self._primed_fresh_reset = False
        self._transitions.append(transition)
        # A terminal step may have physically happened, but cannot become a
        # DART candidate because the generic collector has no terminal-row
        # semantics.  Fail closed before it writes a result.
        if bool(terminated[0]) or bool(truncated[0]):
            raise RecoveryContractError("official environment terminated during live DART step")
        return receipt

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for owner in (self._evaluator, getattr(self._evaluator, "env", None)):
            close = getattr(owner, "close", None)
            if callable(close):
                close()


class _PoseTeacherDartBridge:
    """Make the existing privileged GRASP lifecycle explicit to DART loops."""

    def __init__(self, *, teacher: PrivilegedPoseGraspTeacher, runtime: FreshRolloutRaw23Runtime) -> None:
        self._teacher, self._runtime = teacher, runtime
        self._pending = None

    def clean_action(self, observation: DartObservation, *, intent_bundle_id: str):
        command = self._teacher.clean_action(observation, intent_bundle_id=intent_bundle_id)
        self._pending = command
        return command

    def acknowledge(self, applied: AppliedActionReceipt) -> None:
        command = self._pending
        if command is None:
            raise RecoveryContractError("DART raw23 step has no pending GRASP teacher command")
        try:
            if tuple(applied.applied23) == canonical_native_raw23_float32(command.clean_intended23):
                self._teacher.acknowledge_runtime_clean_application(self._runtime, applied)
            else:
                # This may retire the teacher if Gaussian/bounded noise changed
                # a gripper channel.  It must not clip or relabel the action.
                self._teacher.acknowledge_runtime_noisy_application(self._runtime, applied)
        finally:
            self._pending = None


class _AcknowledgingRuntime:
    """DartRuntime facade that preserves native GRASP lifecycle acknowledgements."""

    def __init__(self, *, runtime: FreshRolloutRaw23Runtime, teacher: _PoseTeacherDartBridge) -> None:
        self._runtime, self._teacher = runtime, teacher

    def observe(self) -> DartObservation:
        return self._runtime.observe()

    def apply_raw23(self, requested23: Sequence[float]) -> AppliedActionReceipt:
        applied = self._runtime.apply_raw23(requested23)
        self._teacher.acknowledge(applied)
        return applied


class DartOgGraspFactory:
    """A live one-environment GRASP session with all private objects contained."""

    def __init__(
        self,
        *,
        hooks: DartOgRuntimeHooks,
        teacher: PrivilegedPoseGraspTeacher,
        evaluator: Any,
        pins: DartOgFactoryPins,
        train_selection: TrainSourceSelection,
        runtime_session: RuntimeSessionReceipt,
        fresh_reset_payload: Mapping[str, object],
    ) -> None:
        self.hooks, self.teacher, self.evaluator = hooks, teacher, evaluator
        self.pins, self.train_selection = pins, train_selection
        self.runtime_session = runtime_session
        self.fresh_reset_payload = dict(fresh_reset_payload)
        self.runtime = FreshRolloutRaw23Runtime(hooks)
        self.teacher_bridge = _PoseTeacherDartBridge(teacher=teacher, runtime=self.runtime)
        self.dart_runtime = _AcknowledgingRuntime(runtime=self.runtime, teacher=self.teacher_bridge)

    @classmethod
    def create(
        cls,
        *,
        pins: DartOgFactoryPins,
        source: CandidateSourceReceipt,
        train_selection: TrainSourceSelection,
        verified_membership: VerifiedDartSourceMembership,
        binding: SealedGraspBinding,
        teacher_spec: Mapping[str, object],
        teacher_reference: Mapping[str, object],
        run_id: str,
        max_control_steps: int,
        runtime_plan: FreshRuntimeSessionPlan,
    ) -> "DartOgGraspFactory":
        """Create the actual pinned evaluator / R1Pro / teacher wiring lazily."""

        if source.original_split != "train":
            raise RecoveryContractError("live DART factory only accepts original TRAIN sources")
        if binding.source_group_id != source.source_group_id or binding.source_release_sha256 != source.source_release_sha256:
            raise RecoveryContractError("sealed GRASP binding and DART source differ")
        if not isinstance(train_selection, TrainSourceSelection):
            raise RecoveryContractError("live DART factory requires an explicit sealed TRAIN selection")
        train_selection.assert_source(source)
        # Validate the full canonical member/index/role binding before an
        # evaluator is constructed.  Matching just group/release would let a
        # calibration-role binding borrow a student-candidate source receipt.
        binding.assert_verified_dart_membership(verified_membership)
        if (
            verified_membership.candidate.source_group_id != source.source_group_id
            or verified_membership.candidate.source_release_sha256 != source.source_release_sha256
            or verified_membership.candidate.task_index != source.parent_task_index
            or verified_membership.candidate.task_instance_id != source.parent_task_instance_id
            or verified_membership.candidate.original_split != "train"
            or verified_membership.candidate.usage_role != "student_candidate"
        ):
            raise RecoveryContractError("live DART factory source is not verified TRAIN student_candidate membership")
        if (
            runtime_plan.runtime_session_id != run_id
            or run_id != source.collection_run_id
            or runtime_plan.task_instance_id != source.parent_task_instance_id
            or runtime_plan.runtime_build_sha256 != pins.runtime_build_sha256
            or runtime_plan.asset_config_sha256 != pins.asset_config_sha256
        ):
            raise RecoveryContractError("fresh runtime session plan differs from the sealed source/pins")
        try:
            from omegaconf import OmegaConf  # lazy: unavailable in CPU contract-only environments
            import torch
            import omnigibson as og
            from omnigibson.eval.evaluator import BatchedEvaluator
        except ImportError as exc:
            raise RecoveryContractError("pinned OmniGibson evaluator runtime is unavailable; live DART cannot start") from exc

        evaluator: Any | None = None
        try:
            cfg = OmegaConf.load(str(pins.evaluator_config_path))
            if str(cfg.get("mode", "")) != "train":
                raise RecoveryContractError("evaluator config must explicitly set mode=train; public_test defaults are forbidden")
            if str(cfg.task.name) != train_selection.task_name or int(cfg.seed) != train_selection.task_seed:
                raise RecoveryContractError("evaluator task/seed does not match sealed TRAIN selection")
            cfg.num_envs = 1
            evaluator = BatchedEvaluator(cfg)
            evaluator.load_batch({0: train_selection.task_instance_id})
            runtime_task = getattr(getattr(evaluator, "env", None), "task", None)
            if getattr(runtime_task, "scene_name", None) != train_selection.scene_name:
                raise RecoveryContractError("fresh evaluator scene differs from sealed TRAIN selection")
        except RecoveryContractError:
            if evaluator is not None:
                close = getattr(evaluator, "close", None)
                if callable(close):
                    close()
            raise
        except Exception as exc:
            if evaluator is not None:
                close = getattr(evaluator, "close", None)
                if callable(close):
                    close()
            raise RecoveryContractError("pinned evaluator could not fresh-reset the requested TRAIN task instance") from exc

        try:
            project_root = Path(__file__).resolve().parents[3]
            native_root = project_root / "scripts" / "vlm_sft"
            if str(native_root) not in __import__("sys").path:
                __import__("sys").path.insert(0, str(native_root))
            from native_teacher_og import PrivilegedReader
            from native_teacher_policy import PoseTeacher, validate_spec
            from semantic_robot.v2.og_calibration import CalibratedRobot
            from semantic_robot.v2.servo import SafeServo
            from common import token_to_action

            validate_spec(dict(teacher_spec), dict(teacher_reference))
            if teacher_spec.get("verb") != "GRASP" or teacher_spec.get("hand") != binding.hand:
                raise RecoveryContractError("complete sealed PoseTeacher spec does not match GRASP binding")
            if teacher_spec.get("target") != binding.target_name:
                raise RecoveryContractError("complete sealed PoseTeacher target differs from GRASP binding")
            state = evaluator.instance_eval_states[0]
            robot = state.env_accessor.robot
            kin = CalibratedRobot(robot)
            model = kin.calibrate(grounded=True)
            reader = PrivilegedReader(PinnedSingleEnvNativeReaderView.from_evaluator(evaluator), dict(teacher_spec))

            def teacher_ctor(minimal: Mapping[str, object]) -> Any:
                if dict(minimal) != {"verb": "GRASP", "hand": binding.hand}:
                    raise RecoveryContractError("pose engine requested a different sealed GRASP intent")
                # The engine's minimal lifecycle key is deliberate, but the
                # native teacher must receive the complete validated spec.
                return PoseTeacher(dict(teacher_spec))

            def fresh_private_receipt(observation: DartObservation, frame: Mapping[str, object], robot_state: Any) -> str:
                # No target/contact data leaves this process; the opaque hash
                # proves a current private query was performed at this state.
                return canonical_sha256(
                    {
                        "schema": "p107-dart-live-private-query-v1",
                        "observation_sha256": observation.observation_sha256,
                        "policy_clock": observation.policy_clock,
                        "target_uid": frame.get("target_uid"),
                        "state_gripper": list(getattr(robot_state, "gripper")),
                        "teacher_spec_sha256": canonical_sha256(dict(teacher_spec)),
                    }
                )

            hooks = DartOgRuntimeHooks(
                evaluator=evaluator, torch_module=torch, og_module=og, run_id=run_id, max_control_steps=max_control_steps
            )
            initial = hooks.prime_fresh_reset_observation()
            fresh_reset_payload = {
                "schema": "p107-dart-live-fresh-reset-v1",
                "runtime_session_id": runtime_plan.runtime_session_id,
                "source_group_id": source.source_group_id,
                "source_release_sha256": source.source_release_sha256,
                "parent_task_id": source.parent_task_id,
                "parent_task_index": source.parent_task_index,
                "task_instance_id": train_selection.task_instance_id,
                "task_seed": train_selection.task_seed,
                "source_selection_sha256": train_selection.source_selection_sha256,
                "og_source_commit": pins.og_source_commit,
                "evaluator_config_sha256": pins.evaluator_config_sha256,
                "r1pro_config_sha256": pins.r1pro_config_sha256,
                "runtime_build_sha256": pins.runtime_build_sha256,
                "asset_config_sha256": pins.asset_config_sha256,
                "evaluator_mode": "train",
                "num_envs": 1,
                "load_batch": {"0": train_selection.task_instance_id},
                "initial_observation": {
                    "policy_clock": initial.dart.policy_clock,
                    "state61_sha256": initial.dart.state61_sha256,
                    "observation_sha256": initial.dart.observation_sha256,
                    "simulation_tick": initial.sim_step,
                    "simulation_time_seconds": initial.sim_time_seconds,
                    "views": [view.public() for view in initial.rgb],
                },
            }
            runtime_session = RuntimeSessionReceipt(
                runtime_session_id=runtime_plan.runtime_session_id,
                task_instance_id=runtime_plan.task_instance_id,
                runtime_build_sha256=runtime_plan.runtime_build_sha256,
                asset_config_sha256=runtime_plan.asset_config_sha256,
                reset_load_task_instance_receipt_sha256=canonical_sha256(fresh_reset_payload),
                snapshot_mode="no_restore_fresh_rollout",
            )
            world = ExistingPrivilegedReaderWorld(
                privileged_reader=reader, kinematics=kin, servo_model=model, fresh_query_receipt=fresh_private_receipt
            )
            engine = ExistingPoseServoEngine(
                pose_teacher_class=teacher_ctor, safe_servo_class=SafeServo, token_to_action=token_to_action
            )
            teacher = PrivilegedPoseGraspTeacher(binding=binding, world=world, engine=engine)
            return cls(
                hooks=hooks,
                teacher=teacher,
                evaluator=evaluator,
                pins=pins,
                train_selection=train_selection,
                runtime_session=runtime_session,
                fresh_reset_payload=fresh_reset_payload,
            )
        except Exception:
            for owner in (evaluator, getattr(evaluator, "env", None)):
                close = getattr(owner, "close", None)
                if callable(close):
                    close()
            raise

    def close(self) -> None:
        self.hooks.close()


def run_live_grasp_collection(
    *,
    factory: DartOgGraspFactory,
    mode: str,
    teacher_receipt: TeacherReceipt,
    source: CandidateSourceReceipt,
    source_membership_reader: Any,
    runtime_session: RuntimeSessionReceipt,
    calibration: CalibrationReceipt,
    intent_bundle_id: str,
    writer: Callable[..., Any],
    original_noise: OriginalGaussianNoise | None = None,
    original_steps: int | None = None,
    original_binding: OriginalGaussianCalibrationBinding | None = None,
    bounded_noise: DartInspiredBoundedNoise | None = None,
    noisy_injection_steps: int | None = None,
    recovery_chunk_lengths: Sequence[int] | None = None,
    limits: CollectionLimits | None = None,
) -> DartCollectionResult:
    """Run exactly one bounded live collection and hand exact RGB/action joins to s1's writer.

    ``writer`` receives private media bytes only after every DART record has
    been joined one-to-one to its live transition receipt.  It must persist a
    new output and preserve all candidate-only gates; it is intentionally
    supplied by the dataset owner rather than duplicated here.
    """

    if not callable(writer):
        raise RecoveryContractError("live DART factory requires the reviewed episode/RGB writer callable")
    if teacher_receipt.teacher_kind != "privileged_pose_grasp_unqualified":
        raise RecoveryContractError(
            "generic live GRASP teacher remains privileged_pose_grasp_unqualified until clean/noisy runtime qualification"
        )
    if teacher_receipt.label_source != "privileged_oracle" or teacher_receipt.feedback_mode != "closed_loop_fresh_observation":
        raise RecoveryContractError("live GRASP teacher receipt must declare a fresh privileged-oracle query, not a replay label")
    if runtime_session.snapshot_mode != "no_restore_fresh_rollout":
        raise RecoveryContractError("live GRASP factory does not restore demo or external snapshots")
    if runtime_session.task_instance_id != source.parent_task_instance_id:
        raise RecoveryContractError("runtime session instance differs from live TRAIN source")
    factory_session = getattr(factory, "runtime_session", None)
    if factory_session is not None and factory_session != runtime_session:
        raise RecoveryContractError("live DART runtime session was not minted by this fresh factory reset")
    factory.train_selection.assert_source(source)
    if runtime_session.task_instance_id != factory.train_selection.task_instance_id:
        raise RecoveryContractError("runtime session instance differs from loaded TRAIN selection")
    if (
        runtime_session.runtime_build_sha256 != factory.pins.runtime_build_sha256
        or runtime_session.asset_config_sha256 != factory.pins.asset_config_sha256
    ):
        raise RecoveryContractError("runtime session pins differ from the constructed live evaluator")
    try:
        if mode == "original_gaussian_one_pass_partial_dart":
            if original_noise is None or original_steps is None or original_binding is None:
                raise RecoveryContractError("original DART mode requires exact Gaussian noise, step count and calibration binding")
            result = collect_original_gaussian_clean_feedback(
                runtime=factory.dart_runtime,
                teacher_callback=factory.teacher_bridge,
                teacher_receipt=teacher_receipt,
                source=source,
                source_membership_reader=source_membership_reader,
                runtime_session=runtime_session,
                calibration=calibration,
                noise=original_noise,
                intent_bundle_id=intent_bundle_id,
                steps=original_steps,
                covariance_update_mode="frozen_one_pass_partial_dart",
                original_calibration_binding=original_binding,
            )
            execution_provenance = _collection_execution_provenance(
                mode=mode,
                original_noise=original_noise,
                original_steps=original_steps,
                original_binding=original_binding,
            )
        elif mode == "dart_inspired_bounded_actual_clean_recovery":
            if bounded_noise is None or noisy_injection_steps is None or recovery_chunk_lengths is None or limits is None:
                raise RecoveryContractError("bounded path requires its explicit noise/injection/recovery/limit configuration")
            result = collect_dart_inspired_actual_clean_recovery(
                runtime=factory.dart_runtime,
                teacher_callback=factory.teacher_bridge,
                teacher_receipt=teacher_receipt,
                source=source,
                source_membership_reader=source_membership_reader,
                runtime_session=runtime_session,
                calibration=calibration,
                noise=bounded_noise,
                intent_bundle_id=intent_bundle_id,
                noisy_injection_steps=noisy_injection_steps,
                recovery_chunk_lengths=recovery_chunk_lengths,
                limits=limits,
            )
            execution_provenance = _collection_execution_provenance(
                mode=mode,
                bounded_noise=bounded_noise,
                noisy_injection_steps=noisy_injection_steps,
                recovery_chunk_lengths=recovery_chunk_lengths,
                limits=limits,
            )
        else:
            raise RecoveryContractError("unknown live DART collection mode")
        records = result.public()["records"]
        transitions = factory.hooks.transitions
        if len(records) != len(transitions):
            raise RecoveryContractError("DART records and captured live transitions differ in count")
        by_receipt = {item.applied.runtime_step_receipt_sha256: item for item in transitions}
        if len(by_receipt) != len(transitions):
            raise RecoveryContractError("live DART transitions have duplicate runtime receipts")
        for record in records:
            receipt = record.get("runtime_step_receipt_sha256")
            if not isinstance(receipt, str) or receipt not in by_receipt:
                raise RecoveryContractError("DART record has no exact captured live transition")
        writer(result, transitions, collection_execution_provenance=execution_provenance)
        return result
    finally:
        factory.close()


def _construct(cls: Any, payload: object, *, field: str) -> Any:
    """Instantiate one explicit dataclass from JSON without default synthesis."""

    raw = _mapping(payload, field=field)
    try:
        return cls(**dict(raw))
    except (TypeError, RecoveryContractError, ValueError) as exc:
        raise RecoveryContractError(f"{field} is not a valid {cls.__name__}") from exc


def _parse_binding(payload: object) -> SealedGraspBinding:
    raw = dict(_mapping(payload, field="binding"))
    raw["pose_review"] = _construct(RootModelPoseReview, raw.get("pose_review"), field="binding.pose_review")
    return _construct(SealedGraspBinding, raw, field="binding")


def _parse_factory_pins(payload: object) -> DartOgFactoryPins:
    raw = dict(_mapping(payload, field="pins"))
    for field in ("evaluator_config_path", "r1pro_config_path"):
        value = raw.get(field)
        if not isinstance(value, str) or not value:
            raise RecoveryContractError(f"pins.{field} must be a nonempty path")
        raw[field] = Path(value)
    return _construct(DartOgFactoryPins, raw, field="pins")


def _parse_calibration(payload: object) -> CalibrationReceipt:
    """Read the DATA receipt's public purpose marker without relaxing it."""

    raw = dict(_mapping(payload, field="calibration"))
    purpose = raw.pop("purpose", None)
    if purpose not in {None, "train_only_covariance_calibration"}:
        raise RecoveryContractError("calibration purpose must remain train_only_covariance_calibration")
    return _construct(CalibrationReceipt, raw, field="calibration")


def _parse_fresh_runtime_plan(payload: object) -> FreshRuntimeSessionPlan:
    """Reject a purported reset receipt before this process performs reset."""

    raw = dict(_mapping(payload, field="runtime_session"))
    supplied_reset = raw.pop("reset_load_task_instance_receipt_sha256", None)
    if supplied_reset is not None:
        raise RecoveryContractError(
            "runtime_session.reset_load_task_instance_receipt_sha256 must be null or absent; "
            "this invocation mints it after load_batch and its first capture"
        )
    supplied_snapshot = raw.pop("same_session_snapshot_receipt_sha256", None)
    if supplied_snapshot is not None:
        raise RecoveryContractError("runtime_session may not supply a snapshot receipt for fresh no-restore collection")
    return _construct(FreshRuntimeSessionPlan, raw, field="runtime_session")


def _parse_original_collection(payload: object) -> tuple[OriginalGaussianNoise, int, OriginalGaussianCalibrationBinding]:
    raw = _mapping(payload, field="original_gaussian")
    noise = _construct(OriginalGaussianNoise, raw.get("noise"), field="original_gaussian.noise")
    binding = _construct(
        OriginalGaussianCalibrationBinding,
        raw.get("calibration_binding"),
        field="original_gaussian.calibration_binding",
    )
    steps = raw.get("steps")
    if type(steps) is not int or steps < 1:
        raise RecoveryContractError("original_gaussian.steps must be a positive integer")
    return noise, steps, binding


def _parse_bounded_collection(payload: object) -> tuple[DartInspiredBoundedNoise, int, tuple[int, ...], CollectionLimits]:
    raw = _mapping(payload, field="bounded_dart")
    profile_raw = _mapping(raw.get("profile"), field="bounded_dart.profile")
    layout_raw = _mapping(profile_raw.get("layout"), field="bounded_dart.profile.layout")
    parts = layout_raw.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        raise RecoveryContractError("bounded_dart.profile.layout.parts must be a sequence")
    layout = Raw23ActionLayout(
        parts=tuple(_construct(ActionPart, item, field="bounded_dart.profile.layout.parts[]") for item in parts),
        embodiment_metadata_sha256=layout_raw.get("embodiment_metadata_sha256"),
        model_projection_manifest_sha256=layout_raw.get("model_projection_manifest_sha256"),
    )
    rules_raw = _mapping(profile_raw.get("rules_by_part"), field="bounded_dart.profile.rules_by_part")
    profile = BoundedNoiseProfile(
        layout=layout,
        rules_by_part={
            key: _construct(BoundedPartRule, value, field=f"bounded_dart.profile.rules_by_part.{key}")
            for key, value in rules_raw.items()
        },
        temporal_rho=profile_raw.get("temporal_rho"),
        profile_id=profile_raw.get("profile_id"),
    )
    seed = raw.get("seed")
    if type(seed) is not int:
        raise RecoveryContractError("bounded_dart.seed must be an integer")
    injection = raw.get("noisy_injection_steps")
    chunks = raw.get("recovery_chunk_lengths")
    if type(injection) is not int or injection < 1 or not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise RecoveryContractError("bounded_dart needs positive noisy_injection_steps and recovery_chunk_lengths")
    limits = _construct(CollectionLimits, raw.get("limits"), field="bounded_dart.limits")
    return DartInspiredBoundedNoise(profile, seed=seed), injection, tuple(chunks), limits


def collect_dart_live_candidate(request: Mapping[str, object]) -> DartCollectionResult:
    """Executable ``collect_dart_demonstrations.py`` factory for one live GRASP.

    The request must have schema ``p107-dart-live-grasp-request-v1`` and name
    every sealed path/SHA, source-index root, TRAIN task/scene/instance/seed,
    full private teacher spec/reference, unqualified teacher receipt, noise
    mode, and a fresh private bundle output.  It deliberately has no defaults
    for evaluator mode, split, task instance, output path, control budget,
    teacher, or actor projection.  Example CLI invocation is:

    ``python scripts/data/collect_dart_demonstrations.py --request REQUEST.json
    --factory g05.recovery.dart_og_factory:collect_dart_live_candidate
    --output fresh-candidate-summary.json``.

    This only becomes a live simulator run when the caller separately provides
    the pinned runtime/assets/licence and authorized execution environment.
    CPU tests exercise the request boundary with fake modules only.
    """

    raw = _mapping(request, field="live DART request")
    expected = {
        "schema",
        "pins",
        "source",
        "train_selection",
        "binding",
        "teacher_spec",
        "teacher_reference",
        "teacher_receipt",
        "runtime_session",
        "calibration",
        "source_membership",
        "private_writer",
        "run_id",
        "max_control_steps",
        "mode",
        "original_gaussian",
        "bounded_dart",
    }
    if set(raw) != expected or raw.get("schema") != "p107-dart-live-grasp-request-v1":
        raise RecoveryContractError("live DART request needs the exact p107-dart-live-grasp-request-v1 fields")
    source = _construct(CandidateSourceReceipt, raw["source"], field="source")
    train_selection = _construct(TrainSourceSelection, raw["train_selection"], field="train_selection")
    binding = _parse_binding(raw["binding"])
    pins = _parse_factory_pins(raw["pins"])
    teacher_spec = dict(_mapping(raw["teacher_spec"], field="teacher_spec"))
    teacher_reference = dict(_mapping(raw["teacher_reference"], field="teacher_reference"))
    teacher = _construct(TeacherReceipt, raw["teacher_receipt"], field="teacher_receipt")
    runtime_plan = _parse_fresh_runtime_plan(raw["runtime_session"])
    calibration = _parse_calibration(raw["calibration"])
    membership_raw = _mapping(raw["source_membership"], field="source_membership")
    index_root = membership_raw.get("index_root")
    inventory_seal = membership_raw.get("expected_inventory_seal_sha256")
    protocol_path = membership_raw.get("protocol_path")
    protocol_sha = membership_raw.get("expected_protocol_source_sha256")
    if not isinstance(index_root, str) or not index_root:
        raise RecoveryContractError("source_membership.index_root must be a nonempty path")
    if protocol_path is not None and (not isinstance(protocol_path, str) or not protocol_path):
        raise RecoveryContractError("source_membership.protocol_path must be null or a nonempty path")
    reader = DataSealedSourceGroupIndexReader(
        Path(index_root),
        expected_inventory_seal_sha256=inventory_seal,
        protocol_path=None if protocol_path is None else Path(protocol_path),
        expected_protocol_source_sha256=protocol_sha,
    )
    verified_membership = reader.verify_dart_membership(source, calibration)
    writer_raw = dict(_mapping(raw["private_writer"], field="private_writer"))
    allowed_writer = {
        "output",
        "source_episode_id",
        "collector_code_sha256",
        "capture_adapter_sha256",
        "actor_observation_schema_sha256",
        "outcome_evidence",
    }
    if set(writer_raw) != allowed_writer:
        raise RecoveryContractError("private_writer must name exact output/provenance and explicit outcome_evidence")
    outcome = writer_raw.pop("outcome_evidence")
    if outcome is not None:
        raise RecoveryContractError(
            "live factory request private_writer.outcome_evidence must be null; "
            "this fresh rollout has no post-run outcome evidence provider"
        )
    writer_output = _preflight_private_publication_output(writer_raw.pop("output"))
    run_id, max_control_steps = raw["run_id"], raw["max_control_steps"]
    if not isinstance(run_id, str) or not run_id or run_id != source.collection_run_id:
        raise RecoveryContractError("run_id must equal the sealed candidate source collection_run_id")
    if type(max_control_steps) is not int or max_control_steps < 1:
        raise RecoveryContractError("max_control_steps must be a positive integer")
    mode = raw["mode"]
    if mode == "original_gaussian_one_pass_partial_dart":
        collection_args = _parse_original_collection(raw["original_gaussian"])
    elif mode == "dart_inspired_bounded_actual_clean_recovery":
        collection_args = _parse_bounded_collection(raw["bounded_dart"])
    else:
        raise RecoveryContractError(
            "live DART request mode must be original_gaussian_one_pass_partial_dart or "
            "dart_inspired_bounded_actual_clean_recovery"
        )
    factory = DartOgGraspFactory.create(
        pins=pins,
        source=source,
        train_selection=train_selection,
        verified_membership=verified_membership,
        binding=binding,
        teacher_spec=teacher_spec,
        teacher_reference=teacher_reference,
        run_id=run_id,
        max_control_steps=max_control_steps,
        runtime_plan=runtime_plan,
    )
    # ``run_live_grasp_collection`` owns shutdown once it starts.  Keep the
    # small gap between factory creation and that call covered too: config or
    # callback construction cannot leak a fresh evaluator/reset when it fails.
    try:
        runtime_session = factory.runtime_session
        writer_config = PrivateEpisodeWriterConfig(
            output=writer_output,
            source=source,
            verified_membership=verified_membership,
            runtime_session=runtime_session,
            teacher=teacher,
            outcome_evidence=None,
            fresh_reset_payload=factory.fresh_reset_payload,
            **writer_raw,
        )
        writer = build_private_episode_bundle_writer(writer_config)
    except BaseException:
        factory.close()
        raise
    if mode == "original_gaussian_one_pass_partial_dart":
        noise, steps, original_binding = collection_args
        return run_live_grasp_collection(
            factory=factory,
            mode=mode,
            teacher_receipt=teacher,
            source=source,
            source_membership_reader=reader,
            runtime_session=runtime_session,
            calibration=calibration,
            intent_bundle_id=binding.intent_bundle_id,
            writer=writer,
            original_noise=noise,
            original_steps=steps,
            original_binding=original_binding,
        )
    noise, injection, chunks, limits = collection_args
    return run_live_grasp_collection(
        factory=factory,
        mode="dart_inspired_bounded_actual_clean_recovery",
        teacher_receipt=teacher,
        source=source,
        source_membership_reader=reader,
        runtime_session=runtime_session,
        calibration=calibration,
        intent_bundle_id=binding.intent_bundle_id,
        writer=writer,
        bounded_noise=noise,
        noisy_injection_steps=injection,
        recovery_chunk_lengths=chunks,
        limits=limits,
    )
