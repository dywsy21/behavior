"""Candidate-only fresh-rollout bridge for a privileged R1Pro GRASP teacher.

This module intentionally has no OmniGibson import.  An LC-owned integration
may inject the already reviewed ``PoseTeacher``, ``SafeServo``,
``PrivilegedReader`` and ``LocalOutcome`` objects into the small protocols
below.  Keeping that boundary explicit makes it impossible for a CPU fake to
look like a live simulator adapter or for scene truth to become actor input.

The bridge is deliberately limited to a one-skill GRASP pilot.  It does not
replay demonstration actions, restore a snapshot, use a model checkpoint, or
claim official task success.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import struct
from typing import Any, Callable, Mapping, Protocol, Sequence

from .common import RecoveryContractError, canonical_sha256, require_sha256, validate_raw23_action
from .dart_collection import (
    AppliedActionReceipt,
    CanonicalSourceGroupMembership,
    DartObservation,
    TeacherCommand,
    VerifiedDartSourceMembership,
)


_PRIVATE_KEYS = frozenset(
    {
        "base_world",
        "contact_pairs",
        "finger_contact",
        "forbidden_contacts",
        "goal_world",
        "hand_poses",
        "held",
        "object_scope",
        "prim_path",
        "private",
        "target_pose",
        "target_uid",
    }
)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecoveryContractError(f"{field} must be non-empty text")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise RecoveryContractError(f"{field} must be finite")
    return float(value)


def _mapping_bool(value: Mapping[str, object], name: str, *, allow_unknown: bool = False) -> dict[str, bool | None]:
    if set(value) != {"left", "right"}:
        raise RecoveryContractError(f"{name} must name exactly left and right")
    result: dict[str, bool | None] = {}
    for arm, item in value.items():
        if item is None and allow_unknown:
            result[arm] = None
        elif type(item) is bool:
            result[arm] = item
        else:
            raise RecoveryContractError(f"{name}.{arm} must be a boolean" + (" or None" if allow_unknown else ""))
    return result


def raw23_wire_bytes(action: Sequence[float]) -> bytes:
    """Canonical little-endian float32 wire representation for native R1Pro action."""

    values = validate_raw23_action(action)
    try:
        return struct.pack("<23f", *values)
    except OverflowError as exc:  # float64 finite values can still overflow float32.
        raise RecoveryContractError("raw23 values must be representable as float32") from exc


def raw23_wire_sha256(action: Sequence[float]) -> str:
    return sha256(raw23_wire_bytes(action)).hexdigest()


def encode_r1pro_raw23(*, base_qvel: Sequence[float], joint_q18: Sequence[float], gripper: Sequence[float]) -> tuple[float, ...]:
    """Encode the reviewed R1Pro order without 27-D model padding.

    ``joint_q18`` is [trunk4, left_arm7, right_arm7].  The result is
    [base3, trunk4, left_arm7, left_gripper, right_arm7, right_gripper].
    """

    if len(base_qvel) != 3 or len(joint_q18) != 18 or len(gripper) != 2:
        raise RecoveryContractError("R1Pro requires base3, joint18 and gripper2")
    values = [
        *base_qvel,
        *joint_q18[:4],
        *joint_q18[4:11],
        gripper[0],
        *joint_q18[11:18],
        gripper[1],
    ]
    return tuple(validate_raw23_action(values))


@dataclass(frozen=True)
class RootModelPoseReview:
    """Static object-local geometry review, honestly non-human and non-actor."""

    review_id: str
    evidence_sha256: str
    human_reviewed: bool = False
    reviewer_kind: str = "root_model_visual_geometry"

    def __post_init__(self) -> None:
        _text(self.review_id, "review_id")
        require_sha256(self.evidence_sha256, field="evidence_sha256")
        if self.human_reviewed is not False or self.reviewer_kind != "root_model_visual_geometry":
            raise RecoveryContractError("pose review must not masquerade as a human review")

    def public(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "evidence_sha256": self.evidence_sha256,
            "human_reviewed": False,
            "reviewer_kind": self.reviewer_kind,
            "actor_visible": False,
        }


@dataclass(frozen=True)
class SealedGraspBinding:
    """The sealed, TRAIN-only intent an adapter may use for one generic GRASP."""

    source_group_id: str
    source_release_sha256: str
    source_group_member_sha256: str
    source_group_index_manifest_sha256: str
    source_role: str
    original_split: str
    intent_bundle_id: str
    intent_sha256: str
    target_name: str
    target_category: str
    hand: str
    goal_pose_local_sha256: str
    pose_review: RootModelPoseReview

    def __post_init__(self) -> None:
        for field in ("intent_bundle_id", "target_name", "target_category"):
            _text(getattr(self, field), field)
        for field in (
            "source_group_id",
            "source_release_sha256",
            "source_group_member_sha256",
            "source_group_index_manifest_sha256",
            "intent_sha256",
            "goal_pose_local_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        if self.source_role not in {"student_candidate", "annotation_calibration"} or self.original_split != "train":
            raise RecoveryContractError("privileged pose GRASP is limited to canonical TRAIN candidate/calibration roles")
        if self.hand not in {"left", "right"}:
            raise RecoveryContractError("GRASP teacher requires an explicit left/right hand")

    def teacher_public(self) -> dict[str, object]:
        """Public receipt deliberately omits raw object names and all geometry."""

        return {
            "source_group_id": self.source_group_id,
            "source_release_sha256": self.source_release_sha256,
            "source_group_member_sha256": self.source_group_member_sha256,
            "source_group_index_manifest_sha256": self.source_group_index_manifest_sha256,
            "source_role": self.source_role,
            "original_split": "train",
            "intent_bundle_id": self.intent_bundle_id,
            "intent_sha256": self.intent_sha256,
            "target_category": self.target_category,
            "hand": self.hand,
            "goal_pose_local_sha256": self.goal_pose_local_sha256,
            "pose_review": self.pose_review.public(),
            "actor_visible": False,
        }

    def assert_canonical_membership(self, membership: CanonicalSourceGroupMembership) -> None:
        """Bind the pose seed to the same DATA-owned source group as DART."""

        if not isinstance(membership, CanonicalSourceGroupMembership):
            raise RecoveryContractError("GRASP binding needs DATA canonical source membership")
        if (
            membership.source_group_id != self.source_group_id
            or membership.source_release_sha256 != self.source_release_sha256
            or membership.source_group_member_sha256 != self.source_group_member_sha256
            or membership.original_split != self.original_split
            or membership.usage_role != self.source_role
        ):
            raise RecoveryContractError("GRASP binding differs from DATA sealed source membership")

    def assert_verified_dart_membership(self, membership: VerifiedDartSourceMembership) -> None:
        """Require the source/index seal already returned by the DART reader."""

        if not isinstance(membership, VerifiedDartSourceMembership):
            raise RecoveryContractError("GRASP binding needs verified DART source membership")
        if membership.source_group_index_manifest_sha256 != self.source_group_index_manifest_sha256:
            raise RecoveryContractError("GRASP binding differs from DART source-index manifest")
        matches = [
            group
            for group in (membership.candidate, *membership.calibration_groups)
            if group.source_group_id == self.source_group_id
        ]
        if len(matches) != 1:
            raise RecoveryContractError("GRASP binding must identify exactly one verified DART source group")
        self.assert_canonical_membership(matches[0])


@dataclass(frozen=True)
class GenericGraspPilot:
    """Two-or-more-object registration for one shared controller implementation."""

    bindings: tuple[SealedGraspBinding, ...]
    controller_code_sha256: str
    controller_config_sha256: str

    def __post_init__(self) -> None:
        if len(self.bindings) < 2 or not all(isinstance(item, SealedGraspBinding) for item in self.bindings):
            raise RecoveryContractError("generic GRASP pilot needs at least two sealed bindings")
        for field in ("controller_code_sha256", "controller_config_sha256"):
            require_sha256(getattr(self, field), field=field)
        groups = [item.source_group_id for item in self.bindings]
        categories = {item.target_category for item in self.bindings}
        if len(set(groups)) != len(groups):
            raise RecoveryContractError("generic GRASP pilot cannot reuse a source group as a second object")
        if len(categories) < 2:
            raise RecoveryContractError("generic GRASP pilot needs at least two target object categories")

    def public(self) -> dict[str, object]:
        return {
            "schema": "p107_generic_privileged_pose_grasp_pilot_v1",
            "controller_code_sha256": self.controller_code_sha256,
            "controller_config_sha256": self.controller_config_sha256,
            "bindings": [item.teacher_public() for item in self.bindings],
            "training_eligible": False,
            "official_task_success": False,
        }


@dataclass(frozen=True)
class ActorContext:
    """Actor-allowed task and causal intent; private teacher values cannot enter."""

    task: str
    causal_intent: str

    def __post_init__(self) -> None:
        _text(self.task, "task")
        _text(self.causal_intent, "causal_intent")
        for value in (self.task, self.causal_intent):
            if any(key in value.lower() for key in _PRIVATE_KEYS):
                raise RecoveryContractError("actor text may not name a private teacher field")


def project_actor_observation(observation: DartObservation, context: ActorContext) -> dict[str, object]:
    """The only actor projection emitted by this bridge."""

    return {
        "task": context.task,
        "causal_intent": context.causal_intent,
        "observation": observation.public(),
        "actor_visible": True,
    }


@dataclass(frozen=True)
class PrivateGraspSnapshot:
    """Opaque teacher-side state, bound to one fresh actor observation.

    ``robot_state`` and ``servo_model`` deliberately never have public methods
    here.  They are handed to existing PoseTeacher / SafeServo only inside the
    teacher-side engine.
    """

    policy_clock: int
    state61_sha256: str
    observation_sha256: str
    target_name: str
    teacher_frame: Mapping[str, object]
    robot_state: Any
    servo_model: Any
    goal_world: Any
    base_world: Any
    grips: Sequence[float]
    fresh_query_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy_clock, int) or self.policy_clock < 0:
            raise RecoveryContractError("private snapshot policy_clock must be current and nonnegative")
        for field in ("state61_sha256", "observation_sha256", "fresh_query_receipt_sha256"):
            require_sha256(getattr(self, field), field=field)
        _text(self.target_name, "target_name")
        if not isinstance(self.teacher_frame, Mapping):
            raise RecoveryContractError("teacher_frame must be an explicit private mapping")
        if self.teacher_frame.get("target_uid") != self.target_name:
            raise RecoveryContractError("private frame target must match sealed target identity")
        _mapping_bool(self.teacher_frame.get("held", {}), "held", allow_unknown=True)
        _mapping_bool(self.teacher_frame.get("finger_contact", {}), "finger_contact")
        if self.teacher_frame.get("contacts_known") is not True:
            raise RecoveryContractError("missing contact registration is UNKNOWN, not a teacher action")
        if self.teacher_frame.get("payload_ok") is not True:
            raise RecoveryContractError("missing/protected payload evidence is not eligible for GRASP teaching")
        if self.teacher_frame.get("forbidden_contacts"):
            raise RecoveryContractError("new forbidden physical contact blocks teacher action")
        if len(self.grips) != 2:
            raise RecoveryContractError("private R1Pro gripper state must have two entries")
        tuple(_finite(value, "grips") for value in self.grips)

    @classmethod
    def from_observation(
        cls,
        observation: DartObservation,
        *,
        target_name: str,
        teacher_frame: Mapping[str, object],
        robot_state: Any,
        servo_model: Any,
        goal_world: Any,
        base_world: Any,
        grips: Sequence[float],
        fresh_query_receipt_sha256: str,
    ) -> "PrivateGraspSnapshot":
        return cls(
            policy_clock=observation.policy_clock,
            state61_sha256=observation.state61_sha256,
            observation_sha256=observation.observation_sha256,
            target_name=target_name,
            teacher_frame=teacher_frame,
            robot_state=robot_state,
            servo_model=servo_model,
            goal_world=goal_world,
            base_world=base_world,
            grips=tuple(_finite(value, "grips") for value in grips),
            fresh_query_receipt_sha256=fresh_query_receipt_sha256,
        )


class PrivilegedWorld(Protocol):
    """Live owner supplies current private state; no demo or snapshot API exists."""

    def fresh_snapshot(self, observation: DartObservation, binding: SealedGraspBinding) -> PrivateGraspSnapshot: ...


class ExistingPrivilegedReaderWorld:
    """Adapter for the existing PrivilegedReader / OGKinematics APIs.

    It intentionally receives already-created objects from the LC session owner:
    this module neither imports OmniGibson nor creates an evaluator.  The query
    receipt factory must return an opaque digest and may not publish its private
    inputs elsewhere.
    """

    def __init__(
        self,
        *,
        privileged_reader: Any,
        kinematics: Any,
        servo_model: Any,
        fresh_query_receipt: Callable[[DartObservation, Mapping[str, object], Any], str],
    ) -> None:
        if not all(callable(getattr(privileged_reader, name, None)) for name in ("read", "goal", "base")):
            raise RecoveryContractError("PrivilegedReader must provide read, goal and base")
        if not callable(getattr(kinematics, "state", None)):
            raise RecoveryContractError("OG kinematics must provide current state")
        if not callable(fresh_query_receipt):
            raise RecoveryContractError("fresh private query receipt factory must be callable")
        self._reader = privileged_reader
        self._kinematics = kinematics
        self._servo_model = servo_model
        self._fresh_query_receipt = fresh_query_receipt

    def fresh_snapshot(self, observation: DartObservation, binding: SealedGraspBinding) -> PrivateGraspSnapshot:
        # All four reads occur at clean_action time.  This is deliberately not a
        # cache and cannot reuse the previous reached state's scene truth.
        frame = self._reader.read(observation.policy_clock)
        state = self._kinematics.state()
        if not isinstance(frame, Mapping) or not hasattr(state, "gripper"):
            raise RecoveryContractError("existing private reader/kinematics returned incomplete current state")
        receipt = self._fresh_query_receipt(observation, frame, state)
        require_sha256(receipt, field="fresh private query receipt")
        return PrivateGraspSnapshot.from_observation(
            observation,
            target_name=binding.target_name,
            teacher_frame=frame,
            robot_state=state,
            servo_model=self._servo_model,
            goal_world=self._reader.goal(),
            base_world=self._reader.base(),
            grips=state.gripper,
            fresh_query_receipt_sha256=receipt,
        )


class PoseServoEngine(Protocol):
    def ranked_tokens(self, snapshot: PrivateGraspSnapshot, binding: SealedGraspBinding) -> Sequence[str]: ...

    def raw23_for_token(self, snapshot: PrivateGraspSnapshot, token: str) -> Sequence[float] | None: ...


class ExistingPoseServoEngine:
    """Thin injection wrapper around the existing PoseTeacher + SafeServo APIs.

    Constructors are injected instead of imported to keep this package CPU-only
    and to make the LC owner pin the exact reviewed implementation hashes.
    """

    def __init__(
        self,
        *,
        pose_teacher_class: Callable[..., Any],
        safe_servo_class: Callable[..., Any],
        token_to_action: Callable[[str], Any],
        servo_limits: Any = None,
    ) -> None:
        if not all(callable(value) for value in (pose_teacher_class, safe_servo_class, token_to_action)):
            raise RecoveryContractError("existing PoseTeacher, SafeServo and token codec must be explicit callables")
        self._pose_teacher_class = pose_teacher_class
        self._safe_servo_class = safe_servo_class
        self._token_to_action = token_to_action
        self._servo_limits = servo_limits

    def ranked_tokens(self, snapshot: PrivateGraspSnapshot, binding: SealedGraspBinding) -> Sequence[str]:
        # A fresh object is rebuilt from current verified grasp state.  The
        # persistent CLOSE flag is reconstructed only from actual current target
        # holding, never from a prior demo, teacher proposal, or action request.
        teacher = self._pose_teacher_class({"verb": "GRASP", "hand": binding.hand})
        held = _mapping_bool(snapshot.teacher_frame["held"], "held", allow_unknown=True)
        if held[binding.hand] is None:
            raise RecoveryContractError("UNKNOWN current target hold blocks GRASP teacher")
        if held[binding.hand] is True:
            teacher.close_issued = True
        ranked = teacher.ranked(
            snapshot.robot_state,
            snapshot.teacher_frame,
            snapshot.goal_world,
            snapshot.base_world,
            snapshot.grips,
            snapshot.servo_model,
        )
        if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
            raise RecoveryContractError("PoseTeacher must return an ordered token sequence")
        result = tuple(_text(token, "PoseTeacher token") for token in ranked)
        if len(set(result)) != len(result):
            raise RecoveryContractError("PoseTeacher emitted duplicate candidate tokens")
        return result

    def raw23_for_token(self, snapshot: PrivateGraspSnapshot, token: str) -> Sequence[float] | None:
        action = self._token_to_action(token)
        servo = self._safe_servo_class(
            snapshot.servo_model,
            snapshot.robot_state,
            gripper_command=snapshot.grips,
            limits=self._servo_limits,
        )
        if servo.begin(action, snapshot.robot_state, carry=True) is not True:
            return None
        if getattr(servo, "done", False):
            return None
        return validate_raw23_action(servo.next_action(snapshot.robot_state))


class PrivilegedPoseGraspTeacher:
    """Fresh-query, private-pose teacher satisfying ``DartTeacher``.

    The teacher never exposes private world values.  A live owner must supply a
    `TeacherReceipt` separately with hashes of this adapter, the injected
    existing components, configuration and postcondition spec.
    """

    def __init__(self, *, binding: SealedGraspBinding, world: PrivilegedWorld, engine: PoseServoEngine) -> None:
        self.binding = binding
        self._world = world
        self._engine = engine

    def clean_action(self, observation: DartObservation, *, intent_bundle_id: str) -> TeacherCommand:
        if not isinstance(observation, DartObservation) or observation.observation_fresh is not True:
            raise RecoveryContractError("teacher requires a fresh DartObservation")
        if intent_bundle_id != self.binding.intent_bundle_id:
            raise RecoveryContractError("teacher intent differs from sealed GRASP intent")
        snapshot = self._world.fresh_snapshot(observation, self.binding)
        if not isinstance(snapshot, PrivateGraspSnapshot):
            raise RecoveryContractError("privileged world must return a bound PrivateGraspSnapshot")
        if (
            snapshot.policy_clock != observation.policy_clock
            or snapshot.state61_sha256 != observation.state61_sha256
            or snapshot.observation_sha256 != observation.observation_sha256
            or snapshot.target_name != self.binding.target_name
        ):
            raise RecoveryContractError("private teacher state is stale, mismatched, or targets another object")
        tokens = self._engine.ranked_tokens(snapshot, self.binding)
        for token in tokens:
            proposed = self._engine.raw23_for_token(snapshot, token)
            if proposed is not None:
                action = tuple(validate_raw23_action(proposed))
                receipt = canonical_sha256(
                    {
                        "schema": "p107_privileged_pose_grasp_query_v1",
                        "binding": self.binding.teacher_public(),
                        "policy_clock": observation.policy_clock,
                        "state61_sha256": observation.state61_sha256,
                        "observation_sha256": observation.observation_sha256,
                        "private_query_receipt_sha256": snapshot.fresh_query_receipt_sha256,
                        "selected_token": token,
                        "raw23_wire_sha256": raw23_wire_sha256(action),
                    }
                )
                return TeacherCommand(
                    clean_intended23=action,
                    observed_policy_clock=observation.policy_clock,
                    observed_state61_sha256=observation.state61_sha256,
                    observed_observation_sha256=observation.observation_sha256,
                    intent_bundle_id=intent_bundle_id,
                    fresh_query_receipt_sha256=receipt,
                )
        raise RecoveryContractError("no eligible current-state PoseTeacher/SafeServo candidate")


class FreshRuntimeHooks(Protocol):
    """Only seam a future LC owner may bind to one fresh official session."""

    def observe_current(self) -> DartObservation: ...

    def apply_current_raw23(self, requested23: Sequence[float], observation: DartObservation) -> AppliedActionReceipt: ...


class FreshRolloutRaw23Runtime:
    """No-restore runtime bridge with strict raw byte and freshness checks."""

    snapshot_mode = "no_restore_fresh_rollout"

    def __init__(self, hooks: FreshRuntimeHooks) -> None:
        self._hooks = hooks
        self._last: DartObservation | None = None
        self._last_clock: int | None = None

    def observe(self) -> DartObservation:
        observation = self._hooks.observe_current()
        if not isinstance(observation, DartObservation) or observation.observation_fresh is not True:
            raise RecoveryContractError("runtime hook must provide current fresh RGB/proprio observation")
        if self._last_clock is not None and observation.policy_clock <= self._last_clock:
            raise RecoveryContractError("runtime observations need strictly increasing policy clocks")
        self._last = observation
        self._last_clock = observation.policy_clock
        return observation

    def apply_raw23(self, requested23: Sequence[float]) -> AppliedActionReceipt:
        requested = tuple(validate_raw23_action(requested23))
        observation = self._last
        if observation is None:
            raise RecoveryContractError("apply_raw23 requires a fresh preceding observation")
        # Invalidate before calling an external runtime.  If it throws after
        # physically applying a control, the caller cannot retry on a stale
        # observation and accidentally execute twice.
        self._last = None
        applied = self._hooks.apply_current_raw23(requested, observation)
        if not isinstance(applied, AppliedActionReceipt) or applied.status != "APPLIED":
            raise RecoveryContractError("live runtime did not apply the requested raw23 action")
        if (
            applied.pre_action_policy_clock != observation.policy_clock
            or applied.pre_action_state61_sha256 != observation.state61_sha256
            or applied.pre_action_observation_sha256 != observation.observation_sha256
        ):
            raise RecoveryContractError("runtime action receipt is stale or bound to another observation")
        if tuple(applied.applied23) != requested:
            raise RecoveryContractError("raw23 controller substitution is not a clean execution")
        if (
            applied.applied_action_bytes_sha256 != raw23_wire_sha256(requested)
            or applied.applied_action_bytes_sha256 != raw23_wire_sha256(applied.applied23)
        ):
            raise RecoveryContractError("runtime raw23 byte receipt does not match requested action")
        return applied


@dataclass(frozen=True)
class GraspPostconditionConfig:
    """Pinned local-GRASP physics requirement; never an official task predicate."""

    physics_dt_seconds: float
    stable_ticks: int = 12
    stable_seconds: float = 0.5
    target_in_hand_translation_m: float = 0.004
    target_in_hand_rotation_deg: float = 3.0
    target_lift_m: float = 0.03
    hand_lift_m: float = 0.025

    def __post_init__(self) -> None:
        for field in (
            "physics_dt_seconds",
            "stable_seconds",
            "target_in_hand_translation_m",
            "target_in_hand_rotation_deg",
            "target_lift_m",
            "hand_lift_m",
        ):
            if _finite(getattr(self, field), field) <= 0:
                raise RecoveryContractError(f"{field} must be positive")
        if type(self.stable_ticks) is not int or self.stable_ticks < 12:
            raise RecoveryContractError("GRASP requires at least twelve distinct physics ticks")
        if self.stable_ticks * self.physics_dt_seconds + 1e-12 < self.stable_seconds:
            raise RecoveryContractError("physics cadence cannot substantiate the pinned stable hold duration")

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": "p107_local_grasp_postcondition_v2",
                "physics_dt_seconds": self.physics_dt_seconds,
                "stable_ticks": self.stable_ticks,
                "stable_seconds": self.stable_seconds,
                "target_in_hand_translation_m": self.target_in_hand_translation_m,
                "target_in_hand_rotation_deg": self.target_in_hand_rotation_deg,
                "target_lift_m": self.target_lift_m,
                "hand_lift_m": self.hand_lift_m,
                "official_task_success": False,
            }
        )


@dataclass(frozen=True)
class LocalGraspEvidence:
    """Teacher-only physical evidence, deliberately not a training/outcome release."""

    status: str
    reason: str
    evidence_sha256: str
    postcondition_sha256: str
    official_task_success: bool = False
    actor_visible: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"SUCCEEDED", "FAILED", "UNKNOWN", "IN_PROGRESS"}:
            raise RecoveryContractError("invalid local GRASP evidence status")
        _text(self.reason, "local GRASP evidence reason")
        require_sha256(self.evidence_sha256, field="evidence_sha256")
        require_sha256(self.postcondition_sha256, field="postcondition_sha256")
        if self.official_task_success is not False or self.actor_visible is not False:
            raise RecoveryContractError("local GRASP evidence may not claim official success or actor visibility")


def certify_local_grasp(
    *,
    binding: SealedGraspBinding,
    config: GraspPostconditionConfig,
    frames: Sequence[Mapping[str, object]],
    issued_tokens: Sequence[str | None],
    local_outcome_class: Callable[..., Any],
) -> LocalGraspEvidence:
    """Run the reviewed LocalOutcome predicate with extra cadence validation.

    The supplied frames remain private.  A missing timestamp, duplicate tick,
    UNKNOWN hold/contact, or any unavailable native API yields UNKNOWN rather
    than invented solid contact or recovery success.
    """

    if len(frames) != len(issued_tokens) or not frames:
        raise RecoveryContractError("physical evidence needs one token per nonempty frame sequence")
    ticks: list[int] = []
    times: list[float] = []
    copied: list[dict[str, object]] = []
    for index, raw in enumerate(frames):
        if not isinstance(raw, Mapping):
            raise RecoveryContractError("physical evidence frame must be a mapping")
        frame = dict(raw)
        if frame.get("target_uid") != binding.target_name:
            raise RecoveryContractError("physical evidence target does not match sealed GRASP target")
        tick = frame.get("tick")
        timestamp = frame.get("physics_time_seconds")
        if type(tick) is not int or tick < 0 or (ticks and tick != ticks[-1] + 1):
            raise RecoveryContractError("physical evidence needs distinct contiguous simulator ticks")
        now = _finite(timestamp, "physics_time_seconds")
        if times and now <= times[-1]:
            raise RecoveryContractError("physical evidence timestamps must be strictly increasing")
        if times and not math.isclose(now - times[-1], config.physics_dt_seconds, rel_tol=0.0, abs_tol=1e-9):
            raise RecoveryContractError("physics timestamps do not match the pinned physics_dt_seconds")
        ticks.append(tick)
        times.append(now)
        # LocalOutcome does not know this additional receipt field.
        frame.pop("physics_time_seconds", None)
        copied.append(frame)
    if len(copied) < config.stable_ticks or times[-1] - times[-config.stable_ticks] + 1e-12 < config.stable_seconds:
        status, reason = "UNKNOWN", "INSUFFICIENT_DISTINCT_TICKS_OR_STABLE_SECONDS"
    else:
        verifier = local_outcome_class(
            {"verb": "GRASP", "hand": binding.hand, "target": binding.target_name},
            stable_ticks=config.stable_ticks,
        )
        verdict: Mapping[str, object] = {"outcome": "UNKNOWN", "reason": "NO_PHYSICS_EVIDENCE"}
        try:
            for frame, token in zip(copied, issued_tokens):
                verdict = verifier.update(frame, token)
            status = str(verdict.get("outcome", "UNKNOWN"))
            reason = str(verdict.get("reason", "LOCAL_OUTCOME_UNAVAILABLE"))
        except (KeyError, TypeError, ValueError, RuntimeError):
            status, reason = "UNKNOWN", "LOCAL_OUTCOME_UNAVAILABLE"
        if status == "SUCCEEDED":
            # Existing LocalOutcome pins 3cm/2.5cm lift and 4mm/3deg
            # target-in-hand stability. Reject a caller pretending another
            # configuration certified the same evidence.
            expected = (0.03, 0.025, 0.004, 3.0)
            actual = (
                config.target_lift_m,
                config.hand_lift_m,
                config.target_in_hand_translation_m,
                config.target_in_hand_rotation_deg,
            )
            if actual != expected:
                status, reason = "UNKNOWN", "LOCAL_OUTCOME_THRESHOLD_CONFIG_MISMATCH"
    evidence = canonical_sha256(
        {
            "schema": "p107_local_grasp_evidence_v2",
            "target_name_sha256": sha256(binding.target_name.encode()).hexdigest(),
            "hand": binding.hand,
            "ticks": ticks,
            "times": times,
            "issued_tokens": list(issued_tokens),
            "status": status,
            "reason": reason,
            "postcondition_sha256": config.sha256,
            "official_task_success": False,
        }
    )
    return LocalGraspEvidence(status=status, reason=reason, evidence_sha256=evidence, postcondition_sha256=config.sha256)
