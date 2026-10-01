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

# Deliberately module-private construction gate.  It is not a cryptographic
# boundary against a hostile Python process; it prevents ordinary callers from
# manufacturing an "acknowledged" action by filling a public dataclass.
_TRUSTED_TRACE_MINT_CAPABILITY = object()


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
    public_task_description_sha256: str
    public_causal_intent_sha256: str
    public_description_registry_sha256: str
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
            "public_task_description_sha256",
            "public_causal_intent_sha256",
            "public_description_registry_sha256",
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
            "public_task_description_sha256": self.public_task_description_sha256,
            "public_causal_intent_sha256": self.public_causal_intent_sha256,
            "public_description_registry_sha256": self.public_description_registry_sha256,
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
class ApprovedPublicText:
    """One already-rendered actor description, pinned by bytes and registry."""

    text: str
    text_sha256: str
    description_registry_sha256: str
    kind: str

    def __post_init__(self) -> None:
        _text(self.text, "public actor text")
        require_sha256(self.text_sha256, field="public actor text_sha256")
        require_sha256(self.description_registry_sha256, field="public description registry")
        if self.kind not in {"task", "causal_intent"}:
            raise RecoveryContractError("public actor text kind must be task or causal_intent")
        if sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise RecoveryContractError("public actor text bytes do not match its sealed description hash")


@dataclass(frozen=True)
class OpaqueMediaReference:
    """A hash-only reference to current RGB/proprio; no path or free-form ref."""

    observation_sha256: str
    kind: str = "fresh_rgb_proprio_sha256_only"

    def __post_init__(self) -> None:
        require_sha256(self.observation_sha256, field="opaque media observation_sha256")
        if self.kind != "fresh_rgb_proprio_sha256_only":
            raise RecoveryContractError("actor media reference must be the fixed hash-only form")


@dataclass(frozen=True)
class ActorContext:
    """Structured actor input; raw private identifiers are rejected at projection."""

    task: ApprovedPublicText
    causal_intent: ApprovedPublicText
    media: OpaqueMediaReference

    def __post_init__(self) -> None:
        if not isinstance(self.task, ApprovedPublicText) or self.task.kind != "task":
            raise RecoveryContractError("actor task must be a sealed public task description")
        if not isinstance(self.causal_intent, ApprovedPublicText) or self.causal_intent.kind != "causal_intent":
            raise RecoveryContractError("actor intent must be a sealed public causal description")
        if not isinstance(self.media, OpaqueMediaReference):
            raise RecoveryContractError("actor media must use the strict opaque media reference")


def project_actor_observation(
    observation: DartObservation, context: ActorContext, binding: SealedGraspBinding
) -> dict[str, object]:
    """Emit only binding-approved public text and hash-only current media receipt."""

    if not isinstance(observation, DartObservation) or not isinstance(context, ActorContext):
        raise RecoveryContractError("actor projection requires typed fresh observation and context")
    if context.media.observation_sha256 != observation.observation_sha256:
        raise RecoveryContractError("opaque media reference must bind the current fresh observation bytes")
    expected = {
        "task": binding.public_task_description_sha256,
        "causal_intent": binding.public_causal_intent_sha256,
    }
    for value in (context.task, context.causal_intent):
        if value.description_registry_sha256 != binding.public_description_registry_sha256:
            raise RecoveryContractError("actor text is not from the binding's approved public description registry")
        if value.text_sha256 != expected[value.kind]:
            raise RecoveryContractError("actor text does not match the binding's approved public description")
        lowered = value.text.casefold()
        if binding.target_name.casefold() in lowered or binding.source_group_id in lowered:
            raise RecoveryContractError("actor text leaks a private target or source identifier")
        if any(key in lowered for key in _PRIVATE_KEYS):
            raise RecoveryContractError("actor text may not name a private teacher field")
    # Do not call DartObservation.public(): it would forward arbitrary
    # observation_ref text that this adapter cannot prove to be non-private.
    return {
        "task": context.task.text,
        "causal_intent": context.causal_intent.text,
        "description_registry_sha256": binding.public_description_registry_sha256,
        "observation": {
            "policy_clock": observation.policy_clock,
            "state61_sha256": observation.state61_sha256,
            "observation_sha256": observation.observation_sha256,
            "media_kind": context.media.kind,
            "observation_fresh": True,
            "actor_visible": True,
        },
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

    def acknowledge_actual_clean_application(self, applied: AppliedActionReceipt) -> None: ...

    def discard_unapplied_proposal(self, applied: AppliedActionReceipt) -> None: ...


@dataclass(frozen=True)
class _PendingCleanProposal:
    snapshot: PrivateGraspSnapshot
    token: str
    intended23: tuple[float, ...]


def _applied_receipt_identity(applied: AppliedActionReceipt) -> str:
    """Stable identity for a one-time runtime receipt, never an actor value."""

    if not isinstance(applied, AppliedActionReceipt):
        raise RecoveryContractError("expected an applied raw23 receipt")
    return canonical_sha256(
        {
            "status": applied.status,
            "applied23": list(applied.applied23),
            "applied_action_bytes_sha256": applied.applied_action_bytes_sha256,
            "runtime_step_receipt_sha256": applied.runtime_step_receipt_sha256,
            "pre_action_policy_clock": applied.pre_action_policy_clock,
            "pre_action_state61_sha256": applied.pre_action_state61_sha256,
            "pre_action_observation_sha256": applied.pre_action_observation_sha256,
        }
    )


def _assert_pending_receipt(
    pending: _PendingCleanProposal, applied: AppliedActionReceipt, *, require_different: bool
) -> None:
    """Bind an actual runtime step to a preview without advancing on a proposal."""

    if not isinstance(applied, AppliedActionReceipt) or applied.status != "APPLIED":
        raise RecoveryContractError("a pending teacher proposal needs an APPLIED runtime receipt")
    snapshot = pending.snapshot
    if (
        applied.pre_action_policy_clock != snapshot.policy_clock
        or applied.pre_action_state61_sha256 != snapshot.state61_sha256
        or applied.pre_action_observation_sha256 != snapshot.observation_sha256
    ):
        raise RecoveryContractError("runtime receipt is stale or belongs to another clean query")
    actual = tuple(validate_raw23_action(applied.applied23))
    if applied.applied_action_bytes_sha256 != raw23_wire_sha256(actual):
        raise RecoveryContractError("runtime receipt bytes do not match its applied raw23 action")
    exact = actual == pending.intended23 and applied.applied_action_bytes_sha256 == raw23_wire_sha256(pending.intended23)
    if require_different and exact:
        raise RecoveryContractError("a clean executed action must be acknowledged, not discarded")
    if not require_different and not exact:
        raise RecoveryContractError("runtime receipt did not execute the exact clean raw23 proposal")


class TrustedGraspAction:
    """Teacher-private proof of one exact clean command actually run by its runtime.

    It has no public constructor: it is minted only after a fresh runtime's
    one-time receipt has been consumed and the corresponding native
    ``PoseTeacher.executed`` callback accepted the action.
    """

    def __init__(
        self,
        *,
        mint_capability: object,
        authority: object,
        binding_identity: str,
        token: str,
        command: TeacherCommand,
        applied: AppliedActionReceipt,
    ) -> None:
        if mint_capability is not _TRUSTED_TRACE_MINT_CAPABILITY:
            raise RecoveryContractError("trusted GRASP actions may only be minted after runtime acknowledgement")
        self._authority = authority
        self._binding_identity = binding_identity
        self._token = _text(token, "executed PoseTeacher token")
        self._command = command
        self._applied = applied

    @property
    def token(self) -> str:
        """Teacher-only selected native token; never place this in actor input."""

        return self._token

    @property
    def runtime_step_receipt_sha256(self) -> str:
        return self._applied.runtime_step_receipt_sha256


class TrustedGraspEvidenceTrace:
    """Private local-physics trace rooted in a runtime-acknowledged command.

    A caller cannot upgrade arbitrary ``frames``/``RIGHT_CLOSE`` strings to a
    certified outcome: the trace is opened by the teacher with an opaque,
    instance-specific action proof.  A future live recorder still must supply
    the private physics frames; this CPU seam deliberately does not claim such
    a recorder is already bound.
    """

    def __init__(
        self, *, mint_capability: object, authority: object, binding_identity: str, first_action: TrustedGraspAction
    ) -> None:
        if mint_capability is not _TRUSTED_TRACE_MINT_CAPABILITY:
            raise RecoveryContractError("local evidence traces may only be opened by the teacher")
        if first_action._authority is not authority or first_action._binding_identity != binding_identity:
            raise RecoveryContractError("local evidence trace needs this teacher's trusted executed action")
        self._authority = authority
        self._binding_identity = binding_identity
        self._actions = [first_action]
        self._entries: list[tuple[dict[str, object], TrustedGraspAction | None]] = []

    def record_private_physics_frame(
        self, frame: Mapping[str, object], *, action: TrustedGraspAction | None = None
    ) -> None:
        if not isinstance(frame, Mapping):
            raise RecoveryContractError("private physics frame must be a mapping")
        if action is not None:
            if action._authority is not self._authority or action._binding_identity != self._binding_identity:
                raise RecoveryContractError("physics frame action belongs to another teacher/binding")
            received = frame.get("runtime_step_receipt_sha256")
            if received != action.runtime_step_receipt_sha256:
                raise RecoveryContractError("physics frame lacks the matching actual runtime-step receipt")
            if action not in self._actions:
                self._actions.append(action)
        self._entries.append((dict(frame), action))

    def _trusted_entries(
        self, *, authority: object, binding_identity: str
    ) -> tuple[tuple[dict[str, object], TrustedGraspAction | None], ...]:
        if authority is not self._authority or binding_identity != self._binding_identity:
            raise RecoveryContractError("local evidence trace belongs to another teacher/binding")
        return tuple(self._entries)


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
        self._binding_identity: str | None = None
        self._teacher: Any | None = None
        self._pending: _PendingCleanProposal | None = None

    @staticmethod
    def _binding_identity_for(binding: SealedGraspBinding) -> str:
        # Private identity stays in-process; no actor/public receipt receives it.
        return canonical_sha256(
            {
                "source_group_id": binding.source_group_id,
                "target_name": binding.target_name,
                "hand": binding.hand,
                "intent_sha256": binding.intent_sha256,
            }
        )

    def _teacher_for(self, snapshot: PrivateGraspSnapshot, binding: SealedGraspBinding) -> Any:
        identity = self._binding_identity_for(binding)
        held = _mapping_bool(snapshot.teacher_frame["held"], "held", allow_unknown=True)
        if held[binding.hand] is None:
            raise RecoveryContractError("UNKNOWN current target hold blocks GRASP teacher")
        if self._binding_identity is None:
            # A new session cannot infer a previous CLOSE lifecycle from scene
            # truth.  Fresh GRASP collection starts empty; joining an existing
            # hold would make teacher state unverifiable.
            if held[binding.hand] is True:
                raise RecoveryContractError("new GRASP teacher session cannot infer an already-held target lifecycle")
            self._binding_identity = identity
            self._teacher = self._pose_teacher_class({"verb": "GRASP", "hand": binding.hand})
        elif self._binding_identity != identity:
            raise RecoveryContractError("PoseTeacher engine may not switch sealed binding mid-session")
        if self._teacher is None:
            raise RecoveryContractError("PoseTeacher session was not initialized")
        return self._teacher

    def ranked_tokens(self, snapshot: PrivateGraspSnapshot, binding: SealedGraspBinding) -> Sequence[str]:
        if self._pending is not None:
            raise RecoveryContractError("previous PoseTeacher proposal needs explicit apply acknowledgement or discard")
        teacher = self._teacher_for(snapshot, binding)
        try:
            ranked = teacher.ranked(
                snapshot.robot_state,
                snapshot.teacher_frame,
                snapshot.goal_world,
                snapshot.base_world,
                snapshot.grips,
                snapshot.servo_model,
            )
        except RuntimeError as exc:
            raise RecoveryContractError("PoseTeacher rejected the current GRASP lifecycle") from exc
        if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
            raise RecoveryContractError("PoseTeacher must return an ordered token sequence")
        result = tuple(_text(token, "PoseTeacher token") for token in ranked)
        if len(set(result)) != len(result):
            raise RecoveryContractError("PoseTeacher emitted duplicate candidate tokens")
        return result

    def raw23_for_token(self, snapshot: PrivateGraspSnapshot, token: str) -> Sequence[float] | None:
        if self._pending is not None:
            raise RecoveryContractError("cannot preview another action while a PoseTeacher proposal is pending")
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
        result = tuple(validate_raw23_action(servo.next_action(snapshot.robot_state)))
        self._pending = _PendingCleanProposal(snapshot=snapshot, token=token, intended23=result)
        return result

    def acknowledge_actual_clean_application(self, applied: AppliedActionReceipt) -> None:
        """Advance native teacher lifecycle only after exact clean raw23 apply."""

        pending = self._pending
        if pending is None:
            raise RecoveryContractError("no PoseTeacher proposal awaits an actual clean acknowledgement")
        _assert_pending_receipt(pending, applied, require_different=False)
        if self._teacher is None:
            raise RecoveryContractError("missing PoseTeacher lifecycle for acknowledged action")
        try:
            self._teacher.executed(pending.token, pending.snapshot.teacher_frame)
        except RuntimeError as exc:
            raise RecoveryContractError("PoseTeacher rejected acknowledged native lifecycle transition") from exc
        self._pending = None

    def discard_unapplied_proposal(self, applied: AppliedActionReceipt) -> None:
        """Clear a noisy/rejected proposal without advancing PoseTeacher state."""

        pending = self._pending
        if pending is None:
            raise RecoveryContractError("no PoseTeacher proposal is available to discard")
        _assert_pending_receipt(pending, applied, require_different=True)
        self._pending = None


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
        self._pending: tuple[_PendingCleanProposal, TeacherCommand] | None = None
        # This capability is deliberately per teacher session.  It is never
        # serialized or projected and makes an unacknowledged proposal unable
        # to seed a local-success trace.
        self._trace_authority = object()

    @property
    def _binding_identity(self) -> str:
        return ExistingPoseServoEngine._binding_identity_for(self.binding)

    def clean_action(self, observation: DartObservation, *, intent_bundle_id: str) -> TeacherCommand:
        if self._pending is not None:
            raise RecoveryContractError("previous clean teacher proposal needs runtime acknowledgement or noisy discard")
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
                command = TeacherCommand(
                    clean_intended23=action,
                    observed_policy_clock=observation.policy_clock,
                    observed_state61_sha256=observation.state61_sha256,
                    observed_observation_sha256=observation.observation_sha256,
                    intent_bundle_id=intent_bundle_id,
                    fresh_query_receipt_sha256=receipt,
                )
                self._pending = (_PendingCleanProposal(snapshot=snapshot, token=token, intended23=action), command)
                return command
        raise RecoveryContractError("no eligible current-state PoseTeacher/SafeServo candidate")

    def acknowledge_runtime_clean_application(
        self, runtime: "FreshRolloutRaw23Runtime", applied: AppliedActionReceipt
    ) -> TrustedGraspAction:
        """Consume an exact runtime step, then (and only then) advance lifecycle.

        ``clean_action`` merely previews.  The separate runtime receipt prevents
        a caller from claiming a token was run, and advancing the existing
        native PoseTeacher is delayed until that receipt passes all raw23 and
        pre-state checks.
        """

        pending_pair = self._pending
        if pending_pair is None:
            raise RecoveryContractError("no clean teacher proposal awaits runtime acknowledgement")
        pending, command = pending_pair
        _assert_pending_receipt(pending, applied, require_different=False)
        if not isinstance(runtime, FreshRolloutRaw23Runtime):
            raise RecoveryContractError("clean teacher acknowledgement requires FreshRolloutRaw23Runtime")
        runtime._consume_verified_receipt(applied)
        try:
            self._engine.acknowledge_actual_clean_application(applied)
        finally:
            # The real control already occurred.  Never allow a retry to make
            # a stale raw23 receipt look like a second authorized control.
            self._pending = None
        return TrustedGraspAction(
            mint_capability=_TRUSTED_TRACE_MINT_CAPABILITY,
            authority=self._trace_authority,
            binding_identity=self._binding_identity,
            token=pending.token,
            command=command,
            applied=applied,
        )

    def acknowledge_runtime_noisy_application(
        self, runtime: "FreshRolloutRaw23Runtime", applied: AppliedActionReceipt
    ) -> None:
        """Discard an actually applied noisy action without advancing clean lifecycle."""

        pending_pair = self._pending
        if pending_pair is None:
            raise RecoveryContractError("no clean teacher proposal is available for noisy discard")
        pending, _command = pending_pair
        _assert_pending_receipt(pending, applied, require_different=True)
        if not isinstance(runtime, FreshRolloutRaw23Runtime):
            raise RecoveryContractError("noisy teacher discard requires FreshRolloutRaw23Runtime")
        runtime._consume_verified_receipt(applied)
        try:
            self._engine.discard_unapplied_proposal(applied)
        finally:
            self._pending = None

    def begin_local_grasp_evidence(self, action: TrustedGraspAction) -> TrustedGraspEvidenceTrace:
        """Open a private evidence trace only from one trusted runtime action."""

        return TrustedGraspEvidenceTrace(
            mint_capability=_TRUSTED_TRACE_MINT_CAPABILITY,
            authority=self._trace_authority,
            binding_identity=self._binding_identity,
            first_action=action,
        )


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
        self._unconsumed_receipt_ids: set[str] = set()

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
        self._unconsumed_receipt_ids.add(_applied_receipt_identity(applied))
        return applied

    def _consume_verified_receipt(self, applied: AppliedActionReceipt) -> None:
        """One-time internal handoff to the teacher acknowledgement boundary."""

        identity = _applied_receipt_identity(applied)
        if identity not in self._unconsumed_receipt_ids:
            raise RecoveryContractError("teacher acknowledgement needs an unconsumed receipt from this exact runtime")
        self._unconsumed_receipt_ids.remove(identity)


@dataclass(frozen=True)
class GraspPostconditionConfig:
    """Pinned local-GRASP physics requirement; never an official task predicate."""

    physics_dt_seconds: float
    physics_clock_receipt_sha256: str
    stable_ticks: int = 12
    stable_seconds: float = 0.5
    target_in_hand_translation_m: float = 0.004
    target_in_hand_rotation_deg: float = 3.0
    target_lift_m: float = 0.03
    hand_lift_m: float = 0.025

    def __post_init__(self) -> None:
        require_sha256(self.physics_clock_receipt_sha256, field="physics_clock_receipt_sha256")
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
                "physics_clock_receipt_sha256": self.physics_clock_receipt_sha256,
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
    trace: object,
    local_outcome_class: Callable[..., Any],
) -> LocalGraspEvidence:
    """Run LocalOutcome only over a teacher/runtime-bound private trace.

    Bare caller-supplied ``frames`` and ``issued_tokens`` used to permit a
    forged ``RIGHT_CLOSE`` to obtain local success.  This entry point now
    accepts only a trace opened from an exact runtime-acknowledged clean action.
    An absent/unbound trace is candidate-only UNKNOWN, never a local certified
    success.  Its frames remain private.
    """

    if not isinstance(trace, TrustedGraspEvidenceTrace):
        return _unknown_local_evidence(
            binding=binding,
            config=config,
            reason="UNTRUSTED_OR_UNBOUND_EXECUTION_TRACE",
            trace_kind=type(trace).__name__,
        )
    try:
        entries = trace._trusted_entries(
            authority=trace._authority,
            binding_identity=ExistingPoseServoEngine._binding_identity_for(binding),
        )
    except RecoveryContractError:
        return _unknown_local_evidence(
            binding=binding,
            config=config,
            reason="UNTRUSTED_OR_WRONG_BINDING_EXECUTION_TRACE",
            trace_kind=type(trace).__name__,
        )
    if not entries:
        return _unknown_local_evidence(
            binding=binding,
            config=config,
            reason="NO_TRUSTED_PRIVATE_PHYSICS_EVIDENCE",
            trace_kind=type(trace).__name__,
        )
    if not any(action is not None and action.token == binding.hand.upper() + "_CLOSE" for _, action in entries):
        return _unknown_local_evidence(
            binding=binding,
            config=config,
            reason="NO_RUNTIME_ACKNOWLEDGED_GRASP_CLOSE",
            trace_kind=type(trace).__name__,
        )
    ticks: list[int] = []
    times: list[float] = []
    copied: list[dict[str, object]] = []
    issued_tokens: list[str | None] = []
    for raw, action in entries:
        frame = dict(raw)
        if frame.get("target_uid") != binding.target_name:
            return _unknown_local_evidence(
                binding=binding,
                config=config,
                reason="PHYSICS_EVIDENCE_TARGET_MISMATCH",
                trace_kind=type(trace).__name__,
            )
        tick = frame.get("tick")
        timestamp = frame.get("physics_time_seconds")
        if frame.get("physics_clock_receipt_sha256") != config.physics_clock_receipt_sha256:
            return _unknown_local_evidence(
                binding=binding,
                config=config,
                reason="PHYSICS_CLOCK_RECEIPT_MISMATCH_OR_MISSING",
                trace_kind=type(trace).__name__,
            )
        if type(tick) is not int or tick < 0 or (ticks and tick != ticks[-1] + 1):
            return _unknown_local_evidence(
                binding=binding,
                config=config,
                reason="MISSING_OR_NONCONTIGUOUS_PHYSICS_TICKS",
                trace_kind=type(trace).__name__,
            )
        try:
            now = _finite(timestamp, "physics_time_seconds")
        except RecoveryContractError:
            return _unknown_local_evidence(
                binding=binding,
                config=config,
                reason="MISSING_OR_INVALID_PHYSICS_TIMESTAMP",
                trace_kind=type(trace).__name__,
            )
        if times and now <= times[-1]:
            return _unknown_local_evidence(
                binding=binding,
                config=config,
                reason="STALE_OR_DUPLICATE_PHYSICS_TIMESTAMP",
                trace_kind=type(trace).__name__,
            )
        if times and not math.isclose(now - times[-1], config.physics_dt_seconds, rel_tol=0.0, abs_tol=1e-9):
            return _unknown_local_evidence(
                binding=binding,
                config=config,
                reason="PHYSICS_CADENCE_DIFFERS_FROM_PINNED_CONFIG",
                trace_kind=type(trace).__name__,
            )
        if not _local_grasp_frame_complete(frame):
            return _unknown_local_evidence(
                binding=binding,
                config=config,
                reason="INCOMPLETE_LOCAL_GRASP_SENSOR_EVIDENCE",
                trace_kind=type(trace).__name__,
            )
        ticks.append(tick)
        times.append(now)
        # LocalOutcome does not know this additional receipt field.
        frame.pop("physics_time_seconds", None)
        frame.pop("physics_clock_receipt_sha256", None)
        frame.pop("runtime_step_receipt_sha256", None)
        copied.append(frame)
        issued_tokens.append(None if action is None else action.token)
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
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
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


def _local_grasp_frame_complete(frame: Mapping[str, object]) -> bool:
    """Preflight native LocalOutcome's required private fields to yield UNKNOWN."""

    try:
        if not isinstance(frame.get("target_pose"), Sequence) or not isinstance(frame.get("hand_poses"), Mapping):
            return False
        if set(frame["hand_poses"]) != {"left", "right"}:
            return False
        _mapping_bool(frame.get("held", {}), "held", allow_unknown=True)
        fingers = _mapping_bool(frame.get("finger_contact", {}), "finger_contact", allow_unknown=True)
        if any(value is None for value in fingers.values()):
            return False
        if frame.get("contacts_known") is not True or type(frame.get("payload_ok")) is not bool:
            return False
        if not isinstance(frame.get("forbidden_contacts"), Sequence) or isinstance(frame.get("forbidden_contacts"), (str, bytes)):
            return False
    except (KeyError, TypeError, RecoveryContractError):
        return False
    return True


def _unknown_local_evidence(
    *, binding: SealedGraspBinding, config: GraspPostconditionConfig, reason: str, trace_kind: str
) -> LocalGraspEvidence:
    """Return an explicit candidate-only UNKNOWN without invoking LocalOutcome."""

    evidence = canonical_sha256(
        {
            "schema": "p107_local_grasp_evidence_v3",
            "target_name_sha256": sha256(binding.target_name.encode()).hexdigest(),
            "hand": binding.hand,
            "status": "UNKNOWN",
            "reason": reason,
            "trace_kind": trace_kind,
            "postcondition_sha256": config.sha256,
            "official_task_success": False,
        }
    )
    return LocalGraspEvidence(
        status="UNKNOWN", reason=reason, evidence_sha256=evidence, postcondition_sha256=config.sha256
    )
