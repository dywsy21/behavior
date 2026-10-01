"""CPU contracts for the candidate-only privileged-pose GRASP adapter."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import sys

try:
    import pytest
except ImportError:  # The minimal CPU image has no pytest.
    class _Raises:
        def __init__(self, error, match):
            self.error, self.match = error, match

        def __enter__(self):
            return None

        def __exit__(self, error_type, error, traceback):
            if error_type is None:
                raise AssertionError(f"expected {self.error.__name__}")
            if not issubclass(error_type, self.error) or not re.search(self.match, str(error)):
                return False
            return True

    class _PytestFallback:
        @staticmethod
        def raises(error, *, match=""):
            return _Raises(error, match)

    pytest = _PytestFallback()

from g05.recovery.common import RecoveryContractError
from g05.recovery.dart_collection import (
    AppliedActionReceipt,
    CanonicalSourceGroupMembership,
    DartObservation,
    VerifiedDartSourceMembership,
)
from g05.recovery.privileged_pose_grasp import (
    ApprovedPublicText,
    ActorContext,
    ExistingPrivilegedReaderWorld,
    ExistingPoseServoEngine,
    FreshRolloutRaw23Runtime,
    GenericGraspPilot,
    GraspPostconditionConfig,
    OpaqueMediaReference,
    PrivateGraspSnapshot,
    PrivilegedPoseGraspTeacher,
    RootModelPoseReview,
    SealedGraspBinding,
    certify_local_grasp,
    encode_r1pro_raw23,
    project_actor_observation,
    raw23_wire_sha256,
)


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def observation(clock: int) -> DartObservation:
    return DartObservation(
        policy_clock=clock,
        state61=[float(clock)] * 61,
        observation_sha256=digest(f"rgb:{clock}"),
        observation_ref=f"fresh-rgb-proprio-{clock}",
    )


def binding(**overrides):
    data = dict(
        source_group_id=digest("train-group-1"),
        source_release_sha256=digest("release"),
        source_group_member_sha256=digest("member"),
        source_group_index_manifest_sha256=digest("index"),
        source_role="student_candidate",
        original_split="train",
        intent_bundle_id="grasp-intent-1",
        intent_sha256=digest("intent"),
        target_name="toy_train_target_1",
        target_category="toy",
        hand="right",
        goal_pose_local_sha256=digest("pose"),
        public_task_description_sha256=digest("grasp object"),
        public_causal_intent_sha256=digest("causal grasp"),
        public_description_registry_sha256=digest("public-description-registry"),
        pose_review=RootModelPoseReview("root-model-review-1", digest("review")),
    )
    data.update(overrides)
    return SealedGraspBinding(**data)


def actor_context(obs: DartObservation, *, task: str = "grasp object", causal_intent: str = "causal grasp") -> ActorContext:
    registry = digest("public-description-registry")
    return ActorContext(
        task=ApprovedPublicText(task, digest(task), registry, "task"),
        causal_intent=ApprovedPublicText(causal_intent, digest(causal_intent), registry, "causal_intent"),
        media=OpaqueMediaReference(obs.observation_sha256),
    )


def private_snapshot(obs: DartObservation, *, marker: float = 0.0, target: str = "toy_train_target_1") -> PrivateGraspSnapshot:
    return PrivateGraspSnapshot.from_observation(
        obs,
        target_name=target,
        teacher_frame={
            "target_uid": target,
            "held": {"left": False, "right": False},
            "finger_contact": {"left": False, "right": False},
            "contacts_known": True,
            "payload_ok": True,
            "forbidden_contacts": [],
        },
        robot_state={"marker": marker},
        servo_model=object(),
        goal_world=object(),
        base_world=object(),
        grips=(0.0, 0.0),
        fresh_query_receipt_sha256=digest(f"private:{obs.policy_clock}:{marker}"),
    )


class World:
    def __init__(self, *, stale: bool = False):
        self.stale = stale
        self.calls = 0

    def fresh_snapshot(self, obs, source_binding):
        self.calls += 1
        return private_snapshot(observation(obs.policy_clock - 1) if self.stale else obs, marker=float(obs.policy_clock))


class Engine:
    def __init__(self):
        self.seen = []
        self.applied = 0

    def ranked_tokens(self, snap, source_binding):
        self.seen.append(snap.robot_state["marker"])
        return ("RIGHT_FORWARD",)

    def raw23_for_token(self, snap, token):
        # A current-state-dependent one-step command, not a demo action.
        return [snap.robot_state["marker"]] * 23

    def acknowledge_actual_clean_application(self, applied):
        self.applied += 1

    def discard_unapplied_proposal(self, applied):
        self.discarded = getattr(self, "discarded", 0) + 1


def test_r1pro_mapping_rejects_padding_and_nonfinite():
    raw = encode_r1pro_raw23(base_qvel=(1, 2, 3), joint_q18=tuple(range(10, 28)), gripper=(90, 91))
    assert raw == (1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 90.0,
                   21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 91.0)
    with pytest.raises(RecoveryContractError, match="base3"):
        encode_r1pro_raw23(base_qvel=(0,) * 4, joint_q18=(0,) * 18, gripper=(0, 0))
    with pytest.raises(RecoveryContractError):
        encode_r1pro_raw23(base_qvel=(0, 0, 0), joint_q18=(0,) * 17 + (float("nan"),), gripper=(0, 0))


def test_teacher_requeries_current_harmless_motion_noise_without_side_effects():
    engine, world = Engine(), World()
    teacher = PrivilegedPoseGraspTeacher(binding=binding(), world=world, engine=engine)
    runtime = FreshRolloutRaw23Runtime(RuntimeHooks())
    first_observation = runtime.observe()
    first = teacher.clean_action(first_observation, intent_bundle_id="grasp-intent-1")
    # This base-only perturbation preserves both gripper raw23 channels.  It
    # remains a noisy execution, not a clean label or native lifecycle event.
    noisy = list(first.clean_intended23)
    noisy[0] += 0.01
    teacher.acknowledge_runtime_noisy_application(runtime, runtime.apply_raw23(noisy))
    second_observation = runtime.observe()
    second = teacher.clean_action(second_observation, intent_bundle_id="grasp-intent-1")
    assert first.clean_intended23 == (1.0,) * 23
    assert second.clean_intended23 == (2.0,) * 23
    assert engine.seen == [1.0, 2.0]
    # clean_action never has a runtime/apply handle: it only queried a fresh state.
    assert engine.applied == 0 and engine.discarded == 1 and world.calls == 2


def test_existing_pose_teacher_and_safe_servo_are_preview_only_until_runtime_apply():
    calls = []

    class FakePoseTeacher:
        def __init__(self, spec):
            self.close_issued = False
            self.spec = spec

        def ranked(self, state, frame, goal, base, grips, model):
            calls.append(("ranked", state["marker"], frame["target_uid"], self.close_issued))
            return ("RIGHT_FORWARD",)

    class FakeServo:
        def __init__(self, model, state, *, gripper_command, limits):
            self.done = False
            calls.append(("servo-init", state["marker"], tuple(gripper_command), limits))

        def begin(self, action, state, *, carry):
            calls.append(("begin", action, state["marker"], carry))
            return True

        def next_action(self, state):
            calls.append(("preview-next", state["marker"]))
            return [0.125] * 23

    engine = ExistingPoseServoEngine(
        pose_teacher_class=FakePoseTeacher,
        safe_servo_class=FakeServo,
        token_to_action=lambda token: "decoded:" + token,
    )
    teacher = PrivilegedPoseGraspTeacher(binding=binding(), world=World(), engine=engine)
    command = teacher.clean_action(observation(6), intent_bundle_id="grasp-intent-1")
    assert command.clean_intended23 == (0.125,) * 23
    assert [row[0] for row in calls] == ["ranked", "servo-init", "begin", "preview-next"]
    # No hook / env.step exists in this teacher path.  Only FreshRolloutRaw23Runtime applies.


def test_existing_privileged_reader_world_reads_current_state_only_when_queried():
    calls = []

    class Reader:
        def read(self, tick):
            calls.append(("read", tick))
            return {
                "target_uid": "toy_train_target_1",
                "held": {"left": False, "right": False},
                "finger_contact": {"left": False, "right": False},
                "contacts_known": True,
                "payload_ok": True,
                "forbidden_contacts": [],
            }

        def goal(self):
            calls.append(("goal",))
            return object()

        def base(self):
            calls.append(("base",))
            return object()

    class Kinematics:
        def state(self):
            calls.append(("state",))
            return type("State", (), {"gripper": (0.0, 0.0)})()

    world = ExistingPrivilegedReaderWorld(
        privileged_reader=Reader(),
        kinematics=Kinematics(),
        servo_model=object(),
        fresh_query_receipt=lambda obs, frame, state: digest("current-private-query"),
    )
    assert calls == []
    snap = world.fresh_snapshot(observation(7), binding())
    assert snap.policy_clock == 7
    assert [row[0] for row in calls] == ["read", "state", "goal", "base"]


def test_teacher_rejects_stale_private_state_unknown_or_missing_candidate():
    teacher = PrivilegedPoseGraspTeacher(binding=binding(), world=World(stale=True), engine=Engine())
    with pytest.raises(RecoveryContractError, match="stale"):
        teacher.clean_action(observation(3), intent_bundle_id="grasp-intent-1")

    class EmptyEngine(Engine):
        def ranked_tokens(self, snap, source_binding):
            return ()

    teacher = PrivilegedPoseGraspTeacher(binding=binding(), world=World(), engine=EmptyEngine())
    with pytest.raises(RecoveryContractError, match="no eligible"):
        teacher.clean_action(observation(4), intent_bundle_id="grasp-intent-1")


def test_train_role_root_model_provenance_and_private_projection_are_strict():
    with pytest.raises(RecoveryContractError, match="TRAIN"):
        binding(source_role="evaluation_only", original_split="eval")
    with pytest.raises(RecoveryContractError, match="human"):
        RootModelPoseReview("wrong", digest("x"), human_reviewed=True)
    with pytest.raises(RecoveryContractError, match="sealed public"):
        ActorContext(task="grasp", causal_intent="target_pose is visible", media=OpaqueMediaReference(digest("media")))

    obs = observation(5)
    projected = project_actor_observation(obs, actor_context(obs), binding())
    text = repr(projected)
    assert "target_uid" not in text and "hand_poses" not in text and "toy_train_target_1" not in text
    assert "observation_ref" not in text
    assert projected["observation"]["actor_visible"] is True
    # A sealed hash alone is insufficient: the public text itself may never
    # include teacher-only target identity or a free-form media reference.
    leaked_intent = "grasp toy_train_target_1"
    leaked_binding = binding(public_causal_intent_sha256=digest(leaked_intent))
    with pytest.raises(RecoveryContractError, match="leaks"):
        project_actor_observation(obs, actor_context(obs, causal_intent=leaked_intent), leaked_binding)
    with pytest.raises(RecoveryContractError, match="opaque media"):
        project_actor_observation(
            obs,
            ActorContext(
                task=actor_context(obs).task,
                causal_intent=actor_context(obs).causal_intent,
                media=OpaqueMediaReference(digest("another-observation")),
            ),
            binding(),
        )

    source_binding = binding()
    membership = CanonicalSourceGroupMembership(
        source_group_id=source_binding.source_group_id,
        source_release_sha256=source_binding.source_release_sha256,
        task_index=1,
        task_instance_id=2,
        original_split="train",
        usage_role="student_candidate",
        source_group_member_sha256=source_binding.source_group_member_sha256,
    )
    source_binding.assert_canonical_membership(membership)
    verified = VerifiedDartSourceMembership(
        index_inventory_seal_sha256=digest("inventory"),
        source_group_index_manifest_sha256=source_binding.source_group_index_manifest_sha256,
        source_membership_protocol_sha256=digest("protocol"),
        candidate=membership,
        calibration_groups=(
            CanonicalSourceGroupMembership(
                source_group_id=digest("calibration"),
                source_release_sha256=source_binding.source_release_sha256,
                task_index=2,
                task_instance_id=3,
                original_split="train",
                usage_role="annotation_calibration",
                source_group_member_sha256=digest("calibration-member"),
            ),
        ),
    )
    source_binding.assert_verified_dart_membership(verified)
    with pytest.raises(RecoveryContractError, match="differs"):
        source_binding.assert_canonical_membership(
            CanonicalSourceGroupMembership(
                source_group_id=source_binding.source_group_id,
                source_release_sha256=source_binding.source_release_sha256,
                task_index=1,
                task_instance_id=2,
                original_split="eval",
                usage_role="evaluation_only",
                source_group_member_sha256=source_binding.source_group_member_sha256,
            )
        )

    other = binding(
        source_group_id=digest("train-group-2"),
        source_group_member_sha256=digest("member-2"),
        target_name="can_train_target_2",
        target_category="can",
    )
    pilot = GenericGraspPilot(
        bindings=(source_binding, other),
        controller_code_sha256=digest("shared-pose-controller"),
        controller_config_sha256=digest("shared-pose-controller-config"),
    )
    assert len(pilot.public()["bindings"]) == 2
    with pytest.raises(RecoveryContractError, match="categories"):
        GenericGraspPilot(
            bindings=(source_binding, binding(source_group_id=digest("third"), source_group_member_sha256=digest("third-member"))),
            controller_code_sha256=digest("shared-pose-controller"),
            controller_config_sha256=digest("shared-pose-controller-config"),
        )


class RuntimeHooks:
    def __init__(self, *, stale: bool = False, substitute: bool = False):
        self.clock = 0
        self.stale = stale
        self.substitute = substitute
        self.applies = 0

    def observe_current(self):
        self.clock += 1
        return observation(self.clock)

    def apply_current_raw23(self, requested, obs):
        self.applies += 1
        applied = tuple([0.0] * 23) if self.substitute else tuple(requested)
        bound = observation(obs.policy_clock - 1) if self.stale else obs
        return AppliedActionReceipt(
            status="APPLIED",
            applied23=applied,
            applied_action_bytes_sha256=raw23_wire_sha256(requested),
            runtime_step_receipt_sha256=digest(f"step:{self.applies}"),
            pre_action_policy_clock=bound.policy_clock,
            pre_action_state61_sha256=bound.state61_sha256,
            pre_action_observation_sha256=bound.observation_sha256,
        )


def test_fresh_runtime_requires_one_authorized_apply_and_exact_bytes():
    hooks = RuntimeHooks()
    runtime = FreshRolloutRaw23Runtime(hooks)
    with pytest.raises(RecoveryContractError, match="preceding"):
        runtime.apply_raw23([0.0] * 23)
    current = runtime.observe()
    receipt = runtime.apply_raw23([0.25] * 23)
    assert receipt.pre_action_policy_clock == current.policy_clock and hooks.applies == 1
    with pytest.raises(RecoveryContractError, match="preceding"):
        runtime.apply_raw23([0.25] * 23)

    stale = FreshRolloutRaw23Runtime(RuntimeHooks(stale=True))
    stale.observe()
    with pytest.raises(RecoveryContractError, match="stale"):
        stale.apply_raw23([0.0] * 23)
    substituted = FreshRolloutRaw23Runtime(RuntimeHooks(substitute=True))
    substituted.observe()
    with pytest.raises(RecoveryContractError, match="substitution"):
        substituted.apply_raw23([0.25] * 23)


def pose(z):
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def evidence_frame(tick: int, *, held: bool, contact: bool, z: float, hand_z: float):
    return {
        "tick": tick,
        "physics_time_seconds": tick * 0.05,
        "target_uid": "toy_train_target_1",
        "target_pose": pose(z),
        "hand_poses": {"left": pose(0.0), "right": pose(hand_z)},
        "held": {"left": False, "right": held},
        "finger_contact": {"left": False, "right": contact},
        "contacts_known": True,
        "payload_ok": True,
        "forbidden_contacts": [],
    }


def close_teacher_and_runtime():
    """CPU fake with the native CLOSE lifecycle shape, no simulator execution."""

    class LifecyclePoseTeacher:
        instances = []

        def __init__(self, spec):
            self.close_issued = False
            self.executed_tokens = []
            type(self).instances.append(self)

        def ranked(self, state, frame, goal, base, grips, model):
            # The wrapper, rather than this convenient fake, must prevent a
            # second CLOSE after an executed lifecycle acknowledgement.
            return ("RIGHT_CLOSE",)

        def executed(self, token, frame):
            self.executed_tokens.append(token)
            if token == "RIGHT_CLOSE":
                self.close_issued = True

    class CloseServo:
        def __init__(self, model, state, *, gripper_command, limits):
            self.done = False

        def begin(self, action, state, *, carry):
            return action == "decoded:RIGHT_CLOSE" and carry is True

        def next_action(self, state):
            return [0.125] * 23

    engine = ExistingPoseServoEngine(
        pose_teacher_class=LifecyclePoseTeacher,
        safe_servo_class=CloseServo,
        token_to_action=lambda token: "decoded:" + token,
    )
    teacher = PrivilegedPoseGraspTeacher(binding=binding(), world=World(), engine=engine)
    runtime = FreshRolloutRaw23Runtime(RuntimeHooks())
    return teacher, runtime, LifecyclePoseTeacher


def test_native_close_lifecycle_advances_only_after_exact_runtime_acknowledgement():
    teacher, runtime, native = close_teacher_and_runtime()
    first_obs = runtime.observe()
    command = teacher.clean_action(first_obs, intent_bundle_id="grasp-intent-1")
    assert native.instances[0].executed_tokens == []

    forged = AppliedActionReceipt(
        status="APPLIED",
        applied23=[0.0] * 23,
        applied_action_bytes_sha256=raw23_wire_sha256([0.0] * 23),
        runtime_step_receipt_sha256=digest("forged-step"),
        pre_action_policy_clock=first_obs.policy_clock,
        pre_action_state61_sha256=first_obs.state61_sha256,
        pre_action_observation_sha256=first_obs.observation_sha256,
    )
    with pytest.raises(RecoveryContractError, match="exact clean"):
        teacher.acknowledge_runtime_clean_application(runtime, forged)
    assert native.instances[0].executed_tokens == []

    applied = runtime.apply_raw23(command.clean_intended23)
    trusted_close = teacher.acknowledge_runtime_clean_application(runtime, applied)
    assert trusted_close.token == "RIGHT_CLOSE"
    assert native.instances[0].executed_tokens == ["RIGHT_CLOSE"]

    # A fresh scene reporting NOT_HELD cannot cause a blind second CLOSE after
    # the one actual close acknowledgement, even if native ranked() offers it.
    with pytest.raises(RecoveryContractError, match="CLOSE after an acknowledged"):
        teacher.clean_action(runtime.observe(), intent_bundle_id="grasp-intent-1")


def test_unapplied_close_proposal_does_not_advance_native_lifecycle():
    teacher, runtime, native = close_teacher_and_runtime()
    teacher.clean_action(runtime.observe(), intent_bundle_id="grasp-intent-1")
    with pytest.raises(RecoveryContractError, match="previous clean"):
        teacher.clean_action(observation(99), intent_bundle_id="grasp-intent-1")
    assert native.instances[0].executed_tokens == []


def test_base_noisy_close_acknowledges_only_native_lifecycle_not_clean_authority():
    teacher, runtime, native = close_teacher_and_runtime()
    command = teacher.clean_action(runtime.observe(), intent_bundle_id="grasp-intent-1")
    noisy = list(command.clean_intended23)
    noisy[0] += 0.01

    # The runtime applied a different raw23 vector, but both gripper float32
    # channels remained byte-identical to the CLOSE preview.  This may advance
    # only the private native lifecycle; no TrustedGraspAction is minted.
    assert teacher.acknowledge_runtime_noisy_application(runtime, runtime.apply_raw23(noisy)) is None
    assert native.instances[0].executed_tokens == ["RIGHT_CLOSE"]
    with pytest.raises(RecoveryContractError, match="CLOSE after an acknowledged"):
        teacher.clean_action(runtime.observe(), intent_bundle_id="grasp-intent-1")


def test_noisy_gripper_change_retires_episode_without_lifecycle_acknowledgement():
    teacher, runtime, native = close_teacher_and_runtime()
    command = teacher.clean_action(runtime.observe(), intent_bundle_id="grasp-intent-1")
    noisy = list(command.clean_intended23)
    noisy[22] += 0.01

    with pytest.raises(RecoveryContractError, match="retired"):
        teacher.acknowledge_runtime_noisy_application(runtime, runtime.apply_raw23(noisy))
    assert native.instances[0].executed_tokens == []
    with pytest.raises(RecoveryContractError, match="retired"):
        teacher.clean_action(runtime.observe(), intent_bundle_id="grasp-intent-1")


def local_outcome():
    root = Path(__file__).resolve().parents[1]
    scripts = str(root / "scripts" / "vlm_sft")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from native_teacher_outcomes import LocalOutcome
    return LocalOutcome


def test_local_grasp_certification_requires_bound_runtime_close_and_complete_contact_evidence():
    config = GraspPostconditionConfig(
        physics_dt_seconds=0.05, physics_clock_receipt_sha256=digest("runtime-physics-clock-0.05")
    )
    teacher, runtime, _native = close_teacher_and_runtime()
    command = teacher.clean_action(runtime.observe(), intent_bundle_id="grasp-intent-1")
    close = teacher.acknowledge_runtime_clean_application(runtime, runtime.apply_raw23(command.clean_intended23))
    trace = teacher.begin_local_grasp_evidence(close)
    frames = [evidence_frame(0, held=False, contact=False, z=0.0, hand_z=0.0)]
    # The CLOSE occurs while still at the original pose; qualified lift begins
    # only on the following real physics tick.
    frames.append(evidence_frame(1, held=False, contact=True, z=0.0, hand_z=0.0))
    frames[-1]["runtime_step_receipt_sha256"] = close.runtime_step_receipt_sha256
    for tick in range(2, 14):
        frames.append(evidence_frame(tick, held=True, contact=True, z=0.04, hand_z=0.03))
    for frame in frames:
        frame["physics_clock_receipt_sha256"] = config.physics_clock_receipt_sha256
    for index, frame in enumerate(frames):
        trace.record_private_physics_frame(frame, action=close if index == 1 else None)
    verdict = certify_local_grasp(binding=binding(), config=config, trace=trace, local_outcome_class=local_outcome())
    assert verdict.status == "SUCCEEDED" and verdict.official_task_success is False

    broken_trace = teacher.begin_local_grasp_evidence(close)
    broken = [dict(row) for row in frames]
    broken[6]["physics_time_seconds"] = broken[5]["physics_time_seconds"]
    for index, frame in enumerate(broken):
        broken_trace.record_private_physics_frame(frame, action=close if index == 1 else None)
    assert certify_local_grasp(
        binding=binding(), config=config, trace=broken_trace, local_outcome_class=local_outcome()
    ).status == "UNKNOWN"

    incomplete_trace = teacher.begin_local_grasp_evidence(close)
    unknown = [dict(row) for row in frames]
    unknown[-1]["finger_contact"] = None
    for index, frame in enumerate(unknown):
        incomplete_trace.record_private_physics_frame(frame, action=close if index == 1 else None)
    verdict = certify_local_grasp(binding=binding(), config=config, trace=incomplete_trace, local_outcome_class=local_outcome())
    assert verdict.status == "UNKNOWN"

    # Caller strings alone cannot manufacture a local success predicate.
    forged = certify_local_grasp(binding=binding(), config=config, trace={"frames": frames, "issued_tokens": ["RIGHT_CLOSE"]},
                                  local_outcome_class=local_outcome())
    assert forged.status == "UNKNOWN" and forged.reason == "UNTRUSTED_OR_UNBOUND_EXECUTION_TRACE"


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("privileged pose GRASP adapter tests passed")
