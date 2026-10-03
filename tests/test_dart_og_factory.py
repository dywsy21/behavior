"""Fixture-only contracts for the lazy live OmniGibson DART factory.

These tests do not import OmniGibson, Isaac, or simulator assets.  They prove
the local adapter's pin/source/clock/action contracts only; a separately
authorized pinned-runtime preflight is still required before collection.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

from g05.recovery.common import RecoveryContractError, canonical_sha256
from g05.recovery.dart_collection import (
    CalibrationReceipt,
    CandidateSourceReceipt,
    canonical_native_raw23_float32,
    CanonicalSourceGroupMembership,
    CollectionLimits,
    DartCollectionResult,
    OriginalGaussianCalibrationBinding,
    OriginalGaussianNoise,
    RuntimeSessionReceipt,
    TeacherCommand,
    TeacherReceipt,
    VerifiedDartSourceMembership,
)
from g05.recovery.dart_noise import ActionPart, BoundedNoiseProfile, BoundedPartRule, DartInspiredBoundedNoise, Raw23ActionLayout
from g05.recovery.dart_og_factory import (
    PrivateEpisodeWriterConfig,
    DartOgFactoryPins,
    DartOgGraspFactory,
    DartOgRuntimeHooks,
    PinnedSingleEnvNativeReaderView,
    TrainSourceSelection,
    build_private_episode_bundle_writer,
    collect_dart_live_candidate,
    load_private_episode_publication,
    run_live_grasp_collection,
    validate_live_r1pro_controller,
)
from g05.recovery.dart_dataset import load_bundle
from g05.recovery.privileged_pose_grasp import FreshRolloutRaw23Runtime, RootModelPoseReview, SealedGraspBinding


def _sha(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _fixture_execution_provenance(record_mode: str) -> dict[str, object]:
    """Minimal complete private controls for direct writer fixture tests."""

    if record_mode == "original_gaussian_clean_intended_feedback":
        covariance = [[0.25 if row == column else 0.0 for column in range(23)] for row in range(23)]
        calibration_binding = {
            "calibration_trajectory_sha256": _sha("calibration-trajectory"),
            "learner_checkpoint_sha256": _sha("learner"),
            "teacher_checkpoint_sha256": _sha("teacher"),
            "covariance_estimator_code_sha256": _sha("covariance-estimator"),
            "estimated_covariance_sha256": _sha("estimated-covariance"),
            "scaled_covariance_sha256": _sha("scaled-covariance"),
            "alpha": 5.75,
            "horizon": 1,
            "source_group_index_manifest_sha256": _sha("source-group-index"),
        }
        return {
            "schema": "p107-dart-live-execution-provenance-v1",
            "factory_mode": "original_gaussian_one_pass_partial_dart",
            "record_collection_mode": record_mode,
            "original_gaussian": {
                "algorithm_id": "dart_original_gaussian_unbounded",
                "sampling_seed": 31,
                "covariance23": covariance,
                "covariance23_sha256": canonical_sha256(covariance),
                "steps": 1,
                "covariance_update_mode": "frozen_one_pass_partial_dart",
                "calibration_binding": calibration_binding,
            },
        }
    return {
        "schema": "p107-dart-live-execution-provenance-v1",
        "factory_mode": "dart_inspired_bounded_actual_clean_recovery",
        "record_collection_mode": "dart_inspired_bounded_actual_clean_recovery",
        "bounded_dart": {
            "algorithm_id": "dart_inspired_bounded_correlated",
            "sampling_seed": 17,
            "profile": {
                "profile_id": "fixture",
                "temporal_rho": 0.0,
                "layout": {
                    "parts": [{"name": "raw23", "dimension": 23, "unit": "fixture"}],
                    "embodiment_metadata_sha256": _sha("fixture-embodiment"),
                    "model_projection_manifest_sha256": _sha("fixture-projection"),
                },
                "rules_by_part": {
                    "raw23": {"active": True, "standard_deviation": 0.1, "cap": 0.2, "inactive_reason": None}
                },
            },
            "noisy_injection_steps": 1,
            "recovery_chunk_lengths": [1],
            "limits": {"max_total_steps": 2, "max_wall_seconds": 30.0},
        },
    }


def _controller(name: str, **fields: object) -> object:
    cls = type(name, (), {})
    value = cls()
    for field, item in fields.items():
        setattr(value, field, item)
    return value


class _FakeRobot:
    model = "r1pro"
    action_dim = 23
    controller_action_idx = {
        "base": [0, 1, 2],
        "trunk": [3, 4, 5, 6],
        "arm_left": [7, 8, 9, 10, 11, 12, 13],
        "gripper_left": [14],
        "arm_right": [15, 16, 17, 18, 19, 20, 21],
        "gripper_right": [22],
    }

    def __init__(self) -> None:
        self.controllers = {
            "base": _controller("HolonomicBaseJointController"),
            "trunk": _controller("JointController"),
            "arm_left": _controller("JointController"),
            "gripper_left": _controller("MultiFingerGripperController", _mode="smooth", _inverted=False),
            "arm_right": _controller("JointController"),
            "gripper_right": _controller("MultiFingerGripperController", _mode="smooth", _inverted=True),
        }


class _FakeTorch:
    float32 = np.float32

    @staticmethod
    def tensor(value: object, *, dtype: object) -> np.ndarray:
        return np.asarray(value, dtype=dtype)


class _FakeSim:
    current_time_step_index = 17
    current_time = 0.5


class _FakeOg:
    sim = _FakeSim()


class _FakeEvaluator:
    num_envs = 1
    robot_camera_names = {
        "left_wrist": "r1pro::left_wrist",
        "right_wrist": "r1pro::right_wrist",
        "head": "r1pro::head",
    }

    def __init__(self, *, terminal: bool = False, static_rgb: bool = False) -> None:
        self.static_rgb = static_rgb
        self.robot = _FakeRobot()
        self.scene = SimpleNamespace(idx=0, robots=[self.robot])
        self.env = SimpleNamespace(
            scenes=[self.scene],
            task=SimpleNamespace(object_scopes=[{"bound_target": object()}], scene_name="scene_from_source_index"),
        )
        self.state = SimpleNamespace(
            env_accessor=SimpleNamespace(robot=self.robot, scene=self.scene),
            obs=self._obs(0),
        )
        self.instance_eval_states = [self.state]
        self.terminal = terminal
        self.closed = False
        self.applied: list[np.ndarray] = []
        self.loaded_batches: list[dict[int, int]] = []

    def _obs(self, clock: int) -> dict[str, np.ndarray]:
        image_clock = 0 if self.static_rgb else clock
        return {
            "r1pro::proprio": np.full(61, float(clock), dtype=np.float32),
            "r1pro::left_wrist::rgb": np.full((2, 3, 3), image_clock + 10, dtype=np.uint8),
            "r1pro::right_wrist::rgb": np.full((2, 3, 3), image_clock + 20, dtype=np.uint8),
            "r1pro::head::rgb": np.full((2, 3, 3), image_clock + 30, dtype=np.uint8),
        }

    def _apply_actions(self, action: np.ndarray, env_indices: list[int]):
        assert action.shape == (1, 23)
        assert env_indices == [0]
        self.applied.append(action.copy())
        _FakeOg.sim.current_time_step_index += 1
        _FakeOg.sim.current_time += 1.0 / 30.0
        self.state.obs = self._obs(len(self.applied))
        return np.asarray([self.terminal]), np.asarray([False]), [{}]

    def close(self) -> None:
        self.closed = True

    def load_batch(self, batch: dict[int, int]) -> None:
        self.loaded_batches.append(dict(batch))


def _source() -> CandidateSourceReceipt:
    return CandidateSourceReceipt(
        source_group_id=_sha("source-group"),
        original_split="train",
        source_release_sha256=_sha("release"),
        parent_task_id="sealed-task-id",
        parent_task_index=7,
        parent_task_instance_id=19,
        parent_task_seed=123,
        trajectory_source_kind="fresh_dart_trajectory",
        collection_run_id="run-1",
    )


def _selection(source: CandidateSourceReceipt) -> TrainSourceSelection:
    return TrainSourceSelection(
        source_group_id=source.source_group_id,
        source_release_sha256=source.source_release_sha256,
        parent_task_id=source.parent_task_id,
        task_name="task_name_from_source_index",
        scene_name="scene_from_source_index",
        task_instance_id=source.parent_task_instance_id,
        task_seed=source.parent_task_seed,
        source_selection_sha256=_sha("source-index-row"),
    )


def _binding(source: CandidateSourceReceipt) -> SealedGraspBinding:
    return SealedGraspBinding(
        source_group_id=source.source_group_id,
        source_release_sha256=source.source_release_sha256,
        source_group_member_sha256=_sha("source-member"),
        source_group_index_manifest_sha256=_sha("source-index"),
        source_role="student_candidate",
        original_split="train",
        intent_bundle_id="sealed-grasp-intent",
        intent_sha256=_sha("intent"),
        target_name="target_from_sealed_teacher_spec",
        target_category="toy",
        hand="right",
        goal_pose_local_sha256=_sha("pose"),
        public_task_description_sha256=_sha("task-description"),
        public_causal_intent_sha256=_sha("causal-intent"),
        public_description_registry_sha256=_sha("public-registry"),
        pose_review=RootModelPoseReview("root-model-review", _sha("review")),
    )


def _verified_membership(source: CandidateSourceReceipt) -> VerifiedDartSourceMembership:
    candidate = CanonicalSourceGroupMembership(
        source_group_id=source.source_group_id,
        source_release_sha256=source.source_release_sha256,
        task_index=source.parent_task_index,
        task_instance_id=source.parent_task_instance_id,
        original_split="train",
        usage_role="student_candidate",
        source_group_member_sha256=_sha("source-member"),
    )
    calibration = CanonicalSourceGroupMembership(
        source_group_id=_sha("calibration-group"),
        source_release_sha256=source.source_release_sha256,
        task_index=source.parent_task_index,
        task_instance_id=source.parent_task_instance_id + 1,
        original_split="train",
        usage_role="annotation_calibration",
        source_group_member_sha256=_sha("calibration-member"),
    )
    return VerifiedDartSourceMembership(
        index_inventory_seal_sha256=_sha("inventory"),
        source_group_index_manifest_sha256=_sha("group-index"),
        source_membership_protocol_sha256=_sha("source-protocol"),
        candidate=candidate,
        calibration_groups=(calibration,),
    )


def _runtime_session(source: CandidateSourceReceipt) -> RuntimeSessionReceipt:
    return RuntimeSessionReceipt(
        runtime_session_id=source.collection_run_id,
        task_instance_id=source.parent_task_instance_id,
        runtime_build_sha256=_sha("runtime"),
        asset_config_sha256=_sha("assets"),
        reset_load_task_instance_receipt_sha256=_sha("fresh-reset"),
    )


def _unqualified_teacher() -> TeacherReceipt:
    return TeacherReceipt(
        teacher_id="privileged-pose-grasp-live-candidate",
        teacher_kind="privileged_pose_grasp_unqualified",
        label_source="privileged_oracle",
        feedback_mode="closed_loop_fresh_observation",
        postcondition_spec_sha256=_sha("local-grasp-postcondition"),
        teacher_code_sha256=_sha("teacher-code"),
        teacher_weights_sha256=_sha("teacher-weights"),
        teacher_config_sha256=_sha("teacher-config"),
        runtime_code_sha256=_sha("runtime-code"),
    )


def _full_teacher_spec(binding: SealedGraspBinding) -> dict[str, object]:
    return {
        "schema": "h09t-private-teacher-v1",
        "verb": "GRASP",
        "hand": binding.hand,
        "support_hand": None,
        "target": binding.target_name,
        "destination": "",
        "payloads": [],
        "goal_pose_local": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "goal_frame": "target",
        "pose_reviewer": "root-model-review",
        "pose_evidence_sha256": _sha("pose-evidence"),
    }


def _install_fake_pinned_runtime(monkeypatch: pytest.MonkeyPatch, *, cfg: object, evaluator: _FakeEvaluator) -> dict[str, object]:
    """Inject narrow fake modules to exercise lazy factory wiring only."""

    captured: dict[str, object] = {}
    omegaconf = ModuleType("omegaconf")
    omegaconf.OmegaConf = SimpleNamespace(load=lambda _path: cfg)
    torch = ModuleType("torch")
    torch.tensor, torch.float32 = _FakeTorch.tensor, _FakeTorch.float32
    og = ModuleType("omnigibson")
    og.sim = _FakeOg.sim
    og_eval = ModuleType("omnigibson.eval")
    og_evaluator = ModuleType("omnigibson.eval.evaluator")
    og_evaluator.BatchedEvaluator = lambda actual_cfg: (captured.update(cfg=actual_cfg) or evaluator)
    reader = ModuleType("native_teacher_og")

    class FakeReader:
        def __init__(self, env: object, spec: object) -> None:
            captured["reader_env"], captured["reader_spec"] = env, spec

        def read(self, _clock: int) -> dict[str, object]:
            return {"held": {"left": False, "right": False}, "finger_contact": {"left": False, "right": False}}

        def goal(self) -> object:
            return object()

        def base(self) -> object:
            return object()

    reader.PrivilegedReader = FakeReader
    policy = ModuleType("native_teacher_policy")

    class FakePoseTeacher:
        def __init__(self, spec: object) -> None:
            self.spec = spec

    def validate_spec(spec: object, reference: object) -> None:
        required = {
            "schema", "verb", "hand", "support_hand", "target", "destination", "payloads", "goal_pose_local",
            "goal_frame", "pose_reviewer", "pose_evidence_sha256",
        }
        if not isinstance(spec, dict) or set(spec) != required or not isinstance(reference, dict):
            raise ValueError("complete test teacher spec required")
        captured["validated_spec"], captured["reference"] = spec, reference

    policy.PoseTeacher, policy.validate_spec = FakePoseTeacher, validate_spec
    calibration = ModuleType("semantic_robot.v2.og_calibration")

    class FakeCalibratedRobot:
        def __init__(self, robot: object) -> None:
            captured["kin_robot"] = robot

        def calibrate(self, *, grounded: bool) -> object:
            assert grounded is True
            return object()

        def state(self) -> object:
            return SimpleNamespace(gripper=(0.0, 0.0))

    calibration.CalibratedRobot = FakeCalibratedRobot
    servo = ModuleType("semantic_robot.v2.servo")
    servo.SafeServo = type("SafeServo", (), {})
    token = ModuleType("common")
    token.token_to_action = lambda _token: object()
    for name, module in {
        "omegaconf": omegaconf,
        "torch": torch,
        "omnigibson": og,
        "omnigibson.eval": og_eval,
        "omnigibson.eval.evaluator": og_evaluator,
        "native_teacher_og": reader,
        "native_teacher_policy": policy,
        "semantic_robot.v2.og_calibration": calibration,
        "semantic_robot.v2.servo": servo,
        "common": token,
    }.items():
        monkeypatch.setitem(__import__("sys").modules, name, module)
    return captured


class _Config(SimpleNamespace):
    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)


def test_r1pro_controller_receipt_records_but_does_not_interpret_gripper_direction() -> None:
    receipt = validate_live_r1pro_controller(_FakeRobot())
    assert receipt["action_dim"] == 23
    assert receipt["controllers"]["gripper_left"] == {"class": "MultiFingerGripperController", "mode": "smooth", "inverted": False}
    assert receipt["controllers"]["gripper_right"]["inverted"] is True


def test_cli_factory_request_rejects_missing_sealed_fields_without_importing_runtime() -> None:
    with pytest.raises(RecoveryContractError, match="exact p107-dart-live-grasp-request-v1 fields"):
        collect_dart_live_candidate({"schema": "p107-dart-live-grasp-request-v1"})


def test_r1pro_rejects_wrong_raw23_controller_layout_before_step() -> None:
    robot = _FakeRobot()
    robot.controller_action_idx = {**robot.controller_action_idx, "gripper_right": [21]}
    with pytest.raises(RecoveryContractError, match="layout"):
        validate_live_r1pro_controller(robot)


def test_runtime_observes_pre_step_applies_exact_raw23_and_captures_fresh_post_step() -> None:
    _FakeOg.sim.current_time_step_index, _FakeOg.sim.current_time = 17, 0.5
    evaluator = _FakeEvaluator(static_rgb=True)
    hooks = DartOgRuntimeHooks(evaluator=evaluator, torch_module=_FakeTorch, og_module=_FakeOg, run_id="live-1", max_control_steps=2)

    pre = hooks.observe_current()
    receipt = hooks.apply_current_raw23([0.125] * 23, pre)

    assert evaluator.applied[0].tolist() == [[0.125] * 23]
    assert receipt.pre_action_policy_clock == 0
    assert len(hooks.transitions) == 1
    transition = hooks.transitions[0]
    assert transition.pre.dart.policy_clock == 0
    assert transition.post.dart.policy_clock == 1
    assert transition.pre.sim_step == 17 and transition.post.sim_step == 18
    assert transition.post.sim_time_seconds > transition.pre.sim_time_seconds
    assert [view.camera_id for view in transition.pre.rgb] == ["left_wrist", "right_wrist", "head"]
    # Static scene pixels may be byte-identical.  The immediately preceding
    # physical step and independently advancing clocks prove the post capture
    # is fresh; hash inequality would be an invalid requirement.
    assert transition.pre.rgb[0].payload == transition.post.rgb[0].payload
    hooks.close()
    assert evaluator.closed is True


def test_terminal_after_physical_step_fails_closed_without_reusable_pre_state() -> None:
    _FakeOg.sim.current_time_step_index, _FakeOg.sim.current_time = 17, 0.5
    evaluator = _FakeEvaluator(terminal=True)
    hooks = DartOgRuntimeHooks(evaluator=evaluator, torch_module=_FakeTorch, og_module=_FakeOg, run_id="live-1", max_control_steps=2)
    pre = hooks.observe_current()
    with pytest.raises(RecoveryContractError, match="terminated"):
        hooks.apply_current_raw23([0.0] * 23, pre)
    assert len(evaluator.applied) == 1
    with pytest.raises(RecoveryContractError, match="unconsumed"):
        hooks.apply_current_raw23([0.0] * 23, pre)


def test_runtime_rejects_preprocessed_float_or_chw_rgb_before_any_step() -> None:
    evaluator = _FakeEvaluator()
    evaluator.state.obs["r1pro::head::rgb"] = np.zeros((3, 2, 3), dtype=np.float32)
    hooks = DartOgRuntimeHooks(evaluator=evaluator, torch_module=_FakeTorch, og_module=_FakeOg, run_id="live-1", max_control_steps=2)
    with pytest.raises(RecoveryContractError, match="native uint8 HxWx3"):
        hooks.observe_current()
    assert evaluator.applied == []


def test_runtime_rejects_non_float32_gaussian_outlier_before_controller_step() -> None:
    evaluator = _FakeEvaluator()
    hooks = DartOgRuntimeHooks(evaluator=evaluator, torch_module=_FakeTorch, og_module=_FakeOg, run_id="live-1", max_control_steps=2)
    pre = hooks.observe_current()
    with pytest.raises(RecoveryContractError, match="float32"):
        hooks.apply_current_raw23([1.0e39] + [0.0] * 22, pre)
    assert evaluator.applied == []


def test_runtime_receipt_hashes_the_exact_nontrivial_float32_tensor_submitted_to_evaluator() -> None:
    _FakeOg.sim.current_time_step_index, _FakeOg.sim.current_time = 17, 0.5
    evaluator = _FakeEvaluator()
    hooks = DartOgRuntimeHooks(evaluator=evaluator, torch_module=_FakeTorch, og_module=_FakeOg, run_id="live-1", max_control_steps=2)
    pre = hooks.observe_current()
    requested = [1.0 / 3.0] + [-(index + 1.0) / 17.0 for index in range(22)]
    receipt = hooks.apply_current_raw23(requested, pre)
    assert receipt.applied23 == canonical_native_raw23_float32(requested)
    assert receipt.applied23 != tuple(requested)
    assert sha256(evaluator.applied[0].tobytes()).hexdigest() == receipt.applied_action_bytes_sha256
    assert evaluator.applied[0].dtype == np.float32


def test_private_writer_encodes_captured_rgb_and_keeps_raw_state61_out_of_actor_projection(tmp_path) -> None:
    _FakeOg.sim.current_time_step_index, _FakeOg.sim.current_time = 17, 0.5
    evaluator = _FakeEvaluator(static_rgb=True)
    hooks = DartOgRuntimeHooks(evaluator=evaluator, torch_module=_FakeTorch, og_module=_FakeOg, run_id="live-1", max_control_steps=2)
    pre = hooks.observe_current()
    applied = hooks.apply_current_raw23([0.125] * 23, pre)
    source, teacher, membership = _source(), _unqualified_teacher(), _verified_membership(_source())
    runtime = _runtime_session(source)
    intent = "sealed-grasp-intent"
    requested_native = canonical_native_raw23_float32([0.125] * 23)
    record = {
        "schema": "p107_dart_candidate_v2",
        "label_kind": "dart_inspired_actual_clean_recovery",
        "collection_mode": "dart_inspired_bounded_actual_clean_recovery",
        "source": source.public(),
        "verified_source_membership": membership.public(),
        "runtime_session": runtime.public(),
        "teacher": teacher.public(),
        "teacher_query_receipt_sha256": _sha("teacher-query"),
        "intent_bundle_id": intent,
        "clean_intended23": [0.125] * 23,
        "requested_noisy23": [0.125] * 23,
        "sampled_noise23": [0.0] * 23,
        "requested_native23": list(requested_native),
        "requested_native_action_bytes_sha256": sha256(evaluator.applied[0].tobytes()).hexdigest(),
        "native_action_wire_dtype": "float32_le",
        "applied23": list(applied.applied23),
        "applied_action_bytes_sha256": applied.applied_action_bytes_sha256,
        "runtime_step_receipt_sha256": applied.runtime_step_receipt_sha256,
        "noise_profile": {
            "algorithm_id": "dart_inspired_bounded_correlated",
            "profile_id": "fixture",
            "profile_sha256": _sha("noise-profile"),
            "embodiment_metadata_sha256": _sha("fixture-embodiment"),
            "model_projection_manifest_sha256": _sha("fixture-projection"),
            "sampling_seed": 17,
            "original_dart_equivalence": False,
        },
        "calibration_receipt": {
            "calibration_trajectory_sha256": _sha("calibration-trajectory"),
            "learner_checkpoint_sha256": _sha("learner"),
            "teacher_checkpoint_sha256": _sha("teacher"),
            "source_group_ids": [_sha("calibration-group")],
        },
        "candidate_only": True,
        "training_eligible": False,
        "ready_for_training": False,
        "low_action_supervision_positive": False,
        "authority_minted": False,
        "authority_status": "NO_AUTHORITY",
    }
    writer_config = PrivateEpisodeWriterConfig(
        output=tmp_path / "fresh-private-bundle",
        source_episode_id="selected-source-episode-19",
        source=source,
        verified_membership=membership,
        runtime_session=runtime,
        teacher=teacher,
        collector_code_sha256=_sha("collector-code"),
        capture_adapter_sha256=_sha("capture-adapter"),
        actor_observation_schema_sha256=_sha("actor-schema"),
    )
    writer = build_private_episode_bundle_writer(writer_config)
    receipt = writer(
        DartCollectionResult((record,), "OUTCOME_UNKNOWN", None),
        hooks.transitions,
        collection_execution_provenance=_fixture_execution_provenance(record["collection_mode"]),
    )
    loaded = load_bundle(receipt.episode_bundle_path, expected_manifest_sha256=receipt.episode_manifest_sha256)
    projection = loaded.actor_projection(0)
    assert projection.proprioception is None
    assert projection.rgb_by_view["head"]
    serialized = (receipt.episode_bundle_path / "records.jsonl").read_text(encoding="utf-8")
    assert '"state61":' not in serialized and "target_name" not in serialized and "contact" not in serialized
    sidecar = json.loads((receipt.path / "private_collection.json").read_text(encoding="utf-8"))
    assert sidecar["collection_result"]["records"][0]["noise_profile"] == record["noise_profile"]
    assert sidecar["collection_result"]["records"][0]["calibration_receipt"] == record["calibration_receipt"]
    assert sidecar["training_eligible"] is False
    tampered_receipt = _sha("different-runtime-step-receipt")
    tampered_transition = replace(
        hooks.transitions[0],
        applied=replace(hooks.transitions[0].applied, runtime_step_receipt_sha256=tampered_receipt),
    )
    tampered_record = {**record, "runtime_step_receipt_sha256": tampered_receipt}
    tampered_writer = build_private_episode_bundle_writer(
        replace(writer_config, output=tmp_path / "tampered-private-bundle")
    )
    with pytest.raises(RecoveryContractError, match="post-observation digest"):
        tampered_writer(
            DartCollectionResult((tampered_record,), "OUTCOME_UNKNOWN", None),
            (tampered_transition,),
            collection_execution_provenance=_fixture_execution_provenance(tampered_record["collection_mode"]),
        )
    tampered_native_writer = build_private_episode_bundle_writer(
        replace(writer_config, output=tmp_path / "tampered-native-private-bundle")
    )
    with pytest.raises(RecoveryContractError, match="native raw23 wire provenance"):
        tampered_native_writer(
            DartCollectionResult(({**record, "requested_native23": [0.0] * 23},), "OUTCOME_UNKNOWN", None),
            hooks.transitions,
            collection_execution_provenance=_fixture_execution_provenance(record["collection_mode"]),
        )
    missing_provenance_output = tmp_path / "missing-provenance-private-bundle"
    with pytest.raises(RecoveryContractError, match="noise and calibration provenance"):
        build_private_episode_bundle_writer(replace(writer_config, output=missing_provenance_output))(
            DartCollectionResult(({key: value for key, value in record.items() if key != "noise_profile"},), "OUTCOME_UNKNOWN", None),
            hooks.transitions,
            collection_execution_provenance=_fixture_execution_provenance(record["collection_mode"]),
        )
    assert not missing_provenance_output.exists()
    missing_execution_output = tmp_path / "missing-execution-private-bundle"
    with pytest.raises(RecoveryContractError, match="execution provenance"):
        build_private_episode_bundle_writer(replace(writer_config, output=missing_execution_output))(
            DartCollectionResult((record,), "OUTCOME_UNKNOWN", None), hooks.transitions
        )
    assert not missing_execution_output.exists()
    (receipt.path / "private_collection.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RecoveryContractError, match="file receipt"):
        load_private_episode_publication(receipt.path, expected_manifest_sha256=receipt.manifest_sha256)


def test_private_writer_preserves_original_gaussian_math_and_native_wire_in_outer_publication(tmp_path) -> None:
    _FakeOg.sim.current_time_step_index, _FakeOg.sim.current_time = 17, 0.5
    evaluator = _FakeEvaluator(static_rgb=True)
    hooks = DartOgRuntimeHooks(evaluator=evaluator, torch_module=_FakeTorch, og_module=_FakeOg, run_id="live-1", max_control_steps=2)
    source, teacher, membership = _source(), _unqualified_teacher(), _verified_membership(_source())
    runtime = _runtime_session(source)
    clean_value, sampled_value = 1.0 / 3.0, 0.25
    requested_value = clean_value + sampled_value
    pre = hooks.observe_current()
    applied = hooks.apply_current_raw23([requested_value] * 23, pre)
    native = canonical_native_raw23_float32([requested_value] * 23)
    assert tuple(applied.applied23) == native and native != tuple([requested_value] * 23)
    record = {
        "schema": "p107_dart_candidate_v2",
        "label_kind": "dart_clean_supervisor_feedback",
        "collection_mode": "original_gaussian_clean_intended_feedback",
        "source": source.public(),
        "verified_source_membership": membership.public(),
        "runtime_session": runtime.public(),
        "teacher": teacher.public(),
        "teacher_query_receipt_sha256": _sha("original-teacher-query"),
        "intent_bundle_id": "sealed-private-intent",
        "clean_intended23": [clean_value] * 23,
        "requested_noisy23": [requested_value] * 23,
        "sampled_noise23": [sampled_value] * 23,
        "requested_native23": list(native),
        "requested_native_action_bytes_sha256": applied.applied_action_bytes_sha256,
        "native_action_wire_dtype": "float32_le",
        "applied23": list(applied.applied23),
        "applied_action_bytes_sha256": applied.applied_action_bytes_sha256,
        "runtime_step_receipt_sha256": applied.runtime_step_receipt_sha256,
        "noise_profile": {
            "algorithm_id": "dart_original_gaussian_unbounded",
            "covariance_sha256": _fixture_execution_provenance(
                "original_gaussian_clean_intended_feedback"
            )["original_gaussian"]["covariance23_sha256"],
            "sampling_seed": 31,
            "covariance_update_mode": "frozen_one_pass_partial_dart",
            "covariance_calibration_binding": _fixture_execution_provenance(
                "original_gaussian_clean_intended_feedback"
            )["original_gaussian"]["calibration_binding"],
        },
        "calibration_receipt": {
            "calibration_trajectory_sha256": _sha("calibration-trajectory"),
            "learner_checkpoint_sha256": _sha("learner"),
            "teacher_checkpoint_sha256": _sha("teacher"),
            "source_group_ids": [_sha("calibration-group")],
        },
        "candidate_only": True,
        "training_eligible": False,
        "ready_for_training": False,
        "low_action_supervision_positive": False,
        "authority_minted": False,
        "authority_status": "NO_AUTHORITY",
    }
    receipt = build_private_episode_bundle_writer(
        PrivateEpisodeWriterConfig(
            output=tmp_path / "original-gaussian-private-publication",
            source_episode_id="selected-source-episode-20",
            source=source,
            verified_membership=membership,
            runtime_session=runtime,
            teacher=teacher,
            collector_code_sha256=_sha("collector-code"),
            capture_adapter_sha256=_sha("capture-adapter"),
            actor_observation_schema_sha256=_sha("actor-schema"),
        )
    )(
        DartCollectionResult((record,), "OUTCOME_UNKNOWN", None),
        hooks.transitions,
        collection_execution_provenance=_fixture_execution_provenance(record["collection_mode"]),
    )
    verified = load_private_episode_publication(receipt.path, expected_manifest_sha256=receipt.manifest_sha256)
    assert verified.collection_result_sha256
    stored = json.loads((receipt.path / "private_collection.json").read_text(encoding="utf-8"))
    row = stored["collection_result"]["records"][0]
    assert row["requested_noisy23"] != row["requested_native23"]
    assert row["requested_native23"] == row["applied23"]


def test_factory_collection_loops_publish_bounded_and_original_candidate_only_bundles(tmp_path) -> None:
    """Fixture wiring only: real factory loop -> native receipt -> atomic writer."""

    source, teacher, membership = _source(), _unqualified_teacher(), _verified_membership(_source())
    runtime_session = _runtime_session(source)
    calibration = CalibrationReceipt(
        source_group_ids=(membership.calibration_groups[0].source_group_id,),
        calibration_trajectory_sha256=_sha("calibration-trajectory"),
        learner_checkpoint_sha256=_sha("learner"),
        teacher_checkpoint_sha256=_sha("teacher"),
    )

    class Reader:
        def verify_dart_membership(self, requested_source, requested_calibration):
            assert requested_source == source and requested_calibration == calibration
            return membership

    class CurrentTeacher:
        def clean_action(self, observation, *, intent_bundle_id):
            return TeacherCommand(
                clean_intended23=[1.0 / 3.0] * 23,
                observed_policy_clock=observation.policy_clock,
                observed_state61_sha256=observation.state61_sha256,
                observed_observation_sha256=observation.observation_sha256,
                intent_bundle_id=intent_bundle_id,
                fresh_query_receipt_sha256=_sha(f"factory-teacher:{observation.policy_clock}"),
            )

    def make_factory() -> SimpleNamespace:
        _FakeOg.sim.current_time_step_index, _FakeOg.sim.current_time = 17, 0.5
        hooks = DartOgRuntimeHooks(
            evaluator=_FakeEvaluator(static_rgb=True),
            torch_module=_FakeTorch,
            og_module=_FakeOg,
            run_id=source.collection_run_id,
            max_control_steps=3,
        )
        return SimpleNamespace(
            dart_runtime=FreshRolloutRaw23Runtime(hooks),
            teacher_bridge=CurrentTeacher(),
            hooks=hooks,
            train_selection=_selection(source),
            pins=SimpleNamespace(
                runtime_build_sha256=runtime_session.runtime_build_sha256,
                asset_config_sha256=runtime_session.asset_config_sha256,
            ),
            close=hooks.close,
        )

    def writer(output):
        return build_private_episode_bundle_writer(
            PrivateEpisodeWriterConfig(
                output=output,
                source_episode_id="factory-loop-fixture",
                source=source,
                verified_membership=membership,
                runtime_session=runtime_session,
                teacher=teacher,
                collector_code_sha256=_sha("collector-code"),
                capture_adapter_sha256=_sha("capture-adapter"),
                actor_observation_schema_sha256=_sha("actor-schema"),
            )
        )

    original_noise = OriginalGaussianNoise(
        tuple(tuple(0.25 if row == column else 0.0 for column in range(23)) for row in range(23)), seed=31
    )
    original_binding = OriginalGaussianCalibrationBinding(
        calibration_trajectory_sha256=calibration.calibration_trajectory_sha256,
        learner_checkpoint_sha256=calibration.learner_checkpoint_sha256,
        teacher_checkpoint_sha256=calibration.teacher_checkpoint_sha256,
        covariance_estimator_code_sha256=_sha("covariance-estimator"),
        estimated_covariance23=tuple(tuple(1.0 if row == column else 0.0 for column in range(23)) for row in range(23)),
        alpha=5.75,
        horizon=1,
        source_group_index_manifest_sha256=membership.source_group_index_manifest_sha256,
    )
    original_output = tmp_path / "original-factory-publication"
    original_result = run_live_grasp_collection(
        factory=make_factory(),
        mode="original_gaussian_one_pass_partial_dart",
        teacher_receipt=teacher,
        source=source,
        source_membership_reader=Reader(),
        runtime_session=runtime_session,
        calibration=calibration,
        intent_bundle_id="factory-loop-intent",
        writer=writer(original_output),
        original_noise=original_noise,
        original_steps=1,
        original_binding=original_binding,
    )
    original_loaded = load_private_episode_publication(original_output)
    assert len(original_result.records) == original_loaded.transition_count == 1
    assert original_result.records[0]["requested_noisy23"] != original_result.records[0]["applied23"]
    original_sidecar = json.loads((original_output / "private_collection.json").read_text(encoding="utf-8"))
    original_execution = original_sidecar["collection_execution_provenance"]
    assert original_execution["original_gaussian"]["covariance23"] == [list(row) for row in original_noise.covariance]
    assert original_execution["original_gaussian"]["calibration_binding"] == original_binding.public()

    layout = Raw23ActionLayout(
        (
            ActionPart("base_qvel", 3, "velocity"),
            ActionPart("trunk_qpos", 4, "position"),
            ActionPart("left_arm", 7, "position"),
            ActionPart("left_gripper", 1, "position"),
            ActionPart("right_arm", 7, "position"),
            ActionPart("right_gripper", 1, "position"),
        ),
        embodiment_metadata_sha256=_sha("embodiment"),
        model_projection_manifest_sha256=_sha("projection"),
    )
    rules = {
        part.name: BoundedPartRule(
            active=part.name == "left_arm",
            standard_deviation=0.1 if part.name == "left_arm" else 0.0,
            cap=0.2 if part.name == "left_arm" else 0.0,
            inactive_reason=None if part.name == "left_arm" else "fixture inactive",
        )
        for part in layout.parts
    }
    bounded_noise = DartInspiredBoundedNoise(
        BoundedNoiseProfile(layout=layout, rules_by_part=rules, temporal_rho=0.0, profile_id="factory-fixture-bounded"),
        seed=17,
    )
    bounded_output = tmp_path / "bounded-factory-publication"
    bounded_result = run_live_grasp_collection(
        factory=make_factory(),
        mode="dart_inspired_bounded_actual_clean_recovery",
        teacher_receipt=teacher,
        source=source,
        source_membership_reader=Reader(),
        runtime_session=runtime_session,
        calibration=calibration,
        intent_bundle_id="factory-loop-intent",
        writer=writer(bounded_output),
        bounded_noise=bounded_noise,
        noisy_injection_steps=1,
        recovery_chunk_lengths=(1,),
        limits=CollectionLimits(max_total_steps=2, max_wall_seconds=30),
    )
    bounded_loaded = load_private_episode_publication(bounded_output)
    assert len(bounded_result.records) == bounded_loaded.transition_count == 2
    assert {record["label_kind"] for record in bounded_result.records} == {
        "dart_inspired_noisy_injection",
        "dart_inspired_actual_clean_recovery",
    }
    assert all(record["training_eligible"] is False for record in bounded_result.records)
    bounded_sidecar = json.loads((bounded_output / "private_collection.json").read_text(encoding="utf-8"))
    bounded_execution = bounded_sidecar["collection_execution_provenance"]["bounded_dart"]
    assert bounded_execution["profile"]["temporal_rho"] == 0.0
    assert bounded_execution["profile"]["rules_by_part"] == {
        part.name: {
            "active": rule.active,
            "standard_deviation": rule.standard_deviation,
            "cap": rule.cap,
            "inactive_reason": rule.inactive_reason,
        }
        for part, rule in ((part, rules[part.name]) for part in layout.parts)
    }
    assert bounded_execution["noisy_injection_steps"] == 1
    assert bounded_execution["recovery_chunk_lengths"] == [1]
    assert bounded_execution["limits"] == {"max_total_steps": 2, "max_wall_seconds": 30}


def test_private_writer_rejects_record_that_loses_candidate_only_gate(tmp_path) -> None:
    source, teacher, membership = _source(), _unqualified_teacher(), _verified_membership(_source())
    config = PrivateEpisodeWriterConfig(
        output=tmp_path / "fresh-private-bundle",
        source_episode_id="selected-source-episode-19",
        source=source,
        verified_membership=membership,
        runtime_session=_runtime_session(source),
        teacher=teacher,
        collector_code_sha256=_sha("collector-code"),
        capture_adapter_sha256=_sha("capture-adapter"),
        actor_observation_schema_sha256=_sha("actor-schema"),
    )
    evaluator = _FakeEvaluator()
    hooks = DartOgRuntimeHooks(evaluator=evaluator, torch_module=_FakeTorch, og_module=_FakeOg, run_id="live-1", max_control_steps=2)
    pre = hooks.observe_current()
    hooks.apply_current_raw23([0.0] * 23, pre)
    with pytest.raises(RecoveryContractError, match="candidate-only"):
        build_private_episode_bundle_writer(config)(
            DartCollectionResult(({"candidate_only": False},), "OUTCOME_UNKNOWN", None),
            hooks.transitions,
        )


def test_native_reader_projection_uses_exact_selected_env0_objects() -> None:
    evaluator = _FakeEvaluator()
    view = PinnedSingleEnvNativeReaderView.from_evaluator(evaluator)
    assert view.robots == [evaluator.robot]
    assert view.scene is evaluator.scene
    assert view.task.object_scope is evaluator.env.task.object_scopes[0]


def test_native_reader_projection_rejects_accessor_from_another_scene() -> None:
    evaluator = _FakeEvaluator()
    evaluator.state.env_accessor.scene = SimpleNamespace(idx=9, robots=[evaluator.robot])
    with pytest.raises(RecoveryContractError, match="selected env0"):
        PinnedSingleEnvNativeReaderView.from_evaluator(evaluator)


def test_train_selection_binds_every_opaque_source_field() -> None:
    source = _source()
    _selection(source).assert_source(source)
    changed = CandidateSourceReceipt(
        **{**source.public(), "parent_task_instance_id": source.parent_task_instance_id + 1}
    )
    with pytest.raises(RecoveryContractError, match="differs"):
        _selection(source).assert_source(changed)


def test_factory_pins_reject_config_byte_tamper(tmp_path) -> None:
    evaluator_config, r1pro_config = tmp_path / "evaluator.yaml", tmp_path / "r1pro.yaml"
    evaluator_config.write_bytes(b"mode: train\n")
    r1pro_config.write_bytes(b"robot: r1pro\n")
    pins = DartOgFactoryPins(
        og_source_commit="a" * 40,
        evaluator_config_path=evaluator_config,
        evaluator_config_sha256=sha256(evaluator_config.read_bytes()).hexdigest(),
        r1pro_config_path=r1pro_config,
        r1pro_config_sha256=sha256(r1pro_config.read_bytes()).hexdigest(),
        runtime_build_sha256=_sha("runtime"),
        asset_config_sha256=_sha("assets"),
    )
    assert pins.og_source_commit == "a" * 40
    evaluator_config.write_bytes(b"mode: public_test\n")
    with pytest.raises(RecoveryContractError, match="evaluator config bytes"):
        DartOgFactoryPins(
            og_source_commit="a" * 40,
            evaluator_config_path=evaluator_config,
            evaluator_config_sha256=pins.evaluator_config_sha256,
            r1pro_config_path=r1pro_config,
            r1pro_config_sha256=pins.r1pro_config_sha256,
            runtime_build_sha256=_sha("runtime"),
            asset_config_sha256=_sha("assets"),
        )


def test_factory_lazy_wiring_loads_only_the_sealed_train_instance_and_scalar_reader_view(monkeypatch, tmp_path) -> None:
    config, r1pro = tmp_path / "evaluator.yaml", tmp_path / "r1pro.yaml"
    config.write_text("fixture evaluator config", encoding="utf-8")
    r1pro.write_text("fixture r1pro config", encoding="utf-8")
    pins = DartOgFactoryPins(
        og_source_commit="b" * 40,
        evaluator_config_path=config,
        evaluator_config_sha256=sha256(config.read_bytes()).hexdigest(),
        r1pro_config_path=r1pro,
        r1pro_config_sha256=sha256(r1pro.read_bytes()).hexdigest(),
        runtime_build_sha256=_sha("runtime"),
        asset_config_sha256=_sha("assets"),
    )
    source, evaluator = _source(), _FakeEvaluator()
    cfg = _Config(mode="train", task=SimpleNamespace(name="task_name_from_source_index"), seed=123, num_envs=99)
    captured = _install_fake_pinned_runtime(monkeypatch, cfg=cfg, evaluator=evaluator)
    binding = _binding(source)
    factory = DartOgGraspFactory.create(
        pins=pins,
        source=source,
        train_selection=_selection(source),
        binding=binding,
        teacher_spec=_full_teacher_spec(binding),
        teacher_reference={"private_original_semantic_json": "[]"},
        run_id="live-test",
        max_control_steps=3,
    )
    assert evaluator.loaded_batches == [{0: source.parent_task_instance_id}]
    assert captured["cfg"].num_envs == 1
    assert isinstance(captured["reader_env"], PinnedSingleEnvNativeReaderView)
    assert captured["reader_env"].scene is evaluator.scene
    assert captured["validated_spec"] == _full_teacher_spec(binding)
    factory.close()
    assert evaluator.closed is True


def test_factory_rejects_implicit_public_test_before_evaluator_creation(monkeypatch, tmp_path) -> None:
    config, r1pro = tmp_path / "evaluator.yaml", tmp_path / "r1pro.yaml"
    config.write_text("fixture evaluator config", encoding="utf-8")
    r1pro.write_text("fixture r1pro config", encoding="utf-8")
    pins = DartOgFactoryPins(
        og_source_commit="b" * 40,
        evaluator_config_path=config,
        evaluator_config_sha256=sha256(config.read_bytes()).hexdigest(),
        r1pro_config_path=r1pro,
        r1pro_config_sha256=sha256(r1pro.read_bytes()).hexdigest(),
        runtime_build_sha256=_sha("runtime"),
        asset_config_sha256=_sha("assets"),
    )
    source, evaluator = _source(), _FakeEvaluator()
    cfg = _Config(mode="public_test", task=SimpleNamespace(name="task_name_from_source_index"), seed=123)
    captured = _install_fake_pinned_runtime(monkeypatch, cfg=cfg, evaluator=evaluator)
    binding = _binding(source)
    with pytest.raises(RecoveryContractError, match="mode=train"):
        DartOgGraspFactory.create(
            pins=pins,
            source=source,
            train_selection=_selection(source),
            binding=binding,
            teacher_spec=_full_teacher_spec(binding),
            teacher_reference={"private_original_semantic_json": "[]"},
            run_id="live-test",
            max_control_steps=3,
        )
    assert "cfg" not in captured
    assert evaluator.closed is False


def test_factory_closes_evaluator_when_loaded_scene_differs_from_sealed_train_selection(monkeypatch, tmp_path) -> None:
    config, r1pro = tmp_path / "evaluator.yaml", tmp_path / "r1pro.yaml"
    config.write_text("fixture evaluator config", encoding="utf-8")
    r1pro.write_text("fixture r1pro config", encoding="utf-8")
    pins = DartOgFactoryPins(
        og_source_commit="b" * 40,
        evaluator_config_path=config,
        evaluator_config_sha256=sha256(config.read_bytes()).hexdigest(),
        r1pro_config_path=r1pro,
        r1pro_config_sha256=sha256(r1pro.read_bytes()).hexdigest(),
        runtime_build_sha256=_sha("runtime"),
        asset_config_sha256=_sha("assets"),
    )
    source, evaluator = _source(), _FakeEvaluator()
    evaluator.env.task.scene_name = "wrong_scene"
    cfg = _Config(mode="train", task=SimpleNamespace(name="task_name_from_source_index"), seed=123)
    _install_fake_pinned_runtime(monkeypatch, cfg=cfg, evaluator=evaluator)
    binding = _binding(source)
    with pytest.raises(RecoveryContractError, match="scene differs"):
        DartOgGraspFactory.create(
            pins=pins,
            source=source,
            train_selection=_selection(source),
            binding=binding,
            teacher_spec=_full_teacher_spec(binding),
            teacher_reference={"private_original_semantic_json": "[]"},
            run_id="live-test",
            max_control_steps=3,
        )
    assert evaluator.closed is True
