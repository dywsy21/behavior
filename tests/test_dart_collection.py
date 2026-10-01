from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

try:
    import pytest
except ImportError:  # Keep these focused contract tests runnable without installing pytest.
    import re

    class _Raises:
        def __init__(self, error: type[Exception], match: str) -> None:
            self.error = error
            self.match = match

        def __enter__(self) -> None:
            return None

        def __exit__(self, error_type: type[Exception] | None, error: Exception | None, traceback: object) -> bool:
            if error_type is None:
                raise AssertionError(f"expected {self.error.__name__}")
            if not issubclass(error_type, self.error) or not re.search(self.match, str(error)):
                return False
            return True

    class _PytestFallback:
        @staticmethod
        def raises(error: type[Exception], *, match: str) -> _Raises:
            return _Raises(error, match)

    pytest = _PytestFallback()

from g05.recovery.common import RecoveryContractError, canonical_sha256
from g05.recovery.dart_collection import (
    AppliedActionReceipt,
    CalibrationReceipt,
    CandidateSourceReceipt,
    CollectionLimits,
    DartObservation,
    OutcomeEvidenceReceipt,
    DartCollectionResult,
    RuntimeSessionReceipt,
    TeacherCommand,
    TeacherReceipt,
    collect_dart_inspired_actual_clean_recovery,
    collect_original_gaussian_clean_feedback,
)
from g05.recovery.dart_noise import (
    ActionPart,
    BoundedNoiseProfile,
    BoundedPartRule,
    DartInspiredBoundedNoise,
    OriginalGaussianNoise,
    Raw23ActionLayout,
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _action(value: float) -> tuple[float, ...]:
    return (value,) * 23


def _diagonal(value: float) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(value if row == column else 0.0 for column in range(23)) for row in range(23))


class FakeRuntime:
    def __init__(self, *, abort: bool = False, receipt_clock_offset: int = 0) -> None:
        self.clock = 0
        self.abort = abort
        self.receipt_clock_offset = receipt_clock_offset
        self.requests: list[tuple[float, ...]] = []

    def observe(self) -> DartObservation:
        return DartObservation(
            policy_clock=self.clock,
            state61=(float(self.clock),) * 61,
            observation_sha256=_digest(f"observation:{self.clock}"),
            observation_ref=f"rgb://candidate/{self.clock}",
        )

    def apply_raw23(self, requested23: tuple[float, ...]) -> AppliedActionReceipt:
        requested = tuple(requested23)
        self.requests.append(requested)
        receipt = AppliedActionReceipt(
            status="ABORTED" if self.abort else "APPLIED",
            applied23=requested,
            applied_action_bytes_sha256=_digest(f"raw-float-bytes:{requested!r}"),
            runtime_step_receipt_sha256=_digest(f"step:{self.clock}"),
            pre_action_policy_clock=self.clock + self.receipt_clock_offset,
            pre_action_state61_sha256=canonical_sha256((float(self.clock + self.receipt_clock_offset),) * 61),
            pre_action_observation_sha256=_digest(f"observation:{self.clock + self.receipt_clock_offset}"),
        )
        if not self.abort:
            self.clock += 1
        return receipt


class FakeTeacher:
    def __init__(self, *, future_leak: bool = False) -> None:
        self.future_leak = future_leak
        self.observed_clocks: list[int] = []

    def clean_action(self, observation: DartObservation, *, intent_bundle_id: str) -> TeacherCommand:
        self.observed_clocks.append(observation.policy_clock)
        bad_clock = observation.policy_clock + 1 if self.future_leak else observation.policy_clock
        return TeacherCommand(
            clean_intended23=_action(float(observation.policy_clock + 1)),
            observed_policy_clock=bad_clock,
            observed_state61_sha256=observation.state61_sha256,
            observed_observation_sha256=observation.observation_sha256,
            intent_bundle_id=intent_bundle_id,
            fresh_query_receipt_sha256=_digest(f"teacher-query:{observation.policy_clock}:{intent_bundle_id}"),
        )


def _source(group: str = "candidate-group") -> CandidateSourceReceipt:
    return CandidateSourceReceipt(
        source_group_id=group,
        original_split="train",
        source_release_sha256=_digest("source-release"),
        source_episode_sha256=_digest("source-episode"),
        task_id="task-42",
        task_instance_id="task-42-instance-9",
        task_seed=9,
    )


def _calibration() -> CalibrationReceipt:
    return CalibrationReceipt(
        source_group_ids=("calibration-group",),
        calibration_trajectory_sha256=_digest("calibration-trajs"),
        learner_checkpoint_sha256=_digest("learner"),
        teacher_checkpoint_sha256=_digest("teacher-calibration"),
    )


def _runtime_session() -> RuntimeSessionReceipt:
    return RuntimeSessionReceipt(
        runtime_session_id="fresh-session-17",
        task_instance_id="task-42-instance-9",
        runtime_build_sha256=_digest("runtime-build"),
        asset_config_sha256=_digest("asset-config"),
        reset_load_task_instance_receipt_sha256=_digest("reset-load-task-instance"),
    )


def _teacher_receipt() -> TeacherReceipt:
    return TeacherReceipt(
        teacher_id="frozen-expert-baseline",
        teacher_kind="planner_or_human_must_be_declared_by_adapter",
        label_source="planner",
        feedback_mode="closed_loop_fresh_observation",
        postcondition_spec_sha256=_digest("teacher-postcondition"),
        teacher_code_sha256=_digest("teacher-code"),
        teacher_weights_sha256=_digest("teacher-weights"),
        teacher_config_sha256=_digest("teacher-config"),
        runtime_code_sha256=_digest("runtime-code"),
    )


def _bounded_noise() -> DartInspiredBoundedNoise:
    layout = Raw23ActionLayout(
        (
            ActionPart("base_qvel", 3, "velocity"),
            ActionPart("trunk_qpos", 4, "position"),
            ActionPart("left_arm", 7, "position"),
            ActionPart("left_gripper", 1, "position"),
            ActionPart("right_arm", 7, "position"),
            ActionPart("right_gripper", 1, "position"),
        )
    )
    return DartInspiredBoundedNoise(
        BoundedNoiseProfile(
            layout=layout,
            rules_by_part={
                "base_qvel": BoundedPartRule(False, inactive_reason="declared initial hypothesis"),
                "trunk_qpos": BoundedPartRule(False, inactive_reason="declared initial hypothesis"),
                "left_arm": BoundedPartRule(True, standard_deviation=1.0, cap=0.2),
                "left_gripper": BoundedPartRule(False, inactive_reason="declared initial hypothesis"),
                "right_arm": BoundedPartRule(False, inactive_reason="declared initial hypothesis"),
                "right_gripper": BoundedPartRule(False, inactive_reason="declared initial hypothesis"),
            },
            temporal_rho=0.25,
            profile_id="bounded-pilot-v1",
        ),
        seed=17,
    )


def build_cli_candidate(request: dict[str, object]) -> DartCollectionResult:
    """Importable integration-factory fixture for the inert CLI contract."""

    record = {
        "schema": "p107_dart_candidate_v1",
        "label_kind": "dart_clean_supervisor_feedback",
        "request_id": request["request_id"],
        "candidate_only": True,
        "training_eligible": False,
        "ready_for_training": False,
        "low_action_supervision_positive": False,
        "authority_minted": False,
    }
    record["candidate_sha256"] = canonical_sha256(record)
    return DartCollectionResult((record,), outcome_label="OUTCOME_UNKNOWN", stop_reason=None)


def test_original_mode_records_clean_feedback_separately_from_noisy_applied_execution() -> None:
    runtime = FakeRuntime()
    teacher = FakeTeacher()
    result = collect_original_gaussian_clean_feedback(
        runtime=runtime,
        teacher_callback=teacher,
        teacher_receipt=_teacher_receipt(),
        source=_source(),
        runtime_session=_runtime_session(),
        calibration=_calibration(),
        noise=OriginalGaussianNoise(_diagonal(0.25), seed=31),
        intent_bundle_id="intent-v1",
        steps=2,
        covariance_update_mode="frozen_one_pass_partial_dart",
    )

    assert teacher.observed_clocks == [0, 1]
    assert len(result.records) == 2
    first, second = result.records
    assert first["label_kind"] == "dart_clean_supervisor_feedback"
    assert first["clean_target_steps"] == 1
    assert first["clean_intended23"] != first["applied23"]
    assert first["requested_noisy23"] == first["applied23"]
    assert first["noise_profile"]["empirical_faithfulness"] is False
    assert second["observation"]["policy_clock"] == 1
    assert all(not record["training_eligible"] for record in result.records)
    assert all(record["candidate_sha256"] == canonical_sha256({key: value for key, value in record.items() if key != "candidate_sha256"}) for record in result.records)


def test_bounded_mode_audits_noisy_control_and_only_actual_clean_recovery_as_candidate() -> None:
    runtime = FakeRuntime()
    teacher = FakeTeacher()
    evidence = OutcomeEvidenceReceipt(
        label="SURVIVAL_NONFAILURE",
        observed_policy_clock=0,
        available_policy_clock=3,
        evidence_sha256=_digest("physical-evidence"),
    )
    result = collect_dart_inspired_actual_clean_recovery(
        runtime=runtime,
        teacher_callback=teacher,
        teacher_receipt=_teacher_receipt(),
        source=_source(),
        runtime_session=_runtime_session(),
        calibration=_calibration(),
        noise=_bounded_noise(),
        intent_bundle_id="intent-v1",
        noisy_injection_steps=1,
        recovery_chunk_lengths=(2,),
        limits=CollectionLimits(max_total_steps=4, max_wall_seconds=60),
        outcome_evidence=evidence,
    )

    injection, clean_one, clean_two = result.records
    assert injection["label_kind"] == "dart_inspired_noisy_injection"
    assert injection["requested_noisy23"] == injection["applied23"]
    assert injection["clean_intended23"] != injection["applied23"]
    assert injection["execution_role"] == "noisy_injection_excluded_from_fm_supervision"
    assert clean_one["label_kind"] == "dart_inspired_actual_clean_recovery"
    assert clean_one["clean_intended23"] == clean_one["applied23"]
    assert clean_one["recovery_chunk_index"] == 0
    assert clean_one["recovery_chunk_step"] == 0
    assert clean_two["recovery_chunk_step"] == 1
    assert clean_one["prediction_contract"] == {
        "future_prediction_steps": 32,
        "execution_start": 0,
        "max_executed_per_replan_chunk": 16,
        "stored_clean_future_tail": False,
    }
    assert all("clean_intended32" not in record and "actions27" not in record for record in result.records)
    assert all(record["training_eligible"] is False for record in result.records)
    assert result.outcome_label == "SURVIVAL_NONFAILURE"


def test_collection_rejects_future_teacher_label_calibration_reuse_abort_and_overlength_chunk() -> None:
    with pytest.raises(RecoveryContractError, match="freshly queried"):
        TeacherReceipt(
            teacher_id="replayed-demo",
            teacher_kind="offline-action-array",
            label_source="planner",
            feedback_mode="precomputed_demo_replay",
            postcondition_spec_sha256=_digest("teacher-postcondition"),
            teacher_code_sha256=_digest("teacher-code"),
            teacher_weights_sha256=_digest("teacher-weights"),
            teacher_config_sha256=_digest("teacher-config"),
            runtime_code_sha256=_digest("runtime-code"),
        )
    with pytest.raises(RecoveryContractError, match="held-out calibration"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source("calibration-group"),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
        )
    with pytest.raises(RecoveryContractError, match="current reached policy clock"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(),
            teacher_callback=FakeTeacher(future_leak=True),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
        )
    with pytest.raises(RecoveryContractError, match="aborted"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(abort=True),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
        )
    with pytest.raises(RecoveryContractError, match="current observed clock/state/observation"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(receipt_clock_offset=1),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
        )
    with pytest.raises(RecoveryContractError, match="1..16"):
        collect_dart_inspired_actual_clean_recovery(
            runtime=FakeRuntime(),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=_bounded_noise(),
            intent_bundle_id="intent-v1",
            noisy_injection_steps=1,
            recovery_chunk_lengths=(17,),
            limits=CollectionLimits(max_total_steps=20, max_wall_seconds=60),
        )


def test_budget_end_stays_unknown_instead_of_synthesizing_outcome() -> None:
    values = iter((0.0, 0.0, 5.0))
    result = collect_dart_inspired_actual_clean_recovery(
        runtime=FakeRuntime(),
        teacher_callback=FakeTeacher(),
        teacher_receipt=_teacher_receipt(),
        source=_source(),
        runtime_session=_runtime_session(),
        calibration=_calibration(),
        noise=_bounded_noise(),
        intent_bundle_id="intent-v1",
        noisy_injection_steps=1,
        recovery_chunk_lengths=(1,),
        limits=CollectionLimits(max_total_steps=2, max_wall_seconds=1),
        clock=lambda: next(values),
    )
    assert len(result.records) == 1
    assert result.outcome_label == "OUTCOME_UNKNOWN"
    assert result.stop_reason == "collection_budget_exhausted"


def test_cli_writes_only_new_hashed_candidate_output() -> None:
    project_root = Path(__file__).resolve().parent.parent
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        request = root / "request.json"
        output = root / "candidate.json"
        request.write_text(json.dumps({"request_id": "cli-fixture"}), encoding="utf-8")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = f"{project_root / 'src'}:{project_root / 'tests'}"
        command = [
            sys.executable,
            str(project_root / "scripts/data/collect_dart_demonstrations.py"),
            "--request", str(request),
            "--factory", "test_dart_collection:build_cli_candidate",
            "--output", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True, env=environment)
        assert json.loads(completed.stdout)["status"] == "candidate_written"
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["authority_minted"] is False
        assert result["records"][0]["training_eligible"] is False
        assert result["collection_result_sha256"] == canonical_sha256(
            {key: value for key, value in result.items() if key != "collection_result_sha256"}
        )


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("dart collection contract tests passed")
