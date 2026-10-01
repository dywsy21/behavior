from __future__ import annotations

from dataclasses import replace
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
    CANDIDATE_ONLY_FLAGS,
    CalibrationReceipt,
    CanonicalSourceGroupMembership,
    CandidateSourceReceipt,
    CollectionLimits,
    DataSealedSourceGroupIndexReader,
    DartObservation,
    OutcomeEvidenceReceipt,
    DartCollectionResult,
    RuntimeSessionReceipt,
    OriginalGaussianCalibrationBinding,
    TeacherCommand,
    TeacherReceipt,
    VerifiedDartSourceMembership,
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


_CANDIDATE_GROUP = _digest("candidate-group")
_CALIBRATION_GROUP = _digest("calibration-group")
_DATA_PROTOCOL_DD060F3_SOURCE_SHA256 = "efdd20642fed24241f38bbdeb4abff6cf4faf1c72a86fe7c2acb32ba7496193b"


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


def _source(group: str = _CANDIDATE_GROUP) -> CandidateSourceReceipt:
    return CandidateSourceReceipt(
        source_group_id=group,
        original_split="train",
        source_release_sha256=_digest("source-release"),
        parent_task_id="task-42",
        parent_task_index=42,
        parent_task_instance_id=9,
        parent_task_seed=9,
        trajectory_source_kind="fresh_dart_trajectory",
        collection_run_id="fresh-session-17",
    )


def _calibration() -> CalibrationReceipt:
    return CalibrationReceipt(
        source_group_ids=(_CALIBRATION_GROUP,),
        calibration_trajectory_sha256=_digest("calibration-trajs"),
        learner_checkpoint_sha256=_digest("learner"),
        teacher_checkpoint_sha256=_digest("teacher-calibration"),
    )


class FakeSourceGroupIndexReader:
    """Test-only stand-in for the DATA-owned sealed membership reader."""

    manifest_sha256 = _digest("sealed-data-source-group-index")

    def __init__(self, *, spoof_role: str | None = None) -> None:
        self.spoof_role = spoof_role
        self.calls = 0

    def verify_dart_membership(
        self, source: CandidateSourceReceipt, calibration: CalibrationReceipt
    ) -> VerifiedDartSourceMembership:
        self.calls += 1
        candidate = CanonicalSourceGroupMembership(
            source_group_id=source.source_group_id,
            source_release_sha256=source.source_release_sha256,
            task_index=source.parent_task_index,
            task_instance_id=source.parent_task_instance_id,
            original_split=("eval" if self.spoof_role == "evaluation_only" else "protected")
            if self.spoof_role else source.original_split,
            usage_role=self.spoof_role or "student_candidate",
            source_group_member_sha256=_digest(f"member:{source.source_group_id}"),
        )
        calibration_groups = tuple(
            CanonicalSourceGroupMembership(
                source_group_id=group_id,
                source_release_sha256=source.source_release_sha256,
                task_index=source.parent_task_index,
                task_instance_id=source.parent_task_instance_id,
                original_split="train",
                usage_role="annotation_calibration",
                source_group_member_sha256=_digest(f"member:{group_id}"),
            )
            for group_id in calibration.source_group_ids
        )
        return VerifiedDartSourceMembership(
            index_inventory_seal_sha256=_digest("sealed-data-inventory"),
            source_group_index_manifest_sha256=self.manifest_sha256,
            source_membership_protocol_sha256=_digest("sealed-data-protocol"),
            candidate=candidate,
            calibration_groups=calibration_groups,
        )


def _source_membership_reader() -> FakeSourceGroupIndexReader:
    return FakeSourceGroupIndexReader()


def _original_calibration_binding() -> OriginalGaussianCalibrationBinding:
    # Eq.4: trace(diag(1)) is 23, so alpha=5.75 and T=1 yields diag(0.25),
    # exactly matching the Gaussian test sampler below.
    return OriginalGaussianCalibrationBinding(
        calibration_trajectory_sha256=_calibration().calibration_trajectory_sha256,
        learner_checkpoint_sha256=_calibration().learner_checkpoint_sha256,
        teacher_checkpoint_sha256=_calibration().teacher_checkpoint_sha256,
        covariance_estimator_code_sha256=_digest("dart-author-code-estimator"),
        estimated_covariance23=_diagonal(1.0),
        alpha=5.75,
        horizon=1,
        source_group_index_manifest_sha256=FakeSourceGroupIndexReader.manifest_sha256,
    )


def _runtime_session() -> RuntimeSessionReceipt:
    return RuntimeSessionReceipt(
        runtime_session_id="fresh-session-17",
        task_instance_id=9,
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
        ),
        embodiment_metadata_sha256=_digest("r1pro-embodiment-metadata"),
        model_projection_manifest_sha256=_digest("r1pro-23to27-projection"),
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
        "schema": "p107_dart_candidate_v2",
        "label_kind": "dart_clean_supervisor_feedback",
        "request_id": request["request_id"],
        "candidate_only": True,
        "training_eligible": False,
        "ready_for_training": False,
        "low_action_supervision_positive": False,
        "authority_minted": False,
        "authority_status": "NO_AUTHORITY",
    }
    record["candidate_sha256"] = canonical_sha256(record)
    return DartCollectionResult((record,), outcome_label="OUTCOME_UNKNOWN", stop_reason=None)


def build_cli_one_bad_candidate_flag(request: dict[str, object]) -> DartCollectionResult:
    result = build_cli_candidate(request)
    record = dict(result.records[0])
    field = request["bad_field"]
    if field not in CANDIDATE_ONLY_FLAGS:
        raise AssertionError("test requested a field outside the exact candidate flag contract")
    expected = CANDIDATE_ONLY_FLAGS[field]
    record[field] = not expected if isinstance(expected, bool) else "UNKNOWN"
    record["candidate_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "candidate_sha256"}
    )
    return DartCollectionResult((record,), outcome_label="OUTCOME_UNKNOWN", stop_reason=None)


def test_original_mode_records_clean_feedback_separately_from_noisy_applied_execution() -> None:
    runtime = FakeRuntime()
    teacher = FakeTeacher()
    source_membership_reader = _source_membership_reader()
    result = collect_original_gaussian_clean_feedback(
        runtime=runtime,
        teacher_callback=teacher,
        teacher_receipt=_teacher_receipt(),
        source=_source(),
        source_membership_reader=source_membership_reader,
        runtime_session=_runtime_session(),
        calibration=_calibration(),
        noise=OriginalGaussianNoise(_diagonal(0.25), seed=31),
        intent_bundle_id="intent-v1",
        steps=2,
        covariance_update_mode="frozen_one_pass_partial_dart",
        original_calibration_binding=_original_calibration_binding(),
    )

    assert teacher.observed_clocks == [0, 1]
    assert len(result.records) == 2
    first, second = result.records
    assert first["label_kind"] == "dart_clean_supervisor_feedback"
    assert first["clean_target_steps"] == 1
    assert first["clean_intended23"] != first["applied23"]
    assert first["requested_noisy23"] == first["applied23"]
    assert first["noise_profile"]["empirical_faithfulness"] is False
    assert first["verified_source_membership"]["source_group_index_manifest_sha256"] == (
        FakeSourceGroupIndexReader.manifest_sha256
    )
    assert source_membership_reader.calls == 1
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
        source_membership_reader=_source_membership_reader(),
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
    assert injection["noise_profile"]["embodiment_metadata_sha256"] == _digest("r1pro-embodiment-metadata")
    assert injection["noise_profile"]["model_projection_manifest_sha256"] == _digest("r1pro-23to27-projection")
    assert result.outcome_label == "SURVIVAL_NONFAILURE"


def test_collection_rejects_future_teacher_label_calibration_reuse_abort_and_overlength_chunk() -> None:
    with pytest.raises(RecoveryContractError, match="fresh_dart_trajectory"):
        CandidateSourceReceipt(
            **{**_source().public(), "trajectory_source_kind": "original_demo_replay"}
        )
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
            source=_source(_CALIBRATION_GROUP),
            source_membership_reader=_source_membership_reader(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
            original_calibration_binding=_original_calibration_binding(),
        )
    with pytest.raises(RecoveryContractError, match="current reached policy clock"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(),
            teacher_callback=FakeTeacher(future_leak=True),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            source_membership_reader=_source_membership_reader(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
            original_calibration_binding=_original_calibration_binding(),
        )
    with pytest.raises(RecoveryContractError, match="aborted"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(abort=True),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            source_membership_reader=_source_membership_reader(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
            original_calibration_binding=_original_calibration_binding(),
        )
    with pytest.raises(RecoveryContractError, match="current observed clock/state/observation"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(receipt_clock_offset=1),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            source_membership_reader=_source_membership_reader(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
            original_calibration_binding=_original_calibration_binding(),
        )
    with pytest.raises(RecoveryContractError, match="1..16"):
        collect_dart_inspired_actual_clean_recovery(
            runtime=FakeRuntime(),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            source_membership_reader=_source_membership_reader(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=_bounded_noise(),
            intent_bundle_id="intent-v1",
            noisy_injection_steps=1,
            recovery_chunk_lengths=(17,),
            limits=CollectionLimits(max_total_steps=20, max_wall_seconds=60),
        )
    with pytest.raises(RecoveryContractError, match="collection_run_id"):
        collect_dart_inspired_actual_clean_recovery(
            runtime=FakeRuntime(),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=replace(_source(), collection_run_id="different-live-run"),
            source_membership_reader=_source_membership_reader(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=_bounded_noise(),
            intent_bundle_id="intent-v1",
            noisy_injection_steps=1,
            recovery_chunk_lengths=(1,),
            limits=CollectionLimits(max_total_steps=2, max_wall_seconds=60),
        )


def test_budget_end_stays_unknown_instead_of_synthesizing_outcome() -> None:
    values = iter((0.0, 0.0, 5.0))
    result = collect_dart_inspired_actual_clean_recovery(
        runtime=FakeRuntime(),
        teacher_callback=FakeTeacher(),
        teacher_receipt=_teacher_receipt(),
        source=_source(),
        source_membership_reader=_source_membership_reader(),
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


def test_collection_rejects_spoofed_eval_protected_membership_and_covariance_binding() -> None:
    for role in ("evaluation_only", "protected_holdout"):
        with pytest.raises(RecoveryContractError, match="canonical sealed source-group membership"):
            collect_dart_inspired_actual_clean_recovery(
                runtime=FakeRuntime(),
                teacher_callback=FakeTeacher(),
                teacher_receipt=_teacher_receipt(),
                source=_source(),  # Caller still says train: sealed membership must win.
                source_membership_reader=FakeSourceGroupIndexReader(spoof_role=role),
                runtime_session=_runtime_session(),
                calibration=_calibration(),
                noise=_bounded_noise(),
                intent_bundle_id="intent-v1",
                noisy_injection_steps=1,
                recovery_chunk_lengths=(1,),
                limits=CollectionLimits(max_total_steps=2, max_wall_seconds=60),
            )
    mismatched = OriginalGaussianCalibrationBinding(
        calibration_trajectory_sha256=_calibration().calibration_trajectory_sha256,
        learner_checkpoint_sha256=_calibration().learner_checkpoint_sha256,
        teacher_checkpoint_sha256=_calibration().teacher_checkpoint_sha256,
        covariance_estimator_code_sha256=_digest("dart-author-code-estimator"),
        estimated_covariance23=_diagonal(1.0),
        alpha=1.0,
        horizon=1,
        source_group_index_manifest_sha256=FakeSourceGroupIndexReader.manifest_sha256,
    )
    with pytest.raises(RecoveryContractError, match="sampled covariance"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            source_membership_reader=_source_membership_reader(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="frozen_one_pass_partial_dart",
            original_calibration_binding=mismatched,
        )
    with pytest.raises(RecoveryContractError, match="only records frozen"):
        collect_original_gaussian_clean_feedback(
            runtime=FakeRuntime(),
            teacher_callback=FakeTeacher(),
            teacher_receipt=_teacher_receipt(),
            source=_source(),
            source_membership_reader=_source_membership_reader(),
            runtime_session=_runtime_session(),
            calibration=_calibration(),
            noise=OriginalGaussianNoise(_diagonal(0.25), seed=1),
            intent_bundle_id="intent-v1",
            steps=1,
            covariance_update_mode="full_iterative_covariance_update",
            original_calibration_binding=_original_calibration_binding(),
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _write_real_sealed_source_group_index(root: Path) -> tuple[str, str]:
    """Create the DATA protocol's real minimal source-group-only boundary.

    The intentionally absent event-candidates payload proves the DART reader
    uses only DATA's source-membership API, not a hidden event scan.
    """

    source_release = _source().source_release_sha256
    rows = [
        {
            "source_group_id": _CANDIDATE_GROUP,
            "source_release_manifest_sha256": source_release,
            "task_index": 42,
            "task_instance_id": 9,
            "original_split": "train",
            "usage_role": "student_candidate",
        },
        {
            "source_group_id": _CALIBRATION_GROUP,
            "source_release_manifest_sha256": source_release,
            "task_index": 42,
            "task_instance_id": 9,
            "original_split": "train",
            "usage_role": "annotation_calibration",
        },
    ]
    source_groups_bytes = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    source_groups_path = root / "source_groups.jsonl"
    source_groups_path.write_bytes(source_groups_bytes)
    empty_event_payload = b""
    files = {
        "source_groups.jsonl": {
            "sha256": sha256(source_groups_bytes).hexdigest(),
            "bytes": len(source_groups_bytes),
            "rows": len(rows),
        },
        # Do not create this file: source-group membership must not touch it.
        "event_candidates.jsonl": {
            "sha256": sha256(empty_event_payload).hexdigest(),
            "bytes": 0,
            "rows": 0,
        },
    }
    manifest = {
        "schema_version": "memlite-event-index-v1",
        "source_release_manifest_sha256": source_release,
        "files": files,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    seal = {
        "schema_version": "memlite-event-inventory-seal-v1",
        "index_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "source_release_manifest_sha256": source_release,
        "coverage_expectations_sha256": _digest("fixture-coverage"),
        "expected_payload_files": ["event_candidates.jsonl", "source_groups.jsonl"],
        "payload_files": files,
    }
    seal_bytes = _canonical_json_bytes(seal)
    (root / "inventory_seal.json").write_bytes(seal_bytes)
    return sha256(seal_bytes).hexdigest(), sha256(manifest_bytes).hexdigest()


def _dd060f3_protocol_bytes(project_root: Path) -> bytes:
    completed = subprocess.run(
        ["git", "show", "dd060f3:src/g05/data/memlite_event_protocol.py"],
        cwd=project_root,
        capture_output=True,
        check=True,
    )
    assert sha256(completed.stdout).hexdigest() == _DATA_PROTOCOL_DD060F3_SOURCE_SHA256
    return completed.stdout


def test_data_source_adapter_uses_real_sealed_index_without_importing_g05_data() -> None:
    """Run DATA's committed API under ``-S``: no fake module or ML dependency."""

    project_root = Path(__file__).resolve().parent.parent
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        index_root = root / "index"
        index_root.mkdir()
        inventory_seal_sha256, manifest_sha256 = _write_real_sealed_source_group_index(index_root)
        protocol_path = root / "memlite_event_protocol.py"
        protocol_bytes = _dd060f3_protocol_bytes(project_root)
        protocol_path.write_bytes(protocol_bytes)
        protocol_sha256 = sha256(protocol_bytes).hexdigest()
        child = f"""
import json
import sys
from pathlib import Path
from g05.recovery.dart_collection import CalibrationReceipt, CandidateSourceReceipt, DataSealedSourceGroupIndexReader
source = CandidateSourceReceipt(
    source_group_id={_CANDIDATE_GROUP!r}, original_split='train', source_release_sha256={_source().source_release_sha256!r},
    parent_task_id='task-42', parent_task_index=42, parent_task_instance_id=9, parent_task_seed=9,
    trajectory_source_kind='fresh_dart_trajectory', collection_run_id='run-fixture')
calibration = CalibrationReceipt(source_group_ids=({_CALIBRATION_GROUP!r},), calibration_trajectory_sha256={_calibration().calibration_trajectory_sha256!r}, learner_checkpoint_sha256={_calibration().learner_checkpoint_sha256!r}, teacher_checkpoint_sha256={_calibration().teacher_checkpoint_sha256!r})
reader = DataSealedSourceGroupIndexReader(Path({str(index_root)!r}), expected_inventory_seal_sha256={inventory_seal_sha256!r}, protocol_path=Path({str(protocol_path)!r}), expected_protocol_source_sha256={protocol_sha256!r})
# Loading happened at construction. Deleting the only group payload makes a
# second filesystem scan fail, so two successful lookups prove cached use.
Path({str(index_root / 'source_groups.jsonl')!r}).unlink()
first = reader.verify_dart_membership(source, calibration)
second = reader.verify_dart_membership(source, calibration)
assert 'g05.data' not in sys.modules
print(json.dumps({{'candidate_role': first.candidate.usage_role, 'calibration_role': second.calibration_groups[0].usage_role, 'inventory': first.index_inventory_seal_sha256, 'manifest': first.source_group_index_manifest_sha256, 'protocol': first.source_membership_protocol_sha256}}))
"""
        environment = {"PATH": os.environ["PATH"], "PYTHONPATH": str(project_root / "src")}
        completed = subprocess.run(
            [sys.executable, "-S", "-c", child], capture_output=True, text=True, check=True, env=environment
        )
        result = json.loads(completed.stdout)
        assert result == {
            "candidate_role": "student_candidate",
            "calibration_role": "annotation_calibration",
            "inventory": inventory_seal_sha256,
            "manifest": manifest_sha256,
            "protocol": protocol_sha256,
        }
        with pytest.raises(RecoveryContractError, match="requires expected_protocol_source_sha256"):
            DataSealedSourceGroupIndexReader(
                index_root,
                expected_inventory_seal_sha256=inventory_seal_sha256,
                protocol_path=protocol_path,
            )
        with pytest.raises(RecoveryContractError, match="do not match"):
            DataSealedSourceGroupIndexReader(
                index_root,
                expected_inventory_seal_sha256=inventory_seal_sha256,
                protocol_path=protocol_path,
                expected_protocol_source_sha256=_digest("wrong-protocol-bytes"),
            )
        missing_api_path = root / "old_protocol_without_membership_api.py"
        missing_api_bytes = b"SOURCE_PROTOCOL_VERSION = 'old'\n"
        missing_api_path.write_bytes(missing_api_bytes)
        with pytest.raises(RecoveryContractError, match="does not expose"):
            DataSealedSourceGroupIndexReader(
                index_root,
                expected_inventory_seal_sha256=inventory_seal_sha256,
                protocol_path=missing_api_path,
                expected_protocol_source_sha256=sha256(missing_api_bytes).hexdigest(),
            )


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
        assert result["authority_status"] == "NO_AUTHORITY"
        assert result["records"][0]["training_eligible"] is False
        assert result["collection_result_sha256"] == canonical_sha256(
            {key: value for key, value in result.items() if key != "collection_result_sha256"}
        )


def test_cli_rejects_hash_valid_records_with_any_non_candidate_authority_flag() -> None:
    project_root = Path(__file__).resolve().parent.parent
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = f"{project_root / 'src'}:{project_root / 'tests'}"
        for field, expected in CANDIDATE_ONLY_FLAGS.items():
            request = root / f"request-{field}.json"
            output = root / f"candidate-{field}.json"
            request.write_text(
                json.dumps({"request_id": f"cli-bad-{field}", "bad_field": field}), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts/data/collect_dart_demonstrations.py"),
                    "--request", str(request),
                    "--factory", "test_dart_collection:build_cli_one_bad_candidate_flag",
                    "--output", str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            assert completed.returncode == 2
            assert f"{field}={expected!r}" in completed.stderr
            assert not output.exists()


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("dart collection contract tests passed")
