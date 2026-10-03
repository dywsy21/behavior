from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
from PIL import Image

from g05.recovery.common import RecoveryContractError, canonical_json
from g05.recovery.dart_collection import (
    AppliedActionReceipt,
    CanonicalSourceGroupMembership,
    CandidateSourceReceipt,
    DartObservation,
    RuntimeSessionReceipt,
    TeacherCommand,
    TeacherReceipt,
    VerifiedDartSourceMembership,
)
from g05.recovery.dart_dataset import (
    ActorObservation,
    DartEpisodeBundle,
    DartEpisodeStep,
    RGBAsset,
    RunProvenance,
    StepClock,
    load_bundle,
    write_episode_bundle,
)
from g05.recovery.dart_collection import OutcomeEvidenceReceipt


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _action(value: float) -> tuple[float, ...]:
    return (value,) * 23


def _png(value: int) -> bytes:
    image = Image.new("RGB", (10, 8), (value, 20, 30))
    from io import BytesIO

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _asset(view: str, clock: int, value: int) -> RGBAsset:
    return RGBAsset(
        view=view,
        camera_key=f"observation.rgb.{view}",
        payload=_png(value),
        width=10,
        height=8,
        policy_clock=clock,
        source_locator=f"fixture://episode/step-{clock}/{view}",
        capture_time_s=float(clock) / 30.0,
        pts=float(clock),
    )


def _actor(clock: int, value: int) -> ActorObservation:
    return ActorObservation(
        policy_clock=clock,
        assets={view: _asset(view, clock, value + index) for index, view in enumerate(("head", "left_wrist", "right_wrist"))},
        proprioception=(float(clock), 0.25, -0.5),
    )


def _observation(clock: int) -> DartObservation:
    return DartObservation(
        policy_clock=clock,
        state61=(float(clock),) * 61,
        observation_sha256=_digest(f"rgb-proprio-observation-{clock}"),
        observation_ref=f"fixture://observation/{clock}",
    )


def _source() -> CandidateSourceReceipt:
    return CandidateSourceReceipt(
        source_group_id=_digest("candidate-group"),
        original_split="train",
        source_release_sha256=_digest("source-release"),
        parent_task_id="task-18",
        parent_task_index=18,
        parent_task_instance_id=7,
        parent_task_seed=11,
        trajectory_source_kind="fresh_dart_trajectory",
        collection_run_id="dart-run-7",
    )


def _membership(source: CandidateSourceReceipt) -> VerifiedDartSourceMembership:
    candidate = CanonicalSourceGroupMembership(
        source_group_id=source.source_group_id,
        source_release_sha256=source.source_release_sha256,
        task_index=source.parent_task_index,
        task_instance_id=source.parent_task_instance_id,
        original_split="train",
        usage_role="student_candidate",
        source_group_member_sha256=_digest("candidate-member"),
    )
    calibration = CanonicalSourceGroupMembership(
        source_group_id=_digest("calibration-group"),
        source_release_sha256=source.source_release_sha256,
        task_index=source.parent_task_index,
        task_instance_id=source.parent_task_instance_id,
        original_split="train",
        usage_role="annotation_calibration",
        source_group_member_sha256=_digest("calibration-member"),
    )
    return VerifiedDartSourceMembership(
        index_inventory_seal_sha256=_digest("inventory"),
        source_group_index_manifest_sha256=_digest("group-index"),
        source_membership_protocol_sha256=_digest("membership-protocol"),
        candidate=candidate,
        calibration_groups=(calibration,),
    )


def _runtime() -> RuntimeSessionReceipt:
    return RuntimeSessionReceipt(
        runtime_session_id="dart-run-7",
        task_instance_id=7,
        runtime_build_sha256=_digest("runtime-build"),
        asset_config_sha256=_digest("asset-config"),
        reset_load_task_instance_receipt_sha256=_digest("fresh-reset"),
    )


def _teacher() -> TeacherReceipt:
    return TeacherReceipt(
        teacher_id="fixture-teacher",
        teacher_kind="planner",
        label_source="planner",
        feedback_mode="closed_loop_fresh_observation",
        postcondition_spec_sha256=_digest("postcondition"),
        teacher_code_sha256=_digest("teacher-code"),
        teacher_weights_sha256=_digest("teacher-weights"),
        teacher_config_sha256=_digest("teacher-config"),
        runtime_code_sha256=_digest("runtime-code"),
    )


def _run() -> RunProvenance:
    return RunProvenance(
        run_id="fixture-run",
        source_episode_id="episode-7",
        run_origin="cpu_fixture",
        collector_code_sha256=_digest("collector-code"),
        capture_adapter_sha256=_digest("capture-adapter"),
        actor_observation_schema_sha256=_digest("actor-schema"),
    )


def _step(index: int, *, values: tuple[int, int] = (30, 50)) -> DartEpisodeStep:
    pre_clock = index
    post_clock = index + 1
    pre = _observation(pre_clock)
    post = _observation(post_clock)
    clean = _action(float(index + 1))
    requested = _action(float(index + 2))
    applied = AppliedActionReceipt(
        status="APPLIED",
        applied23=requested,
        applied_action_bytes_sha256=_digest(f"applied-bytes-{index}"),
        runtime_step_receipt_sha256=_digest(f"runtime-step-{index}"),
        pre_action_policy_clock=pre_clock,
        pre_action_state61_sha256=pre.state61_sha256,
        pre_action_observation_sha256=pre.observation_sha256,
    )
    command = TeacherCommand(
        clean_intended23=clean,
        observed_policy_clock=pre_clock,
        observed_state61_sha256=pre.state61_sha256,
        observed_observation_sha256=pre.observation_sha256,
        intent_bundle_id="intent-v1",
        fresh_query_receipt_sha256=_digest(f"query-{index}"),
    )
    return DartEpisodeStep(
        step_index=index,
        pre_observation=pre,
        pre_actor_observation=_actor(pre_clock, values[0] + index),
        teacher_command=command,
        requested_noisy23=requested,
        sampled_noise23=_action(0.1 + index),
        applied=applied,
        post_observation=post,
        post_actor_observation=_actor(post_clock, values[1] + index),
        clock=StepClock(simulation_tick_start=10 + index, simulation_tick_end=11 + index, action_start_time_s=float(index), action_end_time_s=float(index) + 0.1),
    )


def _bundle(*, collection_mode: str = "dart_inspired_actual_clean_recovery", steps: tuple[DartEpisodeStep, ...] | None = None) -> DartEpisodeBundle:
    source = _source()
    if steps is None:
        steps = (_step(0), _step(1))
        if collection_mode == "original_gaussian_clean_intended_feedback":
            steps = tuple(replace(step, label_kind="dart_clean_supervisor_feedback") for step in steps)
    return DartEpisodeBundle(
        source=source,
        verified_membership=_membership(source),
        runtime_session=_runtime(),
        teacher=_teacher(),
        intent_bundle_id="intent-v1",
        collection_mode=collection_mode,
        label_kind=(
            "dart_clean_supervisor_feedback"
            if collection_mode == "original_gaussian_clean_intended_feedback"
            else "dart_inspired_actual_clean_recovery"
        ),
        run_provenance=_run(),
        steps=steps,
        outcome_evidence=OutcomeEvidenceReceipt(
            label="OUTCOME_UNKNOWN",
            observed_policy_clock=1,
            available_policy_clock=2,
            evidence_sha256=_digest("unknown-outcome-evidence"),
        ),
    )


def test_bundle_roundtrips_distinct_step_images_and_keeps_outcome_private(tmp_path: Path) -> None:
    output = tmp_path / "episode"
    receipt = write_episode_bundle(output, _bundle())
    loaded = load_bundle(output, expected_manifest_sha256=receipt.manifest_sha256)

    assert receipt.transition_count == 2
    assert receipt.image_count == 12
    assert loaded.read_asset(0, "pre", "head") != loaded.read_asset(1, "pre", "head")
    assert loaded.read_asset(0, "pre", "left_wrist") != loaded.read_asset(0, "pre", "right_wrist")
    projection = loaded.actor_projection(0)
    assert projection.policy_clock == 0
    assert projection.clean_intended23 == _action(1.0)
    assert set(projection.rgb_by_view) == {"head", "left_wrist", "right_wrist"}
    assert loaded.manifest["outcome_evidence_separate"] is True
    assert '"state61":' not in (output / "records.jsonl").read_text()
    assert "post_observation" not in projection.__dict__
    assert loaded.inspect()["run_origin"] == "cpu_fixture"
    assert loaded.inspect()["training_eligible"] is False


def test_array_capture_is_encoded_and_rejected_if_dimensions_are_wrong() -> None:
    asset = RGBAsset.from_array(
        view="head",
        camera_key="observation.rgb.zed_link_camera_0",
        array=Image.new("RGB", (4, 3), (1, 2, 3)),
        policy_clock=4,
        source_locator="fixture://array",
    )
    assert asset.width == 4 and asset.height == 3
    with pytest.raises(RecoveryContractError, match="width"):
        RGBAsset(
            view="head",
            camera_key="observation.rgb.zed_link_camera_0",
            payload=asset.payload,
            width=3,
            height=3,
            policy_clock=4,
            source_locator="fixture://bad-dimensions",
        )


def test_wrong_clock_action_dimensions_and_camera_mixup_fail_closed() -> None:
    with pytest.raises(RecoveryContractError, match="23-D"):
        replace(_step(0), requested_noisy23=(0.0,) * 22)
    with pytest.raises(RecoveryContractError, match="simulation_tick_end"):
        StepClock(simulation_tick_start=4, simulation_tick_end=4)
    assets = {view: _asset(view, 0, 1) for view in ("head", "left_wrist", "right_wrist")}
    assets["left_wrist"] = _asset("head", 0, 2)
    with pytest.raises(RecoveryContractError, match="view"):
        ActorObservation(policy_clock=0, assets=assets)
    assets = {view: _asset(view, 0, 1) for view in ("head", "left_wrist", "right_wrist")}
    assets["right_wrist"] = replace(assets["right_wrist"], camera_key=assets["head"].camera_key)
    with pytest.raises(RecoveryContractError, match="camera_key"):
        ActorObservation(policy_clock=0, assets=assets)


def test_privileged_projection_and_source_identity_are_rejected() -> None:
    with pytest.raises(RecoveryContractError, match="proprioception"):
        ActorObservation(
            policy_clock=0,
            assets={view: _asset(view, 0, 1) for view in ("head", "left_wrist", "right_wrist")},
            proprioception={"target_pose": [1, 2, 3]},
        )
    source = _source()
    bad_candidate = replace(
        _membership(source).candidate,
        source_group_id=_digest("different-group"),
    )
    bad_membership = replace(_membership(source), candidate=bad_candidate)
    with pytest.raises(RecoveryContractError, match="sealed TRAIN"):
        DartEpisodeBundle(
            source=source,
            verified_membership=bad_membership,
            runtime_session=_runtime(),
            teacher=_teacher(),
            intent_bundle_id="intent-v1",
            collection_mode="dart_inspired_actual_clean_recovery",
            label_kind="dart_inspired_actual_clean_recovery",
            run_provenance=_run(),
            steps=(_step(0),),
        )


def test_cpu_fixture_cannot_claim_private_outcome() -> None:
    evidence = OutcomeEvidenceReceipt(
        label="SURVIVAL_NONFAILURE",
        observed_policy_clock=1,
        available_policy_clock=2,
        evidence_sha256=_digest("fixture-claimed-survival"),
    )
    with pytest.raises(RecoveryContractError, match="CPU fixtures"):
        replace(_bundle(), outcome_evidence=evidence)


def test_output_is_fresh_only_and_tampered_png_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "episode"
    receipt = write_episode_bundle(output, _bundle())
    with pytest.raises(RecoveryContractError, match="overwrite"):
        write_episode_bundle(output, _bundle())
    png = output / "assets/step-000000/pre-head.png"
    png.write_bytes(png.read_bytes() + b"tampered")
    with pytest.raises(RecoveryContractError, match="bytes/SHA"):
        load_bundle(output, expected_manifest_sha256=receipt.manifest_sha256)


def test_original_gaussian_mode_rejects_runtime_clipping() -> None:
    step = _step(0)
    clipped = replace(
        step.applied,
        applied23=_action(99.0),
    )
    clipped_step = replace(step, applied=clipped, label_kind="dart_clean_supervisor_feedback")
    with pytest.raises(RecoveryContractError, match="runtime clipping"):
        _bundle(
            collection_mode="original_gaussian_clean_intended_feedback",
            steps=(clipped_step,),
        )


def test_bounded_episode_preserves_mixed_injection_and_recovery_kinds(tmp_path: Path) -> None:
    mixed = replace(_bundle().steps[1], label_kind="dart_inspired_noisy_injection")
    bundle = replace(_bundle(), steps=(_step(0), mixed), label_kind="mixed")
    receipt = write_episode_bundle(tmp_path / "mixed", bundle)
    loaded = load_bundle(tmp_path / "mixed", expected_manifest_sha256=receipt.manifest_sha256)
    assert [row["label_kind"] for row in loaded.rows] == [
        "dart_inspired_actual_clean_recovery",
        "dart_inspired_noisy_injection",
    ]


def test_manifest_actor_projection_is_not_allowed_to_gain_future_fields(tmp_path: Path) -> None:
    output = tmp_path / "episode"
    write_episode_bundle(output, _bundle())
    records = output / "records.jsonl"
    row_lines = records.read_text().splitlines()
    row = json.loads(row_lines[0])
    row["actor_projection"]["post_observation"] = {"policy_clock": 1}
    records.write_text(
        canonical_json(row)
        + "\n"
        + "\n".join(row_lines[1:])
        + "\n"
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records_bytes = records.read_bytes()
    manifest["files"]["records.jsonl"] = {
        "sha256": sha256(records_bytes).hexdigest(),
        "bytes": len(records_bytes),
    }
    manifest_path.write_bytes((canonical_json(manifest) + "\n").encode())
    with pytest.raises(RecoveryContractError, match="forbidden actor field"):
        load_bundle(output)
