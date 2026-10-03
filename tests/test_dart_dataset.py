from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import struct

import pytest
from PIL import Image

from g05.recovery.common import RecoveryContractError, canonical_json
from g05.recovery.dart_collection import (
    AppliedActionReceipt,
    CanonicalSourceGroupMembership,
    CandidateSourceReceipt,
    DartObservation,
    native_raw23_float32_sha256,
    RuntimeSessionReceipt,
    TeacherCommand,
    TeacherReceipt,
    VerifiedDartSourceMembership,
)
from g05.recovery.dart_dataset import (
    ActorObservation,
    COLLECTION_MODE_BOUNDED_ACTUAL,
    COLLECTION_MODE_ORIGINAL_GAUSSIAN,
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
        proprioception_schema_sha256=_digest("actor-schema"),
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


def _step(index: int, *, values: tuple[int, int] = (30, 50), noisy: bool = False) -> DartEpisodeStep:
    pre_clock = index
    post_clock = index + 1
    pre = _observation(pre_clock)
    post = _observation(post_clock)
    clean = _action(float(index + 1))
    sampled = _action(0.03125) if noisy else _action(0.0)
    requested_value = float(index + 1) + (0.03125 if noisy else 0.0)
    requested = _action(requested_value)
    native_request_sha = native_raw23_float32_sha256(requested)
    applied_action = requested if noisy else clean
    applied = AppliedActionReceipt(
        status="APPLIED",
        applied23=applied_action,
        applied_action_bytes_sha256=native_request_sha,
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
        sampled_noise23=sampled,
        requested_native23=requested,
        requested_native_action_bytes_sha256=native_request_sha,
        applied=applied,
        post_observation=post,
        post_actor_observation=_actor(post_clock, values[1] + index),
        clock=StepClock(simulation_tick_start=10 + index, simulation_tick_end=11 + index, action_start_time_s=float(index), action_end_time_s=float(index) + 0.1),
        label_kind=("dart_inspired_noisy_injection" if noisy else "dart_inspired_actual_clean_recovery"),
    )


def _bundle(*, collection_mode: str = COLLECTION_MODE_BOUNDED_ACTUAL, steps: tuple[DartEpisodeStep, ...] | None = None) -> DartEpisodeBundle:
    source = _source()
    if steps is None:
        steps = (_step(0), _step(1))
        if collection_mode == COLLECTION_MODE_ORIGINAL_GAUSSIAN:
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
            if collection_mode == COLLECTION_MODE_ORIGINAL_GAUSSIAN
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
    assert projection.public_causal_instruction is None
    assert set(projection.rgb_by_view) == {"head", "left_wrist", "right_wrist"}
    assert loaded.manifest["outcome_evidence_separate"] is True
    assert '"state61":' not in (output / "records.jsonl").read_text()
    assert "post_observation" not in projection.__dict__
    assert loaded.inspect()["run_origin"] == "cpu_fixture"
    assert loaded.inspect()["training_eligible"] is False


def test_actual_bounded_collector_mode_is_preserved_verbatim_and_unknown_upgrade_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "bounded"
    receipt = write_episode_bundle(output, _bundle(collection_mode=COLLECTION_MODE_BOUNDED_ACTUAL))
    loaded = load_bundle(output, expected_manifest_sha256=receipt.manifest_sha256)
    assert loaded.manifest["collection_mode"] == "dart_inspired_bounded_actual_clean_recovery"
    with pytest.raises(RecoveryContractError, match="existing DART mode"):
        replace(_bundle(), collection_mode="dart_inspired_actual_clean_recovery")
    with pytest.raises(RecoveryContractError, match="existing DART mode"):
        replace(_bundle(), collection_mode="dart_inspired_bounded_actual_recovery")


def test_original_gaussian_collection_shape_preserves_clean_plus_noise(tmp_path: Path) -> None:
    sampled = replace(_step(0, noisy=True), label_kind="dart_clean_supervisor_feedback")
    bundle = _bundle(collection_mode=COLLECTION_MODE_ORIGINAL_GAUSSIAN, steps=(sampled,))
    receipt = write_episode_bundle(tmp_path / "gaussian", bundle)
    loaded = load_bundle(tmp_path / "gaussian", expected_manifest_sha256=receipt.manifest_sha256)
    row = loaded.rows[0]
    assert row["requested_noisy23"] == [1.03125] * 23
    assert row["sampled_noise23"] == [0.03125] * 23
    assert row["native_action_wire_dtype"] == "float32_le"
    assert row["requested_native23"] == row["requested_noisy23"]
    assert row["applied"]["applied23"] == row["requested_noisy23"]


def test_native_float32_request_is_distinct_from_math_request_but_valid() -> None:
    step = _step(0)
    clean = _action(0.1)
    sampled = _action(0.2)
    requested = _action(0.1 + 0.2)
    native_value = struct.unpack(">f", struct.pack(">f", requested[0]))[0]
    native = _action(native_value)
    native_sha = native_raw23_float32_sha256(native)
    command = replace(step.teacher_command, clean_intended23=clean)
    applied = replace(
        step.applied,
        applied23=native,
        applied_action_bytes_sha256=native_sha,
    )
    valid = replace(
        step,
        teacher_command=command,
        requested_noisy23=requested,
        sampled_noise23=sampled,
        requested_native23=native,
        requested_native_action_bytes_sha256=native_sha,
        applied=applied,
        label_kind="dart_clean_supervisor_feedback",
    )
    assert valid.requested_noisy23[0] != valid.requested_native23[0]
    with pytest.raises(RecoveryContractError, match="canonical float32"):
        replace(valid, requested_native23=requested)


def test_private_intent_id_stays_metadata_only_and_explicit_public_instruction_roundtrips(tmp_path: Path) -> None:
    output = tmp_path / "public-intent"
    bundle = replace(_bundle(), public_causal_instruction="pick up the declared target")
    receipt = write_episode_bundle(output, bundle)
    loaded = load_bundle(output, expected_manifest_sha256=receipt.manifest_sha256)
    projection = loaded.actor_projection(0)
    assert projection.public_causal_instruction == "pick up the declared target"
    row = loaded.rows[0]
    assert "intent_bundle_id" not in row["actor_projection"]
    assert loaded.manifest["intent_bundle_id"] == "intent-v1"


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


def test_collector_action_semantics_reject_malformed_noise_and_phase_labels() -> None:
    noisy = _step(0, noisy=True)
    with pytest.raises(RecoveryContractError, match="clean_intended23 plus sampled_noise23"):
        replace(noisy, sampled_noise23=_action(0.03126))
    with pytest.raises(RecoveryContractError, match="applied23 must equal"):
        replace(
            noisy,
            applied=replace(
                noisy.applied,
                applied23=_action(1.0),
                applied_action_bytes_sha256=native_raw23_float32_sha256(_action(1.0)),
            ),
        )
    with pytest.raises(RecoveryContractError, match="requested_native_action_bytes_sha256"):
        replace(noisy, requested_native_action_bytes_sha256=_digest("wrong-request-wire"))
    with pytest.raises(RecoveryContractError, match="applied_action_bytes_sha256"):
        replace(noisy, applied=replace(noisy.applied, applied_action_bytes_sha256=_digest("wrong-applied-wire")))
    changed_applied = _action(1.0)
    with pytest.raises(RecoveryContractError, match="applied23 must equal"):
        replace(
            noisy,
            applied=replace(
                noisy.applied,
                applied23=changed_applied,
                applied_action_bytes_sha256=native_raw23_float32_sha256(changed_applied),
            ),
        )
    clean = _step(0)
    with pytest.raises(RecoveryContractError, match="applied23 must equal"):
        replace(
            clean,
            applied=replace(
                clean.applied,
                applied23=_action(1.03125),
                applied_action_bytes_sha256=native_raw23_float32_sha256(_action(1.03125)),
            ),
        )
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
            collection_mode=COLLECTION_MODE_BOUNDED_ACTUAL,
            label_kind="dart_inspired_actual_clean_recovery",
            run_provenance=_run(),
            steps=(_step(0),),
        )
    mismatched_actor = replace(_step(0).pre_actor_observation, proprioception_schema_sha256=_digest("other-actor-schema"))
    with pytest.raises(RecoveryContractError, match="schema pin"):
        replace(_bundle(), steps=(replace(_step(0), pre_actor_observation=mismatched_actor),))


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
        applied_action_bytes_sha256=native_raw23_float32_sha256(_action(99.0)),
    )
    with pytest.raises(RecoveryContractError, match="applied23 must equal"):
        replace(step, applied=clipped, label_kind="dart_clean_supervisor_feedback")


def test_bounded_episode_preserves_mixed_injection_and_recovery_kinds(tmp_path: Path) -> None:
    mixed = _step(1, noisy=True)
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
