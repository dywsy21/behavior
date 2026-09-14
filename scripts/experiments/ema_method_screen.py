"""Strict declaration of one EMA screen, not a method-combination search."""
from pathlib import Path

BASE = Path("/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913")
RUN = "fm_trainable_ema_v1"
COT_RUN = "ar_native_subtask_cot_fulltrain_v1"
COT_SPEC_SHA = "3dffcd78de30bff9d19c3e2b1658433bda29e0a215d4b48bf7630bf3d2d37863"
RECIPE = dict(beta=0.99, update_every=1, initialization="copy_parent_before_update_1",
    scope="all_trainable_action_expert_and_vlm_lora", expected_parameter_tensors=514,
    master_dtypes=["torch.float32"], per_process_gpu_memory_fraction=0.55,
    paired_online_evaluation=True, frozen_buffers_shared=True,
    copy_average_back_to_training=False, separate_ema_eval_every=100,
    control_run="fm_action_control_v3",
    control_checkpoint_sha256="efce4dfe232f85ac18f7fca66b562360ba23d839f3f74748f658b5911b5c33f9")


def validate_screen(spec):
    selected = spec.get("ema_recipe")
    if selected is None:
        if Path(spec["output"]).name == RUN:
            raise RuntimeError("The EMA run cannot silently execute a non-EMA control")
        return
    dependency = spec.get("after_action") or {}
    if (selected != RECIPE or Path(spec["output"]) != BASE / RUN
            or (spec["route"], spec["initialization"], spec["conditioning"]) != ("fm", "a4", "skills")
            or spec.get("marker_rows") is not False or spec.get("after_fm")
            or spec.get("recovery_from") or spec.get("reference_gate_recovery")
            or dependency.get("kind") != "action" or dependency.get("root") != str(BASE / COT_RUN)
            or dependency.get("method_sha256") != COT_SPEC_SHA):
        raise RuntimeError("EMA requires one isolated original-FM screen after the exact existing CoT run")


def validate_smoke(inspection):
    evidence = inspection.get("ema") or {}
    if (inspection.get("passed") is not True or inspection.get("actual_updates") != 5
            or evidence.get("num_updates") != 5 or evidence.get("parameter_tensors") != 514
            or evidence.get("full_state_roundtrip") is not True
            or evidence.get("online_restoration_checks") != 1):
        raise RuntimeError("Formal EMA requires its own five-step full-shadow/clock/online-restore gate")
