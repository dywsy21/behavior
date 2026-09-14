"""One lower-tail LR curve, isolated from EMA/capacity/loss changes."""
from copy import deepcopy
from pathlib import Path

BASE = Path("/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913")
RUN = "fm_tail_lr_v1"
EMA_RUN = "fm_trainable_ema_v1"
EMA_SPEC_SHA = "d195fd580899e5d947df5daa8d768d6be44315ca61a7c441beab159e2d8fd2b7"
TAIL_LR = 1e-5 * 0.1  # Match the actual saved A4 scheduler endpoint.
RECIPE = dict(mode="constant_parent_tail", learning_rate=TAIL_LR,
    warmup_steps=0, lr_min_ratio=1.0, scheduler="cosine",
    changes_only="learning_rate_curve", restores_parent_optimizer=False,
    additional_ema=False, control_run="fm_action_control_v3",
    control_checkpoint_sha256="efce4dfe232f85ac18f7fca66b562360ba23d839f3f74748f658b5911b5c33f9")


def training_recipe(original, selected):
    result = deepcopy(original)
    if selected is not None:
        if selected != RECIPE:
            raise RuntimeError("Only the declared constant-parent-tail LR curve is allowed")
        result.update({k: selected[k] for k in ("learning_rate", "warmup_steps", "lr_min_ratio")})
    return result


def validate_variant(route, initialization, conditioning, marker_rows, selected):
    if selected is not None and (selected != RECIPE
            or (route, initialization, conditioning) != ("fm", "a4", "skills") or marker_rows):
        raise RuntimeError("Tail LR is a standalone original-FM/A4/skills screen, not an AR/marker combination")


def validate_screen(spec):
    selected = spec.get("tail_lr_recipe")
    if selected is None:
        if Path(spec["output"]).name == RUN:
            raise RuntimeError("The tail-LR run cannot silently execute the old restarted schedule")
        return
    validate_variant(spec["route"], spec["initialization"], spec["conditioning"],
                     spec.get("marker_rows"), selected)
    dependency = spec.get("after_action") or {}
    if (Path(spec["output"]) != BASE / RUN or spec.get("marker_rows") is not False
            or spec.get("ema_recipe") is not None or spec.get("after_fm")
            or spec.get("recovery_from") or spec.get("reference_gate_recovery")
            or dependency.get("kind") != "action" or dependency.get("root") != str(BASE / EMA_RUN)
            or dependency.get("method_sha256") != EMA_SPEC_SHA):
        raise RuntimeError("Tail LR requires its own finite run after the exact already queued EMA run")


def validate_ema_completion(inspection):
    evidence = inspection.get("ema") or {}
    if (inspection.get("passed") is not True or inspection.get("actual_updates") != 500
            or evidence.get("num_updates") != 500 or evidence.get("parameter_tensors") != 514
            or evidence.get("full_state_roundtrip") is not True
            or evidence.get("online_restoration_checks") != 5):
        raise RuntimeError("Queued EMA has not completed its original full-shadow/clock/restore verification")
