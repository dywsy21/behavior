"""Explicit H09Y reference budget; absence preserves the old H09U contract."""
from pathlib import Path

PROFILE = "h09y-train114-reference-v1"
PURPOSE = "TRAIN_POSE_SEED_REFERENCE_NOT_NATIVE_BC"
ROOT = Path("/mnt/nvme_tmp/robodojo_vlm_sft_20260919/h09y_grasp_only")
SOURCE = [1, 264, 114]
HELDOUT = [[1, 1], [1, 71]]
BUDGET = {"max_controls": 1199, "tail_controls": 12, "final_hold_controls": 1,
          "seconds_after_reset": 1200, "initialization_seconds": 900,
          "run_MiB": 80, "total_MiB": 6144}


def validate_profile(release, manifest, row):
    profile = release.get("reference_profile")
    if profile is None:
        if "reference_profile" in manifest or "purpose" in manifest:
            raise ValueError("New preparation cannot use an old reference authorization")
        return None
    if profile != PROFILE or manifest.get("reference_profile") != PROFILE:
        raise ValueError("Unknown/mixed reference profile")
    if (release.get("purpose") != PURPOSE or manifest.get("purpose") != PURPOSE or
            release.get("source") != SOURCE or [row[k] for k in ("task", "episode", "instance")] != SOURCE or
            row.get("start") != 596 or row.get("end") != 1186 or row.get("verb") != "GRASP" or
            row.get("hand") != "right" or row.get("support_hand") is not None or row.get("payloads") != [] or
            release.get("held_out_instance_groups") != HELDOUT or manifest.get("held_out_instance_groups") != HELDOUT or
            release.get("physical_gpu") != 3 or type(release.get("physical_gpu")) is not int or
            Path(release.get("experiment_root", "")).resolve() != ROOT or
            release.get("resets") != 1 or type(release.get("resets")) is not int or
            release.get("model_calls") != 0 or type(release.get("model_calls")) is not int):
        raise ValueError("Exact TRAIN114 purpose/partition/GPU/root/reset profile required")
    budget = release.get("budget")
    if (not isinstance(budget, dict) or set(budget) != set(BUDGET) or
            any(type(budget[k]) is not int or budget[k] != v for k, v in BUDGET.items())):
        raise ValueError("Exact explicit reference budget required")
    return PROFILE


def validate_execution_location(release, output, gpu):
    """Checked before output creation and evaluator initialization."""
    if release.get("reference_profile") is None:
        if gpu != 1: raise ValueError("Legacy reference owns GPU1 only")
        return
    if release.get("reference_profile") != PROFILE or type(gpu) is not int or gpu != 3:
        raise ValueError("H09Y reference requires explicitly handed-over GPU3")
    if Path(output).resolve() != ROOT/"reference_train114_v1":
        raise ValueError("Only the single new reference run is registered")


def check_initialization_deadline(release, elapsed):
    if release.get("reference_profile") == PROFILE and elapsed >= BUDGET["initialization_seconds"]:
        raise TimeoutError("Reference initialization deadline exceeded")
