import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v10_launcher", ROOT / "scripts/train_memlite_v10.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("mode,steps,warmup", [("smoke", 5, 1), ("train", 5000, 100)])
def test_actual_optimizer_command_and_fresh_initialization(mode, steps, warmup):
    argv = launcher.command(ROOT, mode)
    assert f"model.max_steps={steps}" in argv and f"model.warmup_steps={warmup}" in argv
    assert "--nproc_per_node=4" in argv and "resume_ckpt=null" in argv
    assert "model.batch_size=8" in argv and not any("baseline" in a for a in argv)
    assert f"tokenizer.vq_config.ckpt_dir={launcher.INIT_RUN / 'action_tokenizer.pt'}" in argv


def test_review_hash_covers_actual_payload_not_just_an_approved_flag():
    payload = {"ready_for_training": False, "samples": {"train": ["original"], "eval": []}}
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload.update(ready_for_training=True, review_payload_sha256=h, manual_review={"payload_sha256": h})
    launcher.verify_manual_payload(payload)
    payload["samples"]["train"][0] = "changed_after_review"
    with pytest.raises(ValueError, match="no longer matches"):
        launcher.verify_manual_payload(payload)


def test_v10_recipe_declares_full_anti_spin_training_contract(monkeypatch):
    for key, value in {"G05_OUTPUT_DIR": "/tmp/test-v10-output", "MEMLITE_INIT_CKPT": "/tmp/v9/step_5000.pt",
                       "MEMLITE_TASK_EVENT_INDEX": "/tmp/motion_recovery_index_v2.json",
                       "MEMLITE_RECOVERY_MANIFEST": "/tmp/approved_recovery.json"}.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["task=r1pro_memlite_ar_v10"])
    assert cfg.model.max_steps == 5000 and cfg.resume_ckpt is None
    arch = cfg.model.model_arch
    assert arch.discrete_action and not arch.continuous_action
    assert arch.memlite_conditioning.enabled and arch.memlite_conditioning.inference_velocity == "masked"
    assert list(arch.memlite_conditioning.velocity_indices) == [24,25,26]
    assert cfg.data.memlite_branch_sampling.strategy == "motion_recovery"
    assert cfg.memlite_recovery.enabled and cfg.memlite_runtime.progress_monitor.enabled
    assert cfg.data.val_split_mode == "task_stratified" and cfg.data.val_set_proportion == .05
    assert cfg.data.obs_size == 6 and cfg.data.action_size == 32
    assert arch.memlite_action_group_loss.lower_body_weight == 2.
