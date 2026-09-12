from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch


def test_old_high_training_env_is_not_required_for_serving(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from serve_policy_memlite_fm import load_serving_config
    for key in ("MISSING_OLD_INIT", "MISSING_OLD_INDEX", "MISSING_OLD_RECOVERY"):
        monkeypatch.delenv(key, raising=False)
    config = OmegaConf.create({"output_dir":"${hydra:runtime.output_dir}",
        "model":{"pretrained_ckpt":"${oc.env:MISSING_OLD_INIT}", "model_arch":{"pretrained_model_path":None}},
        "data":{"memlite_branch_sampling":{"sampling_index_path":"${oc.env:MISSING_OLD_INDEX}"}},
        "memlite_recovery":{"enabled":True,"manifest":"${oc.env:MISSING_OLD_RECOVERY}"},
        "logger":{"task":"${hydra:runtime.choices.task}"}})
    (tmp_path/".hydra").mkdir()
    OmegaConf.save(config, tmp_path/".hydra/config.yaml")
    checkpoint = tmp_path/"step_5.pt"
    checkpoint.touch()
    bridge = SimpleNamespace(_safe_apply_action_tokenizer_sidecar=lambda cfg,run:False)
    actual = load_serving_config(str(checkpoint), bridge=bridge)
    assert actual.ckpt_path == str(checkpoint) and actual.model.pretrained_ckpt is None
    assert actual.memlite_recovery.enabled is False and actual.logger.task == "eval"


def test_fm_launch_saves_explicit_provenance_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_memlite_fm_v11 import command
    for mode in ("smoke", "train"):
        settings = dict(x.split("=",1) for x in command(mode) if "=" in x and not x.startswith("--"))
        for key in ("model.pretrained_ckpt", "memlite_fm.controller_init", "memlite_fm.high_checkpoint",
                    "memlite_recovery.manifest", "data.memlite_branch_sampling.sampling_index_path"):
            assert settings[key].startswith("/mnt/sdc1/") and "${" not in settings[key]


def test_continuous_gripper_projection_matches_official_saturation_and_preserves_other_groups(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from serve_policy_memlite_fm import project_fm_grippers
    from g05.utils.data.normalizer import ActionNormalizationError
    source = {"left_gripper":torch.tensor([[[-1.02],[.3],[1.1]]]),
        "right_gripper":torch.tensor([[[1.02],[-.3],[-1.1]]]), "base_qvel":torch.randn(1,3,3),
        "left_arm":torch.randn(1,3,7), "_normalization_diagnostics":{"arm":{"clipped_count":0}}}
    result = project_fm_grippers(source)
    assert result["base_qvel"] is source["base_qvel"] and result["left_arm"] is source["left_arm"]
    for key in ("left_gripper","right_gripper"):
        torch.testing.assert_close(result[key], source[key].clip(-1,1), rtol=0,atol=0)
        assert result["_normalization_diagnostics"][f"fm_command.{key}"]["clipped_count"] == 2
        assert source[key].abs().max() > 1
    assert source["_normalization_diagnostics"] == {"arm":{"clipped_count":0}}
    for bad in (2.01,float("nan"),float("inf")):
        changed = {**source,"left_gripper":torch.tensor([[[bad]]])}
        with pytest.raises(ActionNormalizationError):
            project_fm_grippers(changed)
    with pytest.raises(ActionNormalizationError):
        project_fm_grippers({"left_gripper":source["left_gripper"]})
