"""CPU checks for the explicit AR GPU-probe lifecycle/config, no GPU needed."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from probe_action_training_gpu import architecture_for, prepare_action_updates, take_row
from g05.utils.training.ar_training_methods import ActionTrainingSettings


def test_non_module_action_codec_is_moved_explicitly():
    calls = []
    model = SimpleNamespace(configure_coordination_trainability=lambda: calls.append("freeze") or {"ok": True},
        apply_fp32_params=lambda: calls.append("precision"),
        action_tokenizer=SimpleNamespace(to=lambda device: calls.append(("codec", device))))
    assert prepare_action_updates(model, "cuda:0") == {"ok": True}
    assert calls == ["freeze", "precision", ("codec", "cuda:0")]


@pytest.mark.parametrize("route", ["ar", "joint", "ki"])
def test_recipe_records_real_route_and_auxiliary_loss_settings(route):
    cfg = OmegaConf.create(dict(tokenizer=dict(vq_config=dict(dropout_noop_parts=True)),
        model=dict(model_arch=dict(fm={}, AT_CONFIG={}, coordination_train={}))))
    result = architecture_for(cfg, ActionTrainingSettings(route=route))
    assert result.discrete_action is True
    assert result.continuous_action == (route != "ar")
    assert result.fm.joint_training == (route == "joint")
    assert not result.predict_cot and not result.AT_CONFIG.dropout_noop_parts
    assert result.coordination_train.stage == "experimental_" + route
    assert result.action_training.route == route
    assert cfg.resume_ckpt is None


def test_one_row_is_a_real_slice_not_a_repeat_or_audit_payload():
    batch = dict(samples=[{"name": "first"}, {"name": "second"}],
        action=torch.arange(2 * 32 * 27).reshape(2, 32, 27),
        action_is_pad=torch.zeros(2, 32, dtype=torch.bool),
        action_dim_is_pad=torch.zeros(2, 27, dtype=torch.bool),
        pixel_values={"head": torch.arange(4).reshape(2, 2)}, audit_only={"must": "not enter"})
    result = take_row(batch, 1)
    assert result["samples"] == [{"name": "second"}]
    assert torch.equal(result["action"], batch["action"][1:2])
    assert torch.equal(result["pixel_values"]["head"], batch["pixel_values"]["head"][1:2])
    assert "audit_only" not in result
    result["action"].zero_()
    assert batch["action"][1].count_nonzero()
