"""CPU checks for the explicit AR GPU-probe lifecycle/config, no GPU needed."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from probe_action_training_gpu import architecture_for, prepare_action_updates, take_row, task_generation_rows
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


def test_native_parent_override_is_explicit_without_changing_default_recipe():
    cfg = OmegaConf.create(dict(tokenizer=dict(vq_config=dict(dropout_noop_parts=True)),
        model=dict(model_arch=dict(fm={}, AT_CONFIG={}, coordination_train={}))))
    result = architecture_for(cfg, ActionTrainingSettings(conditioning="native_task"), parent="/native/model.pt")
    assert cfg.model.pretrained_ckpt == "/native/model.pt"
    assert result.action_training.conditioning == "native_task"
    assert not result.continuous_action and not result.predict_cot


def test_generation_rows_cover_real_five_tasks_without_repeating_one_row():
    batches, receipts = [], []
    for task in range(5):
        batches.append(dict(samples=[dict(task=task)], action=torch.full((1, 32, 27), float(task)),
            action_is_pad=torch.zeros(1, 32, dtype=torch.bool), action_dim_is_pad=torch.zeros(1, 27, dtype=torch.bool),
            pixel_values={"head": torch.full((1, 2), float(task))}))
        receipts.append(dict(sources=[dict(task=task, episode=task + 100, frame=42)]))
    rows = task_generation_rows(batches, dict(batches=receipts))
    for task, (batch, source) in rows.items():
        assert batch["samples"] == [dict(task=task)] and source == receipts[task]["sources"][0]
        assert batch["action"].eq(task).all()
    with pytest.raises(RuntimeError, match="all five"):
        task_generation_rows(batches[:4], dict(batches=receipts[:4]))


def test_cot_recipe_requires_native_parent_and_supervises_the_eov_boundary():
    from native_action_initialization import NATIVE_PARENT
    cfg = OmegaConf.create(dict(tokenizer=dict(vq_config=dict(dropout_noop_parts=True)),
        model=dict(model_arch=dict(fm={}, AT_CONFIG={}, coordination_train={}, input_preprocessor={}))))
    settings = ActionTrainingSettings(conditioning="native_subtask_cot")
    with pytest.raises(ValueError, match="native G0.5"):
        architecture_for(cfg, settings)
    arch = architecture_for(cfg, settings, parent=NATIVE_PARENT)
    assert arch.predict_cot and arch.input_preprocessor.pred_eov and arch.native_cot_max_new_tokens == 256
    assert not arch.register_memlite_hl_end and not arch.continuous_action
