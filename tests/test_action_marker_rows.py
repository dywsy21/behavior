"""Real CPU tensors: tied row deltas, gradients, state and eager-CE contract."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from g05.models.g05.helpers.action_marker_rows import (
    ActionMarkerRows, required_marker_ids, install_marker_rows, assert_marker_bindings, restore_marker_rows,
)


def vlm():
    model = nn.Module()
    model.input_proj = nn.Embedding(19, 4)
    model.output_proj = nn.Linear(4, 19, bias=False)
    model.output_proj.weight = model.input_proj.weight
    return model


def test_required_markers_follow_codec_levels_not_reserved_rows():
    codec = SimpleNamespace(action_token_begin_idx=100, action_token_end_idx=130, _codebook_size=10,
        serializer=SimpleNamespace(nn_key_names=["arm", "body"], rule_key_names=["gripper"],
            num_residuals=2, max_residuals=4,
            group_marker_action_indices={"<arm_0>": 10, "<body_0>": 11, "<arm_1>": 12,
                "<body_1>": 13, "<gripper>": 18, "<body_3>": 17}))
    assert required_marker_ids(codec) == (110, 111, 112, 113, 118)
    codec.serializer.group_marker_action_indices["<body_0>"] = 10
    with pytest.raises(ValueError, match="unique"):
        required_marker_ids(codec)


@pytest.mark.parametrize("ids", [(), (3, 2), (2, 2), (-1,), (19,), (True,), (2.0,)])
def test_marker_ids_reject_ambiguous_or_invalid_values(ids):
    with pytest.raises(ValueError):
        ActionMarkerRows(ids, vocab_size=19, hidden_size=4, device="cpu", dtype=torch.float32)


def test_zero_init_preserves_parent_and_only_selected_outputs_change():
    model, ids = vlm(), torch.tensor([[0, 2, 4, 7]])
    hidden = torch.randn(2, 3, 4)
    old_embed, old_logits = model.input_proj(ids), model.output_proj(hidden)
    saved = deepcopy(model.state_dict())
    adapter = install_marker_rows(model, (2, 7))
    assert_marker_bindings(model, adapter)
    assert torch.equal(old_embed, model.input_proj(ids))
    assert torch.equal(old_logits, model.output_proj(hidden))
    assert set(model.state_dict()) == set(saved)
    with torch.no_grad():
        adapter.delta.copy_(torch.randn_like(adapter.delta))
    expected = saved["input_proj.weight"].clone()
    expected[[2, 7]] += adapter.delta.detach()
    torch.testing.assert_close(model.input_proj(ids), F.embedding(ids, expected))
    torch.testing.assert_close(model.output_proj(hidden), F.linear(hidden, expected))
    other = [index for index in range(19) if index not in (2, 7)]
    assert torch.equal(old_logits[..., other], model.output_proj(hidden)[..., other])
    assert torch.equal(model.input_proj.weight, saved["input_proj.weight"])


def test_both_input_and_output_share_real_gradients_and_tiny_optimizer_state():
    model = vlm()
    adapter = install_marker_rows(model, (2, 7))
    model.add_module("action_marker_rows", adapter)
    ids = torch.tensor([[2, 0, 7]])
    optimizer = torch.optim.AdamW([adapter.delta], lr=.01, weight_decay=0.)
    base = model.input_proj.weight.detach().clone()
    # Each route separately reaches the ONE registered parameter.
    for objective in (lambda: model.input_proj(ids).square().sum(),
                      lambda: F.cross_entropy(model.output_proj(torch.randn(2, 4)), torch.tensor([2, 7]))):
        optimizer.zero_grad(set_to_none=True)
        objective().backward()
        assert adapter.delta.grad is not None and torch.count_nonzero(adapter.delta.grad)
        assert model.input_proj.weight.grad is None
        optimizer.step()
    assert len(optimizer.state) == 1 and optimizer.state[adapter.delta]["exp_avg"].numel() == 8
    assert torch.equal(model.input_proj.weight, base)
    assert len([name for name, _ in model.named_parameters() if "delta" in name]) == 1


def test_deepcopy_device_dtype_and_strict_state_restore_preserve_tying():
    model = vlm()
    model.add_module("action_marker_rows", install_marker_rows(model, (2, 7)))
    with torch.no_grad():
        model.action_marker_rows.delta.fill_(.125)
    restored = deepcopy(model).double()
    assert_marker_bindings(restored, restored.action_marker_rows)
    assert restored.action_marker_rows is not model.action_marker_rows
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.output_proj(torch.ones(2, 4, dtype=torch.float64)),
                               model.output_proj(torch.ones(2, 4)).double())


def test_partial_or_different_identity_adapter_resume_is_rejected():
    adapter = ActionMarkerRows((2, 7), vocab_size=19, hidden_size=4, device="cpu", dtype=torch.float32)
    assert restore_marker_rows(adapter, {})["mode"] == "zero_init"
    state = {"model.action_marker_rows." + k: v.clone() for k, v in adapter.state_dict().items()}
    state["model.action_marker_rows.delta"].fill_(.2)
    assert restore_marker_rows(adapter, state)["restored"] == 2
    assert torch.all(adapter.delta == .2)
    bad = dict(state)
    bad["model.action_marker_rows.token_ids"] = torch.tensor([7, 2])
    with pytest.raises(RuntimeError, match="incompatible"):
        restore_marker_rows(adapter, bad)
    del bad["model.action_marker_rows.token_ids"]
    with pytest.raises(RuntimeError, match="Partial"):
        restore_marker_rows(adapter, bad)
    with pytest.raises(RuntimeError, match="namespace"):
        restore_marker_rows(adapter, {"model.vlm.action_marker_rows.delta": adapter.delta.detach().clone()})


def test_untied_or_twice_installed_adapter_rejected():
    model = vlm()
    model.output_proj.weight = nn.Parameter(model.output_proj.weight.detach().clone())
    with pytest.raises(ValueError, match="tied"):
        install_marker_rows(model, (2, 7))
    model.output_proj.weight = model.input_proj.weight
    install_marker_rows(model, (2, 7))
    with pytest.raises(ValueError, match="plain embedding"):
        install_marker_rows(model, (2, 7))


@pytest.mark.parametrize("train_rows", [False, True])
def test_policy_route_checks_reject_fused_ce_and_unexpected_trainables(train_rows):
    from g05.models.g05.g05_policy_memlite_action_rows import G05PolicyMEMLiteActionRows
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    policy = G05PolicyMEMLiteActionRows.__new__(G05PolicyMEMLiteActionRows)
    nn.Module.__init__(policy)
    policy.train_marker_rows = train_rows
    policy.action_training = ActionTrainingSettings(route="ar")
    policy._coordination_post_load_receipt = {"fixture": True}
    policy.model = nn.Module()
    policy.model.vlm = vlm()
    policy.model.vlm.lora_A = nn.Linear(4, 1, bias=False)
    policy.model.vlm.lora_B = nn.Linear(1, 4, bias=False)
    policy.model.vlm.get_base_model = lambda: policy.model.vlm
    policy.model.action_marker_rows = install_marker_rows(policy.model.vlm, (2, 7))
    policy.model.ar_helper = SimpleNamespace(use_fused_ce=False)
    receipt = policy.configure_coordination_trainability()
    assert receipt["expected_trainable_groups"] == (["action_marker_rows", "vlm_lora"] if train_rows else ["vlm_lora"])
    policy.model.ar_helper.use_fused_ce = True
    with pytest.raises(RuntimeError, match="fused bypass"):
        policy._assert_action_training_contract()
    policy.model.ar_helper.use_fused_ce = False
    policy.model.vlm.input_proj.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="declared groups"):
        policy._assert_action_training_contract()


def test_marker_probe_counts_actual_single_row_markers_and_rejects_partial_cache():
    import importlib.util
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts/experiments"))
    from probe_ar_marker_learning import marker_cache_metrics
    logits = torch.tensor([[0., 4., 2., 1.], [1., 2., 4., 0.]])
    cache = dict(shift_labels_masked=torch.tensor([1, 2]), shift_logits=logits,
                 token_loss=F.cross_entropy(logits, torch.tensor([1, 2]), reduction="none"))
    rows = marker_cache_metrics(cache, [1, 2])
    assert [row["rank"] for row in rows] == [1, 1]
    assert [row["target_index"] for row in rows] == [0, 1]
    with pytest.raises(RuntimeError, match="exactly once"):
        marker_cache_metrics(cache, [1, 3])
    cache["shift_logits"] = None
    with pytest.raises(RuntimeError, match="eager"):
        marker_cache_metrics(cache, [1, 2])


def test_marker_recipe_is_explicit_eager_and_has_one_training_difference():
    from omegaconf import OmegaConf
    from probe_ar_marker_learning import marker_architecture
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    cfg = OmegaConf.create(dict(tokenizer=dict(vq_config={}), model=dict(model_arch=dict(
        fm={}, ar={"use_fused_ce": True}, AT_CONFIG={}, coordination_train={}))))
    control = marker_architecture(deepcopy(cfg), ActionTrainingSettings(route="ar"), "control")
    candidate = marker_architecture(deepcopy(cfg), ActionTrainingSettings(route="ar"), "markers")
    assert not control.ar.use_fused_ce and not candidate.ar.use_fused_ce
    assert not control.marker_row_adaptation.train_rows and candidate.marker_row_adaptation.train_rows
    candidate.marker_row_adaptation.train_rows = False
    assert candidate == control
