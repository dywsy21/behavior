"""CPU checks against the actual FM helper, not a separate imitation of it."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from g05.models.g05.helpers.fm_helper import FMHelper
from g05.utils.training.fm_training_methods import (
    FMTrainingMethods, executed_prefix_codec_input, fm_policy_training_methods,
    fm_training_methods, stratified_beta_times,
)


def helper():
    return FMHelper(SimpleNamespace(time_convention="pi_convention", num_inference_steps=4,
        horizon_steps=32, action_dim=27, fm_weight=1., final_action_clip_value=None,
        flow_sig_min=.001, padding_action_weight=0., zero_pad_action_target=False,
        joint_training=True, num_flow_samples=4, flow_sampling="beta"))


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(27, 27, bias=False)
        self.layers = [SimpleNamespace(mlp=SimpleNamespace(gate_proj=self.projection))]

    def embed(self, x):
        return x

    def encode_time(self, t):
        return t

    def forward(self, inputs_embeds, **kwargs):
        return self.projection(inputs_embeds)

    def decode(self, h):
        return h


def model_and_args():
    model = SimpleNamespace(action_expert=Expert(), attn_implementation="eager")
    def masks(mask, action_len, **kwargs):
        return torch.ones(mask.shape[0], 1, action_len, mask.shape[1] + action_len), torch.zeros(mask.shape[0], action_len).long()
    model.build_action_mask_and_position_ids = masks
    dim_pad = torch.zeros(2, 27).bool()
    dim_pad[:, [7, 8, 17, 18]] = True
    args = (model, [], torch.ones(2, 3).long(), torch.zeros(2, 3).long(),
            torch.randn(2, 32, 27), torch.zeros(2, 32).bool(), dim_pad, torch.float32)
    return model, args


def test_disabled_scope_preserves_loss_gradient_rng_and_instance():
    h = helper()
    model, args = model_and_args()
    original_keys = set(h.__dict__)
    torch.manual_seed(91)
    expected = h.train_step(*args)
    expected.backward()
    gradient = model.action_expert.projection.weight.grad.clone()
    rng = torch.random.get_rng_state().clone()
    model.action_expert.zero_grad()
    torch.manual_seed(91)
    with fm_training_methods(h, FMTrainingMethods(), batch_size=2):
        actual = h.train_step(*args)
    actual.backward()
    assert torch.equal(actual, expected)
    assert torch.equal(gradient, model.action_expert.projection.weight.grad)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(h.__dict__) == original_keys


@pytest.mark.parametrize("convention", ["pi_convention", "galaxea_convention"])
def test_time_strata_cover_original_beta_quantiles_and_moments(convention):
    torch.manual_seed(19)
    t = stratified_beta_times(4, 10000, flow_sig_min=.001, time_convention=convention)
    z = (t - .001) / .999 if convention == "pi_convention" else 1 - t / .999
    q = z.pow(1.5)
    for j in range(4):
        assert (q[j] >= j / 4 - 1e-6).all()
        assert (q[j] <= (j + 1) / 4 + 1e-6).all()
    assert abs(float(z.mean()) - .6) < .002
    assert abs(float(z.square().mean()) - 3 / 7) < .002


def test_weighted_objective_retains_all_23_controls_and_padding_has_no_gradient():
    h = helper()
    pred = torch.ones(1, 32, 27, requires_grad=True)
    noise = torch.zeros_like(pred)
    target = torch.zeros_like(pred)
    temporal = torch.zeros(1, 32).bool()
    temporal[:, -1] = True
    dimensions = torch.zeros(1, 27).bool()
    dimensions[:, [7, 8, 17, 18]] = True
    with fm_training_methods(h, FMTrainingMethods(execution_weight=2), batch_size=1):
        loss = h.cal_fm_loss(pred, noise, target, temporal, dimensions)
    loss.backward()
    assert loss.item() == 1
    assert pred.grad[:, :, dimensions[0]].eq(0).all()
    assert pred.grad[:, -1].eq(0).all()
    assert pred.grad[:, :31, ~dimensions[0]].gt(0).all()
    assert torch.equal(pred.grad[:, 0], 2 * pred.grad[:, 16])
    assert pred.grad[:, :, 20:27].abs().sum() > 0


def test_opt_in_real_forward_backward_and_reference_restoration():
    h = helper()
    model, args = model_and_args()
    torch.manual_seed(37)
    reference = h.train_step(*args)
    with fm_training_methods(h, FMTrainingMethods("beta_stratified", 16, 2), batch_size=2):
        loss = h.train_step(*args)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_expert.projection.weight.grad.abs().sum() > 0
    torch.manual_seed(37)
    assert torch.equal(reference, h.train_step(*args))


def test_scope_restores_on_exception_and_refuses_nesting_and_incomplete_draws():
    h = helper()
    old = dict(h.__dict__)
    with pytest.raises(RuntimeError, match="nested"):
        with fm_training_methods(h, FMTrainingMethods(), batch_size=2):
            with fm_training_methods(h, FMTrainingMethods(), batch_size=2):
                pass
    assert set(old) == set(h.__dict__)
    with pytest.raises(RuntimeError, match="complete"):
        with fm_training_methods(h, FMTrainingMethods("beta_stratified"), batch_size=2):
            h.sample_time(2)
    assert set(old) == set(h.__dict__)


@pytest.mark.parametrize("kwargs", [{"time_sampling": "uniform"}, {"execution_weight": 0},
                                    {"execution_weight": float("nan")}, {"execution_horizon": 0}])
def test_bad_settings_fail(kwargs):
    with pytest.raises(ValueError):
        replace(FMTrainingMethods(), **kwargs)


def test_codec_prefix_is_exact_and_independent_of_unused_future():
    actions = torch.randn(2, 32, 27)
    original = actions.clone()
    padded = executed_prefix_codec_input(actions)
    assert torch.equal(actions, original)
    assert torch.equal(padded[:, :16], actions[:, :16])
    assert torch.equal(padded[:, 16:], actions[:, 15:16].expand(-1, 16, -1))
    actions[:, 16:] = torch.randn_like(actions[:, 16:]) * 100
    assert torch.equal(padded, executed_prefix_codec_input(actions))
    with pytest.raises(ValueError):
        executed_prefix_codec_input(actions, 33)


def test_codec_short_valid_prefix_holds_last_real_action_and_rejects_holes():
    actions = torch.randn(2, 32, 27)
    padding = torch.zeros(2, 32).bool()
    padding[0, 5:] = True
    result = executed_prefix_codec_input(actions, action_is_pad=padding)
    assert torch.equal(result[0, :5], actions[0, :5])
    assert torch.equal(result[0, 5:], actions[0, 4:5].expand(27, -1))
    actions[0, 5:] = 1e6
    assert torch.equal(result, executed_prefix_codec_input(actions, action_is_pad=padding))
    padding[0, 6] = False
    with pytest.raises(ValueError, match="holes"):
        executed_prefix_codec_input(actions, action_is_pad=padding)
    padding[0] = True
    with pytest.raises(ValueError, match="at least one"):
        executed_prefix_codec_input(actions, action_is_pad=padding)


def test_policy_hook_is_train_only_preserves_inherited_forward_and_restores():
    class Parent(nn.Module):
        def forward(self, batch, inference_mode=False):
            return getattr(self.model.fm_helper, "_fm_training_methods_active", False)
    class Policy(Parent):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(fm_helper=helper())
    policy = Policy()
    batch = {"action": torch.zeros(2, 32, 27)}
    original = Policy.forward
    with fm_policy_training_methods(Policy, FMTrainingMethods(execution_weight=2)):
        assert policy(batch)
        assert not policy(batch, inference_mode=True)
        with torch.no_grad():
            assert not policy(batch)
        policy.eval()
        assert not policy(batch)
        assert not getattr(policy.model.fm_helper, "_fm_training_methods_active", False)
        with pytest.raises(RuntimeError, match="already installed"):
            with fm_policy_training_methods(Policy, FMTrainingMethods()):
                pass
    assert Policy.forward is original
    assert "forward" not in Policy.__dict__
    with pytest.raises(ValueError):
        with fm_policy_training_methods(Policy, FMTrainingMethods()):
            policy.train()
            policy({"action": torch.zeros(2, 27)})
    assert Policy.forward is original
