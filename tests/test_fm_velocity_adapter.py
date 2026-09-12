"""Exercise the real shared FM helper's opt-in extension and legacy defaults."""
from types import SimpleNamespace

import torch
from torch import nn

from g05.models.g05.helpers.fm_helper import FMHelper


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [SimpleNamespace(mlp=SimpleNamespace(gate_proj=nn.Linear(3,3)))]

    def embed(self, x):
        return x

    def encode_time(self, t):
        return t

    def forward(self, inputs_embeds, **kwargs):
        return inputs_embeds

    def decode(self, h):
        return torch.zeros_like(h)


def fixture():
    helper = FMHelper(SimpleNamespace(time_convention="pi_convention", num_inference_steps=4,
        horizon_steps=4, action_dim=3, fm_weight=1., final_action_clip_value=None,
        flow_sig_min=0., padding_action_weight=0., zero_pad_action_target=False,
        joint_training=False, num_flow_samples=2, flow_sampling="uniform"))
    def masks(mask, action_len, **kwargs):
        b = mask.shape[0]
        return torch.ones(b, 1, action_len, mask.shape[1]+action_len), torch.zeros(b,action_len).long()
    model = SimpleNamespace(action_expert=Expert(), attn_implementation="sdpa",
                            build_action_mask_and_position_ids=masks)
    return helper, model


def test_legacy_fm_training_and_opt_in_adapter_gradient():
    helper, model = fixture()
    args = (model, [], torch.ones(2,3).long(), torch.zeros(2,3).long(),
            torch.randn(2,4,3), torch.zeros(2,4).bool(), torch.zeros(2,3).bool(), torch.float32)
    torch.manual_seed(61)
    expected = helper.train_step(*args)
    torch.manual_seed(61)
    actual = helper.train_step(*args, velocity_adapter=lambda h,v: v)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    weight = nn.Parameter(torch.tensor(.1))
    calls = []
    def adapt(h, v):
        calls.append(h.shape)
        return v + weight * h
    loss = helper.train_step(*args, velocity_adapter=adapt)
    loss.backward()
    assert calls == [torch.Size([4,4,3])]
    assert weight.grad is not None and torch.isfinite(weight.grad) and weight.grad.abs() > 0


def test_inference_adapter_is_inside_every_flow_step_and_off_is_exact():
    helper, model = fixture()
    args = (model, torch.ones(2,3).long(), {"camera": torch.zeros(2,1,3,2,2)}, [])
    torch.manual_seed(63)
    expected = helper.infer(*args)
    torch.manual_seed(63)
    zero = helper.infer(*args, velocity_adapter=lambda h,v: v)
    torch.testing.assert_close(expected, zero, rtol=0, atol=0)
    calls = []
    def adapt(h,v):
        calls.append(h.shape)
        return v + .25
    torch.manual_seed(63)
    corrected = helper.infer(*args, velocity_adapter=adapt)
    assert len(calls) == helper.num_inference_steps
    torch.testing.assert_close(corrected, expected-.25)
