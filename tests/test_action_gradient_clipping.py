"""Small CPU tensors only; no policy weights/data or training jobs loaded."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


spec = importlib.util.spec_from_file_location("tested_action_gradient_clipping",
    Path(__file__).resolve().parents[1] / "src/g05/utils/training/action_gradient_clipping.py")
clipping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clipping)


def fixture(*, missing=False):
    # Interleave modules; group iteration order must not replace model order.
    entries = [("lora.A", nn.Parameter(torch.ones(2, 3))),
               ("expert.input", nn.Parameter(torch.ones(3, 2))),
               ("lora.B", nn.Parameter(torch.ones(3, dtype=torch.float64))),
               ("frozen", nn.Parameter(torch.ones(2), requires_grad=False)),
               ("expert.output", nn.Parameter(torch.ones(1)))]
    for index, (_, parameter) in enumerate(entries):
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, .15 + index)
    if missing:
        entries[2][1].grad = None
    groups = {"action_expert": [entries[1], entries[4]], "vlm_lora": [entries[0], entries[2]]}
    return entries, groups


def assert_same_gradients(first, second):
    for (_, a), (_, b) in zip(first, second):
        assert torch.equal(a, b)
        if a.grad is None:
            assert b.grad is None
        else:
            assert b.grad is not None and torch.equal(a.grad, b.grad)


def copied_fixture(entries):
    result = [(name, nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad)) for name, p in entries]
    for (_, a), (_, b) in zip(entries, result):
        b.grad = None if a.grad is None else a.grad.clone()
    return result


@pytest.mark.parametrize("foreach", [None, False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_global_is_bitwise_original_gradients_rng_and_parameters(foreach, missing):
    entries, groups = fixture(missing=missing)
    reference = copied_fixture(entries)
    rng = torch.random.get_rng_state().clone()
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in reference if p.requires_grad], 1.,
        error_if_nonfinite=True, foreach=foreach)
    receipt = clipping.clip_action_gradients(iter(entries), groups, foreach=foreach)
    assert_same_gradients(entries, reference)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert receipt["pre_clip_total_norm"] == float(norm)
    assert not receipt["per_loss_gradient_measurement"] and not receipt["learning_rate_multiplier_claim"]
    assert receipt["groups"]["vlm_lora"]["gradient_tensors"] == (1 if missing else 2)


@pytest.mark.parametrize("foreach", [None, False, True])
def test_per_module_matches_independent_pytorch_clips(foreach):
    entries, groups = fixture()
    reference = copied_fixture(entries)
    for indices in ([1, 4], [0, 2]):
        torch.nn.utils.clip_grad_norm_([reference[i][1] for i in indices], 1., foreach=foreach)
    receipt = clipping.clip_action_gradients(entries, groups, mode="per_module", foreach=foreach)
    assert_same_gradients(entries, reference)
    assert receipt["mode"] == "per_module"
    norms = [float(torch.nn.utils.get_total_norm([p.grad for _, p in rows])) for rows in groups.values()]
    assert all(norm <= 1.000001 for norm in norms)
    assert sum(norm**2 for norm in norms)**.5 > 1.4  # Not secretly globally recapped.


def one_per_group(expert_grad, lora_grad):
    a, b = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
    a.grad, b.grad = torch.tensor([expert_grad]), torch.tensor([lora_grad])
    entries = [("expert", a), ("lora", b)]
    return entries, {"action_expert": [entries[0]], "vlm_lora": [entries[1]]}


def test_other_module_gradient_no_longer_changes_expert_clipping():
    actual = {}
    for mode in ("global", "per_module"):
        for lora_gradient in (1., 100.):
            entries, groups = one_per_group(2., lora_gradient)
            clipping.clip_action_gradients(entries, groups, mode=mode)
            actual[mode, lora_gradient] = entries[0][1].grad.clone()
    assert torch.equal(actual["per_module", 1.], actual["per_module", 100.])
    assert not torch.equal(actual["global", 1.], actual["global", 100.])


def test_global_adamw_updates_and_moments_are_identical_to_original():
    actual, groups = fixture(missing=True)
    expected = copied_fixture(actual)
    optimizers = [torch.optim.AdamW([p for _, p in rows if p.requires_grad],
        lr=1e-5, betas=(.9, .95), weight_decay=.03) for rows in (actual, expected)]
    for step in range(3):
        for rows in (actual, expected):
            for index, (_, p) in enumerate(rows):
                if p.requires_grad and index != 2:
                    p.grad = torch.full_like(p, step * .2 + index + .1)
        clipping.clip_action_gradients(actual, groups)
        torch.nn.utils.clip_grad_norm_([p for _, p in expected if p.requires_grad], 1., error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        assert_same_gradients(actual, expected)
    a, b = [optimizer.state_dict() for optimizer in optimizers]
    assert a["param_groups"] == b["param_groups"]
    assert set(a["state"]) == set(b["state"]) and len(a["state"]) == 3
    for key, state in a["state"].items():
        assert all(torch.equal(value, b["state"][key][name]) for name, value in state.items())


@pytest.mark.parametrize("mode", ["global", "per_module"])
def test_zero_and_missing_gradients_are_not_fabricated(mode):
    entries, groups = one_per_group(0., 0.)
    entries[0][1].grad = None
    receipt = clipping.clip_action_gradients(entries, groups, mode=mode)
    assert entries[0][1].grad is None and entries[1][1].grad.item() == 0.
    assert receipt["pre_clip_total_norm"] == 0.
    assert receipt["groups"]["action_expert"]["gradient_tensors"] == 0
    assert all(group["applied_coefficient"] == 1. for group in receipt["groups"].values())


def test_lora_only_global_and_per_module_are_identical():
    a, groups = one_per_group(0., 7.)
    a = [a[1]]
    b = copied_fixture(a)
    clipping.clip_action_gradients(a, {"vlm_lora": a}, mode="global")
    clipping.clip_action_gradients(b, {"vlm_lora": b}, mode="per_module")
    assert_same_gradients(a, b)


@pytest.mark.parametrize("mode", ["global", "per_module"])
def test_mixed_bfloat16_and_fp32_gradients_follow_pytorch(mode):
    a, b = nn.Parameter(torch.ones(4, dtype=torch.bfloat16)), nn.Parameter(torch.ones(3))
    a.grad, b.grad = torch.full_like(a, 3.), torch.full_like(b, 7.)
    entries = [("expert", a), ("lora", b)]
    reference = copied_fixture(entries)
    groups = {"action_expert": [entries[0]], "vlm_lora": [entries[1]]}
    if mode == "global":
        torch.nn.utils.clip_grad_norm_([p for _, p in reference], 1.)
    else:
        for _, p in reference:
            torch.nn.utils.clip_grad_norm_([p], 1.)
    clipping.clip_action_gradients(entries, groups, mode=mode)
    assert_same_gradients(entries, reference)


def test_nonoverlapping_ddp_bucket_views_preserve_global_operation():
    entries, groups = fixture()
    bucket = torch.arange(12, dtype=torch.float32)
    entries[0][1].grad = bucket[:6].reshape(2, 3)
    entries[1][1].grad = bucket[6:].reshape(2, 3).t()  # Noncontiguous view.
    expected = copied_fixture(entries)
    torch.nn.utils.clip_grad_norm_([p for _, p in expected if p.requires_grad], 1.)
    clipping.clip_action_gradients(entries, groups)
    assert_same_gradients(entries, expected)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "cross_duplicate", "unknown_group",
    "frozen", "wrong_name", "wrong_tensor", "duplicate_model_name", "duplicate_model_tensor", "empty"])
def test_invalid_coverage_rejected_before_any_gradient_mutation(problem):
    entries, groups = fixture()
    saved = copied_fixture(entries)
    if problem == "missing": groups["vlm_lora"].pop()
    elif problem == "duplicate": groups["vlm_lora"].append(entries[0])
    elif problem == "cross_duplicate": groups["action_expert"].append(entries[0])
    elif problem == "unknown_group": groups["extra"] = [entries[0]]
    elif problem == "frozen": groups["action_expert"].append(entries[3])
    elif problem == "wrong_name": groups["vlm_lora"][0] = ("not.a.parameter", entries[0][1])
    elif problem == "wrong_tensor": groups["vlm_lora"][0] = (entries[0][0], nn.Parameter(torch.ones(2, 3)))
    elif problem == "duplicate_model_name": entries.append((entries[0][0], nn.Parameter(torch.ones(1))))
    elif problem == "duplicate_model_tensor": entries.append(("alias", entries[0][1]))
    elif problem == "empty": groups["vlm_lora"] = []
    with pytest.raises(ValueError):
        clipping.clip_action_gradients(entries, groups, mode="per_module")
    assert_same_gradients(entries[:len(saved)], saved)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 3e38])
@pytest.mark.parametrize("mode", ["global", "per_module"])
def test_nonfinite_or_overflow_norm_stops_before_clipping_another_group(value, mode):
    entries, groups = fixture()
    expert_before = entries[1][1].grad.clone()
    entries[0][1].grad.fill_(value)
    with pytest.raises(RuntimeError, match="non-finite"):
        clipping.clip_action_gradients(entries, groups, mode=mode)
    assert torch.equal(expert_before, entries[1][1].grad)


@pytest.mark.parametrize("kwargs", [{"max_norm": 0}, {"max_norm": -1}, {"max_norm": True},
    {"max_norm": float("inf")}, {"max_norm": float("nan")}, {"foreach": 1}, {"mode": "implicit"}])
def test_invalid_settings_rejected(kwargs):
    entries, groups = fixture()
    with pytest.raises(ValueError):
        clipping.clip_action_gradients(entries, groups, **kwargs)


def test_sparse_gradient_rejected_before_dense_group_changes():
    entries, groups = fixture()
    expected = entries[1][1].grad.clone()
    entries[0][1].grad = torch.sparse_coo_tensor([[0], [1]], [2.], (2, 3))
    with pytest.raises(ValueError, match="dense"):
        clipping.clip_action_gradients(entries, groups)
    assert torch.equal(entries[1][1].grad, expected)


def test_frozen_stale_gradient_is_rejected():
    entries, groups = fixture()
    entries[3][1].grad = torch.ones_like(entries[3][1])
    with pytest.raises(ValueError, match="Frozen"):
        clipping.clip_action_gradients(entries, groups)
