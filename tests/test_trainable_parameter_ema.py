"""CPU toy modules only; no policy/data/GPU/optimizer training job is loaded."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn


HELPER = Path(__file__).resolve().parents[1] / "src/g05/utils/training/trainable_parameter_ema.py"
spec = importlib.util.spec_from_file_location("tested_trainable_parameter_ema", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EMA = module.TrainableParameterEMA


class Toy(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.expert = nn.Parameter(torch.tensor([1., -2.], dtype=dtype))
        self.lora = nn.Parameter(torch.tensor([.1, .2], dtype=dtype))
        self.unused = nn.Parameter(torch.tensor([3.], dtype=dtype))
        self.frozen = nn.Parameter(torch.tensor([9.], dtype=dtype), requires_grad=False)
        self.register_buffer("stats", torch.tensor([7.], dtype=dtype))


def assert_states_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            assert_states_equal(a[key], b[key])
        elif isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key])
        else:
            assert a[key] == b[key]


@pytest.mark.parametrize("beta", [0., .5, .99])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_recurrence_and_no_online_or_rng_mutation(beta, dtype):
    model = Toy(dtype)
    before_rng = torch.get_rng_state().clone()
    ema = EMA(model, beta=beta)
    assert ema.parameter_names == ("expert", "lora", "unused")
    assert ema.shadow_bytes == 5 * model.expert.element_size()
    expected = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    for step in range(1, 9):
        with torch.no_grad():
            model.expert.add_(.25)
            model.lora.mul_(.95)
        online = deepcopy(model.state_dict())
        for name, target in expected.items():
            target.mul_(beta).add_(dict(model.named_parameters())[name].detach(), alpha=1 - beta)
        ema.update(optimizer_step=step)
        assert ema.num_updates == step
        assert_states_equal(expected, ema.state_dict()["shadow"])
        assert_states_equal(online, model.state_dict())
    assert torch.equal(before_rng, torch.get_rng_state())


def test_online_adam_gradients_moments_and_rng_match_without_ema():
    reference, tracked = Toy(), Toy()
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(.9, .95)) for m in (reference, tracked)]
    ema = EMA(tracked)
    rng = torch.get_rng_state().clone()
    for step in range(1, 9):
        for model, opt in zip((reference, tracked), optimizers):
            opt.zero_grad(set_to_none=True)
            loss = (model.expert * (.2 + step) + model.lora).square().mean()
            loss.backward()
            opt.step()
        ema.update(optimizer_step=step)
        with ema.average_parameters():
            assert not torch.is_grad_enabled()
            tracked.expert.square().mean()
        assert_states_equal(reference.state_dict(), tracked.state_dict())
        assert_states_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
        for (_, a), (_, b) in zip(reference.named_parameters(), tracked.named_parameters()):
            assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)
        assert len(optimizers[1].state) == 2  # No fabricated state for unused/frozen parameters.
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("raise_inside", [False, True])
def test_temporary_average_restores_on_exit_and_keeps_parameter_references(raise_inside):
    model = Toy()
    ema = EMA(model, beta=.5)
    with torch.no_grad():
        model.expert.add_(4)
    ema.update(optimizer_step=1)
    saved = deepcopy(model.state_dict())
    identities = {n: id(p) for n, p in model.named_parameters()}
    average = ema.state_dict()["shadow"]
    try:
        with ema.average_parameters():
            assert torch.equal(model.expert, average["expert"])
            assert torch.equal(model.frozen, saved["frozen"])
            assert torch.equal(model.stats, saved["stats"])
            for forbidden in (lambda: ema.update(optimizer_step=2), ema.state_dict,
                    lambda: ema.load_state_dict({})):
                with pytest.raises(RuntimeError):
                    forbidden()
            with pytest.raises(RuntimeError):
                with ema.average_parameters():
                    pass
            if raise_inside:
                raise LookupError("evaluation failed")
    except LookupError:
        assert raise_inside
    assert {n: id(p) for n, p in model.named_parameters()} == identities
    assert_states_equal(saved, model.state_dict())
    assert ema.restoration_checks == 1
    ema.update(optimizer_step=2)


@pytest.mark.parametrize("bad_step", [0, 2, -1, True, 1.0, None])
def test_bad_clock_is_rejected_before_mutation(bad_step):
    ema = EMA(Toy())
    saved = ema.state_dict()
    with pytest.raises(ValueError):
        ema.update(optimizer_step=bad_step)
    assert_states_equal(saved, ema.state_dict())


@pytest.mark.parametrize("key,value", [("num_updates", -1), ("num_updates", True),
    ("num_updates", 9), ("beta", .8), ("update_every", 10), ("format_version", 2),
    ("initialization", "copy_first_updated_model"), ("kind", "other"), ("parameters", [])])
def test_bad_restoration_is_atomic(key, value):
    ema = EMA(Toy())
    saved = ema.state_dict()
    bad = deepcopy(saved)
    bad[key] = value
    with pytest.raises(ValueError):
        ema.load_state_dict(bad, expected_updates=0)
    assert_states_equal(saved, ema.state_dict())


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "dtype", "nan", "inf", "grad", "no_clock"])
def test_bad_shadow_is_not_partly_loaded(fault):
    ema = EMA(Toy())
    saved = ema.state_dict()
    bad = deepcopy(saved)
    bad["shadow"]["expert"].fill_(42)  # Must not be partly applied if a later field is bad.
    if fault == "missing":
        del bad["shadow"]["lora"]
    elif fault == "extra":
        bad["shadow"]["frozen"] = torch.ones(1)
    elif fault == "shape":
        bad["shadow"]["lora"] = torch.ones(3)
    elif fault == "dtype":
        bad["shadow"]["lora"] = bad["shadow"]["lora"].double()
    elif fault in ("nan", "inf"):
        bad["shadow"]["lora"].fill_(float(fault))
    elif fault == "grad":
        bad["shadow"]["lora"].requires_grad_(True)
    else:
        del bad["num_updates"]
    with pytest.raises(ValueError):
        ema.load_state_dict(bad, expected_updates=0)
    assert_states_equal(saved, ema.state_dict())


def test_state_is_detached_and_restore_continues_exactly_from_disk(tmp_path):
    first, second = Toy(), Toy()
    a, b = EMA(first), EMA(second)
    with torch.no_grad():
        first.expert.add_(1)
        second.expert.add_(1)
    a.update(optimizer_step=1)
    state = a.state_dict()
    torch.save(state, tmp_path / "ema.pt")
    state["shadow"]["expert"].zero_()
    assert not torch.equal(state["shadow"]["expert"], a.state_dict()["shadow"]["expert"])
    state = torch.load(tmp_path / "ema.pt", weights_only=True)
    b.load_state_dict(state, expected_updates=1)
    a.assert_matches_state(state, expected_updates=1)
    for step in range(2, 5):
        with torch.no_grad():
            first.expert.add_(step)
            second.expert.add_(step)
        a.update(optimizer_step=step)
        b.update(optimizer_step=step)
    assert_states_equal(a.state_dict(), b.state_dict())
    # New CPU process, not merely a newly constructed object in this interpreter.
    program = '''import importlib.util, sys, torch
s=importlib.util.spec_from_file_location("ema", sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
net=torch.nn.Module()
net.register_parameter("expert", torch.nn.Parameter(torch.tensor([2., -1.])))
net.register_parameter("lora", torch.nn.Parameter(torch.tensor([.1, .2])))
net.register_parameter("unused", torch.nn.Parameter(torch.tensor([3.])))
e=m.TrainableParameterEMA(net)
state=torch.load(sys.argv[2], weights_only=True); e.load_state_dict(state, expected_updates=1)
e.assert_matches_state(state, expected_updates=1)
for step in range(2,5):
 with torch.no_grad(): net.expert.add_(step)
 e.update(optimizer_step=step)
torch.save(e.state_dict(), sys.argv[3])
'''
    subprocess.run([sys.executable, "-c", program, str(HELPER), str(tmp_path / "ema.pt"),
                    str(tmp_path / "continued.pt")], check=True, timeout=60)
    assert_states_equal(a.state_dict(), torch.load(tmp_path / "continued.pt", weights_only=True))


@pytest.mark.parametrize("fault", ["replace", "dtype", "freeze", "unfreeze", "shape", "nan"])
def test_online_contract_faults_cannot_advance_ema(fault):
    model = Toy()
    ema = EMA(model)
    shadow = deepcopy(ema._shadow)
    if fault == "replace":
        model.lora = nn.Parameter(model.lora.detach().clone())
    elif fault == "dtype":
        model.double()
    elif fault == "freeze":
        model.lora.requires_grad_(False)
    elif fault == "unfreeze":
        model.frozen.requires_grad_(True)
    elif fault == "shape":
        with torch.no_grad():
            model.lora.set_(torch.ones(3))
    else:
        with torch.no_grad():
            model.lora.fill_(float("nan"))
    with pytest.raises(ValueError):
        ema.update(optimizer_step=1)
    assert ema.num_updates == 0
    assert_states_equal(shadow, ema._shadow)


@pytest.mark.parametrize("beta", [1., -.1, float("nan"), float("inf"), True, "0.99"])
def test_invalid_beta(beta):
    with pytest.raises(ValueError):
        EMA(Toy(), beta=beta)


def test_low_precision_master_parameters_are_explicitly_rejected():
    with pytest.raises(ValueError, match="FP32/FP64"):
        EMA(Toy(torch.bfloat16))
