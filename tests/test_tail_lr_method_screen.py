"""CPU contract and actual PyTorch schedule/Adam toy tests; no policy/data."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

import train_action_method_probe as trainer
import tail_lr_method_screen as tail
import ema_method_screen as ema


def candidate():
    return dict(output=str(tail.BASE / tail.RUN), route="fm", initialization="a4", conditioning="skills",
        marker_rows=False, ema_recipe=None, tail_lr_recipe=deepcopy(tail.RECIPE),
        after_action=dict(kind="action", root=str(tail.BASE / tail.EMA_RUN), method_sha256=tail.EMA_SPEC_SHA))


def test_only_lr_curve_changes_in_effective_core_recipe():
    before = deepcopy(trainer.RECIPE)
    result = tail.training_recipe(trainer.RECIPE, tail.RECIPE)
    assert {k for k in before if before[k] != result[k]} == {"learning_rate", "warmup_steps", "lr_min_ratio"}
    assert result["learning_rate"] == 1e-5 * .1
    assert result["warmup_steps"] == 0 and result["lr_min_ratio"] == 1.
    assert result["max_updates"] == 500 and result["smoke_updates"] == 5
    assert trainer.RECIPE == before
    assert tail.training_recipe(trainer.RECIPE, None) == before
    result["betas"][0] = 0
    assert trainer.RECIPE == before  # Returned nested values must not mutate default settings.


def test_single_declared_tail_and_prior_default_screens():
    tail.validate_screen(candidate())
    tail.validate_screen(dict(output=str(tail.BASE / "fm_action_control_v3")))
    tail.validate_screen(dict(output=str(tail.BASE / tail.EMA_RUN)))


@pytest.mark.parametrize("key,value", [("route", "ki"), ("initialization", "native"),
    ("conditioning", "native_task"), ("marker_rows", True), ("marker_rows", None),
    ("ema_recipe", deepcopy(ema.RECIPE)), ("after_fm", {"run": "other"}),
    ("recovery_from", {"old": "completed"}), ("reference_gate_recovery", {"reuse": True}),
    ("output", str(tail.BASE / "another_tail_run")), ("tail_lr_recipe", None)])
def test_rejects_hidden_combinations_and_duplicate_budget(key, value):
    spec = candidate()
    spec[key] = value
    with pytest.raises(RuntimeError):
        tail.validate_screen(spec)


@pytest.mark.parametrize("key,value", [("learning_rate", 2e-6), ("warmup_steps", 50),
    ("lr_min_ratio", .1), ("scheduler", "other"), ("restores_parent_optimizer", True),
    ("additional_ema", True)])
def test_declared_curve_cannot_be_changed_or_claim_full_optimizer_continuation(key, value):
    selected = deepcopy(tail.RECIPE)
    selected[key] = value
    with pytest.raises(RuntimeError):
        tail.training_recipe(trainer.RECIPE, selected)


@pytest.mark.parametrize("key,value", [("root", str(tail.BASE / ema.COT_RUN)),
    ("kind", "fm"), ("method_sha256", "0" * 64)])
def test_tail_must_not_jump_ahead_of_existing_ema(key, value):
    spec = candidate()
    spec["after_action"][key] = value
    with pytest.raises(RuntimeError):
        tail.validate_screen(spec)


def test_new_ema_validator_cannot_accept_tail_combination():
    spec = dict(output=str(ema.BASE / ema.RUN), route="fm", initialization="a4", conditioning="skills",
        marker_rows=False, ema_recipe=deepcopy(ema.RECIPE),
        after_action=dict(kind="action", root=str(ema.BASE / ema.COT_RUN), method_sha256=ema.COT_SPEC_SHA))
    ema.validate_screen(spec)
    spec["tail_lr_recipe"] = deepcopy(tail.RECIPE)
    with pytest.raises(RuntimeError):
        ema.validate_screen(spec)


def complete_ema():
    return dict(passed=True, actual_updates=500, ema=dict(num_updates=500, parameter_tensors=514,
        full_state_roundtrip=True, online_restoration_checks=5))


def test_predecessor_needs_complete_ema_not_just_online_checkpoint():
    tail.validate_ema_completion(complete_ema())
    with pytest.raises(RuntimeError):
        tail.validate_ema_completion(dict(passed=True, actual_updates=500))


@pytest.mark.parametrize("key,value", [("num_updates", 5), ("parameter_tensors", 504),
    ("full_state_roundtrip", False), ("online_restoration_checks", 1)])
def test_tail_does_not_ignore_an_ema_gate_failure(key, value):
    state = complete_ema()
    state["ema"][key] = value
    with pytest.raises(RuntimeError):
        tail.validate_ema_completion(state)


def test_actual_dependency_resolver_binds_queued_ema_recipe_and_sha(monkeypatch):
    root = tail.BASE / tail.EMA_RUN
    method = dict(output=str(root), route="fm", initialization="a4", conditioning="skills",
        recipe=deepcopy(trainer.RECIPE), parent_sha256=trainer.PARENT_SHA,
        marker_rows=False, ema_recipe=deepcopy(ema.RECIPE),
        after_action=dict(kind="action", root=str(ema.BASE / ema.COT_RUN), method_sha256=ema.COT_SPEC_SHA))
    monkeypatch.setattr(trainer, "read", lambda p: {"supervisor_pid": 1707751} if p.name == "launch.json" else method)
    monkeypatch.setattr(trainer, "sha", lambda p: tail.EMA_SPEC_SHA if p.name == "method_spec.json" else "launch-sha")
    identity = trainer.dependency_identity(root, "action")
    assert identity["supervisor_pid"] == 1707751 and identity["method_sha256"] == tail.EMA_SPEC_SHA
    method["ema_recipe"] = None
    with pytest.raises(RuntimeError):
        trainer.dependency_identity(root, "action")
    method["ema_recipe"] = deepcopy(ema.RECIPE)
    monkeypatch.setattr(trainer, "sha", lambda p: "different-source")
    with pytest.raises(RuntimeError):
        trainer.dependency_identity(root, "action")


def test_actual_completion_resolver_requires_shadow_evidence(monkeypatch):
    root = tail.BASE / tail.EMA_RUN
    inspection = dict(complete_ema(), checkpoint=str(root / "formal/checkpoints/step_500.pt"),
                      checkpoint_sha256="checkpoint-sha")
    monkeypatch.setattr(trainer, "read", lambda p: {"state": "complete", "verified_updates": 500}
                        if p.name == "status.json" else inspection)
    monkeypatch.setattr(trainer, "sha", lambda p: "checkpoint-sha")
    assert trainer.verify_dependency_completed(root, "action") == root / "formal/checkpoint_inspection.json"
    del inspection["ema"]
    with pytest.raises(RuntimeError):
        trainer.verify_dependency_completed(root, "action")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = nn.Parameter(torch.tensor([1., -2.]))
        self.lora = nn.Parameter(torch.tensor([.1, .2]))

    def get_optim_param_groups(self, *, lr, weight_decay, **kwargs):
        return [dict(params=[self.ae], lr=lr, weight_decay=weight_decay),
                dict(params=[self.lora], lr=lr, weight_decay=0.)]


def assert_state_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            assert_state_equal(a[key], b[key])
        elif isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key])
        else:
            assert a[key] == b[key]


def step_toy(model, optimizer, scheduler):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .25)
    rates = [float(g["lr"]) for g in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    return rates


def test_actual_500_update_tail_curve_and_all_groups_use_the_declared_rate():
    rng = torch.get_rng_state().clone()
    model = Tiny()
    optimizer, scheduler = trainer.make_optimizer(model, tail_lr_recipe=tail.RECIPE)
    rates = [step_toy(model, optimizer, scheduler) for _ in range(500)]
    assert all(values == [tail.TAIL_LR, tail.TAIL_LR] for values in rates)
    assert scheduler.last_epoch == 500
    assert optimizer.param_groups[0]["betas"] == (.9, .95)
    assert [g["weight_decay"] for g in optimizer.param_groups] == [.03, 0.]
    assert all(int(state["step"]) == 500 for state in optimizer.state.values())
    assert torch.equal(rng, torch.get_rng_state())


def test_default_schedule_gradients_updates_and_adam_stay_bitwise_original():
    from g05.utils.training.get_scheduler import get_scheduler
    actual, reference = Tiny(), Tiny()
    optimizer, scheduler = trainer.make_optimizer(actual)
    old_optimizer = torch.optim.AdamW(reference.get_optim_param_groups(lr=1e-5, weight_decay=.03),
                                     lr=1e-5, betas=(.9, .95))
    old_scheduler = get_scheduler("cosine", old_optimizer, num_warmup_steps=50,
                                  num_training_steps=500, lr_min_ratio=.1)
    applied = []
    for _ in range(500):
        rates = step_toy(actual, optimizer, scheduler)
        assert rates == step_toy(reference, old_optimizer, old_scheduler)
        applied.append(rates[0])
        assert_state_equal(actual.state_dict(), reference.state_dict())
        assert_state_equal(optimizer.state_dict(), old_optimizer.state_dict())
    assert applied[0] == 0. and applied[50] == 1e-5
    assert math.isclose(sum(applied), 0.0027245, rel_tol=1e-12)
    assert scheduler.state_dict() == old_scheduler.state_dict()


@pytest.mark.parametrize("selected", [None, deepcopy(tail.RECIPE)])
def test_cpu_checkpoint_resume_preserves_next_actual_lr_and_adam_trajectory(selected, tmp_path):
    a, b = Tiny(), Tiny()
    opt_a, sched_a = trainer.make_optimizer(a, tail_lr_recipe=selected)
    for _ in range(200):
        step_toy(a, opt_a, sched_a)
    torch.save(dict(model=a.state_dict(), optimizer=opt_a.state_dict(), scheduler=sched_a.state_dict()), tmp_path / "state.pt")
    saved = torch.load(tmp_path / "state.pt", weights_only=True)
    opt_b, sched_b = trainer.make_optimizer(b, tail_lr_recipe=selected)
    b.load_state_dict(saved["model"])
    opt_b.load_state_dict(saved["optimizer"])
    sched_b.load_state_dict(saved["scheduler"])
    for _ in range(300):
        assert step_toy(a, opt_a, sched_a) == step_toy(b, opt_b, sched_b)
    assert_state_equal(a.state_dict(), b.state_dict())
    assert_state_equal(opt_a.state_dict(), opt_b.state_dict())
    assert sched_a.state_dict() == sched_b.state_dict()


def test_exported_model_config_matches_actual_tail_recipe_and_old_default_is_unchanged():
    base = dict(model=dict(model_arch=dict(_target_="G05PolicyMEMLiteSkillFM", discrete_action=False,
                                          continuous_action=True, predict_cot=False)))
    old, selected = OmegaConf.create(base), OmegaConf.create(base)
    trainer.configure(old, "fm")
    trainer.configure(selected, "fm", tail_lr_recipe=tail.RECIPE)
    assert old.model.learning_rate == 1e-5 and old.model.warmup_steps == 50 and old.model.lr_min_ratio == .1
    assert selected.model.learning_rate == tail.TAIL_LR
    assert selected.model.warmup_steps == 0 and selected.model.lr_min_ratio == 1.
    assert old.model.model_arch == selected.model.model_arch
    with pytest.raises(RuntimeError):
        trainer.configure(OmegaConf.create(base), "ar", tail_lr_recipe=tail.RECIPE)
    with pytest.raises(RuntimeError):
        trainer.make_optimizer(Tiny(), marker_rows=True, tail_lr_recipe=tail.RECIPE)
