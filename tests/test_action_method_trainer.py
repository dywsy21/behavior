"""Small contract tests for the full-data action experiment trainer."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import train_action_method_probe as recipe


def test_fixed_budget_is_an_actual_global_sixteen_batch():
    values = recipe.RECIPE
    assert values["batch_size"] * values["accumulation"] * values["world_size"] == 16
    assert values["max_updates"] == 500 and values["smoke_updates"] == 5
    assert values["history"] == 6 and values["prediction"] == 32 and values["execution"] == [0, 16]
    assert values["workers"] == 4


@pytest.mark.parametrize("route,states", [("ar", 192), ("joint", 514), ("ki", 514), ("fm", 504)])
def test_actual_adam_count_distinguishes_fm_only_unused_lora(route, states):
    assert recipe.expected_adam_states(route) == states


def test_reference_fm_flags_restore_on_exception():
    model = SimpleNamespace(discrete_action=True, continuous_action=False)
    with pytest.raises(ValueError):
        with recipe.original_fm_flags(model):
            assert not model.discrete_action and model.continuous_action
            raise ValueError("test")
    assert model.discrete_action and not model.continuous_action


def test_reference_path_cannot_be_used_for_training():
    with pytest.raises(RuntimeError, match="Reference-only"):
        recipe.reference_fm(SimpleNamespace(training=True), {})


def test_diagnostic_is_uniform_over_tasks_not_over_total_rows():
    rows = [dict(task_id=f"task{i}", ce_loss=float(i), reference_fm_loss=i / 10) for i in range(5)]
    result = recipe.summarize(rows + [deepcopy(rows[4])] * 9)
    assert result["aggregate"]["ce_loss"] == 2.
    assert result["aggregate"]["reference_fm_loss"] == pytest.approx(.2)
    with pytest.raises(RuntimeError, match="missed"):
        recipe.summarize(rows[:4])


def test_original_fm_architecture_is_not_silently_converted_to_ce():
    cfg = OmegaConf.create(dict(model=dict(model_arch=dict(
        _target_="g05.models.g05.g05_policy_memlite_skill_fm.G05PolicyMEMLiteSkillFM",
        discrete_action=False, continuous_action=True, predict_cot=False))))
    arch = recipe.configure(cfg, "fm")
    assert not arch.discrete_action and arch.continuous_action
    assert cfg.model.learning_rate == 1e-5 and cfg.model.num_workers == 4
    arch.discrete_action = True
    with pytest.raises(RuntimeError, match="unchanged"):
        recipe.configure(cfg, "fm")


def test_real_optimizer_schedule_and_frozen_parameter_exclusion():
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    def groups(**kwargs):
        assert not kwargs["apply_decay_on_norm_and_bias"]
        return [dict(params=[model.weight], lr=kwargs["lr"], weight_decay=kwargs["weight_decay"], name="backbone_decay")]
    model.get_optim_param_groups = groups
    optimizer, scheduler = recipe.make_optimizer(model)
    assert optimizer.param_groups[0]["lr"] == 0.
    before = model.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
    assert not torch.equal(before, model.weight)
    assert len(optimizer.state) == 1 and int(optimizer.state[model.weight]["step"]) == 2
    assert scheduler.last_epoch == 2


def test_duplicate_optimizer_parameters_are_rejected():
    model = torch.nn.Linear(2, 2, bias=False)
    model.get_optim_param_groups = lambda **kwargs: [dict(params=[model.weight, model.weight])]
    with pytest.raises(RuntimeError, match="once"):
        recipe.make_optimizer(model)


def checkpoint_fixture():
    state = {f"p{i}": torch.tensor([float(i)]) for i in range(1138)}
    adam = dict(param_groups=[dict(params=list(range(192)), lr=1e-6)], state={
        i: dict(step=torch.tensor(5.), exp_avg=torch.ones(1), exp_avg_sq=torch.full((1,), .1))
        for i in range(192)})
    schedule = dict(last_epoch=5, _last_lr=[1e-6])
    rng = [dict(rank=i, cpu=torch.tensor([i]), cuda=torch.tensor([i + 1]), loader=torch.tensor([i + 2]))
           for i in range(4)]
    model = SimpleNamespace(state_dict=lambda: state)
    optimizer = SimpleNamespace(state_dict=lambda: adam)
    scheduler = SimpleNamespace(state_dict=lambda: schedule)
    spec = dict(route="ar", provenance="unit-test-only")
    saved = deepcopy(dict(model_state_dict=state, optimizer_state_dict=adam, scheduler_state_dict=schedule,
        step=5, next_microbatch=10, rng_by_rank=rng, action_experiment_spec=spec))
    return saved, model, optimizer, scheduler, spec, rng


def test_full_checkpoint_and_optimizer_roundtrip_contract():
    saved, model, optimizer, scheduler, spec, rng = checkpoint_fixture()
    recipe.assert_saved_state(saved, model, optimizer, scheduler, spec, 5, rng)


@pytest.mark.parametrize("corruption", ["model", "moment", "clock", "rng", "spec", "cursor", "missing"])
def test_corrupt_checkpoint_never_passes_inspection(corruption):
    saved, model, optimizer, scheduler, spec, rng = checkpoint_fixture()
    if corruption == "model":
        saved["model_state_dict"]["p100"].add_(1)
    elif corruption == "moment":
        saved["optimizer_state_dict"]["state"][0]["exp_avg"].fill_(float("nan"))
    elif corruption == "clock":
        saved["optimizer_state_dict"]["state"][0]["step"].zero_()
    elif corruption == "rng":
        saved["rng_by_rank"][0]["cpu"].add_(1)
    elif corruption == "spec":
        saved["action_experiment_spec"]["route"] = "ki"
    elif corruption == "cursor":
        saved["next_microbatch"] = 8
    else:
        del saved["model_state_dict"]["p100"]
    with pytest.raises(RuntimeError):
        recipe.assert_saved_state(saved, model, optimizer, scheduler, spec, 5, rng)


def test_dependency_is_narrow_not_an_arbitrary_process_or_training_queue():
    with pytest.raises(ValueError, match="predeclared"):
        recipe.dependency_identity("/mnt/sdc1/robodojo")
    with pytest.raises(ValueError, match="predeclared"):
        recipe.dependency_identity(recipe.BASE / "unknown_other_training")


@pytest.mark.parametrize("state,expected", [("S", 123456), ("R", 123456), ("Z", None), ("X", None)])
def test_process_identity_handles_parentheses_in_name_and_terminal_state(tmp_path, state, expected):
    root = tmp_path / "42"
    root.mkdir()
    fields = [state] + ["0"] * 18 + ["123456"] + ["0"] * 8
    (root / "stat").write_text("42 (worker (with) spaces) " + " ".join(fields))
    assert recipe.process_start_ticks(42, tmp_path) == expected


def test_missing_process_is_not_reported_as_live(tmp_path):
    assert recipe.process_start_ticks(42, tmp_path) is None


def test_real_current_process_identity_without_pidfd():
    import os
    assert recipe.process_start_ticks(os.getpid()) > 0
