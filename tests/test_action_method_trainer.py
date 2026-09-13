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


@pytest.mark.parametrize("initialization,conditioning,expected", [
    ("a4", "skills", recipe.PARENT), ("native", "skills", recipe.NATIVE_PARENT),
    ("native", "native_task", recipe.NATIVE_PARENT)])
def test_training_parent_is_explicit_and_not_the_serial_predecessor(initialization, conditioning, expected):
    path, digest = recipe.declared_parent("ar", initialization, conditioning)
    assert path == expected and len(digest) == 64


@pytest.mark.parametrize("route,initialization,conditioning", [
    ("ki", "native", "skills"), ("fm", "native", "native_task"),
    ("ar", "a4", "native_task"), ("ar", "native", "oracle"), ("ar", "unknown", "skills")])
def test_undeclared_native_training_combinations_are_rejected(route, initialization, conditioning):
    with pytest.raises(ValueError):
        recipe.declared_parent(route, initialization, conditioning)


def test_native_full_data_recipe_does_not_claim_a4_initialization():
    cfg = OmegaConf.create(dict(tokenizer=dict(vq_config={}),
        model=dict(model_arch=dict(fm={}, AT_CONFIG={}, coordination_train={})) ))
    arch = recipe.configure(cfg, "ar", "native", "native_task")
    assert cfg.model.pretrained_ckpt == str(recipe.NATIVE_PARENT)
    assert arch.action_training.conditioning == "native_task"
    assert not arch.continuous_action and not arch.predict_cot


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
    with pytest.raises(ValueError, match="predeclared"):
        recipe.dependency_identity(recipe.BASE / "ar_native_unapproved", "action")


@pytest.mark.parametrize("name,kind,choice", [
    ("ar_a4_fulltrain_v2", "action", ("ar", "a4", "skills")),
    ("ar_native_task_fulltrain_v1", "action", ("ar", "native", "native_task")),
    ("fm_action_control_v1", "action", ("fm", "a4", "skills")),
    ("joint_a4_fulltrain_v1", "action", ("joint", "a4", "skills")),
    ("fm_ae_lr2x_v1", "fm", "ae_lr2x"),
    ("fm_beta_stratified_v1", "fm", "beta_stratified"),
    ("fm_exec_weight2_v1", "fm", "exec_weight2"),
])
def test_each_screening_predecessor_binds_method_parent_and_finite_budget(monkeypatch, name, kind, choice):
    from train_fm_method_probe import TRIALS
    if kind == "fm":
        method = dict(trial=choice, settings=dict(TRIALS[choice]), max_updates=500, parent_sha256=recipe.PARENT_SHA)
    else:
        route, initialization, conditioning = choice
        method = dict(route=route, initialization=initialization, conditioning=conditioning,
            recipe=deepcopy(recipe.RECIPE), parent_sha256=recipe.declared_parent(*choice)[1])
    monkeypatch.setattr(recipe, "read", lambda p: dict(supervisor_pid=42) if p.name == "launch.json" else method)
    monkeypatch.setattr(recipe, "sha", lambda p: "fixed")
    identity = recipe.dependency_identity(recipe.BASE / name, kind)
    assert identity["supervisor_pid"] == 42 and identity["kind"] == kind
    method["parent_sha256"] = "wrong-parent"
    with pytest.raises(RuntimeError, match="declared finite"):
        recipe.dependency_identity(recipe.BASE / name, kind)


@pytest.mark.parametrize("damage", ["route", "initialization", "budget"])
def test_predecessor_name_alone_cannot_authorize_a_different_native_run(monkeypatch, damage):
    method = dict(route="ar", initialization="native", conditioning="native_task", recipe=deepcopy(recipe.RECIPE),
        parent_sha256=recipe.NATIVE_PARENT_SHA)
    if damage == "budget":
        method["recipe"]["max_updates"] = 5000
    else:
        method[damage] = "ki" if damage == "route" else "a4"
    monkeypatch.setattr(recipe, "read", lambda p: dict(supervisor_pid=42) if p.name == "launch.json" else method)
    with pytest.raises(RuntimeError, match="declared finite"):
        recipe.dependency_identity(recipe.BASE / "ar_native_task_fulltrain_v1", "action")


@pytest.mark.parametrize("kind", ["fm", "action"])
@pytest.mark.parametrize("damage", [None, "running", "steps", "sha", "wrong_checkpoint", "failed_inspection"])
def test_dependency_requires_terminal_complete_and_its_own_exact_500_weight(tmp_path, monkeypatch, kind, damage):
    state = dict(state="complete", verified_optimizer_steps=500, verified_updates=500)
    inspection = dict(passed=True, step=500, actual_updates=500,
        checkpoint=str(tmp_path / "formal/checkpoints/step_500.pt"), checkpoint_sha256="correct")
    if damage == "running":
        state["state"] = "running"
    elif damage == "steps":
        state["verified_optimizer_steps"] = state["verified_updates"] = 499
    elif damage == "sha":
        inspection["checkpoint_sha256"] = "corrupt"
    elif damage == "wrong_checkpoint":
        inspection["checkpoint"] = str(tmp_path / "other/checkpoints/step_500.pt")
    elif damage == "failed_inspection":
        inspection["passed"] = False
    monkeypatch.setattr(recipe, "read", lambda p: state if p.name == "status.json" else inspection)
    monkeypatch.setattr(recipe, "sha", lambda p: "correct")
    if damage:
        with pytest.raises(RuntimeError, match="did not finish"):
            recipe.verify_dependency_completed(tmp_path, kind)
    else:
        result = recipe.verify_dependency_completed(tmp_path, kind)
        assert result.name == ("formal_checkpoint_inspection.json" if kind == "fm" else "checkpoint_inspection.json")


def test_two_independent_queues_cannot_race_for_the_same_gpus():
    with pytest.raises(RuntimeError, match="one serial"):
        recipe.wait_for_dependency(dict(after_fm={"x": 1}, after_action={"x": 2}), None)


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
