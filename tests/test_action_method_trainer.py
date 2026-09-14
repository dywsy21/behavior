"""Small contract tests for the full-data action experiment trainer."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import train_action_method_probe as recipe


def test_real_source_launcher_keeps_cpu_hidden_and_restores_only_the_gpu_child(tmp_path, monkeypatch):
    import coordination_launcher as launcher
    from train_fm_method_probe import gpu_training_environment
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    for key in ('MEMLITE_COORDINATION_CONFIG_ONLY', 'MEMLITE_COORDINATION_CONFIG_RESOURCE_RECEIPT',
                'MEMLITE_COORDINATION_TRAINER_COMMAND_JSON'):
        monkeypatch.delenv(key, raising=False)
    plan = dict(source_root=str(launcher.ROOT), config_root=str(launcher.ROOT / 'configs'),
                source_root_sha256='source-digest', config_tree_sha256='config-digest',
                identity=dict(world_size=4))
    cpu = launcher.trainer_config_preflight_environment(plan, tmp_path / 'not-created.json', ['trainer'])
    gpu = gpu_training_environment(launcher, plan)
    assert cpu['CUDA_VISIBLE_DEVICES'] == '' and cpu['MEMLITE_COORDINATION_CONFIG_ONLY'] == '1'
    assert gpu['CUDA_VISIBLE_DEVICES'] == '0,1,2,3' and 'MEMLITE_COORDINATION_CONFIG_ONLY' not in gpu
    assert cpu['PYTHONPATH'] == gpu['PYTHONPATH']
    assert gpu['MEMLITE_COORDINATION_SOURCE_ROOT_SHA256'] == 'source-digest'
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''


def test_fixed_budget_is_an_actual_global_sixteen_batch():
    values = recipe.RECIPE
    assert values["batch_size"] * values["accumulation"] * values["world_size"] == 16
    assert values["max_updates"] == 500 and values["smoke_updates"] == 5
    assert values["history"] == 6 and values["prediction"] == 32 and values["execution"] == [0, 16]
    assert values["workers"] == 4


@pytest.mark.parametrize("initialization,conditioning,expected", [
    ("a4", "skills", recipe.PARENT), ("native", "skills", recipe.NATIVE_PARENT),
    ("native", "native_task", recipe.NATIVE_PARENT), ("native", "native_subtask_cot", recipe.NATIVE_PARENT)])
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


def test_cot_and_eov_supervision_cannot_contaminate_the_original_fm_reference():
    model = SimpleNamespace(discrete_action=True, continuous_action=False, predict_cot=True,
                            processor=SimpleNamespace(pred_eov=True))
    with pytest.raises(ValueError):
        with recipe.original_fm_flags(model):
            assert not model.predict_cot and not model.processor.pred_eov
            raise ValueError("test")
    assert model.predict_cot and model.processor.pred_eov


def test_subtask_cot_full_training_keeps_native_weights_and_explicit_supervision():
    cfg = OmegaConf.create(dict(tokenizer=dict(vq_config={}), model=dict(model_arch=dict(
        fm={}, AT_CONFIG={}, coordination_train={}, input_preprocessor={})) ))
    arch = recipe.configure(cfg, "ar", "native", "native_subtask_cot")
    assert cfg.model.pretrained_ckpt == str(recipe.NATIVE_PARENT)
    assert arch.predict_cot and arch.input_preprocessor.pred_eov and not arch.continuous_action


def test_optional_token_components_use_same_uniform_task_window_averaging():
    rows = [dict(task_id=f"task{i}", ce_loss=1., reference_fm_loss=.2,
                 action_token_ce=float(i), text_boundary_token_ce=0.) for i in range(5)]
    assert recipe.summarize(rows + [deepcopy(rows[-1])] * 3)["aggregate"]["action_token_ce"] == 2.
    del rows[1]["action_token_ce"]
    with pytest.raises(RuntimeError, match="Incomplete token"):
        recipe.summarize(rows)


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


def test_marker_full_recipe_keeps_original_a4_and_small_independent_learning_rate():
    cfg = OmegaConf.create(dict(tokenizer=dict(vq_config={}), model=dict(model_arch=dict(
        fm={}, ar={"use_fused_ce": True}, AT_CONFIG={}, coordination_train={}))))
    arch = recipe.configure(cfg, "ar", marker_rows=True)
    assert cfg.model.pretrained_ckpt == str(recipe.PARENT)
    assert "G05PolicyMEMLiteActionRows" in arch._target_
    assert arch.marker_row_adaptation.train_rows and not arch.ar.use_fused_ce
    assert recipe.expected_adam_states("ar", True) == 193
    assert recipe.MARKER_RECIPE["peak_learning_rate"] == 1e-3
    assert cfg.model.learning_rate == 1e-5 and cfg.model.max_steps == 500


@pytest.mark.parametrize("route,initialization,conditioning,flag", [
    ("joint", "a4", "skills", True), ("ki", "a4", "skills", True),
    ("ar", "native", "native_task", True), ("ar", "native", "native_subtask_cot", True),
    ("ar", "a4", "skills", 1)])
def test_marker_full_recipe_refuses_undeclared_cross_route_combinations(route, initialization, conditioning, flag):
    with pytest.raises(ValueError, match="explicitly pure AR"):
        recipe.validate_marker_variant(route, initialization, conditioning, flag)


def test_marker_optimizer_registers_only_one_small_state_and_preserves_lr_ratio():
    model = torch.nn.Module()
    model.lora = torch.nn.Parameter(torch.ones(2, 2))
    model.model = torch.nn.Module()
    model.model.action_marker_rows = torch.nn.Module()
    model.model.action_marker_rows.delta = torch.nn.Parameter(torch.zeros(8, 2048))
    model.frozen_base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.get_optim_param_groups = lambda **kwargs: [dict(
        params=[p for p in model.parameters() if p.requires_grad], lr=kwargs["lr"], weight_decay=.03)]
    optimizer, scheduler = recipe.make_optimizer(model, marker_rows=True)
    assert scheduler.base_lrs == [1e-5, 1e-3]
    assert [len(group["params"]) for group in optimizer.param_groups] == [1, 1]
    for _ in range(2):
        optimizer.zero_grad()
        (model.lora.sum() + model.model.action_marker_rows.delta.sum()).backward()
        optimizer.step()
        scheduler.step()
    assert len(optimizer.state) == 2
    assert optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"] == pytest.approx(100.)
    assert torch.count_nonzero(model.model.action_marker_rows.delta)
    assert model.frozen_base.grad is None


def test_a4_marker_initialization_refuses_trained_cache_adapter_or_changed_base():
    from g05.models.g05.helpers.action_marker_rows import install_marker_rows, required_marker_ids
    codec = SimpleNamespace(action_token_begin_idx=100, action_token_end_idx=124, _codebook_size=10,
        serializer=SimpleNamespace(nn_key_names=["left", "right", "body"], rule_key_names=["lg", "rg"],
            num_residuals=2, max_residuals=4, group_marker_action_indices={
                "<left_0>": 10, "<right_0>": 11, "<body_0>": 12,
                "<left_1>": 13, "<right_1>": 14, "<body_1>": 15, "<lg>": 22, "<rg>": 23}))
    vlm = torch.nn.Module()
    vlm.input_proj = torch.nn.Embedding(124, 2048)
    vlm.output_proj = torch.nn.Linear(2048, 124, bias=False)
    vlm.output_proj.weight = vlm.input_proj.weight
    adapter = install_marker_rows(vlm, required_marker_ids(codec))
    expected = {f"base_{i}": torch.tensor([float(i)]) for i in range(946)}
    expected.update({f"model.vlm.lora_{i}": torch.ones(1) for i in range(192)})
    actual = {name: value.clone() for name, value in expected.items()}
    actual.update({"model.action_marker_rows." + key: value for key, value in adapter.state_dict().items()})
    model = SimpleNamespace(state_dict=lambda: actual, model=SimpleNamespace(action_marker_rows=adapter),
                            _marker_vlm=lambda: vlm, action_tokenizer=codec)
    parent = dict(step=2500, model_state_dict=expected)
    assert recipe.verify_a4_marker_initialization(model, parent)["cache_adapter_loaded"] is False
    with torch.no_grad():
        adapter.delta[0, 0] = 1.
    with pytest.raises(RuntimeError, match="zero function"):
        recipe.verify_a4_marker_initialization(model, parent)
    with torch.no_grad():
        adapter.delta.zero_()
    actual["base_0"].add_(1.)
    with pytest.raises(RuntimeError, match="not restored exactly"):
        recipe.verify_a4_marker_initialization(model, parent)


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


@pytest.mark.parametrize("name,route", [
    ("fm_action_control_v3", "fm"),
    ("joint_a4_fulltrain_v3", "joint"),
    ("ki_a4_fulltrain_v3", "ki"),
])
@pytest.mark.parametrize("valid_evidence", [True, False])
def test_reference_recovery_predecessor_requires_its_full_evidence(monkeypatch, name, route, valid_evidence):
    method = dict(output=str(recipe.BASE / name), route=route, initialization="a4", conditioning="skills",
                  recipe=deepcopy(recipe.RECIPE), parent_sha256=recipe.PARENT_SHA)
    calls = []

    def validate(spec):
        calls.append(spec)
        if not valid_evidence:
            raise RuntimeError("Reference recovery evidence changed")

    monkeypatch.setattr(recipe, "read", lambda p: dict(supervisor_pid=42) if p.name == "launch.json" else method)
    monkeypatch.setattr(recipe, "sha", lambda p: "fixed")
    monkeypatch.setattr(recipe.reference_gate_recovery, "validate_recovery", validate)
    if valid_evidence:
        identity = recipe.dependency_identity(recipe.BASE / name, "action")
        assert identity["supervisor_pid"] == 42 and identity["root"] == method["output"]
    else:
        with pytest.raises(RuntimeError, match="Reference recovery evidence changed"):
            recipe.dependency_identity(recipe.BASE / name, "action")
    assert calls == [method]
    method["parent_sha256"] = "wrong-parent"
    with pytest.raises(RuntimeError, match="declared finite"):
        recipe.dependency_identity(recipe.BASE / name, "action")


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
