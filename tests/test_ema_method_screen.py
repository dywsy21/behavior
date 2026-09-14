from copy import deepcopy

import pytest

import ema_method_screen as screen


def candidate():
    return dict(output=str(screen.BASE / screen.RUN), route="fm", initialization="a4", conditioning="skills",
        marker_rows=False, after_fm=None, recovery_from=None, reference_gate_recovery=None,
        ema_recipe=deepcopy(screen.RECIPE), after_action=dict(kind="action", root=str(screen.BASE / screen.COT_RUN),
            method_sha256=screen.COT_SPEC_SHA))


def test_one_new_ema_screen_and_unmodified_old_recipes():
    screen.validate_screen(candidate())
    screen.validate_screen(dict(output=str(screen.BASE / "fm_action_control_v3")))
    assert screen.RECIPE["beta"] == .99
    assert screen.RECIPE["copy_average_back_to_training"] is False


@pytest.mark.parametrize("key,value", [("route", "ki"), ("initialization", "native"),
    ("conditioning", "native_task"), ("marker_rows", True), ("marker_rows", None),
    ("after_fm", {"root": "another"}), ("recovery_from", {"previous": "old"}),
    ("reference_gate_recovery", {"reuse": True}), ("output", str(screen.BASE / "fm_ema_other")),
    ("ema_recipe", None)])
def test_rejects_undeclared_combinations_or_hidden_control(key, value):
    spec = candidate()
    spec[key] = value
    with pytest.raises(RuntimeError):
        screen.validate_screen(spec)


@pytest.mark.parametrize("key,value", [("beta", .999), ("update_every", 10),
    ("initialization", "first_online"), ("copy_average_back_to_training", True),
    ("per_process_gpu_memory_fraction", .95), ("expected_parameter_tensors", 504)])
def test_recipe_is_fixed_and_does_not_confuse_adam_with_all_trainable_tensors(key, value):
    spec = candidate()
    spec["ema_recipe"][key] = value
    with pytest.raises(RuntimeError):
        screen.validate_screen(spec)


@pytest.mark.parametrize("key,value", [("root", str(screen.BASE / "ar_a4_marker_fulltrain_v3")),
    ("kind", "fm"), ("method_sha256", "f" * 64)])
def test_does_not_bypass_existing_cot_queue(key, value):
    spec = candidate()
    spec["after_action"][key] = value
    with pytest.raises(RuntimeError):
        screen.validate_screen(spec)


def inspection():
    return dict(passed=True, actual_updates=5, ema=dict(num_updates=5, parameter_tensors=514,
        full_state_roundtrip=True, online_restoration_checks=1))


def test_requires_own_actual_ema_smoke_not_an_old_fm_gate():
    screen.validate_smoke(inspection())
    with pytest.raises(RuntimeError):
        screen.validate_smoke(dict(passed=True, actual_updates=5))


@pytest.mark.parametrize("key,value", [("num_updates", 0), ("parameter_tensors", 504),
    ("full_state_roundtrip", False), ("online_restoration_checks", 0)])
def test_shadow_and_clock_must_both_be_valid_before_formal(key, value):
    result = inspection()
    result["ema"][key] = value
    with pytest.raises(RuntimeError):
        screen.validate_smoke(result)
