from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import probe_m04_actions as probe


def batch():
    pad = torch.zeros(1, 27, dtype=torch.bool)
    pad[:, [7, 8, 17, 18]] = True
    return dict(samples=[dict(command="task", proprio=dict(value=torch.zeros(6, 27)))],
        pixel_values={"exterior": torch.zeros(1, 6, 3, 4, 4)}, action_dim_is_pad=pad,
        action=torch.full((1, 32, 27), 999.), action_is_pad=torch.zeros(1, 32, dtype=torch.bool))


def test_finite_output_names_do_not_allow_old_screen_repeats_or_undeclared_arms():
    assert len({probe.output_path(name) for name in probe.ARMS}) == 3
    assert probe.output_path("ki_a4_fulltrain_v3").name == "m04_actions_ki_v1"
    for name in ("fm_control_v1", "ar_a4_fulltrain_v2", "ki_a4_fulltrain_v4"):
        with pytest.raises(ValueError, match="three declared"):
            probe.output_path(name)


@pytest.mark.parametrize("route,name", [(r, n) for n, (r, _) in probe.ARMS.items()])
@pytest.mark.parametrize("damage", [None, "adam", "parent", "rows", "frozen", "steps", "split", "conditioning"])
def test_only_completed_original_five_task_full500_weights_qualify(route, name, damage):
    method = dict(route=route, initialization="a4", conditioning="skills", marker_rows=False,
        recipe=deepcopy(probe.RECIPE), parent_sha256=probe.PARENT_SHA,
        train_split="original950", eval_split="original50")
    inspection = dict(passed=True, actual_updates=500, actual_adam_states=probe.expected_adam_states(route),
        actual_train_rows=8000, frozen_unchanged=True, full_model_optimizer_rng_roundtrip=True)
    if damage == "adam":
        inspection["actual_adam_states"] = 192
    elif damage == "parent":
        method["parent_sha256"] = "different-parent"
    elif damage == "rows":
        inspection["actual_train_rows"] = 10
    elif damage == "frozen":
        inspection["frozen_unchanged"] = False
    elif damage == "steps":
        inspection["actual_updates"] = 499
    elif damage == "split":
        method["train_split"] = "includes-heldout"
    elif damage == "conditioning":
        method["conditioning"] = "native_task"
    if damage:
        with pytest.raises(RuntimeError, match="exact completed"):
            probe.validate_training(name, method, inspection)
    else:
        probe.validate_training(name, method, inspection)


@pytest.mark.parametrize("route", ["fm", "joint", "ki"])
def test_continuous_path_calls_fm_once_and_never_passes_outer_teacher_targets(route):
    data, recorded = batch(), []
    expected = torch.zeros(1, 32, 27)

    def fm(**kwargs):
        recorded.append(kwargs)
        return expected

    original_ar = lambda **kwargs: (_ for _ in ()).throw(AssertionError("AR called"))
    model = SimpleNamespace(training=False, predict_cot=False, discrete_action=route != "fm", continuous_action=True,
        model=SimpleNamespace(inference_ar=original_ar, inference_fm=fm),
        action_training=SimpleNamespace(route=route, conditioning="skills"))

    def call(samples, pixels, **kwargs):
        assert samples is data["samples"] and pixels is data["pixel_values"]
        assert kwargs == {"action_dim_is_pad": data["action_dim_is_pad"]}
        return dict(action=model.model.inference_fm(**kwargs), selected_action_source="fm")

    model.forward_inference = call
    model.prefill = lambda samples, pixels: (samples, pixels)

    def generate(state, samples, *, action_dim_is_pad, action_gt):
        assert state == (samples, data["pixel_values"]) and action_gt is None
        return call(samples, data["pixel_values"], action_dim_is_pad=action_dim_is_pad)

    model.generate_action = generate
    result, calls = probe.infer_continuous(model, route, data)
    assert result is expected and calls == dict(fm=1, ar=0) and len(recorded) == 1
    assert model.model.inference_fm is fm and model.model.inference_ar is original_ar


def test_ar_guard_restores_methods_even_when_a_wrong_path_is_called():
    ar, fm = lambda: None, lambda: None
    model = SimpleNamespace(model=SimpleNamespace(inference_ar=ar, inference_fm=fm))
    with pytest.raises(RuntimeError, match="must not generate AR"):
        with probe.count_continuous_only(model) as calls:
            model.model.inference_ar()
    assert calls == dict(fm=0, ar=1)
    assert model.model.inference_ar is ar and model.model.inference_fm is fm


@pytest.mark.parametrize("damage", ["base_mask", "teacher", "history_batch", "training", "cot"])
def test_actor_contract_refuses_suppressed_controls_or_teacher_fields_before_neural_calls(damage):
    data = batch()
    model = SimpleNamespace(training=False, predict_cot=False)
    if damage == "base_mask":
        data["action_dim_is_pad"][0, -1] = True
    elif damage == "teacher":
        data["samples"][0]["action"] = data["action"]
    elif damage == "history_batch":
        data["samples"] *= 2
    elif damage == "training":
        model.training = True
    elif damage == "cot":
        model.predict_cot = True
    with pytest.raises(ValueError):
        probe.infer_continuous(model, "ki", data)


def test_processed_input_fingerprint_keeps_dtype_shape_and_exact_values():
    value = dict(value=torch.arange(12, dtype=torch.bfloat16).view(3, 4), mask=torch.zeros(27, dtype=torch.bool))
    original = probe.tensor_fingerprint(value)
    assert original == probe.tensor_fingerprint(deepcopy(value))
    assert original["value"]["dtype"] == "torch.bfloat16" and original["value"]["shape"] == [3, 4]
    value["value"][0, 0] = 1
    assert original["value"]["sha256"] != probe.tensor_fingerprint(value)["value"]["sha256"]
    with pytest.raises(ValueError, match="actual processed"):
        probe.tensor_fingerprint("a-seed-is-not-a-pixel-fingerprint")
