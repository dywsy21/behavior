from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import probe_ar_marker_actions as probe
from test_m04_actions import batch


@pytest.mark.parametrize("damage", [None, "route", "parent", "markers", "marker_recipe", "recipe",
                                   "split", "conditioning", "step", "draws", "adam", "frozen", "save"])
def test_final_marker_recipe_requires_real_full500_completion(damage):
    method = dict(route="ar", initialization="a4", conditioning="skills", marker_rows=True,
        recipe=deepcopy(probe.RECIPE), marker_recipe=deepcopy(probe.MARKER_RECIPE),
        parent_sha256=probe.PARENT_SHA, train_split="original950", eval_split="original50")
    inspection = dict(passed=True, actual_updates=500, actual_train_rows=8000, actual_adam_states=193,
        frozen_unchanged=True, full_model_optimizer_rng_roundtrip=True)
    if damage in {"recipe", "marker_recipe"}:
        method[damage] = {}
    elif damage == "route":
        method["route"] = "ki"
    elif damage == "parent":
        method["initialization"] = "native"
    elif damage == "markers":
        method["marker_rows"] = False
    elif damage == "split":
        method["eval_split"] = "recycled"
    elif damage == "conditioning":
        method["conditioning"] = "native_subtask_cot"
    elif damage:
        key = dict(step="actual_updates", draws="actual_train_rows", adam="actual_adam_states",
                   frozen="frozen_unchanged", save="full_model_optimizer_rng_roundtrip")[damage]
        inspection[key] = 0
    if damage:
        with pytest.raises(RuntimeError, match="exact completed"):
            probe.validate_training(method, inspection)
    else:
        probe.validate_training(method, inspection)


def test_runtime_waiver_is_exactly_one_unused_registration_not_other_source_changes():
    import hashlib
    old = b"a\nb\n"
    new = b"a\n" + probe.EMA_LINE + b"b\n"
    assert probe.original_runtime_digest(new) == hashlib.sha256(old).hexdigest()
    assert probe.original_runtime_digest(new + b"unexpected_edit\n") != hashlib.sha256(old).hexdigest()
    for bad in (old, new + probe.EMA_LINE):
        with pytest.raises(RuntimeError, match="exactly one"):
            probe.original_runtime_digest(bad)


def marker_state():
    state = {f"model.vlm.lora_{i}": torch.tensor([i], dtype=torch.float32) for i in range(192)}
    state.update({f"frozen.{i}": torch.tensor([i], dtype=torch.float32) for i in range(946)})
    state["model.action_marker_rows.token_ids"] = torch.tensor(probe.MARKER_IDS)
    state["model.action_marker_rows.delta"] = torch.ones(8, 2048)
    return state


@pytest.mark.parametrize("damage", [None, "missing", "step", "delta_zero", "delta_nan", "ids", "lora", "frozen"])
def test_full_restoration_compares_every_tensor_and_both_trained_marker_entries(damage):
    actual = marker_state()
    saved = dict(step=500, model_state_dict=deepcopy(actual))
    if damage == "missing":
        del actual["model.action_marker_rows.delta"]
    elif damage == "step":
        saved["step"] = 400
    elif damage == "delta_zero":
        actual["model.action_marker_rows.delta"].zero_()
    elif damage == "delta_nan":
        actual["model.action_marker_rows.delta"][0, 0] = float("nan")
    elif damage == "ids":
        actual["model.action_marker_rows.token_ids"][0] += 1
    elif damage == "lora":
        actual["model.vlm.lora_191"] += 1
    elif damage == "frozen":
        actual["frozen.945"] += 1
    model = SimpleNamespace(state_dict=lambda: actual)
    if damage:
        with pytest.raises(RuntimeError):
            probe.verify_marker_state(model, saved)
    else:
        probe.verify_marker_state(model, saved)


def actor(data, failure=None):
    def ar(**kwargs):
        return dict(generated_ids=torch.tensor([[252173, 91]]))

    def never(*args, **kwargs):
        raise AssertionError("Forbidden original FM/teacher encode was reached")

    model = SimpleNamespace(training=False, predict_cot=False, discrete_action=True, continuous_action=False,
        action_training=SimpleNamespace(route="ar", conditioning="skills"),
        model=SimpleNamespace(inference_ar=ar, inference_fm=never),
        action_tokenizer=SimpleNamespace(_encode_action_indices=never))

    def forward(samples, pixels, **kwargs):
        assert samples is data["samples"] and pixels is data["pixel_values"]
        assert set(kwargs) == {"action_dim_is_pad"}
        model.model.inference_ar(max_new_tokens=300, stop_token_ids=[91])
        if failure:
            raise RuntimeError(failure)
        return dict(action=torch.zeros(1, 32, 27), selected_action_source="ar",
                    execution_start=0, execution_steps=16, ar_complete_block_receipts=[{"complete": True}])

    model.forward_inference = forward
    return model


@pytest.mark.parametrize("failure", [None, "Free AR generation omitted action groups; no GT or FM fallback is allowed",
                                    "Free AR generation has incomplete or malformed codec blocks: repeated body1"])
def test_one_actual_free_call_and_invalid_outputs_are_not_filled(failure):
    data = batch()
    model = actor(data, failure)
    original = (model.model.inference_ar, model.model.inference_fm, model.action_tokenizer._encode_action_indices)
    prediction, report = probe.infer_free_ar(model, data)
    assert (model.model.inference_ar, model.model.inference_fm, model.action_tokenizer._encode_action_indices) == original
    assert report["valid"] == (failure is None) and report["failure"] == failure
    assert (prediction is None) == (failure is not None)
    assert report["calls"] == dict(ar=1, fm=0, teacher_encodes=0,
        calls=[dict(ids=[[252173, 91]], max_new_tokens=300, stop_token_ids=[91])])


def test_unexpected_model_errors_are_not_counted_as_format_failures():
    data = batch()
    with pytest.raises(RuntimeError, match="CUDA bug"):
        probe.infer_free_ar(actor(data, "CUDA bug"), data)


@pytest.mark.parametrize("kind", ["fm", "encode", "schema", "tokens", "second_call"])
def test_observer_stops_fallback_teacher_encoding_schema_budget_changes_and_duplicate_calls(kind):
    model = actor(batch())
    original = (model.model.inference_ar, model.model.inference_fm, model.action_tokenizer._encode_action_indices)
    with pytest.raises(RuntimeError):
        with probe.observe_free_ar(model):
            if kind == "fm":
                model.model.inference_fm()
            elif kind == "encode":
                model.action_tokenizer._encode_action_indices()
            elif kind == "schema":
                model.model.inference_ar(max_new_tokens=300, logits_processor=object())
            elif kind == "tokens":
                model.model.inference_ar(max_new_tokens=301)
            else:
                model.model.inference_ar(max_new_tokens=300)
                model.model.inference_ar(max_new_tokens=300)
    assert (model.model.inference_ar, model.model.inference_fm, model.action_tokenizer._encode_action_indices) == original


@pytest.mark.parametrize("kind", ["mask", "teacher", "training", "cot", "fm", "native"])
def test_bad_actor_contract_stops_before_any_neural_call(kind):
    data = batch()
    model = actor(data)
    if kind == "mask":
        data["action_dim_is_pad"][0, 24] = True
    elif kind == "teacher":
        data["samples"][0]["action"] = data["action"]
    elif kind == "training":
        model.training = True
    elif kind == "cot":
        model.predict_cot = True
    elif kind == "fm":
        model.continuous_action = True
    elif kind == "native":
        model.action_training.conditioning = "native_task"
    with pytest.raises(ValueError):
        probe.infer_free_ar(model, data)


def test_pairing_checks_processed_pixels_not_only_source_labels():
    data = batch()
    baseline = dict(pixel_fingerprints=probe.tensor_fingerprint(data["pixel_values"]),
        proprio_fingerprints=probe.tensor_fingerprint(data["samples"][0]["proprio"]),
        mask_fingerprints=probe.tensor_fingerprint({k: data[k] for k in ("action_is_pad", "action_dim_is_pad")}))
    assert probe.paired_input_fingerprints(data, baseline) == baseline
    data["pixel_values"]["exterior"][0, 0, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="processed inputs differ"):
        probe.paired_input_fingerprints(data, baseline)


def test_cached_reduction_tolerance_is_not_permission_to_change_targets_or_support():
    metrics = probe.interval_metrics(torch.zeros(1, 32, 27), batch(), 0, 16)
    other = deepcopy(metrics)
    other["squared_error_sum"] += 1e-8
    probe.verify_cached_score(metrics, other)
    other["squared_error_sum"] += 1
    with pytest.raises(RuntimeError):
        probe.verify_cached_score(metrics, other)
    other = deepcopy(metrics)
    other["valid_scalar_targets"] -= 1
    with pytest.raises(RuntimeError, match="support"):
        probe.verify_cached_score(metrics, other)


@pytest.mark.parametrize("valid_count", [0, 2, 5])
def test_pooled_scores_use_identical_valid_subset_and_report_failures(valid_count):
    rows = []
    for split in ("train", "heldout"):
        for i in range(5):
            row = dict(label=f"{split}{i}", source=dict(diagnostic_split=split), valid=i < valid_count)
            for interval in ("executed", "unexecuted"):
                for prefix in ("", "fm_", "a4_"):
                    row[prefix + interval] = dict(valid_scalar_targets=2, squared_error_sum=2 * (i + 1)**2)
            if not row["valid"]:
                row["executed"] = row["unexecuted"] = None
            rows.append(row)
    pools = probe.pooled_comparison(rows)
    for split, report in pools.items():
        assert report["attempts"] == 5 and report["invalid"] == 5 - valid_count
        assert report["paired_labels"] == [f"{split}{i}" for i in range(valid_count)]
        scores = report["paired_valid_subset"]
        assert scores["executed"] == scores["fm_executed"] == scores["a4_executed"]
        assert scores["executed"]["valid_scalar_targets"] == 2 * valid_count
        assert (scores["executed"]["normalized_rmse"] is None) == (valid_count == 0)
