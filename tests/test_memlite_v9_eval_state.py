"""Periodic eval must exercise MEM history and leave train state intact."""

import pytest
import torch
from contextlib import nullcontext
from types import SimpleNamespace

from g05.utils.training.accuracy_accumulator import TrainAccuracyAccumulator
from utils.train_eval import PeriodicEvaluator, evaluation_state
from utils import train_eval
from utils.metric import rollout_and_calculate_metrics
from utils import metric
from g05.utils.memlite_protocol import ActionDecodeError
from g05.models.g05.g05_policy import G05Policy


def policy():
    model = torch.nn.Sequential(torch.nn.Dropout(0.9), torch.nn.Linear(2, 2))
    model.train()
    model[1].eval()
    model._train_acc = TrainAccuracyAccumulator()
    model._train_acc.push("overall", 0.25)
    model.train_action_accuracy = 0.25
    return model


@pytest.mark.parametrize("fail", [False, True])
def test_eval_mode_and_metric_state_restored_on_success_and_error(fail):
    model = policy()
    accumulator = model._train_acc
    try:
        with evaluation_state(model):
            assert all(not module.training for module in model.modules())
            assert torch.equal(model[0](torch.ones(40)), torch.ones(40))
            model._train_acc.push("overall", 1.0)
            model.train_action_accuracy = 1.0
            model.train_cot_accuracy = 1.0
            if fail:
                raise RuntimeError("infrastructure failure must propagate")
    except RuntimeError as error:
        assert fail and "infrastructure" in str(error)
    assert model.training and model[0].training and not model[1].training
    assert model._train_acc is accumulator
    assert accumulator.flush() == {"overall": 0.25}
    assert model.train_action_accuracy == 0.25
    assert not hasattr(model, "train_cot_accuracy")


def test_periodic_evaluator_uses_eval_context(monkeypatch, tmp_path):
    model = policy()
    observed = []

    def metrics(batch, passed_model, *args, **kwargs):
        assert passed_model is model
        assert all(not module.training for module in model.modules())
        model._train_acc.push("overall", 0.9)
        observed.append(batch)
        return {"teacher_forcing/high_ce_loss": 0.2, "teacher_forcing/low_ce_loss": 2.0}, {}, None

    monkeypatch.setattr(train_eval, "rollout_and_calculate_metrics", metrics)
    monkeypatch.setattr(train_eval.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(train_eval, "save_eval_snapshot", lambda *args, **kwargs: None)
    evaluator = PeriodicEvaluator([{"id": 1}], None, None, None, tmp_path)
    result = evaluator.evaluate(model, None, 100)
    assert observed == [{"id": 1}]
    assert result["eval/action/teacher_forcing/high_ce_loss"] == 0.2
    assert result["eval/action/teacher_forcing/low_ce_loss"] == 2.0
    assert model.training and not model[1].training
    assert model._train_acc.flush() == {"overall": 0.25}


def test_existing_eval_mode_remains_eval():
    model = policy().eval()
    with evaluation_state(model):
        assert not model.training
    assert all(not module.training for module in model.modules())


def test_metric_reports_actual_branch_ce_not_mixed_ce_under_low_name():
    class Model:
        training = False
        train_action_accuracy = 0.5
        train_cot_accuracy = 0.9

        def __call__(self, batch, inference_mode=False):
            if inference_mode:
                assert [row["memlite_branch"] for row in batch["samples"]] == ["low"]
                return {"action": batch["action"].clone()}
            return torch.tensor(1.1), {
                "ce_loss": torch.tensor(1.1),
                "high_ce_loss": torch.tensor(0.2),
                "low_ce_loss": torch.tensor(2.0),
            }

    batch = {
        "samples": [{"memlite_branch": "high"}, {"memlite_branch": "low"}],
        "action": torch.zeros(2, 2, 1),
        "action_dim_is_pad": torch.zeros(2, 1, dtype=torch.bool),
        "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
        "embodiment": ["r1", "r1"],
    }
    metrics = rollout_and_calculate_metrics(
        batch, model=Model(), accelerator=SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext),
    )
    assert metrics["teacher_forcing/high_ce_loss"] == pytest.approx(0.2)
    assert metrics["teacher_forcing/low_ce_loss"] == pytest.approx(2.0)
    assert metrics["teacher_forcing/ar_action_ce_loss"] == pytest.approx(1.1)


def decode_failure_model(error):
    class Model:
        training = False
        train_action_accuracy = 0.5
        train_cot_accuracy = 0.9

        def __call__(self, batch, inference_mode=False):
            if inference_mode:
                assert len(batch["samples"]) == 2
                assert all(row["memlite_branch"] == "low" for row in batch["samples"])
                raise error
            return torch.tensor(1.1), {"ce_loss": torch.tensor(1.1),
                                       "high_ce_loss": torch.tensor(0.2),
                                       "low_ce_loss": torch.tensor(2.0)}
    return Model()


def decode_failure_batch():
    return {"samples": [{"memlite_branch": "high"}, {"memlite_branch": "low"},
                        {"memlite_branch": "low"}],
            "action": torch.zeros(3, 2, 1),
            "action_dim_is_pad": torch.zeros(3, 1, dtype=torch.bool),
            "embodiment": ["r1"] * 3}


def test_decode_failure_retains_ce_but_never_scores_zeros_or_high_rows():
    result, per_emb, predictions = rollout_and_calculate_metrics(
        decode_failure_batch(), model=decode_failure_model(ActionDecodeError("missing markers")),
        accelerator=SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext),
        return_per_emb_raw=True, return_preds=True,
    )
    assert result["teacher_forcing/high_ce_loss"] == pytest.approx(0.2)
    assert result["teacher_forcing/low_ce_loss"] == pytest.approx(2.0)
    assert result["rollout/attempted_sample_count"] == 2
    assert result["rollout/attempted_batch_count"] == 1
    assert result["rollout/decode_failed_batch_count"] == 1
    assert result["rollout/unscored_due_to_decode_sample_count"] == 2
    assert result["rollout/scored_sample_count"] == 0
    assert not any("l1" in key or key.startswith("rollout/ar_action_acc") for key in result)
    assert per_emb == {} and predictions is None


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), OSError("I/O failure"),
                                   ValueError("unexpected programming defect")])
def test_infrastructure_and_unexpected_errors_still_propagate(error):
    with pytest.raises(type(error), match=str(error)):
        rollout_and_calculate_metrics(
            decode_failure_batch(), model=decode_failure_model(error),
            accelerator=SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext),
        )


def test_different_rank_decode_outcomes_sum_counts_without_faking_action_scores(monkeypatch):
    failed = {"teacher_forcing/low_ce_loss": 2.0,
              "rollout/attempted_sample_count": 7, "rollout/attempted_batch_count": 1,
              "rollout/decode_failed_batch_count": 1, "rollout/scored_sample_count": 0,
              "rollout/unscored_due_to_decode_sample_count": 7}
    valid = {"teacher_forcing/low_ce_loss": 4.0,
             "rollout/attempted_sample_count": 7, "rollout/attempted_batch_count": 1,
             "rollout/decode_failed_batch_count": 0, "rollout/scored_sample_count": 7,
             "rollout/unscored_due_to_decode_sample_count": 0,
             "rollout/ar_action_l1": 0.4}
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda result, data: result.__setitem__(slice(None), [failed, valid]))
    result = metric.reduce_payload(failed)
    assert result["teacher_forcing/low_ce_loss"] == 3.0
    assert result["rollout/attempted_sample_count"] == 14
    assert result["rollout/attempted_batch_count"] == 2
    assert result["rollout/decode_failed_batch_count"] == 1
    assert result["rollout/unscored_due_to_decode_sample_count"] == 7
    assert result["rollout/scored_sample_count"] == 7
    assert result["rollout/ar_action_l1"] == 0.4  # only scored rows, not zeros for failures


def inference_shell():
    policy = object.__new__(G05Policy)
    policy.discrete_action = True
    policy.continuous_action = False
    policy.predict_cot = True
    policy.memlite_train_mode = "mixed"
    policy.prefill = lambda *args, **kwargs: pytest.fail("legacy prefill must not run for MEM-Lite eval")
    policy.generate_text = lambda *args, **kwargs: pytest.fail("LL action eval must not generate CoT")
    return policy


def test_memlite_periodic_inference_uses_deployed_ll_api_with_exact_intents():
    policy = inference_shell()
    samples = [{"memlite_branch": "low", "intent": "pick up can"},
               {"memlite_branch": "low", "intent": "close cabinet"}]
    pixels = {"head": torch.zeros(2, 6, 3, 4, 4)}
    mask = torch.zeros(2, 1, dtype=torch.bool)
    gt = torch.ones(2, 2, 1)
    captured = {}

    def low(samples_arg, pixels_arg, **kwargs):
        assert samples_arg is samples and pixels_arg is pixels
        captured.update(kwargs)
        return {"action": gt, "selected_action_source": "ar"}

    policy.generate_low_level_action = low
    result = policy.forward_inference(samples, pixels, actions=gt, action_dim_is_pad=mask)
    assert captured["intent_text"] == ["pick up can", "close cabinet"]
    assert captured["action_gt"] is gt and captured["action_dim_is_pad"] is mask
    assert result["selected_action_source"] == "ar"


@pytest.mark.parametrize("sample", [{"memlite_branch": "high", "intent": "x"},
                                     {"memlite_branch": "low", "intent": ""},
                                     {"memlite_branch": "low"}])
def test_memlite_action_inference_rejects_high_rows_or_missing_intent(sample):
    policy = inference_shell()
    policy.generate_low_level_action = lambda *args, **kwargs: pytest.fail("invalid condition must be rejected")
    with pytest.raises(ValueError):
        policy.forward_inference([sample], {})


def test_non_mem_policy_retains_legacy_inference_entry():
    policy = inference_shell()
    policy.memlite_train_mode = "off"

    def legacy(*args, **kwargs):
        raise RuntimeError("legacy prefill reached")

    policy.prefill = legacy
    policy.generate_low_level_action = lambda *args, **kwargs: pytest.fail("non-MEM policy must not use LL branch")
    with pytest.raises(RuntimeError, match="legacy prefill reached"):
        policy.forward_inference([{}], {})
