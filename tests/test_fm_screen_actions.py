import pytest
import torch

from probe_fm_screen_actions import interval_metrics, verify_screen_state


def batch():
    pad = torch.zeros(1, 27, dtype=torch.bool)
    pad[:, [7, 8, 17, 18]] = True
    return dict(action=torch.zeros(1, 32, 27), action_is_pad=torch.zeros(1, 32, dtype=torch.bool), action_dim_is_pad=pad)


def test_front_back_metrics_keep_all_23_real_dimensions_and_exclude_padding():
    target = batch()
    pred = torch.ones(1, 32, 27)
    pred[:, 16:] = 2
    pred[:, :, [7, 8, 17, 18]] = 100
    a, b = interval_metrics(pred, target, 0, 16), interval_metrics(pred, target, 16, 32)
    assert a["valid_scalar_targets"] == b["valid_scalar_targets"] == 16 * 23
    assert a["normalized_rmse"] == 1 and b["normalized_rmse"] == 2
    assert not a["weighted"] and not b["weighted"]


def test_unexecuted_padding_is_not_invented_zero_error_evidence():
    target = batch()
    target["action_is_pad"][:, 16:] = True
    score = interval_metrics(torch.ones_like(target["action"]), target, 16, 32)
    assert not score["defined"] and score["normalized_rmse"] is None and score["valid_scalar_targets"] == 0


def test_bad_horizon_masks_or_nonfinite_predictions_fail():
    target = batch()
    with pytest.raises(ValueError):
        interval_metrics(target["action"], target, 5, 37)
    with pytest.raises(ValueError):
        interval_metrics(target["action"] + float("nan"), target, 0, 16)
    target["action_dim_is_pad"] = target["action_dim_is_pad"].float()
    with pytest.raises(ValueError):
        interval_metrics(target["action"], target, 0, 16)


def test_partial_or_different_step_model_is_not_a_deployable_screen_checkpoint():
    with pytest.raises(RuntimeError, match="complete 500-step"):
        verify_screen_state(torch.nn.Linear(2, 2), dict(step=5, model_state_dict={}))
