"""Numerical diagnostic summary tests; actual model comparison runs on robo."""
import pytest
import torch

from probe_ar_decode_consistency import compare_logits


def test_identical_logits_and_tied_rank():
    value = torch.tensor([0., 2., 2., -1.])
    result = compare_logits(value, value.clone(), 2)
    assert result["argmax_equal"] and result["max_abs_diff"] == 0
    assert result["rms_diff"] == 0 and result["full_target_rank"] == 1
    assert result["full_target_ce"] == result["cached_target_ce"]


def test_rank_argmax_and_loss_changes_are_not_hidden_by_a_boolean_pass():
    a, b = torch.tensor([5., 1., 0.]), torch.tensor([1., 0., 5.])
    result = compare_logits(a, b, 2)
    assert not result["argmax_equal"] and result["max_abs_diff"] == 5
    assert result["full_target_rank"] == 3 and result["cached_target_rank"] == 1
    assert result["cached_target_ce"] < result["full_target_ce"]


@pytest.mark.parametrize("actual,target", [(torch.ones(2), 0), (torch.tensor([1., float("nan"), 2.]), 0),
    (torch.ones(3), -1), (torch.ones(3), 3)])
def test_corrupt_shape_numerics_or_target_rejected(actual, target):
    with pytest.raises(ValueError):
        compare_logits(torch.ones(3), actual, target)
