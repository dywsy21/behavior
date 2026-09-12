import pytest
import torch

from g05.models.g05.helpers.action_group_loss import ActionGroupLoss


def helper(weight=2):
    return ActionGroupLoss(100, 4, {"<left_control_0>": 104, "<lower_body_0>": 105,
                                  "<right_gripper>": 106, "<lower_body_1>": 107}, weight)


def test_group_markers_text_and_batch_boundaries():
    y = torch.tensor([[105, 100, 101, 106, 100, 10, 100, -100],
                      [100, 104, 101, 107, 103, 102, -100, -100]])
    weights, lower = helper().masks(y)
    assert lower.tolist() == [[False, True, True, False, False, False, False, False],
                              [False, False, False, False, True, True, False, False]]
    assert torch.equal(weights, 1 + lower.float())


def test_weighted_ce_gradient_and_plain_ce_are_distinct():
    labels = torch.tensor([[105, 100, 101, 104, 100, -100]])
    losses = torch.arange(1., 6., requires_grad=True)
    objective, weights, _ = helper().reduce(losses, labels)
    torch.testing.assert_close(objective, (losses * torch.tensor([1., 2., 2., 1., 1.])).sum() / 7)
    objective.backward()
    torch.testing.assert_close(losses.grad, weights / 7)


def test_unit_weights_preserve_original_objective_and_gradient():
    labels = torch.tensor([[105, 100, -100, 10]])
    losses = torch.tensor([.1, 3., 2.], requires_grad=True)
    objective, _, _ = helper(1).reduce(losses, labels)
    torch.testing.assert_close(objective, losses.mean())
    objective.backward()
    torch.testing.assert_close(losses.grad, torch.full_like(losses, 1/3))


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_weights_fail(weight):
    with pytest.raises(ValueError):
        helper(weight)
