from copy import deepcopy

import pytest
import torch

from score_saved_action_groups import action_group_layout, group_scores, summarize_groups


def fixture():
    parts = dict(left_control=9, left_gripper=1, right_control=9, right_gripper=1, lower_body=7)
    pad = torch.zeros(1, 27, dtype=torch.bool)
    pad[:, [7, 8, 17, 18]] = True
    batch = dict(action=torch.zeros(1, 32, 27), action_is_pad=torch.zeros(1, 32, dtype=torch.bool),
        action_dim_is_pad=pad)
    action = torch.ones_like(batch["action"])
    action[:, :, [7, 8, 17, 18]] = 1000.
    return parts, batch, action


def test_metadata_order_defines_groups_instead_of_hardcoded_offsets():
    assert action_group_layout(dict(body=7, hand=1, arm=9), 17) == dict(body=(0, 7), hand=(7, 8), arm=(8, 17))


@pytest.mark.parametrize("parts,total", [({}, 0), ({"arm": True}, 1), ({"arm": -1}, -1),
    ({"arm": 1.5}, 1.5), ({1: 2}, 2), ({"arm": 9}, 27)])
def test_wrong_or_ambiguous_metadata_is_refused(parts, total):
    with pytest.raises(ValueError, match="actual ordered"):
        action_group_layout(parts, total)


def test_group_sum_matches_all23_controls_and_never_counts_padding_errors():
    parts, batch, action = fixture()
    result = group_scores(action, batch, parts, 0, 16)
    assert result["whole"]["valid_scalar_targets"] == 16 * 23
    assert [g["real_dimensions_per_sample"] for g in result["groups"].values()] == [[7], [1], [7], [1], [7]]
    assert [g["valid_scalar_targets"] for g in result["groups"].values()] == [112, 16, 112, 16, 112]
    assert all(g["normalized_rmse"] == 1 for g in result["groups"].values())
    assert sum(g["squared_error_sum"] for g in result["groups"].values()) == result["whole"]["squared_error_sum"]


def test_unexecuted_padding_has_no_targets_and_is_not_reported_as_perfect_prediction():
    parts, batch, action = fixture()
    batch["action_is_pad"][:, 16:] = True
    result = group_scores(action, batch, parts, 16, 32)
    assert all(g["valid_scalar_targets"] == 0 and g["normalized_rmse"] is None for g in result["groups"].values())
    assert not result["whole"]["defined"]


def test_pooling_uses_effective_scalar_counts_not_average_window_rmse():
    parts, batch, action = fixture()
    first = dict(run="model", source=dict(diagnostic_split="train"),
        executed=group_scores(action, batch, parts, 0, 16), unexecuted=group_scores(action, batch, parts, 16, 32))
    short = deepcopy(batch)
    short["action_is_pad"][:, 1:] = True
    second = dict(run="model", source=dict(diagnostic_split="train"),
        executed=group_scores(action * 3, short, parts, 0, 16), unexecuted=group_scores(action * 3, short, parts, 16, 32))
    summary = summarize_groups([first, second])
    left = next(g for g in summary if g["group"] == "left_control" and g["interval"] == "executed")
    assert left["normalized_rmse"] == pytest.approx((25 / 17)**.5)
    assert left["valid_scalar_targets"] == 17 * 7 and left["windows_with_targets"] == 2
    assert left["squared_error_share"] == pytest.approx(7 / 23)


def test_split_and_intervals_remain_separate():
    parts, batch, action = fixture()
    first = dict(run="model", source=dict(diagnostic_split="train"),
        executed=group_scores(action, batch, parts, 0, 16), unexecuted=group_scores(action, batch, parts, 16, 32))
    second = deepcopy(first)
    second["source"]["diagnostic_split"] = "heldout"
    assert len(summarize_groups([first, second])) == 20


def test_group_specific_error_cannot_leak_into_another_control_group():
    parts, batch, action = fixture()
    action[:] = 0
    action[:, :, 20:] = 2
    result = group_scores(action, batch, parts, 0, 16)
    assert result["groups"]["lower_body"]["normalized_rmse"] == 2
    assert all(g["normalized_rmse"] == 0 for k, g in result["groups"].items() if k != "lower_body")
