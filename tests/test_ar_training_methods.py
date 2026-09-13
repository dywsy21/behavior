"""Rendering/target contract tests; no model or simulator construction."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from g05.utils.training.ar_training_methods import (
    ActionTrainingSettings, action_prefix_samples, action_training_samples,
)


def fixture():
    samples = [dict(template=("<chat_user_prefix><image0_image_!><bos>"
        "Embodiment: <embodiment_text_!>; Task: <command_text_!_200>; "
        "Parent goal: <parent_goal_text_!>; <active_skills_text_text_!> State: <proprio_proprio_!>;"
        "<chat_user_suffix><chat_assistant_prefix>Action: <EOV><EOC><eos>"),
        command="turn on the radio", parent_goal="grasp the radio", active_skills_text="Active skills: GRASP",
        active_skills_semantic_json="[]", proprio=dict(value=torch.randn(6, 27))) for _ in range(2)]
    actions = torch.randn(2, 32, 27)
    temporal = torch.zeros(2, 32).bool()
    dimensions = torch.zeros(2, 27).bool()
    dimensions[:, [7, 8, 17, 18]] = True
    return samples, actions, temporal, dimensions


def test_full_targets_keep_all_controls_and_do_not_mutate_inputs():
    samples, actions, temporal, dimensions = fixture()
    before, values = deepcopy(samples), actions.clone()
    result = action_training_samples(samples, actions, temporal, dimensions, ActionTrainingSettings())
    for i, row in enumerate(result):
        assert torch.equal(row["action"]["value"], actions[i])
        assert row["action"]["action_op_mask"].sum() == 23
        assert row["action"]["action_op_mask"][20:27].all()
        assert row["template"].count("<action_action>") == 1
        assert row["template"].split("<EOC>")[0] == samples[i]["template"].split("<EOC>")[0]
        assert "action" not in samples[i]
        assert torch.equal(samples[i]["proprio"]["value"], before[i]["proprio"]["value"])
    result[0]["action"]["value"].fill_(999)
    assert torch.equal(actions, values)


def test_executed_prefix_padding_ignores_future_and_real_temporal_pad():
    samples, actions, temporal, dimensions = fixture()
    temporal[1, 5:] = True
    settings = ActionTrainingSettings(codec_mode="prefix16_holdpad32")
    result = action_training_samples(samples, actions, temporal, dimensions, settings)
    actions[0, 16:] = 999
    actions[1, 5:] = 999
    other = action_training_samples(samples, actions, temporal, dimensions, settings)
    for a, b in zip(result, other):
        assert torch.equal(a["action"]["value"], b["action"]["value"])


def test_task_only_removes_skill_inputs_but_preserves_observed_state_and_task():
    samples, *_ = fixture()
    result = action_prefix_samples(samples, ActionTrainingSettings(conditioning="task"))
    for original, row in zip(samples, result):
        assert "action" not in row
        assert "parent_goal" not in row and "active_skills_text" not in row
        assert "parent_goal_text" not in row["template"] and "active_skills_text" not in row["template"]
        assert row["command"] == original["command"]
        assert torch.equal(row["proprio"]["value"], original["proprio"]["value"])


@pytest.mark.parametrize("key", ["action", "gt_action", "future_state", "teacher_action"])
def test_target_free_rendering_rejects_teacher_fields(key):
    samples, *_ = fixture()
    samples[0][key] = torch.ones(32, 27)
    with pytest.raises(ValueError):
        action_prefix_samples(samples, ActionTrainingSettings())


def test_bad_mask_or_template_fails_without_silently_discarding_controls():
    samples, actions, temporal, dimensions = fixture()
    dimensions[:, 24:27] = True
    with pytest.raises(ValueError, match="23 controls"):
        action_training_samples(samples, actions, temporal, dimensions, ActionTrainingSettings())
    dimensions[:, 24:27] = False
    with pytest.raises(ValueError, match="temporal"):
        action_training_samples(samples, actions, None, dimensions, ActionTrainingSettings())
    samples[0]["template"] += "extra_unchecked_suffix"
    with pytest.raises(ValueError, match="template"):
        action_prefix_samples(samples, ActionTrainingSettings())


@pytest.mark.parametrize("settings", [dict(route="fm"), dict(conditioning="oracle"),
    dict(codec_mode="direct16"), dict(route="ki", codec_mode="prefix16_holdpad32"),
    dict(route="joint", conditioning="task")])
def test_undeclared_mixed_ablation_settings_fail(settings):
    with pytest.raises(ValueError):
        ActionTrainingSettings(**settings)


def test_gradient_route_declaration():
    original = ActionTrainingSettings()
    assert not original.uses_fm
    assert replace(original, route="joint").uses_fm
    assert replace(original, route="ki").uses_fm
