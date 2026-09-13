from types import SimpleNamespace

import pytest
import torch

from native_action_observations import checked_observation, derive_static_action_mask, camera_feature_layout, CAMERAS, STATE_WIDTHS, ACTION_WIDTHS


def test_real_source_collator_cannot_consume_the_reference_sample():
    from g05.utils.data.data_utils import collate_fn_pad_sequences
    from probe_native_action_observations import collate_reference
    reference = dict(samples=dict(command="actual task", proprio=torch.ones(6, 27)),
        action=torch.ones(32, 27), idx=0)
    pipeline = SimpleNamespace(collate=collate_fn_pad_sequences)
    batch = collate_reference(pipeline, reference)
    assert reference["samples"]["command"] == batch["samples"][0]["command"]
    batch["samples"][0]["proprio"].zero_()
    assert reference["samples"]["proprio"].eq(1).all()


def actor_fixture():
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    batch = dict(samples=[dict(command="task", proprio="observed")],
        pixel_values={"exterior": torch.zeros(1, 6, 3, 4, 4)},
        action_dim_is_pad=torch.zeros(1, 27, dtype=torch.bool))
    calls = []
    def predict(**kwargs):
        assert not torch.is_grad_enabled()
        calls.append(kwargs)
        return {"test": "neural-output-placeholder"}
    adapter = SimpleNamespace(preprocess_for_inference=lambda observed: observed,
        collate=lambda prepared: batch, postprocess_action=lambda predicted, prepared: {"test": "raw-placeholder"})
    policy = SimpleNamespace(training=False, action_training=ActionTrainingSettings(conditioning="native_task"),
        forward_inference=predict)
    return policy, adapter, batch, calls


def test_native_actor_call_has_no_planner_teacher_or_raw_diagnostic_arguments():
    from native_action_observations import infer_native_task_observation
    policy, adapter, batch, calls = actor_fixture()
    result = infer_native_task_observation(policy, adapter, {}, device="cpu")
    assert len(calls) == 1 and set(calls[0]) == {"samples", "pixel_values", "action_dim_is_pad"}
    assert calls[0]["samples"] == batch["samples"]
    assert not result["teacher_or_planner_in_actor"] and not result["static_format_forced"]


@pytest.mark.parametrize("field", ["action", "gt_action", "action_gt", "action_is_pad"])
def test_native_actor_refuses_target_emission_before_any_neural_call(field):
    from native_action_observations import infer_native_task_observation
    policy, adapter, batch, calls = actor_fixture()
    batch[field] = torch.zeros(1)
    with pytest.raises(ValueError, match="forbidden"):
        infer_native_task_observation(policy, adapter, {}, device="cpu")
    assert calls == []


@pytest.mark.parametrize("damage", ["training", "skill_route", "cot_schema", "implicit_schema"])
def test_native_actor_cannot_silently_change_a_conditioning_or_decoding_route(damage):
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    from native_action_observations import infer_native_task_observation
    policy, adapter, _, calls = actor_fixture()
    constraint = False
    if damage == "training":
        policy.training = True
    elif damage == "skill_route":
        policy.action_training = ActionTrainingSettings(conditioning="skills")
    elif damage == "cot_schema":
        policy.action_training = ActionTrainingSettings(conditioning="native_subtask_cot")
        constraint = True
    else:
        constraint = "false"
    with pytest.raises(ValueError):
        infer_native_task_observation(policy, adapter, {}, device="cpu", constrain_format=constraint)
    assert calls == []


def observation():
    return dict(task="turn on the radio", embodiment_type="galaxea_r1pro", frequency=30.,
        images={camera: torch.zeros(6, 3, 8, 8, dtype=torch.uint8) for camera in CAMERAS},
        state={name: torch.zeros(6, width) for name, width in STATE_WIDTHS.items()},
        image_is_pad=torch.zeros(6, dtype=torch.bool), state_is_pad=torch.zeros(6, dtype=torch.bool))


def processor():
    return SimpleNamespace(shape_meta=dict(action=[dict(key=key, shape=width, time_offset=0)
                                                   for key, width in ACTION_WIDTHS.items()]),
        action_state_merger=SimpleNamespace(merge=True,
            max_action_shape_meta=dict(left_control=9, left_gripper=1, right_control=9, right_gripper=1, lower_body=7),
            merge_spec=dict(left_control=["left_arm", "left_ee_pose"], left_gripper=["left_gripper"],
                right_control=["right_arm", "right_ee_pose"], right_gripper=["right_gripper"],
                lower_body=["lower_body", "torso"])))


def test_native_observations_preserve_history_without_teacher_or_input_mutation():
    raw = observation()
    result = checked_observation(raw)
    result["state"]["left_arm"].fill_(3)
    assert raw["state"]["left_arm"].count_nonzero() == 0
    assert set(result) == set(raw)


@pytest.mark.parametrize("field", ["action", "gt_action", "memlite_model_projection", "atomic_task", "success", "idx"])
def test_teacher_planner_or_locator_is_not_required_or_accepted(field):
    raw = observation()
    raw[field] = "not an observation"
    with pytest.raises(ValueError, match="seven observable"):
        checked_observation(raw)


@pytest.mark.parametrize("damage", ["rgb_float", "nan_state", "missing_trunk", "five_frames", "pad_hole", "pad_disagreement"])
def test_real_history_and_all_channels_are_required(damage):
    raw = observation()
    if damage == "rgb_float":
        raw["images"]["head_rgb"] = raw["images"]["head_rgb"].float()
    elif damage == "nan_state":
        raw["state"]["left_arm"][0, 0] = float("nan")
    elif damage == "missing_trunk":
        del raw["state"]["trunk_qpos"]
    elif damage == "five_frames":
        raw["state"]["base_qvel"] = raw["state"]["base_qvel"][:5]
    elif damage == "pad_hole":
        raw["state_is_pad"][2] = True
    else:
        raw["state_is_pad"][0] = True
    with pytest.raises(ValueError):
        checked_observation(raw)


def test_static_mask_preserves_base_trunk_and_both_grippers():
    mask = derive_static_action_mask(processor())
    assert mask.dtype == torch.bool and mask.nonzero().flatten().tolist() == [7, 8, 17, 18]
    assert not mask[20:].any() and not mask[[9, 19]].any()


def camera_processor():
    types = ("exterior", "wrist_left", "wrist_right")
    return SimpleNamespace(shape_meta=dict(images=[dict(key=key, camera_type=kind, shape=[3, 256, 256])
        for key, kind in zip(CAMERAS, types)]))


def test_raw_camera_names_map_to_original_model_feature_types_in_the_same_order():
    layout = camera_feature_layout(camera_processor())
    assert list(layout) == ["exterior", "wrist_left", "wrist_right"]
    assert list(layout) != list(CAMERAS)
    assert all(shape == (6, 3, 256, 256) for shape in layout.values())


@pytest.mark.parametrize("damage", ["raw_order", "duplicate_type", "wrong_channels", "missing_camera"])
def test_feature_alias_fix_does_not_relax_image_layout_validation(damage):
    value = camera_processor()
    records = value.shape_meta["images"]
    if damage == "raw_order":
        records[0], records[1] = records[1], records[0]
    elif damage == "duplicate_type":
        records[1]["camera_type"] = records[0]["camera_type"]
    elif damage == "wrong_channels":
        records[0]["shape"][0] = 1
    else:
        records.pop()
    with pytest.raises(ValueError):
        camera_feature_layout(value)


@pytest.mark.parametrize("damage", ["old_action_offset", "missing_base", "group_order", "group_width"])
def test_unknown_robot_or_old_action_clock_is_rejected(damage):
    value = processor()
    if damage == "old_action_offset":
        value.shape_meta["action"][0]["time_offset"] = 5
    elif damage == "missing_base":
        value.shape_meta["action"] = value.shape_meta["action"][1:]
    elif damage == "group_order":
        value.action_state_merger.max_action_shape_meta = dict(reversed(list(value.action_state_merger.max_action_shape_meta.items())))
    else:
        value.action_state_merger.max_action_shape_meta["lower_body"] = 6
    with pytest.raises(ValueError):
        derive_static_action_mask(value)
