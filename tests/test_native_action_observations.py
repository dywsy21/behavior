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
