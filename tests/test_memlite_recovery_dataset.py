from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g05.data.memlite_recovery_dataset import recovery_raw_sample


def fixture():
    shape = {"state": [{"key": "base_qvel", "start_index": 0, "raw_shape": 3},
                       {"key": "trunk_qpos", "start_index": 53, "raw_shape": 4}],
             "action": [{"key": "base_qvel", "start_index": 0, "raw_shape": 3},
                        {"key": "trunk_qpos", "start_index": 3, "raw_shape": 4}]}
    processor = SimpleNamespace(num_obs_steps=6, shape_meta=shape)
    states = np.arange(160*61, dtype=np.float32).reshape(160, 61)
    actions = np.arange(160*23, dtype=np.float32).reshape(160, 23)
    row = {"step": 96, "index": 10000, "command": "turn on radio", "memlite_branch": "low",
           "memory": "Task=0; Completed=none.", "previous_intent": "pick up radio",
           "intent": "face radio", "memory_update": "Task=0; Recovery=aligning.",
           "intent_status": "CONTINUE", "execution_feedback": "Repeated rotation observed."}
    def frame(path):
        t = int(path.stem.split("_")[1])
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[..., 0] = t
        rgba[..., 1] = 12
        rgba[..., 2] = 23
        rgba[..., 3] = 255
        return {suffix: rgba for suffix in ("zed_link:Camera:0::rgb", "left_realsense_link:Camera:0::rgb", "right_realsense_link:Camera:0::rgb")}
    return row, processor, {"states": states, "actions": actions}, frame


def test_recovery_uses_causal_six_frame_history_and_matching_raw_actions():
    row, p, source, loader = fixture()
    result = recovery_raw_sample(row, Path("/data"), p, source, loader)
    np.testing.assert_array_equal(result["images"]["head_rgb"][:, 0, 0, 0], [16,32,48,64,80,96])
    assert result["images"]["head_rgb"].shape == (6, 3, 8, 8)
    assert (result["images"]["head_rgb"][:, 2] == 23).all()  # no RGB/BGR reversal
    np.testing.assert_array_equal(result["state"]["trunk_qpos"], source["states"][[16,32,48,64,80,96], 53:57])
    np.testing.assert_array_equal(result["action"]["base_qvel"], source["actions"][96:128, :3])
    assert not result["action_is_pad"].any()
    assert result["execution_feedback"] == row["execution_feedback"]
    assert "geometry_diagnostic_only" not in result and "position" not in result


@pytest.mark.parametrize("step", [79, 81, 144])
def test_partial_or_wrong_cadence_recovery_samples_fail(step):
    row, p, source, loader = fixture()
    row["step"] = step
    with pytest.raises(ValueError):
        recovery_raw_sample(row, Path("/data"), p, source, loader)


def test_high_level_has_no_action_supervision_but_real_targets_stay_intact():
    row, p, source, loader = fixture()
    row["memlite_branch"] = "high"
    sample = recovery_raw_sample(row, Path("/data"), p, source, loader)
    assert sample["action_is_pad"].all() and sample["memlite_action_valid_steps"] == 0
    np.testing.assert_array_equal(sample["action"]["base_qvel"], source["actions"][96:128, :3])


def test_fm_current_frame_view_preserves_all_raw_continuous_targets():
    row, p, source, loader = fixture()
    six = recovery_raw_sample(row, Path("/data"), p, source, loader)
    p.num_obs_steps = 1
    one = recovery_raw_sample(row, Path("/data"), p, source, loader)
    for key in one["state"]:
        np.testing.assert_array_equal(one["state"][key], six["state"][key][-1:])
    for key in one["images"]:
        np.testing.assert_array_equal(one["images"][key], six["images"][key][-1:])
    for key in one["action"]:
        np.testing.assert_array_equal(one["action"][key], six["action"][key])
    assert one["intent"] == six["intent"]
    assert one["state_is_pad"].shape == (1,)
