"""CPU regressions for BEHAVIOR/R1Pro lower-body processing."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from g05.data_processor.processor.galaxea_cot_processor import GalaxeaCoTProcessor
from g05.data_processor.transforms.action_filter import BaseActionFilter
from g05.data_processor.transforms.action_state_merger import GroupedPaddingMerger
from g05.data_processor.transforms.relative_action import BehaviorPerKeyTransform
from g05.models.g05 import g05_policy
from g05.models.g05.inferencer import PolicyInferencer


PARTS_META = {
    "left_arm": 7,
    "right_arm": 7,
    "left_gripper": 1,
    "right_gripper": 1,
    "left_ee_pose": 9,
    "right_ee_pose": 9,
    "lower_body": 7,
}
MERGE_SPEC = {
    "left_control": ["left_arm", "left_ee_pose"],
    "left_gripper": ["left_gripper"],
    "right_control": ["right_arm", "right_ee_pose"],
    "right_gripper": ["right_gripper"],
    "lower_body": ["lower_body"],
}


def _meta(key: str, raw_shape: int, shape: int | None = None) -> dict:
    return {"key": key, "raw_shape": raw_shape, "shape": raw_shape if shape is None else shape}


RAW_SHAPE_META = {
    "action": [
        _meta("base_qvel", 3),
        _meta("trunk_qpos", 4),
        _meta("left_arm", 7),
        _meta("left_gripper", 1),
        _meta("right_arm", 7),
        _meta("right_gripper", 1),
    ],
    "state": [
        _meta("base_qvel", 3),
        _meta("trunk_qpos", 4),
        _meta("left_arm", 7),
        _meta("left_gripper", 2, 1),
        _meta("right_arm", 7),
        _meta("right_gripper", 2, 1),
    ],
    "images": [],
}


def _merger() -> GroupedPaddingMerger:
    merger = GroupedPaddingMerger(
        max_action_shape_meta=PARTS_META,
        max_state_shape_meta=PARTS_META,
        merge_spec=MERGE_SPEC,
    )
    merger.set_shape_meta(deepcopy(RAW_SHAPE_META))
    return merger


def _raw_batch(*, batched: bool) -> dict:
    if batched:
        action_shape, state_shape, mask_shape = (2, 3), (2, 1), (2,)
    else:
        action_shape, state_shape, mask_shape = (3,), (1,), ()

    def action_values(dim: int, start: float) -> torch.Tensor:
        return torch.arange(
            start, start + int(torch.tensor(action_shape).prod()) * dim, dtype=torch.float32
        ).reshape(*action_shape, dim)

    def state_values(dim: int, start: float) -> torch.Tensor:
        return torch.arange(
            start, start + int(torch.tensor(state_shape).prod()) * dim, dtype=torch.float32
        ).reshape(*state_shape, dim)

    def mask_values(dim: int, start: int) -> torch.Tensor:
        values = torch.tensor([(start + i) % 2 == 0 for i in range(dim)], dtype=torch.bool)
        return values.expand(*mask_shape, dim).clone()

    return {
        "action": {
            "base_qvel": action_values(3, 1),
            "trunk_qpos": action_values(4, 101),
            "left_arm": action_values(7, 201),
            "left_gripper": action_values(1, 301),
            "right_arm": action_values(7, 401),
            "right_gripper": action_values(1, 501),
        },
        "state": {
            "base_qvel": state_values(3, 11),
            "trunk_qpos": state_values(4, 21),
            "left_arm": state_values(7, 31),
            "left_gripper": state_values(2, 41),
            "right_arm": state_values(7, 51),
            "right_gripper": state_values(2, 61),
        },
        "action_op_mask": {
            "base_qvel": mask_values(3, 0),
            "trunk_qpos": mask_values(4, 1),
            "left_arm": mask_values(7, 2),
            "left_gripper": mask_values(1, 3),
            "right_arm": mask_values(7, 4),
            "right_gripper": mask_values(1, 5),
        },
        "action_is_pad": torch.zeros(action_shape[-1], dtype=torch.bool),
        "state_is_pad": torch.zeros(state_shape[-1], dtype=torch.bool),
        "idx": 0,
    }


@pytest.mark.parametrize("batched", [False, True])
def test_behavior_transform_keeps_lower_body_mask_in_trunk_base_order(batched: bool):
    raw = _raw_batch(batched=batched)
    transformed = BehaviorPerKeyTransform().forward(deepcopy(raw))

    assert "lower_body" in transformed["action"]
    assert "lower_body" in transformed["action_op_mask"]
    torch.testing.assert_close(
        transformed["action_op_mask"]["lower_body"],
        torch.cat(
            [raw["action_op_mask"]["trunk_qpos"], raw["action_op_mask"]["base_qvel"]],
            dim=-1,
        ),
    )
    assert "trunk_qpos" not in transformed["action_op_mask"]
    assert "base_qvel" not in transformed["action_op_mask"]


def test_merger_and_behavior_backward_restore_raw_lower_body_and_mask():
    raw = _raw_batch(batched=False)
    transform = BehaviorPerKeyTransform()
    merger = _merger()

    flattened = merger.forward(transform.forward(deepcopy(raw)))
    assert not flattened["action_dim_is_pad"][-7:].any()
    assert flattened["action_op_mask"][-7:].any()

    restored = transform.backward(merger.backward(flattened))
    assert "lower_body" not in restored["action"]
    assert "lower_body" not in restored["action_op_mask"]
    for key in ("base_qvel", "trunk_qpos"):
        torch.testing.assert_close(restored["action"][key], raw["action"][key])
        torch.testing.assert_close(restored["action_op_mask"][key], raw["action_op_mask"][key])


class _IdentityNormalizer:
    def forward(self, data):
        return data

    def backward(self, data):
        return data


class _NoopSamplesBuilder:
    def __init__(self, **_kwargs):
        pass

    def build(self, _data, sample):
        return {"template": "test", "action": sample.get("action")}


def _galaxea_processor() -> GalaxeaCoTProcessor:
    processor = GalaxeaCoTProcessor(
        shape_meta=deepcopy(RAW_SHAPE_META),
        num_obs_steps=1,
        num_output_cameras=0,
        action_state_transforms=[BehaviorPerKeyTransform()],
        use_stepwise_action_norm=False,
        norm_default_mode="dummy",
        action_state_merger=_merger(),
        action_filter=BaseActionFilter(),
        train_transforms=None,
        val_transforms=None,
        drop_high_level_prob=0.0,
        use_zh_instruction=False,
        pad_token_id=0,
        image_token_index=0,
        max_text_tokens=1,
        num_input_cameras=0,
        discrete_action=True,
        samples_builder=_NoopSamplesBuilder,
    )
    processor._normalizer = _IdentityNormalizer()
    processor.process_images = lambda _data: {}
    processor.build_pixel_values = lambda _pixel_values: torch.empty(0)
    return processor


def test_galaxea_gt_action_is_captured_after_behavior_transform():
    raw = _raw_batch(batched=False)
    raw["task"] = "move the object"
    expected_lower_body = torch.cat(
        [raw["action"]["trunk_qpos"], raw["action"]["base_qvel"]], dim=-1
    )
    expected_lower_body[..., :3] -= raw["state"]["trunk_qpos"][..., :3]
    processor = _galaxea_processor()

    sample = processor._process_tensors(raw)

    assert "gt_action" in sample
    assert sample["gt_action"].shape[-1] == 27
    assert sample["gt_action"][..., -7:].abs().sum() > 0
    torch.testing.assert_close(sample["gt_action"][..., -7:], expected_lower_body)


@pytest.mark.parametrize(
    ("group_name", "expected_raw_absent"),
    [
        ("lower_body", {"base_qvel", "trunk_qpos"}),
        ("left_control", {"left_arm"}),
    ],
)
def test_ar_absent_groups_map_to_raw_wire_keys(
    group_name: str, expected_raw_absent: set[str]
):
    # Use the real grouped merger + Behavior transform postprocessor, not a
    # fake grouped key, so this covers the exact wire-key namespace.
    processor = _galaxea_processor()
    action = torch.full((1, 2, 27), 0.25)
    group_slices = {
        "left_control": slice(0, 9),
        "left_gripper": slice(9, 10),
        "right_control": slice(10, 19),
        "right_gripper": slice(19, 20),
        "lower_body": slice(20, 27),
    }
    action[..., group_slices[group_name]] = -100.0

    wire_action = PolicyInferencer._postprocess_single(
        {
            "action": action,
            "proprio": torch.zeros(1, 1, 27),
            "action_dim_is_pad": torch.zeros(1, 27, dtype=torch.bool),
            "selected_action_source": "ar",
            "ar_absent_keys": [{group_name}],
        },
        index=0,
        sub_processor=processor,
    )

    assert wire_action["_absent_keys"] == expected_raw_absent
    delivered = dict(wire_action)
    for key in delivered.pop("_absent_keys"):
        delivered.pop(key, None)
    assert not (set(delivered) & expected_raw_absent)


def test_fm_output_never_inherits_ar_absent_groups():
    processor = _galaxea_processor()
    wire_action = PolicyInferencer._postprocess_single(
        {
            "action": torch.ones(1, 2, 27),
            "proprio": torch.zeros(1, 1, 27),
            "action_dim_is_pad": torch.zeros(1, 27, dtype=torch.bool),
            "selected_action_source": "fm",
            "ar_absent_keys": [{"lower_body"}],
        },
        index=0,
        sub_processor=processor,
    )

    assert "_absent_keys" not in wire_action
    assert {"base_qvel", "trunk_qpos"} <= set(wire_action)


def test_unmatched_grouped_slots_do_not_create_virtual_other_arm_actions():
    right_only_meta = {
        "action": [_meta("right_arm", 7)],
        "state": [_meta("right_arm", 7)],
        "images": [],
    }
    merger = GroupedPaddingMerger(
        max_action_shape_meta=PARTS_META,
        max_state_shape_meta=PARTS_META,
        merge_spec=MERGE_SPEC,
    )
    merger.set_shape_meta(right_only_meta)
    restored = merger.backward(
        {
            "action": torch.ones(2, 27),
            "state": torch.ones(1, 27),
        }
    )

    assert set(restored["action"]) == {"right_arm"}
    assert set(restored["state"]) == {"right_arm"}


class _FakeInferenceState:
    def __init__(self):
        self.attention_mask = torch.ones(1, 1, dtype=torch.long)
        self.pixel_values = torch.zeros(1, 1)
        self.kv_cache = None
        self.position_ids = torch.zeros(1, 1, dtype=torch.long)
        self.last_hidden = torch.zeros(1, 1, 1)
        self.input_ids = torch.ones(1, 1, dtype=torch.long)
        self.generated_ids = None
        self.device = torch.device("cpu")

    def check_invariants(self, _where: str):
        return None


class _FakeModel:
    def inference_fm(self, **_kwargs):
        return torch.full((1, 2, 3), 5.0)

    def inference_ar(self, *_args, **_kwargs):
        return {
            "generated_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": None,
            "past_key_values": None,
            "last_hidden": None,
        }


class _FakeArProcessor:
    def decode_ar(self, *_args, **_kwargs):
        return [torch.full((2, 3), 9.0)], [], None, [{"lower_body"}]


@pytest.mark.parametrize(
    ("continuous", "discrete", "expected_source", "expected_value"),
    [(True, True, "fm", 5.0), (False, True, "ar", 9.0), (True, False, "fm", 5.0)],
)
def test_generate_action_marks_the_selected_head(
    monkeypatch,
    continuous: bool,
    discrete: bool,
    expected_source: str,
    expected_value: float,
):
    monkeypatch.setattr(g05_policy, "_sync_if_cuda_available", lambda: None)
    policy = SimpleNamespace(
        continuous_action=continuous,
        discrete_action=discrete,
        model=_FakeModel(),
        processor=_FakeArProcessor(),
        model_config=SimpleNamespace(horizon_steps=2, action_dim=3),
        _get_action_stop_token_ids=lambda: [],
        _get_action_generation_max_new_tokens=lambda: 1,
    )

    result = g05_policy.G05Policy.generate_action(
        policy,
        _FakeInferenceState(),
        samples=[{}],
        action_dim_is_pad=torch.zeros(1, 3, dtype=torch.bool),
    )

    assert result["selected_action_source"] == expected_source
    assert result["action"].eq(expected_value).all()


@pytest.mark.parametrize("predict_cot", [False, True])
def test_forward_inference_rejects_config_without_any_action_head(predict_cot: bool):
    policy = SimpleNamespace(
        continuous_action=False,
        discrete_action=False,
        predict_cot=predict_cot,
    )

    with pytest.raises(ValueError, match="at least one action head"):
        g05_policy.G05Policy.forward_inference(
            policy,
            samples=[],
            pixel_values=torch.empty(0),
        )
