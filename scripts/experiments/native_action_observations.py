"""Observed-only task-AR processor and the unchanged R1Pro inverse path.

This is a distinct native task route, not a fake MEM-Lite skill projection.
The caller must first admit real history with the existing history transport.
A dedicated processor copy uses the source public preprocess API with a
camera-major Base template. Shared training processors are never modified.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
import time

import torch

from g05.data_processor.processor.samples_builder import _CameraMajorHistoryBuilder
from g05.models.g05.inferencer import PolicyInferencer
from g05.utils.training.ar_training_methods import native_task_actor_samples

CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
STATE_WIDTHS = dict(base_qvel=3, trunk_qpos=4, left_arm=7, left_gripper=2,
                    right_arm=7, right_gripper=2)
ACTION_WIDTHS = {**STATE_WIDTHS, "left_gripper": 1, "right_gripper": 1}
OBSERVATION_FIELDS = {"task", "embodiment_type", "frequency", "images", "state",
                      "image_is_pad", "state_is_pad"}


def checked_observation(observation):
    """Validate the already history-admitted seven observable fields only."""
    if not isinstance(observation, dict) or set(observation) != OBSERVATION_FIELDS:
        raise ValueError("Native task actor accepts only seven observable fields; no teacher or planner")
    if (not isinstance(observation["task"], str) or not observation["task"].strip()
            or observation["embodiment_type"] != "galaxea_r1pro"
            or type(observation["frequency"]) not in (int, float) or observation["frequency"] != 30):
        raise ValueError("Require a real task and declared 30 Hz R1Pro embodiment")
    if (not isinstance(observation["images"], dict) or set(observation["images"]) != set(CAMERAS)
            or not isinstance(observation["state"], dict) or set(observation["state"]) != set(STATE_WIDTHS)):
        raise ValueError("Native task actor requires all three RGB and six proprio channels")
    for name in CAMERAS:
        value = observation["images"][name]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8
                or value.ndim != 4 or value.shape[:2] != (6, 3) or min(value.shape[2:]) < 1):
            raise ValueError("Expected actual CPU uint8 RGB history [6,3,H,W]")
    for name, width in STATE_WIDTHS.items():
        value = observation["state"][name]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or not value.is_floating_point() or value.shape != (6, width) or not torch.isfinite(value).all()):
            raise ValueError("Invalid actual six-frame proprio channel: " + name)
    for name in ("image_is_pad", "state_is_pad"):
        pad = observation[name]
        if (not isinstance(pad, torch.Tensor) or pad.device.type != "cpu" or pad.dtype != torch.bool
                or pad.shape != (6,) or bool(pad[-1]) or (pad[1:] & ~pad[:-1]).any()):
            raise ValueError("History padding must be a contiguous unavailable prefix, not future history")
    if not torch.equal(observation["image_is_pad"], observation["state_is_pad"]):
        raise ValueError("Image and proprio history masks disagree")
    return deepcopy(observation)


def derive_static_action_mask(processor):
    """Same audited robot metadata as the existing aligned FM entry; no GT."""
    widths = {}
    for item in processor.shape_meta["action"]:
        key, width = item["key"], item["shape"]
        if key in widths or type(width) is not int or width < 1 or item.get("time_offset") != 0:
            raise ValueError("Only unambiguous future-only action metadata is supported")
        widths[key] = width
    merger = processor.action_state_merger
    groups = [("left_control", 9), ("left_gripper", 1), ("right_control", 9),
              ("right_gripper", 1), ("lower_body", 7)]
    spec = dict(left_control=["left_arm", "left_ee_pose"], left_gripper=["left_gripper"],
                right_control=["right_arm", "right_ee_pose"], right_gripper=["right_gripper"],
                lower_body=["lower_body", "torso"])
    if (widths != ACTION_WIDTHS or not merger.merge or list(merger.max_action_shape_meta.items()) != groups
            or dict(merger.merge_spec) != spec):
        raise ValueError("Unsupported R1Pro 23-to-27 grouping; do not guess the padding mask")
    active = dict(left_control=widths["left_arm"], left_gripper=widths["left_gripper"],
        right_control=widths["right_arm"], right_gripper=widths["right_gripper"],
        lower_body=widths["trunk_qpos"] + widths["base_qvel"])
    mask = torch.tensor([flag for key, width in groups
                         for flag in [False] * active[key] + [True] * (width - active[key])])
    if mask.shape != (27,) or mask.nonzero().flatten().tolist() != [7, 8, 17, 18]:
        raise ValueError("Expected all 23 actual R1Pro controls")
    return mask


def camera_feature_layout(processor):
    """Raw camera keys and the model's camera-type keys are different names."""
    records = processor.shape_meta["images"]
    if [record["key"] for record in records] != list(CAMERAS):
        raise ValueError("Raw camera metadata must retain the three official cameras in order")
    layout = {}
    for record in records:
        key, shape = record["camera_type"], tuple(record["shape"])
        if (not isinstance(key, str) or not key or key in layout or len(shape) != 3
                or shape[0] != 3 or any(type(size) is not int or size < 1 for size in shape)):
            raise ValueError("Invalid or duplicate model camera-type stream metadata")
        layout[key] = (6, *shape)
    return layout


@dataclass
class PreparedNativeObservation:
    sample: dict
    raw_state_anchor: dict


class NativeTaskObservationProcessor:
    """New public observed-only adapter; keep source tensor math and inverse."""

    def __init__(self, source_processor):
        builder = source_processor.samples_builder
        if (source_processor.num_obs_steps != 6 or source_processor.num_input_images != 18
                or source_processor.action_execution_start_index not in (None, 0)
                or builder.embodiment_type != "galaxea_r1pro"
                or builder.hardcode_instruction is not None or builder.hardcode_proprio_pad_zeros
                or builder.hardcode_action_pad_ones):
            raise ValueError("Native task processor needs the audited six-frame R1Pro configuration")
        self.processor = deepcopy(source_processor)
        self.action_mask = derive_static_action_mask(self.processor)
        self.pixel_layout = camera_feature_layout(self.processor)
        self.processor.samples_builder = _CameraMajorHistoryBuilder(18, dict(builder._image_sizes),
            embodiment_type=builder.embodiment_type)
        self.processor.set_action_execution_start_index(0)
        self.processor.eval()

    def preprocess_for_inference(self, observation):
        raw = checked_observation(observation)
        anchor = {name: raw["state"][name][-1:].detach().float().clone().unsqueeze(0)
                  for name in ("left_arm", "right_arm", "trunk_qpos")}
        raw["idx"] = 0  # Neutral collator bookkeeping; not a dataset/episode locator.
        sample = self.processor.preprocess(raw)
        if any(key in sample for key in ("action", "gt_action", "action_gt", "action_is_pad", "_vlm_action")):
            raise ValueError("Observed-only public preprocessing emitted an expert target")
        sample["samples"] = native_task_actor_samples([sample["samples"]], num_images=18)[0]
        if (sample["proprio"].shape != (6, 27) or not torch.isfinite(sample["proprio"]).all()
                or not torch.equal(sample["proprio_dim_is_pad"], self.action_mask)
                or list(sample["pixel_values"]) != list(self.pixel_layout)
                or any(tuple(value.shape) != self.pixel_layout.get(key)
                       for key, value in sample["pixel_values"].items())):
            raise ValueError("Native processor changed the real history/order/embodiment layout: " + repr(dict(
                proprio_shape=list(sample["proprio"].shape),
                proprio_padding=sample["proprio_dim_is_pad"].nonzero().flatten().tolist(),
                actual_pixel_shapes={key: list(value.shape) for key, value in sample["pixel_values"].items()},
                expected_pixel_shapes=self.pixel_layout)))
        sample["action_dim_is_pad"] = self.action_mask.clone()
        return PreparedNativeObservation(sample=sample, raw_state_anchor=anchor)

    def collate(self, prepared):
        return PolicyInferencer._collate([prepared.sample], padding_input_id=self.processor.pad_token_id)

    def postprocess_action(self, prediction, prepared):
        """Model-normalized output -> all raw controls; no direct 27D slicing."""
        action = prediction.get("action")
        if (prediction.get("selected_action_source") != "ar"
                or prediction.get("ar_complete_block_receipts") != [{"token_count": 60, "complete_blocks": 8}]
                or prediction.get("ar_absent_keys") is None or len(prediction["ar_absent_keys"]) != 1
                or any(prediction["ar_absent_keys"])
                or not isinstance(action, torch.Tensor) or action.shape != (1, 32, 27)
                or not torch.isfinite(action).all() or prediction.get("execution_start") != 0
                or prediction.get("execution_steps") != 16):
            raise ValueError("Require a complete finite AR action with explicit 0:16 execution")
        sample = prepared.sample
        batch = dict(action=action.detach().cpu(), selected_action_source="ar", ar_absent_keys=[set()],
            proprio=sample["proprio"].unsqueeze(0), action_dim_is_pad=self.action_mask.unsqueeze(0),
            proprio_dim_is_pad=sample["proprio_dim_is_pad"].unsqueeze(0))
        raw = PolicyInferencer._postprocess_single(batch, 0, self.processor,
                                                   raw_state_anchor=prepared.raw_state_anchor)
        if {key for key in raw if not key.startswith("_")} != set(ACTION_WIDTHS) or raw.get("_absent_keys"):
            raise ValueError("Inverse processor omitted real raw controls")
        for name, width in ACTION_WIDTHS.items():
            value = raw[name]
            if (not isinstance(value, torch.Tensor) or value.shape != (1, 32, width)
                    or not torch.isfinite(value).all()):
                raise ValueError("Invalid raw inverse action or wrong execution start: " + name)
        return raw


def infer_native_task_observation(policy, adapter, observation, *, device, constrain_format=False):
    """One observed-only native actor call, followed by the checked inverse.

    History admission and session RNG are owned by the external controller.
    No installed skill, teacher token, future action, or physical truth is an
    argument to this entry. Schema constraints are an explicit optional variant.
    """
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.models.g05.helpers.action_schema_decoding import constrained_action_schema

    settings = policy.action_training
    if (policy.training or settings.route != "ar" or settings.conditioning not in {"native_task", "native_subtask_cot"}
            or type(constrain_format) is not bool or constrain_format and settings.predicts_cot):
        raise ValueError("Require eval-mode native task AR; initial constrained variant is non-CoT only")
    device = torch.device(device)
    started = time.monotonic()
    prepared = adapter.preprocess_for_inference(observation)
    batch = dict_apply(adapter.collate(prepared),
        lambda value: value.to(device) if isinstance(value, torch.Tensor) else value)
    if any(key in batch for key in ("action", "action_gt", "gt_action", "action_is_pad")):
        raise ValueError("Native actor collator emitted forbidden future/teacher targets")
    prepared_at = time.monotonic()
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), precision:
        with constrained_action_schema(policy) if constrain_format else nullcontext() as sampler:
            prediction = policy.forward_inference(samples=batch["samples"], pixel_values=batch["pixel_values"],
                                                   action_dim_is_pad=batch["action_dim_is_pad"])
    generated_at = time.monotonic()
    raw = adapter.postprocess_action(prediction, prepared)
    return dict(grouped_raw_action=raw, generated=prediction,
        schema=sampler.require_complete() if sampler is not None else None,
        schema_trace=sampler.trace if sampler is not None else [],
        timing=dict(preprocess_ms=(prepared_at - started) * 1000,
                    infer_ms=(generated_at - prepared_at) * 1000,
                    postprocess_ms=(time.monotonic() - generated_at) * 1000),
        actor_route=settings.conditioning + "_ar", teacher_or_planner_in_actor=False,
        static_format_forced=constrain_format, execution_start=0, execution_steps=16)
