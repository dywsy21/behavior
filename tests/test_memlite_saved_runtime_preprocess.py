#!/usr/bin/env python3
"""CPU regression for the exact saved MEM-Lite run's high/low preprocessing.

This intentionally loads only the saved OmegaConf, processor, stats, and
branch templates.  It never loads checkpoint tensors, a VLM, CUDA, or an
OmniGibson scene.  Run it from the GalaxeaVLA repository root: the saved
OmegaConf legitimately uses repository-relative ``oc.load`` references, and
the production launcher starts the policy server from that same directory.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from g05.models.g05.g05_policy import G05Policy
from g05.models.g05.inferencer import PolicyInferencer, resolve_processor
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.data.processor_utils import build_processors
from scripts.serve_policy_mem import (
    _configure_action_execution_start_index,
    build_obs_dict,
)
from serve_behavior_policy_mem import (
    behavior_action_to_vector,
    behavior_obs_to_g05,
    load_runtime_config,
)


def _official_observation() -> dict:
    return {
        "robot_r1::proprio": np.arange(61, dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros((12, 16, 4), dtype=np.uint8),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros((10, 14, 3), dtype=np.uint8),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros((3, 8, 6), dtype=np.uint8),
        "task_id": np.asarray([0], dtype=np.int64),
    }


def _branch_prepared(inferencer: PolicyInferencer, data: dict, **kwargs):
    runtime = inferencer._with_memlite_runtime_branch([data], **kwargs)
    prepared, _ = inferencer._prepare_branch_batch(runtime)
    assert len(prepared) == 1
    return prepared[0]


def _assert_saved_processor_marker_mapping(sub_processor) -> None:
    """Document the saved config's exact 18-frame -> 3-frame behavior.

    This is diagnostic only: it does not alter the processor.  Distinct camera
    / time values show that eval selects exterior t=0,1,2 and then assigns
    them to the three output keys.  Training uses its seeded random three
    indices and likewise reassigns the selected frames to output keys.
    """
    camera_keys = [meta["camera_type"] for meta in sub_processor.shape_meta["images"]]
    assert len(camera_keys) == 3
    marker_streams = {
        camera: torch.arange(offset, offset + 6, dtype=torch.float32).reshape(6, 1, 1, 1)
        for offset, camera in zip((0, 10, 20), camera_keys)
    }
    flat_markers = torch.cat([marker_streams[key] for key in camera_keys], dim=0)

    sub_processor.eval()
    eval_pixels = sub_processor.build_pixel_values(
        {key: value.clone() for key, value in marker_streams.items()}
    )
    eval_values = [float(eval_pixels[key].reshape(-1)[0]) for key in camera_keys]
    assert eval_values == [0.0, 1.0, 2.0]

    sub_processor.train()
    torch.manual_seed(7)
    expected_indices = torch.randperm(18)[:3].sort().values
    assert expected_indices.tolist() == [6, 9, 11]
    torch.manual_seed(7)
    train_pixels = sub_processor.build_pixel_values(
        {key: value.clone() for key, value in marker_streams.items()}
    )
    train_values = torch.tensor(
        [float(train_pixels[key].reshape(-1)[0]) for key in camera_keys]
    )
    assert torch.equal(train_values, flat_markers[expected_indices].reshape(-1))
    assert train_values.tolist() == [10.0, 13.0, 15.0]
    sub_processor.eval()


def _assert_actual_preprocessed_pixel_values(prepared) -> None:
    pixels = prepared.sample["pixel_values"]
    assert isinstance(pixels, dict)
    assert len(pixels) == 3
    shapes = {str(key): tuple(value.shape) for key, value in pixels.items()}
    assert all(value.shape[0] == 1 for value in pixels.values()), shapes


def _assert_saved_ar_future_only_execution_rows(sub_processor) -> None:
    """Exercise the real saved processor's inverse AR path without a model.

    The saved run labels decoded action rows 0..31 as current-to-future
    actions and declares ``past_action_size=0``.  A row-distinct sentinel
    therefore catches any accidental image-history-derived offset before the
    BEHAVIOR bridge can execute a chunk.
    """
    action = (
        torch.arange(32, dtype=torch.float32).reshape(1, 32, 1)
        .repeat(1, 1, 27)
        / 100.0
    )
    proprio = torch.zeros(1, 6, 27, dtype=torch.float32)
    action_dim_is_pad = torch.zeros(1, 27, dtype=torch.bool)
    action_dim_is_pad[:, 7:9] = True
    action_dim_is_pad[:, 17:19] = True
    ar_absent = set()
    action_op_mask = PolicyInferencer._build_ar_presence_mask(
        action, action_dim_is_pad, ar_absent, sub_processor
    )
    expected_data = {
        "action": action.clone(),
        "state": proprio.clone(),
        "action_dim_is_pad": action_dim_is_pad.clone(),
        "action_op_mask": action_op_mask,
    }
    expected_data = sub_processor.action_state_merger.backward(expected_data)
    expected_data = sub_processor.normalizer.backward(expected_data)
    if sub_processor.action_state_transforms is not None:
        for transform in reversed(sub_processor.action_state_transforms):
            expected_data = transform.backward(expected_data)
    expected_data = sub_processor.action_filter.backward(expected_data)
    expected = expected_data["action"]

    actual = PolicyInferencer._postprocess_single(
        {
            "action": action,
            "proprio": proprio,
            "action_dim_is_pad": action_dim_is_pad,
            "selected_action_source": "ar",
            "ar_absent_keys": [ar_absent],
        },
        index=0,
        sub_processor=sub_processor,
    )
    assert "_absent_keys" not in actual
    assert set(actual) == set(expected)
    for key in sorted(expected):
        assert actual[key].shape[1] == 32, (key, actual[key].shape)
        torch.testing.assert_close(actual[key], expected[key])
        # This is the exact first official chunk: decoded rows 0..15, not
        # rows 5..20 inherited from six image observations.
        torch.testing.assert_close(actual[key][:, :16], expected[key][:, :16])
    for row in range(16):
        actual_vector = behavior_action_to_vector(
            {key: value[0, row].detach().cpu().numpy() for key, value in actual.items()}
        )
        expected_vector = behavior_action_to_vector(
            {key: value[0, row].detach().cpu().numpy() for key, value in expected.items()}
        )
        np.testing.assert_allclose(actual_vector, expected_vector)
        assert np.isfinite(actual_vector).all()


class _SavedRuntimeHighCapCapture:
    """CPU-only policy facade proving the saved processor receives the HL cap."""

    def __init__(self) -> None:
        self.max_new_tokens: int | None = None
        self.memory_text: str | None = None

    def generate_high_level(self, *, samples, pixel_values, memory_text, max_new_tokens):
        self.max_new_tokens = int(max_new_tokens)
        self.memory_text = str(memory_text)
        assert samples[0]["memlite_branch"] == "high"
        assert len(pixel_values) == 3
        return {
            "intent": ["synthetic intent"],
            "memory": [memory_text],
            "status": ["CONTINUE"],
            "high_level_text": [
                "Intent: synthetic intent|Updated Memory: "
                f"{memory_text}|Status: CONTINUE|<HL_END>"
            ],
            "high_level_generation_metadata": [
                {
                    "generated_token_count": 7,
                    "max_new_tokens": int(max_new_tokens),
                    "budget_exhausted": False,
                    "stop_token_id": 1,
                    "stop_reason": "hl_end",
                }
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_runtime_config(args.checkpoint, [])
    processors = build_processors(config)
    run_dir = config.run_dir
    processors.set_normalizer_from_stats(load_dataset_stats_from_json(f"{run_dir}/dataset_stats.json"))
    processors.eval()
    # Match ``scripts.serve_policy_mem.setup`` without loading its model.
    action_horizon = int(config.data.action_size)
    processor_values = (
        processors.processors.values() if hasattr(processors, "processors") else [processors]
    )
    for processor in processor_values:
        processor.action_horizon = action_horizon
    configured_execution_start = _configure_action_execution_start_index(processors, config)
    assert configured_execution_start == 0

    raw_obs = behavior_obs_to_g05(_official_observation(), {0: "Turn on the radio."})
    data = build_obs_dict(
        raw_obs,
        processors,
        mem_state=SimpleNamespace(memory_text="", intent_text="", status="INVALID"),
    )
    inferencer = PolicyInferencer(policy=None, processor=processors, device="cpu")
    high_prepared = _branch_prepared(
        inferencer, data, branch="high", memory_texts=[""]
    )
    high_sample = high_prepared.sample["samples"]
    sub_processor = resolve_processor(processors, data)
    assert sub_processor.action_execution_start_index == 0
    _assert_saved_processor_marker_mapping(sub_processor)
    _assert_saved_ar_future_only_execution_rows(sub_processor)
    _assert_actual_preprocessed_pixel_values(high_prepared)
    assert type(sub_processor.samples_builder).__name__ == "MEMLiteMixedBuilder"
    assert high_sample["memlite_branch"] == "high"
    assert high_sample["memory"] == "No prior memory."
    assert high_sample["intent"] == "Intent: __memlite_runtime_target_placeholder__"
    assert high_sample["memory_update"] == "Updated Memory: __memlite_runtime_target_placeholder__"
    assert high_sample["intent_status"] == "Status: CONTINUE"

    policy = object.__new__(G05Policy)
    policy.num_input_images = int(config.model.model_arch.num_input_images)
    high_prefill = policy._clone_memlite_samples([high_sample], branch="high", memory="")[0]
    assert high_prefill["memory"] == ""
    assert "prompt" in high_prefill
    assert all(key not in high_prefill for key in ("intent", "memory_update", "intent_status"))
    assert high_prefill["template"].index("<EOC>") < high_prefill["template"].index("<EOV>")
    assert high_prefill["template"].count("_image_!") == 3

    intent = "turn toward the radio and press its power control"
    low_prepared = _branch_prepared(
        inferencer, data, branch="low", intent_texts=[intent]
    )
    low_sample = low_prepared.sample["samples"]
    _assert_actual_preprocessed_pixel_values(low_prepared)
    assert low_sample["memlite_branch"] == "low"
    assert low_sample["intent"] == intent
    low_prefill = policy._clone_memlite_samples([low_sample], branch="low", intent=intent)[0]
    assert low_prefill["intent"] == intent
    assert low_prefill["template"].index("<EOV>") < low_prefill["template"].index("<EOC>")
    assert low_prefill["template"].count("_image_!") == 3

    # This follows the exact saved MixedBuilder high preprocess path but uses
    # a CPU facade instead of loading a VLM/checkpoint tensor. It proves the
    # bridge/wrapper's configured 768-token budget reaches the model API and
    # metadata returns without changing the high text protocol.
    capture = _SavedRuntimeHighCapCapture()
    cap_inferencer = PolicyInferencer(policy=capture, processor=processors, device="cpu")
    seeded_memory = "Task=0; Completed=none."
    cap_result = cap_inferencer.infer_high_level(
        [data], [seeded_memory], max_new_tokens=768
    )[0]
    assert capture.max_new_tokens == 768
    assert capture.memory_text == seeded_memory
    assert cap_result["raw"].endswith("<HL_END>")
    assert cap_result["generation_metadata"]["max_new_tokens"] == 768
    print("SAVED_MEMLITE_MIXED_BUILDER_HIGH_PREPROCESS=PASS")
    print("SAVED_MEMLITE_MIXED_BUILDER_LOW_PREPROCESS=PASS")
    print("HIGH_PREFILL_CURRENT_MEMORY_WITHOUT_TARGET=PASS")
    print("LOW_PREFILL_EXPLICIT_CURRENT_INTENT=PASS")
    print("SAVED_PROCESSOR_EVAL_18_TO_3_EXTERIOR_T0_T1_T2_REASSIGNED=PASS")
    print("SAVED_PROCESSOR_TRAIN_SEED7_LEFT_T0_T3_T5_REASSIGNED=PASS")
    print("ACTUAL_PREPROCESSED_PIXELS_THREE_OUTPUT_KEYS_ONE_FRAME_EACH=PASS")
    print("SAVED_RUNTIME_HIGH_CAP_768_TO_POLICY=PASS")
    print("SAVED_AR_FUTURE_ONLY_SENTINEL_ROWS_0_TO_15=PASS")
    print("SAVED_AR_FUTURE_ONLY_BRIDGE_ROWS_0_TO_15=PASS")


if __name__ == "__main__":
    main()
