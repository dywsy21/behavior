"""Two checkpoints, processors and caches; semantic state stays in the runtime."""
from __future__ import annotations

from copy import deepcopy

import torch


def latest_observation(obs: dict) -> dict:
    """Reduce ONLY observation history, never actions or physical state values."""
    result = deepcopy(obs)
    for group in ("images", "state"):
        if not isinstance(result.get(group), dict) or not result[group]:
            raise ValueError(f"Separated FM controller requires observation {group}")
        for key, value in result[group].items():
            if not hasattr(value, "shape") or len(value.shape) < 2 or value.shape[0] < 1:
                raise ValueError(f"Missing chronological observation axis: {group}.{key}")
            result[group][key] = value[-1:].clone() if isinstance(value, torch.Tensor) else value[-1:].copy()
    for key in ("state_is_pad", "image_is_pad"):
        if key in result:
            value = result[key]
            result[key] = value[-1:].clone() if isinstance(value, torch.Tensor) else value[-1:].copy()
    return result


class SeparatedMEMLiteFMInferencer:
    """Duck-type the existing causal runtime, without sharing policy weights."""

    def __init__(self, *, high, low):
        if high is low or high.policy is low.policy:
            raise ValueError("MEM-Lite FM high/low policies must be independent objects")
        if not getattr(high.policy, "predict_cot", False):
            raise ValueError("High policy must support AR memory/intent generation")
        if not getattr(low.policy, "continuous_action", False) or getattr(low.policy, "discrete_action", True):
            raise ValueError("Low policy must execute FM, never AR action decoding")
        self.high = high
        self.low = low
        # The causal wrapper identifies planner capability through .policy;
        # actual action calls always go to self.low below.
        self.policy = high.policy
        self.processor = high.processor

    def infer_high_level(self, obs_dicts, memory_texts=None, *, max_new_tokens=1024):
        return self.high.infer_high_level(obs_dicts, memory_texts,
                                          max_new_tokens=max_new_tokens)

    def infer_low_level_action(self, obs_dicts, intent_texts):
        return self.low.infer_low_level_action([latest_observation(obs) for obs in obs_dicts], intent_texts)

    def infer(self, obs_dicts):
        raise RuntimeError("Separated MEM-Lite must not silently fall back to a different control route")
