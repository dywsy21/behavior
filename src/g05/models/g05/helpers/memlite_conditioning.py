"""Versioned MEM-Lite execution feedback and anti-shortcut conditioning.

Only model input copies are modified. Ground-truth actions, physical proprio
and the raw-state anchors used by postprocessing remain untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import torch


@dataclass(frozen=True)
class MEMLiteConditioning:
    enabled: bool = False
    velocity_dropout: float = 0.5
    inference_velocity: str = "masked"
    velocity_indices: tuple[int, ...] = (24, 25, 26)
    proprio_dim: int = 27

    @classmethod
    def from_config(cls, value: Mapping | None):
        value = dict(value or {})
        result = cls(enabled=bool(value.get("enabled", False)),
                     velocity_dropout=float(value.get("velocity_dropout", .5)),
                     inference_velocity=str(value.get("inference_velocity", "masked")),
                     velocity_indices=tuple(value.get("velocity_indices", (24, 25, 26))),
                     proprio_dim=int(value.get("proprio_dim", 27)))
        if not math.isfinite(result.velocity_dropout) or not 0 <= result.velocity_dropout <= 1:
            raise ValueError("velocity_dropout must be finite and in [0, 1]")
        if result.inference_velocity not in {"observed", "masked"}:
            raise ValueError("inference_velocity must be observed or masked")
        if len(result.velocity_indices) != 3 or len(set(result.velocity_indices)) != 3:
            raise ValueError("Exactly three distinct base velocity dimensions are required")
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < result.proprio_dim
               for i in result.velocity_indices):
            raise ValueError("Base velocity dimensions must be valid integer indices")
        return result

    def prepare(self, samples, *, training: bool):
        if not self.enabled:
            return samples
        output = []
        for sample in samples:
            branch = sample.get("memlite_branch")
            if branch not in {"high", "low"}:
                raise ValueError("Versioned conditioning requires explicit high/low MEM-Lite samples")
            item = dict(sample)
            if item.get("memlite_conditioning_version") == 1:
                output.append(item)
                continue
            marker = "State: <proprio_proprio_!>;"
            if item.get("template", "").count(marker) != 1:
                raise ValueError("MEM-Lite template must contain exactly one proprio conditioning marker")
            item["template"] = item["template"].replace(marker,
                "Execution feedback: <execution_feedback_text_!> "
                "Base velocity observation: <base_velocity_observation_text_!> " + marker)
            feedback = item.get("execution_feedback", "none")
            if not isinstance(feedback, str) or not feedback.strip() or len(feedback) > 1024:
                raise ValueError("execution_feedback must be a nonempty bounded observation string")
            item["execution_feedback"] = feedback
            mode = "observed"
            if branch == "low":
                override = item.get("memlite_velocity_override")
                if override is not None:
                    if override not in {"observed", "masked"}:
                        raise ValueError("Invalid explicit velocity observation override")
                    mode = override
                elif training:
                    mode = "masked" if float(torch.rand(())) < self.velocity_dropout else "observed"
                else:
                    mode = self.inference_velocity
            item["base_velocity_observation"] = "withheld (unknown, not zero speed)" if mode == "masked" else "observed"
            if mode == "masked":
                prop = item.get("proprio")
                value = prop.get("value") if isinstance(prop, dict) else prop
                if value is None:
                    raise ValueError("Cannot mask missing proprio")
                value = torch.as_tensor(value)
                if value.ndim != 2 or value.shape[-1] != self.proprio_dim:
                    raise ValueError(f"Expected [history, {self.proprio_dim}] proprio; got {tuple(value.shape)}")
                masked = value.clone()
                masked[:, self.velocity_indices] = 0
                if isinstance(prop, dict):
                    item["proprio"] = {**prop, "value": masked}
                else:
                    item["proprio"] = masked
            item["memlite_conditioning_version"] = 1
            output.append(item)
        return output
