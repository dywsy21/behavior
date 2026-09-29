"""Pure serialization for the official BEHAVIOR step diagnostic trace."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_sim_step_payload(
    *, action: Any, terminated: bool, truncated: bool, info: Any, phase: int
) -> dict[str, Any]:
    """Use only values returned by ``env.step``; never query simulator state."""
    done = info.get("done", {}) if isinstance(info, dict) else {}
    conditions = done.get("termination_conditions", {}) if isinstance(done, dict) else {}
    timeout = conditions.get("timeout") if isinstance(conditions, dict) else None
    # BehaviorTask adds goal_status to the done-info dictionary before
    # TaskBase wraps it under info["done"]. Retain the top-level fallback for
    # other task implementations and older evaluator variants.
    goal_status = done.get("goal_status", info.get("goal_status")) if isinstance(done, dict) else None
    return {
        "source": "official_sim",
        "event": "env_step",
        "wall_time_unix": time.time(),
        "chunk_phase": int(phase),
        "actual_action": json_safe(action),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": json_safe(done.get("success")) if isinstance(done, dict) else None,
        "timeout": json_safe(timeout),
        "goal_status": json_safe(goal_status),
        "termination_conditions": json_safe(conditions),
    }
