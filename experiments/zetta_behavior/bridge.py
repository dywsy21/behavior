"""CPU-only boundaries for a frozen G0.5 + Zetta integration.

This is not a replacement recovery policy or an OmniGibson runner. G0.5's
existing server owns normalization, the 27-D internal representation, history,
and chunk caching. This module consumes its *postprocessed, single-step* reply.
Only an environment owner may subsequently send the 23-D command to the robot.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


# BEHAVIOR R1Pro raw controller order, not the G0.5 internal padded layout.
# Source: configs/data/behavior5_r1pro_memlite.yaml, shape_meta.action.
ACTION_PARTS = (
    ("base_qvel", 3),
    ("trunk_qpos", 4),
    ("left_arm", 7),
    ("left_gripper", 1),
    ("right_arm", 7),
    ("right_gripper", 1),
)


def action23_from_g05(response: Mapping[str, Any]) -> np.ndarray:
    """Validate and concatenate an existing serve_policy.py single-step reply.

    Missing AR components are an error, not permission to silently drop base or
    torso control. No clipping, gripper rescaling, padding removal, or relative
    joint conversion occurs here: those operations belong to the checkpoint's
    original processor and the simulator's existing action adapter.
    """
    if not isinstance(response, Mapping) or "error" in response:
        raise ValueError("G0.5 returned an error or a non-mapping response")
    action = response.get("action")
    expected = {name for name, _ in ACTION_PARTS}
    if not isinstance(action, Mapping) or set(action) != expected:
        raise ValueError("G0.5 must return all six raw BEHAVIOR action parts")
    parts = []
    for name, width in ACTION_PARTS:
        value = np.asarray(action[name])
        if value.shape != (width,) or value.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a numeric single-step ({width},) array")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains a non-finite value")
        if np.any(np.abs(value.astype(np.float64)) > np.finfo(np.float32).max):
            raise ValueError(f"{name} exceeds float32 range")
        parts.append(value.astype(np.float32, copy=True))
    return np.concatenate(parts)


class FrozenG05ClientBoundary:
    """Transport-independent policy boundary, with explicit recovery handoff.

    ``request`` must implement the existing msgpack WebSocket request/reply
    protocol. It is injected so CPU tests need neither a model nor a socket.
    This class never opens a simulator, adjudicates a proposal, or runs recovery.
    """

    def __init__(self, request: Callable[[dict], Mapping[str, Any]]) -> None:
        self._request = request
        self._needs_reset = True
        self._last_step: int | None = None
        self._fresh_chunk = False

    def reset_episode(self) -> None:
        # Mark unready *before* I/O: a failed reset must not reuse a stale chunk.
        self._needs_reset = True
        reply = self._request({"__reset__": True})
        if not isinstance(reply, Mapping) or dict(reply) != {"__reset__": True}:
            raise ValueError("G0.5 did not acknowledge cache reset")
        self._last_step = None
        self._fresh_chunk = True
        self._needs_reset = False

    def recovery_started(self) -> None:
        """Stop issuing policy actions until an explicit cache-reset handoff."""
        self._needs_reset = True

    def recovery_finished(self) -> None:
        """Discard the pre-recovery suffix; require a fresh observation next."""
        previous_step = self._last_step
        self.reset_episode()
        self._last_step = previous_step

    def policy_step(self, raw_obs: dict, *, step_index: int) -> np.ndarray:
        if self._needs_reset:
            raise RuntimeError("reset/acknowledged recovery handoff required")
        if type(step_index) is not int or step_index < 0:
            raise ValueError("step_index must be a nonnegative control-step integer")
        if self._last_step is not None:
            if step_index <= self._last_step or (
                not self._fresh_chunk and step_index != self._last_step + 1
            ):
                self._needs_reset = True
                raise ValueError("control clock discontinuity requires a cache-reset handoff")
        if not isinstance(raw_obs, dict) or not all(
            key in raw_obs for key in ("images", "state", "task")
        ) or "__reset__" in raw_obs:
            raise ValueError("fresh raw images/state/task required; no cached empty request")
        # Callers retain the existing checkpoint-specific history construction.
        # Errors leave this bridge blocked: transport failure may have advanced
        # the remote cache even if its reply never reached this process.
        self._needs_reset = True
        command = action23_from_g05(self._request(raw_obs))
        self._last_step = step_index
        self._fresh_chunk = False
        self._needs_reset = False
        return command


def inspect_rule_requirements(
    payload: Mapping[str, Any], *, public_features: frozenset[str]
) -> dict[str, Any]:
    """Report missing/privileged features without inventing feature substitutes.

    A complete CandidateBundle also needs frozen recovery steps and an audited
    embodiment-specific executor. Feature compatibility alone never enables it.
    """
    rules = payload.get("critic_rules", [])
    if not isinstance(rules, list) or not rules:
        raise ValueError("critic_rules must be a non-empty array")
    required = set()
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise ValueError("each critic rule must be an object")
        conditions = rule.get("activation_conditions", [])
        if not isinstance(conditions, list):
            raise ValueError("activation_conditions must be an array")
        for item in [rule, *conditions]:
            if not isinstance(item, Mapping):
                raise ValueError("each activation condition must be an object")
            feature = item.get("feature")
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError("every rule and condition needs a named feature")
            required.add(feature)
    privileged = sorted(name for name in required if name.startswith("privileged."))
    unavailable = sorted(required - public_features)
    return {
        "required_features": sorted(required),
        "privileged_features": privileged,
        "unavailable_public_features": unavailable,
        "public_observation_compatible": not privileged and not unavailable,
    }
