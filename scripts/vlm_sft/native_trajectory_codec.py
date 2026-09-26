"""Offline R1Pro composite-action candidate, not a deployed servo replacement.

The answer is a bounded 16-tick plan. Joint groups use piecewise-linear offsets
from the *current* rounded robot joint positions. Base commands use normalized
velocities; grippers use zero-order hold, preserving every quantized switch.
This module decodes controls but never steps a robot or certifies their safety.
"""
from __future__ import annotations

import json

import numpy as np


VERSION = "r1-composite16-v1"
HORIZON = 16
HZ = 30
JOINT_UNIT = 1e-4
COMMAND_UNIT = 1e-3
JOINT_FIT_UNITS = 5
BASE_FIT_UNITS = 2
MAX_TEXT_BYTES = 32768
MAX_INTEGER = 1_000_000
GROUP_WIDTHS = {"t": 4, "l": 7, "r": 7, "b": 3, "g": 2}
JOINT_ACTION = {"t": slice(3, 7), "l": slice(7, 14), "r": slice(15, 22)}
JOINT_ANCHOR = {"t": slice(0, 4), "l": slice(4, 11), "r": slice(11, 18)}
ERROR_LIMITS = {"joint_rad": (JOINT_FIT_UNITS + .5) * JOINT_UNIT,
                "base_normalized": (BASE_FIT_UNITS + .5) * COMMAND_UNIT,
                "gripper_command": .5 * COMMAND_UNIT}
SYSTEM = """Use current head, left-wrist and right-wrist RGB plus current robot proprioception to plan the next 16 control ticks of the R1Pro mobile two-arm robot at 30 Hz. LEFT and RIGHT mean robot arms, not image sides. Return only JSON with keys v,t,l,r,b,g; v is r1-composite16-v1. Each group is a list of [tick,values...] knots, beginning at tick 0, strictly increasing, at most tick 15. t,l,r contain 4,7,7 integer joint offsets in 0.0001 radians from the corresponding CURRENT q_rad joints (torso, left, right). Offsets are not cumulative. b contains 3 integer normalized base velocity commands in 0.001 units, for forward,left,yaw; physical scales are 0.75 m/s,0.75 m/s,1 rad/s. g contains left,right gripper commands in 0.001 units, +1000 open and -1000 close. Interpolate t,l,r,b linearly between knots; use either one constant knot or include tick 15. Hold g piecewise constant, changing only at the listed ticks. All groups execute together. No reasoning, future observation, success claim or object-state assertion. This is a control plan, not a claim of task completion."""


def finite(value, shape):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu" or a.shape != shape or not np.isfinite(a).all():
        raise ValueError("Finite numeric array with exact shape required")
    return a.astype(np.float64)


def current_anchor(state):
    s = finite(state, (61,))
    # The same six-decimal anchor is included in the model input and decoder.
    return np.r_[s[53:57], s[3:10], s[28:35]].round(6)


def _interpolate(knots, width, *, step=False):
    ticks = np.asarray([row[0] for row in knots], dtype=int)
    values = np.asarray([row[1:] for row in knots], dtype=float)
    x = np.arange(HORIZON)
    if step:
        return values[np.searchsorted(ticks, x, side="right") - 1]
    return np.stack([np.interp(x, ticks, values[:, i]) for i in range(width)], axis=1)


def _linear_knots(values, tolerance):
    """Bound every tick/channel, not only the final endpoint or mean error."""
    width = values.shape[1]
    if np.max(np.abs(values - values[0])) <= tolerance:
        return [[0, *values[0].tolist()]]
    indices = [0, HORIZON - 1]
    while True:
        knots = [[i, *values[i].tolist()] for i in indices]
        error = np.max(np.abs(_interpolate(knots, width) - values), axis=1)
        index = int(np.argmax(error))
        if error[index] <= tolerance + 1e-12:
            return knots
        if index in indices:
            raise RuntimeError("Interpolation failed at an existing knot")
        indices.append(index)
        indices.sort()


def _step_knots(values):
    indices = [0] + [i for i in range(1, HORIZON) if not np.array_equal(values[i], values[i - 1])]
    return [[i, *values[i].tolist()] for i in indices]


def _quantized(value, unit):
    scaled = np.rint(value / unit)
    if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) > MAX_INTEGER:
        raise ValueError("Control outside codec numeric bound; never clip")
    return scaled.astype(np.int64)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite JSON constant")


def parse(text):
    if not isinstance(text, str) or not 1 <= len(text.encode("utf-8")) <= MAX_TEXT_BYTES:
        raise ValueError("Bounded JSON text required")
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid control JSON") from exc
    if type(obj) is not dict or set(obj) != {"v", *GROUP_WIDTHS} or obj["v"] != VERSION:
        raise ValueError("Exact composite protocol and all five groups required")
    for group, width in GROUP_WIDTHS.items():
        knots = obj[group]
        if type(knots) is not list or not 1 <= len(knots) <= HORIZON:
            raise ValueError("One to sixteen knots per group required")
        previous = -1
        for row in knots:
            if type(row) is not list or len(row) != width + 1 or any(type(v) is not int for v in row):
                raise ValueError("Integer tick and exact-width integer commands required")
            tick, *values = row
            if not previous < tick < HORIZON or tick < 0 or max(map(abs, values)) > MAX_INTEGER:
                raise ValueError("Unordered tick or out-of-bound integer")
            if group in ("b", "g") and max(map(abs, values)) > 1000:
                raise ValueError("Normalized base/gripper range exceeded")
            previous = tick
        if knots[0][0] != 0 or (group != "g" and len(knots) > 1 and knots[-1][0] != HORIZON - 1):
            raise ValueError("Missing initial knot or final linear endpoint")
    return obj


def decode(text, anchor):
    """Return all sixteen native23 control vectors. Does not execute them.

    Any future live consumer must additionally validate the current observation
    clock, calibrated joint limits, per-tick tracking and collision constraints.
    JSON validity alone is not an execution authorization or safety certificate.
    """
    q = finite(anchor, (18,))
    if not np.array_equal(q, q.round(6)):
        raise ValueError("Decoder must use the exact displayed six-decimal anchor")
    obj = parse(text)
    result = np.zeros((HORIZON, 23), dtype=np.float64)
    for group, sl in JOINT_ACTION.items():
        result[:, sl] = q[JOINT_ANCHOR[group]] + JOINT_UNIT * _interpolate(obj[group], GROUP_WIDTHS[group])
    result[:, :3] = COMMAND_UNIT * _interpolate(obj["b"], 3)
    result[:, [14, 22]] = COMMAND_UNIT * _interpolate(obj["g"], 2, step=True)
    return result


def errors(original, reconstructed):
    a = finite(original, (HORIZON, 23))
    b = finite(reconstructed, (HORIZON, 23))
    return {"joint_rad": float(max(np.abs(a[:, sl] - b[:, sl]).max() for sl in JOINT_ACTION.values())),
            "base_normalized": float(np.abs(a[:, :3] - b[:, :3]).max()),
            "gripper_command": float(np.abs(a[:, [14, 22]] - b[:, [14, 22]]).max())}


def encode(actions, state):
    a = finite(actions, (HORIZON, 23))
    q = current_anchor(state)
    if np.abs(a[:, [0, 1, 2, 14, 22]]).max() > 1.00002:
        raise ValueError("Source normalized commands outside accepted native range")
    obj = {"v": VERSION}
    for group, sl in JOINT_ACTION.items():
        quantized = _quantized(a[:, sl] - q[JOINT_ANCHOR[group]], JOINT_UNIT)
        obj[group] = _linear_knots(quantized, JOINT_FIT_UNITS)
    obj["b"] = _linear_knots(_quantized(a[:, :3], COMMAND_UNIT), BASE_FIT_UNITS)
    obj["g"] = _step_knots(_quantized(a[:, [14, 22]], COMMAND_UNIT))
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    error = errors(a, decode(text, q))
    if any(error[k] > limit + 1e-10 for k, limit in ERROR_LIMITS.items()):
        raise RuntimeError("Composite target exceeds explicit round-trip error")
    return text


def actor_from_state(task, instruction, state):
    s = finite(state, (61,))
    actor = {"protocol": VERSION, "task": task, "active_instruction": instruction,
             "proprio": {"q_rad": current_anchor(s).tolist(),
                         "base_velocity_local": s[:3].round(6).tolist(),
                         "eef_base_m": [s[17:20].round(6).tolist(), s[42:45].round(6).tolist()],
                         "eef_quat_xyzw": [s[20:24].round(6).tolist(), s[45:49].round(6).tolist()],
                         "finger_joints_m": [s[24:26].round(6).tolist(), s[49:51].round(6).tolist()]}}
    validate_actor(actor)
    return actor


def validate_actor(actor):
    if (type(actor) is not dict or set(actor) != {"protocol", "task", "active_instruction", "proprio"}
            or actor["protocol"] != VERSION):
        raise ValueError("Current-only actor whitelist required")
    if any(not isinstance(actor[k], str) or not 1 <= len(actor[k]) <= 5000 for k in ("task", "active_instruction")):
        raise ValueError("Bounded task/instruction text required")
    p = actor["proprio"]
    shapes = {"q_rad": (18,), "base_velocity_local": (3,), "eef_base_m": (2, 3),
              "eef_quat_xyzw": (2, 4), "finger_joints_m": (2, 2)}
    if type(p) is not dict or set(p) != set(shapes):
        raise ValueError("Only current robot proprioception may be serialized")
    for key, shape in shapes.items():
        value = finite(p[key], shape)
        if not np.array_equal(value, value.round(6)):
            raise ValueError("Use the same six-decimal proprioception in training and decoding")
    if np.any(np.abs(np.linalg.norm(p["eef_quat_xyzw"], axis=1) - 1) > .002):
        raise ValueError("Current EEF quaternion must be unit length")
    return actor


def prompt(actor):
    a = validate_actor(actor)
    return (f'Task: {a["task"]}\nActive instruction: {a["active_instruction"]}\n'
            f'Current robot proprioception (left before right): {json.dumps(a["proprio"], separators=(",", ":"), allow_nan=False)}\n'
            'Next simultaneous 16-tick control plan:')
