"""Small, explicit H-09 actor contract and conservative expert motion codec.

The codec targets the *direction* of a same-state expert segment. It does not
assert success and does not claim to reproduce simultaneous 23D commands.
Unrepresentable segments are rejected, never silently mapped to HOLD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = "h09-semantic-motion-v1"
CAMERAS = {"head": "zed_link_camera_0", "left_wrist": "left_realsense_link_camera_0",
           "right_wrist": "right_realsense_link_camera_0"}
POSITIONS = {"left": slice(17, 20), "right": slice(42, 45)}
QUATERNIONS = {"left": slice(20, 24), "right": slice(45, 49)}
JOINTS = {"left": slice(3, 10), "right": slice(28, 35), "torso": slice(53, 57)}
GRIPS = {"left": slice(24, 26), "right": slice(49, 51)}
GRIP_ACTION = {"left": 14, "right": 22}
MOVE_NAMES = ("FORWARD", "LEFT", "UP")
NEG_NAMES = ("BACK", "RIGHT", "DOWN")
ROT_NAMES = ("ROLL", "PITCH", "YAW")
TOKENS = tuple([f"{p}_{d}" for p in ("LEFT", "RIGHT", "BOTH")
                for d in (*MOVE_NAMES, *NEG_NAMES)] +
               [f"{p}_{r}_{s}" for p in ("LEFT", "RIGHT") for r in ROT_NAMES for s in ("PLUS", "MINUS")] +
               [f"{p}_{g}" for p in ("LEFT", "RIGHT") for g in ("OPEN", "CLOSE")] +
               [f"BASE_{d}" for d in ("FORWARD", "BACK", "LEFT", "RIGHT", "YAW_PLUS", "YAW_MINUS")] +
               [f"TORSO_{d}" for d in ("FORWARD", "BACK", "UP", "DOWN")] + ["HOLD"])

SYSTEM = """Control a mobile two-arm robot using its current onboard RGB views and robot-relative proprioception. Images are not mirrored. The robot base frame is FORWARD +x, LEFT +y, UP +z. LEFT and RIGHT name the robot arms, not the image side. Choose one small semantic motion from the supplied vocabulary. Use the current task and active instruction; previous motions are context, not a command to repeat. OPEN and CLOSE are gripper commands, not claims of success. HOLD safely keeps the current pose. Output the single motion symbol only, without reasoning, punctuation or extra text."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def clean_object(value):
    """Remove simulator identity suffixes; retain ordinary category words."""
    if not value or value in ("NONE", "UNSPECIFIED"):
        return ""
    text = str(value)
    text = re.sub(r"_\d+$", "", text)
    text = re.sub(r"_[a-z0-9]{6}$", "", text)
    # Some annotations have comma-separated objects; each is sanitized alone.
    return text.replace("_", " ").strip()


def skill_text(raw):
    rows = json.loads(raw) if isinstance(raw, str) else raw
    result = []
    for row in rows:
        if set(row) - {"arm", "destination", "source", "target", "target_part", "unbound_relation", "verb"}:
            raise ValueError("Unexpected semantic skill fields")
        if row.get("unbound_relation"):
            raise ValueError("Unbound target relation")
        fields = [(key, clean_object(row.get(key, ""))) for key in
                  ("verb", "target", "source", "destination", "target_part", "arm")]
        result.append("; ".join(f"{key}={value}" for key, value in fields if value))
    if not result:
        raise ValueError("Missing same-state active skill")
    return " | ".join(result)


def actor_state(state):
    s = np.asarray(state, dtype=float)
    if s.shape != (61,) or not np.isfinite(s).all():
        raise ValueError("Expected finite current 61D proprioception")
    return {"eef_base_m": {arm: s[sl].round(3).tolist() for arm, sl in POSITIONS.items()},
            "finger_opening_m": {arm: round(float(s[sl].mean()), 3) for arm, sl in GRIPS.items()},
            "base_velocity_local": s[:3].round(3).tolist(), "torso_joints_rad": s[53:57].round(3).tolist()}


def prompt(task, active_instruction, proprio, history):
    if set(proprio) != {"eef_base_m", "finger_opening_m", "base_velocity_local", "torso_joints_rad"}:
        raise ValueError("Actor state whitelist mismatch")
    if any(x not in TOKENS for x in history) or len(history) > 5:
        raise ValueError("History must be at most five actually executed motion symbols")
    return (f"Task: {task}\nActive instruction: {active_instruction}\n"
            f"Current proprioception: {json.dumps(proprio, separators=(',', ':'))}\n"
            f"Recent executed motions, oldest first: {json.dumps(history)}\n"
            "Available motions: " + ", ".join(TOKENS) + "\nNext motion:")


def directional(v, positive=MOVE_NAMES, negative=NEG_NAMES, minimum=.006, purity=.35):
    v = np.asarray(v, dtype=float)
    i = int(np.argmax(np.abs(v)))
    main = abs(float(v[i]))
    residual = float(np.linalg.norm(np.delete(v, i)))
    if main < minimum or residual > purity * main:
        return None
    return (positive if v[i] > 0 else negative)[i]


def classify_window(states, actions, previous_action=None):
    """Return a token only if one primitive explains the bounded expert window.

    states includes t through t+H; actions includes t through t+H-1. All
    geometric evidence is an offline target. No future field reaches prompt().
    """
    s, a = np.asarray(states, float), np.asarray(actions, float)
    if s.ndim != 2 or s.shape[1] != 61 or a.shape != (len(s)-1, 23) or len(a) < 4:
        raise ValueError("Misaligned same-state expert window")
    if not np.isfinite(s).all() or not np.isfinite(a).all():
        return None, {"reject": "nonfinite_source"}
    dp = {p: s[-1, sl]-s[0, sl] for p, sl in POSITIONS.items()}
    rv = {p: (Rotation.from_quat(s[-1, sl]) * Rotation.from_quat(s[0, sl]).inv()).as_rotvec()
          for p, sl in QUATERNIONS.items()}
    dq = {p: float(np.max(np.ptp(s[:, sl], axis=0))) for p, sl in JOINTS.items()}
    dg = {p: float(s[-1, sl].mean()-s[0, sl].mean()) for p, sl in GRIPS.items()}
    evidence = {"delta_eef_base_m": {p: v.tolist() for p, v in dp.items()},
                "delta_rotation_base_rad": {p: v.tolist() for p, v in rv.items()},
                "joint_range_rad": dq, "delta_finger_m": dg,
                "raw_base_command_mean": a[:, :3].mean(0).tolist()}
    base = a[:, :3] * np.array([.75, .75, 1.])
    base_abs = np.max(np.abs(base), axis=0)
    arm_quiet = max(np.linalg.norm(v) for v in dp.values()) < .007 and max(np.linalg.norm(v) for v in rv.values()) < .06
    grips_quiet = max(abs(v) for v in dg.values()) < .005
    if float(base_abs.max()) > .04:
        # Pure direction, consistent throughout the first eight commands; no
        # conversion of the simulator's biased velocity integral into truth.
        scaled = base / [.12, .12, .25]
        mean = scaled.mean(0)
        d = directional(mean, ("FORWARD", "LEFT", "YAW_PLUS"), ("BACK", "RIGHT", "YAW_MINUS"), .15)
        axis = int(np.argmax(np.abs(mean)))
        stable = bool(np.all(scaled[:8, axis] * np.sign(mean[axis]) > .1))
        if d and stable and arm_quiet and grips_quiet and dq["torso"] < .01:
            return "BASE_"+d, evidence
        return None, dict(evidence, reject="mixed_or_transitional_base")
    # Gripper targets are supervised as commands, never inferred successful.
    changes = [p for p in GRIPS if abs(dg[p]) >= .01]
    if len(changes) == 1 and arm_quiet and dq["torso"] < .01:
        p = changes[0]
        values = a[:, GRIP_ACTION[p]]
        sign = np.sign(dg[p])
        if np.all(values[:4]*sign > .15):
            return p.upper() + ("_OPEN" if sign > 0 else "_CLOSE"), evidence
    if not grips_quiet:
        return None, dict(evidence, reject="mixed_gripper_motion")
    if dq["torso"] >= .01:
        mean = (dp["left"] + dp["right"])/2
        d = directional(mean)
        if (d in ("FORWARD", "BACK", "UP", "DOWN") and max(dq[p] for p in POSITIONS) < .01
                and np.linalg.norm(dp["left"]-dp["right"]) < .005
                and max(np.linalg.norm(v) for v in rv.values()) < .04):
            return "TORSO_"+d, evidence
        return None, dict(evidence, reject="torso_motion_unrepresentable")
    moving = [p for p in POSITIONS if np.linalg.norm(dp[p]) >= .006 or np.linalg.norm(rv[p]) >= .035]
    if len(moving) == 2:
        d = directional((dp["left"]+dp["right"])/2)
        if d and np.linalg.norm(dp["left"]-dp["right"]) < .004 and max(np.linalg.norm(v) for v in rv.values()) < .035:
            return "BOTH_"+d, evidence
        return None, dict(evidence, reject="uncoordinated_two_arm_motion")
    if len(moving) == 1:
        p = moving[0]
        d = directional(dp[p])
        if d and np.linalg.norm(rv[p]) < .035:
            # Reject reversal within a window even if endpoints look pure.
            increments = np.diff(s[:, POSITIONS[p]], axis=0)
            axis = int(np.argmax(np.abs(dp[p])))
            reverse = np.maximum(-increments[:, axis]*np.sign(dp[p][axis]), 0).sum()
            if reverse <= .0015:
                return p.upper()+"_"+d, evidence
        r = directional(rv[p], tuple(x+"_PLUS" for x in ROT_NAMES), tuple(x+"_MINUS" for x in ROT_NAMES), .035)
        if r and np.linalg.norm(dp[p]) < .006:
            return p.upper()+"_"+r, evidence
        return None, dict(evidence, reject="mixed_or_curved_arm_motion")
    if previous_action is not None and np.max(np.abs(np.asarray(previous_action)[:3])) > .08 and base_abs.max() < .01:
        return "HOLD", dict(evidence, hold_reason="expert_base_braking_transition")
    return None, dict(evidence, reject="stationary_or_subthreshold_not_success")


def token_to_action(token):
    from semantic_robot.v2.protocol import Action, HOLD
    if token not in TOKENS:
        raise ValueError("Unknown semantic motion")
    if token == "HOLD":
        return HOLD
    part, move = token.split("_", 1)
    scale = "coarse" if part == "BASE" and move.startswith("YAW") else "fine"
    return Action(part.lower(), move.lower(), scale, "base")
