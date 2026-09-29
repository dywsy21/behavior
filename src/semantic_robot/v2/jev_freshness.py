"""Conservative Jev pilot contract: a decision only applies to its frozen state.

For synchronized simulation, API wall time must not advance physics or change
any input. Fresh render/proprio are checked before servo.begin. A changed view
or state aborts, never automatically replans. This is not a real-robot latency
compensator; real deployment needs a different qualified freshness mechanism.
"""
import hashlib
import json

import numpy as np

VIEWS = ("head", "left_wrist", "right_wrist")


def array_digest(value):
    data = np.asarray(value)
    return {"shape": list(data.shape), "dtype": str(data.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()}


def snapshot(state, images, depths, receipts, controls):
    sensors = {}
    for view in VIEWS:
        native = receipts[view].get("native_time", {})
        if "simulation_time" not in native:
            raise ValueError("Jev freshness requires synchronized sensor timestamps")
        sensors[view] = {"rgb": array_digest(images[view + "_rgb"]),
                         "depth": array_digest(depths[view]),
                         "simulation_time": native["simulation_time"]}
    proprio = {name: array_digest(getattr(state, name)) for name in ("q", "gripper", "base_velocity")}
    fingers = getattr(state, "finger_qpos", None)
    # A JSON round trip removes mutable references from the before snapshot.
    return json.loads(json.dumps({"controls": controls, "sensors": sensors, "proprio": proprio,
                                  "finger_qpos": fingers}, sort_keys=True, allow_nan=False))


def freshness_check(before, after):
    fields = ("controls", "sensors", "proprio", "finger_qpos")
    changed = [key for key in fields if before.get(key) != after.get(key)]
    return {"passed": not changed, "changed_fields": changed,
            "reason": "EXACT_FROZEN_DECISION_STATE" if not changed else "JEV_DECISION_STATE_CHANGED",
            "policy": "synchronous_sim_only_exact_RGBD_proprio_clock_no_threshold_relaxation",
            "new_model_calls": 0, "before": before, "after": after}
