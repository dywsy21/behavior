"""Bind a decision to a frozen physical state, not stochastic renderer pixels.

H75 has six consecutive captures with identical physics clocks but differing
RGB-D hashes. A fresh PathTracing batch is allowed to differ; the pinned native
I/O contract still proves read-buffer identity, zero physics ticks per capture,
and advancing, matched per-camera render batches. This is simulation-only.
"""
import hashlib
import json
import math

import numpy as np

from .render_batch import validate_batch, validate_advance

VIEWS = ("head", "left_wrist", "right_wrist")


def array_digest(value):
    data = np.asarray(value)
    return {"shape": list(data.shape), "dtype": str(data.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()}


def snapshot(state, images, depths, receipts, controls):
    sensors, cameras, clocks, captures, boundaries = {}, {}, [], [], []
    for view in VIEWS:
        receipt = receipts[view]
        native = receipt.get("native_time", {})
        if (receipt.get("freshness_proof") != "native_render_batch_v2" or
                native.get("version") != "render_batch_v2" or
                native.get("render_batch_verified") is not True or native.get("wait_for_render") is not True or
                type(native.get("physics_ticks_in_capture")) is not int or native["physics_ticks_in_capture"] != 0):
            raise ValueError("Jev freshness requires verified synchronized render receipts")
        t, index, capture = native.get("simulation_time"), native.get("physics_index"), native.get("capture")
        if (type(t) not in (int,float) or not math.isfinite(t) or t < 0 or
                type(index) is not int or index < 0 or type(capture) is not int or capture < 1):
            raise ValueError("Invalid native clock/capture identity")
        # Onboard RGB receipts hash the copied HWC buffer; runner images are CHW.
        rgb = array_digest(np.asarray(images[view + "_rgb"]).transpose(1,2,0))
        depth = array_digest(depths[view])
        if rgb["sha256"] != receipt.get("rgb_sha256") or depth["sha256"] != receipt.get("depth_sha256"):
            raise ValueError("Jev snapshot differs from verified native read buffers")
        sensors[view] = {"rgb":rgb, "depth":depth}
        cameras[view] = native["camera"]
        clocks.append({"simulation_time":t, "physics_index":index})
        captures.append(capture)
        boundaries.append({k:native[k] for k in ("scheduled","completed")})
    if any(c != clocks[0] for c in clocks) or len(set(captures)) != 1 or any(b != boundaries[0] for b in boundaries):
        raise ValueError("Torn Jev snapshot across cameras")
    batch = {**boundaries[0], "cameras":cameras}
    validate_batch(batch)
    proprio = {name: array_digest(getattr(state,name)) for name in ("q","gripper","base_velocity")}
    return json.loads(json.dumps({"controls":controls,"sensors":sensors,"proprio":proprio,
        "finger_qpos":getattr(state,"finger_qpos",None), "clock":clocks[0],
        "capture":captures[0], "batch":batch}, sort_keys=True, allow_nan=False))


def freshness_check(before, after):
    changed = [key for key in ("controls","clock","proprio","finger_qpos") if before.get(key) != after.get(key)]
    try:
        validate_advance(after["batch"], before["batch"])
        if after["capture"] <= before["capture"]: raise ValueError("Nonadvancing capture")
        for view in VIEWS:
            for kind in ("rgb","depth"):
                if any(before["sensors"][view][kind][k] != after["sensors"][view][kind][k] for k in ("shape","dtype")):
                    raise ValueError("Sensor layout changed")
    except (KeyError,TypeError,ValueError):
        changed.append("render_batch_or_sensor_layout")
    return {"passed":not changed, "changed_fields":changed,
        "reason":"FROZEN_PHYSICS_FRESH_NATIVE_RENDER" if not changed else "JEV_DECISION_STATE_CHANGED",
        "policy":"synchronous_sim_only_exact_physics_proprio_advancing_verified_render",
        "render_pixels_changed":before.get("sensors") != after.get("sensors"),
        "new_model_calls":0, "before":before, "after":after}
