"""Versioned current-robot-only actor input; no teacher/outcome dependencies.

Capture identities and clocks are verified OUTSIDE the actor prompt. Live and
offline callers use the same projection and prompt, for base and adapter alike.
Legacy H09 inputs are not silently upgraded or changed.
"""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import CAMERAS, TOKENS, SYSTEM as LEGACY_SYSTEM, sha
from live import runtime_proprio as legacy_proprio, validate_proprio as legacy_validate

VERSION = "h09x-current-robot-pose-v1"
SYSTEM = LEGACY_SYSTEM + " Current joint positions and both end-effector rotation matrices are robot proprioception, not object estimates. Each rotation matrix maps end-effector local axes into the robot base frame (columns are the local axes in base coordinates). Rotational motion symbols are about robot BASE axes, not camera or tool axes."
JOINT_GROUPS = {"torso": slice(0, 4), "left": slice(4, 11), "right": slice(11, 18)}
PROPRIO_KEYS = {"eef_base_m", "finger_opening_m", "base_velocity_local", "torso_joints_rad",
                "joint_positions_rad", "eef_rotation_base"}
ACTOR_KEYS = {"protocol", "task", "active_instruction", "proprio", "current_rgb_sha256", "history"}


def check_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Exact SHA256 required")


def check_clock(clock):
    if (not isinstance(clock, dict) or set(clock) != {"prefix_control", "native_control"} or
            any(type(v) is not int or v < 0 for v in clock.values())):
        raise ValueError("Exact nonnegative integer capture clock required")


def finite(value, shape):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu" or a.shape != shape or not np.isfinite(a).all():
        raise ValueError("Finite numeric robot array with exact shape required")
    return a.astype(float)


def rotation(value):
    r = finite(value, (3, 3))
    if (not np.allclose(r.T @ r, np.eye(3), atol=4e-6, rtol=0) or
            abs(np.linalg.det(r)-1) > 4e-6):
        raise ValueError("Proper robot rotation matrix required")
    return r


def validate_proprio(value):
    if not isinstance(value, dict) or set(value) != PROPRIO_KEYS:
        raise ValueError("Only this version's current robot fields are allowed")
    legacy_validate({k: v for k, v in value.items() if k not in {"joint_positions_rad", "eef_rotation_base"}})
    finite(value["base_velocity_local"], (3,)); finite(value["torso_joints_rad"], (4,))
    for arm in ("left", "right"):
        finite(value["eef_base_m"][arm], (3,)); finite(value["finger_opening_m"][arm], ())
    if set(value["joint_positions_rad"]) != set(JOINT_GROUPS) or set(value["eef_rotation_base"]) != {"left", "right"}:
        raise ValueError("Exact torso/left/right joint and two rotation groups required")
    q = {name: finite(value["joint_positions_rad"][name], (sl.stop-sl.start,)) for name, sl in JOINT_GROUPS.items()}
    if not np.allclose(q["torso"], value["torso_joints_rad"], atol=.000501, rtol=0):
        raise ValueError("Duplicate torso proprioception disagrees")
    for arm in ("left", "right"):
        rotation(value["eef_rotation_base"][arm])


def runtime_proprio(model, state, *, clock, expected_clock, calibration_sha256):
    check_clock(clock); check_clock(expected_clock); check_sha(calibration_sha256)
    if clock != expected_clock or model.sha != calibration_sha256:
        raise ValueError("Observation clock or fixed calibration identity mismatch")
    q = finite(state.q, (18,)); finite(state.gripper, (2,)); finite(state.base_velocity, (3,))
    value = legacy_proprio(state)
    value["joint_positions_rad"] = {name: q[sl].round(6).tolist() for name, sl in JOINT_GROUPS.items()}
    value["eef_rotation_base"] = {}
    for arm in ("left", "right"):
        t = finite(model.forward(q, arm), (4, 4))
        if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
            raise ValueError("Invalid FK homogeneous transform")
        rotation(t[:3, :3])
        if not np.allclose(t[:3, 3], finite(state.poses[arm][0], (3,)), atol=1e-6, rtol=0):
            raise ValueError("Current EEF position is stale relative to joint FK")
        value["eef_rotation_base"][arm] = t[:3, :3].round(6).tolist()
    validate_proprio(value)
    return value


def actor_input(task, instruction, proprio, image_hashes, history):
    validate_proprio(proprio)
    if any(not isinstance(s, str) or not 1 <= len(s) <= 5000 for s in (task, instruction)):
        raise ValueError("Legal task and active instruction strings required")
    if not isinstance(image_hashes, dict) or set(image_hashes) != set(CAMERAS):
        raise ValueError("Three current onboard RGB identities required")
    for h in image_hashes.values(): check_sha(h)
    if type(history) is not list or len(history) > 5 or any(t not in TOKENS for t in history):
        raise ValueError("At most five actually completed motion symbols required")
    return {"protocol": VERSION, "task": task, "active_instruction": instruction,
            "proprio": proprio, "current_rgb_sha256": image_hashes, "history": history.copy()}


def validate_actor(value):
    if not isinstance(value, dict) or set(value) != ACTOR_KEYS or value["protocol"] != VERSION:
        raise ValueError("Exact actor protocol whitelist required")
    return actor_input(value["task"], value["active_instruction"], value["proprio"],
                       value["current_rgb_sha256"], value["history"])


def prompt(actor):
    v = validate_actor(actor)
    # No capture clock, calibration hash, dataset instance, teacher pose, goal
    # status or future field is included in the model's text.
    return (f"Task: {v['task']}\nActive instruction: {v['active_instruction']}\n"
            f"Current proprioception: {json.dumps(v['proprio'], sort_keys=True, separators=(',', ':'))}\n"
            f"Recent executed motions, oldest first: {json.dumps(v['history'])}\n"
            "Available motions: " + ", ".join(TOKENS) + "\nNext motion:")


def inference_row(actor):
    return {"protocol": VERSION, "actor": validate_actor(actor), "text": prompt(actor)}


def training_row(actor, target):
    if target not in TOKENS: raise ValueError("Unknown target motion")
    # This constructs a protocol row, NOT an outcome/review release certificate.
    return {**inference_row(actor), "target": target}


def request_payload(actor, images, variant):
    """Use original PNG bytes, so current image identity is actually verifiable."""
    validate_actor(actor)
    if variant not in ("base", "finetuned") or set(images) != set(CAMERAS):
        raise ValueError("Registered model variant and three views required")
    payload = {}
    for view in CAMERAS:
        raw = images[view]
        if type(raw) is not bytes or len(raw) > 3*1024**2 or hashlib.sha256(raw).hexdigest() != actor["current_rgb_sha256"][view]:
            raise ValueError("Current raw image bytes disagree with actor identity")
        payload[view] = base64.b64encode(raw).decode()
    return {"actor": actor, "images": payload, "variant": variant}


def parse_request(value, *, registered_instructions):
    if set(value) != {"actor", "images", "variant"} or value["variant"] not in ("base", "finetuned"):
        raise ValueError("Unexpected request fields or variant")
    actor = validate_actor(value["actor"])
    if actor["active_instruction"] not in registered_instructions:
        raise ValueError("Instruction not registered for this experiment")
    if set(value["images"]) != set(CAMERAS): raise ValueError("Exactly three current images required")
    images = {}
    for view, encoded in value["images"].items():
        if not isinstance(encoded, str) or len(encoded) > 4*1024**2: raise ValueError("Oversized image")
        raw = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(raw).hexdigest() != actor["current_rgb_sha256"][view]: raise ValueError("Wrong current RGB")
        image = Image.open(BytesIO(raw))
        size = 720 if view == "head" else 480
        if image.format != "PNG" or image.size != (size, size): raise ValueError("Wrong raw onboard image layout")
        images[view] = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    return inference_row(actor), images


def from_capture(folder, model, *, capture_sha256, expected_clock, calibration_sha256):
    """Read a CURRENT frozen capture only; never open private teacher ledgers.

    The caller supplies hashes from the independently reviewed run manifest.
    Output binding is audit metadata, never passed to actor_input/prompt.
    """
    folder = Path(folder)
    check_sha(capture_sha256)
    if sha(folder/"capture.json") != capture_sha256: raise ValueError("Capture bytes changed")
    c = json.loads((folder/"capture.json").read_text())
    files = {"head.png", "left_wrist.png", "right_wrist.png", "depth.npz",
             "robot_self_geometry.json", "sensors.json", "proprio.json"}
    if (set(c) != {"clock", "q", "gripper", "kinematic_model_sha256", "source", "scene_truth", "array_layout", "files_sha256", "depth_array_sha256"} or
            c.get("scene_truth") is not False or c.get("source") != "render_only_current_onboard_RGBD_and_robot_joint_FK" or
            set(c["files_sha256"]) != files or c["kinematic_model_sha256"] != calibration_sha256):
        raise ValueError("Not a bound current robot-only capture")
    for name, expected in c["files_sha256"].items():
        check_sha(expected)
        if sha(folder/name) != expected: raise ValueError("Capture payload bytes changed")
    old = json.loads((folder/"proprio.json").read_text()); legacy_validate(old)
    state = model.state(finite(c["q"], (18,)), finite(c["gripper"], (2,)), old["base_velocity_local"])
    if legacy_proprio(state) != old: raise ValueError("Current capture q/gripper disagree with saved proprioception")
    proprio = runtime_proprio(model, state, clock=c["clock"], expected_clock=expected_clock,
                             calibration_sha256=calibration_sha256)
    hashes = {v: c["files_sha256"][v+".png"] for v in CAMERAS}
    binding = {"capture_sha256": capture_sha256, "calibration_sha256": calibration_sha256,
               "clock": dict(c["clock"]), "protocol": VERSION, "not_actor_input": True}
    return proprio, hashes, binding
