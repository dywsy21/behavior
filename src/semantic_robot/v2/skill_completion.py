"""Public evidence for a learned GRASP completion *request*, never task truth.

Only recorded onboard captures and actually completed native micro-actions are
accepted. This module neither advances a goal nor commands another lift. The
caller must separately score the endpoint; public holding is not skill success.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .grounding import GroundedEvidence, localize_target, validate_depth
from .grasp_motion import GraspMotionVerifier
from .grounded_harness import GroundedHarness
from .harness import Goal
from .odometry import RGBDMotion
from .protocol import Action
from .self_odometry import make_frame
from .servo import execution_completed

VERSION = "h43-public-grasp-request-v1"
VIEWS = ("head", "left_wrist", "right_wrist")
FILES = {v + ".png" for v in VIEWS} | {
    "depth.npz", "sensors.json", "proprio.json", "robot_self_geometry.json"}
CAPTURE_KEYS = {"clock", "q", "gripper", "kinematic_model_sha256", "source",
                "scene_truth", "array_layout", "files_sha256", "depth_array_sha256"}
LOCALIZE_SYSTEM = """Locate the named target anew in the CURRENT RAW images. Choose a visible physical textured surface on its rim or side, not a gripper, floor, empty hole or background. The point is only for RGB-D surface tracking, not a complete grasp pose or an assertion of holding. Prefer a wrist view only if the target surface is clearly identifiable. If no target surface can be identified, abstain.
Return one JSON object with exactly: visible(boolean), view(head/left_wrist/right_wrist/none), target_uv(normalized[x,y] in CURRENT RAW view or null), enclosed:null, co_moving:null, supported:null, effect:null, hazard(none/occluded), note(one short visible identity fact), other_views:[], target_reference:unknown. Never infer success from the instruction or from closed fingers."""


@dataclass(frozen=True)
class CaptureRef:
    directory: Path
    sha256: str
    clock: dict


@dataclass(frozen=True)
class ExecutedMotionRef:
    token: str
    execution_path: Path
    execution_sha256: str
    before: CaptureRef
    after: CaptureRef


@dataclass(frozen=True)
class CapturedState:
    """The localizer receives public pixels/proprio only, no execution/result."""
    ref: CaptureRef
    state: object
    images: dict
    depths: dict
    geometry: dict
    png_sha256: dict
    snapshot_id: int

    @property
    def control(self):
        return sum(self.ref.clock.values())


def _deadline(deadline):
    if (type(deadline) not in (int, float) or not np.isfinite(deadline) or
            time.perf_counter() >= deadline):
        raise TimeoutError("Completion verifier deadline expired")


def _sha(value):
    if (type(value) is not str or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Exact SHA256 identity required")
    return value


def _clock(value):
    if (type(value) is not dict or set(value) != {"prefix_control", "native_control"}
            or any(type(n) is not int or n < 0 for n in value.values())):
        raise ValueError("Exact nonnegative integer capture clock required")
    return value


def _array(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError("Finite robot array with exact shape required")
    return a.astype(float)


def _bytes(path, expected, limit=32 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Bounded regular public evidence file required")
    data = path.read_bytes()
    if len(data) > limit or hashlib.sha256(data).hexdigest() != _sha(expected):
        raise ValueError("Public evidence bytes changed")
    return data


def read_capture(ref, model):
    """Hash-bind the complete current sensor snapshot before any VLM call."""
    if type(ref) is not CaptureRef:
        raise ValueError("Explicit public CaptureRef required")
    folder = Path(ref.directory)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Regular capture directory required")
    clock = dict(_clock(ref.clock))
    c = json.loads(_bytes(folder / "capture.json", ref.sha256, 256 * 1024))
    if (set(c) != CAPTURE_KEYS or c["scene_truth"] is not False or
            c["source"] != "render_only_current_onboard_RGBD_and_robot_joint_FK" or
            c["kinematic_model_sha256"] != model.sha or _clock(c["clock"]) != clock or
            set(c["files_sha256"]) != FILES or set(c["depth_array_sha256"]) != set(VIEWS) or
            set(c["array_layout"]) != set(VIEWS)):
        raise ValueError("Not the expected current onboard capture")
    raw = {name: _bytes(folder / name, digest) for name, digest in c["files_sha256"].items()}
    from io import BytesIO
    images = {}
    snapshots = set()
    for view in VIEWS:
        camera = model.spec["metadata"]["cameras"][view]
        with Image.open(BytesIO(raw[view + ".png"])) as im:
            if im.mode != "RGB" or im.size != (camera["width"], camera["height"]):
                raise ValueError("Calibrated raw RGB image required")
            images[view] = np.asarray(im).copy()
    with np.load(BytesIO(raw["depth.npz"]), allow_pickle=False) as archive:
        if set(archive.files) != set(VIEWS):
            raise ValueError("Exactly three onboard depth images required")
        depths = {v: archive[v].copy() for v in VIEWS}
    sensors = json.loads(raw["sensors.json"])
    if set(sensors) != set(VIEWS):
        raise ValueError("Three current sensor receipts required")
    for view in VIEWS:
        camera = model.spec["metadata"]["cameras"][view]
        d = validate_depth(depths[view], camera)
        depth_hash = hashlib.sha256(depths[view].tobytes()).hexdigest()
        sensor = sensors[view]
        layout = c["array_layout"][view]
        if (type(sensor.get("snapshot_id")) is not int or sensor["snapshot_id"] < 0 or
                type(sensor.get("render_barrier_updates")) is not int or sensor["render_barrier_updates"] < 4 or
                sensor.get("modalities") != ["rgb", "depth_linear"] or
                layout.get("rgb_shape") not in ([3, camera["height"], camera["width"]],
                                                [camera["height"], camera["width"], 3]) or
                layout.get("depth_shape") != [camera["height"], camera["width"]] or
                layout.get("depth_dtype") != str(depths[view].dtype) or
                layout.get("rgb_bytes") != images[view].nbytes or layout.get("depth_bytes") != depths[view].nbytes):
            raise ValueError("Original render barrier and calibrated array layout required")
        snapshots.add(sensor["snapshot_id"])
        if (depth_hash != _sha(c["depth_array_sha256"][view]) or
                sensor.get("depth_sha256") != depth_hash or
                sensor.get("rgb_sha256") != hashlib.sha256(images[view].tobytes()).hexdigest() or
                sensor.get("same_sensor_current_render") is not True or
                type(sensor.get("control_steps_in_capture")) is not int or sensor["control_steps_in_capture"] != 0 or
                sensor.get("depth_units") != "metres" or
                sensor.get("depth_convention") != "distance_to_image_plane" or
                sensor.get("shape") != [camera["height"], camera["width"]]):
            raise ValueError("RGB-D clock, unit or array identity mismatch")
        depths[view] = d
    if len(snapshots) != 1:
        raise ValueError("All onboard views must share one actual snapshot id")
    q, gripper = _array(c["q"], (18,)), _array(c["gripper"], (2,))
    proprio = json.loads(raw["proprio.json"])
    if set(proprio) != {"eef_base_m", "finger_opening_m", "base_velocity_local", "torso_joints_rad"}:
        raise ValueError("Only robot proprioception is permitted")
    state = model.state(q, gripper, _array(proprio["base_velocity_local"], (3,)))
    expected = {"eef_base_m": {a: state.poses[a][0].round(3).tolist() for a in ("left", "right")},
                "finger_opening_m": {a: round(float(gripper[i]), 3) for i, a in enumerate(("left", "right"))},
                "base_velocity_local": state.base_velocity.round(3).tolist(), "torso_joints_rad": q[:4].round(3).tolist()}
    if expected != proprio:
        raise ValueError("Capture proprioception disagrees with actual joint FK")
    geometry = json.loads(raw["robot_self_geometry.json"])
    make_frame({v + "_rgb": images[v] for v in VIEWS}, depths, model, state, sum(clock.values()), geometry)
    return CapturedState(CaptureRef(folder.resolve(), ref.sha256, clock), state, images, depths,
                         geometry, {v: c["files_sha256"][v + ".png"] for v in VIEWS}, snapshots.pop())


def _same_state(left, right):
    return (left.ref.clock == right.ref.clock and np.array_equal(left.state.q, right.state.q)
            and np.array_equal(left.state.gripper, right.state.gripper)
            and np.array_equal(left.state.base_velocity, right.state.base_velocity))


def prepare_window(model, goal, records, request_capture, deadline):
    """Bounded same-run CLOSE then consecutive UPs; no skipped or failed motion."""
    _deadline(deadline)
    if type(goal) is not Goal or goal.kind != "pick" or goal.hand not in ("left", "right"):
        raise ValueError("Only an explicit single-hand GRASP request is supported")
    if type(records) not in (list, tuple) or not 3 <= len(records) <= 5:
        raise ValueError("One actual CLOSE and two to four consecutive UPs required")
    cache = {}
    def load(ref):
        _deadline(deadline)
        if type(ref) is not CaptureRef:
            raise ValueError("Explicit public CaptureRef required")
        key = (str(Path(ref.directory).resolve()), ref.sha256, tuple(sorted(_clock(ref.clock).items())))
        if key not in cache:
            cache[key] = read_capture(ref, model)
        return cache[key]
    frames = []; executions = []; run = None; previous = None
    for i, record in enumerate(records):
        if type(record) is not ExecutedMotionRef:
            raise ValueError("Explicit executed-motion record required")
        expected_token = goal.hand.upper() + ("_CLOSE" if i == 0 else "_UP")
        path = Path(record.execution_path)
        before, after = load(record.before), load(record.after)
        scope = path.resolve().parent.parent
        if (record.token != expected_token or before.ref.directory.name != "before" or
                after.ref.directory.name != "after_settle" or
                before.ref.directory.parent != path.resolve().parent or
                after.ref.directory.parent != path.resolve().parent or (run is not None and scope != run)):
            raise ValueError("One same-run actual hand/action sequence required")
        run = scope
        execution = json.loads(_bytes(path, record.execution_sha256, 1024 * 1024))
        start, end = execution.get("control_start"), execution.get("control_end")
        feedback = execution.get("feedback", {})
        status = "GRIPPER_COMMAND_COMPLETED" if i == 0 else "TARGET_REACHED"
        if (execution.get("token") != expected_token or type(start) is not int or type(end) is not int or
                not 0 <= start < end or end - start > 150 or
                before.ref.clock["prefix_control"] != after.ref.clock["prefix_control"] or
                before.ref.clock["native_control"] != start or after.ref.clock["native_control"] != end + 12 or
                feedback.get("status") != status or execution.get("status", status) != status or
                type(feedback.get("control_ticks")) is not int or feedback["control_ticks"] != end - start or
                after.snapshot_id <= before.snapshot_id or
                (previous is not None and (not _same_state(previous, before) or before.snapshot_id < previous.snapshot_id))):
            raise ValueError("Recorded execution and actual settled sensor clocks disagree")
        # Preserve the executor's complete command contract, including explicit
        # visual failures on UP and all versioned safety checks on CLOSE.
        if not execution_completed(Action(goal.hand, "close" if i == 0 else "up", "fine"), feedback):
            raise ValueError("Strict public execution completion receipt failed")
        if i == 0:
            closing = feedback.get("gripper_execution", {})
            if (closing.get("part") != goal.hand or closing.get("move") != "close" or
                    closing.get("command_complete") is not True or closing.get("success_claim") is not False or
                    closing.get("executed_control_ticks") != end - start):
                raise ValueError("Actual CLOSE completion receipt required, not a grasp claim")
        previous = after
        frames.append(after); executions.append(execution)
    current = load(request_capture)
    if (current.ref.directory.parent.parent != run or not _same_state(current, frames[-1]) or
            current.snapshot_id < frames[-1].snapshot_id or
            (current.snapshot_id == frames[-1].snapshot_id and current.ref.sha256 != frames[-1].ref.sha256)):
        raise ValueError("Verification request is stale or from another execution/episode")
    # A new render at the same actual control clock is allowed, never counted as
    # another lift. Use the exact status-query pixels for the final observation.
    frames[-1] = current
    _deadline(deadline)
    return frames, executions


def verify_grasp_request(model, goal, records, request_capture, *, localize, deadline,
                         requested_status="REQUEST_VERIFY"):
    """At most five localization calls, zero controls; invalid input abstains.

    ``localize(CapturedState, Goal)`` must honor the caller's absolute deadline
    and return strict GroundedEvidence. It must not step the simulator or expose
    dataset/private fields. No model self-report of holding is consumed.
    """
    result = {"version": VERSION, "public_holding_verified": False, "status": "UNKNOWN",
              "reason": "NO_VALID_PUBLIC_WINDOW", "not_skill_success": True,
              "not_official_task_success": True, "physical_controls": 0,
              "localization_calls": 0, "frames": []}
    try:
        if requested_status != "REQUEST_VERIFY" or type(requested_status) is not str:
            raise ValueError("An explicit learned verification request is required")
        frames, executions = prepare_window(model, goal, records, request_capture, deadline)
        harness = GroundedHarness([goal]); harness.stage = "VERIFY_GRASP"
        harness.pending_grasp = {a: a == goal.hand for a in ("left", "right")}
        verifier = GraspMotionVerifier(persistent_tracks=True, spatial_seed_features=True)
        motion = RGBDMotion("rgbd_joint", exclude_robot=True, refine_matches=True)
        for i, (frame, execution) in enumerate(zip(frames, executions)):
            _deadline(deadline)
            rgb = {v + "_rgb": frame.images[v] for v in VIEWS}
            robot_frame = make_frame(rgb, frame.depths, model, frame.state, frame.control, frame.geometry)
            odometry = motion.observe(rgb, frame.depths, model, frame.state.q, robot_frame=robot_frame,
                                      gripper=frame.state.gripper, control=frame.control)
            result["localization_calls"] += 1
            evidence = localize(frame, goal)
            _deadline(deadline)
            if type(evidence) is not GroundedEvidence:
                raise ValueError("Strict GroundedEvidence required")
            evidence = GroundedEvidence.parse(json.dumps(asdict(evidence), allow_nan=False))
            if (any(getattr(evidence, k) is not None for k in ("enclosed", "co_moving", "supported", "effect")) or
                    evidence.other_views or evidence.target_reference != "unknown" or
                    evidence.hazard not in ("none", "occluded")):
                raise ValueError("Only surface localization, never a VLM holding/success claim")
            target = localize_target(evidence, frame.depths, model, frame.state.q)
            harness.executions = i
            harness.last_action = Action(goal.hand, "close" if i == 0 else "up")
            harness.feedback = execution["feedback"]; harness.last_gripper = frame.state.gripper.copy()
            measured = verifier.observe(model, frame.state, frame.images, frame.depths, frame.geometry,
                                        harness, {goal.hand: target}, odometry, evidence)
            result["frames"].append({"capture_sha256": frame.ref.sha256, "clock": frame.ref.clock,
                "token": records[i].token, "evidence": asdict(evidence), "target": target,
                "odometry": odometry, "grasp": measured})
            _deadline(deadline)
        # Only the request's current endpoint may verify, never an earlier
        # cherry-picked success. Invalid intervening pairs reset the verifier.
        verified = result["frames"][-1]["grasp"].get("verified") is True
        result.update(public_holding_verified=verified,
                      status="PUBLIC_HOLDING_VERIFIED" if verified else "UNKNOWN",
                      reason="REQUEST_CORROBORATED_NOT_SKILL_SUCCESS" if verified else "PUBLIC_REGISTRATION_INSUFFICIENT",
                      request_capture_sha256=frames[-1].ref.sha256,
                      request_clock=dict(frames[-1].ref.clock), goal=asdict(goal))
    except (ValueError, TypeError, KeyError, OSError, TimeoutError, RuntimeError) as exc:
        result.update(public_holding_verified=False, status="UNKNOWN", reason="INVALID_OR_UNAVAILABLE_PUBLIC_EVIDENCE", error=str(exc))
    return result
