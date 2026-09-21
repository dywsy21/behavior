"""Current robot-only geometry for excluding non-world RGB-D landmarks.

Never infer geometry from scene segmentation or fit residuals. The runtime
captures actual finger FK, not the calibration's fully-open finger transforms.
"""
import copy
import hashlib
import json

import numpy as np

from .grasp_motion import robot_point_mask
from .grounding import validate_depth


VERSION = "current_robot_self_odometry_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sensor_hashes(images, depths, model):
    rgb = np.asarray(images["head_rgb"])
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    camera = model.spec["metadata"]["cameras"]["head"]
    if rgb.shape != (camera["height"], camera["width"], 3) or rgb.dtype != np.uint8:
        raise ValueError("Current calibrated raw head RGB required")
    depth = validate_depth(depths["head"], camera)
    return {"rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest()}


def checked_geometry(geometry, model):
    reference = model.spec["metadata"].get("robot_visual_boxes_reference")
    for value in (geometry, reference):
        if (not isinstance(value, dict) or
                value.get("source") != "robot_visual_link_boxes_actual_joint_fk" or
                value.get("scene_truth") is not False or
                value.get("includes_actual_finger_positions") is not True or
                not isinstance(value.get("boxes"), list) or not value["boxes"]):
            raise ValueError("Complete actual robot-only visual link geometry required")
    # Same immutable asset skin and margin, with only joint transforms changing.
    expected = {box["link"]: box for box in reference["boxes"]}
    actual = {box["link"]: box for box in geometry["boxes"]}
    if (len(expected) != len(reference["boxes"]) or len(actual) != len(geometry["boxes"]) or
            set(actual) != set(expected) or geometry.get("margin_m") != reference.get("margin_m")):
        raise ValueError("Robot self mask must cover the same calibrated asset links and margin")
    for name, box in actual.items():
        if any(not np.array_equal(box.get(key), expected[name].get(key)) for key in ("lower", "upper")):
            raise ValueError("Robot link skin changed; no scene-sized exclusion boxes")
    if robot_point_mask(np.empty((0, 3)), geometry) is None:
        raise ValueError("Invalid robot self geometry")
    return geometry


def make_frame(images, depths, model, state, control, geometry):
    frame = {"version": VERSION, "control": control, "q": state.q.tolist(),
             "gripper": state.gripper.tolist(), "sensors": sensor_hashes(images, depths, model),
             "geometry": copy.deepcopy(geometry)}
    validate_frame(frame, images, depths, model, state.q, state.gripper, control)
    return frame


def validate_frame(frame, images, depths, model, q, gripper, control):
    if (not isinstance(frame, dict) or frame.get("version") != VERSION or
            type(control) is not int or control < 0 or
            type(frame.get("control")) is not int or frame["control"] != control):
        raise ValueError("Current robot self frame and exact integer control clock required")
    for key, actual, shape in (("q", q, (18,)), ("gripper", gripper, (2,))):
        actual = np.asarray(actual, dtype=float)
        recorded = np.asarray(frame.get(key), dtype=float)
        if (actual.shape != shape or recorded.shape != shape or
                not np.isfinite(actual).all() or not np.array_equal(actual, recorded)):
            raise ValueError("Robot self frame has stale or unknown " + key)
    if frame.get("sensors") != sensor_hashes(images, depths, model):
        raise ValueError("Robot self frame does not belong to these RGB-D bytes")
    checked_geometry(frame.get("geometry"), model)
    return digest(frame)


def world_correspondences(oldpoints, pixels, newpoints, before, after, geometry_before, geometry_after):
    """Remove a match if EITHER measured endpoint lies on the robot skin."""
    oldpoints = np.asarray(oldpoints, dtype=float).reshape(-1, 3)
    newpoints = np.asarray(newpoints, dtype=float).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(oldpoints) != len(newpoints) or len(oldpoints) != len(pixels):
        raise ValueError("Aligned RGB-D correspondences required")
    masks = []
    for points, camera, geometry in ((oldpoints, before, geometry_before), (newpoints, after, geometry_after)):
        camera = np.asarray(camera)
        base = (points * [1, -1, -1]) @ camera[:3, :3].T + camera[:3, 3]
        mask = robot_point_mask(base, geometry)
        if mask is None:
            raise ValueError("No unmasked fallback when robot geometry is unavailable")
        masks.append(mask)
    excluded = masks[0] | masks[1]
    return (oldpoints[~excluded], pixels[~excluded], newpoints[~excluded],
            {"raw_depth_matches": len(oldpoints), "self_excluded_before": int(masks[0].sum()),
             "self_excluded_after": int(masks[1].sum()), "self_excluded_either": int(excluded.sum()),
             "remaining_world_matches": int((~excluded).sum()), "fit_residuals_not_used_to_choose_mask": True})
