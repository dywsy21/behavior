"""Fixed-parameter saved-frame KLT diagnosis. No simulator or actor writes.

This is an explicit alternative tracker, NOT a fallback selected after seeing
which solver accepts an action. The existing rigid RGB-D quality gate is reused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image

from semantic_robot.v2.grasp_motion import robot_point_mask
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.grounding import validate_depth
from semantic_robot.v2.odometry import solve_rgbd_correspondences, rigid_fit, body_motion


TRACKER = {"maxCorners": 1000, "qualityLevel": .01, "minDistance": 7,
           "blockSize": 7}
LK = {"winSize": (21, 21), "maxLevel": 3,
      "flags": 0,
      "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}


def read_frame(path, model):
    rgb = np.asarray(Image.open(path / "CURRENT_HEAD_RAW.png"))
    with np.load(path / "depth.npz") as archive:
        depth = archive["head"].copy()
    camera = model.spec["metadata"]["cameras"]["head"]
    if rgb.shape != (camera["height"], camera["width"], 3) or rgb.dtype != np.uint8:
        raise ValueError("Calibrated RGB shape/dtype required")
    depth = validate_depth(depth, camera)
    receipt = json.loads((path / "depth_receipt.json").read_text())["head"]
    for key, value in (("rgb", rgb), ("depth", depth)):
        if hashlib.sha256(value.tobytes()).hexdigest() != receipt[key + "_sha256"]:
            raise ValueError("Original RGB-D hash mismatch")
    q = np.asarray(json.loads((path / "proprio.json").read_text())["q"])
    T = model.forward(q, "camera_head"); K = np.asarray(camera["K"])
    if T.shape != (4, 4) or K.shape != (3, 3) or not np.isfinite(T).all() or not np.isfinite(K).all():
        raise ValueError("Finite calibrated camera required")
    return {"gray": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), "depth": depth,
            "geometry": json.loads((path / "robot_self_geometry.json").read_text()),
            "T": T, "K": K}


def stable_point(uv, frame):
    if not np.isfinite(uv).all():
        return None
    x, y = np.rint(uv).astype(int)
    height, width = frame["gray"].shape
    if frame["depth"].shape != (height, width):
        raise ValueError("Depth and RGB must have equal calibrated shape")
    if not 1 <= x < width - 1 or not 1 <= y < height - 1:
        return None
    patch = frame["depth"][y-1:y+2, x-1:x+2]
    z = float(np.median(patch))
    if patch.shape != (3, 3) or not .05 < z < 5. or not np.isfinite(patch).all() or np.ptp(patch) > .03:
        return None
    return np.linalg.solve(frame["K"], np.r_[uv, 1.]) * z


def spatial_corners(gray):
    """Pre-registered alternative: local contrast, same <=1000 corner budget."""
    height, width = gray.shape; groups = []
    for iy in range(4):
        for ix in range(4):
            mask = np.zeros_like(gray)
            mask[iy*height//4:(iy+1)*height//4, ix*width//4:(ix+1)*width//4] = 255
            found = cv2.goodFeaturesToTrack(gray, mask=mask, **{**TRACKER, "maxCorners": 62})
            groups.append([] if found is None else found.reshape(-1, 2).tolist())
    kept = []
    for i in range(62):
        for group in groups:
            if i >= len(group):
                continue
            point = group[i]
            if kept and np.min(np.linalg.norm(np.asarray(kept)-point, axis=1)) < 7:
                continue
            kept.append(point)
    return np.asarray(kept, dtype=np.float32).reshape(-1, 1, 2) if kept else None


def spatial_quality(result, rows, before, after):
    if not result["valid"]:
        return result
    motion = np.asarray(result["body_transform_current_in_previous"])
    S = np.diag([1., -1., -1., 1.])
    rel = S @ np.linalg.inv(after["T"]) @ np.linalg.inv(motion) @ before["T"] @ S
    a = np.asarray([r[0] for r in rows]); b = np.asarray([r[1] for r in rows])
    inliers = np.linalg.norm(a @ rel[:3, :3].T + rel[:3, 3] - b, axis=1) < .01
    if result.get("joint_reprojection_consensus"):
        fitted = a @ rel[:3, :3].T + rel[:3, 3]
        proj = fitted @ after["K"].T
        inliers &= np.linalg.norm(proj[:, :2]/proj[:, 2, None]-np.asarray([r[2] for r in rows]), axis=1) < 1.8
    result["spatial_support"] = []
    for frame, index in ((before, 3), (after, 2)):
        uv = np.asarray([r[index] for r in rows])[inliers]
        height, width = frame["gray"].shape
        cells = np.floor(uv / [width/4, height/4]).astype(int)
        counts = np.zeros((4, 4), dtype=int)
        for ix, iy in cells:
            if not 0 <= ix < 4 or not 0 <= iy < 4:
                raise ValueError("Support outside calibrated image")
            counts[iy, ix] += 1
        unique = len({tuple(x) for x in np.rint(uv).astype(int)}) == len(uv)
        spans = np.ptp(uv, axis=0) / [width, height]
        passed = unique and np.all(spans >= .25) and np.count_nonzero(counts >= 3) >= 6
        result["spatial_support"].append({"grid_counts": counts.tolist(), "unique_pixels": unique,
                                          "span_fraction": spans.tolist(), "passed": bool(passed)})
    if not all(r["passed"] for r in result["spatial_support"]):
        return {**result, "valid": False, "reason": "CONCENTRATED_IMAGE_SUPPORT"}
    return result


def joint_solve(rows, before, after):
    """Fixed joint RGB/depth consensus; retain all original final quality gates."""
    result = {"valid": False, "matches": len(rows), "joint_reprojection_consensus": True}
    if len(rows) < 25:
        return {**result, "reason": "INSUFFICIENT_STABLE_CORRESPONDENCES"}
    a = np.asarray([r[0] for r in rows]); b = np.asarray([r[1] for r in rows]); uv = np.asarray([r[2] for r in rows])
    K = after["K"]
    if not all(np.isfinite(x).all() for x in (a, b, uv, K, before["T"], after["T"])):
        raise ValueError("Finite correspondences and camera matrices required")
    def errors(R, t):
        fit = a @ R.T + t; point = np.linalg.norm(fit-b, axis=1)
        project = fit @ K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            pixel = np.linalg.norm(project[:, :2]/project[:, 2, None]-uv, axis=1)
        select = (point < .01) & (pixel < 1.8) & (fit[:, 2] > .01)
        return point, pixel, select
    rng = np.random.default_rng(31); best = None; score = (-1, -float("inf"))
    for _ in range(200):
        ids = rng.choice(len(a), 3, replace=False); fit = rigid_fit(a[ids], b[ids])
        if fit is None: continue
        point, pixel, select = errors(*fit)
        value = (int(select.sum()), -float(np.median(pixel[select])) if select.any() else -float("inf"))
        if value > score: best, score = select, value
    if best is None or best.sum() < 25:
        return {**result, "reason": "JOINT_RGBD_NOT_SUPPORTED"}
    for _ in range(2):
        fit = rigid_fit(a[best], b[best])
        if fit is None: return {**result, "reason": "DEGENERATE_3D_SUPPORT"}
        R, t = fit; point, pixel, best = errors(R, t)
        if best.sum() < 25: return {**result, "reason": "JOINT_RGBD_NOT_SUPPORTED"}
    rel = np.eye(4); rel[:3, :3] = R; rel[:3, 3] = t
    motion, delta = body_motion(before["T"], after["T"], rel)
    if not all(np.isfinite(x).all() for x in (motion, delta, point, pixel)):
        return {**result, "reason": "NONFINITE_MOTION_OR_RESIDUAL"}
    result.update(inliers=int(best.sum()), inlier_fraction=float(best.mean()),
                  median_reprojection_px=float(np.median(pixel[best])), median_depth_correspondence_m=float(np.median(point[best])),
                  body_delta=delta.tolist(), body_translation_z_m=float(motion[2, 3]),
                  body_transform_current_in_previous=motion.tolist())
    if best.mean() < .45: return {**result, "reason": "MATCHES_NOT_GEOMETRICALLY_COHERENT"}
    if result["median_reprojection_px"] > 1. or result["median_depth_correspondence_m"] > .015:
        return {**result, "reason": "RGB_DEPTH_MOTION_DISAGREEMENT"}
    if np.linalg.norm(delta[:2]) > .18 or abs(delta[2]) > .30 or abs(motion[2, 3]) > .035:
        return {**result, "reason": "OUTSIDE_SINGLE_MICRO_ACTION_MOTION_BOUND"}
    return {**result, "valid": True, "reason": "JOINT_RGBD_DIAGNOSTIC_ONLY"}


def track(before, after, spatial=False, joint=False):
    corners = (spatial_corners(before["gray"]) if spatial else
               cv2.goodFeaturesToTrack(before["gray"], mask=None, **TRACKER))
    counts = {"corners": 0 if corners is None else len(corners)}
    if corners is None:
        return {"valid": False, "reason": "NO_CORNERS", **counts}
    current, good, error = cv2.calcOpticalFlowPyrLK(before["gray"], after["gray"], corners, None, **LK)
    if current is None or good is None or error is None:
        return {"valid": False, "reason": "NO_FORWARD_TRACKS", **counts}
    valid = ((good.ravel() == 1) & np.isfinite(current.reshape(-1, 2)).all(axis=1)
             & np.isfinite(error.ravel()) & (error.ravel() <= 20.))
    old = corners[valid]; current = current[valid]
    counts["forward_tracks"] = len(current)
    if not len(current):
        return {"valid": False, "reason": "NO_FORWARD_TRACKS", **counts}
    back, good_back, back_error = cv2.calcOpticalFlowPyrLK(after["gray"], before["gray"], current, None, **LK)
    if back is None or good_back is None or back_error is None:
        return {"valid": False, "reason": "NO_BACKWARD_TRACKS", **counts}
    old = old.reshape(-1, 2); new = current.reshape(-1, 2)
    roundtrip = np.linalg.norm(back.reshape(-1, 2) - old, axis=1)
    valid = ((good_back.ravel() == 1) & np.isfinite(back.reshape(-1, 2)).all(axis=1)
             & np.isfinite(back_error.ravel()) & (back_error.ravel() <= 20.) & (roundtrip <= .5))
    counts["bidirectional_tracks"] = int(valid.sum())
    rows = []
    for uv0, uv1 in zip(old[valid], new[valid]):
        a, b = stable_point(uv0, before), stable_point(uv1, after)
        if a is not None and b is not None:
            rows.append((a, b, uv1, uv0))
    counts["stable_depth_tracks"] = len(rows)
    if rows:
        a = np.asarray([r[0] for r in rows]); b = np.asarray([r[1] for r in rows])
        keep = np.ones(len(rows), dtype=bool)
        for points, frame in ((a, before), (b, after)):
            T = frame["T"]
            base = (points * [1., -1., -1.]) @ T[:3, :3].T + T[:3, 3]
            own = robot_point_mask(base, frame["geometry"])
            if own is None:
                raise ValueError("Actual robot self geometry required")
            keep &= ~own
        rows = [r for r, retained in zip(rows, keep) if retained]
    counts["nonrobot_tracks"] = len(rows)
    result = (joint_solve(rows, before, after) if joint else
              solve_rgbd_correspondences([r[0] for r in rows], [r[2] for r in rows],
                                         [r[1] for r in rows], after["K"], before["T"], after["T"]))
    result = spatial_quality(result, rows, before, after)
    return {**result, "track_counts": counts,
            "support_before_optical_m": [r[0].tolist() for r in rows],
            "support_after_optical_m": [r[1].tolist() for r in rows],
            "support_after_uv": [r[2].tolist() for r in rows],
            "support_before_uv": [r[3].tolist() for r in rows]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spatial-corners", action="store_true")
    p.add_argument("--joint-consensus", action="store_true")
    a = p.parse_args()
    if a.output.exists() or len(a.run) > 3:
        raise ValueError("Preserve evidence; at most three registered runs")
    started = time.monotonic(); cv2.setNumThreads(2); rows = []
    for run in a.run:
        model = RobotModel(json.loads((run / "robot_calibration.json").read_text()))
        paths = sorted(path for path in run.glob("decision_*") if re.fullmatch(r"decision_\d{3}", path.name))
        if not 2 <= len(paths) <= 25:
            raise ValueError("Registered <=24 motion pairs per run")
        before = read_frame(paths[0], model)
        for path in paths[1:]:
            if time.monotonic() - started > 300:
                raise TimeoutError("CPU process bound")
            after = read_frame(path, model)
            rows.append({"run": run.name, "decision": int(path.name.split("_")[1]),
                         "motion": track(before, after, a.spatial_corners, a.joint_consensus)})
            before = after
    result = {"diagnostic_only": True, "model_calls": 0, "controls": 0,
              "tracker": TRACKER, "spatial_corners": a.spatial_corners, "joint_consensus": a.joint_consensus, "lk": LK, "forward_backward_max_px": .5,
              "forward_intensity_error_max": 20., "rows": rows,
              "wall_s": time.monotonic() - started,
              "limitation": "Static scene assumption remains; robot geometry is exclusion-only. No deployment or retroactive gate pass."}
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"wall_s": result["wall_s"], "pairs": len(rows),
                      "selected": [{"run": r["run"], "decision": r["decision"],
                                    "motion": {k: v for k, v in r["motion"].items() if not k.startswith("support_")}}
                                   for r in rows if not r["motion"]["valid"] or (r["run"] == "radio_h11_fullstart" and r["decision"] >= 13)]}, indent=2))


if __name__ == "__main__":
    main()
