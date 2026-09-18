"""Pre-registered fixed-support joint objective experiment; diagnostic only."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from semantic_robot.v2.odometry import body_motion
from semantic_robot.v2.kinematics import RobotModel


def refine(row):
    data = row["rigid_residuals"]
    if data is None or "optical_transform" not in data:
        return {"valid": False, "reason": "NO_3D_INITIALIZATION"}
    a = np.asarray(row["points_before"]); b = np.asarray(row["points_after"])
    u0 = np.asarray(row["pixels_before"]); u1 = np.asarray(row["pixels_after"])
    support = np.asarray(data["support_mask"], dtype=bool)
    K = np.asarray(row["K"]); initial = np.asarray(data["optical_transform"])
    if support.sum() < 25 or support.mean() < .45:
        return {"valid": False, "reason": "INSUFFICIENT_ORIGINAL_3D_SUPPORT"}
    if not all(np.isfinite(x).all() for x in (a, b, u0, u1, K, initial)):
        raise ValueError("Finite original measurements required")
    points = a[support], b[support], u0[support], u1[support]
    x0 = np.r_[cv2.Rodrigues(initial[:3, :3])[0].ravel(), initial[:3, 3]]
    def errors(x):
        R = cv2.Rodrigues(x[:3])[0]; t = x[3:]
        aa, bb, uu0, uu1 = points
        forward = aa @ R.T + t; backward = (bb-t) @ R
        if np.any(forward[:, 2] <= .01) or np.any(backward[:, 2] <= .01):
            raise ValueError("Refinement left the positive-depth domain")
        pf = forward @ K.T; pb = backward @ K.T
        pixel = pf[:, :2]/pf[:, 2, None]-uu1
        reverse = pb[:, :2]/pb[:, 2, None]-uu0
        point = forward-bb
        return pixel, reverse, point
    def objective(x):
        pixel, reverse, point = errors(x)
        return np.concatenate([pixel.ravel(), reverse.ravel(), (point/.01).ravel()])
    solution = least_squares(objective, x0, loss="soft_l1", f_scale=1., max_nfev=50)
    if not solution.success or not np.isfinite(solution.x).all():
        return {"valid": False, "reason": "JOINT_OPTIMIZATION_NOT_CONVERGED"}
    pixel, reverse, point = errors(solution.x)
    rel = np.eye(4); rel[:3, :3] = cv2.Rodrigues(solution.x[:3])[0]; rel[:3, 3] = solution.x[3:]
    motion, delta = body_motion(np.asarray(row["camera_before"]), np.asarray(row["camera_after"]), rel)
    unique_old = len(np.unique(np.rint(u0[support]), axis=0)); unique_new = len(np.unique(np.rint(u1[support]), axis=0))
    result = {"valid": False, "reason": "JOINT_FIXED_SUPPORT_DIAGNOSTIC", "support": int(support.sum()),
              "unique_old_support": unique_old, "unique_new_support": unique_new,
              "median_pixel": float(np.median(np.linalg.norm(pixel, axis=1))),
              "median_reverse_pixel": float(np.median(np.linalg.norm(reverse, axis=1))),
              "median_3d_m": float(np.median(np.linalg.norm(point, axis=1))),
              "body_delta": delta.tolist(), "body_z_m": float(motion[2, 3]),
              "body_transform": motion.tolist(), "optical_transform": rel.tolist(),
              "nfev": solution.nfev, "cost_before": float(np.sum(2*(np.sqrt(1+objective(x0)**2)-1))),
              "cost_after": float(2*solution.cost), "fixed_support_no_reselection": True}
    if result["median_pixel"] > 1. or result["median_reverse_pixel"] > 1. or result["median_3d_m"] > .015:
        return {**result, "reason": "RGB_DEPTH_MOTION_DISAGREEMENT"}
    if np.linalg.norm(delta[:2]) > .18 or abs(delta[2]) > .30 or abs(motion[2, 3]) > .035:
        return {**result, "reason": "OUTSIDE_ORIGINAL_MICRO_MOTION_BOUND"}
    return {**result, "valid": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--b11-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-production", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous experiment outputs")
    start = time.monotonic(); cv2.setNumThreads(2)
    original = json.loads(args.input.read_text())
    if len(original["rows"]) != 223:
        raise ValueError("All registered H12 segments required")
    spec = importlib.util.spec_from_file_location("h13_audit", Path(__file__).with_name("audit_substep_residuals.py"))
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    model = RobotModel(json.loads((args.b11_run/"robot_calibration.json").read_text()))
    rows = original["rows"]
    # Every old B11 action boundary, not just its known bad terminal pair.
    for index in range(1, 20):
        before = audit.frame(args.b11_run/f"decision_{index-1:03d}", model)
        after = audit.frame(args.b11_run/f"decision_{index:03d}", model)
        rows.append({"run": args.b11_run.name, "decision": index, "controls": None,
                     **audit.compare(before, after)})
    output = []
    for row in rows:
        if time.monotonic()-start > 300:
            raise TimeoutError("Finite CPU experiment exceeded")
        result = refine(row)
        production = None
        if args.include_production:
            from semantic_robot.v2.joint_odometry import solve_joint_correspondences
            production = solve_joint_correspondences(row["points_before"], row["pixels_after"], row["points_after"],
                                                      row["K"], row["camera_before"], row["camera_after"])
        output.append({"run": row["run"], "decision": row["decision"], "controls": row["controls"],
                       "rigid": row["rigid"], "pnp": row["pnp"], "joint": result, "production": production})
    report = {"diagnostic_only": True, "wall_s": time.monotonic()-start, "rows": output,
              "new_model_calls": 0, "new_controls": 0,
              "recipe": {"pixel_scale": 1., "depth_scale_m": .01, "loss": "soft_l1", "f_scale": 1., "max_nfev": 50},
              "scope": "223 complete H12 substeps and 19 earlier B11 boundary pairs; no deployment or true-pose certification"}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"wall_s": report["wall_s"], "pairs": len(output),
                      "joint_pass": sum(x["joint"]["valid"] for x in output),
                      "production_pass": sum(bool(x["production"] and x["production"]["valid"]) for x in output),
                      "selected": [x for x in output if not x["joint"]["valid"] or (x["production"] and not x["production"]["valid"]) or (x["run"] == "radio_b11" and x["decision"] == 19)
                                   or (x["run"] == "radio_h12_fullstart" and x["controls"] == [420, 426])]}, indent=2))


if __name__ == "__main__":
    main()
