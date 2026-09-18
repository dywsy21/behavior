"""Offline H13 diagnosis on recorded frames, never a deployed solver fallback.

Reproduce the head SIFT/3D correspondences and compare fixed 3D-fit and PnP
objectives on exactly the same data. Neither comparison retroactively accepts
an action. No simulator, model calls, hidden pose, or actor mutations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from semantic_robot.v2.grounding import validate_depth
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.odometry import solve_rgbd_correspondences, solve_correspondences


def frame(path, model):
    rgb = np.asarray(Image.open(path / "CURRENT_HEAD_RAW.png"))
    with np.load(path / "depth.npz") as archive:
        depth = archive["head"].copy()
    camera = model.spec["metadata"]["cameras"]["head"]
    if rgb.dtype != np.uint8 or rgb.shape != (camera["height"], camera["width"], 3):
        raise ValueError("Original calibrated RGB required")
    depth = validate_depth(depth, camera)
    receipt = json.loads((path / "depth_receipt.json").read_text())["head"]
    for name, value in (("rgb", rgb), ("depth", depth)):
        if hashlib.sha256(value.tobytes()).hexdigest() != receipt[name + "_sha256"]:
            raise ValueError("Recorded RGB-D hash mismatch")
    q = np.asarray(json.loads((path / "proprio.json").read_text())["q"])
    T = model.forward(q, "camera_head")
    K = np.asarray(camera["K"], dtype=float)
    if not np.isfinite(T).all() or not np.isfinite(K).all():
        raise ValueError("Finite recorded calibration required")
    keys, desc = cv2.SIFT_create(nfeatures=2000).detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    return {"rgb": rgb, "depth": depth, "K": K, "T": T, "keys": keys, "desc": desc,
            "rgb_sha256": receipt["rgb_sha256"], "depth_sha256": receipt["depth_sha256"]}


def correspondences(before, after):
    pairs = (cv2.BFMatcher().knnMatch(before["desc"], after["desc"], k=2)
             if before["desc"] is not None and after["desc"] is not None else [])
    rows = []
    for pair in pairs:
        if len(pair) != 2 or pair[0].distance >= .7 * pair[1].distance:
            continue
        m = pair[0]
        pixels = [np.asarray(before["keys"][m.queryIdx].pt), np.asarray(after["keys"][m.trainIdx].pt)]
        points = []
        for uv, item in zip(pixels, (before, after)):
            x, y = np.rint(uv).astype(int)
            patch = item["depth"][max(0, y-1):y+2, max(0, x-1):x+2]
            z = float(np.median(patch))
            if not .05 < z < 5. or not np.isfinite(patch).all() or np.ptp(patch) > .03:
                break
            points.append(np.linalg.solve(item["K"], np.r_[uv, 1.]) * z)
        if len(points) == 2:
            rows.append([*points, *pixels])
    return [np.asarray([row[i] for row in rows], dtype=float).reshape(-1, 3 if i < 2 else 2) for i in range(4)]


def residuals(result, points, before, after, frozen_support=None):
    if "body_transform_current_in_previous" not in result:
        return None
    a, b, u0, u1 = points
    S = np.diag([1., -1., -1., 1.])
    motion = np.asarray(result["body_transform_current_in_previous"])
    rel = S @ np.linalg.inv(after["T"]) @ np.linalg.inv(motion) @ before["T"] @ S
    forward = a @ rel[:3, :3].T + rel[:3, 3]
    backward = (b - rel[:3, 3]) @ rel[:3, :3]
    def project(xyz, K):
        p = xyz @ K.T
        return p[:, :2] / p[:, 2, None]
    pixel = np.linalg.norm(project(forward, after["K"]) - u1, axis=1)
    reverse = np.linalg.norm(project(backward, before["K"]) - u0, axis=1)
    distance = np.linalg.norm(forward-b, axis=1)
    select = distance < .01 if frozen_support is None else np.asarray(frozen_support, dtype=bool)
    if select.shape != distance.shape:
        raise ValueError("Aligned fixed residual support required")
    if not select.any():
        return {"support": 0}
    return {"support": int(select.sum()), "fraction": float(select.mean()),
            "pixel_median_on_3d_support": float(np.median(pixel[select])),
            "reverse_pixel_median_on_3d_support": float(np.median(reverse[select])),
            "depth_median_on_3d_support": float(np.median(distance[select])),
            "optical_transform": rel.tolist(),
            "pixel_errors": pixel.tolist(), "reverse_pixel_errors": reverse.tolist(),
            "point_errors": distance.tolist(), "support_mask": select.tolist()}


def compare(before, after):
    points = correspondences(before, after)
    a, b, u0, u1 = points
    args = a, u1, b, after["K"], before["T"], after["T"]
    rigid = solve_rgbd_correspondences(*args)
    pnp = solve_correspondences(*args)
    rigid_errors = residuals(rigid, points, before, after)
    frozen = None if rigid_errors is None else rigid_errors.get("support_mask")
    return {"matches": len(a), "unique_old_pixels": len(np.unique(np.rint(u0), axis=0)),
            "unique_new_pixels": len(np.unique(np.rint(u1), axis=0)),
            "rigid": rigid, "pnp": pnp,
            "rigid_residuals": rigid_errors,
            "pnp_residuals": residuals(pnp, points, before, after),
            "pnp_on_frozen_rigid_support": None if frozen is None else residuals(pnp, points, before, after, frozen),
            "points_before": a.tolist(), "points_after": b.tolist(),
            "pixels_before": u0.tolist(), "pixels_after": u1.tolist(),
            "K": after["K"].tolist(), "camera_before": before["T"].tolist(), "camera_after": after["T"].tolist()}


def annotated_failure(row, run, output):
    """Measured-point overlay for human audit, not target or material labels."""
    end = row["controls"][1]
    image_path = run / f"decision_{row['decision']:03d}" / "motion_substeps" / f"control_{end:06d}" / "CURRENT_HEAD_RAW.png"
    rgb = np.asarray(Image.open(image_path)).copy()
    values = row["rigid_residuals"]
    for i, (uv, valid) in enumerate(zip(row["pixels_after"], values["support_mask"])):
        color = (20, 230, 60) if valid else (255, 40, 40)
        center = tuple(np.rint(uv).astype(int))
        cv2.circle(rgb, center, 4, color, 1)
        cv2.putText(rgb, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
    Image.fromarray(rgb).save(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    expected_counts = {"radio_h12_fullstart": 71, "gate_radio_h12": 76, "gate_plates_h12": 76}
    if (args.output.exists() or len({p.resolve() for p in args.run}) != 3
            or {p.name for p in args.run} != set(expected_counts)):
        raise ValueError("Exactly three registered H12 runs, new output only")
    cv2.setNumThreads(2)
    start = time.monotonic()
    rows = []; identities = []; seen = set()
    for run in args.run:
        model = RobotModel(json.loads((run / "robot_calibration.json").read_text()))
        terminal = json.loads((run / "result.json").read_text())
        identities.append({"run": run.name,
                           "result_sha256": hashlib.sha256((run/"result.json").read_bytes()).hexdigest(),
                           "calibration_sha256": hashlib.sha256((run/"robot_calibration.json").read_bytes()).hexdigest(),
                           "implementation_digest": terminal["implementation_digest"]})
        for action in terminal["decisions"]:
            root = run / f"decision_{action['decision']:03d}"
            if not (root / "action_motion.json").exists():
                if action["control_end"] > action["control_start"]:
                    raise ValueError("Executed action missing required substep chain")
                continue
            chain = json.loads((root / "action_motion.json").read_text())
            before = frame(root, model)
            for segment in chain["segments"]:
                key = (run.name, action["decision"], segment["control_start"], segment["control_end"])
                if key in seen:
                    raise ValueError("Duplicate recorded segment")
                seen.add(key)
                if len(rows) >= 300 or time.monotonic() - start > 600:
                    raise TimeoutError("Registered CPU scope exceeded")
                location = root / "motion_substeps" / f"control_{segment['control_end']:06d}"
                after = frame(location, model)
                for item, label in ((before, "before"), (after, "after")):
                    for key in ("rgb_sha256", "depth_sha256"):
                        if item[key] != segment[label][key]:
                            raise ValueError("Segment endpoint identity mismatch")
                    if not np.allclose(item["T"], np.asarray(segment[label]["camera_fk"]), atol=1e-10, rtol=1e-10):
                        raise ValueError("Recorded segment camera FK mismatch")
                result = compare(before, after)
                original = segment["measurement"]
                for key in ("valid", "reason", "matches", "inliers", "inlier_fraction", "median_reprojection_px", "median_depth_correspondence_m",
                            "body_delta", "body_transform_current_in_previous", "body_translation_z_m"):
                    if key in original and result["rigid"].get(key) != original[key]:
                        # SVD can differ in last bits across BLAS builds; never hide real drift.
                        if not isinstance(original[key], (float, list)) or not np.allclose(result["rigid"][key], original[key], atol=1e-8, rtol=1e-7):
                            raise ValueError(f"Original solver reproduction mismatch: {key}")
                rows.append({"run": run.name, "decision": action["decision"],
                             "controls": [segment["control_start"], segment["control_end"]],
                             "recorded_valid": original["valid"], **result})
                before = after
    if {name: sum(r["run"] == name for r in rows) for name in expected_counts} != expected_counts:
        raise ValueError("Registered 71+76+76 segment coverage mismatch")
    report = {"diagnostic_only": True, "new_model_calls": 0, "new_controls": 0,
              "wall_s": time.monotonic()-start, "rows": rows,
              "identities": identities, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
              "limitation": "Same recorded static-world assumption, no GT or deployment acceptance; comparison is not a fallback."}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    for index, row in enumerate(rows):
        if not row["recorded_valid"] and row["rigid_residuals"] is not None:
            run = next(path for path in args.run if path.name == row["run"])
            annotated_failure(row, run, args.output.with_name(args.output.stem + f"_failed_{index}.png"))
    print(json.dumps({"wall_s": report["wall_s"], "pairs": len(rows),
                      "rigid_pass": sum(r["rigid"]["valid"] for r in rows),
                      "pnp_pass": sum(r["pnp"]["valid"] for r in rows),
                      "differences": [{k: r[k] for k in ("run", "decision", "controls", "matches", "unique_old_pixels", "unique_new_pixels", "rigid", "pnp")}
                                      for r in rows if not r["rigid"]["valid"] or not r["pnp"]["valid"]]}, indent=2))


if __name__ == "__main__":
    main()
