"""Saved RGB-D/body-box swept-volume diagnostic; never an actor/GT oracle.

Boxes overapproximate robot visual geometry. A new depth-box incursion is only
a candidate veto, not proof of contact, free space, or a collision-free path.
All links are rigidly transported for base commands; steering articulation and
continuous sweep extrema are not certified. The original depth gate is replayed
unchanged for comparison. No scene-object state is loaded.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.arm_observation_guard import box_distance
from semantic_robot.v2.grasp_motion import robot_point_mask
from semantic_robot.v2.grounding import LocalDepthGuard, observed_cloud
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.protocol import Action, TRANSLATIONS, VIEWS


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sweep(points, boxes, action, margin=.008):
    amount = action.amount(False)
    delta = np.asarray(TRANSLATIONS.get(action.move, (0., 0., 0.))) * amount
    yaw = amount * (1 if action.move == "yaw_plus" else -1) if action.move.startswith("yaw_") else 0.
    hits = []
    for box in boxes:
        start = np.asarray(box["T_base_link"], dtype=float)
        lo, hi = np.asarray(box["lower"]), np.asarray(box["upper"])
        old = box_distance(points, start, lo, hi)
        for u in np.linspace(0., 1., 13)[1:]:
            c, s = math.cos(yaw * u), math.sin(yaw * u)
            body = np.eye(4)
            body[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            body[:3, 3] = delta * u
            distance = box_distance(points, body @ start, lo, hi)
            indices = np.flatnonzero((distance < margin) & (distance < old - .002))
            if len(indices):
                worst = int(indices[np.argmin(distance[indices])])
                hits.append({"link": box["link"], "fraction": float(u),
                             "points": len(indices), "min_distance_m": float(distance[worst]),
                             "start_distance_same_point_m": float(old[worst]),
                             "worst_point_base_m": points[worst].tolist()})
                break
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--approach", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--gate", action="append", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise ValueError("Do not overwrite a diagnostic")
    if len(a.gate) != 2:
        raise ValueError("Exactly both registered gates required")
    started = time.monotonic()
    calibration = Path(a.calibration)
    model = RobotModel(read(calibration))
    models = {sha(calibration): model}
    calibrations = {sha(calibration): str(calibration)}
    sources = []
    completed = [json.loads(line) for line in (Path(a.approach) / "steps.jsonl").read_text().splitlines()]
    completed = [row for row in completed if "action" in row and "feedback" in row]
    for i in range(38, 55):
        d = Path(a.approach) / f"decision_{i:03d}"
        matching = [row for row in completed if row["decision"] == i]
        if len(matching) != 1:
            raise ValueError("Exactly one actual completed action for each registered state")
        selected = Action(**matching[0]["action"])
        sources.append((d, selected, "H13_original_start", sha(calibration)))
    for gate in a.gate:
        root = Path(gate)
        result = read(root / "result.json")
        if not result.get("gate_ok"):
            raise ValueError("Expected completed engineering gate")
        cal = root / "robot_calibration.json"
        digest = sha(cal)
        if digest not in models:
            models[digest] = RobotModel(read(cal))
            calibrations[digest] = str(cal)
        for row in result["decisions"]:
            action = Action(**row["action"])
            if action.part == "base":
                sources.append((root / f"decision_{row['decision']:03d}", action, root.name, digest))
    rows = []
    for d, action, group, digest in sources:
        if time.monotonic() - started > 900:
            raise TimeoutError("Registered CPU wall budget")
        model = models[digest]
        files = ["proprio.json", "robot_self_geometry.json", "depth.npz", "depth_receipt.json", "action_motion.json"]
        if (d / "candidates.json").exists():
            files.append("candidates.json")
        row = {"state": str(d), "group": group, "action": asdict(action),
               "calibration_sha256": digest, "source_sha256": {n: sha(d / n) for n in files}}
        if action.part != "base":
            row["skipped_reason"] = "NOT_A_BASE_ACTION"
            rows.append(row)
            continue
        q = np.asarray(read(d / "proprio.json")["q"])
        geometry = read(d / "robot_self_geometry.json")
        with np.load(d / "depth.npz", allow_pickle=False) as raw:
            depths = {v: raw[v].copy() for v in VIEWS}
        receipt = read(d / "depth_receipt.json")
        for v in VIEWS:
            rgb = np.asarray(Image.open(d / ("CURRENT_" + v.upper() + "_RAW.png")))
            for kind, value in (("rgb", rgb), ("depth", depths[v])):
                if hashlib.sha256(value.tobytes()).hexdigest() != receipt[v][kind + "_sha256"]:
                    raise ValueError("Saved RGB-D receipt mismatch")
        points = observed_cloud(depths, model, q)
        old_guard = LocalDepthGuard(points, model, q, depths)
        original_ok, original_reason = old_guard.check(action, False)
        candidate = (next(x for x in read(d / "candidates.json")["tested"] if x["action"] == asdict(action))
                     if (d / "candidates.json").exists() else None)
        own = robot_point_mask(points, geometry)
        if own is None:
            raise ValueError("Current actual robot geometry required")
        nonself = points[~own]
        # Separately retain old-guard masking vs complete current-box masking.
        # Neither is a scene-object semantic mask or verified free space.
        row.update(original_depth_ok=original_ok, original_depth_reason=original_reason,
                   original_candidate_accepted=candidate["accepted"] if candidate else None,
                   raw_cloud_points=len(points), outside_current_robot_boxes=len(nonself),
                   proposed_margin_m=.008, sampled_fractions=12,
                   old_guard_environment_sweep=sweep(old_guard.environment_points, geometry["boxes"], action),
                   outside_all_current_boxes_sweep=sweep(nonself, geometry["boxes"], action))
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("state", "original_depth_ok", "outside_all_current_boxes_sweep")}), flush=True)
    payload = {"rows": rows, "wall_s": time.monotonic() - started,
               "calibrations_by_sha256": calibrations, "script_sha256": sha(Path(__file__)),
               "approach_steps_sha256": sha(Path(a.approach) / "steps.jsonl"),
               "gate_result_sha256": {str(Path(g) / "result.json"): sha(Path(g) / "result.json") for g in a.gate},
               "source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
               "source_dirty": bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()),
               "new_controls": 0, "new_model_calls": 0, "not_a_contact_or_clearance_certificate": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
