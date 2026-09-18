"""CPU-only saved-depth ablation for missed low chassis contacts.

No contact/object state is read. These geometric vetoes are not clearance
certificates. History is transformed only by the saved onboard RGB-D odometry.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from audit_body_sweep import sweep
from semantic_robot.v2.grounding import observed_cloud, LocalDepthGuard
from semantic_robot.v2.grasp_motion import robot_point_mask
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.protocol import Action


def circle_veto(points, action, radius, zmin):
    points = points[(points[:, 2] > zmin) & (points[:, 2] < .75)]
    if not len(points):
        return None
    before = np.linalg.norm(points[:, :2], axis=1) - radius
    from semantic_robot.v2.protocol import TRANSLATIONS
    delta = np.asarray(TRANSLATIONS.get(action.move, (0, 0, 0))) * action.amount(False)
    for u in np.linspace(0, 1, 6)[1:]:
        after = np.linalg.norm((points - u * delta)[:, :2], axis=1) - radius
        hit = (after < .015) & (after < before - .002)
        if hit.any():
            i = np.flatnonzero(hit)[np.argmin(after[hit])]
            return {"point": points[i].tolist(), "clearance": float(after[i]), "fraction": float(u)}
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root, out = Path(args.run), Path(args.output)
    if out.exists():
        raise ValueError("Preserve prior evidence")
    model = RobotModel(json.loads((root / "robot_calibration.json").read_text()))
    result = json.loads((root / "result.json").read_text())
    memory, rows = [], []
    started = time.monotonic()
    last_base = max(row["decision"] for row in result["decisions"] if row["action"]["part"] == "base")
    for command in result["decisions"]:
        if command["decision"] > last_base:
            break
        if time.monotonic() - started > 600:
            raise TimeoutError("Bounded CPU ablation")
        d = root / f"decision_{command['decision']:03d}"
        if command["decision"]:
            prior = root / f"decision_{command['decision'] - 1:03d}" / "action_motion.json"
            previous_command = result["decisions"][command["decision"] - 1]
            if previous_command["control_start"] == previous_command["control_end"]:
                transform = np.eye(4)
            else:
                motion = json.loads(prior.read_text())
                if not motion["valid"]:
                    raise ValueError("No history across invalid motion")
                transform = np.asarray(motion["body_transform_current_in_previous"])
            memory = [(x - transform[:3, 3]) @ transform[:3, :3] for x in memory]
        q = np.asarray(json.loads((d / "proprio.json").read_text())["q"])
        geom = json.loads((d / "robot_self_geometry.json").read_text())
        with np.load(d / "depth.npz") as z:
            depths = {v: z[v] for v in ("head", "left_wrist", "right_wrist")}
        points = observed_cloud(depths, model, q)
        own = robot_point_mask(points, geom)
        if own is None:
            raise ValueError("Actual robot-only self mask required")
        # Apply the source-frame self mask ONCE, not to accumulated obstacles
        # after the robot may already have driven into their old location.
        current = points[~own]
        current = current[(current[:, 2] > .025) & (current[:, 2] < 1.8)]
        memory.append(current)
        memory = memory[-16:]
        history = np.concatenate(memory)
        history = history[np.linalg.norm(history[:, :2], axis=1) < 2.5]
        _, unique = np.unique(np.floor(history / .01).astype(int), axis=0, return_index=True)
        history = history[unique]
        action = Action(**command["action"])
        if action.part != "base":
            continue
        chassis = [b for b in geom["boxes"] if b["link"] == "base_link" or b["link"].startswith(("wheel_", "steer_"))]
        corners = []
        for box in chassis:
            lo, hi = box["lower"], box["upper"]
            local = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            t = np.asarray(box["T_base_link"])
            corners.extend(local @ t[:3, :3].T + t[:3, 3])
        radius = float(np.max(np.linalg.norm(np.asarray(corners)[:, :2], axis=1)))
        old = LocalDepthGuard(points, model, q, depths)
        variants = {}
        for name, cloud in (("current_boxes_mask", current), ("old_capsule_mask", old.environment_points), ("history16", history)):
            for rname, rad in (("old_radius", .34), ("robot_box_radius", radius)):
                for zname, zmin in (("old_height", .1), ("low_height", .025)):
                    variants[f"{name}/{rname}/{zname}"] = circle_veto(cloud, action, rad, zmin)
        rows.append({"decision": command["decision"], "action": asdict(action), "old_result": old.check(action),
                     "robot_visual_corner_radius_m": radius, "current_points": len(current), "history_points": len(history),
                     "circle_variants": variants, "current_chassis_boxes": sweep(current, chassis, action),
                     "history_chassis_boxes": sweep(history, chassis, action)})
    payload = {"run": str(root), "new_controls": 0, "scene_truth_read": False,
               "history_decisions": 16, "history_voxel_m": .01, "wall_s": time.monotonic() - started, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"wall_s": payload["wall_s"], "base_actions": len(rows),
                     "near_contact": [r for r in rows if 49 <= r["decision"] <= 54]}, indent=2))


if __name__ == "__main__":
    main()
