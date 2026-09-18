"""Counterfactual CPU preflight on immutable saved observations, no simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.servo import SafeServo, ServoLimits
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.grounding import LocalDepthGuard, observed_cloud


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError("Preserve diagnostic identity")
    started, runs = time.monotonic(), []
    for path in a.run:
        root = Path(path)
        spec = json.loads((root / "robot_calibration.json").read_text())
        reference = json.loads((root / "decision_000/proprio.json").read_text())
        if np.max(np.abs(np.asarray(reference["q"]) - spec["q_reference"])) > 1e-5:
            raise ValueError("Reset/reference robot geometry mismatch")
        spec["metadata"]["robot_visual_boxes_reference"] = json.loads((root / "decision_000/robot_self_geometry.json").read_text())
        spec["metadata"]["parallel_gripper_open_envelope"] = "R1Pro_parallel_prismatic_jaws"
        model = RobotModel(spec)
        result = json.loads((root / "result.json").read_text())
        trace = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
        latches = {r["control"]: np.asarray(r["action23"])[[14, 22]] for r in trace if "action23" in r}
        rows = []
        for command in result["decisions"]:
            if time.monotonic() - started > 480:
                raise TimeoutError("Registered fixed-source preflight budget")
            d = root / f"decision_{command['decision']:03d}"
            saved = json.loads((d / "proprio.json").read_text())
            state = model.state(np.asarray(saved["q"]), np.asarray(saved["gripper"]), np.zeros(3))
            action = Action(**command["action"])
            # The reference envelope must be open; subsequent gate CLOSE/OPEN
            # states need their actual last commanded latch, not a reset latch.
            latch = (latches[command["control_start"]] if command["control_start"] else
                     np.clip(state.gripper / .05 * 2 - 1, -1, 1))
            new = SafeServo(model, state, latch, ServoLimits(robot_geometry_guards=True))
            ok = new.begin(action, state)
            row = {"decision": command["decision"], "action": command["action"],
                   "old_actually_accepted": command["accepted_before_motion"],
                   "new_servo_accepted": ok, "new_servo_reason": new.status,
                   "new_initial_hand_body_clearance": new.collision.hand_body.clearance(state.q)}
            if action.part == "base":
                with np.load(d / "depth.npz") as z:
                    depths = {v: z[v] for v in ("head", "left_wrist", "right_wrist")}
                points = observed_cloud(depths, model, state.q)
                geom = json.loads((d / "robot_self_geometry.json").read_text())
                row["old_depth"] = LocalDepthGuard(points, model, state.q, depths).check(action)
                row["new_depth"] = LocalDepthGuard(points, model, state.q, depths, self_geometry=geom).check(action)
            rows.append(row)
        runs.append({"run": path, "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ("manifest.json", "result.json", "steps.jsonl", "robot_calibration.json", "decision_000/robot_self_geometry.json")},
                     "rows": rows})
    payload = {"runs": runs, "wall_s": time.monotonic() - started, "new_controls": 0, "model_calls": 0,
               "no_scene_truth": True, "not_physical_effect_proof": True,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"wall_s": payload["wall_s"], "runs": [{"run": r["run"], "changed": [x for x in r["rows"]
          if x["old_actually_accepted"] != x["new_servo_accepted"] or x.get("old_depth") != x.get("new_depth")]} for r in runs]}, indent=2))


if __name__ == "__main__":
    main()
