"""Finite, robot-only saved-state diagnostic; never imports the simulator.

Tests existing wrist primitives and a subsequent existing coarse translation.
Predicted endpoint states are NOT physical observations or scene-safe plans.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.protocol import Action, ROTATIONS
from semantic_robot.v2.servo import SafeServo, ServoLimits


class DiagnosticServo(SafeServo):
    def _solve(self, q, targets, step):
        result = super()._solve(q, targets, step)
        self.last_solver_q = result.copy()
        self.solver_steps = getattr(self, "solver_steps", 0) + 1
        return result


def run(args):
    source, output = Path(args.run), Path(args.output)
    if output.exists():
        raise ValueError("Do not overwrite earlier diagnostics")
    if len(args.decision) > 3 or len(set(args.decision)) != len(args.decision):
        raise ValueError("At most three distinct saved decisions")
    started = time.monotonic()
    model = RobotModel(json.loads((source / "robot_calibration.json").read_text()))
    limits = ServoLimits(robot_geometry_guards=True)
    calls = 0

    def trial(action, state, latch, point):
        nonlocal calls
        if calls >= 144 or time.monotonic() - started > 590:
            raise TimeoutError("Fixed 144-trial/600-second CPU block exhausted")
        calls += 1
        servo = DiagnosticServo(model, state, latch, limits)
        checks = {"count": 0, "negative": 0, "minimum_clearance_m": None}
        clearance = servo.collision.clearance

        def observed_clearance(q):
            value = clearance(q)
            checks["count"] += 1
            checks["negative"] += int(value < 0)
            checks["minimum_clearance_m"] = min(value, checks["minimum_clearance_m"]) if checks["minimum_clearance_m"] is not None else value
            return value

        servo.collision.clearance = observed_clearance
        before = float(np.linalg.norm(model.grasp_centers(state.q)["right"] - point))
        ok = servo.begin(action, state, carry=False)
        q = servo.joint_plan[-1] if ok and servo.joint_plan is not None else getattr(servo, "last_solver_q", state.q)
        margin = np.minimum(q - model.lower, model.upper - q)
        record = {"action": asdict(action), "accepted": bool(ok), "reason": servo.status,
                  "solver_steps": getattr(servo, "solver_steps", 0), "collision_queries": checks,
                  "endpoint_q": q.tolist(), "endpoint_joint_margin_min": float(margin.min()),
                  "closest_joint_index": int(margin.argmin()),
                  "endpoint_robot_clearance_m": float(clearance(q)),
                  "endpoint_hand_body_clearance": servo.collision.hand_body.clearance(q),
                  "before_distance_m": before,
                  "endpoint_distance_m": float(np.linalg.norm(model.grasp_centers(q)["right"] - point)),
                  "pose_errors": {name: error.tolist() for name, error in servo._errors(model.poses(q, servo.targets), servo.targets).items()},
                  "ticks_if_accepted": servo.total_ticks if ok else None}
        # Failed solver iterates are diagnostic only; never chain them as states.
        after = model.state(q, state.gripper.copy(), np.zeros(3)) if ok else None
        return record, after

    result = {"run": str(source), "new_controls": 0, "new_model_calls": 0,
              "scene_truth": False, "physical_effect_claim": False,
              "limits": asdict(limits), "rows": [],
              "model_file_sha256": hashlib.sha256((source / "robot_calibration.json").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        for decision in args.decision:
            folder = source / f"decision_{decision:03d}"
            saved = json.loads((folder / "proprio.json").read_text())
            harness = json.loads((folder / "harness.json").read_text())
            candidates = json.loads((folder / "candidates.json").read_text())
            if harness["stage"] != "APPROACH" or harness["goal"]["hand"] != "right" or harness["carry_constraints"]:
                raise ValueError("Fixed diagnostic requires unloaded right-arm APPROACH")
            state = model.state(np.asarray(saved["q"]), np.asarray(saved["gripper"]), np.zeros(3))
            latch = np.asarray(candidates["command_grip_latch"])
            point = np.asarray(harness["target_surface_estimate"]["point_base_m"])
            row = {"decision": decision, "source_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                   for name in ("proprio.json", "harness.json", "candidates.json")},
                   "target_point": point.tolist(), "baseline": [], "rotations": []}
            result["rows"].append(row)
            for move in ("forward", "left", "up"):
                record, _ = trial(Action("right", move, "coarse", "base"), state, latch, point)
                row["baseline"].append(record)
            # First two states have fine and coarse; final state fine only.
            scales = ("fine", "coarse") if decision != args.decision[-1] else ("fine",)
            for move in ROTATIONS:
                for scale in scales:
                    record, after = trial(Action("right", move, scale, "tool"), state, latch, point)
                    row["rotations"].append({"rotation": record, "following_translations": []})
                    if after is not None:
                        for direction in ("forward", "left", "up"):
                            following, _ = trial(Action("right", direction, "coarse", "base"), after, latch, point)
                            row["rotations"][-1]["following_translations"].append(following)
    finally:
        result["wall_s"] = time.monotonic() - started
        result["trial_count"] = calls
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"wall_s": result["wall_s"], "trial_count": calls, "rows": [
        {"decision": row["decision"], "baseline": [{"action": r["action"], "ok": r["accepted"], "reason": r["reason"],
         "closest_joint": r["closest_joint_index"], "margin": r["endpoint_joint_margin_min"], "negative_collision_checks": r["collision_queries"]["negative"]} for r in row["baseline"]],
         "rotations_unlocking_coarse": [{"action": r["rotation"]["action"], "following": [t["action"]["move"] for t in r["following_translations"] if t["accepted"]]}
         for r in row["rotations"] if any(t["accepted"] for t in r["following_translations"])]} for row in result["rows"]]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--decision", action="append", type=int, required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
