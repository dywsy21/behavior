"""Saved-state camera/arm candidate audit. Never controls physics or calls a VLM."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.observer_geometry import anchor_camera_views, angle_degrees
from semantic_robot.v2.protocol import Action, ROTATIONS, TRANSLATIONS
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.grounding import GroundedEvidence, LocalDepthGuard, observed_cloud
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import HOLD


def read(path):
    return json.loads(path.read_text())


def audit(run, decisions, timeout, integrated=False):
    started = time.monotonic()
    model = RobotModel(read(run / "robot_calibration.json"))
    results = []
    for decision in decisions:
        directory = run / f"decision_{decision:03d}"
        h = read(directory / "harness.json")
        inspection = h.get("held_inspection", {})
        if not inspection.get("valid"):
            raise ValueError("Saved registered held anchor is required")
        anchor = inspection["anchor"]
        if anchor["source"] != "previous_verified_onboard_RGBD_surface_not_current_affordance":
            raise ValueError("Unknown anchor source")
        reference = inspection["reference_hand"]
        free = "left" if reference == "right" else "right"
        if h["holding_verified_by_observation_and_proprio"].get(free) or h["unverified_close_latches"].get(free):
            raise ValueError("Other hand is not free")
        if h["carry_constraints"]:
            raise ValueError("This saved-state audit is registered for nonlevel loads only")
        proprio = read(directory / "proprio.json")
        state = model.state(np.array(proprio["q"]), np.array(proprio["gripper"]), np.zeros(3))
        with np.load(directory / "depth.npz") as bundle:
            depths = {name: bundle[name] for name in bundle.files}
        before = anchor_camera_views(model, state.q, reference, anchor["point_hand_m"], depths)
        palette = [Action(arm, move, scale, "tool" if move in ROTATIONS else "base")
                   for arm in (reference, free) for scale in ("coarse", "fine")
                   for move in (*TRANSLATIONS, *ROTATIONS)]
        grips = np.array([1., 1.]); grips[("left", "right").index(reference)] = -1.
        row = {"decision": decision, "reference_hand": reference, "free_hand": free, "before": before, "candidates": []}
        if integrated:
            manager = GroundedHarness([Goal(**h["goal"])], held_inspection=True, inspection_budget_aware=True,
                                      multicamera_inspection=True, reference_from_planner=True)
            manager.held = h["held_target_claims"].copy()
            manager.hold_verified = h["holding_verified_by_observation_and_proprio"].copy()
            manager.pending_grasp = h["unverified_close_latches"].copy()
            manager.bind_reference("held_" + reference, "saved_sensor_registered_reference")
            # No new image label is fabricated: this is the saved missing-target
            # state, not a neural prediction on a generated image.
            manager.observation = GroundedEvidence(False, "none", None, None, None, None, None, "none", "Saved missing affordance")
            manager.stage = h["stage"]
            servo = SafeServo(model, state, grips.copy())
            inspector = GroundedController(model, servo, manager).inspector
            inspector.anchors[reference] = anchor
            inspector.observe(model, state, manager, depths, read(directory / "robot_self_geometry.json"))
            inspector.states[manager.index].update(attempts=inspection["attempts"], path_m=inspection["relative_path_m"],
                                                   rotation_rad=np.deg2rad(inspection["relative_rotation_deg"]))
            allowed, receipt = inspector.candidates(model, state, manager, servo,
                LocalDepthGuard(observed_cloud(depths, model, state.q), model, state.q, depths))
            row.update(context=inspector.context(model, state, manager), receipt=receipt,
                       allowed=[asdict(a) for a in allowed], stop_reason=manager.stop_reason,
                       history_scope="current pose only; historical path/attempt totals retained; not a replay of all prior poses",
                       free_guard_valid=inspector.free_guard.valid, free_guard_reason=inspector.free_guard.reason)
            if time.monotonic() - started > timeout:
                raise TimeoutError("Registered CPU audit budget exceeded")
            results.append(row)
            continue
        for action in palette:
            if time.monotonic() - started > timeout:
                raise TimeoutError("Registered CPU audit budget exceeded")
            servo = SafeServo(model, state, grips.copy())
            accepted = servo.begin(action, state, carry=False)
            entry = {"action": asdict(action), "ik_accepted": accepted, "reason": servo.status}
            if accepted:
                after = anchor_camera_views(model, servo.joint_plan[-1], reference, anchor["point_hand_m"])
                entry["predicted_cameras"] = after
                entry["pointing_improvement_deg"] = {view: before[view]["bearing_error_deg"] - after[view]["bearing_error_deg"] for view in before}
                entry["view_direction_change_deg"] = {view: angle_degrees(before[view]["camera_direction_in_reference_hand"], after[view]["camera_direction_in_reference_hand"]) for view in before}
                entry["planned_ticks"] = servo.total_ticks
            row["candidates"].append(entry)
        results.append(row)
    return {"run": str(run), "wall_seconds": time.monotonic() - started, "controls": 0, "model_calls": 0,
            "rows": results, "scope": "CPU saved-sensor prediction, not new images or executed effects", "integrated": integrated,
            "environment_clearance_certified": False, "scene_truth_actor_input": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--decisions", nargs="+", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=900)
    p.add_argument("--integrated", action="store_true", help="Use opt-in runtime candidates and actual visible-depth sweep veto")
    a = p.parse_args()
    if len(a.decisions) > 4 or not 0 < a.max_seconds <= 900:
        raise ValueError("Bounded saved-state audit required")
    if a.output.exists():
        raise ValueError("Preserve previous result")
    result = audit(a.run, a.decisions, a.max_seconds, a.integrated)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    for row in result["rows"]:
        free = row["free_hand"]
        if a.integrated:
            accepted = [entry for entry in row["receipt"]["tested"] if entry["accepted"] and entry["action"]["part"] == free]
            print(json.dumps({"decision": row["decision"], "free_guard_valid": row["free_guard_valid"],
                              "free_guard_reason": row["free_guard_reason"], "free_accepted": len(accepted),
                              "best": sorted(accepted, key=lambda r: r["inspection_after"]["pointing_gain_deg"], reverse=True)[:2]}))
            continue
        choices = [entry for entry in row["candidates"] if entry["ik_accepted"] and entry["action"]["part"] == free]
        best = sorted(choices, key=lambda entry: entry["pointing_improvement_deg"][free + "_wrist"], reverse=True)[:3]
        print(json.dumps({"decision": row["decision"], "before": row["before"], "best_free_camera_pointing": best}, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
