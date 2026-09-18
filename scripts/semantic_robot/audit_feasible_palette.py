"""Bounded CPU-only audit of actions hidden by the grounded candidate sampler."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.servo import SafeServo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--decision", type=int, action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= len(args.decision) <= 4:
        raise ValueError("At most four saved snapshots per CPU audit")
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = RobotModel(json.loads((run / "robot_calibration.json").read_text()))
    results = []
    for decision in args.decision:
        source = run / f"decision_{decision:03d}"
        saved = json.loads((source / "harness.json").read_text())
        proprio = json.loads((source / "proprio.json").read_text())
        state = model.state(np.asarray(proprio["q"]), np.asarray(proprio["gripper"]), np.zeros(3))
        original = json.loads((source / "observation.json").read_text())
        refined = json.loads((source / "refinement.json").read_text())
        obs = GroundedEvidence.parse(json.dumps(refined.get("refined_evidence", json.loads(original["result"]["text"]))))
        manager = GroundedHarness([Goal(**saved["goal"])])
        manager.stage = saved["stage"]
        manager.held = saved["held_target_claims"]
        manager.hold_verified = saved["holding_verified_by_observation_and_proprio"]
        candidate_receipt = json.loads((source / "candidates.json").read_text())
        servo = SafeServo(model, state, candidate_receipt["command_grip_latch"])
        controller = GroundedController(model, servo, manager)
        controller.observe(obs, state, dict(np.load(source / "depth.npz")),
                           json.loads((source / "depth_receipt.json").read_text()))
        # Audit exactly the saved stage; an observe() transition is not a real
        # control or a new observation in this offline enumeration.
        manager.stage = saved["stage"]
        old = {json.dumps(x["action"], sort_keys=True): x for x in candidate_receipt["tested"]}
        controller.candidates(state)
        new_sampler = manager.candidate_receipt
        rows = []
        palette = manager.palette()
        if len(palette) > 160:
            raise ValueError("Unexpectedly large palette")
        for action in palette:
            ok, why = controller.depth_guard.check(action, manager.carry)
            trial = SafeServo(model, state, servo.grips.copy(), servo.limits)
            if ok:
                ok = trial.begin(action, state, manager.carry)
                why = trial.status
            row = {"action": asdict(action), "feasible": bool(ok), "reason": why,
                   "tested_in_saved_sampler": json.dumps(asdict(action), sort_keys=True) in old}
            if ok and controller.target.get("valid"):
                distance = controller._expected_point(action, state, trial)
                if distance is not None:
                    row["predicted_gain_m"] = controller.target["distance_to_active_closing_center_m"] - distance
            rows.append(row)
        result = {"source": str(source), "saved_stage": saved["stage"], "rows": rows, "new_sampler": new_sampler,
                  "model_calls": 0, "controls": 0, "full_palette_size": len(palette)}
        results.append(result)
        (output / f"decision_{decision:03d}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"decision": decision, "full_palette_size": len(palette),
                          "hidden_feasible_count": sum(r["feasible"] and not r["tested_in_saved_sampler"] for r in rows),
                          "new_sampler_feasible": [r for r in new_sampler["tested"] if r["accepted"]]}, indent=2), flush=True)
    (output / "result.json").write_text(json.dumps({"results": results, "model_calls": 0, "controls": 0}, indent=2))


if __name__ == "__main__":
    main()
