"""Run the opt-in production sampler on at most three fixed saved states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.grounding import GroundedEvidence, LocalDepthGuard, observed_cloud
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.servo import SafeServo, ServoLimits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--decision", action="append", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-mode",choices=("reorientation","body","combined"),default="reorientation")
    args = parser.parse_args()
    root, out = Path(args.run), Path(args.output)
    if out.exists() or not 1 <= len(args.decision) <= 3 or len(set(args.decision)) != len(args.decision):
        raise ValueError("New output and <=3 distinct registered decisions required")
    started = time.monotonic()
    deadline=time.perf_counter()+180
    model = RobotModel(json.loads((root / "robot_calibration.json").read_text()))
    rows = []
    for decision in args.decision:
        source = root / f"decision_{decision:03d}"
        saved = json.loads((source / "harness.json").read_text())
        proprio = json.loads((source / "proprio.json").read_text())
        old = json.loads((source / "candidates.json").read_text())
        state = model.state(np.asarray(proprio["q"]), np.asarray(proprio["gripper"]), np.zeros(3))
        raw = json.loads((source / "observation.json").read_text())
        refined = json.loads((source / "refinement.json").read_text())
        evidence = GroundedEvidence.parse(json.dumps(refined.get("refined_evidence", json.loads(raw["result"]["text"]))))
        h = GroundedHarness([Goal(**saved["goal"])], contact_geometry=False,
                            approach_reorientation=args.candidate_mode in ("reorientation","combined"),
                            approach_body_options=args.candidate_mode in ("body","combined"))
        h.stage, h.observation, h.last_gripper = saved["stage"], evidence, state.gripper.copy()
        h.grounding = saved["target_surface_estimate"]
        h.pending_grasp = saved["unverified_close_latches"].copy()
        h.possible_contact_after_close = saved["possible_contact_after_any_close"].copy()
        h.hold_verified = saved["holding_verified_by_observation_and_proprio"].copy()
        servo = SafeServo(model, state, old["command_grip_latch"], ServoLimits(robot_geometry_guards=True))
        c = GroundedController(model, servo, h)
        c.target = h.grounding
        c.centers = model.grasp_centers(state.q)
        with np.load(source / "depth.npz") as z:
            depths = {key: z[key].copy() for key in z.files}
        geometry = json.loads((source / "robot_self_geometry.json").read_text())
        c.depth_guard = LocalDepthGuard(observed_cloud(depths, model, state.q), model, state.q, depths, self_geometry=geometry)
        before = state.q.copy()
        allowed = c.candidates(state,deadline=deadline)
        if not np.array_equal(before, state.q) or servo.status != "IDLE":
            raise AssertionError("Read-only sampler mutated live state")
        rows.append({"decision": decision, "allowed": [a.text() for a in allowed], "receipt": h.candidate_receipt,
                     "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}})
        print(json.dumps({"decision": decision, "allowed_count": len(allowed), "preflight_s": h.candidate_receipt["preflight_s"],
                          "allowed_body": [a.text() for a in allowed if a.part=="base"],
                          "rotation_summaries": [{"action": r["action"], "preview": r["reorientation_after"]}
                                                 for r in h.candidate_receipt["tested"] if "reorientation_after" in r]}), flush=True)
    with out.open("x") as f:
        json.dump({"rows": rows,"candidate_mode":args.candidate_mode, "wall_s": time.monotonic()-started, "new_controls": 0, "model_calls": 0,
                   "model_sha256": hashlib.sha256((root / "robot_calibration.json").read_bytes()).hexdigest(),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
