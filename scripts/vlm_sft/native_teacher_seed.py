"""Compile a pose SEED from a separately budgeted *actual* expert replay.

Input is a full consecutive physics ledger, not annotation endpoints/17 frames.
No expert action becomes a native BC label. This CPU helper never runs replay.
"""
import numpy as np
from native_teacher_outcomes import LocalOutcome, rigid


def extract_seed(spec, rows):
    if not rows: raise ValueError("No actual full-segment physics evidence")
    oracle = LocalOutcome(spec)
    selected = None
    previous_command = None
    for row in rows:
        frame = row["frame"]
        raw = row.get("actual_action23")
        token = None
        if raw is not None:
            command = np.asarray(raw, float)
            if command.shape != (23,) or not np.isfinite(command).all():
                raise ValueError("Actual finite full native23 action required")
            grip = command[14 if spec["hand"] == "left" else 22]
            # Commands only establish a causal attempt; outcome independently
            # requires contact/lift/edge/support/stability, never command alone.
            if grip < -.5: token=spec["hand"].upper()+"_CLOSE"
            elif grip > .5: token=spec["hand"].upper()+"_OPEN"
            previous_command = command
        elif previous_command is not None:
            raise ValueError("A later physical sample lacks its actual action")
        result = oracle.update(frame, token)
        if result["outcome"] in ("FAILED", "UNKNOWN"):
            raise ValueError("Reference replay not valid: "+result["reason"])
        if spec["verb"] == "PRESS" and oracle.edge_seen and selected is None:
            selected = np.linalg.inv(rigid(frame["goal_parent_pose"]))@rigid(frame["hand_poses"][spec["hand"]])
    if previous_command is None or oracle.result["outcome"] != "SUCCEEDED":
        raise ValueError("Annotation tail is not a completed physical skill")
    last = rows[-1]["frame"]
    if spec["verb"] == "GRASP":
        selected = np.linalg.inv(rigid(last["target_pose"]))@rigid(last["hand_poses"][spec["hand"]])
    elif spec["verb"].startswith("PLACE"):
        selected = np.linalg.inv(rigid(last["goal_parent_pose"]))@rigid(last["target_pose"])
    return {"goal_pose_local": rigid(selected).tolist(), "outcome": oracle.result,
            "physics_controls": len(rows)-1, "first_control": rows[0]["frame"]["tick"],
            "last_control": last["tick"], "is_native_bc_label": False,
            "source_and_independent_visual_review_required": True}
