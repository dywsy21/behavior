"""Compile a pose SEED from a separately budgeted *actual* expert replay.

Input is a full consecutive physics ledger, not annotation endpoints/17 frames.
No expert action becomes a native BC label. This CPU helper never runs replay.
"""
import json
from pathlib import Path
import numpy as np
from native_teacher_outcomes import LocalOutcome, rigid
from native_teacher_contract import digest
from common import sha


def seed_identity(reference, spec):
    source, label = reference["source"], reference["source_label"]
    return {**{k:source[k] for k in ("task","episode","instance","extracted_arrays_and_labels_sha256")},
            **{k:spec[k] for k in ("verb","hand","target","destination","support_hand","payloads","goal_frame")},
            "segment_start":label["segment_start"], "segment_end":label["segment_end"], "official_mode":"train", "seed":0}


def validate_seed_release(seed_path, review_path, spec, reference, preparation_sha):
    """Bind source, exact pose, successful full ledger AND independent review."""
    seed_path, review_path = Path(seed_path), Path(review_path)
    if sha(seed_path) != spec["pose_evidence_sha256"]: raise ValueError("Seed bytes changed")
    seed, review = json.loads(seed_path.read_text()), json.loads(review_path.read_text())
    identity = seed_identity(reference,spec)
    if (seed.get("schema") != "h09u-measured-reference-seed-v1" or seed.get("identity") != identity or
            seed.get("is_native_bc_label") is not False or seed.get("training_eligible") is not False or
            not isinstance(preparation_sha,str) or len(preparation_sha)!=64 or
            seed.get("reference_preparation_sha256")!=preparation_sha or
            not np.array_equal(rigid(seed["goal_pose_local"]),rigid(spec["goal_pose_local"]))):
        raise ValueError("Seed source/skill/hand/target/destination/pose mismatch")
    needed = {"PRIVATE_reference_trace.jsonl","result.json","captures.json"}
    if set(seed.get("evidence_sha256",{})) != needed:
        raise ValueError("Complete reference replay evidence required")
    for name, expected in seed["evidence_sha256"].items():
        path=seed_path.parent/name
        if path.stat().st_size > 32*1024**2 or sha(path)!=expected:raise ValueError("Reference evidence changed")
    rows=[json.loads(line) for line in (seed_path.parent/"PRIVATE_reference_trace.jsonl").read_text().splitlines()]
    if not 2<=len(rows)<=2001:raise ValueError("Bounded complete reference ledger required")
    if (rows[0]["frame"]["tick"]!=identity["segment_start"] or
            rows[-1]["frame"]["tick"]!=identity["segment_end"]+13):
        raise ValueError("Reference full segment/12 stability/final hold clock mismatch")
    measured=extract_seed(spec,rows)
    # Recomputing inverse/matmul on another CPU/BLAS changes final ulps.
    # Only this DERIVED matrix comparison tolerates numerical roundoff. Seed
    # bytes, seed/spec equality and independent review digests remain exact.
    if measured["outcome"]!=seed.get("outcome") or not np.allclose(
            rigid(measured["goal_pose_local"]),rigid(seed["goal_pose_local"]),atol=1e-12,rtol=0):
        raise ValueError("Seed not reproduced by its actual successful trajectory")
    result=json.loads((seed_path.parent/"result.json").read_text())
    if (result.get("status")!="REFERENCE_LOCAL_SUCCEEDED" or result.get("identity")!=identity or
            result.get("final_hold_completed") is not True or result.get("failure") is not None or
            result.get("completed_controls")!=identity["segment_end"]+13 or result.get("model_calls")!=0 or
            result.get("reference_preparation_sha256")!=preparation_sha):
        raise ValueError("Reference replay was interrupted/failed or a different source")
    captures=json.loads((seed_path.parent/"captures.json").read_text())
    required_captures={"before":identity["segment_start"],"segment_end":identity["segment_end"],
                       "after_stability":identity["segment_end"]+12,"after_final_hold":identity["segment_end"]+13}
    files={"head.png","left_wrist.png","right_wrist.png","depth.npz","robot_self_geometry.json",
           "sensors.json","proprio.json","capture.json"}
    for folder,tick in required_captures.items():
        entries=[c for c in captures if set(c["files_sha256"])=={folder+"/"+f for f in files}]
        if len(entries)!=1 or entries[0]["control"]!=tick:
            raise ValueError("Four clock-bound complete physical reference captures required")
    for capture in captures:
        for name, expected in capture["files_sha256"].items():
            path=(seed_path.parent/name).resolve()
            if not path.is_relative_to(seed_path.parent.resolve()) or sha(path)!=expected:
                raise ValueError("Reference visual review evidence changed")
    required={"seed_sha256","identity_sha256","goal_pose_sha256","reviewer","decision",
              "reviewed_source_current_and_terminal_views","reviewed_contact_update_and_control_ledger",
              "local_outcome_and_pose_accepted","reason"}
    if (set(review)!=required or review["seed_sha256"]!=sha(seed_path) or
            review["identity_sha256"]!=digest(identity) or review["goal_pose_sha256"]!=digest(seed["goal_pose_local"]) or
            review["reviewer"]!=spec["pose_reviewer"] or not review["reviewer"] or review["decision"]!="approve" or
            any(review[k] is not True for k in ("reviewed_source_current_and_terminal_views",
                 "reviewed_contact_update_and_control_ledger","local_outcome_and_pose_accepted")) or
            not isinstance(review["reason"],str) or len(review["reason"].strip())<20):
        raise ValueError("Independent source/pose/physical-evidence review is missing or stale")
    return seed


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
            edges=[e for e in frame["toggle_events"] if e["before"]["value"] is False and e["after"]["value"] is True]
            if len(edges)!=1:raise ValueError("Unique actual update edge pose required")
            m=edges[0]["before"]["measurement"]
            selected = np.linalg.inv(rigid(m["goal_parent_pose"]))@rigid(m["hand_poses"][spec["hand"]])
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
