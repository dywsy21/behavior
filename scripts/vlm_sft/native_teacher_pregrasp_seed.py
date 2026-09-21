"""Explicit offline GRASP pre-contact seed; old terminal seeds stay immutable.

The selected frame can be part-way through CLOSE. Its aperture is evidence,
NEVER an execution target, rotation qualification, or an actor input. The
complete successful reference and its existing independent review are still
validated by the unchanged v1 path before any new candidate can be compiled.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import sha
from native_teacher_contract import digest
from native_teacher_outcomes import rigid, stable
from native_teacher_seed import extract_seed, seed_identity, validate_seed_release

PROFILE = "h09y-grasp-precontact-v2"
SCHEMA = "h09y-precontact-reference-seed-v2"
REVIEW_SCHEMA = "h09y-precontact-seed-review-v2"
RULE = "FIRST_TARGET_FINGER_CONTACT_PREVIOUS_ACTUAL_FRAME_IN_SINGLE_CLOSE_ATTEMPT"
# Fixed successful TRAIN references only. No heldout-derived candidate/search.
TERMINAL_SHAS = {
    (1, 310, 192): "bc8865c719e84ab73b43bb2f0ba7cbaa975ce0bf1612ab1053f6073804cf4b7d",
    (1, 264, 114): "476a6ef7f262a9c189e78f05bd9646e27e3a0c1c490095c72750b44c0b20f9a0",
}
SPEC_KEYS = ("verb", "hand", "support_hand", "target", "destination", "payloads", "goal_frame")


def select_precontact(spec, rows):
    """Deterministic first-attempt rule, not a best-pose or future-score search."""
    if spec["verb"] != "GRASP" or spec["goal_frame"] != "target":
        raise ValueError("Precontact seed supports GRASP only")
    # Retain full causal lift/stability/clock checks, not merely first contact.
    terminal = extract_seed(spec, rows)
    arm = spec["hand"]; grip_index = 14 if arm == "left" else 22
    commands = [None if r.get("actual_action23") is None else
                float(np.asarray(r["actual_action23"])[grip_index]) for r in rows]
    closes = [i for i, command in enumerate(commands) if command is not None and command < -.5]
    if not closes or closes[0] == 0:
        raise ValueError("Measured open predecessor and actual CLOSE attempt required")
    first_close = closes[0]
    if commands[first_close-1] is None or commands[first_close-1] <= .5:
        raise ValueError("First CLOSE must follow an actual OPEN command")
    # No reopening, dropping and choosing a later successful retry. Exactly the
    # first contiguous causal CLOSE attempt must establish the terminal hold.
    if any(c is None or c >= -.5 for c in commands[first_close:]):
        raise ValueError("Interrupted/reopened CLOSE attempt; no alternate seed search")
    contacts = []
    for i, row in enumerate(rows):
        f = row["frame"]
        if any(type(f["held"].get(a)) is not bool or type(f["finger_contact"].get(a)) is not bool
               for a in ("left", "right")):
            raise ValueError("Known actual target hold/contact is required")
        if f["finger_contact"][arm]: contacts.append(i)
    if not contacts or contacts[0] < first_close:
        raise ValueError("First target contact must follow this actual CLOSE")
    contact = contacts[0]; selected = contact-1
    if selected < first_close-1 or any(any(r["frame"]["held"].values()) for r in rows[:contact]):
        raise ValueError("No pre-held target or missing contact predecessor")
    f = rows[selected]["frame"]
    if any(f["finger_contact"].values()):
        raise ValueError("Selected frame must have no target finger contact in either hand")
    held = [i for i in range(contact, len(rows)) if rows[i]["frame"]["held"][arm]]
    if not held or any(r["frame"]["held"][arm] is not True or
                       r["frame"]["finger_contact"][arm] is not True for r in rows[held[0]:]):
        raise ValueError("First hold must retain target-identity contact through terminal hold")
    aperture = f["finger_opening"][arm]
    if (type(aperture) not in (float, int) or not np.isfinite(aperture) or not 0 < aperture <= .051):
        raise ValueError("Measured finite precontact finger aperture required")
    target_window = [r["frame"]["target_pose"] for r in rows[max(0, selected-11):selected+1]]
    if len(target_window) != 12 or not stable(target_window):
        raise ValueError("Twelve actual stable-target frames before first contact required")
    pose = np.linalg.inv(rigid(f["target_pose"])) @ rigid(f["hand_poses"][arm])
    selection = {
        "rule": RULE, "selected_tick": f["tick"],
        "first_close_tick": rows[first_close]["frame"]["tick"],
        "first_contact_tick": rows[contact]["frame"]["tick"],
        "first_held_tick": rows[held[0]]["frame"]["tick"],
        "selected_frame_sha256": digest(rows[selected]),
        "measured_finger_aperture_m": aperture,
        "actual_grip_command": commands[selected],
        "aperture_is_full_open_evidence": False, "aperture_is_execution_target": False,
        "target_stability_first_tick": rows[selected-11]["frame"]["tick"],
        "target_stability_frames": 12,
        "target_stability_translation_m": .004, "target_stability_angle_deg": 3.,
    }
    return {"goal_pose_local": rigid(pose).tolist(), "selection": selection,
            "terminal_goal_pose_local": terminal["goal_pose_local"], "outcome": terminal["outcome"]}


def _bound_file(path, expected=None):
    p = Path(path)
    if not p.is_absolute() or str(p.resolve()) != str(p) or not p.is_file() or p.stat().st_size > 32*1024**2:
        raise ValueError("Canonical bounded regular reference file required")
    if expected is not None and sha(p) != expected:
        raise ValueError("Bound reference/parent-review bytes changed")
    return p


def build_sidecar(terminal_seed_path, terminal_review_path, reference_path):
    """Read/revalidate the original evidence in place; never edit or copy it."""
    terminal_path = _bound_file(terminal_seed_path)
    review_path = _bound_file(terminal_review_path)
    source_path = _bound_file(reference_path)
    old = json.loads(terminal_path.read_text()); review = json.loads(review_path.read_text())
    reference = json.loads(source_path.read_text()); identity = old["identity"]
    key = tuple(identity[k] for k in ("task", "episode", "instance"))
    if (key not in TERMINAL_SHAS or sha(terminal_path) != TERMINAL_SHAS[key] or
            identity["verb"] != "GRASP" or identity["official_mode"] != "train" or
            identity["seed"] != 0 or terminal_path.name != "QUARANTINED_pose_seed.json"):
        raise ValueError("Only the two registered successful TRAIN terminal references are accepted")
    spec = {"schema": "h09t-private-teacher-v1", **{k: identity[k] for k in SPEC_KEYS},
            "goal_pose_local": old["goal_pose_local"], "pose_evidence_sha256": sha(terminal_path),
            "pose_reviewer": review["reviewer"]}
    from native_teacher_policy import validate_spec
    validate_spec(spec, reference)
    validate_seed_release(terminal_path, review_path, spec, reference, old["reference_preparation_sha256"])
    root = terminal_path.parent
    rows = [json.loads(s) for s in (root/"PRIVATE_reference_trace.jsonl").read_text().splitlines()]
    measured = select_precontact(spec, rows)
    return {"schema": SCHEMA, "seed_profile": PROFILE, "identity": identity,
            "reference_root": str(root), "reference_preparation_sha256": old["reference_preparation_sha256"],
            "reference_source_path": str(source_path), "reference_source_sha256": sha(source_path),
            "terminal_seed_sha256": sha(terminal_path), "terminal_review_path": str(review_path),
            "terminal_review_sha256": sha(review_path), "evidence_sha256": old["evidence_sha256"],
            **measured, "is_native_bc_label": False, "training_eligible": False}


def validate_pregrasp_release(seed_path, review_path, spec, reference, preparation_sha):
    seed_path = _bound_file(seed_path, spec["pose_evidence_sha256"])
    review_path = _bound_file(review_path)
    seed = json.loads(seed_path.read_text()); review = json.loads(review_path.read_text())
    if seed.get("schema") != SCHEMA or seed.get("seed_profile") != PROFILE:
        raise ValueError("Explicit precontact sidecar/profile required; no legacy fallback")
    root = Path(seed["reference_root"])
    if (not root.is_absolute() or str(root.resolve()) != str(root) or not root.is_dir() or
            seed_path.is_relative_to(root)):
        raise ValueError("Precontact sidecar must be separate from its preserved canonical reference")
    terminal_path = _bound_file(root/"QUARANTINED_pose_seed.json", seed["terminal_seed_sha256"])
    old_review = _bound_file(seed["terminal_review_path"], seed["terminal_review_sha256"])
    source_path = _bound_file(seed["reference_source_path"], seed["reference_source_sha256"])
    fresh = build_sidecar(terminal_path, old_review, source_path)
    # Exact metadata/selection and spec equality; only recomputed SE(3) allows
    # the same 1e-12 BLAS-roundoff tolerance as the unchanged terminal validator.
    matrices = ("goal_pose_local", "terminal_goal_pose_local")
    if ({k: v for k, v in seed.items() if k not in matrices} !=
            {k: v for k, v in fresh.items() if k not in matrices} or
            any(not np.allclose(rigid(seed[k]), rigid(fresh[k]), atol=1e-12, rtol=0) for k in matrices) or
            seed["identity"] != seed_identity(reference, spec) or
            seed["reference_preparation_sha256"] != preparation_sha or
            not np.array_equal(rigid(seed["goal_pose_local"]), rigid(spec["goal_pose_local"]))):
        raise ValueError("Precontact source/pose/selection/full-success reproduction mismatch")
    required = {"schema", "seed_profile", "seed_sha256", "identity_sha256", "goal_pose_sha256",
                "selection_sha256", "reviewer", "decision", "reviewed_source_current_and_terminal_views",
                "reviewed_contact_update_and_control_ledger", "reviewed_precontact_phase_and_half_closed_aperture",
                "local_outcome_and_pose_accepted", "reason"}
    flags = ("reviewed_source_current_and_terminal_views", "reviewed_contact_update_and_control_ledger",
             "reviewed_precontact_phase_and_half_closed_aperture", "local_outcome_and_pose_accepted")
    if (set(review) != required or review["schema"] != REVIEW_SCHEMA or review["seed_profile"] != PROFILE or
            review["seed_sha256"] != sha(seed_path) or review["identity_sha256"] != digest(seed["identity"]) or
            review["goal_pose_sha256"] != digest(seed["goal_pose_local"]) or
            review["selection_sha256"] != digest(seed["selection"]) or
            review["reviewer"] != spec["pose_reviewer"] or not review["reviewer"] or
            review["decision"] != "approve" or any(review[k] is not True for k in flags) or
            not isinstance(review["reason"], str) or len(review["reason"].strip()) < 20):
        raise ValueError("Independent precontact-phase/complete-reference review is missing or stale")
    return seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-seed", required=True)
    parser.add_argument("--terminal-review", required=True)
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_sidecar(args.terminal_seed, args.terminal_review, args.reference_source)
    output = args.output.absolute()
    if output.resolve() != output or output.is_relative_to(Path(value["reference_root"])):
        raise ValueError("Fresh separate canonical sidecar path required")
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if len(encoded.encode()) > 32768:
        raise ValueError("Bounded sidecar exceeds 32KiB")
    with output.open("x") as stream: stream.write(encoded)
    print(json.dumps({"path": str(output), "sha256": sha(output), "selection": value["selection"],
                      "training_eligible": False, "new_resets": 0, "parent_review_required": True}))
