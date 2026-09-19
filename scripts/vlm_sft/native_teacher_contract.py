"""H09S: proposals are NOT labels; current-state manual teaching is mandatory.

Pure CPU helpers. No simulator/model import, success oracle, or auto-label path.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from common import TOKENS, POSITIONS, QUATERNIONS, GRIPS, token_to_action, sha
from live import validate_proprio

SCHEMA = "h09s-native-manual-teacher-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def verify_prepared_source(prepared, release):
    """Bind all reset inputs BEFORE creating an evaluator; never rewrite them."""
    from native_teacher_near_grasp import SCHEMA as NEAR_SCHEMA,verify_prepared
    if release.get("schema")==NEAR_SCHEMA:return verify_prepared(prepared,release)
    prepared = Path(prepared).resolve()
    manifest_path = prepared.parent / "preparation.json"
    if sha(manifest_path) != release.get("preparation_manifest_sha256"):
        raise ValueError("Unregistered preparation manifest")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != SCHEMA or
            manifest.get("status") != "PREPARED_NOT_COLLECTED_NOT_TRAINING_DATA" or
            len(manifest.get("sources", [])) != 3 or
            {r["task"] for r in manifest["sources"]} != {0, 1, 3}):
        raise ValueError("Exactly the reviewed three TRAIN preparations required")
    ref_path = prepared / "teacher_reference.json"
    ref = json.loads(ref_path.read_text())
    pilot = ref["pilot"]
    rows = [r for r in manifest["sources"] if r["task"] == pilot["task"]]
    if len(rows) != 1 or prepared.name != f"task_{pilot['task']}":
        raise ValueError("Prepared source ownership mismatch")
    row = rows[0]
    if (ref.get("schema") != SCHEMA or any(row[k] != pilot[k] for k in ("task", "episode", "instance", "frame", "verb")) or
            ref["prefix_controls"] != pilot["frame"]):
        raise ValueError("Prepared source identity mismatch")
    hashes = {"teacher_reference.json": row["reference_sha256"],
              "window.json": row["window_sha256"], "prefix.npy": row["prefix_sha256"]}
    if any(sha(prepared / name) != expected for name, expected in hashes.items()):
        raise ValueError("Prepared window/prefix/reference bytes changed")
    window = json.loads((prepared / "window.json").read_text())
    if (window["official_mode"] != "train" or window["seed"] != 0 or
            window["instance_id"] != pilot["instance"] or not window.get("task_name") or
            window["prefix_actions_sha256"] != hashes["prefix.npy"] or
            Path(window["prefix_actions_path"]).resolve() != prepared / "prefix.npy"):
        raise ValueError("Prepared reset or prefix path mismatch")
    prefix = np.load(prepared / "prefix.npy", allow_pickle=False)
    if prefix.shape != (pilot["frame"], 23) or prefix.dtype != np.float32 or not np.isfinite(prefix).all():
        raise ValueError("Exact finite full 23D prefix required")
    binding = {"preparation_manifest_sha256": sha(manifest_path), "files_sha256": hashes,
               "task": pilot["task"], "task_name": window["task_name"],
               "instance": pilot["instance"], "official_mode": "train", "seed": 0}
    return ref, prefix, binding


def verify_loaded_window(window, prefix, expected_prefix, binding):
    if (window.task_name != binding["task_name"] or window.official_mode != binding["official_mode"] or
            window.instance_id != binding["instance"] or window.seed != binding["seed"] or
            not np.array_equal(np.asarray(prefix), expected_prefix)):
        raise ValueError("Factory loaded a different task/reset/full prefix")


def select_sources(counts):
    blocked = {tuple(x) for group in counts["exclusions"].values() for x in group}
    selected = []
    for task in (0, 1, 3):
        pool = [r for r in counts["sources"] if r["task"] == task and r["cohort"] == "additional_train"
                and (task, r["instance"]) not in blocked]
        if not pool:
            raise ValueError("No unprotected TRAIN source")
        selected.append(min(pool, key=lambda r: hashlib.sha256(
            f'h09s:teacher:1:{task}:{r["instance"]}'.encode()).hexdigest()))
    return selected


def compile_candidates(states, actions):
    """Mixed expert motion is a REVIEW AID only, never decomposed BC truth.

    Refuse spatial ranking if the base moves: future body-frame coordinates
    cannot be transformed to today's body frame with the available 61D state.
    Even a stationary-base ranking needs current paused-state manual approval.
    """
    s, a = np.asarray(states, float), np.asarray(actions, float)
    if s.shape != (17, 61) or a.shape != (16, 23) or not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError("Expected aligned finite expert 17 states / 16 actions")
    out = {"label": None, "training_eligible": False, "ranked_proposals": [],
           "manual_choices": list(TOKENS), "not_executor_equivalence": True}
    if (np.linalg.norm(s[:, :2], axis=1).max() >= .02 or abs(s[:, 2]).max() >= .03
            or abs(a[:, :3]).max() >= .04):
        return dict(out, reason="MOVING_SOURCE_BASE_NO_COMMON_SPATIAL_FRAME")
    ranked = []
    positive, negative = ("FORWARD", "LEFT", "UP"), ("BACK", "RIGHT", "DOWN")
    for arm in ("left", "right"):
        dp = s[-1, POSITIONS[arm]] - s[0, POSITIONS[arm]]
        for i, amount in enumerate(dp):
            if abs(amount) >= .006:
                ranked.append((abs(amount) / .01, arm.upper()+"_"+(positive if amount > 0 else negative)[i]))
        rv = (Rotation.from_quat(s[-1, QUATERNIONS[arm]]) *
              Rotation.from_quat(s[0, QUATERNIONS[arm]]).inv()).as_rotvec()
        for i, amount in enumerate(rv):
            if abs(amount) >= np.deg2rad(2):
                ranked.append((abs(amount) / np.deg2rad(3), arm.upper()+"_"+
                               ("ROLL", "PITCH", "YAW")[i]+("_PLUS" if amount > 0 else "_MINUS")))
        change = s[-1, GRIPS[arm]].mean() - s[0, GRIPS[arm]].mean()
        if abs(change) >= .01:
            ranked.append((abs(change) / .01, arm.upper()+("_OPEN" if change > 0 else "_CLOSE")))
    out["ranked_proposals"] = [t for _, t in sorted(ranked, key=lambda x: (-x[0], x[1]))[:6]]
    out["reason"] = "OFFLINE_COMPONENT_PROPOSALS_REQUIRE_RETEACHING_AT_PAUSED_STATE"
    return out


def actor_input(task, instruction, proprio, image_hashes, history):
    """Explicit projection; no source frame, hidden ID/pose, future or verdict."""
    validate_proprio(proprio)
    if not isinstance(task, str) or not task or not isinstance(instruction, str) or not instruction:
        raise ValueError("Task and independently revalidated instruction required")
    if set(image_hashes) != {"head", "left_wrist", "right_wrist"}:
        raise ValueError("Three current RGB identities required")
    if any(not isinstance(v, str) or len(v) != 64 for v in image_hashes.values()):
        raise ValueError("Image hashes required")
    if len(history) > 5 or any(t not in TOKENS for t in history):
        raise ValueError("Only actually executed native history")
    return {"task": task, "active_instruction": instruction, "proprio": proprio,
            "current_rgb_sha256": image_hashes, "history": list(history)}


def validate_approval(value, request):
    required = {"request_sha256", "reviewer", "decision", "token", "intent_still_valid",
                "judged_correct_next_action", "reason", "reviewed_current_and_source_views"}
    if set(value) != required or value["request_sha256"] != digest(request):
        raise ValueError("Approval must bind this exact paused observation and intent")
    if value["decision"] == "reject":
        return None
    if (value["decision"] != "approve" or value["token"] not in TOKENS or value["token"] not in request["allowed_tokens"] or
            value["intent_still_valid"] is not True or value["judged_correct_next_action"] is not True or
            value["reviewed_current_and_source_views"] is not True or
            not isinstance(value["reviewer"], str) or not value["reviewer"].strip() or
            not isinstance(value["reason"], str) or len(value["reason"].strip()) < 20):
        raise ValueError("Manual current-state teaching, not a bare PASS, required")
    return value["token"]


def release_reviewed_record(record, post):
    """Only a manual judgment can promote a measured demonstration.

    This checks evidence binding, not the truth of an expert's assessment.
    Output must still pass the H09R dataset coverage and independent audit gates.
    """
    if (post.get("record_sha256") != digest(record) or post.get("decision") != "approve" or
            post.get("correct_for_current_intent") is not True or
            post.get("reviewed_full_before_after_and_native_trace") is not True or
            post.get("not_based_only_on_safety_or_distance") is not True or
            not isinstance(post.get("reviewer"), str) or not post["reviewer"].strip() or
            not isinstance(post.get("reason"), str) or len(post["reason"].strip()) < 20):
        raise ValueError("Missing independently reviewable post-action judgment")
    token = validate_approval(record["approval"], record["request"])
    executed = record["execution"]
    if (token is None or executed["token"] != token or executed["action"] != asdict(token_to_action(token)) or
            executed["status"] != "TARGET_REACHED" or executed["interrupted"] or
            executed["native_controls"] <= 0 or not executed["native_trace_sha256"] or
            not record.get("post_observation_sha256") or record.get("settle_passed") is not True):
        raise ValueError("Approved primitive must actually finish; legality is not correctness")
    return {"actor": record["request"]["actor"], "target": token,
            "source_group": record["request"]["source_group"],
            "label_basis": "manual_current_state_teacher_and_reviewed_native_execution",
            "record_sha256": digest(record), "post_review_sha256": digest(post),
            "official_success_claim": False}
