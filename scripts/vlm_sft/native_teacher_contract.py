"""H09S: proposals are NOT labels; current-state manual teaching is mandatory.

Pure CPU helpers. No simulator/model import, success oracle, or auto-label path.
"""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

from common import TOKENS, POSITIONS, QUATERNIONS, GRIPS, token_to_action
from live import validate_proprio

SCHEMA = "h09s-native-manual-teacher-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


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
            not record.get("post_observation_sha256")):
        raise ValueError("Approved primitive must actually finish; legality is not correctness")
    return {"actor": record["request"]["actor"], "target": token,
            "source_group": record["request"]["source_group"],
            "label_basis": "manual_current_state_teacher_and_reviewed_native_execution",
            "record_sha256": digest(record), "post_review_sha256": digest(post),
            "official_success_claim": False}
