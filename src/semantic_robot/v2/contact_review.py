"""Bounded semantic review of near-field contact rays, not grasp certification.

Depth validity cannot establish object identity. Previous points only schedule
a fresh visual question; they NEVER replace the current target or guide motion.
"""
from dataclasses import asdict
import hashlib
import json

import numpy as np


def binding(harness, state, evidence, target, images):
    if not isinstance(images, dict) or set(images) != {"head", "left_wrist", "right_wrist"}:
        raise ValueError("Contact review requires the exact three current RAW views")
    pixels = {}
    for name, image in images.items():
        value = np.asarray(image)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError("Current RAW uint8 RGB required")
        pixels[name] = {"shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    value = {"goal_index": harness.index, "stage": harness.stage,
             "q": np.asarray(state.q).tolist(), "gripper": np.asarray(state.gripper).tolist(),
             "evidence": asdict(evidence), "raw_images": pixels, "target": target}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class NearContactReview:
    """Uses the policy's EXISTING16 total surface choices, no additional calls."""
    near_m = .25
    jump_m = .07

    def __init__(self):
        self.goal = None
        self.previous = None
        self.confirmed = False

    def request(self, harness, state, evidence, target, model):
        if harness.index != self.goal:
            self.goal = harness.index
            self.previous = None
            self.confirmed = False
        eligible = (harness.goal.kind == "pick" and harness.goal.hand in ("left", "right")
                    # SEARCH/RECOVER may transition to APPROACH in this very
                    # observation; do not give their first near command a bypass.
                    and harness.stage in ("SEARCH", "RECOVER", "APPROACH", "ALIGN") and not harness.stop_reason
                    and not harness.carry and not any(harness.pending_grasp.values())
                    and not any(harness.hold_verified.values())
                    and not any(harness.possible_contact_after_close.values()))
        receipt = {"required": False, "reason": "OUTSIDE_UNLADEN_SINGLE_HAND_APPROACH",
                   "near_m": self.near_m, "point_jump_m": None}
        if not eligible or not target.get("valid"):
            self.previous = None
            self.confirmed = False
            return receipt
        point = np.asarray(target["point_base_m"], dtype=float)
        center = model.grasp_centers(state.q)[harness.goal.hand]
        distance = float(np.linalg.norm(point-center))
        receipt["distance_m"] = distance
        if distance > self.near_m:
            self.previous = None
            self.confirmed = False
            receipt["reason"] = "OUTSIDE_NEAR_FIELD"
            return receipt
        reason = None
        if not self.confirmed or self.previous is None:
            reason = "FIRST_OR_UNCONFIRMED_NEAR_CONTACT"
        elif evidence.view != self.previous["view"]:
            reason = "CONTACT_VIEW_CHANGED"
        else:
            motion = harness.motion_receipt
            body = np.asarray(motion.get("body_transform_current_in_previous", []), dtype=float)
            if not (motion.get("valid") and body.shape == (4, 4) and np.isfinite(body).all()
                    and np.allclose(body[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                    and np.allclose(body[:3,:3].T@body[:3,:3], np.eye(3), atol=1e-5, rtol=0)
                    and abs(np.linalg.det(body[:3,:3])-1) < 1e-5):
                reason = "NO_MEASURED_CONTACT_CONTINUITY"
            else:
                expected = np.linalg.inv(body) @ np.r_[self.previous["point"], 1.]
                jump = float(np.linalg.norm(point-expected[:3]))
                receipt["point_jump_m"] = jump
                if jump > self.jump_m:
                    reason = "CURRENT_CONTACT_POINT_JUMP"
        receipt.update(required=reason is not None, reason=reason or "CURRENT_CONTACT_CONTINUITY")
        return receipt

    def finish(self, request, harness, state, evidence, target, selected_current_surface, images):
        approved = bool(request["required"] and selected_current_surface and target.get("valid"))
        if request["required"]:
            self.confirmed = approved
        if "distance_m" in request and request["distance_m"] <= self.near_m and target.get("valid"):
            self.previous = {"view": evidence.view, "point": np.asarray(target["point_base_m"]).copy()}
        return {"version": "near_contact_review_v1", **request,
                "confirmed_current_contact": approved,
                "snapshot_binding": binding(harness, state, evidence, target, images),
                "success_or_grasp_claim": False}


def apply_review(target, receipt, harness, state, evidence, images):
    """Preserve visible-object evidence; an unconfirmed ray is not actionable."""
    if (not isinstance(receipt, dict) or receipt.get("version") != "near_contact_review_v1"
            or type(receipt.get("required")) is not bool
            or type(receipt.get("confirmed_current_contact")) is not bool
            or receipt.get("snapshot_binding") != binding(harness, state, evidence, target, images)
            or receipt.get("success_or_grasp_claim") is not False):
        raise ValueError("Same-frame contact review receipt required")
    if receipt["confirmed_current_contact"] and (not receipt["required"] or not target.get("valid")):
        raise ValueError("Contact confirmation requires a current valid selected ray")
    result = {**target, "near_contact_review": receipt}
    if receipt["required"] and not receipt["confirmed_current_contact"]:
        result.update(valid=False, reason="VISIBLE_TARGET_CONTACT_UNCONFIRMED")
    return result
