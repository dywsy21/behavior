"""One sensor-based displacement adjudicator for policy and engineering gates.

This does not certify task success or repair hard controller failures. The
caller must supply the quality-gated RGB-D pair spanning the executed action.
"""
import copy

import numpy as np

from .protocol import TRANSLATIONS


def observed_motion_feedback(action, raw_feedback, receipt):
    if receipt.get("valid") is not True or receipt.get("initial", False):
        raise ValueError("A valid non-initial before/after RGB-D motion pair is required")
    delta = np.asarray(receipt.get("body_delta"), dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("Finite three-dimensional observed body displacement required")
    feedback = copy.deepcopy(raw_feedback)
    feedback["base_velocity_integral_raw"] = copy.deepcopy(raw_feedback["base_integral"])
    feedback["base_integral"] = delta.tolist()
    feedback["base_motion_source"] = "onboard_RGBD_not_joint_velocity_integration"
    feedback["base_motion_convention"] = "displacement_in_previous_body_frame"
    if (action is not None and action.part == "base" and
            feedback["status"] in ("TARGET_REACHED", "BASE_TRACKING_FAILED")):
        expected = np.zeros(3)
        amount = action.amount(feedback.get("carry", False))
        if action.move in TRANSLATIONS:
            expected[:2] = np.asarray(TRANSLATIONS[action.move])[:2] * amount
        elif action.move in ("yaw_plus", "yaw_minus"):
            expected[2] = amount * (1 if action.move == "yaw_plus" else -1)
        residual = delta - expected
        feedback["base_tracking_status_from_velocity_raw"] = feedback["status"]
        feedback["visual_base_residual"] = residual.tolist()
        feedback["status"] = ("TARGET_REACHED" if np.linalg.norm(residual[:2]) < .012
                              and abs(residual[2]) < np.deg2rad(2.) else "BASE_TRACKING_FAILED")
    return feedback
