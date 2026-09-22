"""Explicit small-pose options in the existing8--10cm stage gap.

Qualification is not proof of no contact. Every added option requires a
negative-only visible arm sweep, including a NEW observation at execution.
"""
from dataclasses import asdict

import numpy as np

from .protocol import ROTATIONS, TRANSLATIONS


def public_checks(harness):
    ground = harness.grounding
    distance = ground.get("distance_to_active_closing_center_m")
    obs = harness.observation
    checks = {"enabled": bool(getattr(harness, "near_pose_gap", False)),
              "single_pick_approach": harness.stage == "APPROACH" and harness.goal.kind == "pick"
                                     and harness.goal.hand in ("left", "right"),
              "not_stopped": not harness.stop_reason,
              "original_stage_gap": isinstance(distance, (int, float)) and np.isfinite(distance) and .08 < distance <= .10,
              "visible_without_collision_or_enclosure": bool(obs and obs.visible and obs.hazard in ("none", "occluded")
                                                              and obs.enclosed is not True),
              "current_semantic_contact_confirmed": bool(ground.get("valid") and
                  ground.get("near_contact_review", {}).get("confirmed_current_contact") is True)}
    for name in ("pending_grasp", "hold_verified", "possible_contact_after_close", "workspace_close_seen"):
        value = getattr(harness, name, None)
        checks[name+"_clear"] = bool(isinstance(value, dict) and set(value) == {"left", "right"}
                                     and all(v is False for v in value.values()))
    checks["not_carrying"] = bool(all(checks[name+"_clear"] for name in
        ("pending_grasp", "hold_verified", "possible_contact_after_close", "workspace_close_seen")) and not harness.carry)
    fingers = harness.last_gripper
    checks["last_measured_open"] = bool(fingers is not None and np.shape(fingers) == (2,)
                                         and np.isfinite(fingers).all() and np.all(np.asarray(fingers) >= .0495))
    return checks


def eligible(harness):
    return all(public_checks(harness).values())


def qualification(harness, state, grips, model, limits):
    checks = public_checks(harness)
    checks["robot_geometry_guards"] = bool(limits.robot_geometry_guards)
    checks["current_calibrated_open_and_latches"] = False
    try:
        meta = model.spec["metadata"]
        reference = np.asarray(meta["grasp_region_reference_gripper_m"], dtype=float)
        measured, commanded = np.asarray(state.gripper), np.asarray(grips)
        checks["current_calibrated_open_and_latches"] = bool(
            reference.shape == measured.shape == commanded.shape == (2,)
            and np.isfinite(np.r_[reference, measured, commanded]).all()
            and all(meta["grasp_region_reference_fully_open"].get(a) is True for a in ("left", "right"))
            and np.all(abs(measured-reference) <= .0005) and np.all(measured >= .0495)
            and np.all((commanded >= .999) & (commanded <= 1.)))
    except (KeyError, TypeError, ValueError):
        pass
    return {"eligible": all(checks.values()), "checks": checks, "goal_index": harness.index,
            "stage": harness.stage, "not_contact_or_holding_certificate": True}


def execution_check(harness, state, grips, model, limits, action):
    result = qualification(harness, state, grips, model, limits)
    receipt = harness.candidate_receipt
    previous = receipt.get("near_pose_gap") or {}
    rows = [r for r in receipt.get("tested", []) if r.get("action") == asdict(action)]
    checks = result["checks"]
    checks["same_goal_stage"] = previous.get("goal_index") == harness.index and previous.get("stage") == harness.stage
    checks["current_offered_sweep_checked_option"] = bool(len(rows) == 1 and rows[0].get("accepted") is True
        and rows[0].get("near_pose_gap_option") is True and rows[0].get("near_pose_sweep", {}).get("safe") is True)
    checks["bounded_current_arm_action"] = bool(action.part == harness.goal.hand and
        ((action.move in ROTATIONS and action.scale in ("micro", "fine") and action.frame == "tool")
         or (action.move in TRANSLATIONS and action.scale == "fine")))
    before = np.asarray(receipt.get("state_q", []))
    checks["unchanged_observed_robot_state"] = bool(before.shape == state.q.shape and np.isfinite(before).all()
        and np.max(np.abs(before-state.q)) <= 1e-5 and np.array_equal(np.asarray(receipt.get("command_grip_latch", [])),grips))
    result["eligible"] = all(checks.values())
    result["fresh_RGBD_sweep_still_required"] = True
    return result


def execution_sweep(harness, selected_state, fresh_state, grips, model, limits, action, depths, geometry, joint_plan):
    """Caller must obtain depths/geometry AFTER model selection, with no motion."""
    result = execution_check(harness, fresh_state, grips, model, limits, action)
    unchanged = bool(np.max(np.abs(fresh_state.q-selected_state.q)) <= 1e-5 and
                     np.max(np.abs(fresh_state.gripper-selected_state.gripper)) <= 1e-5)
    result["checks"]["robot_unchanged_during_fresh_capture"] = unchanged
    if not result["eligible"] or not unchanged:
        return False, {**result, "eligible": False, "reason": "NEAR_POSE_QUALIFICATION_CHANGED"}
    from .arm_observation_guard import ObservingArmGuard
    from .grounding import observed_cloud
    guard = ObservingArmGuard(model, fresh_state.q, observed_cloud(depths,model,fresh_state.q,stride=6),geometry,action.part)
    safe, sweep = guard.check(joint_plan)
    return bool(safe), {**result, "reason": sweep["reason"], "sweep": sweep,
                       "fresh_RGBD_sweep_still_required": False}
