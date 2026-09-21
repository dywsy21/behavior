"""Public, bounded body-first workspace previews; not queued motion or success."""
from dataclasses import asdict

import numpy as np

from .protocol import TRANSLATIONS
from .servo import SafeServo
from .wall_budget import require_time


def qualification(harness, state, grips, model):
    """Calibrated open commands qualify timing, never certify no contact/load."""
    checks = {
        "enabled": bool(getattr(harness, "workspace_posture", False)),
        "single_hand_pick": harness.goal.kind == "pick" and harness.goal.hand in ("left", "right"),
        "approach_or_align": harness.stage in ("APPROACH", "ALIGN"),
        "not_stopped": not harness.stop_reason,
        "not_carrying": not harness.carry,
    }
    for name in ("pending_grasp", "hold_verified", "possible_contact_after_close", "workspace_close_seen"):
        value = getattr(harness, name, None)
        checks[name + "_clear"] = bool(isinstance(value, dict) and set(value) == {"left", "right"}
                                           and all(v is False for v in value.values()))
    obs = harness.observation
    checks["visible_without_hazard_or_enclosure"] = bool(obs is not None and obs.visible
                                                       and obs.hazard == "none" and obs.enclosed is not True)
    ground = harness.grounding
    distance = ground.get("distance_to_active_closing_center_m")
    checks["valid_current_target"] = bool(ground.get("valid") and isinstance(distance, (int, float))
                                           and np.isfinite(distance) and distance > 0)
    checks["calibrated_full_open_both_and_open_latches"] = False
    try:
        metadata = model.spec["metadata"]
        reference = np.asarray(metadata["grasp_region_reference_gripper_m"], dtype=float)
        current = np.asarray(state.gripper, dtype=float)
        commanded = np.asarray(grips, dtype=float)
        checks["calibrated_full_open_both_and_open_latches"] = bool(
            reference.shape == current.shape == commanded.shape == (2,)
            and np.isfinite(np.r_[reference, current, commanded]).all()
            and all(metadata["grasp_region_reference_fully_open"].get(a) is True for a in ("left", "right"))
            and np.all(abs(current - reference) <= .0005) and np.all(current >= .0495)
            and np.all((commanded >= .999) & (commanded <= 1.)))
    except (KeyError, TypeError, ValueError):
        pass
    return {"eligible": all(checks.values()), "checks": checks,
            "goal_index": harness.index, "stage": harness.stage,
            "not_contact_or_holding_certificate": True}


def execution_check(harness, state, grips, model, limits, action):
    """Recheck public qualification and exact offered FIRST command after latency."""
    result=qualification(harness,state,grips,model)
    receipt=harness.candidate_receipt
    previous=receipt.get("workspace_posture") or {}
    match=[row for row in receipt.get("tested",[]) if row.get("action")==asdict(action)]
    checks=result["checks"]
    checks["robot_geometry_guards"]=bool(limits.robot_geometry_guards)
    checks["exact_fine_torso_command"]=bool(action.part=="torso" and action.scale=="fine"
        and action.frame=="base" and action.move in ("up","down","forward","back"))
    checks["exact_current_goal_and_stage"]=bool(previous.get("goal_index")==harness.index
        and previous.get("stage")==harness.stage)
    checks["exact_offered_first_action"]=bool(len(match)==1 and match[0].get("accepted") is True
        and match[0].get("offered_to_policy") is True and match[0].get("workspace_posture_after"))
    result["eligible"]=all(checks.values())
    result["actual_servo_recheck_still_required"]=True
    return result


def preview(model, state, grips, limits, arm, point, torso_trials, followups, deadline=None):
    """At most four current-safe torso poses x three already-blocked translations.

    Current-frame depth checks belong to the caller. Future robot-only checks
    do not certify environment clearance; the actor must reobserve after the
    FIRST action, and there is no executable multi-action queue.
    """
    point = np.asarray(point, dtype=float)
    grips = np.asarray(grips, dtype=float)
    if (not limits.robot_geometry_guards or arm not in ("left", "right") or point.shape != (3,) or not np.isfinite(point).all()
            or grips.shape != (2,) or not np.isfinite(grips).all()
            or not np.all((grips >= .999) & (grips <= 1.))
            or len(torso_trials) > 4 or len(followups) > 3
            or len(set(followups)) != len(followups)):
        raise ValueError("Bounded, finite unladen workspace preview required")
    if any(a.part != arm or a.move not in TRANSLATIONS or a.scale not in ("fine", "coarse") for a in followups):
        raise ValueError("Workspace followups are moderate current-arm translations")
    if len({a for a, _ in torso_trials}) != len(torso_trials):
        raise ValueError("Duplicate torso preview")
    before = float(np.linalg.norm(point - model.grasp_centers(state.q)[arm]))
    rows, count = [], 0
    for action, trial in torso_trials:
        require_time(deadline)
        if (action.part != "torso" or action.scale != "fine" or action.frame != "base"
                or action.move not in ("up", "down", "forward", "back")
                or trial.action != action or trial.joint_plan is None or trial.status != "RUNNING"
                or trial.carry or trial.done or trial.model is not model or trial.limits != limits
                or not np.array_equal(trial.start.q, state.q) or not np.array_equal(trial.grips, grips)):
            raise ValueError("Workspace preview requires an exact current accepted torso preflight")
        endpoint = model.state(trial.joint_plan[-1].copy(), state.gripper.copy(), np.zeros(3))
        checks = []
        for following in followups:
            require_time(deadline)
            probe = SafeServo(model, endpoint, grips.copy(), limits)
            ok = probe.begin(following, endpoint, carry=False)
            require_time(deadline)
            count += 1
            row = {"action": asdict(following), "accepted_robot_only": bool(ok), "reason": probe.status}
            if ok:
                after = float(np.linalg.norm(point - model.grasp_centers(probe.joint_plan[-1])[arm]))
                row.update(net_two_command_distance_gain_m=before-after,
                           total_planned_ticks=trial.total_ticks+probe.total_ticks)
            checks.append(row)
        feasible = [r for r in checks if r["accepted_robot_only"]]
        best = max(feasible, key=lambda r: r["net_two_command_distance_gain_m"], default=None)
        rows.append({"torso": asdict(action), "following_robot_only_trials": checks,
                     "summary": {"feasible_followups": len(feasible),
                                 "best_followup": best["action"] if best else None,
                                 "best_two_command_gain_m": best["net_two_command_distance_gain_m"] if best else None,
                                 "prediction_not_execution_or_environment_safety": True}})
    return {"source": "current_RGBD_target_and_robot_only_kinematics", "scene_truth": False,
            "future_state_is_prediction": True, "future_actions_not_authorized": True,
            "max_additional_preflights": 12, "additional_preflights": count, "rows": rows}
