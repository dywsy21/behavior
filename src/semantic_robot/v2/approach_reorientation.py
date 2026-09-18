"""Bounded robot-only two-command preview, never an executable action queue."""
from dataclasses import asdict

import numpy as np

from .protocol import ROTATIONS, TRANSLATIONS
from .servo import SafeServo


def eligible(harness):
    """Only an observed open, unladen single hand may change approach pose."""
    fingers = harness.last_gripper
    observation = harness.observation
    distance = harness.grounding.get("distance_to_active_closing_center_m", 0)
    return bool(harness.approach_reorientation and not harness.stop_reason
                and harness.stage == "APPROACH" and harness.goal.kind == "pick"
                and harness.goal.hand in ("left", "right") and not harness.carry
                and all(isinstance(load, dict) and set(load) == {"left", "right"}
                        and all(value is False for value in load.values())
                        for load in (harness.pending_grasp, harness.hold_verified,
                                     harness.possible_contact_after_close))
                and fingers is not None and np.shape(fingers) == (2,)
                and np.isfinite(fingers).all() and np.all(np.asarray(fingers) >= .0495)
                and observation is not None and observation.visible and observation.hazard == "none"
                and harness.grounding.get("valid")
                and np.isfinite(distance) and distance > .10)


def preview(model, state, grips, limits, arm, point, rotation_trials, translations):
    """At most six accepted rotations x three coarse translations, all read-only.

    Every endpoint is predicted, not observed. These trials have only robot
    geometry: no environment/held-object collision or target-motion guarantee.
    Only the already-currently-preflighted rotation remains selectable.
    """
    point = np.asarray(point, dtype=float)
    if (point.shape != (3,) or not np.isfinite(point).all() or arm not in ("left", "right")
            or len(rotation_trials) > 6 or len(translations) > 3
            or not np.isfinite(grips).all() or np.shape(grips) != (2,)
            or not np.all((np.asarray(grips) >= .999) & (np.asarray(grips) <= 1.))):
        raise ValueError("Bounded, unladen approach preview inputs required")
    if any(a.part != arm or a.move not in TRANSLATIONS or a.scale != "coarse" for a in translations):
        raise ValueError("Preview only current-arm coarse translations")
    before = float(np.linalg.norm(point - model.grasp_centers(state.q)[arm]))
    rows, count = [], 0
    for rotation, trial in rotation_trials:
        if (rotation.part != arm or rotation.move not in ROTATIONS or trial.joint_plan is None
                or trial.status != "RUNNING" or trial.carry
                or not np.array_equal(trial.start.q, state.q)
                or not np.array_equal(trial.grips, grips)):
            raise ValueError("Preview requires exact current accepted unloaded rotation")
        endpoint = model.state(trial.joint_plan[-1].copy(), state.gripper.copy(), np.zeros(3))
        checks = []
        for action in translations:
            count += 1
            following = SafeServo(model, endpoint, np.asarray(grips).copy(), limits)
            ok = following.begin(action, endpoint, carry=False)
            row = {"action": asdict(action), "accepted_robot_only": bool(ok), "reason": following.status}
            if ok:
                after = float(np.linalg.norm(point - model.grasp_centers(following.joint_plan[-1])[arm]))
                row.update(net_two_command_distance_gain_m=before-after,
                           total_planned_ticks=trial.total_ticks+following.total_ticks)
            checks.append(row)
        feasible = [r for r in checks if r["accepted_robot_only"]]
        best = max(feasible, key=lambda r: r["net_two_command_distance_gain_m"], default=None)
        rows.append({"rotation": asdict(rotation), "following_robot_only_trials": checks,
                     "summary": {"feasible_coarse_followups": len(feasible),
                                 "best_followup": best["action"] if best else None,
                                 "best_two_command_gain_m": best["net_two_command_distance_gain_m"] if best else None,
                                 "prediction_not_execution_or_environment_safety": True}})
    return {"trigger": "CURRENT_COARSE_ARM_TRANSLATIONS_BLOCKED",
            "source": "current_RGBD_surface_point_and_robot_only_kinematics",
            "scene_truth": False, "future_state_is_prediction": True,
            "max_additional_preflights": 18, "additional_preflights": count,
            "future_actions_not_authorized": True, "rows": rows}
