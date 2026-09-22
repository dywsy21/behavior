"""Read-only setup-translation hints; no new palette or executable queue."""
from dataclasses import asdict

import numpy as np

from .protocol import TRANSLATIONS
from .servo import SafeServo
from .wall_budget import require_time


ROBOT_PATH_REJECTIONS = frozenset(("UNREACHABLE_OR_COLLISION_BLOCKED", "DURATION_LIMIT_EXCEEDED",
                                  "CARTESIAN_PATH_DEVIATION", "TRAJECTORY_LIMIT_OR_COLLISION"))


def preview(model, state, grips, limits, arm, point, first_trials, followup, deadline=None):
    """At most3 CURRENT accepted1cm translations, each with ONE predicted3cm reach.

    The caller retains the original depth-checked first-action set. Following
    states have no fresh depth: their feasibility is robot-only, not permission.
    """
    require_time(deadline)
    point = np.asarray(point, dtype=float)
    grips = np.asarray(grips, dtype=float)
    if (arm not in ("left", "right") or point.shape != (3,) or not np.isfinite(point).all()
            or not 1 <= len(first_trials) <= 3 or not limits.robot_geometry_guards
            or grips.shape != (2,) or not np.isfinite(grips).all()
            or not np.all((grips >= .999) & (grips <= 1.))
            or np.shape(state.gripper) != (2,) or not np.isfinite(state.gripper).all()
            or not np.all(state.gripper >= .0495)
            or followup.part != arm or followup.move not in TRANSLATIONS or followup.scale != "coarse"):
        raise ValueError("Bounded open-hand translation preview inputs required")
    seen = set()
    for action, trial in first_trials:
        if (action in seen or action.part != arm or action.move not in TRANSLATIONS or action.scale != "fine"
                or trial.model is not model or trial.limits is not limits or trial.action != action
                or trial.joint_plan is None or trial.status != "RUNNING" or trial.carry
                or not np.array_equal(trial.start.q, state.q)
                or not np.array_equal(trial.start.gripper, state.gripper)
                or not np.array_equal(trial.grips, grips)):
            raise ValueError("Exact current accepted unloaded fine translation required")
        seen.add(action)
    before = float(np.linalg.norm(point - model.grasp_centers(state.q)[arm]))
    rows = []
    for action, trial in first_trials:
        require_time(deadline)
        endpoint = model.state(trial.joint_plan[-1].copy(), state.gripper.copy(), np.zeros(3))
        following = SafeServo(model, endpoint, grips.copy(), limits)
        ok = following.begin(followup, endpoint, carry=False)
        require_time(deadline)
        check = {"action": asdict(followup), "accepted_robot_only": bool(ok), "reason": following.status}
        if ok:
            after = float(np.linalg.norm(point - model.grasp_centers(following.joint_plan[-1])[arm]))
            check.update(net_two_command_distance_gain_m=before-after,
                         total_planned_ticks=trial.total_ticks+following.total_ticks)
        rows.append({"translation": asdict(action), "following_robot_only_trial": check,
                     "summary": {"feasible_coarse_followups": int(ok),
                                 "best_followup": asdict(followup) if ok else None,
                                 "best_two_command_gain_m": check.get("net_two_command_distance_gain_m"),
                                 "total_planned_ticks": check.get("total_planned_ticks"),
                                 "prediction_not_execution_or_environment_safety": True}})
    return {"trigger": "CURRENT_COARSE_ARM_TRANSLATIONS_BLOCKED",
            "source": "current_RGBD_surface_point_and_robot_only_kinematics",
            "scene_truth": False, "future_state_is_prediction": True,
            "max_additional_preflights": 3, "additional_preflights": len(rows),
            "future_actions_not_authorized": True, "rows": rows}
