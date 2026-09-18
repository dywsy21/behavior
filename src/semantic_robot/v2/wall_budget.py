"""Cooperative wall budget; never interrupts a simulator physics step.

Checks bracket CPU work and model requests. A single in-progress operation can
finish after the deadline, but it may not authorize a subsequent robot control.
The separately reserved final safe hold is intentionally exempt.
"""
import math
import time


class WallTimeBudgetReached(TimeoutError):
    pass


def remaining(deadline):
    if deadline is None:
        return None
    if not math.isfinite(deadline):
        raise ValueError("Finite absolute wall deadline required")
    return max(0., deadline - time.perf_counter())


def expired(deadline):
    value = remaining(deadline)
    return value is not None and value <= 0


def require_time(deadline):
    value = remaining(deadline)
    if value is not None and value <= 0:
        raise WallTimeBudgetReached("WALL_TIME_BUDGET_REACHED: no new computation or actuation")
    return value


def stop_before_motion(deadline, controls, row, decisions, manager):
    """Record a selected-but-unexecuted action separately from partial motion."""
    if not expired(deadline):
        return False
    reason = "WALL_TIME_BUDGET_REACHED_BEFORE_ACTION"
    row.update(control_end=controls, accepted_before_motion=False, stop_reason=reason,
               feedback={"status":"NOT_STARTED_WALL_TIME_BUDGET", "control_ticks":0})
    decisions.append(row)
    if manager is not None:
        manager.stop_reason = reason
    return True
