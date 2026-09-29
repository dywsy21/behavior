"""Candidate-only bridge: no strategy action may bypass Jev in this mode.

Geometry, contact protocols and budgets can veto; they cannot choose a winner.
Predicted search coverage is never written into measured coverage.
"""
from dataclasses import asdict
import hashlib
import json
import math

from .jev_client import JevError
from .protocol import Action, HOLD
from .servo import SafeServo
from .wall_budget import require_time


def search_candidates(c, state, deadline=None):
    h = c.harness
    if c.is_held_search:
        raise ValueError("Held targets require relative inspection, not world search")
    if h.stop_reason or c.depth_guard is None:
        raise JevError("Stopped or missing current depth; no search choice")
    search = c.search
    if (search.search_travel_m + search.unknown_budget_travel_m > 1.2 or
            search.search_rotation_rad + search.unknown_budget_rotation_rad > 4*math.pi):
        h.stop_reason = "SEARCH_TRAVEL_BUDGET"
        return (), "search_budget_veto"
    palette = h.palette()
    # Torso fine has an additional workspace-mode execution qualification,
    # which does not apply to unseen-target SEARCH. Do not offer it here.
    proposed = [a for a in palette if a == HOLD or a.part == "base" or
                (a.part == "torso" and a.scale == "micro")]
    if len(proposed) > 25:
        raise ValueError("Bounded search palette exceeded; do not silently prune directions")
    camera = c.model.spec["metadata"]["cameras"]["head"]
    half_fov = .7*math.atan(camera["width"]/(2*camera["K"][0][0]))
    allowed, tested = [], []
    for action in proposed:
        require_time(deadline)
        h.authorize(action)
        ok, reason = c.depth_guard.check(action, h.carry)
        if c.approach_monitor is not None and not c.approach_monitor.allowed(action):
            ok, reason = False, "REPEATED_BASE_APPROACH_WITHOUT_CONTACT_PROGRESS"
        trial = None
        if ok:
            trial = SafeServo(c.model, state, c.servo.grips.copy(), c.servo.limits)
            ok = trial.begin(action, state, h.carry)
            reason = "CURRENT_SEARCH_PATH_PREFLIGHTED" if ok else trial.status
        row = {"action": asdict(action), "accepted": bool(ok), "reason": reason}
        if ok:
            row["planned_ticks"] = trial.total_ticks
            prediction = {"is_prediction_not_measured_coverage": True,
                          "target_visibility_not_predicted": True}
            if action.part == "base" and action.move in ("yaw_plus", "yaw_minus"):
                yaw = action.amount(h.carry)*(1 if action.move == "yaw_plus" else -1)
                heading = search.observed_heading+yaw
                bins = [i for i in range(search.bins)
                        if abs(math.atan2(math.sin(2*math.pi*i/search.bins-heading),
                                          math.cos(2*math.pi*i/search.bins-heading))) <= half_fov]
                prediction.update(commanded_yaw_deg=math.degrees(yaw),
                    predicted_heading_deg=math.degrees(heading),
                    predicted_new_heading_bins=[i for i in bins if i not in search.covered])
            row["search_prediction"] = prediction
            allowed.append(action)
        tested.append(row)
    require_time(deadline)
    h.candidate_receipt = {"source": "jev_unseen_target_search_candidates_v2", "tested": tested,
        "search": search.context(), "depth_guard": c.depth_guard.receipt(),
        "command_grip_latch": c.servo.grips.tolist(), "no_command_selected_in_code": True}
    return tuple(allowed), "world_search"


def candidates(c, state, deadline=None):
    """All normal control branches terminate in a palette, never an action."""
    require_time(deadline)
    h = c.harness
    if h.stop_reason:
        raise JevError("Hard stop forbids all Jev motion choices")
    if c.goal_changed:
        h.candidate_receipt = {"source": "new_goal_requires_observation", "tested": [
            {"action": asdict(HOLD), "accepted": True, "reason": "FRESH_GOAL_OBSERVATION_REQUIRED"}]}
        return (HOLD,), "goal_observation_barrier"
    if c.press_cycle is not None and c.press_cycle.owns_selection:
        return c.press_cycle.candidates(state, deadline, model_selects=True), "press_protocol"
    if c.is_held_search:
        return c.inspection_candidates(state), "held_inspection"
    if ((h.observation and not h.observation.visible and h.stage in ("SEARCH", "RECOVER"))
            or c.reposition_left > 0):
        # A recovery recommendation is context, not a queued five-action macro.
        return search_candidates(c, state, deadline)
    allowed = c.candidates(state, deadline)
    if c.grasp_verifier is not None and h.stage == "VERIFY_GRASP":
        obs = h.observation
        observable = (obs is not None and obs.visible and c.target.get("valid") and
                      obs.enclosed is not False and obs.co_moving is not False and obs.hazard == "none")
        lift = Action(h.goal.hand, "up", "fine")
        if not observable or lift not in allowed:
            h.stop_reason = "NO_SAFE_OBSERVABLE_GRASP_LIFT"
            return (), "verification_safety_veto"
        # The registered sensing protocol needs a 1cm lift. Jev selects whether
        # to execute it, observe with HOLD, or abstain; code never issues it.
        allowed = tuple(a for a in allowed if a in (HOLD, lift))
        return allowed, "grasp_verification"
    return allowed, "grounded_control"


def authority(action, receipt, allowed):
    """Fail closed if any route tries to label a script action as a Jev choice."""
    result = receipt.get("result", {})
    proof = result.get("jev_authority", {})
    digest = hashlib.sha256(json.dumps([asdict(a) for a in allowed], sort_keys=True,
                                      allow_nan=False).encode()).hexdigest()
    if action not in allowed:
        raise JevError("Selected action not in current palette")
    option = f"command_{allowed.index(action):03d}"
    if (proof.get("schema") != "jev-action-choice-v2" or proof.get("model") != "jev-1.13.0"
            or type(proof.get("command_call")) is not int or proof["command_call"] < 2
            or proof.get("allowed_sha256") != digest or proof.get("action") != asdict(action)
            or proof.get("selected_option") != option or result.get("decision_call_count") != 2
            or result.get("answers", {}).get("command", {}).get("choice") != option):
        raise JevError("Missing or mismatched Jev action ownership receipt")
    return dict(proof)


def validate_actor_ownership(result, ledger=None, controls=None):
    """Completed run is not accepted as a Jev control test without execution."""
    rows = [r for r in result.get("decisions", []) if r.get("accepted_before_motion")
            and r.get("control_end", 0) > r.get("control_start", 0)]
    if not rows:
        raise JevError("No Jev-selected action executed; control integration not validated")
    calls, probe_controls = [], 0
    validated = {} if ledger is None else {r["call"]: r for r in ledger if r.get("event") == "validated"}
    def check_ledger(call, question, option):
        if ledger is not None and validated.get(call, {}).get("selections", {}).get(question) != option:
            raise JevError("Executed choice differs from durable Jev response")
    for row in rows:
        proof = row.get("jev_authority", {})
        call = proof.get("command_call")
        if (not row.get("selection_source", "").startswith("Jev_all_")
                or proof.get("schema") != "jev-action-choice-v2"
                or proof.get("model") != "jev-1.13.0" or proof.get("action") != row.get("action")
                or not isinstance(proof.get("selected_option"), str)
                or not proof["selected_option"].startswith("command_")
                or type(call) is not int or not 2 <= call <= result.get("jev_requests", 0)):
            raise JevError("Executed non-Jev strategy action in Jev-owned run")
        calls.append(call)
        check_ledger(call, "command", proof.get("selected_option"))
        recovery = row.get("search_reanchor", {})
        if recovery.get("control_end", 0) > recovery.get("control_start", 0):
            authorization = recovery.get("jev_recovery_authorization", {})
            if (not isinstance(authorization, dict) or
                    authorization.get("schema") != "jev-search-recovery-choice-v1" or
                    authorization.get("model") != "jev-1.13.0" or
                    authorization.get("choice") != "measure_stationary_reference" or
                    type(authorization.get("call")) is not int or
                    not call < authorization["call"] <= result["jev_requests"]):
                raise JevError("Unrequested automatic search-recovery probe")
            check_ledger(authorization["call"], "tracking_recovery", authorization["choice"])
            calls.append(authorization["call"])
            probe_controls += recovery["control_end"]-recovery["control_start"]
    if calls != sorted(set(calls)):
        raise JevError("Reused or out-of-order Jev action receipt")
    if controls is not None:
        # Trace every issued control, including the separately allowed emergency
        # safe stop. A planning request cannot launder unowned motor commands.
        ticks = [r for r in controls if "control" in r]
        if [r["control"] for r in ticks] != list(range(1, result["controls"]+1)):
            raise JevError("Missing, duplicated or unaccounted physical control ticks")
        by_decision = {r["decision"]: r for r in rows}
        safe_stops = 0
        for tick in ticks:
            if tick.get("safety_stop") is True:
                safe_stops += 1
                if tick["control"] != result["controls"] or safe_stops > 1:
                    raise JevError("Unbounded or nonterminal safety HOLD")
                continue
            if "decision" in tick:
                row = by_decision.get(tick["decision"], {})
                if not row.get("control_start", 0) < tick["control"] <= row.get("control_end", 0):
                    raise JevError("Issued control without Jev-owned decision")
            else:
                row = by_decision.get(tick.get("reanchor_after_decision"), {})
                recovery = row.get("search_reanchor", {})
                if (not recovery.get("control_start", 0) < tick["control"] <= recovery.get("control_end", 0)
                        or tick.get("jev_recovery_authorization") != recovery.get("jev_recovery_authorization")):
                    raise JevError("Issued unowned recovery control")
    return {"normal_actions_executed": len(rows), "jev_owned_actions": len(rows),
            "jev_requested_probe_controls": probe_controls,
            "non_jev_strategy_actions": 0, "safety_stop_is_not_policy_action": True,
            "durable_choices_checked": ledger is not None, "all_control_ticks_checked": controls is not None}
