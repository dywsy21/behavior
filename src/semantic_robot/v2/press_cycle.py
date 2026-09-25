"""Bounded public-sensor PRESS execution; never a contact/success oracle.

This executor stabilizes the existing grip command, not a newly invented tool
aperture. It owns purposeful holds separately from the ordinary emergency HOLD.
Only the runner's actual post-control samples can advance its execution chain.
"""
from collections import deque
from dataclasses import asdict
import copy

import numpy as np

from .finger_kinematics import FingerKinematics
from .motion_feedback import observed_motion_feedback
from .protocol import Action, HOLD, TRANSLATIONS
from .servo import SafeServo, execution_completed
from .wall_budget import require_time


VERSION = "measured-press-cycle-v1"
STABLE_TICKS = 6
FINGER_SPAN_M = .0005
FINGER_STEP_M = .0001
NEAR_M = .008
DWELL_M = .003
RETRACT_M = .003
MAX_ADVANCES = 4
MAX_RETRACTIONS = 3
MAX_VERIFY_VIEWS = 3


def checked_action_motion(receipt, start, end):
    """Require the full current action's RGB-D chain, not next-decision qvel.

    The source is the runner-owned SubstepMotion; a public VLM output has no
    route here. Recompose every measured segment and bind its control interval.
    """
    from .substep_odometry import checked_transform
    if (not isinstance(receipt, dict) or receipt.get("valid") is not True
            or receipt.get("source") != "fixed_six_control_RGBD_SE3_chain"
            or receipt.get("no_scene_truth") is not True or receipt.get("velocity_integral_not_used") is not True
            or receipt.get("robot_self_exclusion") is not True
            or type(receipt.get("control_start")) is not int or receipt["control_start"] != start
            or type(receipt.get("control_end")) is not int or receipt["control_end"] != end
            or type(receipt.get("substep_interval_controls")) is not int
            or receipt["substep_interval_controls"] != 6):
        raise ValueError("PRESS_CURRENT_ACTION_RGBD_CHAIN_REQUIRED")
    segments = receipt.get("segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= end-start:
        raise ValueError("PRESS_INCOMPLETE_RGBD_SEGMENTS")
    total, previous = np.eye(4), start
    for segment in segments:
        if (type(segment.get("control_start")) is not int or segment["control_start"] != previous
                or type(segment.get("control_end")) is not int
                or not 0 < segment["control_end"]-previous <= 6
                or segment["control_end"] > end):
            raise ValueError("PRESS_RGBD_SEGMENT_CLOCK_MISMATCH")
        measurement = segment.get("measurement", {})
        if measurement.get("valid") is not True or measurement.get("initial", False):
            raise ValueError("PRESS_RGBD_SEGMENT_INVALID")
        total = total @ checked_transform(measurement)
        previous = segment["control_end"]
    if previous != end or not np.allclose(total, checked_transform(receipt), atol=1e-8, rtol=0):
        raise ValueError("PRESS_RGBD_ACTION_COMPOSITION_MISMATCH")
    return receipt


class PressCycle:
    def __init__(self, controller):
        self.c = controller
        if controller.press_asset is None:
            raise ValueError("Press cycle requires the fixed finger surface asset")
        self.fingers = FingerKinematics(controller.model.spec["metadata"].get("finger_kinematics"))
        self.key = None
        self.phase = "IDLE"
        self.selected = None
        self.running = None
        self.last_receipt = None
        self.observation_control = None
        self.last_verify_control = None
        self.verify_views = self.confirmations = 0
        self.verified_claims = []
        self.advances = self.retractions = 0
        self.finger_reference = None
        self.grips = None
        self.reference_id = None
        self.anchor = self.away = self.retreat_origin = None
        self.dwell_end = self.retract_end = None
        self.effect_after_dwell = None
        self.stop_reason = None

    @property
    def manager(self):
        return self.c.harness

    @property
    def active(self):
        return self.manager.goal.kind == "press"

    @property
    def tool_ready(self):
        return self.finger_reference is not None and self.phase != "STOPPED"

    @property
    def owns_selection(self):
        return self.active and self.phase not in ("IDLE", "APPROACH", "FINISHED")

    def goal_key(self):
        g = self.manager.goal
        return (self.manager.index, g.kind, g.target, g.hand, g.done_when,
                g.level, self.manager.search_reference)

    def stop(self, reason):
        self.stop_reason = self.stop_reason or reason
        self.manager.stop_reason = self.manager.stop_reason or self.stop_reason
        self.phase = "STOPPED"
        self.selected = None
        return False

    def transition(self, phase):
        self.phase = phase
        self.manager.transition("APPROACH" if phase == "APPROACH" else "PRESS_"+phase)

    def _values(self, state):
        # Validate every named joint, including the other (possibly loaded) hand.
        identity, _ = self.c.press_asset.context(self.c.model, state.q, state.finger_qpos)
        if hasattr(self, "calibration_id") and identity != self.calibration_id:
            raise ValueError("PRESS_CALIBRATION_CHANGED")
        arm = self.manager.goal.hand
        if arm not in ("left", "right"):
            raise ValueError("PRESS_REQUIRES_ONE_FREE_HAND")
        names, _, lower, upper, _ = self.fingers.arms[arm]
        values = np.array([state.finger_qpos[n] for n in names])
        if any((self.manager.held[arm] is not None, self.manager.hold_verified[arm],
                self.manager.pending_grasp[arm], self.manager.possible_contact_after_close[arm])):
            raise ValueError("PRESS_HAND_POSSIBLY_LOADED")
        grips = np.asarray(self.c.servo.grips, dtype=float)
        if (grips.shape != (2,) or not np.isfinite(grips).all() or np.any(np.abs(grips) > 1)
                or (self.grips is not None and not np.array_equal(grips, self.grips))):
            raise ValueError("PRESS_LATCHED_GRIP_COMMAND_CHANGED")
        target = lower+(grips[("left", "right").index(arm)]+1)*.5*(upper-lower)
        if np.max(np.abs(values-target)) > FINGER_SPAN_M:
            raise ValueError("PRESS_CURRENT_TOOL_NOT_AT_EXISTING_COMMAND")
        if self.finger_reference is not None and np.max(np.abs(values-self.finger_reference)) > FINGER_SPAN_M:
            raise ValueError("PRESS_BOUND_TOOL_DRIFTED")
        return values, identity

    def begin_observation(self, state, control):
        if not self.active:
            return
        if type(control) is not int or control < 0:
            raise ValueError("Press observations require an actual integer control stamp")
        key = self.goal_key()
        if self.key != key:
            if self.phase not in ("IDLE", "FINISHED"):
                self.stop("PRESS_GOAL_OR_TARGET_REFERENCE_CHANGED")
                return
            self.__init__(self.c)
            self.key = key
            self.transition("PREPARE")
        if self.observation_control is not None and control < self.observation_control:
            self.stop("PRESS_OBSERVATION_CLOCK_REVERSED")
            return
        self.observation_control = control
        self.observation_q = state.q.copy()
        self.selected = None
        if self.running is not None:
            self.stop("PRESS_EXECUTION_NOT_FINISHED")
            return
        try:
            _, self.calibration_id = self._values(state)
        except ValueError as error:
            self.stop(str(error))
        if self.grips is None:
            self.grips = self.c.servo.grips.copy()

    def after_observation(self, evidence, state):
        """Called after the ordinary hazard / lost-load / failed-servo gates."""
        if not self.active:
            return False
        if self.manager.stop_reason:
            return self.stop("PRESS_STOPPED_BY_"+self.manager.stop_reason)
        if self.manager.stage == "RECOVER":
            return self.stop("PRESS_ORDINARY_RECOVERY_REQUIRED")
        if self.phase == "PREPARE":
            return False
        target, ref = self.c.target, self.c.press_reference
        if not target.get("valid") or ref is None:
            if self.phase != "APPROACH":
                self.stop("PRESS_FRESH_TARGET_UNAVAILABLE")
            return False
        rid = ref.record()["reference_id"]
        if self.reference_id is not None and rid != self.reference_id:
            return self.stop("PRESS_BOUND_SURFACE_CHANGED")
        self.reference_id = rid
        if self.phase == "APPROACH" and self.c._distance() <= NEAR_M:
            self.anchor = np.asarray(target["point_base_m"], dtype=float).copy()
            self.transition("ADVANCE")
        if self.anchor is not None and np.linalg.norm(np.asarray(target["point_base_m"])-self.anchor) > .01:
            return self.stop("PRESS_NEAR_TARGET_CHANGED")
        if self.phase == "ADVANCE" and self.c._distance() <= DWELL_M:
            point = np.asarray(target["press_reference"]["point_base_m"])
            direction = point-np.asarray(target["point_base_m"])
            length = np.linalg.norm(direction)
            if not np.isfinite(length) or length <= 1e-8:
                return self.stop("PRESS_RETREAT_DIRECTION_UNDEFINED")
            self.away = direction/length
            self.transition("DWELL")
        if self.phase == "RETRACT" and self.effect_after_dwell is None:
            self.effect_after_dwell = {"control": self.observation_control,
                "effect_claim": evidence.effect, "not_completion": True}
        if self.phase == "VERIFY":
            stamp = self.observation_control
            if (self.retract_end is None or stamp < self.retract_end or
                    (self.last_verify_control is not None and stamp <= self.last_verify_control)):
                return False  # Re-observing a frozen frame cannot add confirmation.
            self.last_verify_control = stamp
            self.verify_views += 1
            valid = evidence.visible and evidence.hazard == "none" and evidence.effect is True
            self.confirmations = self.confirmations+1 if valid else 0
            self.verified_claims.append({"control": stamp, "evidence": asdict(evidence), "accepted_claim": valid})
            if self.confirmations >= 2:
                self.transition("FINISHED")
                return True
            if self.verify_views >= MAX_VERIFY_VIEWS:
                self.stop("PRESS_EFFECT_UNVERIFIED")
        return False

    def candidates(self, state, deadline=None):
        """Finite original primitives; no invented target poses or hand aperture."""
        require_time(deadline)
        if not self.owns_selection:
            raise ValueError("The VLM still owns ordinary far-field selection")
        if self.phase == "STOPPED":
            return (HOLD,)
        if self.phase in ("PREPARE", "DWELL", "VERIFY"):
            actions = [HOLD]
        else:
            limit = MAX_ADVANCES if self.phase == "ADVANCE" else MAX_RETRACTIONS
            used = self.advances if self.phase == "ADVANCE" else self.retractions
            if used >= limit:
                self.stop("PRESS_"+self.phase+"_BUDGET_EXHAUSTED")
                return (HOLD,)
            actions = [Action(self.manager.goal.hand, move, "micro") for move in TRANSLATIONS]
        tested, accepted = [], []
        for action in actions:
            require_time(deadline)
            ok, reason = self.c.depth_guard.check(action, self.manager.carry)
            trial = SafeServo(self.c.model, state, self.c.servo.grips.copy(), self.c.servo.limits)
            if ok:
                ok = trial.begin(action, state, self.manager.carry)
                reason = "ORIGINAL_SERVO_CHECKED" if ok else trial.status
            distance = None
            if ok and action != HOLD:
                check = self.c.press_trial_check(state, action, trial, deadline)
                ok, reason = check["eligible"], check["reason"]
                distance = self.c._expected_point(action, state, trial) if ok else None
                gain = self.c._distance()-distance if distance is not None else 0.
                if self.phase == "ADVANCE" and gain < .0002:
                    ok, reason = False, "PRESS_NO_MEASURED_APPROACH_OPTION"
                if self.phase == "RETRACT":
                    direction = np.asarray(TRANSLATIONS[action.move])
                    if direction @ self.away < .5 or gain > -.0002:
                        ok, reason = False, "PRESS_NOT_AN_AWAY_OPTION"
            tested.append({"action": asdict(action), "accepted": bool(ok), "reason": reason,
                           "predicted_distance_m": distance})
            if ok:
                score = (distance if self.phase == "ADVANCE" else -distance) if distance is not None else 0.
                accepted.append((score, action))
        require_time(deadline)
        self.manager.candidate_receipt = {"source": VERSION, "phase": self.phase, "tested": tested}
        if not accepted:
            self.stop("PRESS_NO_CHECKED_"+self.phase+"_ACTION")
            return (HOLD,)
        # Safety HOLD is available as an opt-out, but only this exact internal
        # selection gets a purpose receipt. No VLM-supplied purpose is accepted.
        action = min(accepted, key=lambda item: item[0])[1]
        self.selected = (self.phase, action, self.observation_control)
        return (action,)

    def begin_execution(self, action, state, control):
        if not self.active:
            return True
        if self.key != self.goal_key() or self.manager.stop_reason:
            return self.stop("PRESS_EXECUTION_CONTEXT_CHANGED")
        if control != self.observation_control or self.running is not None:
            return self.stop("PRESS_EXECUTION_CLOCK_CHANGED")
        if np.max(np.abs(state.q-self.observation_q)) > 1e-5:
            return self.stop("PRESS_OBSERVATION_ROBOT_CHANGED")
        purpose = None
        if self.owns_selection:
            if self.selected != (self.phase, action, control):
                return self.stop("PRESS_PURPOSE_NOT_SELECTED_BY_EXECUTOR")
            purpose = self.phase
        elif self.phase != "APPROACH":
            return self.stop("PRESS_NO_ACTIVE_EXECUTION_PHASE")
        try:
            values, _ = self._values(state)
        except ValueError as error:
            return self.stop(str(error))
        self.running = {"purpose": purpose, "phase": self.phase, "action": action,
            "start_control": control, "last_control": control, "samples": deque(maxlen=STABLE_TICKS),
            "start_values": values.copy(), "max_finger_delta_m": 0., "failure": None,
            "reference_id": self.reference_id}
        self.selected = None
        return True

    def post_step(self, state, control, command):
        """Exactly once AFTER each actual environment control, even its last."""
        if not self.active:
            return True
        r = self.running
        if r is None or type(control) is not int or control != r["last_control"]+1:
            return self.stop("PRESS_MISSING_OR_NONCONSECUTIVE_CONTROL_SAMPLE")
        r["last_control"] = control
        try:
            if self.goal_key() != self.key:
                raise ValueError("PRESS_GOAL_CHANGED_DURING_EXECUTION")
            command = np.asarray(command)
            if (command.shape != (23,) or not np.isfinite(command).all() or
                    not np.allclose(command[[14, 22]], self.grips, atol=1e-7, rtol=0)):
                raise ValueError("PRESS_ISSUED_GRIP_COMMAND_CHANGED")
            values, _ = self._values(state)
            if r["phase"] in ("ADVANCE", "DWELL", "RETRACT", "VERIFY"):
                if self.c.press_reference is None or self.c.press_reference.record()["reference_id"] != r["reference_id"]:
                    raise ValueError("PRESS_SURFACE_CHANGED_DURING_EXECUTION")
                try:
                    self.c.press_asset.resolve(self.c.press_reference, self.c.model, state.q, state.finger_qpos,
                                               target_base=self.c.target["point_base_m"])
                except (ValueError, KeyError) as error:
                    raise ValueError("PRESS_EXECUTED_FACE_OR_CHANNEL_INVALID") from error
            r["samples"].append(values.copy())
            r["max_finger_delta_m"] = max(r["max_finger_delta_m"], float(np.max(np.abs(values-r["start_values"]))))
        except ValueError as error:
            r["failure"] = str(error)
            return self.stop(str(error))
        return True

    def finish_execution(self, action, feedback, state, control, *, interrupted=False, motion_receipt=None):
        if not self.active:
            return feedback
        r = self.running
        if r is None:
            self.stop("PRESS_NO_EXECUTION_CHAIN")
            return {**feedback, "press_execution": {"version": VERSION, "command_complete": False,
                                                     "reason": self.stop_reason, "success_claim": False}}
        self.running = None
        actual = control-r["start_control"]
        measured = None
        try:
            checked_action_motion(motion_receipt, r["start_control"], control)
            measured = observed_motion_feedback(action, feedback, motion_receipt)
        except (ValueError, KeyError, TypeError) as error:
            r["failure"] = r["failure"] or str(error)
        complete = bool(not interrupted and not self.manager.stop_reason and not r["failure"]
                        and r["action"] == action and self.key == self.goal_key()
                        and control == r["last_control"] and type(feedback.get("control_ticks")) is int
                        and actual == feedback["control_ticks"] == self.c.servo.total_ticks
                        and actual > 0 and measured is not None and execution_completed(action, measured))
        if r["purpose"] in ("ADVANCE", "DWELL", "RETRACT", "VERIFY"):
            base = np.asarray((measured or {}).get("base_integral", []), dtype=float)
            if (base.shape != (3,) or not np.isfinite(base).all()
                    or np.linalg.norm(base[:2]) > .001 or abs(base[2]) > .003):
                complete = False
                r["failure"] = r["failure"] or "PRESS_BODY_MOVED_DURING_NEAR_INTERACTION"
        samples = np.asarray(r["samples"])
        stable = bool(len(samples) == STABLE_TICKS and
                      np.max(np.ptp(samples, axis=0)) <= FINGER_SPAN_M and
                      np.max(np.abs(np.diff(samples, axis=0))) <= FINGER_STEP_M)
        receipt = {"version": VERSION, "goal_key": list(self.key), "phase": r["phase"],
            "purpose": r["purpose"], "action": asdict(action), "reference_id": r["reference_id"],
            "control_start": r["start_control"], "control_end": control,
            "planned_control_ticks": self.c.servo.total_ticks, "executed_control_ticks": actual,
            "sampled_through_control": r["last_control"], "last_finger_samples_m": samples.tolist(),
            "max_finger_delta_m": r["max_finger_delta_m"], "stable_last_six": stable,
            "command_complete": complete, "interrupted": bool(interrupted), "failure": r["failure"],
            "base_motion_source": None if measured is None else measured["base_motion_source"],
            "measured_body_delta": None if measured is None else measured["base_integral"],
            "physical_contact_evidence": False, "success_claim": False}
        self.last_receipt = receipt
        if not complete:
            self.stop(r["failure"] or "PRESS_EXECUTION_INCOMPLETE")
        elif r["purpose"] == "PREPARE":
            if not stable:
                self.stop("PRESS_TOOL_UNSTABLE")
            else:
                self.finger_reference = samples[-1].copy()
                self.transition("APPROACH")
        elif r["purpose"] == "ADVANCE":
            self.advances += 1
        elif r["purpose"] == "DWELL":
            self.dwell_end = control
            row = self.c.press_asset.resolve(self.c.press_reference, self.c.model, state.q, state.finger_qpos)
            self.retreat_origin = np.asarray(row["point_base_m"])
            self.transition("RETRACT")
        elif r["purpose"] == "RETRACT":
            self.retractions += 1
            row = self.c.press_asset.resolve(self.c.press_reference, self.c.model, state.q, state.finger_qpos)
            retreat = float((np.asarray(row["point_base_m"])-self.retreat_origin) @ self.away)
            receipt["measured_away_displacement_m"] = retreat
            if retreat >= RETRACT_M:
                self.retract_end = control
                self.transition("VERIFY")
        return {**feedback, "press_execution": copy.deepcopy(receipt)}

    def context(self):
        return {"version": VERSION, "phase": self.phase, "goal_key": list(self.key) if self.key else None,
                "tool_ready": self.tool_ready, "reference_id": self.reference_id,
                "advances": self.advances, "retractions": self.retractions,
                "dwell_control_end": self.dwell_end, "retract_control_end": self.retract_end,
                "effect_after_dwell": self.effect_after_dwell, "verification_views": self.verify_views,
                "consecutive_effect_claims": self.confirmations, "effect_claims": copy.deepcopy(self.verified_claims),
                "last_execution": copy.deepcopy(self.last_receipt), "stop_reason": self.stop_reason,
                "physical_contact_evidence": False, "official_success_claim": False}
