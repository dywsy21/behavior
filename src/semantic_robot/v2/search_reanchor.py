"""Bounded new LOCAL reference after lost odometry, never relocalization.

Only an unseen world-target search with two calibrated-open, never-closed
hands is eligible. The failed displacement stays unknown. Twelve real HOLD
controls must pass the unchanged joint RGB-D estimator before resuming.
"""
import copy
import math

import numpy as np

from .odometry import RGBDMotion
from .substep_odometry import SubstepMotion, checked_transform
from .wall_budget import expired


def calibrated_open(controller, state):
    meta = controller.model.spec.get("metadata", {})
    opening = np.asarray(meta.get("grasp_region_reference_gripper_m", []), dtype=float)
    known = meta.get("grasp_region_reference_fully_open", {})
    width, latch = np.asarray(state.gripper), np.asarray(controller.servo.grips)
    return bool(opening.shape == width.shape == latch.shape == (2,) and
                np.isfinite([opening, width, latch]).all() and
                all(known.get(a) is True for a in ("left", "right")) and
                np.all(abs(width-opening) <= .0005) and np.all((latch >= .999) & (latch <= 1.)))


class SearchReanchor:
    max_attempts = 2
    hold_controls = 12

    def __init__(self):
        self.attempts = 0
        self.ever_closed = False
        self.history = []

    def checks(self, c, state, failed, terminal):
        h = c.harness
        action = h.last_action
        segments = failed.get("segments", [])
        start, end = failed.get("control_start"), failed.get("control_end")
        span = (type(start) is int and type(end) is int and 0 <= start < end <= start+24)
        arms={"left","right"}
        flags=(h.pending_grasp,h.hold_verified,h.possible_contact_after_close)
        known_unloaded=all(isinstance(row,dict) and set(row)==arms and
                           all(value is False for value in row.values()) for row in flags)
        known_unheld=(isinstance(h.held,dict) and set(h.held)==arms and
                      all(value is None for value in h.held.values()))
        return {
            "attempt_budget": type(self.attempts) is int and 0 <= self.attempts < self.max_attempts,
            "only_visual_quality_stop": h.stop_reason == "VISUAL_ODOMETRY_UNCERTAIN",
            "not_terminal": terminal is False,
            "unseen_world_search": bool(h.stage == "SEARCH" and h.search_reference == "world" and
                h.observation and h.observation.visible is False and h.observation.hazard == "none" and
                h.goal.kind in ("navigate", "pick")),
            "base_search_not_reposition": bool(action and action.part == "base" and
                action.move in ("yaw_plus", "yaw_minus") and c.reposition_left == 0 and c.pending_exploratory),
            "no_load_or_contact_history": bool(self.ever_closed is False and known_unloaded and
                known_unheld and not h.completed),
            "calibrated_open_latches_and_apertures": calibrated_open(c, state),
            "no_held_or_blocked_anchors": bool((c.inspector is None or not c.inspector.anchors) and
                (c.approach_monitor is None or not c.approach_monitor.blocked)),
            "failed_complete_record": bool(failed.get("valid") is False and
                failed.get("reason") == "SUBSTEP_RGBD_QUALITY_FAILED_NO_PARTIAL_CREDIT" and span and
                segments and segments[-1].get("measurement", {}).get("valid") is False and
                isinstance(c.motion, SubstepMotion) and c.motion.failed is True and
                c.motion.active is False and c.motion.pending is not None and
                c.motion.pending.get("control_start")==start and c.motion.pending.get("control_end")==end and
                c.motion.segments==segments and c.pending_motion is not None),
        }

    def attempt(self, c, failed, *, controls, action_limit, terminal, deadline,
                observe, state_now, issue_hold, save_snapshot, motion_factory=None):
        """Callbacks own recorded controls/captures; no simulator state access.

        issue_hold returns terminal and MUST issue exactly one ordinary control.
        save_snapshot receives all three raw RGB-D views, current q and clock.
        A false receipt never clears the original stop or replaces the odometer.
        """
        state = state_now()
        checks = self.checks(c, state, failed, terminal)
        checks["failed_endpoint_matches_clock"] = type(controls) is int and failed.get("control_end")==controls
        checks["whole_probe_fits_control_budget"] = controls+self.hold_controls <= action_limit
        checks["time_remaining"] = not expired(deadline)
        receipt = {"valid": False, "source": "bounded_new_local_search_reference", "checks": checks,
                   "failed_motion_not_recovered": True, "success_claim": False,
                   "control_start": controls, "control_end": controls, "attempt": self.attempts}
        if not all(checks.values()):
            receipt["reason"] = "SEARCH_REANCHOR_INELIGIBLE"
            return receipt, None
        self.attempts += 1
        receipt["attempt"] = self.attempts
        self.history.append(receipt)
        use_self = c.odometry_self_exclusion
        factory = motion_factory or (lambda: SubstepMotion(RGBDMotion("rgbd_joint", exclude_robot=True)
                                                            if use_self else RGBDMotion("rgbd_joint")))
        motion = factory()
        images, depths, sensor = observe("search_reanchor_before")
        state = state_now()
        robot_frame = save_snapshot(controls, images, depths, sensor, state)
        kwargs = dict(robot_frame=robot_frame, gripper=state.gripper, control=controls) if use_self else {}
        initial = motion.observe(images, depths, c.model, state.q, **kwargs)
        if initial.get("valid") is not True or initial.get("initial") is not True:
            receipt["reason"] = "NEW_REFERENCE_INITIALIZATION_FAILED"
            return receipt, None
        motion.begin(controls)
        for offset in range(1, self.hold_controls+1):
            if expired(deadline) or not calibrated_open(c, state_now()):
                receipt["reason"] = "PROBE_DEADLINE_OR_GRIP_CHANGE"
                return receipt, None
            ended = issue_hold()
            receipt["control_end"] = controls+offset
            if ended:
                receipt["reason"] = "OFFICIAL_EPISODE_TERMINATED_DURING_PROBE"
                return receipt, None
            if offset % 6 == 0:
                images, depths, sensor = observe("search_reanchor_probe")
                state = state_now()
                robot_frame = save_snapshot(controls+offset, images, depths, sensor, state)
                kwargs = dict(robot_frame=robot_frame, gripper=state.gripper) if use_self else {}
                measured = motion.sample(images, depths, c.model, state.q, controls+offset, **kwargs)
                if not measured["valid"]:
                    receipt["chain"] = motion.finish(controls+offset, interrupted=True)
                    receipt["reason"] = "HOLD_RGBD_QUALITY_FAILED_NO_RETRY"
                    return receipt, None
        chain = motion.finish(controls+self.hold_controls)
        receipt["chain"] = chain
        if expired(deadline):
            receipt["reason"] = "PROBE_DEADLINE_AFTER_CAPTURE"
            return receipt, None
        if not chain["valid"]:
            receipt["reason"] = "HOLD_CHAIN_FAILED"
            return receipt, None
        T = checked_transform(chain)
        angle=math.acos(float(np.clip((np.trace(T[:3,:3])-1)/2,-1.,1.)))
        receipt["hold_total_rotation_rad"]=angle
        if (np.linalg.norm(T[:2, 3]) > .01 or abs(math.atan2(T[1, 0], T[0, 0])) > .02 or
                angle > .02 or abs(T[2, 3]) > .005 or not calibrated_open(c, state)):
            receipt["reason"] = "HOLD_NOT_OBSERVED_AT_REST"
            return receipt, None
        receipt.update(valid=True, reason="NEW_LOCAL_REFERENCE_AFTER_MEASURED_HOLD_NOT_OLD_POSE_RECOVERY",
                       unknown_failed_span=[failed["control_start"], failed["control_end"]],
                       probe_is_not_goal_or_motion_progress=True)
        # No credit for either the failed action or this sensing HOLD. A fresh
        # odometer starts only at the final snapshot; old/pending chain is retired.
        c.search.new_local_epoch(failed["control_start"], failed["control_end"])
        c.motion = factory()
        c.pending_motion = None
        c.previous = None
        c.target = {"valid": False, "reason": "NEW_REFERENCE_REQUIRES_FRESH_SEMANTIC_OBSERVATION"}
        c.hand_targets = {}; c.centers = {}; c.depth_guard = None; c.progress = {}
        c.reposition = None; c.reposition_left = 0; c.replan_needed = None
        if c.approach_monitor is not None:
            c.approach_monitor = type(c.approach_monitor)()
        if c.grasp_verifier is not None:
            old = c.grasp_verifier
            c.grasp_verifier = type(old)(persistent_tracks=old.persistent_tracks,
                                         spatial_seed_features=old.spatial_seed_features)
        h = c.harness
        h.observation = h.previous_observation = None
        h.feedback = None
        h.grounding = copy.deepcopy(c.target)
        h.candidate_receipt = {}; h.motion_receipt = {}
        h.last_distance = None; h.confirmations = 0
        h.search_context = c.search.context()
        h.search_reanchor = self.context()
        h.stop_reason = None
        snapshot = (controls+self.hold_controls, state.q.copy(), state.gripper.copy(), images, depths, sensor)
        if use_self:
            snapshot += (robot_frame,)
        return receipt, snapshot

    def context(self):
        return {"attempts": self.attempts, "maximum": self.max_attempts, "ever_closed": self.ever_closed,
                "history": [{k: v for k, v in row.items() if k != "chain"} for row in self.history],
                "unknown_motion_never_credited": True}
