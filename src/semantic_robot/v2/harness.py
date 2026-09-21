"""Explicit subgoals, evidence gates, finite recovery and scoped action palettes.

VLM claims remain claims. Width alone cannot certify holding; self-reported DONE
cannot certify official success. No privileged evaluator data is accepted here.
"""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .protocol import Action, Evidence, HOLD, ROTATIONS, TRANSLATIONS, strict_json
from .servo import execution_completed


@dataclass(frozen=True)
class Goal:
    kind: str
    target: str
    hand: str
    done_when: str
    level: bool = False

    def __post_init__(self):
        if self.kind not in ("pick", "place", "press", "open", "close", "navigate"):
            raise ValueError("Unknown goal kind")
        if self.hand not in ("left", "right", "both") or type(self.level) is not bool:
            raise ValueError("Invalid arm or carry-level flag")
        for value in (self.target, self.done_when):
            if not isinstance(value, str) or not 1 <= len(value) <= 300:
                raise ValueError("Target and visible completion criterion required")


def parse_plan(text):
    values = strict_json(text)
    if not isinstance(values, list) or not 1 <= len(values) <= 16:
        raise ValueError("Plan must contain 1–16 bounded subgoals")
    result = []
    for value in values:
        if not isinstance(value, dict) or set(value) != set(Goal.__dataclass_fields__):
            raise ValueError("Exact goal fields required")
        result.append(Goal(**value))
    occupied = set()
    for goal in result:
        arms = {"left", "right"} if goal.hand == "both" else {goal.hand}
        if goal.kind == "pick":
            if occupied & arms:
                raise ValueError("Plan attempts to pick with an already occupied hand")
            occupied |= arms
        elif goal.kind == "place":
            if not arms <= occupied:
                raise ValueError("Plan places with a hand that has not picked an object")
            occupied -= arms
    return result


class TaskHarness:
    def __init__(self, goals, max_recoveries=3):
        if not goals:
            raise ValueError("A plan is required")
        self.goals = list(goals)
        self.index, self.stage = 0, "SEARCH"
        self.events, self.completed = [], []
        self.held = {"left": None, "right": None}
        self.hold_verified = {"left": False, "right": False}
        self.level = {"left": False, "right": False}
        self.recoveries, self.max_recoveries = 0, max_recoveries
        self.history = deque(maxlen=8)
        self.feedback, self.observation = None, None
        self.previous_observation, self.last_action = None, None
        self.stop_reason = None
        self.stage_age, self.confirmations, self.no_progress = 0, 0, 0
        self.last_distance = None
        self.executions, self.recovery_entered_after = 0, 0

    @property
    def goal(self):
        return self.goals[min(self.index, len(self.goals)-1)]

    @property
    def carry(self):
        return any(self.level[k] and self.hold_verified[k] for k in self.held)

    @property
    def arms(self):
        return ("left", "right") if self.goal.hand == "both" else (self.goal.hand,)

    def transition(self, stage):
        if self.stage != stage:
            self.events.append({"event": "STAGE_CHANGED", "from": self.stage, "to": stage, "goal": self.index})
            self.stage, self.stage_age, self.confirmations = stage, 0, 0
            self.no_progress, self.last_distance = 0, None

    def recover(self, event):
        self.events.append({"event": event, "goal": self.index, "stage": self.stage})
        self.recoveries += 1
        self.recovery_entered_after = self.executions
        if self.recoveries > self.max_recoveries:
            self.stop_reason = "RECOVERY_BUDGET_EXHAUSTED"
            return
        # Never open a potentially held object automatically. The recovery palette
        # requires re-observation; only unverified/empty pick grippers may reopen.
        self.transition("RECOVER")
        # A new attempt gets its own observation window even if we were already
        # recovering. transition() deliberately does nothing for the same stage;
        # retaining its old age would spend all remaining attempts immediately.
        self.stage_age, self.confirmations, self.no_progress = 0, 0, 0
        self.last_distance = None

    def _complete_goal(self):
        g = self.goal
        if g.kind == "pick":
            for arm in self.arms:
                self.held[arm], self.hold_verified[arm], self.level[arm] = g.target, True, g.level
        if g.kind == "place":
            for arm in self.arms:
                self.held[arm], self.hold_verified[arm], self.level[arm] = None, False, False
        self.completed.append({"goal": asdict(g), "evidence": self.observation.as_dict(),
                               "status": "OBSERVATION_VERIFIED_NOT_OFFICIAL_TRUTH"})
        self.index += 1
        if self.index == len(self.goals):
            self.stop_reason = "PLAN_EXHAUSTED_NOT_OFFICIAL_SUCCESS"
        else:
            self.transition("SEARCH")

    def observe(self, evidence: Evidence, state, geometry=None, measured_progress=None):
        self.previous_observation, self.observation = self.observation, evidence
        self.events = []
        if self.stop_reason:
            return
        self.stage_age += 1
        # Only the grounded adapter supplies these sensor-derived measurements;
        # they are not VLM JSON fields and never certify task completion.
        measured_progress = measured_progress or {}
        navigation_ready=(measured_progress.get("navigation_aligned",True) and
                          measured_progress.get("navigation_reach_possible",True))
        if self.stage in ("SEARCH","RECOVER") and measured_progress.get("new_search_coverage") is True:
            self.stage_age = 0
        if evidence.hazard in ("slip", "collision"):
            # Hold first. Repeated danger consumes finite recovery; no auto-release.
            self.recover("VISUAL_"+evidence.hazard.upper()+"_SUSPECTED")
            return
        if self.feedback and not execution_completed(self.last_action, self.feedback):
            self.recover(self.feedback["status"])
            self.feedback = None  # event consumed once, not on every subsequent view
            return
        empty = any(state.gripper[("left", "right").index(arm)] < .0015 for arm in self.arms)
        if self.goal.kind == "pick" and self.stage == "VERIFY_GRASP" and empty:
            self.recover("EMPTY_GRASP_SUSPECTED")
            return
        if any(self.hold_verified.values()):
            lost = [arm for i, arm in enumerate(("left", "right")) if self.hold_verified[arm] and state.gripper[i] < .0015]
            if lost and not (self.stage == "VERIFY_PLACE" and self.goal.kind == "place"):
                for arm in lost:
                    self.hold_verified[arm] = False
                self.recover("PREVIOUS_HOLD_LOST_OR_UNCERTAIN")
                return
        if self.stage == "RECOVER":
            recovered_action = (self.executions > self.recovery_entered_after and self.feedback
                                and execution_completed(self.last_action, self.feedback))
            if recovered_action and self.last_action.move == "open" and self.goal.kind == "pick":
                self.transition("ALIGN" if evidence.visible else "SEARCH")
            elif recovered_action and evidence.visible and evidence.hazard == "none" and self.last_action.move not in ("hold", "close"):
                self.transition("APPROACH")
            elif self.stage_age >= 4:
                self.recover("RECOVERY_NO_PROGRESS")
            return
        if not evidence.visible:
            self.events.append({"event": "TARGET_NOT_VISIBLE", "view": evidence.view})
            if self.stage not in ("VERIFY_GRASP", "VERIFY_PLACE"):
                self.transition("SEARCH")
            if self.stage_age >= 8:
                self.recover("SEARCH_NO_PROGRESS")
            return
        if self.stage == "SEARCH":
            self.transition("APPROACH")
        elif self.stage == "APPROACH":
            wrist = evidence.view == self.goal.hand+"_wrist" or (self.goal.hand == "both" and "wrist" in evidence.view)
            if self.goal.kind == "navigate":
                if (evidence.effect is True and navigation_ready
                        and self.last_action and self.last_action.move != "hold"):
                    self.transition("VERIFY_EFFECT")
                    self.confirmations = 1
            elif (measured_progress.get("target_distance_m",float("inf")) <= .08
                  if "target_distance_m" in measured_progress else wrist or evidence.enclosed is True):
                self.transition("ALIGN")
        elif self.stage == "ALIGN":
            if (self.goal.kind == "pick" and evidence.enclosed is True and
                    measured_progress.get("target_distance_m",0.) <= .06):
                self.transition("GRASP")
            elif self.goal.kind == "place" and evidence.supported is True:
                self.transition("VERIFY_SUPPORT")
                self.confirmations = 1
            elif self.goal.kind not in ("pick", "place"):
                self.transition("INTERACT")
        elif self.stage == "VERIFY_GRASP":
            moved = self.feedback and any(np.linalg.norm(self.feedback["eef_delta_m"][arm]) >= .0015 for arm in self.arms)
            # Two sources: nonempty physical opening + visual enclosure and temporal
            # co-motion after an actual verification displacement. Neither alone wins.
            temporal = self.previous_observation is not None and self.previous_observation.visible
            legacy=(evidence.enclosed is True and evidence.co_moving is True
                    and measured_progress.get("metric_co_motion",True) is True)
            registration=measured_progress.get("registered_grasp_motion",{})
            registered=(registration.get("verified") is True and
                        registration.get("source")=="registered_onboard_RGBD_grasp_motion" and
                        evidence.enclosed is not False and evidence.co_moving is not False)
            if not empty and moved and temporal and (legacy or registered):
                self._complete_goal()
                if registered:self.completed[-1]["registered_motion_evidence"]=registration
            elif self.stage_age >= 4:
                self.recover("GRASP_UNVERIFIED")
        elif self.stage == "VERIFY_PLACE":
            opened = all(state.gripper[("left", "right").index(arm)] > .03 for arm in self.arms)
            self.confirmations = self.confirmations+1 if opened and evidence.supported is True and evidence.enclosed is not True else 0
            if self.confirmations >= 2:
                self._complete_goal()
            elif self.stage_age >= 5:
                self.recover("PLACEMENT_UNVERIFIED")
        elif self.stage == "VERIFY_SUPPORT":
            self.confirmations = self.confirmations+1 if evidence.supported is True else 0
            if self.confirmations >= 2:
                self.transition("RELEASE")
            elif evidence.supported is not True:
                self.transition("ALIGN")
        elif self.stage == "INTERACT":
            if evidence.effect is True and self.last_action and self.last_action.move != "hold":
                self.transition("VERIFY_EFFECT")
                self.confirmations = 1
        elif self.stage == "VERIFY_EFFECT":
            effect=evidence.effect is True and (self.goal.kind!="navigate" or navigation_ready)
            self.confirmations = self.confirmations+1 if effect else 0
            if self.confirmations >= 2:
                self._complete_goal()
            elif not effect:
                self.transition("APPROACH" if self.goal.kind == "navigate" else "INTERACT")
        # Image progress is relative target-to-hand distance, not merely pixels changing.
        if geometry and evidence.target_uv and self.goal.hand in ("left", "right"):
            uv = geometry.get(evidence.view, {}).get(self.goal.hand, {}).get("eef_uv")
            if uv is not None:
                distance = float(np.linalg.norm(np.asarray(uv)-evidence.target_uv))
                if self.last_distance is not None:
                    self.no_progress = 0 if distance < self.last_distance-.008 else self.no_progress+1
                self.last_distance = distance
        if self.stage_age >= 12 or self.no_progress >= 6:
            self.recover("NO_TASK_PROGRESS")

    def palette(self):
        if self.stop_reason:
            return (HOLD,)
        g, stage, obs = self.goal, self.stage, self.observation
        actions = [HOLD]
        def add(part, moves, scales=("fine",), frames=("base",)):
            for move in moves:
                for scale in scales:
                    for frame in frames:
                        actions.append(Action(part, move, scale, frame))
        if stage in ("SEARCH", "APPROACH", "RECOVER"):
            add("base", ("forward", "back", "left", "right", "yaw_plus", "yaw_minus"), ("micro", "fine"))
            add("torso", ("up", "down", "forward", "back"), ("micro", "fine"))
        if stage == "SEARCH":
            return tuple(actions)  # do not manipulate a target that has not been located
        if stage == "GRASP":
            if obs and obs.visible and obs.enclosed is True:
                add(g.hand, ("close",))
            return tuple(actions)
        if stage == "RELEASE":
            if obs and obs.supported is True:
                add(g.hand, ("open",))
            return tuple(actions)
        if stage == "VERIFY_GRASP":
            add(g.hand, ("up",), ("micro", "fine"))
            return tuple(actions)
        if stage in ("VERIFY_PLACE", "VERIFY_SUPPORT", "VERIFY_EFFECT"):
            return tuple(actions)
        if stage == "RECOVER":
            if g.kind == "pick" and not any(self.hold_verified[arm] for arm in self.arms):
                add(g.hand, ("open",))
            add(g.hand, ("back", "up"), ("micro",))
            return tuple(actions)
        if g.kind == "navigate":
            return tuple(actions)
        frames = ("base",) if g.hand == "both" else ("base", obs.view) if obs and obs.visible else ("base",)
        # Near a workspace boundary, 1 cm may be unreachable while 2 mm is
        # feasible. Approach must not hide the only feasible smaller step.
        scales = ("micro", "fine") if stage != "APPROACH" else ("micro", "fine", "coarse")
        add(g.hand, TRANSLATIONS, scales, frames)
        if not self.carry and stage in ("ALIGN", "INTERACT"):
            add(g.hand, ROTATIONS, ("micro", "fine"), ("tool",) if g.hand != "both" else ("base",))
        if g.kind in ("open", "close") and stage == "INTERACT":
            add(g.hand, ("open", "close"))
        if g.kind == "pick" and stage in ("APPROACH", "ALIGN"):
            add(g.hand, ("open",))
        return tuple(dict.fromkeys(actions))

    def authorize(self, action):
        if action not in self.palette():
            raise ValueError("Action outside current stage/arm/evidence palette")

    def executed(self, action, feedback):
        self.executions += 1
        self.last_action, self.feedback = action, feedback
        self.history.append({"action": asdict(action), "feedback": feedback, "stage": self.stage})
        if execution_completed(action, feedback):
            if self.stage == "GRASP" and action.move == "close":
                self.transition("VERIFY_GRASP")
            elif self.stage == "RELEASE" and action.move == "open":
                self.transition("VERIFY_PLACE")
        if len(self.history) >= 4:
            last = list(self.history)[-4:]
            commands = [r["action"] for r in last]
            a, b, c, d = commands
            # There is no "both" key in physical per-hand feedback. Using that
            # missing key silently labelled every bimanual ABAB as zero motion.
            drift = max(np.linalg.norm(sum((np.asarray(r["feedback"].get("eef_delta_m", {}).get(arm, [0,0,0]))
                        for r in last), np.zeros(3))) for arm in self.arms)
            # Repetition is legitimate while approaching. Detect near-zero effect or
            # cancelling ABAB only; never globally penalize valid repeated moves.
            noop = a == b == c == d and all(r["action"]["move"] in ("hold", "close", "open") for r in last)
            cancel = a == c and b == d and a != b and all(x["part"] in ("left", "right", "both") for x in commands) and drift < .004
            if noop or cancel:
                self.recover("REPEATED_NOOP" if noop else "OSCILLATION")

    def context(self):
        return {"goal_index": self.index, "goal": asdict(self.goal), "stage": self.stage,
                "completed_observation_claims": self.completed, "held_target_claims": self.held,
                "holding_verified_by_observation_and_proprio": self.hold_verified,
                "carry_constraints": self.carry, "events": self.events,
                "recent_executed": list(self.history)[-5:], "recoveries": self.recoveries,
                "stop_reason": self.stop_reason}
