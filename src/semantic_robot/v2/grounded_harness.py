"""Sensor-grounded finite task/search/recovery orchestration, not task scripts."""
import math
import time
from dataclasses import asdict

import numpy as np

from .grounding import LocalDepthGuard, localize_target, observed_cloud
from .harness import TaskHarness
from .protocol import Action, HOLD, TRANSLATIONS, strict_json
from .search import CoverageSearch
from .servo import SafeServo


RECOVERY_STRATEGIES=("scan_left","scan_right","move_forward","move_left","move_right","retry_approach","hold")


def parse_recovery(text):
    value=strict_json(text)
    if not isinstance(value,dict) or set(value)!={"strategy","visible_reason"}:
        raise ValueError("Recovery changes strategy only, not goals or completion")
    if value["strategy"] not in RECOVERY_STRATEGIES or not isinstance(value["visible_reason"],str) or not 1<=len(value["visible_reason"])<=240:
        raise ValueError("Invalid bounded recovery strategy")
    return value


class GroundedHarness(TaskHarness):
    def __init__(self, goals):
        super().__init__(goals)
        self.grounding={}
        self.search_context={}
        self.replans=0
        self.replan_history=[]
        self.candidate_receipt={}

    def palette(self):
        palette=super().palette()
        if not self.stop_reason and self.stage in ("SEARCH","RECOVER"):
            palette += (Action("base","yaw_plus","coarse"),Action("base","yaw_minus","coarse"))
        return tuple(dict.fromkeys(palette))

    def context(self):
        context=super().context()
        context.update(target_surface_estimate=self.grounding, search=self.search_context,
                       strategy_replans=self.replans, recent_replans=self.replan_history[-2:])
        return context


def metric_co_motion(previous, current, state, centers, feedback, arm):
    """Abstain on camera switches, tiny motion, absent depth or base motion.

    Surface correspondence is still supplied by the VLM, so this is corroborating
    evidence, never ground truth. A static object cannot pass a 1cm hand lift.
    """
    if not previous or not current.get("valid") or not previous["target"].get("valid") or not feedback:
        return None
    base=np.asarray(feedback.get("base_integral",[0,0,0]))
    if np.linalg.norm(base[:2])>.002 or abs(base[2])>.005:
        return None
    old_views={x["view"] for x in previous["target"]["views"] if x["valid"]}
    new_views={x["view"] for x in current["views"] if x["valid"]}
    if old_views!=new_views:
        return None
    dp=centers[arm]-np.asarray(previous["centers"][arm])
    size=float(np.linalg.norm(dp))
    if size<.005:
        return None
    target_delta=np.asarray(current["point_base_m"])-previous["target"]["point_base_m"]
    return bool(np.linalg.norm(target_delta) >= .5*size and
                target_delta@dp >= .5*size*size and
                np.linalg.norm(target_delta-dp) <= max(.004,.5*size))


class GroundedController:
    # A hard candidate-count limit bounds CPU work; no unbounded IK search.
    max_preflights=10
    max_replans=2

    def __init__(self, model, servo, harness):
        self.model,self.servo,self.harness=model,servo,harness
        self.search=CoverageSearch()
        self.previous=None
        self.target={"valid":False}
        self.centers={}
        self.depth_guard=None
        self.reposition=None
        self.reposition_left=0
        self.replan_needed=None
        self.progress={}

    def observe(self,evidence,state,depths,depth_receipt):
        manager=self.harness
        observed_goal_index=manager.index
        self.target=localize_target(evidence,depths,self.model,state.q)
        self.centers=self.model.grasp_centers(state.q)
        manager.grounding=self.target
        camera=self.model.spec["metadata"]["cameras"]["head"]
        new_coverage=self.search.observe(manager.index,self.model.forward(state.q,"camera_head"),camera["K"],
                                        camera["width"],depth_receipt["head"]["valid_fraction"],evidence.visible)
        self.depth_guard=LocalDepthGuard(observed_cloud(depths,self.model,state.q),self.model,state.q,depths)
        distance=None
        if self.target["valid"]:
            point=np.asarray(self.target["point_base_m"])
            distance=max(float(np.linalg.norm(point-self.centers[a])) for a in manager.arms)
            self.target["distance_to_active_closing_center_m"]=distance
            self.target["target_minus_center_base_m"]={a:(point-self.centers[a]).round(4).tolist() for a in manager.arms}
        co_motion=all(metric_co_motion(self.previous,self.target,state,self.centers,manager.feedback,a) is True for a in manager.arms)
        previous_distance=(self.previous or {}).get("distance")
        same_goal=self.previous and self.previous["goal_index"]==manager.index
        reduced=bool(same_goal and distance is not None and previous_distance is not None and distance<previous_distance-.001)
        self.progress={"new_search_coverage":new_coverage,"target_distance_m":distance,
                       "metric_co_motion":co_motion,"distance_reduced":reduced}
        internal=dict(self.progress)
        internal["target_distance_m"]=distance if distance is not None else math.inf
        if reduced and manager.stage in ("APPROACH","ALIGN"):
            manager.stage_age=0
        # Do not feed off-screen / cross-camera 2D EEF distances into progress.
        manager.observe(evidence,state,geometry=None,measured_progress=internal)
        if self.reposition_left and self.reposition not in manager.palette():
            self.reposition_left=0  # a new stage may forbid the queued strategy
        manager.search_context=self.search.context()
        self.replan_needed=None
        if manager.stop_reason=="RECOVERY_BUDGET_EXHAUSTED":
            self.replan_needed=manager.stop_reason
        if not evidence.visible and self.search.complete and self.reposition_left<=0:
            self.replan_needed="LOCAL_VIEW_SWEEP_COMPLETE_TARGET_NOT_FOUND"
        if self.replan_needed and manager.replans>=self.max_replans:
            manager.stop_reason="STRATEGY_REPLAN_BUDGET_EXHAUSTED"
            self.replan_needed=None
        self.previous={"target":self.target,"centers":{a:p.copy() for a,p in self.centers.items()},
                       "distance":distance,"goal_index":observed_goal_index}

    def apply_recovery(self,value):
        manager=self.harness
        if not self.replan_needed or manager.replans>=self.max_replans:
            raise ValueError("Recovery is not currently authorized")
        if manager.stop_reason not in (None,"RECOVERY_BUDGET_EXHAUSTED"):
            raise ValueError("A safety/plan stop cannot be erased by a VLM")
        # The original goal list/index and all held-object claims are untouched.
        manager.replans+=1
        manager.replan_history.append({"goal_index":manager.index,"trigger":self.replan_needed,**value})
        manager.stop_reason=None
        manager.recoveries=0
        manager.feedback=None
        manager.transition("SEARCH" if not manager.observation.visible else "RECOVER")
        manager.stage_age=0
        manager.recovery_entered_after=manager.executions
        self.reposition=None; self.reposition_left=0
        strategy=value["strategy"]
        if strategy in ("scan_left","scan_right"):
            self.search.direction=1 if strategy=="scan_left" else -1
            if manager.observation.visible:
                self.reposition=Action("base","yaw_plus" if self.search.direction>0 else "yaw_minus","coarse")
                self.reposition_left=1
        elif strategy.startswith("move_"):
            self.reposition=Action("base",strategy[5:],"fine")
            self.reposition_left=5  # <=30cm, 5 fresh depth checks / observations
        elif strategy=="retry_approach" and manager.observation.visible:
            manager.transition("APPROACH")
        elif strategy=="hold":
            manager.stop_reason="MODEL_REQUESTED_SAFE_STOP"
        self.replan_needed=None

    def _expected_point(self, action, state, trial=None):
        point=np.asarray(self.target["point_base_m"])
        centers=self.centers
        if action.part=="base":
            amount=action.amount(self.harness.carry)
            if action.move in TRANSLATIONS:
                point=point-np.asarray(TRANSLATIONS[action.move])*amount
            else:
                angle=-amount*(1 if action.move=="yaw_plus" else -1)
                c,s=math.cos(angle),math.sin(angle)
                point=np.array([[c,-s,0],[s,c,0],[0,0,1]])@point
        elif trial is not None and trial.joint_plan is not None:
            centers=self.model.grasp_centers(trial.joint_plan[-1])
        elif action.move in TRANSLATIONS and action.part in (*self.harness.arms,"both"):
            if action.frame!="base":
                return None
            centers={a:p+(np.asarray(TRANSLATIONS[action.move])*action.amount(self.harness.carry)
                          if action.part in (a,"both") else 0) for a,p in centers.items()}
        else:
            return None
        return max(float(np.linalg.norm(point-centers[a])) for a in self.harness.arms)

    def candidates(self,state):
        palette=self.harness.palette()
        if self.depth_guard is None:
            raise RuntimeError("Current RGB-D observation required before proposing motion")
        rows=[]; proposed=[]
        if self.target.get("valid") and self.harness.stage not in ("GRASP","RELEASE","VERIFY_GRASP","VERIFY_PLACE","VERIFY_SUPPORT","VERIFY_EFFECT"):
            ranked=[]
            before=self.target["distance_to_active_closing_center_m"]
            for action in palette:
                if action.move in TRANSLATIONS and action.part in (*self.harness.arms,"both") and action.frame=="base":
                    after=self._expected_point(action,state)
                    if after is not None:
                        ranked.append((before-after,action))
            ranked.sort(key=lambda x:x[0],reverse=True)
            translation_slots=2 if self.harness.stage in ("ALIGN","INTERACT") and not self.harness.carry else 6
            # Each promising direction gets its 2mm fallback BEFORE spending
            # the budget on a second direction. Micro cannot disappear again.
            for gain,action in ranked:
                if gain<=0 or len(proposed)>=translation_slots: continue
                if action not in proposed: proposed.append(action)
                micro=Action(action.part,action.move,"micro",action.frame)
                if micro in palette and micro not in proposed: proposed.append(micro)
                if len(proposed)>=translation_slots: break
            # Bounded body repositioning is available when the arm is blocked;
            # it still needs CURRENT observed free space, not a guessed room map.
            body=[]
            for action in palette:
                if action.part=="base" and action.move in TRANSLATIONS and action.scale=="fine":
                    distance=self._expected_point(action,state)
                    if distance is not None: body.append((distance,action))
            proposed.extend(a for _,a in sorted(body,key=lambda x:x[0])[:1])
            if self.harness.stage=="RECOVER":
                proposed=[a for a in palette if a.part in (*self.harness.arms,"both") and a.move in ("back","up")]+proposed
            open_action=Action(self.harness.goal.hand,"open")
            if open_action in palette:
                proposed.insert(0,open_action)
            for action in palette:
                if action.move in ("roll_plus","roll_minus","pitch_plus","pitch_minus","yaw_plus","yaw_minus") and action.part in (*self.harness.arms,"both") and action.scale=="micro":
                    if len(proposed)<self.max_preflights: proposed.append(action)
            if self.harness.stage in ("APPROACH","RECOVER"):
                for move in ("up","down"):
                    torso=Action("torso",move,"micro")
                    if torso in palette and len(proposed)<self.max_preflights:proposed.append(torso)
        else:
            # State-machine verification has deliberately small palettes.
            proposed=[a for a in palette if a!=HOLD and a.part!="torso"]
            if self.harness.stage=="VERIFY_GRASP":
                proposed.sort(key=lambda a:a.scale!="fine")  # depth needs >5mm for co-motion evidence
            if not self.target.get("valid") and self.harness.stage in ("APPROACH","ALIGN"):
                proposed=[]  # visible but bad depth: observe/replan, no blind approach
        if self.target.get("valid") and self.harness.goal.kind=="navigate" and self.harness.stage=="APPROACH":
            # A navigation goal must retain turns and alternate translations;
            # the manipulation ranking's one body fallback is insufficient.
            proposed=[a for a in palette if a.part=="base" and a.scale=="fine"]
        allowed=[]; started=time.perf_counter()
        for action in [HOLD,*dict.fromkeys(proposed[:self.max_preflights])]:
            ok,reason=self.depth_guard.check(action,self.harness.carry)
            trial=None
            if ok:
                # Recreate the servo with the actual command latch, NOT measured
                # aperture. A held object's width must never turn CLOSE into OPEN.
                trial=SafeServo(self.model,state,self.servo.grips.copy(),self.servo.limits)
                ok=trial.begin(action,state,self.harness.carry)
                reason="KINEMATIC_PATH_FEASIBLE" if ok else trial.status
            row={"action":asdict(action),"accepted":bool(ok),"reason":reason}
            if ok:
                allowed.append(action)
                row["planned_ticks"]=trial.total_ticks
                if self.target.get("valid"):
                    distance=self._expected_point(action,state,trial)
                    if distance is not None:
                        row["predicted_target_distance_m"]=round(distance,5)
                        row["predicted_distance_gain_m"]=round(self.target["distance_to_active_closing_center_m"]-distance,5)
            rows.append(row)
        self.harness.candidate_receipt={"state_q":state.q.tolist(),"command_grip_latch":self.servo.grips.tolist(),
            "tested":rows,"preflight_s":time.perf_counter()-started,"max_nonhold_preflights":self.max_preflights,
            "depth_guard":self.depth_guard.receipt(),"fresh_execution_recheck_required":True}
        if not allowed:
            self.harness.stop_reason="NO_SAFE_ACTION_AT_CURRENT_STATE"
        return tuple(allowed)

    def search_action(self,state):
        """One finite pulse, not a hidden multi-step macro or VLM fiction."""
        if self.reposition is not None and self.reposition_left>0:
            action,reason=self.reposition,"BOUNDED_REPOSITION_WITH_CURRENT_DEPTH"
        else:
            action,reason=self.search.propose()
        if action is None:
            self.harness.stop_reason=reason
            return HOLD,{"source":"measured_search_controller","reason":reason}
        self.harness.authorize(action)
        ok,why=self.depth_guard.check(action,self.harness.carry)
        trial=SafeServo(self.model,state,self.servo.grips.copy(),self.servo.limits)
        if ok:
            ok=trial.begin(action,state,self.harness.carry)
            why=trial.status
        if not ok:
            self.harness.recover("SEARCH_ACTION_REJECTED_"+why)
            self.reposition_left=0
            return HOLD,{"source":"measured_search_controller","reason":why,"rejected":asdict(action)}
        return action,{"source":"measured_search_controller","reason":reason,"preflight_status":why,
                       "search":self.search.context(),"depth_guard":self.depth_guard.receipt()}

    def executed(self,action,feedback):
        self.search.executed(feedback)
        if action==self.reposition and feedback["status"]=="TARGET_REACHED":
            self.reposition_left-=1
        self.harness.executed(action,feedback)
