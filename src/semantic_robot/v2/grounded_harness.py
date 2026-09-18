"""Sensor-grounded finite task/search/recovery orchestration, not task scripts."""
import math
import time
import copy
from dataclasses import asdict

import numpy as np

from .grounding import LocalDepthGuard, localize_target, observed_cloud
from .harness import TaskHarness
from .protocol import Action, HOLD, TRANSLATIONS, VIEWS, strict_json
from .search import CoverageSearch
from .servo import SafeServo


RECOVERY_STRATEGIES=("scan_left","scan_right","move_forward","move_left","move_right","retry_approach","hold")


def parse_recovery(text):
    value=strict_json(text)
    if not isinstance(value,dict) or set(value)!={"strategy","visible_reason"}:
        raise ValueError("Recovery changes strategy only, not goals or completion")
    # The rationale is audit text, never a command. A harmless 244-character
    # explanation must not abort an otherwise valid safe strategy. Keep a real
    # payload bound and preserve the exact text; no truncation or inferred enum.
    if value["strategy"] not in RECOVERY_STRATEGIES or not isinstance(value["visible_reason"],str) or not 1<=len(value["visible_reason"])<=1024:
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
        self.motion_receipt={}

    def palette(self):
        palette=super().palette()
        if not self.stop_reason and self.stage in ("SEARCH","RECOVER"):
            palette += (Action("base","yaw_plus","coarse"),Action("base","yaw_minus","coarse"))
        if (not self.stop_reason and self.stage == "ALIGN" and self.goal.kind == "pick"
                and not any(self.hold_verified.values())):
            # A near target can still be beyond an extended arm's workspace.
            # Permit a *single* small, fresh-depth-checked body adjustment; do
            # not force wrist-only alignment or move a potentially held load.
            palette += tuple(Action("base", move, "micro") for move in ("forward", "back", "left", "right"))
            palette += tuple(Action("torso", move, "micro") for move in ("up", "down"))
        return tuple(dict.fromkeys(palette))

    def context(self):
        context=super().context()
        context.update(target_surface_estimate=self.grounding, search=self.search_context,
                       strategy_replans=self.replans, recent_replans=self.replan_history[-2:],
                       egocentric_motion=self.motion_receipt)
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
    max_preflights=24
    max_replans=2

    def __init__(self, model, servo, harness, visual_odometry=False):
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
        self.motion=None
        self.pending_motion=None
        if visual_odometry:
            from .odometry import RGBDMotion
            self.motion=RGBDMotion()

    @property
    def can_replan_stop(self):
        return (self.harness.stop_reason == "RECOVERY_BUDGET_EXHAUSTED"
                and self.harness.replans < self.max_replans)

    def update_motion(self,images,depths,state):
        if self.motion is None:raise ValueError("Visual motion was not enabled")
        receipt=self.motion.observe(images,depths,self.model,state.q)
        self.harness.motion_receipt=receipt
        if not receipt["valid"]:
            # Never turn an unreliable velocity integral into new coverage.
            self.harness.stop_reason="VISUAL_ODOMETRY_UNCERTAIN"
            return receipt
        if receipt.get("initial"):
            if self.pending_motion is not None:raise ValueError("Missing pre-action visual reference")
            return receipt
        if self.pending_motion is None:raise ValueError("No executed action matches this visual motion")
        feedback=copy.deepcopy(self.pending_motion)
        feedback["base_velocity_integral_raw"]=feedback["base_integral"]
        feedback["base_integral"]=receipt["body_delta"]
        feedback["base_motion_source"]="onboard_RGBD_not_joint_velocity_integration"
        feedback["base_motion_convention"]="displacement_in_previous_body_frame"
        action=self.harness.last_action
        if action is not None and action.part=="base" and feedback["status"] in ("TARGET_REACHED","BASE_TRACKING_FAILED"):
            # Re-evaluate ONLY the post-motion base residual that was previously
            # judged from the unreliable velocity integral. Never clear a servo
            # interruption, collision, joint-limit or other hard safety failure.
            expected=np.zeros(3);amount=action.amount(feedback.get("carry",False))
            if action.move in TRANSLATIONS:expected[:2]=np.asarray(TRANSLATIONS[action.move])[:2]*amount
            elif action.move in ("yaw_plus","yaw_minus"):expected[2]=amount*(1 if action.move=="yaw_plus" else -1)
            residual=np.asarray(receipt["body_delta"])-expected
            feedback["base_tracking_status_from_velocity_raw"]=feedback["status"]
            feedback["visual_base_residual"]=residual.tolist()
            feedback["status"]=("TARGET_REACHED" if np.linalg.norm(residual[:2])<.012 and abs(residual[2])<np.deg2rad(2.)
                                else "BASE_TRACKING_FAILED")
        self.search.executed(feedback)
        self.harness.feedback=feedback
        if self.harness.history:self.harness.history[-1]={**self.harness.history[-1],"feedback":feedback}
        self.harness.search_context=self.search.context()
        self.pending_motion=None
        return receipt

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
        if (manager.stage == "ALIGN" and manager.goal.kind == "pick" and distance is not None
                and distance > .10 and not any(manager.hold_verified.values())):
            # Hysteresis: enter ALIGN at 8cm, leave above 10cm. A changed contact
            # estimate must not strand a 16cm-away target in wrist-only control.
            manager.transition("APPROACH")
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

    def _translation_direction(self, action, state):
        vector = np.asarray(TRANSLATIONS[action.move], dtype=float)
        if action.frame == "base":
            return vector
        if action.frame in VIEWS:
            # Exactly SafeServo's camera frame: forward into image, left/up in
            # image coordinates. This estimate never replaces its real IK test.
            frame = self.model.forward(state.q, "camera_" + action.frame)[:3, :3]
            return frame @ np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]) @ vector
        return None

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
            direction = self._translation_direction(action, state)
            if direction is None:
                return None
            centers={a:p+(direction*action.amount(self.harness.carry)
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
                if action.move in TRANSLATIONS and action.part in (*self.harness.arms,"both"):
                    after=self._expected_point(action,state)
                    if after is not None:
                        ranked.append((before-after,action))
            ranked.sort(key=lambda x:x[0],reverse=True)
            translation_slots=8
            # Rank *directions*, not repeated scales of one direction. Failed
            # forward IK must not spend every slot and hide feasible up/left.
            # Camera-frame diagonals may be feasible when base axes are not.
            selected_directions=[]
            for gain,action in ranked:
                if gain<=0 or len(proposed)>=translation_slots: continue
                direction=self._translation_direction(action,state)
                if any(action.part==part and float(direction@old)>.995 for part,old in selected_directions):continue
                selected_directions.append((action.part,direction))
                if action not in proposed: proposed.append(action)
                micro=Action(action.part,action.move,"micro",action.frame)
                if micro in palette and micro not in proposed: proposed.append(micro)
                if len(proposed)>=translation_slots: break
            # Bounded body repositioning is available when the arm is blocked;
            # it still needs CURRENT observed free space, not a guessed room map.
            body=[]
            for action in palette:
                if action.part=="base" and action.move in TRANSLATIONS and action.scale==("micro" if self.harness.stage=="ALIGN" else "fine"):
                    distance=self._expected_point(action,state)
                    if distance is not None: body.append((distance,action))
            proposed.extend(a for _,a in sorted(body,key=lambda x:x[0])[:1])
            if self.harness.stage=="RECOVER":
                proposed=[a for a in palette if a.part in (*self.harness.arms,"both") and a.move in ("back","up")]+proposed
            open_action=Action(self.harness.goal.hand,"open")
            if open_action in palette:
                proposed.insert(0,open_action)
            for action in palette:
                if action.move in ("roll_plus","roll_minus","pitch_plus","pitch_minus","yaw_plus","yaw_minus") and action.part in (*self.harness.arms,"both"):
                    if len(proposed)<self.max_preflights: proposed.append(action)
            if self.harness.stage in ("APPROACH","ALIGN","RECOVER"):
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
        if self.motion is None:self.search.executed(feedback)
        else:
            if self.pending_motion is not None:raise ValueError("An executed motion has not been observed")
            self.pending_motion=copy.deepcopy(feedback)
        if action==self.reposition and feedback["status"]=="TARGET_REACHED":
            self.reposition_left-=1
        self.harness.executed(action,feedback)
