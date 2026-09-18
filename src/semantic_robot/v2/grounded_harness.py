"""Sensor-grounded finite task/search/recovery orchestration, not task scripts."""
import math
import time
import copy
from dataclasses import asdict, replace

import numpy as np

from .grounding import LocalDepthGuard, localize_target, observed_cloud
from .harness import TaskHarness
from .protocol import Action, HOLD, TRANSLATIONS, VIEWS, strict_json
from .search import CoverageSearch
from .servo import SafeServo
from .bimanual import localize_hand_contacts, all_claims
from .navigation import navigation_workspace_check


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
    def __init__(self, goals, *, active_grasp_probe=False,contact_geometry=True,held_inspection=False,reference_from_planner=False,inspection_budget_aware=False,multicamera_inspection=False):
        super().__init__(goals)
        self.grounding={}
        self.search_context={}
        self.replans=0
        self.replan_history=[]
        self.candidate_receipt={}
        self.motion_receipt={}
        self.pending_grasp={"left":False,"right":False}
        self.last_gripper=None
        self.contact_geometry=bool(contact_geometry)
        self.active_grasp_probe=bool(active_grasp_probe)
        self.grasp_probe_attempts={}
        self.grasp_probe={"eligible":False,"reason":"NOT_OBSERVED"}
        self.held_inspection_enabled=bool(held_inspection)
        self.inspection_budget_aware=bool(inspection_budget_aware)
        if self.inspection_budget_aware and not self.held_inspection_enabled:raise ValueError("Inspection budget mode requires held inspection")
        self.multicamera_inspection=bool(multicamera_inspection)
        if self.multicamera_inspection and not self.inspection_budget_aware:raise ValueError("Multi-camera inspection requires budget-aware held inspection")
        self.target_references={}
        self.reference_from_planner=bool(reference_from_planner)
        self.reference_receipts={}
        self.held_inspection={}

    @property
    def search_reference(self):
        return self.target_references.get(self.index,"unknown")

    def resolve_reference(self,evidence):
        if not self.held_inspection_enabled:return
        if self.reference_from_planner:return  # Visual invisibility cannot erase a semantic relationship.
        reference=getattr(evidence,"target_reference","unknown")
        if reference=="unknown":return  # Missing relation never fabricates world/held truth.
        self.bind_reference(reference,"legacy_visual_observer_B16")

    def bind_reference(self,reference,source):
        from .target_reference import REFERENCES
        if reference not in REFERENCES:raise ValueError("Invalid reference")
        old=self.target_references.get(self.index)
        if old is not None and old!=reference:
            self.stop_reason="TARGET_REFERENCE_CONTRADICTION";return
        if reference.startswith("held_") and not self.hold_verified[reference[5:]]:
            self.stop_reason="TARGET_REFERENCE_REQUIRES_VERIFIED_HOLD";return
        self.target_references[self.index]=reference
        self.reference_receipts[self.index]={"reference":reference,"source":source,"not_current_attachment_evidence":True}
        if reference=="unknown" and any(self.hold_verified.values()):
            self.stop_reason="TARGET_REFERENCE_UNRESOLVED_WITH_HELD_LOAD"

    def possibly_loaded_arms(self):
        # Unknown aperture after a CLOSE is not evidence of an empty hand.
        return [a for i,a in enumerate(("left","right")) if self.pending_grasp[a]
                and (self.last_gripper is None or not np.isfinite(self.last_gripper[i])
                     or self.last_gripper[i]>=.0015)]

    def observe(self,evidence,state,geometry=None,measured_progress=None):
        self.resolve_reference(evidence)
        self.last_gripper=np.asarray(state.gripper,dtype=float).copy()
        return super().observe(evidence,state,geometry,measured_progress)

    def recover(self,event):
        if self.possibly_loaded_arms():
            # Verification uncertainty is not permission to drop a load. This
            # finite safe stop is NOT success and cannot enter a VLM replan.
            self.events.append({"event":event,"goal":self.index,"stage":self.stage,
                                "possibly_loaded_arms":self.possibly_loaded_arms()})
            self.stop_reason="UNVERIFIED_LOAD_PRESERVED_"+event
            return
        return super().recover(event)

    def _complete_goal(self):
        arms=self.arms;kind=self.goal.kind
        super()._complete_goal()
        if kind in ("pick","place"):
            for a in arms:self.pending_grasp[a]=False

    def update_grasp_probe(self,state):
        """A bounded exploratory close, NEVER an enclosure/holding assertion.

        The 4cm heuristic limits an attempt, not object width or contact pose.
        Explicit contrary evidence, missing contact, load and danger still veto.
        """
        obs=self.observation
        count=self.grasp_probe_attempts.get(self.index,0)
        distance=self.grounding.get("distance_to_active_closing_center_m")
        checks={"enabled":self.active_grasp_probe,"pick_align":self.goal.kind=="pick" and self.stage=="ALIGN",
            "not_stopped":not self.stop_reason,"attempt_budget":count<2,
            "visible_and_not_contradicted":bool(obs and obs.visible and obs.enclosed is None and obs.hazard=="none"),
            "fresh_near_contact":bool(self.grounding.get("valid") and distance is not None and 0<=distance<=.04),
            "no_existing_load":not self.carry and not any(self.hold_verified.values()) and not any(self.pending_grasp.values()),
            "active_fingers_open":all(state.gripper[("left","right").index(a)]>.02 for a in self.arms)}
        self.grasp_probe={"eligible":all(checks.values()),"checks":checks,"attempts_used":count,"max_attempts":2,
            "max_surface_distance_m":.04,"command":"close","success_claim":False,
            "reason":"EXPLORATORY_CLOSE_REQUIRES_SUBSEQUENT_VERIFICATION" if all(checks.values()) else "PRECONDITIONS_NOT_MET"}

    @property
    def carry(self):
        # An unverified close may already support part of a fragile load. Do
        # not unlock orientation/independent alignment merely because the final
        # two-hand verification has not yet passed.
        return super().carry or (self.goal.level and any(self.pending_grasp.values()))

    def executed(self,action,feedback):
        probe=(self.grasp_probe.get("eligible") and self.stage=="ALIGN" and
               action==Action(self.goal.hand,"close"))
        if probe and (feedback.get("control_ticks",0)>0 or feedback["status"]=="TARGET_REACHED"):
            self.grasp_probe_attempts[self.index]=self.grasp_probe_attempts.get(self.index,0)+1
        widths=feedback.get("finger_mean_m")
        self.last_gripper=np.asarray(widths,dtype=float).copy() if widths is not None else None
        # Even an interrupted close may have made contact. Keep the latch until
        # an explicit OPEN completes or grasp evidence really verifies holding.
        if action.move=="close" and (feedback.get("control_ticks",0)>0 or feedback["status"]=="TARGET_REACHED"):
            for arm in (("left","right") if action.part=="both" else (action.part,)):
                if arm in self.pending_grasp and self.goal.kind=="pick":self.pending_grasp[arm]=True
        super().executed(action,feedback)
        if feedback["status"]=="TARGET_REACHED" and action.move in ("close","open"):
            arms=("left","right") if action.part=="both" else (action.part,)
            for arm in arms:
                if arm in self.pending_grasp:
                    if action.move=="open":self.pending_grasp[arm]=False
                    elif self.goal.kind=="pick":self.pending_grasp[arm]=True
        if probe and feedback["status"]=="TARGET_REACHED":
            self.events.append({"event":"UNVERIFIED_GRASP_PROBE","goal":self.index,
                                "attempt":self.grasp_probe_attempts[self.index],"success_claim":False})
            self.transition("VERIFY_GRASP")
        self.grasp_probe={**self.grasp_probe,"eligible":False}

    def palette(self):
        if self.possibly_loaded_arms() and self.stage not in ("VERIFY_GRASP","GRASP"):
            return (HOLD,)
        if self.held_inspection_enabled and self.stage in ("SEARCH","RECOVER") and self.observation and not self.observation.visible:
            if self.stop_reason:return (HOLD,)
            if self.search_reference.startswith("held_"):
                if self.multicamera_inspection:
                    from .multicamera_inspection import multicamera_palette
                    return multicamera_palette(self)
                from .held_inspection import inspection_palette
                arm=self.search_reference[5:]
                return inspection_palette(arm,self.carry,self.inspection_budget_aware) if self.hold_verified[arm] else (HOLD,)
            if self.search_reference=="unknown" and any(self.hold_verified.values()):return (HOLD,)
        palette=super().palette()
        if (not self.stop_reason and self.stage=="ALIGN" and self.goal.kind=="pick"
                and self.grasp_probe.get("eligible")):
            palette += (Action(self.goal.hand,"close"),)
        if (self.contact_geometry and not self.stop_reason and self.goal.kind=="pick" and self.goal.hand in ("left","right")
                and self.stage in ("APPROACH","ALIGN") and not any(self.pending_grasp.values())
                and not any(self.hold_verified.values())):
            scales=("micro","fine","coarse") if self.stage=="APPROACH" else ("micro","fine")
            palette += tuple(Action(self.goal.hand,move,scale,"tool") for move in TRANSLATIONS for scale in scales)
        if (not self.stop_reason and self.goal.kind == "pick" and self.goal.hand == "both"
                and self.stage in ("APPROACH", "ALIGN") and not any(self.hold_verified.values())
                and not any(self.pending_grasp.values())):
            # Before grasping, each hand must be able to reach its own contact.
            # After closing, VERIFY_GRASP retains only the synchronized lift.
            extra=[]
            for arm in self.arms:
                frames=("base", arm+"_wrist", "head")
                scales=("micro","fine","coarse") if self.stage=="APPROACH" else ("micro","fine")
                extra.extend(Action(arm,move,scale,frame) for move in TRANSLATIONS for scale in scales for frame in frames)
                if self.stage=="ALIGN":
                    extra.extend(Action(arm,move,scale,"tool") for move in
                        ("roll_plus","roll_minus","pitch_plus","pitch_minus","yaw_plus","yaw_minus")
                        for scale in ("micro","fine"))
            palette += tuple(extra)
        if not self.stop_reason and self.stage in ("SEARCH","RECOVER"):
            palette += (Action("base","yaw_plus","coarse"),Action("base","yaw_minus","coarse"))
        if not self.stop_reason and self.stage=="APPROACH" and self.goal.kind=="navigate":
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
        if self.held_inspection_enabled:
            context.update(target_reference=self.search_reference,held_inspection=self.held_inspection,
                           reference_binding=self.reference_receipts.get(self.index))
        context.update(target_surface_estimate=self.grounding, search=self.search_context,
                       strategy_replans=self.replans, recent_replans=self.replan_history[-2:],
                       egocentric_motion=self.motion_receipt, unverified_close_latches=self.pending_grasp.copy(),
                       active_grasp_probe=self.grasp_probe.copy())
        if hasattr(self,"approach_progress"):context["approach_progress"]=self.approach_progress
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

    def __init__(self, model, servo, harness, visual_odometry=False,grasp_motion=False,odometry_estimator="pnp",approach_progress=False,persistent_grasp_tracks=False,spatial_grasp_features=False):
        self.model,self.servo,self.harness=model,servo,harness
        if approach_progress and not visual_odometry:raise ValueError("Approach progress requires measured visual motion")
        from .approach_progress import ApproachProgress
        self.approach_monitor=ApproachProgress() if approach_progress else None
        self.search=CoverageSearch()
        self.previous=None
        self.target={"valid":False}
        self.hand_targets={}
        self.centers={}
        self.depth_guard=None
        self.reposition=None
        self.reposition_left=0
        self.replan_needed=None
        self.progress={}
        self.motion=None
        self.pending_motion=None
        self.pending_exploratory=True
        self.goal_changed=False
        from .held_inspection import HeldInspection
        self.inspector=HeldInspection(budget_aware=harness.inspection_budget_aware) if harness.held_inspection_enabled else None
        if harness.multicamera_inspection:
            from .multicamera_inspection import MultiCameraInspection
            self.inspector=MultiCameraInspection()
        self.grasp_verifier=None
        if persistent_grasp_tracks and not grasp_motion:raise ValueError("Persistent features require registered grasp motion")
        if spatial_grasp_features and not persistent_grasp_tracks:raise ValueError("Spatial features require persistent tracking")
        if grasp_motion:
            if not visual_odometry:raise ValueError("Registered grasp motion requires independent RGB-D body motion")
            from .grasp_motion import GraspMotionVerifier
            self.grasp_verifier=GraspMotionVerifier(persistent_tracks=persistent_grasp_tracks,spatial_seed_features=spatial_grasp_features)
        if visual_odometry:
            from .odometry import RGBDMotion
            self.motion=RGBDMotion(odometry_estimator)

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
        self.search.executed(feedback,exploratory=self.pending_exploratory)
        self.harness.feedback=feedback
        if self.harness.history:self.harness.history[-1]={**self.harness.history[-1],"feedback":feedback}
        self.harness.search_context=self.search.context()
        self.pending_motion=None
        return receipt

    def observe(self,evidence,state,depths,depth_receipt,images=None,self_geometry=None):
        manager=self.harness
        self.goal_changed=False
        observed_goal_index=manager.index
        observed_kind,observed_arms=manager.goal.kind,manager.arms
        manager.resolve_reference(evidence)
        bimanual=manager.goal.kind=="pick" and manager.goal.hand=="both"
        self.hand_targets=localize_hand_contacts(evidence,depths,self.model,state.q) if bimanual else {}
        if bimanual:
            valid=all(row["valid"] for row in self.hand_targets.values())
            self.target={"valid":valid,"source":"separate_VLM_hand_pixels_plus_onboard_depth",
                         "reason":"OBSERVED_HAND_CONTACTS" if valid else "MISSING_OR_INVALID_HAND_CONTACT",
                         "hand_contacts":self.hand_targets,"views":[],"surface_point_not_object_pose":True}
            if valid:
                self.target["point_base_m"]=np.mean([row["point_base_m"] for row in self.hand_targets.values()],axis=0).tolist()
                self.target["point_is_display_midpoint_not_grasp_target"]=True
            contacts=getattr(evidence,"hand_contacts",())
            evidence=replace(evidence,enclosed=all_claims(row.enclosed for row in contacts),
                             co_moving=all_claims(row.co_moving for row in contacts))
        else:
            self.target=localize_target(evidence,depths,self.model,state.q)
        self.centers=self.model.grasp_centers(state.q)
        manager.grounding=self.target
        camera=self.model.spec["metadata"]["cameras"]["head"]
        target_bearing=(math.atan2(self.target["point_base_m"][1],self.target["point_base_m"][0])
                        if self.target["valid"] else None)
        new_coverage=self.search.observe(manager.index,self.model.forward(state.q,"camera_head"),camera["K"],
                                        camera["width"],depth_receipt["head"]["valid_fraction"],evidence.visible,target_bearing)
        self.depth_guard=LocalDepthGuard(observed_cloud(depths,self.model,state.q),self.model,state.q,depths)
        distance=None
        if self.target["valid"]:
            points=self._points_for_arms()
            distances={a:float(np.linalg.norm(points[a]-self.centers[a])) for a in manager.arms}
            distance=max(distances.values())
            self.target["distance_to_active_closing_center_m"]=distance
            self.target["mean_contact_distance_m"]=float(np.mean(list(distances.values())))
            self.target["per_hand_distance_m"]=distances
            self.target["target_minus_center_base_m"]={a:(points[a]-self.centers[a]).round(4).tolist() for a in manager.arms}
            if manager.contact_geometry:
                self.target["target_minus_center_tool_m"]={a:(self.model.forward(state.q,a)[:3,:3].T@(points[a]-self.centers[a])).round(4).tolist() for a in manager.arms}
        co_motion=True
        for arm in manager.arms:
            previous=self.previous
            current=self.target
            if bimanual:
                current=self.hand_targets[arm]
                previous=({**previous,"target":previous.get("hand_targets",{}).get(arm,{"valid":False})}
                          if previous else None)
            co_motion=co_motion and metric_co_motion(previous,current,state,self.centers,manager.feedback,arm) is True
        # With two contacts, improving one hand is progress even when the other
        # hand still determines the maximum used by the strict stage gate.
        progress_distance=self.target.get("mean_contact_distance_m",distance)
        previous_distance=(self.previous or {}).get("progress_distance")
        same_goal=self.previous and self.previous["goal_index"]==manager.index
        reduced=bool(same_goal and progress_distance is not None and previous_distance is not None and progress_distance<previous_distance-.001)
        self.progress={"new_search_coverage":new_coverage,"target_distance_m":distance,
                       "metric_co_motion":co_motion,"distance_reduced":reduced}
        internal=dict(self.progress)
        if self.inspector is not None and manager.search_reference.startswith("held_"):
            fresh,context=(self.inspector.observe(self.model,state,manager,depths,self_geometry)
                           if manager.multicamera_inspection else self.inspector.observe(self.model,state,manager))
            manager.held_inspection=context
            internal["new_search_coverage"]=fresh
            self.progress["new_relative_inspection_view"]=fresh
            if not context["valid"]:manager.stop_reason="NO_VERIFIED_HELD_ANCHOR"
        if self.approach_monitor is not None:
            points=self._points_for_arms() if self.target.get("valid") else {}
            poses={a:self.model.forward(state.q,a).copy() for a in manager.arms}
            for arm in poses:poses[arm][:3,3]=self.centers[arm]
            manager.approach_progress=self.approach_monitor.observe(goal_index=manager.index,
                kind=manager.goal.kind,stage=manager.stage,points=points,poses=poses,
                execution=manager.executions,action=manager.last_action,feedback=manager.feedback,
                motion=manager.motion_receipt,loaded=any(manager.pending_grasp.values()) or any(manager.hold_verified.values()))
            self.progress["approach_progress"]=manager.approach_progress
        if self.grasp_verifier is not None:
            if images is None or self_geometry is None:raise ValueError("Fresh raw images and robot-only self geometry required")
            targets=self.hand_targets if bimanual else {a:self.target for a in manager.arms}
            registration=self.grasp_verifier.observe(self.model,state,images,depths,self_geometry,manager,targets,manager.motion_receipt,evidence)
            self.progress["registered_grasp_motion"]=registration
            internal["registered_grasp_motion"]=registration
        internal["target_distance_m"]=distance if distance is not None else math.inf
        if manager.goal.kind=="navigate":
            internal["navigation_aligned"]=bool(self.target["valid"] and abs(self._navigation_geometry()["bearing_deg"])<=15.)
            self.progress["navigation_aligned"]=internal["navigation_aligned"]
            workspace=navigation_workspace_check(self.model,state.q,self.target,manager.arms)
            self.target["navigation_workspace_check"]=workspace
            internal["navigation_reach_possible"]=workspace["valid"] and workspace["within_optimistic_reach"]
            self.progress["navigation_workspace_check"]=workspace
        if (manager.stage == "ALIGN" and manager.goal.kind == "pick" and distance is not None
                and distance > .10 and not any(manager.hold_verified.values())):
            # Hysteresis: enter ALIGN at 8cm, leave above 10cm. A changed contact
            # estimate must not strand a 16cm-away target in wrist-only control.
            manager.transition("APPROACH")
        if reduced and manager.stage in ("APPROACH","ALIGN"):
            manager.stage_age=0
        # Do not feed off-screen / cross-camera 2D EEF distances into progress.
        manager.observe(evidence,state,geometry=None,measured_progress=internal)
        if manager.index!=observed_goal_index:
            if self.inspector is not None and observed_kind=="pick":
                for arm in observed_arms:
                    if manager.hold_verified[arm]:self.inspector.remember(self.model,state,arm,self.hand_targets[arm] if bimanual else self.target)
            self.goal_changed=True
            self.target={"valid":False,"reason":"NEW_GOAL_REQUIRES_FRESH_SEMANTIC_OBSERVATION",
                         "observation_goal_index":observed_goal_index,"current_goal_index":manager.index}
            self.hand_targets={};manager.grounding=self.target
            manager.observation=None
            self.reposition=None;self.reposition_left=0;self.replan_needed=None
            self.previous=None
            self.progress["goal_transition_barrier"]={"from":observed_goal_index,"to":manager.index,
                "old_target_invalidated":True,"permitted_action":"HOLD_ONLY_BEFORE_NEW_GOAL_OBSERVATION"}
            manager.update_grasp_probe(state)
            return
        if (self.inspector is not None and not evidence.visible and any(manager.hold_verified.values())
                and manager.search_reference=="unknown"):
            manager.stop_reason="TARGET_REFERENCE_UNRESOLVED_WITH_HELD_LOAD"
        manager.update_grasp_probe(state)
        if self.reposition_left and self.reposition not in manager.palette():
            self.reposition_left=0  # a new stage may forbid the queued strategy
        manager.search_context=self.search.context()
        self.replan_needed=None
        if manager.stop_reason=="RECOVERY_BUDGET_EXHAUSTED":
            self.replan_needed=manager.stop_reason
        if not self.is_held_search and not evidence.visible and self.search.complete and self.reposition_left<=0:
            self.replan_needed="LOCAL_VIEW_SWEEP_COMPLETE_TARGET_NOT_FOUND"
        if self.replan_needed and manager.replans>=self.max_replans:
            manager.stop_reason="STRATEGY_REPLAN_BUDGET_EXHAUSTED"
            self.replan_needed=None
        self.previous={"target":self.target,"hand_targets":self.hand_targets,"centers":{a:p.copy() for a,p in self.centers.items()},
                       "distance":distance,"progress_distance":progress_distance,"goal_index":observed_goal_index}

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
        if action.frame == "tool" and action.part in ("left","right"):
            return self.model.forward(state.q,action.part)[:3,:3] @ vector
        if action.frame in VIEWS:
            # Exactly SafeServo's camera frame: forward into image, left/up in
            # image coordinates. This estimate never replaces its real IK test.
            frame = self.model.forward(state.q, "camera_" + action.frame)[:3, :3]
            return frame @ np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]) @ vector
        return None

    def _points_for_arms(self):
        contacts=self.target.get("hand_contacts")
        return {a:np.asarray(contacts[a]["point_base_m"] if contacts else self.target["point_base_m"])
                for a in self.harness.arms}

    def _expected_distances(self, action, state, trial=None):
        points=self._points_for_arms()
        centers=self.centers
        if action.part=="base":
            amount=action.amount(self.harness.carry)
            if action.move in TRANSLATIONS:
                points={a:p-np.asarray(TRANSLATIONS[action.move])*amount for a,p in points.items()}
            else:
                angle=-amount*(1 if action.move=="yaw_plus" else -1)
                c,s=math.cos(angle),math.sin(angle)
                R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                points={a:R@p for a,p in points.items()}
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
        return {a:float(np.linalg.norm(points[a]-centers[a])) for a in self.harness.arms}

    def _expected_point(self, action, state, trial=None):
        distances=self._expected_distances(action,state,trial)
        return max(distances.values()) if distances is not None else None

    def _navigation_geometry(self,action=None):
        point=np.asarray(self.target["point_base_m"],dtype=float).copy()
        if action is not None:
            amount=action.amount(self.harness.carry)
            if action.move in TRANSLATIONS:point-=np.asarray(TRANSLATIONS[action.move])*amount
            elif action.move in ("yaw_plus","yaw_minus"):
                angle=-amount*(1 if action.move=="yaw_plus" else -1)
                c,s=math.cos(angle),math.sin(angle)
                point=np.array([[c,-s,0],[s,c,0],[0,0,1]])@point
        return {"range_m":float(np.linalg.norm(point[:2])),
                "bearing_deg":math.degrees(math.atan2(point[1],point[0]))}

    def candidates(self,state):
        if self.goal_changed:return (HOLD,)
        palette=self.harness.palette()
        bimanual=self.harness.goal.kind=="pick" and self.harness.goal.hand=="both"
        limit=40 if bimanual else self.max_preflights
        if self.depth_guard is None:
            raise RuntimeError("Current RGB-D observation required before proposing motion")
        rows=[]; proposed=[]
        if self.target.get("valid") and self.harness.stage not in ("GRASP","RELEASE","VERIFY_GRASP","VERIFY_PLACE","VERIFY_SUPPORT","VERIFY_EFFECT"):
            ranked=[]
            before=self.target.get("mean_contact_distance_m",self.target["distance_to_active_closing_center_m"])
            for action in palette:
                if action.move in TRANSLATIONS and action.part in (*self.harness.arms,"both"):
                    distances=self._expected_distances(action,state)
                    if distances is not None:
                        ranked.append((before-float(np.mean(list(distances.values()))),action))
            ranked.sort(key=lambda x:x[0],reverse=True)
            if bimanual:
                # Reserve a useful direction for EACH hand before common moves
                # consume the finite list. Their errors can point opposite ways.
                first=[next((row for row in ranked if row[1].part==arm and row[0]>0),None)
                       for arm in self.harness.arms]
                ranked=[row for row in first if row is not None]+ranked
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
            close_probe=Action(self.harness.goal.hand,"close")
            if self.harness.grasp_probe.get("eligible") and close_probe in palette:
                proposed.insert(0,close_probe)
            rotations=[action for action in palette if action.move in
                       ("roll_plus","roll_minus","pitch_plus","pitch_minus","yaw_plus","yaw_minus")
                       and action.part in (*self.harness.arms,"both")]
            if bimanual:rotations.sort(key=lambda action:action.part=="both")
            for action in rotations:
                if len(proposed)<limit: proposed.append(action)
            if self.harness.stage in ("APPROACH","ALIGN","RECOVER"):
                for move in ("up","down"):
                    torso=Action("torso",move,"micro")
                    if torso in palette and len(proposed)<limit:proposed.append(torso)
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
            proposed=[a for a in palette if a.part=="base"]
        allowed=[]; started=time.perf_counter()
        queue=[HOLD,*dict.fromkeys(proposed[:limit])]
        tested=set()
        while queue and len(tested-{HOLD})<limit:
            action=queue.pop(0)
            if action in tested:continue
            tested.add(action)
            ok,reason=self.depth_guard.check(action,self.harness.carry)
            if self.approach_monitor is not None and not self.approach_monitor.allowed(action):
                ok,reason=False,"REPEATED_BASE_APPROACH_WITHOUT_CONTACT_PROGRESS"
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
                    distances=self._expected_distances(action,state,trial)
                    if distances is not None:
                        row["predicted_target_distance_m"]=round(max(distances.values()),5)
                        row["predicted_per_hand_distance_m"]={a:round(d,5) for a,d in distances.items()}
                        before=self.target.get("mean_contact_distance_m",self.target["distance_to_active_closing_center_m"])
                        row["predicted_distance_gain_m"]=round(before-float(np.mean(list(distances.values()))),5)
                        row["gain_objective"]="mean_active_contact_distance; stage gates use maximum"
            rows.append(row)
            if (not ok and action.part in (*self.harness.arms,"both")
                    and action.move in TRANSLATIONS and action.scale=="coarse"):
                # A rejected 3cm move does NOT imply that only 2mm is possible.
                # Try the existing 1cm command before its micro sibling; every
                # attempt still counts toward the unchanged preflight budget.
                middle=Action(action.part,action.move,"fine",action.frame)
                if middle in palette and middle not in tested:queue.insert(0,middle)
        navigation=None
        if self.target.get("valid") and self.harness.goal.kind=="navigate" and self.harness.stage=="APPROACH":
            navigation={"current":self._navigation_geometry(),"rule":"face_visible_destination_before_approaching",
                        "alignment_tolerance_deg":15.,"not_a_completion_or_collision_certificate":True}
            toward=[]
            for row in rows:
                action=Action(**row["action"])
                if action.part!="base":continue
                row["navigation_after"]=self._navigation_geometry(action)
                improvement=abs(navigation["current"]["bearing_deg"])-abs(row["navigation_after"]["bearing_deg"])
                if row["accepted"] and action.move in ("yaw_plus","yaw_minus") and improvement>.1:
                    toward.append(action)
            if abs(navigation["current"]["bearing_deg"])>15. and toward:
                allowed=[a for a in allowed if a==HOLD or a in toward]
                navigation["phase"]="FACE_TARGET_FIRST"
            else:
                # If a turn is blocked, keep depth-checked repositioning options.
                # A scalar range is never allowed to declare navigation success.
                navigation["phase"]="APPROACH_OR_SAFE_REPOSITION"
            for row in rows:row["offered_to_policy"]=Action(**row["action"]) in allowed
        self.harness.candidate_receipt={"state_q":state.q.tolist(),"command_grip_latch":self.servo.grips.tolist(),
            "tested":rows,"preflight_s":time.perf_counter()-started,"max_nonhold_preflights":limit,
            "depth_guard":self.depth_guard.receipt(),"fresh_execution_recheck_required":True,
            "navigation":navigation}
        if not allowed:
            self.harness.stop_reason="NO_SAFE_ACTION_AT_CURRENT_STATE"
        return tuple(allowed)

    @property
    def is_held_search(self):
        h=self.harness
        return bool(self.inspector is not None and h.search_reference.startswith("held_") and
                    h.stage in ("SEARCH","RECOVER") and h.observation and not h.observation.visible)

    def inspection_candidates(self,state):
        if not self.is_held_search:raise ValueError("No current held-object search")
        allowed,receipt=self.inspector.candidates(self.model,state,self.harness,self.servo,self.depth_guard)
        self.harness.candidate_receipt=receipt
        return allowed

    def verification_action(self,allowed):
        """A finite sensing action must match its verifier's measurement scale.

        The old VLM could choose a 2mm pulse that could not satisfy the >=3mm
        measured-motion gate. Never lower that gate or invent evidence; use
        one already-preflighted 1cm synchronized lift per fresh observation.
        The original four-observation window bounds this to three probes.
        """
        h=self.harness
        if self.grasp_verifier is None or h.goal.kind!="pick" or h.stage!="VERIFY_GRASP":
            raise ValueError("Registered grasp verification is not active")
        obs=h.observation
        observable=(obs is not None and obs.visible and self.target.get("valid") and
                    obs.enclosed is not False and obs.co_moving is not False and obs.hazard=="none")
        action=Action(h.goal.hand,"up","fine")
        reason=("VERIFICATION_TARGET_UNOBSERVABLE_OR_CONTRADICTED" if not observable else
                "NO_SAFE_OBSERVABLE_GRASP_LIFT" if action not in allowed else None)
        if reason is not None:
            h.recover(reason)
            # Even with a malformed missing latch, no unobservable motion is
            # authorized by this sensing routine. This is a stop, not success.
            if h.stop_reason is None:h.stop_reason=reason
            return HOLD,{"source":"registered_grasp_sensing_controller","reason":reason,"success_claim":False}
        h.authorize(action)
        return action,{"source":"registered_grasp_sensing_controller","reason":"PRECHECKED_1CM_MEASURABLE_LIFT",
            "max_probes_in_original_window":3,"commanded_lift_m":action.amount(h.carry),
            "measurement_gate_unchanged":True,"no_model_action_call":True,"success_claim":False}

    def search_action(self,state):
        """One finite pulse, not a hidden multi-step macro or VLM fiction."""
        if self.goal_changed:return HOLD,{"source":"goal_transition_barrier","reason":"OBSERVE_NEW_GOAL_FIRST"}
        if self.is_held_search:raise ValueError("Held affordance cannot use world-heading search")
        if self.reposition is not None and self.reposition_left>0:
            action,reason=self.reposition,"BOUNDED_REPOSITION_WITH_CURRENT_DEPTH"
        else:
            action,reason=self.search.propose()
        if action is None:
            self.harness.stop_reason=reason
            return HOLD,{"source":"measured_search_controller","reason":reason}
        self.harness.authorize(action)
        if self.approach_monitor is not None and not self.approach_monitor.allowed(action):
            self.reposition_left=0
            self.harness.recover("REPEATED_BASE_APPROACH_WITHOUT_CONTACT_PROGRESS")
            return HOLD,{"source":"observed_progress_veto","reason":"REPOSITION_DIRECTION_STILL_BLOCKED"}
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
        if self.is_held_search:self.inspector.executed(self.harness)
        exploratory=self.harness.stage in ("SEARCH","RECOVER") and not self.goal_changed
        if self.motion is None:self.search.executed(feedback,exploratory=exploratory)
        else:
            if self.pending_motion is not None:raise ValueError("An executed motion has not been observed")
            self.pending_motion=copy.deepcopy(feedback)
            self.pending_exploratory=exploratory
        if action==self.reposition and feedback["status"]=="TARGET_REACHED":
            self.reposition_left-=1
        self.harness.executed(action,feedback)
