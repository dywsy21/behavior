"""Sensor-grounded finite task/search/recovery orchestration, not task scripts."""
import math
import time
import copy
from dataclasses import asdict, replace

import numpy as np

from .grounding import LocalDepthGuard, localize_target, observed_cloud, target_self_veto_reason
from .harness import TaskHarness
from .protocol import Action, HOLD, TRANSLATIONS, ROTATIONS, VIEWS, strict_json
from .search import CoverageSearch
from .servo import SafeServo, execution_completed
from .bimanual import localize_hand_contacts, all_claims
from .navigation import navigation_workspace_check
from .motion_feedback import observed_motion_feedback


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
    def __init__(self, goals, *, active_grasp_probe=False,contact_geometry=True,held_inspection=False,reference_from_planner=False,inspection_budget_aware=False,multicamera_inspection=False,approach_reorientation=False,approach_body_options=False,workspace_posture=False,approach_translation_preview=False,near_pose_gap=False):
        super().__init__(goals)
        self.grounding={}
        self.search_context={}
        self.replans=0
        self.replan_history=[]
        self.candidate_receipt={}
        self.motion_receipt={}
        self.pending_grasp={"left":False,"right":False}
        # Distinct from PICK verification: closing on a door/container handle
        # may also leave a contact/load after its semantic goal is finished.
        self.possible_contact_after_close={"left":False,"right":False}
        self.last_gripper=None
        self.contact_geometry=bool(contact_geometry)
        self.approach_reorientation=bool(approach_reorientation)
        self.approach_translation_preview=bool(approach_translation_preview)
        if self.approach_translation_preview and not self.approach_reorientation:
            raise ValueError("Translation preview requires the unloaded approach eligibility contract")
        self.approach_body_options=bool(approach_body_options)
        self.workspace_posture=bool(workspace_posture)
        self.near_pose_gap=bool(near_pose_gap)
        if self.near_pose_gap and not self.approach_reorientation:
            raise ValueError("Near pose gap requires explicit approach reorientation")
        self.workspace_close_seen={"left":False,"right":False}
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
        self.target_geometry_veto=None

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
        self.last_gripper=np.asarray(state.gripper,dtype=float).copy()
        veto=target_self_veto_reason(self.grounding)
        self.target_geometry_veto=None
        if veto:
            # Preserve the exact model claim for audit, but never use a robot
            # self point to confirm enclosure, support, motion, or task effect.
            self.target_geometry_veto={"reason":veto,"raw_evidence":asdict(evidence),
                "semantic_claims_suppressed":True,"requires_fresh_observation":True}
            fields=dict(visible=False,view="none",target_uv=None,enclosed=None,
                        co_moving=None,supported=None,effect=None)
            for key,value in (("other_views",()),("hand_contacts",()),("target_reference","unknown")):
                if hasattr(evidence,key):fields[key]=value
            self.confirmations=0
            self.grasp_probe={**self.grasp_probe,"eligible":False,"reason":veto}
            # Still process physical loss/hazard/budgets through the usual path.
            result=super().observe(replace(evidence,**fields),state,geometry,measured_progress)
            self.events.append({"event":"TARGET_GEOMETRY_VETO","reason":veto,"goal":self.index})
            return result
        self.resolve_reference(evidence)
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

    def issued(self,command):
        """Record actual robot commands, including partial/failed execution.

        Call at the env.step boundary, not when proposing or beginning an
        action. A zero-tick rejected arm motion can still issue a closing HOLD.
        This irreversible history is not contact or holding evidence.
        """
        value=np.asarray(command,dtype=float)
        if value.shape!=(23,) or not np.isfinite(value).all():
            raise ValueError("Issued robot command must be finite 23D")
        grips=value[[14,22]]
        if np.any(np.abs(grips)>1):
            raise ValueError("Issued gripper commands must lie in [-1,1]")
        for arm,grip in zip(("left","right"),grips):
            if grip<.999:self.workspace_close_seen[arm]=True

    def executed(self,action,feedback):
        probe=(self.grasp_probe.get("eligible") and self.stage=="ALIGN" and
               action==Action(self.goal.hand,"close"))
        if probe and (feedback.get("control_ticks",0)>0 or execution_completed(action,feedback)):
            self.grasp_probe_attempts[self.index]=self.grasp_probe_attempts.get(self.index,0)+1
        widths=feedback.get("finger_mean_m")
        self.last_gripper=np.asarray(widths,dtype=float).copy() if widths is not None else None
        # Even an interrupted close may have made contact. Keep the latch until
        # an explicit OPEN completes or grasp evidence really verifies holding.
        if action.move=="close" and (feedback.get("control_ticks",0)>0 or execution_completed(action,feedback)):
            for arm in (("left","right") if action.part=="both" else (action.part,)):
                if arm in self.workspace_close_seen:self.workspace_close_seen[arm]=True
                if arm in self.possible_contact_after_close:self.possible_contact_after_close[arm]=True
                if arm in self.pending_grasp and self.goal.kind=="pick":self.pending_grasp[arm]=True
        super().executed(action,feedback)
        if execution_completed(action,feedback) and action.move in ("close","open"):
            arms=("left","right") if action.part=="both" else (action.part,)
            for arm in arms:
                if arm in self.pending_grasp:
                    if action.move=="open":self.pending_grasp[arm]=False
                    elif self.goal.kind=="pick":self.pending_grasp[arm]=True
                if action.move=="open" and arm in feedback.get("gripper_open_at_calibrated_aperture",[]):
                    self.possible_contact_after_close[arm]=False
        if probe and execution_completed(action,feedback):
            self.events.append({"event":"UNVERIFIED_GRASP_PROBE","goal":self.index,
                                "attempt":self.grasp_probe_attempts[self.index],"success_claim":False})
            self.transition("VERIFY_GRASP")
        self.grasp_probe={**self.grasp_probe,"eligible":False}

    def palette(self):
        if target_self_veto_reason(self.grounding):return (HOLD,)
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
        if self.workspace_posture and self.goal.kind=="pick" and self.goal.hand in ("left","right") and self.stage in ("APPROACH","ALIGN"):
            # Tentative palette only: current open/history/depth/servo and a
            # useful bounded followup are required before offering any of these.
            palette+=tuple(Action("torso",move,"fine") for move in ("up","down","forward","back"))
        from .approach_reorientation import eligible
        if eligible(self):
            palette += tuple(Action(self.goal.hand, move, scale, "tool")
                             for move in ROTATIONS for scale in ("coarse", "fine"))
        from .near_pose_gap import eligible as near_pose_eligible
        if near_pose_eligible(self):
            palette += tuple(Action(self.goal.hand, move, scale, "tool")
                             for move in ROTATIONS for scale in ("micro", "fine"))
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
        if self.held_inspection_enabled and self.search_reference.startswith("held_"):
            reference=self.search_reference.removeprefix("held_")
            if not self.hold_verified.get(reference):return (HOLD,)
            # A rigidly held target moves with the body. Base translation/yaw
            # cannot bring the working hand closer to it in this same frame.
            palette=tuple(a for a in palette if a.part!="base")
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
        context["target_geometry_veto"]=self.target_geometry_veto
        if self.multicamera_inspection:context["possible_contact_after_any_close"]=self.possible_contact_after_close.copy()
        if hasattr(self,"approach_progress"):context["approach_progress"]=self.approach_progress
        if hasattr(self,"search_reanchor"):context["search_reanchor"]=self.search_reanchor
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

    def __init__(self, model, servo, harness, visual_odometry=False,grasp_motion=False,odometry_estimator="pnp",approach_progress=False,persistent_grasp_tracks=False,spatial_grasp_features=False,search_motion_recovery=False,odometry_self_exclusion=False,odometry_match_refinement=False,near_contact_review=False):
        self.model,self.servo,self.harness=model,servo,harness
        if type(near_contact_review) is not bool or (near_contact_review and not visual_odometry):
            raise ValueError("Contact review requires explicit measured visual motion")
        self.near_contact_review=near_contact_review
        if odometry_self_exclusion and not visual_odometry:
            raise ValueError("Robot-self exclusion requires visual odometry")
        self.odometry_self_exclusion=odometry_self_exclusion
        if (type(odometry_match_refinement) is not bool or
                (odometry_match_refinement and not (visual_odometry and odometry_self_exclusion and odometry_estimator=="rgbd_joint"))):
            raise ValueError("Match refinement requires explicit self-excluded joint RGB-D")
        self.odometry_match_refinement=odometry_match_refinement
        if approach_progress and not visual_odometry:raise ValueError("Approach progress requires measured visual motion")
        from .approach_progress import ApproachProgress
        self.approach_monitor=ApproachProgress() if approach_progress else None
        self.search=CoverageSearch()
        self.previous=None
        self.target={"valid":False}
        self.hand_targets={}
        self.centers={}
        self.depth_guard=None
        self.near_pose_geometry=None
        self.reposition=None
        self.reposition_left=0
        self.replan_needed=None
        self.progress={}
        self.motion=None
        self.pending_motion=None
        self.pending_exploratory=True
        self.goal_changed=False
        self.search_recovery=None
        if search_motion_recovery:
            if not visual_odometry or odometry_estimator!="rgbd_joint":
                raise ValueError("Search reanchor requires unchanged joint RGB-D motion")
            from .search_reanchor import SearchReanchor
            self.search_recovery=SearchReanchor()
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
            self.motion=RGBDMotion(odometry_estimator,exclude_robot=odometry_self_exclusion,
                                  refine_matches=odometry_match_refinement)

    @property
    def can_replan_stop(self):
        return (self.harness.stop_reason == "RECOVERY_BUDGET_EXHAUSTED"
                and self.harness.replans < self.max_replans)

    def update_motion(self,images,depths,state, *, robot_frame=None, control=None):
        if self.motion is None:raise ValueError("Visual motion was not enabled")
        kwargs=(dict(robot_frame=robot_frame,gripper=state.gripper,control=control) if self.odometry_self_exclusion else {})
        if self.odometry_self_exclusion and state.finger_qpos is not None:
            kwargs["finger_qpos"]=state.finger_qpos
        if not self.odometry_self_exclusion and (robot_frame is not None or control is not None):
            raise ValueError("Self frame supplied to disabled robot exclusion")
        receipt=self.motion.observe(images,depths,self.model,state.q,**kwargs)
        self.harness.motion_receipt=receipt
        if not receipt["valid"]:
            # Never turn an unreliable velocity integral into new coverage.
            self.harness.stop_reason="VISUAL_ODOMETRY_UNCERTAIN"
            return receipt
        if receipt.get("initial"):
            if self.pending_motion is not None:raise ValueError("Missing pre-action visual reference")
            return receipt
        if self.pending_motion is None:raise ValueError("No executed action matches this visual motion")
        feedback=observed_motion_feedback(self.harness.last_action,self.pending_motion,receipt)
        self.search.executed(feedback,exploratory=self.pending_exploratory)
        self.harness.feedback=feedback
        if self.harness.history:self.harness.history[-1]={**self.harness.history[-1],"feedback":feedback}
        self.harness.search_context=self.search.context()
        self.pending_motion=None
        return receipt

    def observe(self,evidence,state,depths,depth_receipt,images=None,self_geometry=None,contact_review_receipt=None):
        manager=self.harness
        self.goal_changed=False
        observed_goal_index=manager.index
        observed_kind,observed_arms=manager.goal.kind,manager.arms
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
            if self.near_contact_review:
                from .contact_review import apply_review
                self.target=apply_review(self.target,contact_review_receipt,manager,state,evidence,images)
            elif contact_review_receipt is not None:
                raise ValueError("Contact review receipt supplied to disabled controller")
        self.centers=self.model.grasp_centers(state.q)
        manager.grounding=self.target
        if target_self_veto_reason(self.target) is None:manager.resolve_reference(evidence)
        camera=self.model.spec["metadata"]["cameras"]["head"]
        target_bearing=(math.atan2(self.target["point_base_m"][1],self.target["point_base_m"][0])
                        if self.target["valid"] else None)
        new_coverage=self.search.observe(manager.index,self.model.forward(state.q,"camera_head"),camera["K"],
                                        camera["width"],depth_receipt["head"]["valid_fraction"],evidence.visible,target_bearing)
        precise_geometry = self.servo.limits.robot_geometry_guards
        if precise_geometry and self_geometry is None:
            raise ValueError("Fresh robot-only geometry required by enabled guard")
        self.depth_guard=LocalDepthGuard(observed_cloud(depths,self.model,state.q),self.model,state.q,depths,
                                        self_geometry=self_geometry if precise_geometry else None)
        if manager.near_pose_gap:self.near_pose_geometry=copy.deepcopy(self_geometry)
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
            if not context["valid"]:manager.stop_reason=manager.stop_reason or "NO_VERIFIED_HELD_ANCHOR"
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
        reference=None
        if self.harness.held_inspection_enabled:
            scope=self.harness.search_reference
            if scope=="unknown":return None  # never silently default to world geometry
            if scope.startswith("held_"):
                reference=scope.removeprefix("held_")
                if not self.harness.hold_verified.get(reference):return None
                current=self.model.forward(state.q,reference)
                predicted=current.copy()
                if trial is not None and trial.joint_plan is not None:
                    predicted=self.model.forward(trial.joint_plan[-1],reference)
                elif action.move in TRANSLATIONS and action.part in (reference,"both"):
                    direction=self._translation_direction(action,state)
                    if direction is None:return None
                    predicted[:3,3]+=direction*action.amount(self.harness.carry)
                # CURRENT observed contact (e.g. the button), not the old
                # inspection object's surface anchor, follows the held frame.
                points={a:predicted[:3,:3]@(current[:3,:3].T@(p-current[:3,3]))+predicted[:3,3]
                        for a,p in points.items()}
        if action.part=="base":
            if reference is None:
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
        elif reference is None or action.part not in (reference,"torso","all"):
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

    def candidates(self,state,deadline=None):
        from .wall_budget import require_time
        require_time(deadline)
        if self.goal_changed:return (HOLD,)
        veto=target_self_veto_reason(self.target)
        if veto:
            self.harness.candidate_receipt={"reason":veto,"requires_fresh_observation":True,
                "tested":[{"action":asdict(HOLD),"accepted":True,"reason":"TARGET_GEOMETRY_VETO"}]}
            return (HOLD,)
        if self.near_contact_review and self.target.get("reason")=="VISIBLE_TARGET_CONTACT_UNCONFIRMED":
            self.harness.candidate_receipt={"reason":"CONTACT_REVIEW_ABSTAINED",
                "near_contact_review":self.target["near_contact_review"],
                "tested":[{"action":asdict(HOLD),"accepted":True,"reason":"NO_POINT_GUIDED_MOTION"}]}
            return (HOLD,)
        palette=self.harness.palette()
        from .approach_reorientation import eligible, preview, unladen_pick_approach
        open_command=bool(np.all((self.servo.grips >= .999) & (self.servo.grips <= 1.)) and np.all(state.gripper >= .0495))
        posture_mode=eligible(self.harness) and open_command
        translation_mode=self.harness.approach_translation_preview and posture_mode and self.servo.limits.robot_geometry_guards
        body_mode=self.harness.approach_body_options and unladen_pick_approach(self.harness) and open_command
        from .workspace_posture import qualification as workspace_qualification, preview as workspace_preview
        workspace=workspace_qualification(self.harness,state,self.servo.grips,self.model) if self.harness.workspace_posture else None
        if workspace is not None:
            workspace["checks"]["robot_geometry_guards"]=bool(self.servo.limits.robot_geometry_guards)
            workspace["eligible"]=all(workspace["checks"].values())
        workspace_mode=bool(workspace and workspace["eligible"])
        if self.harness.approach_reorientation and self.harness.stage=="APPROACH" and not posture_mode:
            # A measured open jaw cannot override a pending CLOSE command.
            palette=tuple(a for a in palette if not (a.part in ("left","right","both") and a.move in ROTATIONS))
        bimanual=self.harness.goal.kind=="pick" and self.harness.goal.hand=="both"
        limit=40 if bimanual else 32 if posture_mode or body_mode else self.max_preflights
        if self.depth_guard is None:
            raise RuntimeError("Current RGB-D observation required before proposing motion")
        rows=[]; proposed=[]; ranked=[]; rotation_trials=[]; translation_trials=[]
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
            if body_mode:
                # Keep alternate directions, INCLUDING turns, until they have
                # passed the current depth/servo checks. Pre-pruning to one
                # predicted best translation hid every feasible alternative
                # when that translation was blocked. Negative predicted gains
                # are not filtered: a short reposition may be necessary.
                body=[(self._expected_point(a,state),a) for a in palette
                      if a.part=="base" and a.scale=="fine"]
                proposed.extend(a for d,a in sorted(body,key=lambda x:float("inf") if x[0] is None else x[0]))
            else:
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
            if posture_mode:
                # One direction per slot. A rejected existing 8deg primitive
                # gets its existing 3deg fallback below; no amplitude changes.
                rotations=[action for action in rotations if action.scale=="coarse"]
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
            require_time(deadline)
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
            require_time(deadline)
            row={"action":asdict(action),"accepted":bool(ok),"reason":reason}
            if ok:
                allowed.append(action)
                if posture_mode and action.part==self.harness.goal.hand and action.move in ROTATIONS:
                    rotation_trials.append((action,trial))
                row["planned_ticks"]=trial.total_ticks
                if self.target.get("valid"):
                    distances=self._expected_distances(action,state,trial)
                    if distances is not None:
                        row["predicted_target_distance_m"]=round(max(distances.values()),5)
                        row["predicted_per_hand_distance_m"]={a:round(d,5) for a,d in distances.items()}
                        before=self.target.get("mean_contact_distance_m",self.target["distance_to_active_closing_center_m"])
                        row["predicted_distance_gain_m"]=round(before-float(np.mean(list(distances.values()))),5)
                        row["gain_objective"]="mean_active_contact_distance; stage gates use maximum"
                if (translation_mode and action.part==self.harness.goal.hand and action.move in TRANSLATIONS
                        and action.scale=="fine" and row.get("predicted_distance_gain_m",0)>0):
                    translation_trials.append((action,trial,row["predicted_distance_gain_m"]))
            rows.append(row)
            if (not ok and action.part in (*self.harness.arms,"both")
                    and (action.move in TRANSLATIONS or (posture_mode and action.move in ROTATIONS)) and action.scale=="coarse"):
                # A rejected 3cm move does NOT imply that only 2mm is possible.
                # Try the existing 1cm command before its micro sibling; every
                # attempt still counts toward the unchanged preflight budget.
                middle=Action(action.part,action.move,"fine",action.frame)
                if middle in palette and middle not in tested:queue.insert(0,middle)
            if not ok and body_mode and action.part=="base" and action.scale=="fine":
                # A shorter *existing* pulse still requires its own full
                # preflight. Append so every fine direction is checked first.
                micro=Action("base",action.move,"micro")
                if micro in palette and micro not in tested:queue.append(micro)
        workspace_trials=[]; workspace_lookahead=None
        if workspace_mode and self.target.get("valid"):
            moderate=[]
            for row in rows:
                a=Action(**row["action"])
                if a.part!=self.harness.goal.hand or a.move not in TRANSLATIONS or a.scale not in ("fine","coarse"):continue
                distances=self._expected_distances(a,state)
                if distances is None:continue
                gain=self.target["distance_to_active_closing_center_m"]-float(np.mean(list(distances.values())))
                if gain>1e-6:moderate.append((a,row,gain))
            # Only a currently observed workspace blockage warrants extra body
            # preflights. 2mm progress is retained but is not a moderate reach.
            blocked=bool(moderate and not any(row["accepted"] for a,row,gain in moderate))
            followups=[];directions=[]
            for a,row,gain in sorted(moderate,key=lambda v:(v[0].scale!="fine",-v[2])):
                if row["accepted"] or row["reason"] not in ("UNREACHABLE_OR_COLLISION_BLOCKED","DURATION_LIMIT_EXCEEDED","CARTESIAN_PATH_DEVIATION","TRAJECTORY_LIMIT_OR_COLLISION"):continue
                direction=self._translation_direction(a,state)
                if any(float(direction@old)>.995 for old in directions):continue
                followups.append(a);directions.append(direction)
                if len(followups)==3:break
            if blocked and followups:
                for move in ("up","down","forward","back"):
                    require_time(deadline)
                    action=Action("torso",move,"fine")
                    ok,reason=self.depth_guard.check(action,False)
                    trial=None
                    if ok:
                        trial=SafeServo(self.model,state,self.servo.grips.copy(),self.servo.limits)
                        ok=trial.begin(action,state,False)
                        reason="KINEMATIC_PATH_FEASIBLE" if ok else trial.status
                    require_time(deadline)
                    row={"action":asdict(action),"accepted":bool(ok),"reason":reason,"offered_to_policy":False}
                    if ok:
                        row["planned_ticks"]=trial.total_ticks
                        workspace_trials.append((action,trial))
                    rows.append(row)
                workspace_lookahead=workspace_preview(self.model,state,self.servo.grips.copy(),self.servo.limits,
                    self.harness.goal.hand,self.target["point_base_m"],workspace_trials,followups,deadline)
                for row in rows:
                    found=next((r for r in workspace_lookahead["rows"] if r["torso"]==row["action"]),None)
                    if found is None:continue
                    row["workspace_posture_after"]=found["summary"]
                    gain=found["summary"]["best_two_command_gain_m"]
                    if gain is not None and gain>1e-3:
                        allowed.append(Action(**row["action"]))
                        row["offered_to_policy"]=True
            workspace.update(trigger="MODERATE_IMPROVING_ARM_PATHS_BLOCKED" if blocked else "NO_MODERATE_WORKSPACE_BLOCKAGE",
                             blocked_followups=[asdict(a) for a in followups],preview=workspace_lookahead)
        reorientation=None
        if (posture_mode and rotation_trials and self.target.get("valid")
                and not any(r["accepted"] and r["action"]["part"]==self.harness.goal.hand
                            and r["action"]["move"] in TRANSLATIONS and r["action"]["scale"]=="coarse" for r in rows)):
            future=[]; directions=[]
            for gain,action in ranked:
                if gain<=0 or action.scale!="coarse" or action.part!=self.harness.goal.hand:continue
                direction=self._translation_direction(action,state)
                if any(float(direction@old)>.995 for old in directions):continue
                future.append(action);directions.append(direction)
                if len(future)==3:break
            if future:
                reorientation=preview(self.model,state,self.servo.grips.copy(),self.servo.limits,
                    self.harness.goal.hand,self.target["point_base_m"],rotation_trials,future,deadline=deadline)
                for row in rows:
                    match=next((r for r in reorientation["rows"] if r["rotation"]==row["action"]),None)
                    if match is not None:row["reorientation_after"]=match["summary"]
        translation_lookahead=None
        if (translation_mode and translation_trials and self.target.get("valid")
                and not any(r["accepted"] and r["action"]["part"]==self.harness.goal.hand
                            and r["action"]["move"] in TRANSLATIONS and r["action"]["scale"]=="coarse" for r in rows)):
            from .approach_translation import preview as translation_preview, ROBOT_PATH_REJECTIONS
            blocked={Action(**r["action"]) for r in rows if not r["accepted"] and r["reason"] in ROBOT_PATH_REJECTIONS}
            followup=next((a for gain,a in ranked if gain>0 and a.part==self.harness.goal.hand
                           and a.scale=="coarse" and a in blocked),None)
            if followup is not None:
                first=[];directions=[]
                for action,trial,gain in sorted(translation_trials,key=lambda v:-v[2]):
                    direction=self._translation_direction(action,state)
                    if any(float(direction@old)>.995 for old in directions):continue
                    first.append((action,trial));directions.append(direction)
                    if len(first)==3:break
                translation_lookahead=translation_preview(self.model,state,self.servo.grips.copy(),self.servo.limits,
                    self.harness.goal.hand,self.target["point_base_m"],first,followup,deadline)
                for row in rows:
                    match=next((r for r in translation_lookahead["rows"] if r["translation"]==row["action"]),None)
                    if match is not None:row["translation_after"]=match["summary"]
        near_pose=None;near_additional=0
        if self.harness.near_pose_gap:
            from .near_pose_gap import qualification as near_qualification
            near_pose=near_qualification(self.harness,state,self.servo.grips,self.model,self.servo.limits)
            gains={a:gain for gain,a in ranked}
            improving=[r for r in rows if r["action"]["part"]==self.harness.goal.hand
                       and r["action"]["move"] in TRANSLATIONS and gains.get(Action(**r["action"]),0)>1e-6]
            blocked=bool(improving and not any(r["accepted"] for r in improving))
            near_pose.update(trigger="ALL_IMPROVING_ARM_TRANSLATIONS_BLOCKED" if blocked else "NO_FULL_ARM_BLOCKAGE",
                             max_additional_preflights=15,additional_preflights=0)
            if near_pose["eligible"] and blocked:
                from .arm_observation_guard import ObservingArmGuard
                observed=observed_cloud(self.depth_guard.depth_images,self.model,state.q,stride=6)
                guard=ObservingArmGuard(self.model,state.q,observed,self.near_pose_geometry,self.harness.goal.hand)
                extra=[a for a in self.harness.palette() if a.part==self.harness.goal.hand
                       and a.move in ROTATIONS and a.scale in ("micro","fine") and a.frame=="tool"]
                retreat=[];directions=[]
                for gain,a in sorted(ranked,key=lambda v:v[0]):
                    if gain>=0 or a.part!=self.harness.goal.hand or a.scale!="fine" or a in tested:continue
                    direction=self._translation_direction(a,state)
                    if any(float(direction@old)>.995 for old in directions):continue
                    retreat.append(a);directions.append(direction)
                    if len(retreat)==3:break
                extra=list(dict.fromkeys(extra+retreat))
                if len(extra)>15:raise ValueError("Near pose option budget exceeded")
                for action in extra:
                    if action in tested:continue
                    require_time(deadline);near_additional+=1;tested.add(action)
                    ok,reason=self.depth_guard.check(action,False)
                    trial=None;sweep={"safe":False,"reason":"CURRENT_PREFLIGHT_REJECTED"}
                    if ok:
                        trial=SafeServo(self.model,state,self.servo.grips.copy(),self.servo.limits)
                        ok=trial.begin(action,state,False)
                        reason="KINEMATIC_PATH_FEASIBLE" if ok else trial.status
                    if ok:
                        safe,check=guard.check(trial.joint_plan)
                        sweep={"safe":bool(safe),**check};ok=bool(safe)
                        if not ok:reason=check["reason"]
                    require_time(deadline)
                    row={"action":asdict(action),"accepted":bool(ok),"reason":reason,
                         "near_pose_gap_option":True,"near_pose_sweep":sweep}
                    if ok:
                        allowed.append(action);row["planned_ticks"]=trial.total_ticks
                        distances=self._expected_distances(action,state,trial)
                        if distances is not None:
                            row["predicted_per_hand_distance_m"]={a:round(d,5) for a,d in distances.items()}
                            row["predicted_distance_gain_m"]=round(self.target["distance_to_active_closing_center_m"]-max(distances.values()),5)
                    rows.append(row)
                near_pose["additional_preflights"]=near_additional
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
            "tested":rows,"preflight_s":time.perf_counter()-started,"max_nonhold_preflights":limit+(4 if workspace_mode else 0)+near_additional,
            "depth_guard":self.depth_guard.receipt(),"fresh_execution_recheck_required":True,
            "navigation":navigation}
        if self.harness.approach_reorientation:
            self.harness.candidate_receipt["approach_reorientation"]={"eligible":bool(posture_mode),"preview":reorientation}
        if self.harness.approach_translation_preview:
            self.harness.candidate_receipt["approach_translation_preview"]={"eligible":bool(translation_mode),"preview":translation_lookahead}
        if self.harness.approach_body_options:
            self.harness.candidate_receipt["approach_body_options"]={"eligible":bool(body_mode),
                "all_existing_fine_body_directions_before_rejected_micro_fallback":True,
                "all_options_require_current_depth_servo_and_progress_checks":True,
                "prediction_not_execution_or_complete_environment_safety":True}
        if self.harness.workspace_posture:
            self.harness.candidate_receipt["workspace_posture"]=workspace
        if self.harness.near_pose_gap:
            self.harness.candidate_receipt["near_pose_gap"]=near_pose
        if not allowed:
            self.harness.stop_reason="NO_SAFE_ACTION_AT_CURRENT_STATE"
        return tuple(allowed)

    @property
    def is_held_search(self):
        h=self.harness
        return bool(self.inspector is not None and h.search_reference.startswith("held_") and
                    h.stage in ("SEARCH","RECOVER") and h.observation and not h.observation.visible)

    def inspection_candidates(self,state):
        if target_self_veto_reason(self.target):return self.candidates(state)
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
        veto=target_self_veto_reason(self.target)
        if veto:return HOLD,{"source":"target_geometry_veto","reason":veto,"success_claim":False}
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
        veto=target_self_veto_reason(self.target)
        if veto:return HOLD,{"source":"target_geometry_veto","reason":veto,"requires_fresh_observation":True}
        if self.near_contact_review and self.target.get("reason")=="VISIBLE_TARGET_CONTACT_UNCONFIRMED":
            return HOLD,{"source":"near_contact_review_veto","reason":"CONTACT_REVIEW_ABSTAINED"}
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
        if self.search_recovery is not None and action.move=="close":
            self.search_recovery.ever_closed=True
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
