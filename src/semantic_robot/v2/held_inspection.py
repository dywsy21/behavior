"""Bounded relative-view inspection of an observationally verified held object.

The stored point is a prior onboard RGB-D surface anchor in hand coordinates,
NOT the current affordance, an object pose, segmentation or simulator state.
"""
from dataclasses import asdict
import numpy as np
from scipy.spatial.transform import Rotation
from .protocol import Action,HOLD,TRANSLATIONS,ROTATIONS
from .servo import SafeServo
from .vision import project


def inspection_palette(hand,level=False):
    moves=[Action(hand,move,"fine") for move in TRANSLATIONS]
    if not level:moves += [Action(hand,move,"fine","tool") for move in ROTATIONS]
    return (HOLD,*moves)


class HeldInspection:
    max_attempts=24
    max_path_m=.20
    max_rotation_rad=np.deg2rad(60.)

    def __init__(self):
        self.anchors={};self.states={}

    def remember(self,model,state,arm,target):
        if not target.get("valid"):return
        t=model.forward(state.q,arm);p=np.asarray(target["point_base_m"])
        if not np.isfinite(p).all():raise ValueError("Finite observed anchor required")
        self.anchors[arm]={"point_hand_m":(t[:3,:3].T@(p-t[:3,3])).tolist(),
                          "source":"previous_verified_onboard_RGBD_surface_not_current_affordance"}

    def observe(self,model,state,harness):
        arm=harness.search_reference.removeprefix("held_")
        if arm not in ("left","right") or not harness.hold_verified[arm] or arm not in self.anchors:
            return False,{"valid":False,"reason":"NO_VERIFIED_HELD_ANCHOR"}
        relative=np.linalg.inv(model.forward(state.q,"camera_head"))@model.forward(state.q,arm)
        row=self.states.setdefault(harness.index,{"arm":arm,"last":relative.copy(),"novel":relative.copy(),
            "path_m":0.,"rotation_rad":0.,"attempts":0})
        if row["arm"]!=arm:raise ValueError("Inspection reference changed within one goal")
        row["path_m"]+=float(np.linalg.norm(relative[:3,3]-row["last"][:3,3]))
        row["rotation_rad"]+=float(np.linalg.norm(Rotation.from_matrix(relative[:3,:3]@row["last"][:3,:3].T).as_rotvec()))
        row["last"]=relative.copy()
        fresh=(np.linalg.norm(relative[:3,3]-row["novel"][:3,3])>=.008 or
               np.linalg.norm(Rotation.from_matrix(relative[:3,:3]@row["novel"][:3,:3].T).as_rotvec())>=np.deg2rad(2.))
        if fresh:row["novel"]=relative.copy()
        return bool(fresh),self.context(model,state,harness)

    def view(self,model,q,arm):
        t=model.forward(q,arm);point=t[:3,:3]@np.asarray(self.anchors[arm]["point_hand_m"])+t[:3,3]
        camera=model.spec["metadata"]["cameras"]["head"]
        uv=project(point,model.forward(q,"camera_head"),camera["K"])
        normalized=None if uv is None else (uv/[camera["width"]-1,camera["height"]-1]).tolist()
        return {"prior_held_anchor_head_uv":normalized,"not_current_affordance_detection":True,
                "framing_cost":None if normalized is None else float(np.linalg.norm(np.asarray(normalized)-[.5,.55]))}

    def context(self,model,state,harness):
        row=self.states.get(harness.index);arm=harness.search_reference.removeprefix("held_")
        if row is None or arm not in self.anchors:return {"valid":False,"reason":"NO_VERIFIED_HELD_ANCHOR"}
        return {"valid":True,"reference_hand":arm,"attempts":row["attempts"],"max_attempts":self.max_attempts,
            "relative_path_m":row["path_m"],"relative_rotation_deg":float(np.rad2deg(row["rotation_rad"])),
            "max_path_m":self.max_path_m,"max_rotation_deg":60.,"base_rotation_is_not_relative_inspection":True,
            "anchor":self.anchors[arm],**self.view(model,state.q,arm)}

    def candidates(self,model,state,harness,servo,depth_guard):
        row=self.states.get(harness.index);arm=harness.search_reference.removeprefix("held_")
        receipt={"source":"held_object_relative_view_inspection","tested":[],"command_grip_latch":servo.grips.tolist(),
                 "depth_guard":depth_guard.receipt(),"scene_truth":False}
        if row is None or arm not in self.anchors or not harness.hold_verified.get(arm):
            harness.stop_reason="NO_VERIFIED_HELD_ANCHOR";return (HOLD,),receipt
        if row["attempts"]>=self.max_attempts or row["path_m"]>=self.max_path_m or row["rotation_rad"]>=self.max_rotation_rad:
            harness.stop_reason="HELD_INSPECTION_BUDGET_REACHED";return (HOLD,),receipt
        allowed=[HOLD]
        for action in harness.palette():
            if action==HOLD:continue
            if action.part!=arm or action.move not in (*TRANSLATIONS,*ROTATIONS):raise ValueError("Unscoped held inspection action")
            bound=(row["path_m"]+action.amount(harness.carry)<=self.max_path_m if action.move in TRANSLATIONS else
                   row["rotation_rad"]+action.amount(harness.carry)<=self.max_rotation_rad)
            ok,why=depth_guard.check(action,harness.carry) if bound else (False,"INSPECTION_PATH_BOUND")
            trial=SafeServo(model,state,servo.grips.copy(),servo.limits)
            if ok:ok=trial.begin(action,state,harness.carry);why=trial.status
            result={"action":asdict(action),"accepted":bool(ok),"reason":why}
            if ok:
                predicted=trial.joint_plan[-1] if trial.joint_plan is not None else state.q
                result.update(inspection_after=self.view(model,predicted,arm),planned_ticks=trial.total_ticks)
                allowed.append(action)
            receipt["tested"].append(result)
        if len(allowed)==1:harness.stop_reason="NO_SAFE_HELD_INSPECTION_ACTION"
        return tuple(allowed),receipt

    def executed(self,harness):
        if harness.index in self.states:self.states[harness.index]["attempts"]+=1
