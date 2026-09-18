"""Finite robot-only choice of a feasible free-wrist engineering test pair."""
from dataclasses import asdict
import numpy as np

from .arm_observation_guard import ObservingArmGuard
from .grounding import observed_cloud
from .protocol import Action, ROTATIONS
from .servo import SafeServo


def choose_gate_pair(model,state,servo,depths,self_geometry):
    receipt={"source":"finite_6_axis_robot_only_gate_pair", "tested":[], "scene_truth":False,
             "not_task_policy":True,"scale":"coarse","environment_clearance_certified":False}
    meta=model.spec.get("metadata",{})
    opening=meta.get("grasp_region_reference_gripper_m")
    if (servo.grips[0]<=0 or opening is None or not meta.get("grasp_region_reference_fully_open",{}).get("left")
            or abs(state.gripper[0]-opening[0])>.0005):
        return None,None,{**receipt,"reason":"GATE_REQUIRES_MEASURED_OPEN_FREE_LEFT_HAND"}
    guard=ObservingArmGuard(model,state.q,observed_cloud(depths,model,state.q,stride=6),self_geometry,"left")
    for move in ROTATIONS:
        action=Action("left",move,"coarse","tool")
        axis,sign=move.split("_")
        reverse=Action("left",axis+("_minus" if sign=="plus" else "_plus"),"coarse","tool")
        forward=SafeServo(model,state,servo.grips.copy(),servo.limits)
        row={"forward":asdict(action),"reverse":asdict(reverse)}
        accepted=forward.begin(action,state,False)
        row.update(forward_ik=accepted,forward_reason=forward.status)
        if accepted:
            predicted=model.state(forward.joint_plan[-1],state.gripper,state.base_velocity)
            back=SafeServo(model,predicted,servo.grips.copy(),servo.limits)
            accepted=back.begin(reverse,predicted,False)
            row.update(reverse_ik=accepted,reverse_reason=back.status)
            if accepted:
                path=np.vstack([forward.joint_plan,back.joint_plan])
                if len(path)>64:
                    accepted=False;row["reason"]="ROUND_TRIP_SAMPLING_BUDGET"
                else:
                    accepted,check=guard.check(path)
                    row["sweep"]=check
        row["accepted"]=bool(accepted);receipt["tested"].append(row)
        if accepted:
            return action,reverse,{**receipt,"reason":"FEASIBLE_COARSE_FREE_WRIST_PAIR"}
    return None,None,{**receipt,"reason":"NO_FEASIBLE_COARSE_FREE_WRIST_PAIR"}
