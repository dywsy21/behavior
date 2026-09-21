"""Bounded PRIVATE robot-kinematic lookahead; only its first action is issued.

Predictions are not captured observations, scene clearance, or positive BC.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from native_motion_codec import BODY_TOKENS, token_to_action
from native_execution import (WORKSPACE_PROFILE,WORKSPACE_PROFILES,CARRY_PROFILE,body_limit, action_codec, workspace_qualified,
    servo_limits, preclose_translation,workspace_budget_ok,primitive_tick_limit)
from native_teacher_policy import empty_hand_rotation_allowed
from native_teacher_outcomes import rigid
from semantic_robot.v2.protocol import ROTATIONS, TRANSLATIONS
from semantic_robot.v2.servo import SafeServo


def fallback(*, model,state,frame,goal_world,base_world,hand,history,ranked,rejected,
             profile,used,remaining_controls,remaining_macros,preflight):
    receipt={"not_actor_input":True,"not_success_evidence":True,"predicted_not_executed":True,
        "body_trials":[],"max_body_trials":4,"max_followups_per_body":3,
        "scene_sweep_clearance_certified":False,"selected_first_token":None}
    if profile not in WORKSPACE_PROFILES:return None,receipt
    if profile==CARRY_PROFILE and (type(used) is not int or not 0<=used<=body_limit(profile)):
        raise ValueError("Actual nonnegative integer body macro count required")
    exhausted=used>=body_limit(profile) if profile==CARRY_PROFILE else bool(used)
    prototype=token_to_action(BODY_TOKENS[0],action_codec(profile))
    if (exhausted or frame.get("forbidden_contacts") or remaining_macros<2 or not ranked or len(rejected)!=len(ranked) or
            [r["token"] for r in rejected]!=ranked or
            not workspace_qualified(prototype,state,model,history,profile) or
            not all(empty_hand_rotation_allowed(a,state,frame,history.grips,model,history.close_seen[a])
                    for a in ("left","right"))):
        receipt["reason"]="UNQUALIFIED_OR_ALREADY_USED_OR_NO_BLOCKED_ARM_SET"
        return None,receipt
    follows=[t for t in ranked if token_to_action(t).part==hand and
             token_to_action(t).move in (*TRANSLATIONS,*ROTATIONS)][:3]
    if not follows:return None,receipt
    local_goal=np.linalg.inv(rigid(base_world))@rigid(goal_world)
    def cost(q):
        pose=model.forward(q,hand)
        return float(np.linalg.norm(pose[:3,3]-local_goal[:3,3])+.04*np.linalg.norm(
            Rotation.from_matrix(local_goal[:3,:3]@pose[:3,:3].T).as_rotvec()))
    initial=cost(state.q);choices=[]
    receipt["current_private_cost"]=initial
    for ordinal,token in enumerate(BODY_TOKENS):
        row={"token":token,"following":[]};receipt["body_trials"].append(row)
        try:body,body_receipt=preflight(token)
        except RuntimeError as exc:
            row["rejected"]=str(exc);continue
        if body.joint_plan is None or not body.total_ticks<=40:raise ValueError("Finite body plan required")
        endpoint=model.state(body.joint_plan[-1],state.gripper.copy(),np.zeros(3))
        row.update(planned_ticks=body.total_ticks,predicted_q=endpoint.q.tolist())
        for index,following in enumerate(follows):
            action=token_to_action(following)
            trial=SafeServo(model,endpoint,gripper_command=history.grips,limits=servo_limits(profile))
            carry=not (preclose_translation(action,endpoint,model,history,profile) or action.move in ROTATIONS)
            accepted=trial.begin(action,endpoint,carry=carry)
            item={"token":following,"accepted_original_robot_preflight":bool(accepted),"status":trial.status,
                "position_tolerance_m":trial.pos_tolerance,"angle_tolerance_deg":float(np.rad2deg(trial.rot_tolerance))}
            row["following"].append(item)
            if not accepted:continue
            if trial.total_ticks>primitive_tick_limit(action,carry,profile):continue
            total=body.total_ticks+12+(75 if profile==CARRY_PROFILE else 40)+12+1
            value=cost(trial.joint_plan[-1])
            item.update(planned_ticks=trial.total_ticks,predicted_private_cost=value,
                        reserved_controls_including_settles_and_hold=total)
            if value<initial-1e-6 and workspace_budget_ok(token,body.total_ticks,remaining_controls,remaining_macros,profile):
                choices.append((value,body.total_ticks+trial.total_ticks,ordinal,index,token,body,body_receipt))
    if not choices:
        receipt["reason"]="NO_BOUNDED_ORIGINAL_GATE_TWO_STEP_IMPROVEMENT";return None,receipt
    chosen=min(choices,key=lambda x:x[:4]);receipt["selected_first_token"]=chosen[4]
    receipt["reason"]="ISSUE_ONLY_FIRST_THEN_RECAPTURE_AND_REPLAN"
    return (chosen[4],chosen[5],chosen[6]),receipt
