"""Shared public-only base / adapter / nonvisual NN execution interface.

No teacher spec, pose seed, held flag, contact, target pose or outcome enters
these functions. Current full-open qualification is NOT proof of no contact.
"""
import json
import hashlib
import numpy as np

from native_motion_codec import tokens as motion_tokens,token_to_action,codec_for_protocol,BODY_TOKENS
from native_actor_protocol import VERSION,validate_actor,request_payload,finite,check_clock
from native_teacher_artifacts import json_bytes
from semantic_robot.v2.protocol import ROTATIONS
from semantic_robot.v2.servo import SafeServo
from native_execution import (validate_profile,servo_limits,metadata as execution_metadata,completed,
    PublicGripperHistory,preclose_translation,timing_metadata)
from native_execution import (action_codec,actor_protocol,workspace_qualified,workspace_metadata,WORKSPACE_PROFILE,
    CARRY_PROFILE,WorkspaceQuota,primitive_tick_limit)
from semantic_robot.v2.grounding import LocalDepthGuard,observed_cloud

VARIANTS=("base","finetuned","proprio_history_nn")


def features(actor):
    actor=validate_actor(actor);p=actor["proprio"]
    # Fixed units, no image/hash/instance/clock/teacher stage feature.
    values=[np.array(p["base_velocity_local"])/.01,np.array(p["torso_joints_rad"])/.1]
    for arm in ("left","right"):
        values.extend([np.array(p["eef_base_m"][arm])/.01,np.array(p["joint_positions_rad"][arm])/.1,
            np.array(p["eef_rotation_base"][arm]).reshape(-1)/.05,np.array([p["finger_opening_m"][arm]])/.01])
    vocabulary=motion_tokens(codec_for_protocol(actor["protocol"]))
    history=np.zeros((5,len(vocabulary)+1))
    padded=[None]*(5-len(actor["history"]))+actor["history"]
    for i,t in enumerate(padded):history[i,len(vocabulary) if t is None else vocabulary.index(t)]=2.
    return np.concatenate([*values,history.reshape(-1)])


class NearestNeighbor:
    def __init__(self,rows):
        if not rows:raise ValueError("Reviewed TRAIN rows required")
        self.rows=sorted(rows,key=lambda r:r["id"])
        self.protocol=self.rows[0]["actor"]["protocol"]
        if any(r["actor"]["protocol"]!=self.protocol for r in self.rows):raise ValueError("Mixed NN action protocols")
        self.x=np.array([features(r["actor"]) for r in self.rows])

    def predict(self,actor):
        if actor["protocol"]!=self.protocol:raise ValueError("NN protocol differs from TRAIN")
        x=features(actor)
        available=[i for i,r in enumerate(self.rows) if r["actor"]["active_instruction"]==actor["active_instruction"]]
        if not available:raise ValueError("Unregistered instruction for train-only NN")
        distances=np.sum((self.x[available]-x)**2,axis=1);j=int(np.argmin(distances));i=available[j]
        return {"prediction":self.rows[i]["target"],"variant":"proprio_history_nn","protocol":self.protocol,
            "neighbor_training_id":self.rows[i]["id"],"squared_distance":float(distances[j]),"images_used":False}


class PublicExecution(PublicGripperHistory):
    """Gripper command state changes only at an actual control issuance."""
    def __init__(self,grips,*,execution_profile=None):
        self.execution_profile=validate_profile(execution_profile)
        super().__init__(grips)
        self.workspace_seen=False;self.pending_workspace=False
        self.workspace_quota=WorkspaceQuota(self.execution_profile)

    def issued(self,command):
        super().issued(command)
        if self.pending_workspace:self.workspace_seen=True
        if self.execution_profile==CARRY_PROFILE:self.workspace_quota.issued(command)

    def cancel_pending(self):
        self.pending_workspace=False;self.workspace_quota.cancel_pending()

    def completed(self,token,feedback):
        return completed(token,feedback,profile=self.execution_profile)

    def preflight(self,token,state,model,depths,geometry,*,capture_receipt,expected_clock):
        if self.execution_profile==CARRY_PROFILE:self.cancel_pending()
        check_clock(expected_clock);check_clock(capture_receipt["clock"])
        if (capture_receipt["clock"]!=expected_clock or capture_receipt["kinematic_model_sha256"]!=model.sha or
                not np.array_equal(finite(capture_receipt["q"],(18,)),state.q) or
                not np.array_equal(finite(capture_receipt["gripper"],(2,)),state.gripper) or
                capture_receipt.get("scene_truth") is not False or
                hashlib.sha256(json_bytes(geometry)).hexdigest()!=capture_receipt["files_sha256"]["robot_self_geometry.json"] or
                set(depths)!={"head","left_wrist","right_wrist"} or any(
                    hashlib.sha256(np.ascontiguousarray(depths[v]).tobytes()).hexdigest()!=capture_receipt["depth_array_sha256"][v] for v in depths)):
            raise ValueError("Fresh capture q/grip/depth/actual-box binding required")
        action=token_to_action(token,action_codec(self.execution_profile));arm=action.part
        if token.endswith("_OPEN") and self.close_seen.get(token.split("_")[0].lower(),True):
            raise RuntimeError("Fixed GRASP preserves issued CLOSE latch")
        qualified=preclose_translation(action,state,model,self,self.execution_profile)
        body=workspace_qualified(action,state,model,self,self.execution_profile)
        spent=not self.workspace_quota.available() if self.execution_profile==CARRY_PROFILE else self.workspace_seen
        if action.part=="torso" and (not body or spent):raise RuntimeError("Workspace unqualified or body budget spent")
        carry=not (qualified or body)
        hand_rotation=action.move in ROTATIONS and arm in ("left","right")
        if hand_rotation:
            if not self.rotation_qualified(arm,state,model):raise RuntimeError("No public full-open rotation qualification")
            carry=False
        if geometry is None:raise ValueError("Fresh actual robot geometry required")
        guard=LocalDepthGuard(observed_cloud(depths,model,state.q),model,state.q,depths,self_geometry=geometry)
        allowed,reason=guard.check(action,carry)
        if not allowed:raise RuntimeError(reason)
        servo=SafeServo(model,state,gripper_command=self.grips,limits=servo_limits(self.execution_profile))
        if not servo.begin(action,state,carry=carry) or servo.total_ticks>primitive_tick_limit(action,carry,self.execution_profile):raise RuntimeError(servo.status)
        self.pending_workspace=body
        if self.execution_profile==CARRY_PROFILE:self.workspace_quota.arm(token)
        return servo,{**execution_metadata(self.execution_profile),**timing_metadata(self.execution_profile,qualified),
            **workspace_metadata(action,body,self.execution_profile),
            "carry":carry,"robot_geometry_guards":True,"amount":action.amount(carry),
            "public_full_open_rotation_qualification":hand_rotation,
            "empty_hand_or_environment_contact_not_certified":True,"depth":guard.receipt(),"depth_check":reason}


def choose(variant,actor,raw_images,*,nn=None,remote=None):
    validate_actor(actor)
    if variant not in VARIANTS:raise ValueError("Unregistered comparison arm")
    if variant=="proprio_history_nn":
        if nn is None:raise ValueError("Missing TRAIN-only NN")
        answer=nn.predict(actor)
    else:
        if remote is None:raise ValueError("No separately authorized model service")
        answer=remote(request_payload(actor,raw_images,variant))
    if (answer.get("prediction") not in motion_tokens(codec_for_protocol(actor["protocol"])) or
            answer.get("protocol")!=actor["protocol"] or answer.get("variant")!=variant):
        raise ValueError("Policy/codec/protocol identity drift")
    return answer
