"""Whitelisted deploy inputs for the separately bounded local-skill pilot."""
import base64
from io import BytesIO

import numpy as np
from PIL import Image

from common import CAMERAS,TOKENS,actor_state,prompt

ACTIVE = {0: "verb=GRASP; target=radio; source=coffee table",
          3: "verb=NAVIGATE; target=breakfast table"}


def runtime_proprio(state):
    s=np.zeros(61)
    s[:3]=state.base_velocity;s[53:57]=state.q[:4]
    for i,(arm,position,finger) in enumerate((("left",17,24),("right",42,49))):
        s[position:position+3]=state.poses[arm][0]
        s[finger:finger+2]=state.gripper[i]
    return actor_state(s)


def validate_proprio(value):
    expected={"eef_base_m","finger_opening_m","base_velocity_local","torso_joints_rad"}
    if set(value)!=expected:raise ValueError("Only current robot proprioception is accepted")
    for name in ("eef_base_m","finger_opening_m"):
        if set(value[name])!={"left","right"}:raise ValueError("Exactly two robot arms required")
    for arm in ("left","right"):
        v=np.asarray(value["eef_base_m"][arm],dtype=float)
        if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError("Invalid EEF vector")
        g=float(value["finger_opening_m"][arm])
        if not np.isfinite(g) or not -.001<=g<=.06:raise ValueError("Invalid current gripper opening")
    for name,n in (("base_velocity_local",3),("torso_joints_rad",4)):
        v=np.asarray(value[name],dtype=float)
        if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError("Invalid body proprioception")


def request_payload(task,proprio,history,images,variant,task_id):
    validate_proprio(proprio)
    if task_id not in ACTIVE or variant not in ("base","finetuned"):raise ValueError("Unregistered pilot")
    encoded={}
    for view in CAMERAS:
        image=images[view].convert("RGB").resize((256,256),Image.Resampling.LANCZOS)
        data=BytesIO();image.save(data,format="PNG");encoded[view]=base64.b64encode(data.getvalue()).decode()
    return {"task":task,"active_instruction":ACTIVE[task_id],"proprio":proprio,
            "history":history[-5:],"images":encoded,"variant":variant}


def parse_request(value):
    if set(value)!={"task","active_instruction","proprio","history","images","variant"}:raise ValueError("Unexpected actor fields")
    if value["variant"] not in ("base","finetuned"):raise ValueError("Invalid model variant")
    if not isinstance(value["task"],str) or not 1<=len(value["task"])<=5000:raise ValueError("Invalid task text")
    if value["active_instruction"] not in ACTIVE.values():raise ValueError("Only pre-registered fixed local instructions allowed")
    validate_proprio(value["proprio"])
    if not isinstance(value["history"],list) or len(value["history"])>5 or any(t not in TOKENS for t in value["history"]):raise ValueError("Invalid executed history")
    if set(value["images"])!=set(CAMERAS):raise ValueError("Exactly three current RGB cameras required")
    images={}
    for view,encoded in value["images"].items():
        if not isinstance(encoded,str) or len(encoded)>400000:raise ValueError("Oversized image")
        img=Image.open(BytesIO(base64.b64decode(encoded,validate=True)))
        if img.size!=(256,256):raise ValueError("Expected exact training image dimensions")
        images[view]=img.convert("RGB")
    return {"text":prompt(value["task"],value["active_instruction"],value["proprio"],value["history"])},images
