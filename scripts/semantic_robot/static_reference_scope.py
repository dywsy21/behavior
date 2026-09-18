"""Two same-pixel semantic reference checks plus read-only held-action preflight."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.policy import GroundedPolicy,reference_observation_system
from semantic_robot.v2.grounding import GroundedEvidence,localize_target
from semantic_robot.v2.grounded_harness import GroundedHarness,GroundedController
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import VisualBundle
from audit_grasp_motion import frame,read


def preflight(d,obs,planner_reference=None):
    saved=read(d/"harness.json");m=RobotModel(read(d.parent/"robot_calibration.json"));f=frame(d);pr=read(d/"proprio.json")
    state=m.state(np.asarray(pr["q"]),np.asarray(pr["gripper"]),np.zeros(3))
    h=GroundedHarness([Goal(**saved["goal"])],held_inspection=True,contact_geometry=False)
    h.held=saved["held_target_claims"];h.hold_verified=saved["holding_verified_by_observation_and_proprio"]
    if planner_reference is not None:
        h.reference_from_planner=True
        h.bind_reference(planner_reference,"text_only_static_planner")
    h.level={a:False for a in ("left","right")}
    if saved["carry_constraints"]:raise ValueError("Static fixture requires original non-level hold, not an inferred relaxation")
    # Restore a previously observed, verified anchor, never simulator attachment.
    before=d.parent/f"decision_{int(d.name.split('_')[-1])-1:03d}"
    previous=read(before/"grounded_progress.json").get("registered_grasp_motion",{})
    if not previous.get("verified"):raise ValueError("Prior registered observation required")
    bpr=read(before/"proprio.json");bs=m.state(np.array(bpr["q"]),np.array(bpr["gripper"]),np.zeros(3))
    evidence=GroundedEvidence.parse(read(before/"observation.json")["result"]["text"])
    target=localize_target(evidence,dict(np.load(before/"depth.npz")),m,bs.q)
    servo=SafeServo(m,state,[1 if not h.hold_verified[a] else -1 for a in ("left","right")])
    ctl=GroundedController(m,servo,h)
    for arm in h.hold_verified:
        if h.hold_verified[arm]:ctl.inspector.remember(m,bs,arm,target)
    ctl.observe(obs,state,f["depth"],read(d/"depth_receipt.json"))
    if not ctl.is_held_search:return {"inspection_needed":False,"stage":h.stage,"stop_reason":h.stop_reason}
    actions=ctl.inspection_candidates(state)
    return {"inspection_needed":True,"allowed":[asdict(a) for a in actions],"receipt":h.candidate_receipt,
            "context":h.held_inspection,"stop_reason":h.stop_reason,"new_controls":0}


def main():
    p=argparse.ArgumentParser();p.add_argument("--state",action="append",required=True)
    p.add_argument("--expected-reference",action="append",required=True)
    p.add_argument("--uri",required=True);p.add_argument("--revision",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    if len(a.state)!=2 or len(a.expected_reference)!=2:raise ValueError("Exactly two registered states")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise ValueError("Fixed source required")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[];policy=GroundedPolicy(a.uri,a.revision,max_calls=2)
    try:
        for i,(value,expected) in enumerate(zip(a.state,a.expected_reference)):
            d=Path(value);saved=read(d/"observation.json");old=saved["request_without_pixel_duplicates"]
            labels=[x["label"] for x in saved["result"]["images"]]
            if len(labels) not in (6,9):raise ValueError("Original initial/current/history camera set required")
            bundle=VisualBundle([Image.open(d/(label+".png")).convert("RGB") for label in labels],labels,{}, {})
            result,payload=policy._call("observe",reference_observation_system(old["system"]),old["text"],bundle)
            if result["images"]!=saved["result"]["images"]:raise ValueError("Saved service pixels changed")
            obs=GroundedEvidence.parse(result["text"])
            row={"source":value,"call":result,"request_without_pixels":{k:v for k,v in payload.items() if k!="images"},
                 "expected_reference_not_actor_input":expected,"reference_matches":obs.target_reference==expected,
                 "same_original_images":True,"new_controls":0,"not_task_success":True}
            if obs.target_reference.startswith("held_"):row["preflight"]=preflight(d,obs)
            (out/f"case_{i:02d}.json").write_text(json.dumps(row,indent=2));rows.append(row)
            print(json.dumps({"case":i,"reference":obs.target_reference,"matches":row["reference_matches"],"visible":obs.visible}),flush=True)
        if not all(row["reference_matches"] for row in rows):raise ValueError("Semantic reference mismatch; no physical release")
        (out/"result.json").write_text(json.dumps({"status":"complete","model_calls":policy.calls,"controls":0,"rows":rows},indent=2))
    except BaseException as e:
        (out/"failure.json").write_text(json.dumps({"error":repr(e),"model_calls":policy.calls,"controls":0,"rows":rows},indent=2));raise


if __name__=="__main__":main()
