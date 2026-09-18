"""Causal saved-view budget checks; no new physics or scene-truth input."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO/"src")]
from audit_grasp_motion import read,frame
from semantic_robot.v2.grounded_harness import GroundedHarness,GroundedController
from semantic_robot.v2.grounding import GroundedEvidence,LocalDepthGuard,observed_cloud
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import VisualBundle


def restore(run,index,multicamera=False):
    d=run/f"decision_{index:03d}";saved=read(d/"harness.json");model=RobotModel(read(run/"robot_calibration.json"))
    f=frame(d);pr=read(d/"proprio.json");state=model.state(np.array(pr["q"]),np.array(pr["gripper"]),np.zeros(3))
    if saved["carry_constraints"] or saved["target_reference"]!="held_right":raise ValueError("Original non-level held fixture required")
    h=GroundedHarness([Goal(**saved["goal"])],held_inspection=True,reference_from_planner=True,
                      contact_geometry=False,inspection_budget_aware=True,multicamera_inspection=multicamera)
    h.held=saved["held_target_claims"];h.hold_verified=saved["holding_verified_by_observation_and_proprio"]
    h.bind_reference(saved["target_reference"],"original_saved_semantic_relation_not_new_truth")
    h.stage=saved["stage"];h.observation=GroundedEvidence.parse(read(d/"observation.json")["result"]["text"])
    h.history.extend(saved["recent_executed"])
    if h.history:h.feedback=h.history[-1]["feedback"];h.last_action=Action(**h.history[-1]["action"])
    servo=SafeServo(model,state,[1.,-1.]);ctl=GroundedController(model,servo,h)
    anchor=saved["held_inspection"]["anchor"]
    if anchor["source"]!="previous_verified_onboard_RGBD_surface_not_current_affordance":raise ValueError("Measured historical anchor required")
    ctl.inspector.anchors["right"]=anchor
    # Only poses already observed by this saved decision. No later image/state.
    for j in range(4,index+1):
        past=read(run/f"decision_{j:03d}"/"proprio.json")
        oldstate=model.state(np.array(past["q"]),np.array(past["gripper"]),np.zeros(3))
        if j>4:ctl.inspector.executed(h)
        if multicamera:
            # Earlier poses reconstruct only measured motion accounts. A cloud
            # from another pose is never presented as current free-arm depth.
            ctl.inspector.observe(model,oldstate,h,f["depth"] if j==index else None,
                                  read(d/"robot_self_geometry.json") if j==index else None)
        else:ctl.inspector.observe(model,oldstate,h)
    h.held_inspection=ctl.inspector.context(model,state,h)
    for key in ("attempts","relative_path_m","relative_rotation_deg"):
        if not np.isclose(h.held_inspection[key],saved["held_inspection"][key],rtol=1e-6,atol=1e-6):raise ValueError("Causal history disagrees with saved actual budget")
    ctl.depth_guard=LocalDepthGuard(observed_cloud(f["depth"],model,state.q),model,state.q,f["depth"])
    allowed=ctl.inspection_candidates(state)
    original=read(d/"action.json");labels=[x["label"] for x in original["result"]["images"]]
    bundle=VisualBundle([Image.open(d/(label+".png")).convert("RGB") for label in labels],labels,pr["geometry"],f["rgb"])
    return h,state,bundle,allowed,original


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--uri");p.add_argument("--revision");a=p.parse_args()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise ValueError("Pinned clean source required")
    a.output.mkdir(exist_ok=False);policy=GroundedPolicy(a.uri,a.revision,max_calls=3) if a.uri else None;rows=[]
    try:
        for i in (4,14,25,26):
            h,state,bundle,allowed,original=restore(a.run,i)
            row={"decision":i,"controls":0,"context":h.held_inspection,"candidates":h.candidate_receipt,
                 "allowed":[asdict(x) for x in allowed],"stop_reason":h.stop_reason}
            if h.stop_reason:raise ValueError(h.stop_reason)
            if policy is not None and i!=26:
                action,call=policy.act_feasible(h,state,bundle,allowed)
                if call["result"]["images"]!=original["result"]["images"]:raise ValueError("Saved image pixels changed")
                row.update(action=asdict(action),call=call,same_original_images=True)
            rows.append(row);(a.output/f"decision_{i:03d}.json").write_text(json.dumps(row,indent=2))
            print(json.dumps({"decision":i,"allowed":len(allowed),"selected":row.get("action"),"stop_reason":h.stop_reason}),flush=True)
        undo={"part":"right","move":"roll_minus","scale":"fine","frame":"tool"}
        if undo in rows[-1]["allowed"]:raise ValueError("Original revisited view was not rejected")
        (a.output/"result.json").write_text(json.dumps({"status":"complete","model_calls":policy.calls if policy else 0,
            "controls":0,"rows":rows,"not_task_success":True},indent=2))
    except BaseException as error:
        (a.output/"failure.json").write_text(json.dumps({"error":repr(error),"model_calls":policy.calls if policy else 0,"controls":0},indent=2));raise


if __name__=="__main__":main()
