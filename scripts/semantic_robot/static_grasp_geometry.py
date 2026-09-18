"""Two saved states; new robot guides, same raw RGB-D, <=4 model calls."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.grounded_harness import GroundedController,GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import prepare_views


def main():
    p=argparse.ArgumentParser();p.add_argument("--state",action="append",required=True)
    p.add_argument("--robot-reference",required=True);p.add_argument("--uri",required=True)
    p.add_argument("--revision",required=True);p.add_argument("--output",required=True)
    p.add_argument("--prepare-only",action="store_true");a=p.parse_args()
    if not 1<=len(a.state)<=2:raise ValueError("At most two fixed states")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise RuntimeError("Fixed clean source required")
    ref=Path(a.robot_reference);refmodel=json.loads((ref/"robot_calibration.json").read_text());refmanifest=json.loads((ref/"manifest.json").read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    policy=None if a.prepare_only else GroundedPolicy(a.uri,a.revision,max_calls=2*len(a.state));rows=[]
    keys=("grasp_regions_eef","grasp_region_reference_gripper_m","grasp_region_reference_fully_open","grasp_region_source")
    for i,value in enumerate(a.state):
        d=Path(value);manifest=json.loads((d.parent/"manifest.json").read_text())
        if manifest["robot_sha"]!=refmanifest["robot_sha"]:raise ValueError("Robot asset mismatch")
        spec=json.loads((d.parent/"robot_calibration.json").read_text())
        if spec["metadata"]["joint_names"]!=refmodel["metadata"]["joint_names"]:raise ValueError("Robot joint convention mismatch")
        for key in keys:spec["metadata"][key]=copy.deepcopy(refmodel["metadata"][key])
        m=RobotModel(spec);pr=json.loads((d/"proprio.json").read_text());saved=json.loads((d/"harness.json").read_text())
        state=m.state(np.asarray(pr["q"]),np.asarray(pr["gripper"]),np.zeros(3))
        h=GroundedHarness([Goal(**saved["goal"])],active_grasp_probe=True);h.stage=saved["stage"]
        h.held=saved["held_target_claims"];h.hold_verified=saved["holding_verified_by_observation_and_proprio"]
        h.pending_grasp=saved["unverified_close_latches"]
        h.grasp_probe_attempts[0]=saved.get("active_grasp_probe",{}).get("attempts_used",0)
        history=saved["recent_executed"]
        for r in history:h.history.append(r)
        if history:h.last_action=Action(**history[-1]["action"]);h.feedback=history[-1]["feedback"]
        views=("head","left_wrist","right_wrist");raw={v:np.asarray(Image.open(d/("CURRENT_"+v.upper()+"_RAW.png"))) for v in views}
        previous={v:np.asarray(Image.open(d/("PREVIOUS_"+v.upper()+"_RAW.png"))) for v in views}
        bundle=prepare_views({v+"_rgb":x for v,x in raw.items()},m,state.q,previous,grounded=True,gripper=state.gripper)
        directory=out/f"case_{i:02d}";directory.mkdir()
        for label,img in zip(bundle.labels,bundle.images):img.save(directory/(label+".png"))
        receipt=json.loads((d/"depth_receipt.json").read_text());depths=dict(np.load(d/"depth.npz"))
        for v in views:
            if hashlib.sha256(raw[v].tobytes()).hexdigest()!=receipt[v]["rgb_sha256"] or hashlib.sha256(depths[v].tobytes()).hexdigest()!=receipt[v]["depth_sha256"]:raise ValueError("Saved RGB-D mismatch")
        row={"source":str(d),"robot_reference":str(ref),"raw_rgbd_hashes_unchanged":True,"guides_intentionally_regenerated":True,
            "geometry":bundle.geometry,"controls":0}
        if policy:
            obs,call=policy.observe(h,state,bundle);row["observation_result"]=call["result"]
            original=json.loads((d/"observation.json").read_text())["result"]["images"]
            raw_rows=lambda records:[r for r in records if r["label"].endswith("_RAW")]
            if raw_rows(original)!=raw_rows(call["result"]["images"]):raise ValueError("Service raw image bytes changed")
            row["service_raw_image_hashes_unchanged"]=True
            candidate=json.loads((d/"candidates.json").read_text());servo=SafeServo(m,state,candidate["command_grip_latch"])
            ctl=GroundedController(m,servo,h);ctl.observe(obs,state,depths,receipt)
            allowed=ctl.candidates(state);row["grounding"]=h.grounding;row["probe"]=h.grasp_probe
            if not h.stop_reason and h.grounding.get("valid"):
                action,call=policy.act_feasible(h,state,bundle,allowed);row["action"]=asdict(action);row["action_result"]=call["result"]
            else:row["action_skipped"]=h.stop_reason or "NO_VALID_TARGET_DEPTH"
        (directory/"result.json").write_text(json.dumps(row,indent=2));rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k in ("source","action","action_skipped","raw_rgbd_hashes_unchanged")}),flush=True)
    (out/"result.json").write_text(json.dumps({"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),"model_calls":policy.calls if policy else 0,"controls":0,"rows":rows},indent=2))


if __name__=="__main__":main()
