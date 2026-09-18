"""At most four immutable saved observations; same nine image bytes, zero controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.policy import GroundedPolicy,grasp_tracking_instruction
from semantic_robot.v2.grounding import GroundedEvidence,localize_target
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.vision import VisualBundle
from audit_grasp_motion import frame,read
from semantic_robot.v2.grasp_motion import measure_grasp_motion
from semantic_robot.v2.odometry import RGBDMotion


def main():
    p=argparse.ArgumentParser();p.add_argument("--state",action="append",required=True)
    p.add_argument("--uri",required=True);p.add_argument("--revision",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    if not 1<=len(a.state)<=4:raise ValueError("At most four saved states")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise ValueError("Fixed clean source required")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    policy=GroundedPolicy(a.uri,a.revision,max_calls=len(a.state));rows=[]
    for index,value in enumerate(a.state):
        d=Path(value);root=d.parent;i=int(d.name.split("_")[-1]);after_dir=root/f"decision_{i+1:03d}"
        saved=read(d/"observation.json");original=saved["request_without_pixel_duplicates"]
        h=read(d/"harness.json");scope=SimpleNamespace(goal=Goal(**h["goal"]),stage="VERIFY_GRASP")
        extra=grasp_tracking_instruction(scope)
        if not extra or h["stage"]!="VERIFY_GRASP":raise ValueError("Saved verification observation required")
        labels=[r["label"] for r in saved["result"]["images"]]
        if len(labels)!=9:raise ValueError("Original nine-image comparison required")
        images=[Image.open(d/(label+".png")).convert("RGB") for label in labels]
        bundle=VisualBundle(images,labels,{},{});model=RobotModel(read(root/"robot_calibration.json"))
        before,after=frame(d),frame(after_dir);depth_receipt=read(d/"depth_receipt.json")
        for view in before["rgb"]:
            for kind in ("rgb","depth"):
                if hashlib.sha256(before[kind][view].tobytes()).hexdigest()!=depth_receipt[view][kind+"_sha256"]:raise ValueError("Saved RGB-D hash mismatch")
        result,payload=policy._call("observe",original["system"]+extra,original["text"],bundle)
        if result["images"]!=saved["result"]["images"]:raise ValueError("Original service image hashes changed")
        obs=GroundedEvidence.parse(result["text"]);target=localize_target(obs,before["depth"],model,before["q"])
        motion=RGBDMotion(read(root/"manifest.json")["args"]["odometry_estimator"])
        for f in (before,after):receipt=motion.observe({k+"_rgb":v for k,v in f["rgb"].items()},f["depth"],model,f["q"])
        tracks=measure_grasp_motion(before,after,model,"right",target["point_base_m"],receipt["body_transform_current_in_previous"]) if target.get("valid") and receipt.get("valid") else {"valid":False,"reason":"NO_VALID_TARGET_OR_BODY_MOTION"}
        row={"source":str(d),"next_saved_frame":str(after_dir),"call":result,"request_without_pixels":{k:v for k,v in payload.items() if k!="images"},"target":target,"tracks":tracks,"same_nine_images":True,"counterfactual_seed_only":True,"new_controls":0,"not_grasp_authorization":True}
        (out/f"case_{index:02d}.json").write_text(json.dumps(row,indent=2));rows.append(row)
        print(json.dumps({"case":index,"source":str(d),"target_valid":target["valid"],"tracks":tracks.get("rgbd_tracks"),"consistent":tracks.get("registered_pair_consistent")}),flush=True)
    (out/"result.json").write_text(json.dumps({"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),"model_calls":policy.calls,"controls":0,"rows":rows},indent=2))


if __name__=="__main__":main()
