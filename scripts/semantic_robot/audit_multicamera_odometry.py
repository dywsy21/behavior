"""Saved RGB-D diagnosis only. No simulator, model call or actor intervention."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.grasp_motion import robot_point_mask
from semantic_robot.v2.odometry import solve_rgbd_correspondences


def extract(run,index,view,model,exclude_self):
    p=run/f"decision_{index:03d}";rgb=np.asarray(Image.open(p/f"CURRENT_{view.upper()}_RAW.png"))
    depth=np.load(p/"depth.npz")[view]
    receipt=json.loads((p/"depth_receipt.json").read_text())[view]
    for key,data in (("rgb",rgb),("depth",depth)):
        if hashlib.sha256(data.tobytes()).hexdigest()!=receipt[key+"_sha256"]:raise ValueError("Input hash mismatch")
    q=np.asarray(json.loads((p/"proprio.json").read_text())["q"])
    geometry=json.loads((p/"robot_self_geometry.json").read_text())
    K=np.asarray(model.spec["metadata"]["cameras"][view]["K"]);T=model.forward(q,"camera_"+view)
    keypoints,descriptors=cv2.SIFT_create(nfeatures=2000).detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None)
    rows=[];ids=[]
    for i,k in enumerate(keypoints):
        uv=np.asarray(k.pt);x,y=np.rint(uv).astype(int);patch=depth[max(0,y-1):y+2,max(0,x-1):x+2]
        z=float(np.median(patch))
        if not .05<z<5. or not np.isfinite(patch).all() or np.ptp(patch)>.03:continue
        xyz=np.linalg.solve(K,np.r_[uv,1.])*z
        rows.append((uv,xyz));ids.append(i)
    original=len(rows)
    if exclude_self and rows:
        optical=np.asarray([x[1] for x in rows]);base=(optical*np.array([1.,-1.,-1.]))@T[:3,:3].T+T[:3,3]
        own=robot_point_mask(base,geometry)
        if own is None:raise ValueError("Actual robot self geometry missing")
        rows=[x for x,masked in zip(rows,own) if not masked]
        ids=[x for x,masked in zip(ids,own) if not masked]
    return {"rows":rows,"descriptors":None if not ids else descriptors[ids],"K":K,"T":T,
            "all_keypoints":len(keypoints),"stable_depth_keypoints":original,"retained":len(rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--last",type=int,default=15)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.last<=24:raise ValueError("New bounded audit output required")
    started=time.monotonic();cv2.setNumThreads(2)
    model=RobotModel(json.loads((a.run/"robot_calibration.json").read_text()));results=[]
    for view in ("head","left_wrist","right_wrist"):
        for exclude in (False,True):
            before=extract(a.run,0,view,model,exclude)
            for i in range(1,a.last+1):
                if time.monotonic()-started>300:raise TimeoutError("CPU audit bound")
                after=extract(a.run,i,view,model,exclude)
                pairs=cv2.BFMatcher().knnMatch(before["descriptors"],after["descriptors"],k=2) if before["descriptors"] is not None and after["descriptors"] is not None else []
                old=[];new=[];uv=[]
                for pair in pairs:
                    if len(pair)!=2 or pair[0].distance>=.7*pair[1].distance:continue
                    m=pair[0];old.append(before["rows"][m.queryIdx][1]);uv.append(after["rows"][m.trainIdx][0]);new.append(after["rows"][m.trainIdx][1])
                r=solve_rgbd_correspondences(old,uv,new,after["K"],before["T"],after["T"])
                results.append({"view":view,"exclude_robot":exclude,"decision":i,
                    "before_counts":{k:before[k] for k in ("all_keypoints","stable_depth_keypoints","retained")},
                    "after_counts":{k:after[k] for k in ("all_keypoints","stable_depth_keypoints","retained")},"motion":r})
                before=after
    report={"run":str(a.run),"rows":results,"model_calls":0,"controls":0,"wall_s":time.monotonic()-started,
            "diagnostic_only_not_actor_or_retroactive_gate_pass":True,
            "limitation":"Feature filtering before matching differs from old head path; unmasked is also depth-filtered before matching. No result is retroactively declared valid."}
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({"wall_s":report["wall_s"],"last_pair":[x for x in results if x["decision"]==a.last]},indent=2))


if __name__=="__main__":main()
