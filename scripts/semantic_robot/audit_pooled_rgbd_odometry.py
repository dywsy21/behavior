"""Offline hypothesis: calibrated simultaneous cameras share ONE body motion.

Same existing fit/residual/count bounds; this does not authorize deployment.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np

from audit_multicamera_odometry import extract, RobotModel
from semantic_robot.v2.odometry import rigid_fit


def solve(rows):
    result={"valid":False,"matches":len(rows)}
    if len(rows)<25:return {**result,"reason":"INSUFFICIENT_STABLE_CORRESPONDENCES"}
    a=np.asarray([r["before"] for r in rows]);b=np.asarray([r["after"] for r in rows])
    rng=np.random.default_rng(31);best=None;score=(-1,-float("inf"))
    for _ in range(200):
        choice=rng.choice(len(a),3,replace=False);fit=rigid_fit(a[choice],b[choice])
        if fit is None:continue
        R,t=fit;e=np.linalg.norm(a@R.T+t-b,axis=1);sel=e<.01
        s=(int(sel.sum()),-float(np.median(e[sel])) if sel.any() else -float("inf"))
        if s>score:best,score=sel,s
    if best is None or best.sum()<25:return {**result,"reason":"RGBD_RIGID_NOT_SUPPORTED"}
    for _ in range(2):
        fit=rigid_fit(a[best],b[best])
        if fit is None:return {**result,"reason":"DEGENERATE_3D_SUPPORT"}
        R,t=fit;e=np.linalg.norm(a@R.T+t-b,axis=1);best=e<.01
        if best.sum()<25:return {**result,"reason":"RGBD_RIGID_NOT_SUPPORTED"}
    ids=np.flatnonzero(best);fraction=len(ids)/len(a)
    result.update(inliers=len(ids),inlier_fraction=fraction,
                  inlier_views=dict(Counter(rows[i]["view"] for i in ids)))
    if fraction<.45:return {**result,"reason":"MATCHES_NOT_GEOMETRICALLY_COHERENT"}
    predicted=a@R.T+t;pixels=[]
    for i in ids:
        x=rows[i];T=x["camera_after"]
        optical=((predicted[i]-T[:3,3])@T[:3,:3])*[1.,-1.,-1.]
        if optical[2]<=.01:return {**result,"reason":"INVALID_PROJECTED_DEPTH"}
        uv=x["K"]@optical;pixels.append(np.linalg.norm(uv[:2]/uv[2]-x["uv"]))
    rel=np.eye(4);rel[:3,:3]=R;rel[:3,3]=t;motion=np.linalg.inv(rel)
    delta=np.array([motion[0,3],motion[1,3],np.arctan2(motion[1,0],motion[0,0])])
    pixel=float(np.median(pixels));point=float(np.median(e[ids]))
    result.update(body_delta=delta.tolist(),body_translation_z_m=float(motion[2,3]),
                  body_transform_current_in_previous=motion.tolist(),median_reprojection_px=pixel,
                  median_depth_correspondence_m=point,
                  per_view_median_reprojection_px={view:float(np.median([pe for i,pe in zip(ids,pixels) if rows[i]["view"]==view])) for view in result["inlier_views"]})
    if pixel>1. or point>.015:return {**result,"reason":"RGB_DEPTH_MOTION_DISAGREEMENT"}
    if np.linalg.norm(delta[:2])>.18 or abs(delta[2])>.30 or abs(motion[2,3])>.035:
        return {**result,"reason":"OUTSIDE_SINGLE_MICRO_ACTION_MOTION_BOUND"}
    return {**result,"valid":True,"reason":"POOLED_BODY_RGBD_DIAGNOSTIC_ONLY"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError("Preserve evidence")
    started=time.monotonic();cv2.setNumThreads(2);views=("head","left_wrist","right_wrist")
    model=RobotModel(json.loads((a.run/"robot_calibration.json").read_text()));before={v:extract(a.run,0,v,model,True) for v in views};reports=[]
    for d in range(1,16):
        if time.monotonic()-started>300:raise TimeoutError("CPU budget")
        after={v:extract(a.run,d,v,model,True) for v in views};rows=[]
        for v in views:
            x,y=before[v],after[v]
            pairs=cv2.BFMatcher().knnMatch(x["descriptors"],y["descriptors"],k=2) if x["descriptors"] is not None and y["descriptors"] is not None else []
            for pair in pairs:
                if len(pair)!=2 or pair[0].distance>=.7*pair[1].distance:continue
                m=pair[0];o=x["rows"][m.queryIdx][1];n=y["rows"][m.trainIdx][1]
                rows.append({"view":v,"before":(o*[1.,-1.,-1.])@x["T"][:3,:3].T+x["T"][:3,3],
                    "after":(n*[1.,-1.,-1.])@y["T"][:3,:3].T+y["T"][:3,3],
                    "uv":y["rows"][m.trainIdx][0],"K":y["K"],"camera_after":y["T"]})
        r=solve(rows);reports.append({"decision":d,"motion":r});before=after
    result={"diagnostic_only":True,"model_calls":0,"controls":0,"wall_s":time.monotonic()-started,"rows":reports}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":main()
