"""Independent saved RGB-D motion check. Diagnostic only, no actor corrections."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.v2.kinematics import RobotModel


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();run=Path(args.run);out=Path(args.output)
    if out.exists():raise ValueError("Preserve old odometry audit")
    model=RobotModel(json.loads((run/"robot_calibration.json").read_text()))
    camera=model.spec["metadata"]["cameras"]["head"];K=np.asarray(camera["K"])
    terminal=json.loads((run/("result.json" if (run/"result.json").exists() else "failure.json")).read_text())
    decisions={d["decision"]:d for d in terminal["decisions"]}
    cv2.setNumThreads(2);cv2.setRNGSeed(31);sift=cv2.SIFT_create(nfeatures=2000)
    matcher=cv2.BFMatcher();previous=None;rows=[]
    S=np.diag([1.,-1.,-1.,1.])
    for source in sorted(run.glob("decision_*")):
        i=int(source.name.split("_")[-1])
        image=cv2.imread(str(source/"CURRENT_HEAD_RAW.png"),cv2.IMREAD_GRAYSCALE)
        kp,desc=sift.detectAndCompute(image,None)
        proprio=json.loads((source/"proprio.json").read_text())
        T=model.forward(np.asarray(proprio["q"]),"camera_head")
        depth=np.load(source/"depth.npz")["head"]
        current=(i,kp,desc,T,depth)
        if previous is not None and i==previous[0]+1:
            j,oldkp,olddesc,oldT,olddepth=previous
            row={"from":j,"to":i,"controls":decisions.get(j,{}).get("feedback",{}).get("control_ticks"),
                 "reported_base_integral":decisions.get(j,{}).get("feedback",{}).get("base_integral"),"accepted":False}
            matches=matcher.knnMatch(olddesc,desc,k=2) if olddesc is not None and desc is not None else []
            points=[];pixels=[]
            for pair in matches:
                if len(pair)!=2 or pair[0].distance>=.7*pair[1].distance:continue
                match=pair[0];uv=np.asarray(oldkp[match.queryIdx].pt);x,y=np.rint(uv).astype(int)
                patch=olddepth[max(0,y-1):y+2,max(0,x-1):x+2]
                z=float(np.median(patch))
                if not .05<z<5. or not np.isfinite(patch).all() or np.ptp(patch)>.03:continue
                points.append(np.linalg.solve(K,np.r_[uv,1.])*z);pixels.append(kp[match.trainIdx].pt)
            row["matched_stable_points"]=len(points)
            if len(points)>=25:
                xyz=np.asarray(points);uv=np.asarray(pixels)
                ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=200,reprojectionError=1.8,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
                if ok and inliers is not None and len(inliers)>=25:
                    sel=inliers.ravel();rvec,tvec=cv2.solvePnPRefineLM(xyz[sel],uv[sel],K,None,rvec,tvec)
                    rel=np.eye(4);rel[:3,:3]=cv2.Rodrigues(rvec)[0];rel[:3,3]=tvec.ravel()
                    # Map OLD body coordinates to CURRENT body, then invert to
                    # recover current body displacement in the old body frame.
                    motion=np.linalg.inv(T@S@rel@S@np.linalg.inv(oldT))
                    pred,_=cv2.projectPoints(xyz[sel],rvec,tvec,K,None)
                    error=float(np.median(np.linalg.norm(pred.reshape(-1,2)-uv[sel],axis=1)))
                    row.update(accepted=error<1.0,inliers=len(sel),median_reprojection_px=error,
                               visual_body_yaw_rad=float(np.arctan2(motion[1,0],motion[0,0])),
                               visual_body_translation_m=motion[:3,3].tolist())
            rows.append(row)
        previous=current
    good=[r for r in rows if r["accepted"] and r["reported_base_integral"] is not None and abs(r["reported_base_integral"][2])>.03]
    ratios=[r["visual_body_yaw_rad"]/r["reported_base_integral"][2] for r in good]
    result={"run":str(run),"diagnostic_only":True,"no_scene_truth":True,"new_controls":0,"new_model_calls":0,
            "method":"SIFT stable-depth PnP RANSAC + robot-relative camera FK, independent from velocity integral",
            "accepted_rotating_pairs":len(good),"median_visual_to_reported_yaw_ratio":float(np.median(ratios)) if ratios else None,
            "rows":rows}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
