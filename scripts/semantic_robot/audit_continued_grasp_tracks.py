"""Fixed saved-state candidate, with no model calls or new controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.grasp_motion import measure_grasp_motion
from semantic_robot.v2.kinematics import RobotModel
from audit_grasp_motion import frame,read


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--decision",type=int,action="append",required=True)
    p.add_argument("--continued",action="store_true");p.add_argument("--stable-seed-depth",action="store_true");p.add_argument("--output",required=True);a=p.parse_args()
    run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);model=RobotModel(read(run/"robot_calibration.json"))
    rows=[];history=None;count=0;delta=np.zeros(3)
    for i in a.decision:
        old,current=run/f"decision_{i-1:03d}",run/f"decision_{i:03d}"
        before,after=frame(old),frame(current);h=read(old/"harness.json");target=h["target_surface_estimate"]
        motion=read(current/"visual_odometry.json")
        if not target["valid"] or not motion.get("valid"):raise ValueError("Registered fixtures need valid saved target/body motion")
        digest=hashlib.sha256(before["rgb"]["right_wrist"].tobytes()).hexdigest()
        chain=bool(history and history["next_decision"]==i and history["image_sha"]==digest and history["goal_index"]==h["goal_index"])
        use=bool(a.continued and chain)
        if not chain:count=0;delta=np.zeros(3)
        points=history["points"] if use else None
        result=measure_grasp_motion(before,after,model,"right",target["point_base_m"],motion["body_transform_current_in_previous"],points,a.stable_seed_depth)
        if result.get("registered_pair_consistent"):
            count+=1;delta+=np.asarray(result["hand_delta_m"])
            history={"next_decision":i+1,"image_sha":result["current_rgb_sha256"],"goal_index":h["goal_index"],
                     "points":[p["after"] for p in result["tracked_pixels"]]}
        else:history=None;count=0;delta=np.zeros(3)
        result.update(decision=i,continued_input_used=use,consecutive_qualified_pairs=count,cumulative_hand_delta_m=delta.tolist(),
                      offline_threshold_met=bool(count>=2 and np.linalg.norm(delta)>=.015 and delta[2]>=.012),
                      not_actual_execution_or_task_success=True)
        canvas=Image.new("RGB",(960,480))
        for k,(key,f) in enumerate((("before",before),("after",after))):
            im=Image.fromarray(f["rgb"]["right_wrist"]);draw=ImageDraw.Draw(im)
            for t in result.get("tracked_pixels",[]):
                x,y=t[key];draw.ellipse((x-3,y-3,x+3,y+3),outline="cyan",width=2)
            canvas.paste(im,(480*k,0))
        canvas.save(out/f"decision_{i:03d}_tracks.jpg")
        rows.append(result);print(json.dumps({k:v for k,v in result.items() if k!="tracked_pixels"}),flush=True)
    (out/"result.json").write_text(json.dumps({"run":str(run),"continued_candidate":a.continued,"stable_seed_depth":a.stable_seed_depth,"controls":0,"model_calls":0,"rows":rows},indent=2))


if __name__=="__main__":main()
