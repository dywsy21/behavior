"""Read-only matching of saved RGB-D grasp attempts; zero controls/model calls."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.grasp_motion import measure_grasp_motion
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.odometry import RGBDMotion


def read(p):return json.loads(p.read_text())


def frame(p):
    return {"q":np.asarray(read(p/"proprio.json")["q"]),"depth":dict(np.load(p/"depth.npz")),
        "rgb":{v:np.asarray(Image.open(p/("CURRENT_"+v.upper()+"_RAW.png"))) for v in ("head","left_wrist","right_wrist")}}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--decision",type=int,action="append",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);m=RobotModel(read(root/"robot_calibration.json"));rows=[]
    for i in a.decision:
        d0,d1=(root/f"decision_{j:03d}" for j in (i-1,i));before,after=frame(d0),frame(d1)
        target=read(d0/"harness.json")["target_surface_estimate"]
        if not target["valid"]:rows.append({"decision":i,"valid":False,"reason":"PREVIOUS_TARGET_UNKNOWN"});continue
        motion=RGBDMotion()
        for row in (before,after):r=motion.observe({v+"_rgb":rgb for v,rgb in row["rgb"].items()},row["depth"],m,row["q"])
        if not r["valid"]:rows.append({"decision":i,"valid":False,"reason":"HEAD_MOTION_UNKNOWN","motion":r});continue
        result=measure_grasp_motion(before,after,m,"right",target["point_base_m"],r["body_transform_current_in_previous"])
        result.update(decision=i,head_motion=r,new_model_calls=0,new_controls=0)
        canvas=Image.new("RGB",(960,480))
        for k,(time,f) in enumerate((("before",before),("after",after))):
            img=Image.fromarray(f["rgb"]["right_wrist"]).copy();draw=ImageDraw.Draw(img)
            for n,point in enumerate(result.get("tracked_pixels",[])):
                x,y=point[time];draw.ellipse((x-3,y-3,x+3,y+3),outline="cyan",width=2)
            canvas.paste(img,(480*k,0))
        canvas.save(out/f"decision_{i:03d}_tracks.jpg")
        rows.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ("tracked_pixels","head_motion")}),flush=True)
    (out/"result.json").write_text(json.dumps({"rows":rows,"diagnostic_only":True,"model_calls":0,"controls":0},indent=2))


if __name__=="__main__":main()
