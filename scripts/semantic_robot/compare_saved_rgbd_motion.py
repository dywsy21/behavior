"""Bounded paired pose diagnostics on saved pixels; no simulator or model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.odometry import RGBDMotion


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--spec",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();spec=read(Path(a.spec));out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    if sum(map(len,spec["saved_pairs"].values()))>spec["budget"]["CPU_pairs"]:raise ValueError("Pair budget exceeded")
    for name,indices in spec["saved_pairs"].items():
        root=Path(a.root)/name;model=RobotModel(read(root/"robot_calibration.json"))
        for i in indices:
            if time.monotonic()-started>spec["budget"]["wall_seconds"]:raise TimeoutError("Declared CPU time reached")
            solvers={key:RGBDMotion(key) for key in ("pnp","rgbd_rigid")};result={}
            for j in (i-1,i):
                d=root/f"decision_{j:03d}";q=np.asarray(read(d/"proprio.json")["q"])
                rgb=np.asarray(Image.open(d/"CURRENT_HEAD_RAW.png"));depths=dict(np.load(d/"depth.npz"))
                for key,solver in solvers.items():result[key]=solver.observe({"head_rgb":rgb},depths,model,q)
            row={"run":name,"decision":i,"estimators":result};rows.append(row)
            print(json.dumps({"run":name,"decision":i,"estimates":{k:{f:v.get(f) for f in ("valid","reason","body_delta","body_translation_z_m","median_depth_correspondence_m","median_reprojection_px","inliers")} for k,v in result.items()}}),flush=True)
    result={"spec_sha256":hashlib.sha256(Path(a.spec).read_bytes()).hexdigest(),"rows":rows,
        "wall_s":time.monotonic()-started,"new_model_calls":0,"new_controls":0,"absolute_pose_truth_available":False}
    (out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
