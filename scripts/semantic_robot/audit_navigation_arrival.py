"""Read-only saved navigation workspace checks; no model or simulator calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.navigation import navigation_workspace_check


def main():
    p=argparse.ArgumentParser();p.add_argument("--state",action="append",required=True);p.add_argument("--output",required=True)
    a=p.parse_args()
    if not 1<=len(a.state)<=4:raise ValueError("At most four recorded states")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise RuntimeError("Fixed clean source required")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[]
    for value in a.state:
        d=Path(value);calibration=(d.parent/"robot_calibration.json").read_bytes()
        m=RobotModel(json.loads(calibration));h=json.loads((d/"harness.json").read_text());pr=json.loads((d/"proprio.json").read_text())
        arms=("left","right") if h["goal"]["hand"]=="both" else (h["goal"]["hand"],)
        r=navigation_workspace_check(m,np.asarray(pr["q"]),h["target_surface_estimate"],arms)
        row={"source":str(d),"saved_goal":h["goal"],"saved_stage":h["stage"],"workspace":r,
            "calibration_sha256":hashlib.sha256(calibration).hexdigest(),"controls":0,"model_calls":0}
        rows.append(row);print(json.dumps(row),flush=True)
    (out/"result.json").write_text(json.dumps({"rows":rows,"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),"controls":0,"model_calls":0},indent=2))


if __name__=="__main__":main()
