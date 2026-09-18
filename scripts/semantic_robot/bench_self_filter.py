"""Exact old/new chassis-mask comparison on a saved robot-only calibration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.grounding import observed_cloud,LocalDepthGuard
from semantic_robot.v2.self_filter import ChassisSurface
from semantic_robot.v2.protocol import Action


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--calibration",required=True);p.add_argument("--state",required=True)
    p.add_argument("--output",required=True);args=p.parse_args()
    state_path=Path(args.state);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    saved=json.loads((state_path.parent/"robot_calibration.json").read_text())
    mesh_cal=json.loads(Path(args.calibration).read_text())
    if saved["metadata"]["joint_names"]!=mesh_cal["metadata"]["joint_names"]:
        raise ValueError("Robot joint definitions differ")
    mesh=mesh_cal["metadata"]["base_visual_surface"]
    model=RobotModel(saved)
    q=np.asarray(json.loads((state_path/"proprio.json").read_text())["q"])
    depths=dict(np.load(state_path/"depth.npz"))
    cloud=observed_cloud(depths,model,q);T=model.forward(q,mesh["link"])
    # Run this repository's exact previous implementation in memory; no old
    # source files/worktrees are edited and no simulator is opened.
    baseline_source=subprocess.check_output(["git","-C",str(REPO),"show","39881cf:src/semantic_robot/v2/self_filter.py"],text=True)
    namespace={"__name__":"pinned_chassis_baseline"}
    exec(compile(baseline_source,"39881cf:self_filter.py","exec"),namespace)
    old=namespace["ChassisSurface"](mesh);new=ChassisSurface(mesh)
    start=time.perf_counter();old_mask=old.mask(cloud,T);old_s=time.perf_counter()-start
    start=time.perf_counter();new_mask=new.mask(cloud,T);new_s=time.perf_counter()-start
    if not np.array_equal(old_mask,new_mask):raise ValueError("Optimized mesh mask changed classifications")
    old_guard=LocalDepthGuard(cloud,model,q,depths)
    current=copy.deepcopy(saved);current["metadata"]["base_visual_surface"]=mesh
    updated=RobotModel(current);new_guard=LocalDepthGuard(cloud,updated,q,depths)
    result={"state":str(state_path),"calibration":args.calibration,"points":len(cloud),
            "triangles":len(mesh["faces"]),"mask_equal":True,"masked_points":int(new_mask.sum()),
            "baseline_seconds":old_s,"optimized_seconds":new_s,"speedup":old_s/max(new_s,1e-9),
            "optimized_source_sha256":hashlib.sha256((REPO/"src/semantic_robot/v2/self_filter.py").read_bytes()).hexdigest(),
            "old_guard_without_chassis":old_guard.receipt(),"new_guard":new_guard.receipt(),
            "base_checks":{d:{"before":old_guard.check(Action("base",d)),"after":new_guard.check(Action("base",d))}
                           for d in ("forward","back","left","right")},"model_calls":0,"controls":0}
    (output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
