"""Counterfactual progress-veto audit on fixed old trajectories, not new rollouts."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.v2.approach_progress import ApproachProgress
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.odometry import RGBDMotion
from semantic_robot.v2.protocol import Action


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",action="append",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if len(a.run)>4:raise ValueError("At most four registered trajectories")
    start=time.monotonic();rows=[];count=0
    for value in a.run:
        root=Path(value);model=RobotModel(read(root/"robot_calibration.json"));motion=RGBDMotion("rgbd_rigid");monitor=ApproachProgress()
        commands={r["decision"]:r for r in map(json.loads,(root/"steps.jsonl").read_text().splitlines()) if "action" in r and "feedback" in r}
        for d in sorted(root.glob("decision_*")):
            count+=1
            if count>200 or time.monotonic()-start>600:raise RuntimeError("Fixed CPU audit budget reached")
            i=int(d.name.split("_")[-1]);q=np.asarray(read(d/"proprio.json")["q"])
            rgb=np.asarray(Image.open(d/"CURRENT_HEAD_RAW.png"));depth=dict(np.load(d/"depth.npz"))
            receipt=motion.observe({"head_rgb":rgb},depth,model,q)
            if not (d/"harness.json").exists():continue
            h=read(d/"harness.json");g=h.get("target_surface_estimate",{});arms=("left","right") if h["goal"]["hand"]=="both" else (h["goal"]["hand"],)
            points={a:np.asarray(g["hand_contacts"][a]["point_base_m"] if "hand_contacts" in g else g["point_base_m"]) for a in arms} if g.get("valid") else {}
            poses={a:model.forward(q,a).copy() for a in arms};centers=model.grasp_centers(q)
            for arm in poses:poses[arm][:3,3]=centers[arm]
            prev=commands.get(i-1,{})
            r=monitor.observe(goal_index=h["goal_index"],kind=h["goal"]["kind"],stage=h["stage"],points=points,
                poses=poses,execution=i,action=Action(**prev["action"]) if prev else None,
                feedback=prev.get("feedback"),motion=receipt,loaded=any(h.get("unverified_close_latches",{}).values()) or any(h["holding_verified_by_observation_and_proprio"].values()))
            rows.append({"run":root.name,"decision":i,"motion_valid":receipt["valid"],"progress":r})
            if r["events"]:print(json.dumps(rows[-1]),flush=True)
    result={"saved_frames":count,"wall_s":time.monotonic()-start,"rows":rows,"model_calls":0,"controls":0,
        "counterfactual_only_after_first_veto_original_future_states_are_not_a_new_policy_trajectory":True}
    (out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))


if __name__=="__main__":main()
