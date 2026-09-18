"""Four original B7 states, <=4 observations/0 controls; no rollout replay claim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.grounded_harness import GroundedHarness,metric_co_motion
from semantic_robot.v2.grounding import GroundedEvidence,localize_target
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.vision import prepare_views


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True)
    p.add_argument("--uri");p.add_argument("--revision");a=p.parse_args()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise RuntimeError("Fixed clean source required")
    run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    m=RobotModel(read(run/"robot_calibration.json"));policy=GroundedPolicy(a.uri,a.revision,max_calls=4) if a.uri else None
    if policy:policy.paired_grasp_verification=True  # reproduce the B10 ablation, not current production
    rows=[]
    for i in (10,22,23,24,25):
        d=run/f"decision_{i:03d}";before=run/f"decision_{i-1:03d}"
        saved=read(d/"harness.json");pr=read(d/"proprio.json");oldpr=read(before/"proprio.json")
        state=m.state(np.asarray(pr["q"]),np.asarray(pr["gripper"]),np.zeros(3))
        h=GroundedHarness([Goal(**saved["goal"])],active_grasp_probe=True)
        h.stage="VERIFY_GRASP";h.stage_age=3 if i==25 else 0
        h.pending_grasp=saved["unverified_close_latches"].copy()
        last=saved["recent_executed"][-1];h.last_action=Action(**last["action"]);h.feedback=last["feedback"]
        h.observation=GroundedEvidence.parse(read(before/"observation.json")["result"]["text"])
        for item in saved["recent_executed"]:h.history.append(item)
        raw={v:np.asarray(Image.open(d/("CURRENT_"+v.upper()+"_RAW.png"))) for v in ("head","left_wrist","right_wrist")}
        previous={v:np.asarray(Image.open(d/("PREVIOUS_"+v.upper()+"_RAW.png"))) for v in raw}
        bundle=prepare_views({v+"_rgb":x for v,x in raw.items()},m,state.q,previous,grounded=True,gripper=state.gripper)
        receipt=read(d/"depth_receipt.json");depths=dict(np.load(d/"depth.npz"))
        for v in raw:
            if hashlib.sha256(raw[v].tobytes()).hexdigest()!=receipt[v]["rgb_sha256"] or hashlib.sha256(depths[v].tobytes()).hexdigest()!=receipt[v]["depth_sha256"]:raise ValueError("Raw RGB-D hash mismatch")
        row={"decision":i,"stage_before_observation_explicitly_restored":"VERIFY_GRASP","controls":0,"source":str(d),"raw_rgbd_hashes_unchanged":True}
        obs=GroundedEvidence.parse(read(d/"observation.json")["result"]["text"])
        if policy and i!=25:
            obs,call=policy.observe(h,state,bundle);payload=call.pop("request")
            row["call"]=call;row["request_without_pixels"]={k:v for k,v in payload.items() if k!="images"}
            original={r["label"]:r for r in read(d/"observation.json")["result"]["images"]}
            if any(original[r["label"]]!=r for r in call["result"]["images"]):raise ValueError("Selected raw service pixels changed")
            row["service_raw_subset_hashes_unchanged"]=True
        target=localize_target(obs,depths,m,state.q)
        prev={"target":read(before/"harness.json")["target_surface_estimate"],"centers":m.grasp_centers(np.asarray(oldpr["q"]))}
        metric=metric_co_motion(prev,target,state,m.grasp_centers(state.q),h.feedback,"right")
        h.observe(obs,state,measured_progress={"metric_co_motion":metric is True})
        row.update(observation=obs.as_dict(),metric_co_motion=metric,stage=h.stage,stop_reason=h.stop_reason,
            saved_one_step_completion=bool(h.completed),open_offered=any(x.move=="open" for x in h.palette()),
            not_a_new_episode_or_task_success=True)
        (out/f"decision_{i:03d}.json").write_text(json.dumps(row,indent=2));rows.append(row)
        print(json.dumps({k:row[k] for k in ("decision","metric_co_motion","stop_reason","saved_one_step_completion","open_offered")}),flush=True)
    result={"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
            "model_calls":policy.calls if policy else 0,"controls":0,"rows":rows}
    (out/"result.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":main()
