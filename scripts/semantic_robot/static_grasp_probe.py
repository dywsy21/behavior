"""At most two fixed saved states, recomputed feasible actions, ZERO controls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.grounded_harness import GroundedHarness,GroundedController
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import VisualBundle


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--state",action="append",required=True)
    p.add_argument("--uri",required=True);p.add_argument("--revision",required=True)
    p.add_argument("--output",required=True);p.add_argument("--prepare-only",action="store_true")
    a=p.parse_args()
    if not 1<=len(a.state)<=2:raise ValueError("At most two fixed states")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Clean fixed source required")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    policy=None if a.prepare_only else GroundedPolicy(a.uri,a.revision,max_calls=len(a.state))
    rows=[]
    for i,path in enumerate(a.state):
        d=Path(path);saved=json.loads((d/"harness.json").read_text());pr=json.loads((d/"proprio.json").read_text())
        m=RobotModel(json.loads((d.parent/"robot_calibration.json").read_text()))
        state=m.state(np.asarray(pr["q"]),np.asarray(pr["gripper"]),np.zeros(3))
        h=GroundedHarness([Goal(**saved["goal"])],active_grasp_probe=True)
        h.stage=saved["stage"];h.held=saved["held_target_claims"];h.hold_verified=saved["holding_verified_by_observation_and_proprio"]
        h.pending_grasp=saved["unverified_close_latches"]
        old=json.loads((d/"observation.json").read_text());ref=json.loads((d/"refinement.json").read_text())
        obs=GroundedEvidence.parse(json.dumps(ref.get("refined_evidence",json.loads(old["result"]["text"]))))
        c=json.loads((d/"candidates.json").read_text());servo=SafeServo(m,state,c["command_grip_latch"])
        ctl=GroundedController(m,servo,h)
        ctl.observe(obs,state,dict(np.load(d/"depth.npz")),json.loads((d/"depth_receipt.json").read_text()))
        h.stage=saved["stage"];h.update_grasp_probe(state)
        # Preserve the real past events; do not invent successful attempts.
        for r in saved["recent_executed"]:h.history.append(r)
        allowed=ctl.candidates(state)
        original=json.loads((d/"action.json").read_text())
        labels=[x["label"] for x in original["request_without_pixel_duplicates"]["images"]]
        images=[Image.open(d/(label+".png")).convert("RGB") for label in labels]
        raw={v:np.asarray(Image.open(d/("CURRENT_"+v.upper()+"_RAW.png"))) for v in ("head","left_wrist","right_wrist")}
        bundle=VisualBundle(images,labels,pr["geometry"],raw)
        row={"source":str(d),"probe":h.grasp_probe,"allowed":[asdict(x) for x in allowed],"controls":0}
        if policy:
            action,call=policy.act_feasible(h,state,bundle,allowed)
            row["action"]=asdict(action);row["result"]=call["result"]
            call["request"]["images"]=[{"label":x} for x in labels]
            row["request_without_pixels"]=call["request"]
            row["same_image_bytes_as_original"]=original["result"]["images"]==call["result"]["images"]
            if not row["same_image_bytes_as_original"]:raise ValueError("Static comparison changed images")
        rows.append(row);(out/f"case_{i:02d}.json").write_text(json.dumps(row,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k in ("source","probe","action","same_image_bytes_as_original")}),flush=True)
    (out/"result.json").write_text(json.dumps({"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),"model_calls":policy.calls if policy else 0,"controls":0,"rows":rows},indent=2))


if __name__=="__main__":main()
