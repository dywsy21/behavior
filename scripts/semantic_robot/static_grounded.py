"""Three fixed saved gate states, <=6 calls, no simulator or action execution."""
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
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import prepare_views


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--radio-gate",required=True)
    p.add_argument("--plates-gate",required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--uri",default="http://127.0.0.1:8907")
    args=p.parse_args()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Clean pinned source required")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    policy=GroundedPolicy(args.uri,args.revision,max_calls=6)
    rows=[]
    # Fixed before reading new model outputs. Two radio poses and one invisible
    # table search state; correlated development sanity, not recognition SR.
    for index,(root,step) in enumerate(((args.radio_gate,0),(args.radio_gate,18),(args.plates_gate,0))):
        root=Path(root);source=root/f"decision_{step:03d}"
        model=RobotModel(json.loads((root/"robot_calibration.json").read_text()))
        proprio=json.loads((source/"proprio.json").read_text())
        state=model.state(np.asarray(proprio["q"]),np.asarray(proprio["gripper"]),np.zeros(3))
        images={v+"_rgb":np.asarray(Image.open(source/("CURRENT_"+v.upper()+"_RAW.png"))) for v in ("head","left_wrist","right_wrist")}
        depths=dict(np.load(source/"depth.npz"))
        receipt=json.loads((source/"depth_receipt.json").read_text())
        bundle=prepare_views(images,model,state.q,grounded=True)
        goal=(Goal("pick","radio on the coffee table","right","radio moves with gripper") if index<2 else
              Goal("navigate","breakfast table with pizza plates","right","table is visibly close enough to approach plates"))
        manager=GroundedHarness([goal]);servo=SafeServo(model,state)
        controller=GroundedController(model,servo,manager)
        row={"index":index,"source":str(source),"calibration_sha":model.sha,"calls":[],"actuations":0}
        try:
            obs,call=policy.observe(manager,state,bundle)
            row["calls"].append({"kind":"observe","result":call["result"]})
            row["observation"]=asdict(obs)
            controller.observe(obs,state,depths,receipt)
            if index==2:
                # Exercise the recovery schema without applying it or consuming
                # a real episode's recovery allowance.
                strategy,call=policy.recover(manager,state,bundle,"Saved-state schema check; no actuation")
                row["calls"].append({"kind":"recovery","result":call["result"]})
                row["recovery"]=strategy
            else:
                allowed=controller.candidates(state)
                action,call=policy.act_feasible(manager,state,bundle,allowed)
                row["calls"].append({"kind":"act","result":call["result"]})
                row["action"]=asdict(action)
                row["candidates"]=manager.candidate_receipt
            row["grounding"]=manager.grounding
            row["schema_ok"]=True
        except Exception as exc:
            row["error"]=repr(exc);row["schema_ok"]=False
        rows.append(row)
        (out/f"case_{index:02d}.json").write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)
    result={"status":"complete","code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
            "calls":policy.calls,"actuations":0,"training_updates":0,"model_identity":policy.identity,
            "schema_ok":all(r["schema_ok"] for r in rows),"rows":rows,"human_grounding_review_required":True}
    (out/"result.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":main()
