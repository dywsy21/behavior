"""Two preregistered H-07 saved observations; <=2 choices, no robot controls."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.affordance import surface_candidates, refinement_bundle
from semantic_robot.v2.grounding import GroundedEvidence, localize_target
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import RefinedGroundedPolicy
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import prepare_views


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",required=True,help="Original H-07 root with radio_27b_v1/server_27b_v2")
    p.add_argument("--output",required=True)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--revision")
    p.add_argument("--uri",default="http://127.0.0.1:8907")
    args=p.parse_args()
    dirty=subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip()
    if not args.prepare_only and (dirty or not args.revision):
        raise ValueError("Neural static checks require clean pinned source/revision")
    root=Path(args.source_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    calls={x["call"]:x for x in map(json.loads,(root/"server_27b_v2/calls.jsonl").read_text().splitlines())}
    run=root/"radio_27b_v1"
    model=RobotModel(json.loads((run/"robot_calibration.json").read_text()))
    policy=None if args.prepare_only else RefinedGroundedPolicy(args.uri,args.revision,max_calls=2,max_refinements=2)
    rows=[]
    for decision,call_id in ((0,4),(1,6)):
        source=run/f"decision_{decision:03d}";target=out/f"case_{decision:02d}";target.mkdir()
        proprio=json.loads((source/"proprio.json").read_text())
        state=model.state(np.asarray(proprio["q"]),np.asarray(proprio["gripper"]),np.zeros(3))
        images={v+"_rgb":np.asarray(Image.open(source/("CURRENT_"+v.upper()+"_RAW.png"))) for v in ("head","left_wrist","right_wrist")}
        depths=dict(np.load(source/"depth.npz"))
        evidence=GroundedEvidence.parse(calls[call_id]["text"])
        manager=GroundedHarness([Goal("pick","red and white radio on the coffee table","right","radio moves with gripper")])
        manager.stage="APPROACH"
        bundle=prepare_views(images,model,state.q,grounded=True)
        proposals=surface_candidates(evidence,depths,model,state.q)
        detail=refinement_bundle(bundle,proposals)
        for label,img in zip(detail.labels,detail.images):img.save(target/(label+".png"))
        row={"source":str(source),"old_call":call_id,"original_text":calls[call_id]["text"],
             "calibration_sha":model.sha,"original_target":localize_target(evidence,depths,model,state.q),
             "surface_proposals":proposals,"raw_pixels_sha256":{
                 v:hashlib.sha256(x.tobytes()).hexdigest() for v,x in images.items()},
             "source_depth_file_sha256":hashlib.sha256((source/"depth.npz").read_bytes()).hexdigest(),
             "source_proprio_sha256":hashlib.sha256((source/"proprio.json").read_bytes()).hexdigest(),
             "actuations":0}
        if policy:
            chosen,call,receipt,_=policy.refine(evidence,manager,state,bundle,depths,model)
            if call:
                call["request"]["images"]=[{"label":x["label"]} for x in call["request"]["images"]]
            row.update(call=call,refinement=receipt,selected_evidence=asdict(chosen))
            servo=SafeServo(model,state)
            controller=GroundedController(model,servo,manager)
            controller.observe(chosen,state,depths,json.loads((source/"depth_receipt.json").read_text()))
            allowed=controller.candidates(state)
            row["preflighted_actions_not_executed"]=[asdict(a) for a in allowed]
            row["candidate_receipt"]=manager.candidate_receipt
        rows.append(row);write(target/"result.json",row)
        print(json.dumps({"case":decision,"neural":bool(policy),"proposals":len(proposals["candidates"]),
                          "choice":row.get("refinement",{}).get("choice"),"actuations":0}),flush=True)
    write(out/"result.json",{"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
        "dirty_prepare_only":bool(dirty),"calls":0 if policy is None else policy.calls,
        "model_identity":None if policy is None else policy.identity,"actuations":0,"training_updates":0,
        "human_grounding_review_required":True,"rows":rows})


if __name__=="__main__":main()
