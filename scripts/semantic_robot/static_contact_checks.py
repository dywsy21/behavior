"""At most four saved RGB-D contact checks; no simulator or actuator imports."""
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
from semantic_robot.v2.affordance import surface_candidates,refinement_bundle
from semantic_robot.v2.bimanual import localize_hand_contacts
from semantic_robot.v2.grounding import GroundedEvidence,localize_target
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import RefinedGroundedPolicy
from semantic_robot.v2.vision import prepare_views


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--state",action="append",required=True)
    p.add_argument("--mode",choices=("refine","bimanual","observe"),required=True)
    p.add_argument("--bimanual-target")
    p.add_argument("--uri",required=True);p.add_argument("--revision",required=True)
    p.add_argument("--output",required=True);p.add_argument("--prepare-only",action="store_true")
    args=p.parse_args()
    if not 1<=len(args.state)<=4:raise ValueError("At most four fixed observations per check")
    if (args.mode=="bimanual") != bool(args.bimanual_target):raise ValueError("Explicit counterfactual bimanual target required")
    if not args.prepare_only and subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Clean fixed client source required")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    policy=None if args.prepare_only else RefinedGroundedPolicy(args.uri,args.revision,
        max_calls=len(args.state),max_refinements=len(args.state))
    rows=[]
    for index,name in enumerate(args.state):
        source=Path(name);target=output/f"case_{index:02d}";target.mkdir()
        old=json.loads((source/"observation.json").read_text())
        saved=json.loads((source/"harness.json").read_text())
        model=RobotModel(json.loads((source.parent/"robot_calibration.json").read_text()))
        proprio=json.loads((source/"proprio.json").read_text())
        state=model.state(np.asarray(proprio["q"]),np.asarray(proprio["gripper"]),np.zeros(3))
        images={v+"_rgb":np.asarray(Image.open(source/("CURRENT_"+v.upper()+"_RAW.png")))
                for v in ("head","left_wrist","right_wrist")}
        depths=dict(np.load(source/"depth.npz"))
        receipt=json.loads((source/"depth_receipt.json").read_text())
        for view in ("head","left_wrist","right_wrist"):
            if hashlib.sha256(images[view+"_rgb"].tobytes()).hexdigest()!=receipt[view]["rgb_sha256"]:
                raise ValueError("Saved current image mismatch")
            if hashlib.sha256(depths[view].tobytes()).hexdigest()!=receipt[view]["depth_sha256"]:
                raise ValueError("Saved depth mismatch")
        goal=(Goal("pick",args.bimanual_target,"both","both hands support the same object after a small lift",True)
              if args.mode=="bimanual" else Goal(**saved["goal"]))
        manager=GroundedHarness([goal]);manager.stage="APPROACH" if args.mode=="bimanual" else saved["stage"]
        bundle=prepare_views(images,model,state.q,grounded=True)
        for label,img in zip(bundle.labels,bundle.images):img.save(target/(label+".png"))
        row={"source":str(source),"mode":args.mode,"original_goal":saved["goal"],"checked_goal":asdict(goal),
             "counterfactual_subgoal_not_actual_episode_progress":args.mode=="bimanual",
             "source_pixel_depth_hashes_match":True,"controls":0,"training_updates":0,"manual_review_required":True}
        call=None
        if args.mode=="refine":
            observation=GroundedEvidence.parse(old["result"]["text"])
            proposals=surface_candidates(observation,depths,model,state.q)
            row.update(original_evidence=asdict(observation),original_grounding=localize_target(observation,depths,model,state.q))
            views=refinement_bundle(bundle,proposals)
            for label,img in zip(views.labels,views.images):img.save(target/(label+".png"))
            row["proposals"]=proposals
            if policy:
                selected,call,refinement,_=policy.refine(observation,manager,state,bundle,depths,model)
                row.update(refinement=refinement,selected_evidence=asdict(selected),
                           selected_grounding=localize_target(selected,depths,model,state.q))
        elif policy:
            observed,call=policy.observe(manager,state,bundle)
            row.update(observed_evidence=asdict(observed))
            if args.mode=="bimanual":row["hand_grounding"]=localize_hand_contacts(observed,depths,model,state.q)
            else:row["selected_grounding"]=localize_target(observed,depths,model,state.q)
        if call:
            call["request"]["images"]=[{"label":x["label"]} for x in call["request"]["images"]]
            row["call"]=call
        (target/"result.json").write_text(json.dumps(row,indent=2,allow_nan=False))
        rows.append(row)
        print(json.dumps({"source":str(source),"mode":args.mode,"choice":row.get("refinement",{}).get("choice"),
                          "evidence":row.get("observed_evidence"),"controls":0}),flush=True)
    result={"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
            "model_identity":policy.identity if policy else None,"model_calls":policy.calls if policy else 0,
            "controls":0,"training_updates":0,"rows":rows}
    (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
