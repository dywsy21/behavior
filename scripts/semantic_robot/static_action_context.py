"""Audit saved actor prompts and optionally make <=2 non-actuating selections."""
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
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import GroundedPolicy,observation_context
from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.vision import prepare_views


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--state",action="append",required=True)
    p.add_argument("--output",required=True);p.add_argument("--tokenizer")
    p.add_argument("--uri");p.add_argument("--revision")
    args=p.parse_args()
    if not 1<=len(args.state)<=2:raise ValueError("At most two fixed saved states")
    if bool(args.uri)!=bool(args.revision):raise ValueError("Both service identity arguments required")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise ValueError("Fixed clean client source required")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    processor=None
    if args.tokenizer:
        from transformers import AutoProcessor
        processor=AutoProcessor.from_pretrained(args.tokenizer,local_files_only=True)
    policy=GroundedPolicy(args.uri,args.revision,max_calls=len(args.state)) if args.uri else None
    rows=[]
    for index,source in enumerate(map(Path,args.state)):
        saved=json.loads((source/"harness.json").read_text())
        original=json.loads((source/"observation.json").read_text())
        refine=json.loads((source/"refinement.json").read_text())
        observation=GroundedEvidence.parse(json.dumps(refine.get("refined_evidence",json.loads(original["result"]["text"]))))
        model=RobotModel(json.loads((source.parent/"robot_calibration.json").read_text()))
        proprio=json.loads((source/"proprio.json").read_text())
        state=model.state(np.asarray(proprio["q"]),np.asarray(proprio["gripper"]),np.zeros(3))
        images={v+"_rgb":np.asarray(Image.open(source/("CURRENT_"+v.upper()+"_RAW.png")))
                for v in ("head","left_wrist","right_wrist")}
        previous={v:np.asarray(Image.open(source/("PREVIOUS_"+v.upper()+"_RAW.png")))
                  for v in ("head","left_wrist","right_wrist")} if (source/"PREVIOUS_HEAD_RAW.png").exists() else None
        depths=dict(np.load(source/"depth.npz"));receipt=json.loads((source/"depth_receipt.json").read_text())
        for view in ("head","left_wrist","right_wrist"):
            for kind,value in (("rgb",images[view+"_rgb"]),("depth",depths[view])):
                if hashlib.sha256(value.tobytes()).hexdigest()!=receipt[view][kind+"_sha256"]:
                    raise ValueError("Current saved RGB-D mismatch")
        bundle=prepare_views(images,model,state.q,previous,grounded=True)
        # Preserve actual marked views AND their recorded numeric geometry.
        bundle.images=[Image.open(source/(label+".png")).convert("RGB") for label in bundle.labels]
        bundle.geometry=json.loads(original["request"]["text"])["current_robot"]["projection_guides"]
        manager=GroundedHarness([Goal(**saved["goal"])])
        manager.stage=saved["stage"];manager.observation=observation
        manager.held=saved["held_target_claims"];manager.hold_verified=saved["holding_verified_by_observation_and_proprio"]
        manager.pending_grasp=saved.get("unverified_close_latches",manager.pending_grasp)
        manager.context=lambda:saved  # frozen original snapshot, not claimed episode execution
        manager.candidate_receipt=json.loads((source/"candidates.json").read_text())
        allowed=tuple(Action(**row["action"]) for row in manager.candidate_receipt["tested"]
                      if row["accepted"] and row.get("offered_to_policy",True))
        suffix="\nChoose exactly one of these feasible complete commands:\n"+"\n".join(a.text() for a in allowed)
        old=observation_context(manager,state,bundle)+"\nVisible evidence: "+json.dumps(asdict(observation))
        old+="\nCURRENT preflight receipt (geometric gain is an estimate, NOT grasp success): "+json.dumps(manager.candidate_receipt)+suffix
        new=actor_context(manager,state,bundle,allowed)+"\nChoose one feasible command below; indices bind the scores, output ONLY its JSON:\n"+"\n".join(f"{index}: {a.text()}" for index,a in enumerate(allowed))
        # Capture the exact system/payload builder without an HTTP/model call.
        dry=GroundedPolicy.__new__(GroundedPolicy);captured={}
        def capture(kind,system,text,views,choices):
            captured.update(system=system,text=text)
            return {"text":choices[0].text()},{}
        dry._call=capture;dry.act_feasible(manager,state,bundle,allowed)
        if captured["text"]!=new:raise ValueError("Static/live prompt builder drift")
        row={"source":str(source),"source_pixel_depth_hashes_match":True,"old_chars":len(old),"new_chars":len(new),
             "unchanged_allowed_commands":[a.text() for a in allowed],"images":bundle.labels,
             "new_controls":0,"new_training_updates":0,"manual_review_required":True}
        if processor:
            for name,text in (("old",old),("new",new)):
                content=[]
                for label,im in zip(bundle.labels,bundle.images):
                    image=im.copy();image.thumbnail((640,640))
                    content.extend([{"type":"text","text":label},{"type":"image","image":image}])
                content.append({"type":"text","text":text})
                value=processor.apply_chat_template([{"role":"system","content":captured["system"]},
                    {"role":"user","content":content}],tokenize=True,add_generation_prompt=True,
                    return_dict=True,return_tensors="pt",enable_thinking=False)
                row[name+"_input_tokens"]=int(value["input_ids"].shape[1])
            if row["new_input_tokens"]>12000:raise ValueError("Compacted prompt still exceeds original cap")
        if policy:
            action,call=policy.act_feasible(manager,state,bundle,allowed)
            call["request"]["images"]=[{"label":item["label"]} for item in call["request"]["images"]]
            row.update(selected_action=asdict(action),call=call)
        target=output/f"case_{index:02d}";target.mkdir()
        (target/"old_prompt.txt").write_text(captured["system"]+"\n"+old)
        (target/"new_prompt.txt").write_text(captured["system"]+"\n"+new)
        (target/"result.json").write_text(json.dumps(row,indent=2,allow_nan=False));rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ("call","unchanged_allowed_commands","images")}),flush=True)
    result={"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
            "model_calls":policy.calls if policy else 0,"controls":0,"rows":rows}
    (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
