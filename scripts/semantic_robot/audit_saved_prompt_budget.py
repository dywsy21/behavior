"""Exact saved-candidate prompt CPU checks, including a labelled history stress case.

Never generates, selects an action, runs IK, changes source data, or actuates.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2 import prompt_context
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.protocol import Action, HOLD
from semantic_robot.v2.vision import VisualBundle


class Captured(Exception):pass


class CapturePolicy(GroundedPolicy):
    def __init__(self):self.payload=None

    def _call(self,kind,system,text,bundle,allowed=()):
        self.payload={"kind":kind,"system":system,"text":text,"bundle":bundle,"allowed":allowed}
        raise Captured("CPU-only prompt construction ends before any action choice")


def saved_prompt(run,index,stress_history=False):
    d=run/f"decision_{index:03d}"
    read=lambda p:json.loads(p.read_text())
    saved=read(d/"harness.json");candidates=read(d/"candidates.json")
    if saved["stage"] not in ("SEARCH","RECOVER") or not saved["target_reference"].startswith("held_"):
        raise ValueError("Only saved held-inspection candidate receipts are scoped here")
    if stress_history:
        history=saved["recent_executed"]
        if not history:raise ValueError("No actual history to repeat in synthetic pressure case")
        saved["recent_executed"]=[copy.deepcopy(history[i%len(history)]) for i in range(8)]
    flags=read(run/"manifest.json")["args"]
    h=SimpleNamespace(context=lambda:copy.deepcopy(saved),candidate_receipt=candidates,
        observation=GroundedEvidence.parse(read(d/"observation.json")["result"]["text"]),
        contact_geometry=flags.get("contact_geometry",False),held_inspection_enabled=True,
        search_reference=saved["target_reference"],stage=saved["stage"],
        inspection_budget_aware=True,multicamera_inspection=True,
        grasp_probe=saved.get("active_grasp_probe",{}))
    allowed=(HOLD,)+tuple(Action(**r["action"]) for r in candidates["tested"] if r["accepted"])
    if len(set(allowed))!=len(allowed):raise ValueError("Duplicate saved command")
    model=RobotModel(read(run/"robot_calibration.json"));pr=read(d/"proprio.json")
    state=model.state(np.array(pr["q"]),np.array(pr["gripper"]),np.zeros(3))
    source=d/"action.json"
    if not source.exists():source=d/"observation.json"
    original=read(source);labels=[x["label"] for x in original["result"]["images"]]
    images=[Image.open(d/(name+".png")).convert("RGB") for name in labels]
    bundle=VisualBundle(images,labels,pr["geometry"],{})
    policy=CapturePolicy()
    try:policy.act_feasible(h,state,bundle,allowed)
    except Captured:pass
    else:raise RuntimeError("Prompt capture must never produce an action")
    return policy.payload,original


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",action="append",required=True,help="Absolute run path:decision")
    p.add_argument("--model",required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--legacy-scores",action="store_true")
    a=p.parse_args()
    if a.output.exists():raise ValueError("Preserve previous budget evidence")
    if a.legacy_scores:prompt_context.columnar_scores=lambda scores:scores
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    start=time.monotonic();rows=[]
    for case in a.case:
        path,index=case.rsplit(":",1)
        for stress in (False,True):
            payload,original=saved_prompt(Path(path),int(index),stress)
            content=[];hashes=[]
            for label,image in zip(payload["bundle"].labels,payload["bundle"].images):
                image=image.copy();image.thumbnail((640,640))
                hashes.append({"label":label,"size":list(image.size),"pixels_sha256":hashlib.sha256(image.tobytes()).hexdigest()})
                content.extend([{"type":"text","text":label},{"type":"image","image":image}])
            if hashes!=original["result"]["images"]:raise ValueError("Original nine-view input drift")
            content.append({"type":"text","text":payload["text"]})
            messages=[{"role":"system","content":payload["system"]},{"role":"user","content":content}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors="pt",enable_thinking=False)
            tokens=int(inputs["input_ids"].shape[1]);chars=len(payload["system"])+len(payload["text"])
            old=original.get("request_without_pixel_duplicates",{})
            same=(payload["system"]==old.get("system") and payload["text"]==old.get("text")) if (Path(path)/f"decision_{int(index):03d}"/"action.json").exists() and not stress else None
            rows.append({"case":case,"synthetic_eight_history_pressure_only":stress,"choices":len(payload["allowed"]),
                "input_tokens":tokens,"characters":chars,"same_original_images":True,
                "matches_saved_successful_prompt":same,"within_budget":tokens<=12000 and chars<=32000})
    result={"legacy_scores":a.legacy_scores,"rows":rows,"model_calls":0,"controls":0,
            "wall_s":time.monotonic()-start,"passed":all(r["within_budget"] for r in rows)}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
