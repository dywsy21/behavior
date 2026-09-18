"""Two saved-start plans, no simulator/actions, no retries or output selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.policy import VLMPolicy
from semantic_robot.v2.structured_planning import call_receipt
from semantic_robot.v2.vision import VisualBundle


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--starts",nargs=2,type=Path,required=True)
    p.add_argument("--instruction-runs",nargs=2,type=Path,required=True)
    p.add_argument("--uri",required=True);p.add_argument("--revision",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise ValueError("Clean immutable source required")
    a.output.mkdir(exist_ok=False)
    policy=VLMPolicy(a.uri,a.revision,max_calls=2,structured_planning=True)
    from PIL import Image
    started=time.monotonic()
    try:
        for task,run,old_run in zip((0,3),a.starts,a.instruction_runs):
            if time.monotonic()-started>300:raise TimeoutError("Static planning budget")
            m=json.loads((run/"manifest.json").read_text());old=json.loads((old_run/"manifest.json").read_text())
            if (m["task"],m["instance"],m["seed"])!=(task,old["instance"],old["seed"]) or old["task"]!=task:
                raise ValueError("Instruction task/instance mismatch")
            if m["args"]["prefix"] or m["args"].get("replay_prefix_spec"):
                raise ValueError("Require actual zero-prefix starts")
            f=old_run/"planner.json";text=json.loads(f.read_text())["request_without_pixel_duplicates"]["text"]
            marker="\nTask strategy (not current state): "
            if not text.startswith("Task: ") or marker not in text:raise ValueError("Unknown instruction receipt")
            instruction=text.split(marker,1)[0][len("Task: "):]
            labels=[];images=[];hashes=[]
            for view in ("HEAD","LEFT_WRIST","RIGHT_WRIST"):
                for suffix in ("RAW","ROBOT_GUIDE_NOT_OBJECT_LABELS"):
                    label=f"CURRENT_{view}_{suffix}";img=Image.open(run/(label+".png")).convert("RGB")
                    small=img.copy();small.thumbnail((640,640));labels.append(label);images.append(img)
                    hashes.append({"label":label,"size":list(small.size),"pixels_sha256":hashlib.sha256(small.tobytes()).hexdigest()})
            goals,call=policy.plan(task,instruction,VisualBundle(images,labels,{},{}))
            if call["result"]["images"]!=hashes:raise ValueError("Initial image hashes changed")
            row={"task":task,"source_run":str(run),"instruction_run":str(old_run),
                 "instruction_receipt_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"instruction":instruction,
                 "plan":[asdict(g) for g in goals],"call":call_receipt(call),"same_saved_start_images":True,
                 "semantic_task_coverage_requires_human_review":True,"controls":0}
            (a.output/f"task_{task}.json").write_text(json.dumps(row,indent=2))
            print(json.dumps({"task":task,"plan":row["plan"]}),flush=True)
        if time.monotonic()-started>300:raise TimeoutError("Static planning budget")
        (a.output/"result.json").write_text(json.dumps({"status":"complete","model_calls":policy.calls,
            "controls":0,"wall_s":time.monotonic()-started,"semantic_coverage_not_automatically_certified":True},indent=2))
    except BaseException as exc:
        if policy.last_call is not None:
            (a.output/"last_policy_call_before_failure.json").write_text(json.dumps(call_receipt(policy.last_call),indent=2))
        (a.output/"failure.json").write_text(json.dumps({"error":repr(exc),"model_calls":policy.calls,"controls":0},indent=2))
        raise


if __name__=="__main__":main()
