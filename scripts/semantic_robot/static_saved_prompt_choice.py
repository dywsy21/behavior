"""At most two exact saved-inspection choices; no physics, retries or new images."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from audit_saved_prompt_budget import REPO, saved_prompt
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.protocol import Action


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--uri",required=True);p.add_argument("--revision",required=True)
    a=p.parse_args()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise ValueError("Use immutable clean source")
    a.output.mkdir(exist_ok=False);policy=GroundedPolicy(a.uri,a.revision,max_calls=2)
    started=time.monotonic();rows=[]
    try:
        for index in (4,5):
            if time.monotonic()-started>300:raise TimeoutError("Static gate budget reached")
            payload,original=saved_prompt(a.run,index)
            result,request=policy._call("act",payload["system"],payload["text"],payload["bundle"],payload["allowed"])
            action=Action.parse(result["text"])
            if action not in payload["allowed"]:raise ValueError("Choice outside exact saved feasible commands")
            if result["images"]!=original["result"]["images"]:raise ValueError("Original nine image hashes changed")
            # Retain the request text, command indices and canonical choices;
            # image bytes already reside in the unchanged source observation.
            request={**request,"images":[{"label":r["label"]} for r in request["images"]]}
            row={"decision":index,"action":asdict(action),"result":result,"request":request,
                 "same_original_images":True,"controls":0,"not_task_success":True}
            (a.output/f"decision_{index:03d}.json").write_text(json.dumps(row,indent=2))
            rows.append(row);print(json.dumps({"decision":index,"action":asdict(action),"input_tokens":result["input_tokens"]}),flush=True)
        if time.monotonic()-started>300:raise TimeoutError("Static gate budget reached")
        (a.output/"result.json").write_text(json.dumps({"status":"complete","model_calls":policy.calls,
            "controls":0,"wall_s":time.monotonic()-started,"decisions":[r["decision"] for r in rows],
            "not_task_success":True},indent=2))
    except BaseException as exc:
        (a.output/"failure.json").write_text(json.dumps({"error":repr(exc),"model_calls":policy.calls,"controls":0},indent=2))
        raise


if __name__=="__main__":main()
