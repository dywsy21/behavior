"""JEV-03 non-actuating probe: real saved search sensor snapshot + pinned states.

No simulator imports, GPU, new perception, post-action truth or control output.
Not a rollout: every choice here is hypothetical and must never be replayed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.jev_client import JevClient, JevAbstained
from semantic_robot.v2.jev_control import candidates, authority
from semantic_robot.v2.jev_policy import JevGroundedPolicy, JevDecisionPolicy
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.servo import SafeServo, ServoLimits
from test_jev_saved import saved_context


def search_input(root):
    root=Path(root)
    spec=json.loads((REPO/"configs/semantic_robot/jev_search_source_v1.json").read_text())
    for name,digest in spec["files"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Frozen search source changed: "+name)
    def read(name):return json.loads((root/name).read_text())
    manifest=read("manifest.json")
    if (manifest.get("actor_scene_truth") is not False or manifest.get("prefix_is_expert_not_agent") is not False
            or manifest.get("instance")!=138 or manifest.get("task")!=0):
        raise ValueError("Only frozen original-start nonprivileged actor input")
    prior=read("decision_000/harness.json")
    if prior["stage"]!="SEARCH" or prior.get("goal_index")!=0 or prior.get("recent_executed"):
        raise ValueError("Search reconstruction only supports initial state, no invented history")
    model=RobotModel(read("robot_calibration.json"))
    q=read("decision_000/proprio.json")
    state=model.state(q["q"],q["gripper"],np.zeros(3))
    h=GroundedHarness([Goal(**prior["goal"])])
    h.jev_decision_owner=True
    # Initial open command follows the recorded run's known reset protocol,
    # not aperture -> command inference for a possibly loaded later state.
    servo=SafeServo(model,state,[1,1],ServoLimits(robot_geometry_guards=True))
    c=GroundedController(model,servo,h)
    evidence=GroundedEvidence.parse(read("decision_000/observation.json")["result"]["text"])
    with np.load(root/"decision_000/depth.npz",allow_pickle=False) as archive:
        depths={name:archive[name].copy() for name in archive.files}
    c.observe(evidence,state,depths,read("decision_000/depth_receipt.json"),
              self_geometry=read("decision_000/robot_self_geometry.json"))
    allowed,mode=candidates(c,state)
    if mode!="world_search" or h.stop_reason or not allowed:
        raise ValueError("Saved input failed new search candidate reconstruction")
    return json.loads(actor_context(h,state,SimpleNamespace(geometry={}),allowed)),allowed,{
        "source":spec,"path":str(root.resolve()),"mode":mode,
        "new_palette_not_original_executed_action":True,"reconstructed_preflight":h.candidate_receipt}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-source",required=True)
    parser.add_argument("--state",action="append",default=[])
    parser.add_argument("--output",required=True)
    parser.add_argument("--key-file")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if "TYPESAFE_API_KEY" in os.environ:
        raise ValueError("Inherited TypeSafe environment forbidden; use explicit private key file")
    if not args.dry_run and not args.key_file:
        raise ValueError("Live probe requires explicit --key-file")
    if args.dry_run and args.key_file:
        raise ValueError("Dry probe must not receive credentials")
    if len(args.state)>6:raise ValueError("At most six pinned previous actor states")
    inputs=[search_input(args.search_source),*[saved_context(p) for p in args.state]]
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise ValueError("Freeze clean source before offline experiment")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    budget=3+2*len(inputs)  # two required goals + semantic reference + choices
    result=dict(schema="jev-all-offline-v1",code_commit=subprocess.check_output(
        ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
        mode="dry" if args.dry_run else "live_api",new_controls=0,new_resets=0,training_updates=0,
        not_a_success_rate_evaluation=True,request_budget=budget,wall_seconds=600,rows=[])
    api=None
    with (out/"calls.jsonl").open("x") as journal:
        def record(value):journal.write(json.dumps(value,allow_nan=False)+"\n");journal.flush()
        try:
            if not args.dry_run:
                api=JevClient(key_file=args.key_file,max_calls=budget,journal=record)
                api.deadline=time.perf_counter()+600
                policy=JevGroundedPolicy(SimpleNamespace(calls=0,identity={}),api,task_id=0,
                    task_plan=REPO/"configs/semantic_robot/jev_task0_plan.json")
                goals,receipt=policy.plan(0,policy.plan_identity["official_instruction"],None)
                write("plan.json",receipt)
                h=GroundedHarness(goals);h.held_inspection_enabled=h.reference_from_planner=True
                write("reference.json",policy.resolve_target_reference(h))
                result["planned_goals"]=[asdict(g) for g in goals]
                result["initial_reference"]=h.search_reference
            engine=JevDecisionPolicy(api) if api else None
            for index,(context,allowed,source) in enumerate(inputs):
                write(f"input_{index:03d}.json",dict(context=context,allowed=[asdict(a) for a in allowed],source=source))
                row=dict(index=index,stage=context["harness"]["stage"],goal=context["harness"]["goal"],
                         allowed_count=len(allowed),source_path=source["path"])
                if engine:
                    try:
                        action,receipt=engine.select(context,allowed)
                        row.update(status="selected_not_executed",selected=asdict(action),intent=receipt["result"]["intent"],
                                   jev_authority=authority(action,receipt,allowed))
                    except JevAbstained:
                        receipt=engine.last_call;row.update(status="abstained_no_action")
                    write(f"choice_{index:03d}.json",receipt)
                else:row["status"]="validated_not_called"
                result["rows"].append(row)
            result["completed"]=True
        except Exception as error:
            result.update(completed=False,error=str(error))
            raise
        finally:
            if api:
                result.update(api_requests=api.calls,validated_responses=api.validated_responses,
                    input_tokens=api.input_tokens,output_tokens=api.output_tokens)
            write("result.json",result)
            print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
