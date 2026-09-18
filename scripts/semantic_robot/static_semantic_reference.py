"""Two registered saved-state semantic routes; zero physics and zero image input."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.harness import Goal
from static_reference_scope import preflight,read


def main():
    p=argparse.ArgumentParser();p.add_argument("--state",action="append",required=True)
    p.add_argument("--expected-reference",action="append",required=True)
    p.add_argument("--uri",required=True);p.add_argument("--revision",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    if len(a.state)!=2 or len(a.expected_reference)!=2:raise ValueError("Exactly two registered states")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise ValueError("Fixed source required")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[];policy=GroundedPolicy(a.uri,a.revision,max_calls=2)
    try:
        for i,(value,expected) in enumerate(zip(a.state,a.expected_reference)):
            d=Path(value);saved=read(d/"harness.json")
            h=GroundedHarness([Goal(**saved["goal"])],held_inspection=True,reference_from_planner=True)
            h.held=saved["held_target_claims"];h.hold_verified=saved["holding_verified_by_observation_and_proprio"]
            call=policy.resolve_target_reference(h)
            if call is not None and (call["request"]["images"] or call["result"]["images"]):
                raise ValueError("Semantic routing may not receive images")
            row={"source":value,"call":call,"binding":h.reference_receipts.get(h.index),
                 "expected_reference_not_actor_input":expected,"reference_matches":h.search_reference==expected,
                 "new_controls":0,"not_task_success":True,"stop_reason":h.stop_reason}
            if h.search_reference.startswith("held_") and not h.stop_reason:
                obs=GroundedEvidence.parse(read(d/"observation.json")["result"]["text"])
                row["preflight"]=preflight(d,obs,planner_reference=h.search_reference)
            (out/f"case_{i:02d}.json").write_text(json.dumps(row,indent=2));rows.append(row)
            print(json.dumps({"case":i,"reference":h.search_reference,"matches":row["reference_matches"],
                              "stop_reason":h.stop_reason,"preflight_stop":row.get("preflight",{}).get("stop_reason")}),flush=True)
        if not all(row["reference_matches"] and not row["stop_reason"] for row in rows):
            raise ValueError("Semantic reference mismatch; no physical release")
        held=rows[0].get("preflight",{})
        if not held.get("inspection_needed") or held.get("stop_reason") or len(held.get("allowed",[]))<=1:
            raise ValueError("Saved-pose inspection preflight failed; no physical release")
        (out/"result.json").write_text(json.dumps({"status":"complete","model_calls":policy.calls,"controls":0,"rows":rows},indent=2))
    except BaseException as e:
        (out/"failure.json").write_text(json.dumps({"error":repr(e),"model_calls":policy.calls,"controls":0,"rows":rows},indent=2));raise


if __name__=="__main__":main()
