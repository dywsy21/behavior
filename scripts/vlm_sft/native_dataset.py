"""Release whole reviewed native GRASP trajectories, never partial successes.

Privileged evidence is consumed here only. Exported actor/text/images are the
unchanged H09X whitelist. Original quarantines and image files stay untouched.
"""
import argparse
import hashlib
from io import BytesIO
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from common import CAMERAS,TOKENS,sha,write_json
import native_actor_protocol as protocol
from native_teacher_artifacts import load_calibration
from native_teacher_outcomes import LocalOutcome
from native_teacher_policy import validate_train_group
from native_execution import authorization_profile,metadata as execution_metadata,completed as execution_completed

VERSION="h09y-reviewed-native-grasp-v1"
TRAIN={(1,192),(1,114)}
HELDOUT=[[1,1],[1,71]]


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def check_execution(record,request,execution,profile):
    token=record["token"]
    if any(authorization_profile(value)!=profile for value in (execution,record,request["preflight"])):
        raise ValueError("Mixed execution profile within a trajectory")
    if (token not in TOKENS or request["token"]!=token or execution["token"]!=token or
            not execution_completed(token,execution["feedback"],profile=profile)):
        raise ValueError("Incomplete or action-mismatched execution receipt")
    if profile is not None and execution["control_end"]-execution["control_start"]!=execution["feedback"]["control_ticks"]:
        raise ValueError("Completion receipt does not match the actual control ledger")


def checked_images(row):
    """NO legacy protocol fallback and NO image path trusted without bytes."""
    if row.get("protocol")!=protocol.VERSION or row.get("text")!=protocol.prompt(row["actor"]):
        raise ValueError("Required H09X actor protocol/prefix missing")
    if set(row.get("images",{}))!=set(CAMERAS):raise ValueError("Exactly three image paths required")
    result={}
    for view,path in row["images"].items():
        path=Path(path)
        if not path.is_absolute() or sha(path)!=row["actor"]["current_rgb_sha256"][view]:
            raise ValueError("Actual training image bytes changed")
        image=Image.open(path);n=720 if view=="head" else 480
        if image.format!="PNG" or image.size!=(n,n):raise ValueError("Wrong raw onboard image layout")
        result[view]=image.convert("RGB").resize((256,256),Image.Resampling.LANCZOS)
    return result


def verified_run(entry,counts):
    run=Path(entry["run"]).resolve();inventory=Path(entry["inventory"]);review_path=Path(entry["review"])
    protocol.check_sha(entry["review_sha256"])
    if sha(review_path)!=entry["review_sha256"]:raise ValueError("Independent review bytes changed")
    review=read(review_path);listing=read(inventory)
    if (review.get("candidate_single_trajectory_accepted") is not True or review.get("local_success") is not True or
            review.get("reviewer")!="Codex-parent" or review.get("inventory_sha256")!=sha(inventory)):
        raise ValueError("Bound independent whole-trajectory review required")
    expected={k[len(run.name)+1:]:v for k,v in listing["files"].items() if k.startswith(run.name+"/")}
    actual={str(p.relative_to(run)):p for p in run.rglob("*") if p.is_file()}
    if not expected or set(expected)!=set(actual):raise ValueError("Incomplete or extra run evidence")
    for name,item in expected.items():
        if sha(actual[name])!=item["sha256"] or actual[name].stat().st_size!=item["bytes"]:
            raise ValueError("Reviewed run bytes changed")
    manifest=read(run/"manifest.json");result=read(run/"result.json")
    execution_profile=authorization_profile(manifest["authorization"],collection=True)
    if authorization_profile(review)!=execution_profile:
        raise ValueError("Independent review must bind the same execution profile")
    if (manifest["code"]!=review["source"] or manifest.get("robot_geometry_guards") is not True or
            result.get("status")!="COLLECTED_QUARANTINED_NOT_SFT" or result.get("model_calls")!=0 or
            any("failure" in name.lower() for name in actual)):
        raise ValueError("Only complete successful native collection runs")
    group=validate_train_group({"source":manifest["source"]},counts,HELDOUT)
    if group not in TRAIN:raise ValueError("Unregistered training instance")
    initial=read(run/"PRIVATE_teacher_initial.json");spec=initial["spec"]
    if spec["verb"]!="GRASP":raise ValueError("This dataset is GRASP-only")
    arm=spec["hand"];prefix=manifest["prefix_controls"]
    normal=lines(run/"native_trace.jsonl");issued=lines(run/"issued_trace.jsonl")
    if normal!=issued:raise ValueError("Issued/completed controls diverged")
    for i,row in enumerate(normal,1):
        if (row["prefix_control"]!=min(i,prefix) or row["native_control"]!=max(0,i-prefix) or
                (row["phase"]=="prefix")!=(i<=prefix) or
                len(row["action23"])!=23 or not np.isfinite(row["action23"]).all()):
            raise ValueError("Actual native control ledger is not contiguous")
    prefix_bytes=BytesIO();np.save(prefix_bytes,np.asarray([r["action23"] for r in normal[:prefix]],np.float32),allow_pickle=False)
    if hashlib.sha256(prefix_bytes.getvalue()).hexdigest()!=manifest["prepared_binding"]["files_sha256"]["prefix.npy"]:
        raise ValueError("Actually replayed full prefix differs from bound source")
    hold=read(run/"final_hold.json");final=read(run/"PRIVATE_final_hold_outcome.json")
    if (hold.get("completed") is not True or hold["prefix_controls"]!=prefix or
            hold["native_controls"]!=result["native_controls"] or prefix+hold["native_controls"]!=len(normal)+1):
        raise ValueError("Whole-trajectory final hold missing")
    folders=[run/Path(p).name for p in result["samples"]]
    if folders!=[run/f"teacher_{i:02d}" for i in range(len(folders))] or not 1<=len(folders)<=12:
        raise ValueError("Whole macro sequence required")
    model=load_calibration(run);records=[];history=[];tokens={};last_end=12;rotation_degrees=[]
    for i,folder in enumerate(folders):
        record=read(folder/"QUARANTINED_record.json");request=read(folder/"request.json")
        execution=read(folder/"native_execution.json");token=record["token"]
        check_execution(record,request,execution,execution_profile)
        if (record["native_trace_sha256"]!=sha(folder/"native_execution.json") or
                record["actor"]!=request["actor"] or record["actor"]["history"]!=history[-5:] or
                record["actor"]["proprio"]!=read(folder/"before/proprio.json") or
                execution["control_start"]!=last_end or execution["control_end"]<=last_end):
            raise ValueError("Wrong same-state/action/history or incomplete execution")
        before=read(folder/"before/capture.json");after=read(folder/"after_settle/capture.json")
        expected_clock={"prefix_control":prefix,"native_control":execution["control_start"]}
        proprio,hashes,binding=protocol.from_capture(folder/"before",model,capture_sha256=request["capture_sha256"],
            expected_clock=expected_clock,calibration_sha256=model.sha)
        if record["actor"]["current_rgb_sha256"]!=hashes or before["clock"]!=expected_clock or after["clock"]!={
                "prefix_control":prefix,"native_control":execution["control_end"]+12}:
            raise ValueError("Capture/action clock drift")
        # A rotation label counts as coverage only when the actual robot FK
        # records a nontrivial same-axis movement in the requested direction.
        if any(axis in token for axis in ("_YAW_","_ROLL_","_PITCH_")):
            moved=token.split("_")[0].lower();axis={"ROLL":0,"PITCH":1,"YAW":2}[token.split("_")[1]]
            before_r=np.array(proprio["eef_rotation_base"][moved])
            after_proprio,_,_=protocol.from_capture(folder/"after_settle",model,
                capture_sha256=sha(folder/"after_settle/capture.json"),expected_clock=after["clock"],calibration_sha256=model.sha)
            after_r=np.array(after_proprio["eef_rotation_base"][moved])
            rotation_vector=Rotation.from_matrix(after_r@before_r.T).as_rotvec()
            signed=np.rad2deg(rotation_vector[axis])*(1 if token.endswith("PLUS") else -1)
            if signed<1.:raise ValueError("Rotation token lacks measured same-axis physical rotation")
            rotation_degrees.append(float(signed))
        for control in range(execution["control_start"]+1,execution["control_end"]+1):
            if normal[prefix+control-1]["phase"]!="candidate":raise ValueError("Wrong executed action phase")
            tokens[prefix+control]=token
        last_end=execution["control_end"]+12
        actor=protocol.actor_input(record["actor"]["task"],record["actor"]["active_instruction"],proprio,hashes,history[-5:])
        row={**protocol.training_row(actor,token),"id":sha(inventory)[:16]+f"_{i:02d}",
             "images":{v:str(folder/"before"/(v+".png")) for v in CAMERAS},
             "provenance":{"group":list(group),"episode":manifest["source"]["episode"],"prefix":prefix,
                "macro":i,"hand":arm,"run_inventory_sha256":sha(inventory),"review_sha256":sha(review_path),
                **execution_metadata(execution_profile),**binding}}
        checked_images(row);records.append(row);history.append(token)
    if last_end+1!=result["native_controls"]:raise ValueError("Unaccounted post-macro controls")
    oracle=LocalOutcome(spec);oracle.update(initial["frame"])
    for row in lines(run/"PRIVATE_teacher_trace.jsonl"):
        verdict=oracle.update(row["frame"],tokens.get(row["frame"]["tick"]))
        if verdict!=row["verdict"] or verdict["outcome"] in ("UNKNOWN","FAILED"):
            raise ValueError("Actual full physical outcome chain invalid")
    if oracle.update(final["frame"])!=final["verdict"] or final["verdict"]["outcome"]!="SUCCEEDED":
        raise ValueError("No preserved physical GRASP success at final hold")
    return records,{"group":list(group),"prefix":prefix,"macros":len(records),"hand":arm,
        **execution_metadata(execution_profile),
        "inventory_sha256":sha(inventory),"review_sha256":sha(review_path),"initial_proprio":records[0]["actor"]["proprio"],
        "measured_rotation_degrees":rotation_degrees}


def coverage(rows,runs):
    counts=Counter(tuple(r["group"]) for r in runs);labels=Counter(r["target"] for r in rows)
    close=sum(t.endswith("_CLOSE") for r in rows for t in [r["target"]])
    lift=preclose=0
    for run in runs:
        closed=False
        for r in sorted((r for r in rows if r["provenance"]["run_inventory_sha256"]==run["inventory_sha256"]),key=lambda r:r["provenance"]["macro"]):
            lift+=int(closed and r["target"].endswith("_UP"))
            preclose+=int(not closed and r["target"]!="HOLD" and not r["target"].endswith(("_CLOSE","_OPEN")))
            closed|=r["target"].endswith("_CLOSE")
    rotation=any(any(float(value)>=1. for value in r.get("measured_rotation_degrees",[])) for r in runs)
    duplicates=[]
    for i,a in enumerate(runs):
        for j,b in enumerate(runs[:i]):
            if a["group"]!=b["group"]:continue
            arm=a["hand"];ap,bp=a["initial_proprio"],b["initial_proprio"]
            distance=np.linalg.norm(np.array(ap["eef_base_m"][arm])-bp["eef_base_m"][arm])
            angle=np.linalg.norm(Rotation.from_matrix(np.array(ap["eef_rotation_base"][arm]).T@np.array(bp["eef_rotation_base"][arm])).as_rotvec())
            if distance<.003 and angle<np.deg2rad(.75):duplicates.append([j,i])
    passed=(len(runs)>=4 and set(counts)==TRAIN and min(counts.values())>=2 and len(rows)>=32 and
            close>=4 and lift>=8 and preclose>=8 and len(set(labels)-{"HOLD"})>=4 and rotation and not duplicates)
    return {"passed":bool(passed),"whole_trajectories":len(runs),"group_counts":{str(k):v for k,v in counts.items()},
        "macros":len(rows),"labels":dict(labels),"close":close,"lift":lift,"preclose":preclose,
        "actual_rotation":rotation,"near_duplicate_run_pairs":duplicates,"failed_positive_BC":0}


def load_dataset(folder,*,require_gate=True):
    folder=Path(folder);manifest=read(folder/"dataset.json")
    if manifest.get("schema")!=VERSION or manifest.get("protocol")!=protocol.VERSION or sha(folder/"train.jsonl")!=manifest["rows_sha256"]:
        raise ValueError("Exact new dataset/protocol identity required")
    rows=lines(folder/"train.jsonl")
    if not rows or len({r["id"] for r in rows})!=len(rows):raise ValueError("Unique actual training rows required")
    run_ids=[r["inventory_sha256"] for r in manifest["runs"]]
    if (len(set(run_ids))!=len(run_ids) or any(r["provenance"]["run_inventory_sha256"] not in run_ids for r in rows) or
            any(sum(row["provenance"]["run_inventory_sha256"]==run["inventory_sha256"] for row in rows)!=run["macros"] for run in manifest["runs"])):
        raise ValueError("Exact unique whole-run row membership required")
    if any(tuple(r["provenance"]["group"]) not in TRAIN for r in rows):raise ValueError("Heldout leaked into BC")
    profiles={run["inventory_sha256"]:authorization_profile(run) for run in manifest["runs"]}
    if any(authorization_profile(row["provenance"])!=profiles[row["provenance"]["run_inventory_sha256"]] for row in rows):
        raise ValueError("Row execution provenance differs from its reviewed run")
    if coverage(rows,manifest["runs"])!=manifest["coverage"] or (require_gate and not manifest["coverage"]["passed"]):
        raise ValueError("Registered whole-trajectory coverage is incomplete")
    for row in rows:checked_images(row)
    return rows,manifest


def main():
    p=argparse.ArgumentParser();p.add_argument("--sources",type=Path,required=True);p.add_argument("--counts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    from native_teacher_group_prepare import COUNTS_SHA
    if sha(a.counts)!=COUNTS_SHA:raise ValueError("Immutable TRAIN exclusions required")
    entries=read(a.sources);rows=[];runs=[]
    if not isinstance(entries,list) or not 1<=len(entries)<=6:raise ValueError("Bounded six-run admission")
    for entry in entries:
        batch,receipt=verified_run(entry,read(a.counts));rows.extend(batch);runs.append(receipt)
    if len({r["inventory_sha256"] for r in runs})!=len(runs):raise ValueError("Duplicate whole trajectory")
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"train.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in rows))
    gate=coverage(rows,runs)
    write_json(a.output/"dataset.json",{"schema":VERSION,"protocol":protocol.VERSION,"runs":runs,"coverage":gate,
        "rows_sha256":sha(a.output/"train.jsonl"),"source_list_sha256":sha(a.sources),"counts_sha256":COUNTS_SHA,
        "status":"READY_FOR_SEPARATE_TRAINING_AUTHORIZATION" if gate["passed"] else "INSUFFICIENT_DO_NOT_TRAIN",
        "original_quarantine_unchanged":True,"full_task_success_claim":False})
    print(json.dumps(gate,indent=2))


if __name__=="__main__":main()
