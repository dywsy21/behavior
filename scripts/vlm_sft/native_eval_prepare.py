"""Prepare ONLY the two preregistered heldout starts; no teacher or pose seed.

Original actions define the paid prefix. They never become actor supervision.
The private target identity is used only by the posthoc measurement reader.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import sha,skill_text,write_json
from native_grasp_sources import audit_source,reserve_groups
from native_teacher_group_prepare import COUNTS_SHA,AUDIT_SHA,ROBOT_SHA
from native_reference_profile import ROOT
from native_storage import activate as activate_storage

SCHEMA="h09y-heldout-grasp-evaluation-v1"
STARTS={1:{"episode":200,"prefix":832,"stratum":"expert_left_hand_start_pose_OOD"},
        71:{"episode":247,"prefix":1038,"stratum":"expert_right_hand_start_pose"}}
INSTRUCTION="verb=GRASP; target=trash can; source=floors"


def select(counts,audit,instance):
    if type(instance) is not int or instance not in STARTS:raise ValueError("Only the two fixed heldout instances")
    reserved=reserve_groups(counts)
    if audit.get("counts_sha256")!=COUNTS_SHA or len(audit.get("groups",[]))!=len(reserved):
        raise ValueError("Immutable source partition required")
    for split,source in reserved:
        found=[r for r in audit["groups"] if r["source"]==source and r["split"]==split]
        if len(found)!=1:raise ValueError("Source assignment changed")
    found=[r for r in audit["groups"] if r["split"]=="heldout" and r["source"]["instance"]==instance]
    if len(found)!=1:raise ValueError("Heldout source not unique")
    row=found[0];item=STARTS[instance];selected=row["selected_earliest_eligible"]
    if (row["source"]["task"]!=1 or row["source"]["episode"]!=item["episode"] or
            selected["near_prefix_controls"]!=item["prefix"] or skill_text([selected["skill"]])!=INSTRUCTION):
        raise ValueError("Registered heldout episode/prefix/legal instruction changed")
    return row


def prepare(counts_path,audit_path,robot_template,instance,storage_spec=None,*,execution_profile=None):
    from native_execution import CARRY_PROFILE,episode_limits,metadata
    if execution_profile not in (None,CARRY_PROFILE):raise ValueError("Unknown explicit heldout preparation profile")
    _,control_limit=episode_limits(execution_profile)
    import pyarrow.parquet as pq
    from prepare import LABELS,RELEASE
    from native_teacher_reference_prepare import LABEL_COLUMNS
    if sha(counts_path)!=COUNTS_SHA or sha(audit_path)!=AUDIT_SHA:raise ValueError("Source/count bytes changed")
    counts=json.loads(Path(counts_path).read_text());audit=json.loads(Path(audit_path).read_text())
    registered=select(counts,audit,instance);source=registered["source"]
    folder=ROOT/f"eval_prepare_i{instance}_v1"
    from native_train import check_storage
    storage=activate_storage({} if storage_spec is None else {"storage":storage_spec},folder);check_storage(storage)
    if (sha(LABELS)!=counts["source_identity"]["labels_sha256"] or
            sha(RELEASE/"quarantine_ranges.parquet")!=counts["source_identity"]["quarantine_sha256"]):
        raise ValueError("Annotation/quarantine release changed")
    table=pq.read_table(source["parquet"],filters=[("episode_index","=",source["episode"])],
        columns=["frame_index","action","observation.state"]).sort_by("frame_index")
    labels=pq.read_table(LABELS,filters=[("episode_index","=",source["episode"])],columns=LABEL_COLUMNS).to_pylist()
    fresh=audit_source(source,table,labels,pq.read_table(RELEASE/"quarantine_ranges.parquet").to_pylist())
    if fresh!={k:v for k,v in registered.items() if k!="split"}:raise ValueError("Exact fixed source arrays/labels changed")
    template=json.loads(Path(robot_template).read_text());robot=Path(template["robot_config_path"])
    if template["robot_config_sha256"]!=ROBOT_SHA or sha(robot)!=ROBOT_SHA:raise ValueError("Robot identity changed")
    item=STARTS[instance];prefix=np.asarray(table["action"].to_pylist(),np.float32)[:item["prefix"]]
    check_storage(storage);folder.mkdir(parents=True,exist_ok=False)
    np.save(folder/"prefix.npy",prefix,allow_pickle=False)
    if sha(folder/"prefix.npy")!=fresh["selected_earliest_eligible"]["near_prefix_sha256"]:
        raise ValueError("Exact paid prefix SHA mismatch")
    skill=fresh["selected_earliest_eligible"]["skill"]
    # The reader records both arms. This placeholder is NEVER an actor input
    # or a specified-hand evaluation constraint.
    spec={"verb":"GRASP","hand":"right","support_hand":None,"target":skill["target"],
        "destination":None,"payloads":[],"goal_frame":"target"}
    write_json(folder/"PRIVATE_scoring_spec.json",spec)
    window={"kind":"native_oracle_low_window","immutable":True,"window_id":f"h09y-heldout-t1-i{instance}",
        "task_name":"picking_up_trash","official_mode":"train","instance_id":instance,"seed":0,
        "max_steps":len(prefix)+control_limit+1,"robot_config_path":str(robot),"robot_config_sha256":ROBOT_SHA,
        "max_chunks":1,"execute_steps":1,"prefix_actions_path":str(folder/"prefix.npy"),
        "prefix_actions_sha256":sha(folder/"prefix.npy"),"semantic_subgoal":{"parent_goal":"picking_up_trash",
            "active_skills_semantic_json":"[]","active_skills_text":INSTRUCTION}}
    write_json(folder/"window.json",window)
    manifest={"schema":SCHEMA,"purpose":"HELDOUT_EVALUATION_NEVER_BC","source":[1,item["episode"],instance],
        "prefix_controls":len(prefix),"stratum":item["stratum"],"active_instruction":INSTRUCTION,
        "specified_hand":None,"source_audit_sha256":AUDIT_SHA,"counts_sha256":COUNTS_SHA,
        "files_sha256":{n:sha(folder/n) for n in ("prefix.npy","window.json","PRIVATE_scoring_spec.json")},
        "new_resets":0,"training_eligible":False,"teacher_pose_seed":None}
    if execution_profile is not None:manifest.update(metadata(execution_profile))
    write_json(folder/"preparation.json",manifest);return manifest


def verify(prepared,expected_sha,*,execution_profile=None):
    from native_execution import CARRY_PROFILE,authorization_profile,episode_limits
    _,control_limit=episode_limits(execution_profile)
    folder=Path(prepared).resolve();mp=folder/"preparation.json"
    if sha(mp)!=expected_sha:raise ValueError("Exact heldout preparation identity required")
    m=json.loads(mp.read_text());instance=m["source"][2]
    if authorization_profile(m)!=(CARRY_PROFILE if execution_profile==CARRY_PROFILE else None):
        raise ValueError("Heldout preparation episode profile mismatch")
    if type(instance) is not int or instance not in STARTS:raise ValueError("Not an evaluation instance")
    item=STARTS[instance]
    if (m.get("schema")!=SCHEMA or m.get("purpose")!="HELDOUT_EVALUATION_NEVER_BC" or
            m["source"]!=[1,item["episode"],instance] or m["prefix_controls"]!=item["prefix"] or
            m.get("stratum")!=item["stratum"] or m.get("active_instruction")!=INSTRUCTION or
            m.get("specified_hand") is not None or m.get("teacher_pose_seed") is not None or m.get("training_eligible") is not False or
            m.get("source_audit_sha256")!=AUDIT_SHA or m.get("counts_sha256")!=COUNTS_SHA or
            set(m["files_sha256"])!={"prefix.npy","window.json","PRIVATE_scoring_spec.json"} or
            any(sha(folder/n)!=h for n,h in m["files_sha256"].items())):
        raise ValueError("Heldout purpose/identity/instruction/prefix binding changed")
    prefix=np.load(folder/"prefix.npy",allow_pickle=False);window=json.loads((folder/"window.json").read_text())
    if (prefix.dtype!=np.float32 or prefix.shape!=(item["prefix"],23) or not np.isfinite(prefix).all() or
            not np.all((prefix[-1,[14,22]]>=.999)&(prefix[-1,[14,22]]<=1)) or
            window.get("task_name")!="picking_up_trash" or window.get("official_mode")!="train" or
            window.get("instance_id")!=instance or type(window.get("seed")) is not int or window["seed"]!=0 or
            window.get("max_steps")!=len(prefix)+control_limit+1 or window.get("max_chunks")!=1 or window.get("execute_steps")!=1 or
            window.get("robot_config_sha256")!=ROBOT_SHA or sha(window["robot_config_path"])!=ROBOT_SHA or
            Path(window["prefix_actions_path"]).resolve()!=folder/"prefix.npy" or
            window.get("prefix_actions_sha256")!=m["files_sha256"]["prefix.npy"] or
            window.get("semantic_subgoal")!={"parent_goal":"picking_up_trash","active_skills_semantic_json":"[]","active_skills_text":INSTRUCTION}):
        raise ValueError("Actual heldout factory reset/prefix changed")
    return m,prefix


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--counts",required=True);p.add_argument("--source-audit",required=True)
    p.add_argument("--robot-template",required=True);p.add_argument("--instance",type=int,choices=tuple(STARTS),required=True)
    p.add_argument("--storage-profile",type=Path)
    p.add_argument("--execution-profile")
    a=p.parse_args();print(json.dumps(prepare(a.counts,a.source_audit,a.robot_template,a.instance,
        None if a.storage_profile is None else json.loads(a.storage_profile.read_text()),execution_profile=a.execution_profile),indent=2))
