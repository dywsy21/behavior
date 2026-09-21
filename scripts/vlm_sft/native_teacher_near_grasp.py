"""Bound a paid, later TRAIN prefix to the SAME complete expert skill source.

CPU preparation only. Original source/seed files are immutable. This is not a
phase-start sample, physical permission, successful grasp, or native BC label.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from common import sha,write_json
from native_teacher_policy import validate_train_group

SCHEMA="h09v-near-grasp-native-teacher-v1"


def original_source(folder,expected_manifest,counts,excluded):
    folder=Path(folder).resolve();mp=folder.parent/"preparation.json"
    if sha(mp)!=expected_manifest:raise ValueError("Original full-reference preparation changed")
    manifest=json.loads(mp.read_text())
    if manifest.get("schema")!="h09u-full-expert-reference-v1" or manifest.get("status")!="PREPARED_NO_RESET_NO_OUTCOME_NO_SEED":
        raise ValueError("Full expert reference preparation required")
    matches=[r for r in manifest["sources"] if folder.name==f"task_{r['task']}"]
    if len(matches)!=1:raise ValueError("Unique full-reference source required")
    row=matches[0]
    required={"prefix.npy","segment.npy","source_states.npy","teacher_reference.json","private_spec.json",
              "segment_labels.json","label_selection_audit.json","window.json"}
    if set(row["files_sha256"])!=required or any(sha(folder/k)!=v for k,v in row["files_sha256"].items()):
        raise ValueError("Original full-source bytes changed")
    ref=json.loads((folder/"teacher_reference.json").read_text());validate_train_group(ref,counts,excluded)
    if (any(ref["source"][k]!=row[k] for k in ("task","episode","instance")) or
            ref["source_label"]["segment_start"]!=row["start"] or ref["source_label"]["segment_end"]!=row["end"]):
        raise ValueError("Source identity/skill interval changed")
    pre,seg,states=[np.load(folder/(n+".npy"),allow_pickle=False) for n in ("prefix","segment","source_states")]
    for a,shape in ((pre,(row["start"],23)),(seg,(row["end"]-row["start"],23)),(states,(len(seg)+1,61))):
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError("Full source shape/clock invalid")
    return row,ref,pre,seg,states


def derive(folder,expected_manifest,counts,excluded,start):
    row,ref,pre,seg,states=original_source(folder,expected_manifest,counts,excluded)
    if type(start) is not int or not row["start"]<start<row["end"]-16:
        raise ValueError("Explicit interior skill start with full 17-frame reference required")
    skill=json.loads(ref["private_original_semantic_json"])
    if len(skill)!=1 or skill[0]["verb"]!="GRASP" or skill[0].get("unbound_relation"):
        raise ValueError("This bounded later-prefix pilot supports only original GRASP")
    labels=json.loads((Path(folder)/"segment_labels.json").read_text())
    selected=[]
    for frame in range(start,start+17):
        matched=[r for r in labels if r["frame_index"]==frame and r["source_kind"]=="original_demo" and r["memlite_branch"]=="low"]
        if (len(matched)!=1 or matched[0]["active_skills_semantic_json"]!=ref["private_original_semantic_json"] or
                matched[0]["low_action_supervision_mask"] is not True or matched[0]["segment_start"]!=row["start"] or
                matched[0]["segment_end"]!=row["end"] or matched[0]["action_horizon_end"]<start+17):
            raise ValueError("Conflicting/mixed/unreleased later source frame")
        selected.append(matched[0])
    offset=start-row["start"];prefix=np.concatenate([pre,seg[:offset]])
    if not np.all((prefix[-1,[14,22]]>=.999)&(prefix[-1,[14,22]]<=1)):
        raise ValueError("Last paid original command must keep both grippers OPEN")
    updated=copy.deepcopy(ref);updated.update(schema=SCHEMA,prefix_controls=start,source_label=selected[0],
        source_states=states[offset:offset+17].tolist(),source_actions=seg[offset:offset+16].tolist(),
        proposals={"training_eligible":False,"reason":"CURRENT_STATE_OFFLINE_TEACHER_NOT_FUTURE_ACTION_BC"},
        source_review_media={},is_original_skill_phase_start=False,original_skill_start=row["start"])
    updated["pilot"]={**updated["pilot"],"frame":start}
    return updated,prefix,row


def prepare(folder,output,start,expected_manifest,counts_path,excluded,*,capacity_profile=None):
    from native_teacher_capacity import CARRY_PROFILE,CARRY_STARTS
    if capacity_profile not in (None,CARRY_PROFILE):raise ValueError("Unknown explicit preparation capacity")
    counts=json.loads(Path(counts_path).read_text());ref,prefix,row=derive(folder,expected_manifest,counts,excluded,start)
    if capacity_profile==CARRY_PROFILE and (row['task'],row['episode'],row['instance'],start) not in CARRY_STARTS:
        raise ValueError("Unregistered carry-duration start")
    controls,macros=(640,14) if capacity_profile==CARRY_PROFILE else (420,12)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);task=output/f"task_{row['task']}";task.mkdir()
    np.save(task/"prefix.npy",prefix,allow_pickle=False);write_json(task/"teacher_reference.json",ref)
    window=json.loads((Path(folder)/"window.json").read_text())
    window.update(window_id=f"h09v-train-t{row['task']}-i{row['instance']}-paid{start}",max_steps=start+controls+1,
                  prefix_actions_path=str(task/"prefix.npy"),prefix_actions_sha256=sha(task/"prefix.npy"))
    write_json(task/"window.json",window)
    manifest={"schema":SCHEMA,"status":"PREPARED_NOT_COLLECTED_NOT_TRAINING_DATA","source":[row[k] for k in ("task","episode","instance")],
              "original_reference_path":str(Path(folder).resolve()),"original_reference_preparation_sha256":expected_manifest,
              "train_counts_sha256":sha(counts_path),"held_out_instance_groups":excluded,"paid_prefix_controls":start,
              "files_sha256":{n:sha(task/n) for n in ("prefix.npy","teacher_reference.json","window.json")},
              "native_controls_max":controls,"max_teacher_primitives":macros,"new_controls":0,"new_resets":0,"training_eligible":False}
    if capacity_profile is not None:manifest['capacity_profile']=capacity_profile
    write_json(output/"preparation.json",manifest);return manifest


def verify_prepared(prepared,release):
    from native_teacher_capacity import CARRY_PROFILE,CARRY_STARTS
    prepared=Path(prepared).resolve();mp=prepared.parent/"preparation.json"
    if sha(mp)!=release.get("preparation_manifest_sha256"):raise ValueError("Unregistered later-prefix preparation")
    m=json.loads(mp.read_text());counts_path=Path(release["train_counts_path"])
    extended=release.get('capacity_profile')==CARRY_PROFILE
    controls,macros=(640,14) if extended else (420,12)
    if (m.get('native_controls_max')!=controls or m.get('max_teacher_primitives')!=macros or
            m.get('capacity_profile')!=(CARRY_PROFILE if extended else None) or
            (extended and (*m['source'],m['paid_prefix_controls']) not in CARRY_STARTS)):
        raise ValueError("Preparation must bind the exact old/new control and source profile")
    if (m.get("schema")!=SCHEMA or m.get("status")!="PREPARED_NOT_COLLECTED_NOT_TRAINING_DATA" or
            m["source"]!=release["source"] or m["paid_prefix_controls"]!=release["paid_prefix_controls"] or
            m["train_counts_sha256"]!=release["train_counts_sha256"] or sha(counts_path)!=m["train_counts_sha256"] or
            m["held_out_instance_groups"]!=release["held_out_instance_groups"] or
            m["original_reference_preparation_sha256"]!=release["seed_reference_preparation_sha256"]):
        raise ValueError("Later prefix/source/TRAIN/seed identity mismatch")
    expected_ref,expected_prefix,row=derive(m["original_reference_path"],m["original_reference_preparation_sha256"],
        json.loads(counts_path.read_text()),m["held_out_instance_groups"],m["paid_prefix_controls"])
    if m['source']!=[row[k] for k in ('task','episode','instance')]:
        raise ValueError("Authorization source does not match actual origin")
    expected_files={"prefix.npy","teacher_reference.json","window.json"}
    if prepared.name!=f"task_{row['task']}" or set(m["files_sha256"])!=expected_files or any(sha(prepared/n)!=h for n,h in m["files_sha256"].items()):
        raise ValueError("Prepared later prefix bytes changed")
    ref=json.loads((prepared/"teacher_reference.json").read_text());prefix=np.load(prepared/"prefix.npy",allow_pickle=False)
    if ref!=expected_ref or prefix.dtype!=np.float32 or not np.array_equal(prefix,expected_prefix):
        raise ValueError("Later prefix is not exactly original prefix plus same-skill controls")
    window=json.loads((prepared/"window.json").read_text())
    original=json.loads((Path(m["original_reference_path"])/"window.json").read_text())
    expected=dict(original);expected.update(window_id=f"h09v-train-t{row['task']}-i{row['instance']}-paid{len(prefix)}",
        max_steps=len(prefix)+controls+1,prefix_actions_path=str(prepared/"prefix.npy"),prefix_actions_sha256=m["files_sha256"]["prefix.npy"])
    if window!=expected:raise ValueError("Wrong actual reset/task/window/timing")
    return ref,prefix,{"preparation_manifest_sha256":sha(mp),"files_sha256":m["files_sha256"],
        "task":row["task"],"task_name":window["task_name"],"instance":row["instance"],"official_mode":"train","seed":0}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--reference",required=True);p.add_argument("--output",required=True)
    p.add_argument("--start-control",type=int,required=True);p.add_argument("--reference-manifest-sha",required=True)
    p.add_argument("--counts",required=True);p.add_argument("--held-out-groups",default="[]")
    p.add_argument("--capacity-profile")
    a=p.parse_args();print(json.dumps(prepare(a.reference,a.output,a.start_control,a.reference_manifest_sha,a.counts,json.loads(a.held_out_groups),capacity_profile=a.capacity_profile)))
