"""Extract two hash-pinned COMPLETE TRAIN skill segments; zero simulator imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from common import sha, write_json
from native_teacher_policy import validate_train_group

SCHEMA="h09u-full-expert-reference-v1"
LABEL_COLUMNS=["frame_index","active_skills_semantic_json","low_action_supervision_mask",
               "action_horizon_end","segment_start","segment_end","memlite_branch","source_kind"]


def validate_segment(states, actions, frames, labels, source, start, end, semantic, quarantine):
    s,a=np.asarray(states,dtype=float),np.asarray(actions,dtype=float)
    if (s.shape!=(source["frames"],61) or a.shape!=(source["frames"],23) or
            not np.isfinite(s).all() or not np.isfinite(a).all() or
            not np.array_equal(frames,np.arange(source["frames"])) or not 0<=start<end<len(s)):
        raise ValueError("Exact finite full episode/state/action alignment required")
    h=hashlib.sha256(s.tobytes()+a.tobytes())
    h.update(json.dumps(labels,sort_keys=True,separators=(",",":")).encode())
    if h.hexdigest()!=source["extracted_arrays_and_labels_sha256"]:
        raise ValueError("Original full-episode arrays/labels SHA changed")
    by_frame={row["frame_index"]:row for row in labels}
    if len(by_frame)!=len(labels):raise ValueError("Duplicate annotation frames")
    for f in range(start,end):
        r=by_frame[f]
        if (r["active_skills_semantic_json"]!=semantic or r["source_kind"]!="original_demo" or
                r["memlite_branch"]!="low" or r["low_action_supervision_mask"] is not True or
                r["segment_start"]!=start or r["segment_end"]!=end or r["action_horizon_end"]<=f):
            raise ValueError("Full reference crosses a mixed/unreleased/changed skill boundary")
    if any(r["episode_index"]==source["episode"] and r["frame_start"]<end and r["frame_end"]>=start for r in quarantine):
        raise ValueError("Reference touches quarantined source")
    return a[:start].astype(np.float32),a[start:end].astype(np.float32),s[start:end+1].astype(np.float32)


def prepare(config,counts,h09s,output):
    import pyarrow.parquet as pq
    from prepare import LABELS,RELEASE
    h09s,output=Path(h09s),Path(output)
    if output.exists():raise FileExistsError(output)
    if sha(h09s/"preparation.json")!=config["h09s_preparation_sha256"]:
        raise ValueError("Original preparation identity changed")
    if sha(LABELS)!=counts["source_identity"]["labels_sha256"] or sha(RELEASE/"quarantine_ranges.parquet")!=counts["source_identity"]["quarantine_sha256"]:
        raise ValueError("Original annotation/quarantine release changed")
    old=json.loads((h09s/"preparation.json").read_text())
    quarantine=pq.read_table(RELEASE/"quarantine_ranges.parquet").to_pylist()
    prepared=[];output.mkdir(parents=True)
    for pilot in config["sources"]:
        prior=h09s/f"task_{pilot['task']}"
        oldrow=next(r for r in old["sources"] if r["task"]==pilot["task"])
        if sha(prior/"teacher_reference.json")!=oldrow["reference_sha256"] or sha(prior/"window.json")!=oldrow["window_sha256"]:
            raise ValueError("Frozen H09S task identity changed")
        ref=json.loads((prior/"teacher_reference.json").read_text());source=ref["source"]
        validate_train_group(ref,counts,config["held_out_instance_groups"])
        if any(source[k]!=pilot[k] for k in ("task","episode","instance")) or ref["source_label"]["segment_start"]!=pilot["start"] or ref["source_label"]["segment_end"]!=pilot["end"]:
            raise ValueError("Unregistered source skill span")
        data=pq.read_table(source["parquet"],filters=[("episode_index","=",pilot["episode"])],
                           columns=["frame_index","observation.state","action"]).sort_by("frame_index")
        labels=pq.read_table(LABELS,filters=[("episode_index","=",pilot["episode"])],columns=LABEL_COLUMNS).to_pylist()
        prefix,segment,states=validate_segment(data["observation.state"].to_pylist(),data["action"].to_pylist(),
                data["frame_index"].to_numpy(),labels,source,pilot["start"],pilot["end"],ref["private_original_semantic_json"],quarantine)
        folder=output/f"task_{pilot['task']}";folder.mkdir()
        for name,values in (("prefix",prefix),("segment",segment),("source_states",states)):
            np.save(folder/(name+".npy"),values,allow_pickle=False)
        skill=json.loads(ref["private_original_semantic_json"])[0]
        if skill["verb"]!=pilot["verb"] or skill["arm"] not in ("UNSPECIFIED",pilot["hand"].upper()):
            raise ValueError("Source skill hand/verb changed")
        spec={"verb":skill["verb"],"hand":pilot["hand"],"support_hand":pilot["support_hand"],
              "target":skill["target"],"destination":skill["destination"],"payloads":pilot["payloads"],
              "goal_frame":"toggle_link" if skill["verb"]=="PRESS" else "target"}
        write_json(folder/"teacher_reference.json",ref)
        write_json(folder/"private_spec.json",spec)
        write_json(folder/"segment_labels.json",[r for r in labels if pilot["start"]<=r["frame_index"]<pilot["end"]])
        window=json.loads((prior/"window.json").read_text())
        window.update(window_id=f"h09u-full-t{pilot['task']}-e{pilot['episode']}",max_steps=pilot["end"]+14,
                      prefix_actions_path=str((folder/"prefix.npy").resolve()),prefix_actions_sha256=sha(folder/"prefix.npy"))
        write_json(folder/"window.json",window)
        prepared.append({**pilot,"files_sha256":{p.name:sha(p) for p in folder.iterdir() if p.is_file()},
                         "prefix_controls":len(prefix),"segment_controls":len(segment),
                         "controls_including_tail_and_final_hold":pilot["end"]+13,
                         "source_arrays_and_labels_sha256":source["extracted_arrays_and_labels_sha256"]})
    result={"schema":SCHEMA,"status":"PREPARED_NO_RESET_NO_OUTCOME_NO_SEED", "sources":prepared,
            "source_config":config,"h09r_counts_sha256":config["h09r_counts_sha256"],
            "bytes":sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),"new_controls":0}
    if result["bytes"]>30*1024**2:raise RuntimeError("CPU preparation disk budget exceeded")
    write_json(output/"preparation.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--counts",required=True)
    p.add_argument("--h09s",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    if sha(a.counts)!=cfg["h09r_counts_sha256"]:raise ValueError("Count identity changed")
    print(json.dumps(prepare(cfg,json.loads(Path(a.counts).read_text()),a.h09s,a.output)))
