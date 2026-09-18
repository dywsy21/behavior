"""Derive a small immutable vocabulary-consistent dataset without new frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from common import TOKENS,VERSION,prompt,sha,write_json


def mapped_history(history):
    """Keep only the contiguous suffix after the last unrepresentable action."""
    last=max((i for i,t in enumerate(history) if t not in TOKENS),default=-1)
    return history[last+1:][-5:]


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();repo=Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip():raise RuntimeError("Pinned clean source required")
    if shutil.disk_usage(args.source).free<80*1024**3:raise RuntimeError("Disk reserve")
    args.output.mkdir(exist_ok=False)
    manifest={"version":VERSION,"code_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
        "source":str(args.source),"source_image_manifest_sha256":sha(args.source/"image_manifest.json"),
        "reason":"TORSO expert co-moving EEFs differs from current servo fixed-EEF compensation",
        "allowed_tokens":TOKENS,"counts":{},"class_counts":{},"task_counts":{},"removed":{},"history_changed":{}}
    for split in ("train","validation","test"):
        old=[json.loads(line) for line in (args.source/(split+"_images.jsonl")).read_text().splitlines()]
        rows=[];changed=0
        for row in old:
            if row["target"] not in TOKENS:continue
            history=mapped_history(row["history"]);changed+=history!=row["history"]
            row["history"]=history
            row["text"]=prompt(row["task"],row["active_instruction"],row["proprio"],history)
            row["text_sha256"]=hashlib.sha256(row["text"].encode()).hexdigest()
            for view,value in row["images"].items():
                image=(args.source/value).resolve()
                if sha(image)!=row["image_receipts"][view]["png_sha256"]:raise RuntimeError("Image changed")
                row["images"][view]=str(image)
            rows.append(row)
        for suffix in (".jsonl","_images.jsonl"):
            with (args.output/(split+suffix)).open("x") as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        manifest["counts"][split]=len(rows);manifest["class_counts"][split]=dict(Counter(r["target"] for r in rows))
        manifest["task_counts"][split]=dict(Counter(r["task_id"] for r in rows));manifest["removed"][split]=len(old)-len(rows)
        manifest["history_changed"][split]=changed
    for name in ("groups.json","frame_audit.json"):
        shutil.copyfile(args.source/name,args.output/name)
        manifest[name+"_sha256"]=sha(args.output/name)
    write_json(args.output/"index_manifest.json",manifest)
    write_json(args.output/"image_manifest.json",{"status":"strict_contract_filter_same_reviewed_images","version":VERSION,
        "index_manifest_sha256":sha(args.output/"index_manifest.json"),"source_image_manifest_sha256":manifest["source_image_manifest_sha256"],
        "splits":{s:sha(args.output/(s+"_images.jsonl")) for s in manifest["counts"]},
        "all_frames_within_half_frame_tolerance":True,"new_images":0})
    print(json.dumps({**manifest,"image_manifest_sha256":sha(args.output/"image_manifest.json")}),flush=True)


if __name__=="__main__":main()
