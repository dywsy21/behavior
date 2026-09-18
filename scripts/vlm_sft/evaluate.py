"""Frozen, paired heldout decoding plus an explicit causal motion baseline."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from common import TOKENS,VERSION,directional,sha,write_json
from modeling import decode,encode,load_images,load_model
from prepare import reserve


def summarize(rows):
    classes=sorted({r["target"] for r in rows});byclass={}
    for token in classes:
        subset=[r for r in rows if r["target"]==token]
        byclass[token]={"correct":sum(r["prediction"]==token for r in subset),"n":len(subset)}
    errors=Counter();confusion=Counter()
    opposite={"FORWARD":"BACK","BACK":"FORWARD","LEFT":"RIGHT","RIGHT":"LEFT","UP":"DOWN","DOWN":"UP",
              "OPEN":"CLOSE","CLOSE":"OPEN","YAW_PLUS":"YAW_MINUS","YAW_MINUS":"YAW_PLUS"}
    for r in rows:
        p,t=r["prediction"],r["target"]
        if p==t:continue
        pp,_,pm=p.partition("_");tp,_,tm=t.partition("_")
        category="wrong_body_part" if pp!=tp else "opposite_direction" if opposite.get(tm)==pm else "wrong_direction"
        errors[category]+=1;confusion[(t,p)]+=1
    return {"n":len(rows),"correct":sum(r["prediction"]==r["target"] for r in rows),
        "part_correct":sum(r["prediction"].split("_")[0]==r["target"].split("_")[0] for r in rows),
        "accuracy":float(np.mean([r["prediction"]==r["target"] for r in rows])) if rows else None,
        "macro_recall":float(np.mean([v["correct"]/v["n"] for v in byclass.values()])) if byclass else None,
        "per_class":byclass,"failure_counts":dict(errors),
        "confusion":[{"target":t,"prediction":p,"n":n} for (t,p),n in confusion.most_common()],
        "latency_s_p50_p95":np.quantile([r["latency_s"] for r in rows],[.5,.95]).tolist() if rows else []}


def persistence_rule(row):
    velocity=np.asarray(row["proprio"]["base_velocity_local"])
    d=directional(velocity/[.12,.12,.25],("FORWARD","LEFT","YAW_PLUS"),("BACK","RIGHT","YAW_MINUS"),.15)
    return "BASE_"+d if d else "HOLD"


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--model",required=True);p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--split",choices=("validation","test"),default="test")
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="1":raise RuntimeError("Use owned GPU1")
    args.output.mkdir(parents=True,exist_ok=False);reserve(args.output)
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(41)
    rows=[json.loads(x) for x in (args.data/(args.split+"_images.jsonl")).read_text().splitlines()]
    rows=sorted(rows,key=lambda r:r["source_fingerprint"])
    if args.limit:rows=rows[:args.limit]
    repo=Path(__file__).resolve().parents[2]
    manifest={"protocol":VERSION,"code_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
        "data_sha256":sha(args.data/(args.split+"_images.jsonl")),"adapter_sha256":sha(args.adapter/"adapter_model.safetensors"),
        "split":args.split,"n":len(rows),"sample_ids":[r["id"] for r in rows],"same_inputs_and_grammar":True,
        "base_path":args.model,"paired_order":"alternate base-first/adapter-first by sample","selection_from_test":False}
    write_json(args.output/"manifest.json",manifest)
    model,processor=load_model(args.model,adapter=args.adapter)
    predictions={"base":[],"finetuned":[],"proprio_persistence":[]}
    ledger=(args.output/"predictions.jsonl").open("x",buffering=1)
    start=time.monotonic()
    for i,row in enumerate(rows):
        if time.monotonic()-start>3600:raise RuntimeError("Heldout evaluation one-hour budget reached")
        x=encode(processor,row,load_images(args.data,row),supervised=False)
        outputs={}
        for arm in (["base","finetuned"] if i%2==0 else ["finetuned","base"]):
            if arm=="base":
                with model.disable_adapter():result=decode(model,processor,x)
            else:result=decode(model,processor,x)
            result.update(id=row["id"],task_id=row["task_id"],episode_index=row["episode_index"],target=row["target"],arm=arm)
            predictions[arm].append(result);outputs[arm]=result
            ledger.write(json.dumps(result)+"\n")
        if outputs["base"]["input_ids_sha256"]!=outputs["finetuned"]["input_ids_sha256"]:
            raise RuntimeError("Paired model inputs differ")
        predictions["proprio_persistence"].append({"prediction":persistence_rule(row),"target":row["target"],"latency_s":0.,"task_id":row["task_id"]})
        if (i+1)%20==0:print(json.dumps({"completed":i+1,"n":len(rows),"elapsed_s":time.monotonic()-start}),flush=True)
    result={"status":"complete","metrics":{arm:summarize(values) for arm,values in predictions.items()},
            "by_task":{arm:{str(task):summarize([r for r in values if r["task_id"]==task]) for task in (0,1,3)} for arm,values in predictions.items()},
            "paired":{key:sum(fn(b,f) for b,f in zip(predictions["base"],predictions["finetuned"])) for key,fn in {
                "both_correct":lambda b,f:b["prediction"]==b["target"] and f["prediction"]==f["target"],
                "base_only_correct":lambda b,f:b["prediction"]==b["target"] and f["prediction"]!=f["target"],
                "finetuned_only_correct":lambda b,f:b["prediction"]!=b["target"] and f["prediction"]==f["target"],
                "both_wrong":lambda b,f:b["prediction"]!=b["target"] and f["prediction"]!=f["target"]}.items()},
            "elapsed_s":time.monotonic()-start,"manifest_sha256":sha(args.output/"manifest.json"),
            "limitations":["Heldout instances within three trained tasks; not unseen-task generalization.","Ground-truth same-state active instruction provided equally to both static models.","Direction-projection accuracy, not control or task success.","Proprio persistence rule exposes whether gains merely follow current body motion."]}
    write_json(args.output/"result.json",result)
    print(json.dumps({"complete":True,"n":len(rows),"base":result["metrics"]["base"]["correct"],"finetuned":result["metrics"]["finetuned"]["correct"]}),flush=True)


if __name__=="__main__":main()
