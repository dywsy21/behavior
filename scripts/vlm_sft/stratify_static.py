"""Zero-new-inference audit of velocity/history deployment distribution shift."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from common import sha,write_json


def low_velocity(row):
    v=np.asarray(row["proprio"]["base_velocity_local"])
    return np.linalg.norm(v[:2])<.02 and abs(v[2])<.03


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--evaluation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError("Preserve prior audit")
    source={split:[json.loads(x) for x in (args.data/(split+"_images.jsonl")).read_text().splitlines()] for split in ("train","validation","test")}
    pred={}
    for line in (args.evaluation/"predictions.jsonl").read_text().splitlines():
        r=json.loads(line);pred[(r["id"],r["arm"])]=r["prediction"]
    strata={}
    for name,condition in (("low_current_base_velocity",low_velocity),("nonlow_current_base_velocity",lambda r:not low_velocity(r)),
                           ("empty_action_history",lambda r:not r["history"]),("nonempty_action_history",lambda r:bool(r["history"]))):
        rows=[r for r in source["test"] if condition(r)]
        strata[name]={"n":len(rows),"target_families":dict(Counter(r["target"].split("_")[0] for r in rows)),
            "correct":{arm:sum(pred[(r["id"],arm)]==r["target"] for r in rows) for arm in ("base","finetuned")}}
    result={"new_model_calls":0,"low_velocity_definition":"XY speed <0.02 m/s AND abs yaw speed <0.03 rad/s; thresholds not optimized",
        "split_counts":{s:{"n":len(rows),"low_velocity":sum(low_velocity(r) for r in rows),"empty_history":sum(not r["history"] for r in rows)} for s,rows in source.items()},
        "test_strata":strata,"predictions_sha256":sha(args.evaluation/"predictions.jsonl"),
        "interpretation_limit":"Low body velocity does not imply static arms; observational association, not a causal image/proprio ablation"}
    write_json(args.output,result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
