"""Same saved legal camera inputs; cold/warm latency and action validity, not SR."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.actions import parse_action
from semantic_robot.client import request_action


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--uri",default="http://127.0.0.1:8897")
    p.add_argument("--calls",type=int,default=8)
    a=p.parse_args()
    if not 2<=a.calls<=16:
        raise ValueError("Bounded latency probe allows 2..16 calls")
    source=Path(a.input); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    images={name:np.asarray(Image.open(source/f"{name}.png").convert("RGB"))
            for name in ("head_rgb","left_wrist_rgb","right_wrist_rgb")}
    text=(source/"prompt.txt").read_text()
    rows=[]
    with (out/"calls.jsonl").open("x",buffering=1) as log:
        for i in range(a.calls):
            result=request_action(a.uri,images,text,timeout=120)
            try:
                parse_action(result["action"])
                result["valid"]=True
            except ValueError as exc:
                result.update(valid=False,parse_error=str(exc))
            result["warmup"]=i==0
            log.write(json.dumps(result)+"\n"); rows.append(result)
            print(json.dumps(result),flush=True)
    warm=rows[1:]
    result={"kind":"same_view_latency_not_policy_success","source":str(source),"calls":len(rows),
            "valid":sum(r["valid"] for r in rows),"cold":rows[0],
            "warm_generation_median_s":float(np.median([r["generation_s"] for r in warm])),
            "warm_roundtrip_median_s":float(np.median([r["client_roundtrip_s"] for r in warm])),
            "warm_roundtrip_p95_s":float(np.percentile([r["client_roundtrip_s"] for r in warm],95)),
            "full_success_claim":False}
    (out/"result.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
