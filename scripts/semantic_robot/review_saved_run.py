"""Read-only rollout audit and human panels. Never feeds the actor or simulator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read(path):return json.loads(path.read_text()) if path.exists() else {}
def sha(value):return hashlib.sha256(value).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--output",required=True)
    p.add_argument("--stride",type=int,default=5)
    args=p.parse_args()
    if not 1<=args.stride<=12:raise ValueError("Bounded panel stride required")
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    terminal=read(run/"result.json") or read(run/"failure.json")
    decisions={d["decision"]:d for d in terminal.get("decisions",[])}
    rows=[];checks=0;bad=[]
    views=("head","left_wrist","right_wrist")
    for source in sorted(run.glob("decision_*")):
        i=int(source.name.split("_")[-1]);h=read(source/"harness.json")
        raw=read(source/"observation.json").get("result",{}).get("text")
        obs={} if raw is None else json.loads(raw)
        g=h.get("target_surface_estimate",{})
        row={"decision":i,"stage":h.get("stage"),"goal_index":h.get("goal_index"),
             "visible":obs.get("visible"),"view":obs.get("view"),"raw_observation":raw,
             "target_valid":g.get("valid"),"target_reason":g.get("reason"),
             "target_distance_m":g.get("distance_to_active_closing_center_m"),
             "search":h.get("search"),"refinement":read(source/"refinement.json").get("choice"),
             "execution":decisions.get(i),"completed_claims":h.get("completed_observation_claims",[])}
        rows.append(row)
        receipt=read(source/"depth_receipt.json")
        if receipt:
            with np.load(source/"depth.npz") as depths:
                for view in views:
                    rgb=np.asarray(Image.open(source/("CURRENT_"+view.upper()+"_RAW.png")))
                    for kind,data in (("rgb",rgb),("depth",depths[view])):
                        checks+=1
                        if sha(data.tobytes())!=receipt[view][kind+"_sha256"]:
                            bad.append({"decision":i,"view":view,"kind":kind})
    # Uniform coverage plus every transition / new visual target / refinement.
    selected=[]
    for i,row in enumerate(rows):
        key=lambda r:(r["stage"],r["goal_index"],r["visible"],r["view"])
        if i%args.stride==0 or i==len(rows)-1 or (i and key(row)!=key(rows[i-1])) or row["refinement"] is not None:
            selected.append(row)
    for page in range(0,len(selected),2):
        canvas=Image.new("RGB",(960,704),(22,24,30));draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(selected[page:page+2]):
            y=j*352;i=row["decision"];source=run/f"decision_{i:03d}"
            action=(row["execution"] or {}).get("action",{})
            draw.text((6,y+4),f"{run.name} | decision {i} | {row['stage']} | {action}",fill="white")
            for col,view in enumerate(views):
                img=Image.open(source/("CURRENT_"+view.upper()+"_RAW.png")).convert("RGB").resize((320,320))
                canvas.paste(img,(col*320,y+30))
        canvas.save(out/f"panel_{page//2:02d}.jpg",quality=94)
    result={"run":str(run),"terminal_sha256":sha((run/("result.json" if (run/"result.json").exists() else "failure.json")).read_bytes()),
            "terminal":{k:v for k,v in terminal.items() if k!="decisions"},
            "completed_execution_count":sum("feedback" in d for d in decisions.values()),
            "actions":dict(Counter("/".join((d["action"]["part"],d["action"]["move"],d["action"]["scale"])) for d in decisions.values() if "action" in d)),
            "hash_checks":checks,"hash_mismatches":bad,"selected_panel_decisions":[r["decision"] for r in selected],
            "rows":rows,"human_review_not_automated":True,"new_model_calls":0,"new_controls":0}
    (out/"audit.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","terminal")},indent=2))
    if bad:raise ValueError("Saved RGB-D receipt mismatch")


if __name__=="__main__":main()
