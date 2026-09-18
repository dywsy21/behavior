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


def grasp_timeline(rows):
    """Audit every executed action, not only the first active grasp probes.

    Attachment is privileged *diagnostic* evidence, never actor input or a task
    success predicate. Missing receipts are unknown, not evidence of an empty hand.
    """
    arms=("left","right");previous={arm:None for arm in arms}
    result={"diagnostic_only_not_actor_input":True,"not_task_success":True,
            "close_decisions":[],"open_decisions":[],"missing_post_action_receipts":[],
            "unavailable_attachment_diagnostics":[],
            "attachment_observations":[],"attachment_changes":[],
            "opens_after_observed_attachment":[],"registered_verification_decisions":[]}
    for row in rows:
        execution=row.get("execution")
        if not execution or "feedback" not in execution:continue
        i=row["decision"];action=execution["action"]
        if action["move"] in ("close","open"):
            result[action["move"]+"_decisions"].append(i)
        record=row.get("post_action_grasp_audit")
        if record is None:
            result["missing_post_action_receipts"].append(i)
            previous={arm:None for arm in arms}
        else:
            objects=record.get("assisted_objects")
            if not isinstance(objects,dict):
                result["unavailable_attachment_diagnostics"].append(i)
                previous={arm:None for arm in arms}
                objects={}
            for arm in arms:
                if arm not in objects:
                    previous[arm]=None
                    continue
                obj=objects[arm];old=previous[arm]
                if obj is not None:
                    result["attachment_observations"].append({"decision":i,"arm":arm,"object":obj})
                if old is not None:
                    if old["object"]!=obj:
                        result["attachment_changes"].append({"decision":i,"arm":arm,
                            "before":old["object"],"after":obj,"action":action})
                    if old["object"] is not None and action["move"]=="open" and action["part"] in (arm,"both"):
                        result["opens_after_observed_attachment"].append({"decision":i,"arm":arm,
                            "previous_decision":old["decision"],"object_before":old["object"],"object_after":obj})
                previous[arm]={"decision":i,"object":obj}
        if row.get("registered_grasp_motion",{}).get("verified"):
            result["registered_verification_decisions"].append(i)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--output",required=True)
    p.add_argument("--stride",type=int,default=5)
    args=p.parse_args()
    if not 1<=args.stride<=12:raise ValueError("Bounded panel stride required")
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    terminal_path=next((run/name for name in ("result.json","failure.json","operator_stop.json") if (run/name).exists()),None)
    if terminal_path is None:raise ValueError("No real terminal or explicit operator-stop record")
    terminal=read(terminal_path)
    if terminal_path.name=="operator_stop.json":
        # The evaluator's signal hook may quit Kit before Python's finally.
        # Preserve this distinction; recover only already-written decisions.
        trace=[json.loads(line) for line in (run/"steps.jsonl").read_text().splitlines()]
        terminal["decisions"]=[row for row in trace if "action" in row and "feedback" in row]
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
             "execution":decisions.get(i),"completed_claims":h.get("completed_observation_claims",[]),
             "registered_grasp_motion":read(source/"grounded_progress.json").get("registered_grasp_motion",{}),
             "post_action_grasp_audit":(read(source/"PRIVILEGED_POST_ACTION_GRASP_AUDIT.json")
                                        if (source/"PRIVILEGED_POST_ACTION_GRASP_AUDIT.json").exists() else None)}
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
        elif ((row.get("execution") or {}).get("action",{}).get("move") in ("close","open") or
              row.get("registered_grasp_motion",{}).get("verified") or
              (i and row["post_action_grasp_audit"]!=rows[i-1]["post_action_grasp_audit"])):
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
    result={"run":str(run),"terminal_record":terminal_path.name,"terminal_sha256":sha(terminal_path.read_bytes()),
            "terminal":{k:v for k,v in terminal.items() if k!="decisions"},
            "completed_execution_count":sum("feedback" in d for d in decisions.values()),
            "actions":dict(Counter("/".join((d["action"]["part"],d["action"]["move"],d["action"]["scale"])) for d in decisions.values() if "action" in d)),
            "hash_checks":checks,"hash_mismatches":bad,"selected_panel_decisions":[r["decision"] for r in selected],
            "grasp_diagnostic":grasp_timeline(rows),
            "rows":rows,"human_review_not_automated":True,"new_model_calls":0,"new_controls":0}
    (out/"audit.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","terminal")},indent=2))
    if bad:raise ValueError("Saved RGB-D receipt mismatch")


if __name__=="__main__":main()
