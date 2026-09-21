"""Report all six preregistered slots, including failed and missing attempts."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import sha,write_json
from native_evaluation import VARIANTS
from native_eval_prepare import STARTS,ROOT,SCHEMA
from native_actor_protocol import validate_actor
from native_execution import authorization_profile


def summarize(root):
    root=Path(root);rows=[];actors={};identities=set()
    for instance,item in STARTS.items():
        for variant in VARIANTS:
            run=root/f"eval_t1_i{instance}_{variant}_v1";result_path=run/"result.json"
            row={"instance":instance,"stratum":item["stratum"],"variant":variant,"run":str(run),
                "status":"NOT_RUN","counted_local_success":False,"counted_in_registered_denominator":True}
            if (run/"manifest.json").exists():
                m=json.loads((run/"manifest.json").read_text());auth=m["authorization"];prepared=m["preparation"]
                if (auth.get("schema")!=SCHEMA or auth.get("source")!=[1,item["episode"],instance] or
                        auth.get("variant")!=variant or prepared.get("prefix_controls")!=item["prefix"] or
                        m.get("oracle_actor_feedback") is not False or m.get("specified_hand") is not None):
                    raise ValueError("Registered paired identities changed")
                profile=authorization_profile(auth)
                identities.add((m["code_commit"],m["protocol"],m["dataset_sha256"],profile))
                row["execution_profile"]=profile
                row["status"]="INCOMPLETE_ATTEMPT"
            if result_path.exists():
                r=json.loads(result_path.read_text())
                if r.get("manifest_sha256")!=sha(run/"manifest.json") or r.get("variant")!=variant or r.get("instance")!=instance:
                    raise ValueError("Result is not bound to this paired run")
                valid=(r.get("status")=="COMPLETE" and r.get("final_hold") is True and
                    not (run/"failure.json").exists() and r.get("issued_native")==r.get("native_controls") and
                    r.get("issued_prefix")==r.get("prefix_controls")==item["prefix"] and
                    1<=r["native_controls"]<=420 and len(r.get("decisions",[]))<=12 and
                    r.get("wall_seconds_after_reset",float("inf"))<=1200)
                row.update(status=r.get("status"),valid_bounded_result=valid,result_sha256=sha(result_path),
                    counted_local_success=bool(valid and r.get("score",{}).get("any_hand_local_success") is True),
                    score=r.get("score"),official_success_diagnostic=r.get("official_success"),
                    paid_prefix_controls=r.get("prefix_controls"),new_controls=r.get("native_controls"),
                    stop=r.get("stop"),neural_request_attempts=r.get("neural_request_attempts"),
                    confirmed_service_calls=r.get("confirmed_service_calls"))
            elif (run/"failure.json").exists():row["status"]="FAILED_WITHOUT_RESULT"
            actor_path=run/"decision_00/request.json"
            if actor_path.exists():actors[(instance,variant)]=validate_actor(json.loads(actor_path.read_text())["actor"])
            rows.append(row)
    if len(identities)>1:raise ValueError("Comparison does not share exact source/protocol/TRAIN dataset/execution profile")
    paired=[]
    for instance in STARTS:
        group=[actors.get((instance,v)) for v in VARIANTS]
        record={"instance":instance,"all_three_initial_observations":all(a is not None for a in group)}
        if record["all_three_initial_observations"]:
            if any(a["active_instruction"]!=group[0]["active_instruction"] or a["task"]!=group[0]["task"] or a["history"] for a in group):
                raise ValueError("Paired initial legal task/instruction/history changed")
            record.update(initial_proprio_bit_equal=all(a["proprio"]==group[0]["proprio"] for a in group),
                initial_RGB_bit_equal=all(a["current_rgb_sha256"]==group[0]["current_rgb_sha256"] for a in group),
                max_initial_joint_difference_rad=max(float(np.max(np.abs(np.array(a["proprio"]["joint_positions_rad"][arm])-
                    group[0]["proprio"]["joint_positions_rad"][arm]))) for a in group for arm in ("torso","left","right")))
        paired.append(record)
    complete=all(r["status"] in ("COMPLETE","FAILED","FAILED_WITHOUT_RESULT") for r in rows)
    return {"status":"ALL_SIX_ATTEMPTS_FINISHED" if complete else "INCOMPLETE_DO_NOT_CLAIM_FINISHED_COMPARISON",
        "rows":rows,"paired_start_checks":paired,"per_variant_registered_numerator_denominator":{
            v:[sum(r["counted_local_success"] for r in rows if r["variant"]==v),2] for v in VARIANTS},
        "scope":"Paid near-grasp prefix local skill only; OOD strata have n=1 per policy",
        "all_attempts_reported":True,"statistical_significance_claim":False,"full_task_SR_claim":False,
        "visual_causality_claim":False,"training_loss_is_success_metric":False}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=summarize(a.root);write_json(a.output,result);print(json.dumps(result,indent=2))
