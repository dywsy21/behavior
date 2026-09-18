"""Offline-only physical receipts; privileged audit never feeds the actor."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common import sha,write_json
from stratify_static import low_velocity


RUNS=("radio_base_v1","radio_ft_v1","plates_ft_v1","plates_base_v1")


def rotation_path(quaternions):
    rotations=Rotation.from_quat(quaternions)
    increments=(rotations[1:]*rotations[:-1].inv()).as_rotvec()
    yaw=np.unwrap(rotations.as_euler("xyz")[:,2])
    return {"decision_endpoint_rotation_travel_rad":float(np.linalg.norm(increments,axis=1).sum()),
        "decision_endpoint_signed_yaw_travel_rad":float(yaw[-1]-yaw[0]),
        "decision_endpoint_absolute_yaw_travel_rad":float(np.abs(np.diff(yaw)).sum())}


def summarize(folder):
    result=json.loads((folder/"result.json").read_text());manifest=json.loads((folder/"manifest.json").read_text())
    initial=json.loads((folder/"PRIVILEGED_INITIAL_AUDIT.json").read_text())
    audits=[json.loads(p.read_text()) for p in sorted(folder.glob("decision_*/PRIVILEGED_POST_ACTION_GRASP_AUDIT.json"))]
    poses=[initial,*audits];positions=np.asarray([r["robot_world_position_m"] for r in poses])
    endpoint=positions[-1]-positions[0]
    angular=(Rotation.from_quat(poses[-1]["robot_world_quaternion_xyzw"])*Rotation.from_quat(poses[0]["robot_world_quaternion_xyzw"]).inv()).as_rotvec()
    objects=Counter()
    for r in audits:
        for arm,obj in (r.get("assisted_objects") or {}).items():
            if obj is not None:objects[(arm,obj)]+=1
    decisions=result["decisions"]
    actor_proprio=[json.loads((folder/f"decision_{r['decision']:03d}"/"proprio.json").read_text()) for r in decisions]
    requests=[json.loads((folder/f"decision_{r['decision']:03d}"/"request_without_images.json").read_text()) for r in decisions]
    return {"result_sha256":sha(folder/"result.json"),"manifest_sha256":sha(folder/"manifest.json"),"task":result["task"],
        "variant":result["variant"],"code_commit":manifest["code_commit"],"implementation_digest":result["implementation_digest"],
        "controls":result["controls"],"expert_prefix_controls":result["prefix_controls"],"decisions":len(decisions),
        "official_success":result["official_success"],"stop_reason":result["stop_reason"],"wall_s_after_prefix":result["wall_s_after_prefix"],
        "actor_distribution_audit":{"decisions_with_low_base_velocity":sum(low_velocity({"proprio":p}) for p in actor_proprio),
            "decisions_with_empty_action_history":sum(not p["history"] for p in requests),
            "low_velocity_definition":"XY speed <0.02 m/s AND abs yaw speed <0.03 rad/s, identical to static audit"},
        "action_counts":dict(Counter(r["token"] for r in decisions)),"execution_status":dict(Counter(r["feedback"]["status"] for r in decisions)),
        "rejection_reasons":dict(Counter(r["feedback"].get("reason","") for r in decisions if r["feedback"]["status"]=="REJECTED_NO_MOTION")),
        "actual_world_base_endpoint_delta_m":endpoint.tolist(),"actual_world_base_endpoint_rotation_rad":angular.tolist(),
        **rotation_path([r["robot_world_quaternion_xyzw"] for r in poses]),
        "decision_endpoint_base_path_m":float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
        "eef_base_endpoint_path_m":{a:sum(float(np.linalg.norm(r["feedback"].get("eef_delta_m",{}).get(a,[0,0,0]))) for r in decisions) for a in ("left","right")},
        "assisted_attachment_audit_counts":[{"arm":a,"object":o,"decisions_attached":n} for (a,o),n in objects.items()],
        "initial_pose":poses[0],"final_pose":poses[-1],"video_sha256":sha(folder/"rollout.mp4"),
        "latency_s":np.quantile([r["response"]["latency_s"] for r in decisions],[.5,.95]).tolist() if decisions else [],
        "limits":["Path measured only at decision endpoints, not swept arc length", "Assisted attachment is simulator diagnostic, not independent grip-strength success", "Fixed-skill development start; no full-task success-rate claim"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError("Preserve existing summary")
    rows={name:summarize(args.root/name) for name in RUNS}
    pairs={}
    for a,b in (("radio_base_v1","radio_ft_v1"),("plates_base_v1","plates_ft_v1")):
        first,second=rows[a],rows[b]
        if first["implementation_digest"]!=second["implementation_digest"]:raise ValueError("Mismatched physical code")
        p0,p1=(r["initial_pose"] for r in (first,second))
        pairs[str(first["task"])]= {"initial_world_position_difference_m":float(np.linalg.norm(np.asarray(p0["robot_world_position_m"])-p1["robot_world_position_m"])),
            "initial_world_rotation_difference_rad":float(np.linalg.norm((Rotation.from_quat(p0["robot_world_quaternion_xyzw"])*Rotation.from_quat(p1["robot_world_quaternion_xyzw"]).inv()).as_rotvec())),
            "first_actor_proprio_exact_equal":json.loads((args.root/a/"decision_000/proprio.json").read_text())==json.loads((args.root/b/"decision_000/proprio.json").read_text()),
            "first_head_rgb_sha_equal":sha(args.root/a/"decision_000/head.png")==sha(args.root/b/"decision_000/head.png")}
    result={"runs":rows,"reset_comparison":pairs,"full_task_success_rate_claim":False,"all_evidence_offline_only":True}
    write_json(args.output,result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
