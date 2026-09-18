"""Read-only robot-relative viewpoint audit; never an affordance detector.

The old observed surface anchor is fixed in the hand only as a diagnostic
proxy. No object pose, segmentation, attachment truth or new model call is used.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.protocol import ROTATIONS,TRANSLATIONS


def view_direction(head,hand,anchor_hand):
    """Direction from an old hand-fixed anchor toward camera, in hand axes."""
    vector=hand[:3,:3].T@(head[:3,3]-hand[:3,3])-np.asarray(anchor_hand,float)
    length=np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length<1e-6:raise ValueError("Invalid camera-anchor separation")
    return vector/length


def angle(a,b):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a,b),-1.,1.))))


def audit(run):
    model=RobotModel(json.loads((run/"robot_calibration.json").read_text()))
    terminal=json.loads((run/"result.json").read_text())
    actions={r["decision"]:r["action"] for r in terminal["decisions"] if "action" in r}
    groups=defaultdict(list)
    for folder in sorted(run.glob("decision_*")):
        h=json.loads((folder/"harness.json").read_text());context=h.get("held_inspection",{})
        if not context.get("valid"):continue
        arm=context["reference_hand"];anchor=context["anchor"]
        if anchor["source"]!="previous_verified_onboard_RGBD_surface_not_current_affordance":raise ValueError("Unknown anchor source")
        q=np.asarray(json.loads((folder/"proprio.json").read_text())["q"])
        head=model.forward(q,"camera_head");hand=model.forward(q,arm)
        key=(h["goal_index"],arm,tuple(anchor["point_hand_m"]))
        groups[key].append({"decision":int(folder.name.split("_")[1]),
            "direction":view_direction(head,hand,anchor["point_hand_m"]),"relative":np.linalg.inv(head)@hand,
            "context":context})
    reports=[]
    for (goal,arm,anchor),rows in groups.items():
        pieces=defaultdict(lambda:{"segments":0,"relative_translation_m":0.,"relative_rotation_deg":0.,"view_direction_travel_deg":0.})
        for previous,current in zip(rows,rows[1:]):
            if current["decision"]!=previous["decision"]+1:raise ValueError("Missing observation interval")
            action=actions.get(previous["decision"],{});move=action.get("move")
            kind="rotation" if move in ROTATIONS else "translation" if move in TRANSLATIONS else "other"
            x=pieces[kind];x["segments"]+=1
            x["relative_translation_m"]+=float(np.linalg.norm(current["relative"][:3,3]-previous["relative"][:3,3]))
            x["relative_rotation_deg"]+=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(current["relative"][:3,:3]@previous["relative"][:3,:3].T).as_rotvec())))
            x["view_direction_travel_deg"]+=angle(previous["direction"],current["direction"])
        directions=[r["direction"] for r in rows]
        reports.append({"goal_index":goal,"hand":arm,"first_decision":rows[0]["decision"],"last_decision":rows[-1]["decision"],
            "observations":len(rows),"by_previous_command":dict(pieces),
            "max_direction_from_initial_deg":max(angle(directions[0],v) for v in directions),
            "max_pairwise_direction_deg":max(angle(a,b) for a in directions for b in directions),
            "final_direction_from_initial_deg":angle(directions[0],directions[-1]),
            "rows":[{"decision":r["decision"],"direction_in_hand":r["direction"].tolist(),
                     "angle_from_initial_deg":angle(directions[0],r["direction"])} for r in rows]})
    return {"run":str(run),"controls":0,"model_calls":0,"groups":reports,
        "limitations":["Hand-fixed old surface is a proxy, not a measured current object orientation or button", "No visibility, occlusion, grasp rigidity or information-gain certificate", "Translation is observed endpoint path, not continuous-path collision clearance", "Post-run only; never supplied to actor"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError("Preserve previous audit")
    result=audit(a.run)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({**result,"groups":[{k:v for k,v in r.items() if k!="rows"} for r in result["groups"]]},indent=2))


if __name__=="__main__":main()
