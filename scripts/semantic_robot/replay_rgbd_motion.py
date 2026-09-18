"""Replay production sensor odometry on saved frames; optional audit-only truth."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.odometry import RGBDMotion


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();run=Path(args.run);out=Path(args.output)
    if out.exists():raise ValueError("Preserve prior replay")
    import cv2
    cv2.setNumThreads(2)
    model=RobotModel(json.loads((run/"robot_calibration.json").read_text()));est=RGBDMotion()
    terminal=json.loads((run/("result.json" if (run/"result.json").exists() else "failure.json")).read_text())
    executions={r["decision"]:r for r in terminal["decisions"]}
    audit=run/"odometry_audit.jsonl"
    truth=list(map(json.loads,audit.read_text().splitlines())) if audit.exists() else None
    offset=0;rows=[]
    for source in sorted(run.glob("decision_*")):
        i=int(source.name.split("_")[-1]);q=np.asarray(json.loads((source/"proprio.json").read_text())["q"])
        rgb=np.asarray(Image.open(source/"CURRENT_HEAD_RAW.png"));depth=dict(np.load(source/"depth.npz"))
        receipt=est.observe({"head_rgb":rgb},depth,model,q)
        row={"decision":i,"receipt":receipt,"raw_velocity_feedback":executions.get(i-1,{}).get("feedback",{}).get("base_integral")}
        if truth is not None and i:
            ticks=executions[i-1]["feedback"]["control_ticks"]
            segment=truth[offset:offset+ticks];offset+=ticks
            if len(segment)!=ticks:raise ValueError("Missing physical audit steps")
            motion=np.eye(4)
            for step in segment:
                T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(step["diagnostic_body_rotvec_rad"]).as_matrix()
                T[:3,3]=step["diagnostic_body_delta_m"];motion=motion@T
            expected=np.array([motion[0,3],motion[1,3],np.arctan2(motion[1,0],motion[0,0])])
            row["diagnostic_only_true_body_delta"]=expected.tolist()
            if receipt["valid"]:
                row["diagnostic_only_error"]= (np.asarray(receipt["body_delta"])-expected).tolist()
        rows.append(row)
    valid=[r for r in rows if r["receipt"]["valid"] and not r["receipt"].get("initial")]
    errors=[r["diagnostic_only_error"] for r in valid if "diagnostic_only_error" in r]
    result={"run":str(run),"pairs":max(0,len(rows)-1),"valid_pairs":len(valid),"new_controls":0,"new_model_calls":0,
            "production_estimator_no_truth_inputs":True,"truth_comparison_only_after_estimation":truth is not None,
            "max_abs_body_error":np.max(np.abs(errors),axis=0).tolist() if errors else None,"rows":rows}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
