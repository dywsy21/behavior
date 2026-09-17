"""Read-only kinematic replay against saved native robot poses, no simulator.

This tests FK outside the calibration pose, including old near-limit failures.
It does NOT validate contact dynamics, camera poses absent from old logs, or SR.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.v2.kinematics import RobotModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration",required=True)
    p.add_argument("--saved-result",required=True)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    model = RobotModel(json.loads(Path(args.calibration).read_text()))
    source = Path(args.saved_result)
    rows = []
    for row in json.loads(source.read_text())["rows"]:
        state = row["proprio"]; q = np.asarray(state["q"])
        errors = {}
        for name,(position,orientation) in model.poses(q).items():
            native_p,native_q = state["poses"][name]
            errors[name] = {"position_m":float(np.linalg.norm(position-native_p)),
                           "angle_rad":float(np.linalg.norm((Rotation.from_quat(orientation)*Rotation.from_quat(native_q).inv()).as_rotvec()))}
        rows.append({"index":row["index"],"source":row["source"],"errors":errors,
                     "minimum_joint_margin_rad":float(np.minimum(q-model.lower,model.upper-q).min())})
    result = {"calibration_sha":model.sha,"saved_result_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "cases":len(rows),"new_actuations":0,"rows":rows,
              "max_position_m":max(e["position_m"] for row in rows for e in row["errors"].values()),
              "max_angle_rad":max(e["angle_rad"] for row in rows for e in row["errors"].values()),
              "not_a_physical_contact_or_success_test":True}
    result["passed"] = result["max_position_m"] < .003 and result["max_angle_rad"] < .02
    with Path(args.output).open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
