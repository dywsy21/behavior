"""Read completed gate sensors to preflight a finite round-trip test, no physics."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.observer_gate import choose_gate_pair


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("Preserve prior result")
    started=time.monotonic();directory=a.run/"decision_007"
    model=RobotModel(json.loads((a.run/"robot_calibration.json").read_text()))
    raw=json.loads((directory/"proprio.json").read_text())
    state=model.state(np.array(raw["q"]),np.array(raw["gripper"]),np.zeros(3))
    with np.load(directory/"depth.npz") as bundle:depths={v:bundle[v] for v in bundle.files}
    geometry=json.loads((directory/"robot_self_geometry.json").read_text())
    forward,back,receipt=choose_gate_pair(model,state,SafeServo(model,state,[1,1]),depths,geometry)
    result={"forward":asdict(forward) if forward else None,"reverse":asdict(back) if back else None,
            "receipt":receipt,"controls":0,"model_calls":0,"wall_seconds":time.monotonic()-started}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result))
    if forward is None:raise SystemExit(1)


if __name__=="__main__":main()
