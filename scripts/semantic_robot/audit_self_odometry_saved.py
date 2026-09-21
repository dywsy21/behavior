"""Read-only fixed full-run RGB-D replay; no new simulation or tuned selection.

Transported boxes are ONLY a diagnostic for unchanged, calibrated-open hands.
The deployed runner instead captures actual native robot geometry every sample.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.odometry import RGBDMotion
from semantic_robot.v2.self_odometry import make_frame
from semantic_robot.v2.substep_odometry import signature


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("--seconds",type=int,default=600)
    a=p.parse_args();started=time.monotonic();deadline=started+min(a.seconds,600)
    read=lambda path:json.loads(path.read_text())
    model=RobotModel(read(a.run/"robot_calibration.json"))
    def frame(path):
        rgb=np.asarray(Image.open(path/"CURRENT_HEAD_RAW.png"))
        with np.load(path/"depth.npz") as z:depth=z["head"].copy()
        s=read(path/"proprio.json")
        return {"head_rgb":rgb},{"head":depth},SimpleNamespace(q=np.asarray(s["q"]),gripper=np.asarray(s["gripper"]))
    pairs=regressions=rescued=old_valid=new_valid=0
    for chain_path in sorted(a.run.glob("decision_*/action_motion.json")):
        folder=chain_path.parent;chain=read(chain_path)
        reference=frame(folder);original=read(folder/"robot_self_geometry.json")
        expected=model.spec["metadata"]["grasp_region_reference_gripper_m"]
        if np.max(abs(reference[2].gripper-expected))>.0005:
            raise ValueError("Offline FK transport is only registered for open-hand diagnostics")
        if any("link:"+b["link"] not in model.links for b in original["boxes"]):
            raise ValueError("Incomplete portable robot link coverage")
        def boxes(state):
            if np.max(abs(state.gripper-reference[2].gripper))>=1e-5:
                raise ValueError("Cannot transport finger boxes across aperture changes")
            value=copy.deepcopy(original)
            for b in value["boxes"]:
                key="link:"+b["link"]
                T=model.forward(state.q,key)@np.linalg.inv(model.forward(reference[2].q,key))@np.asarray(b["T_base_link"])
                b["T_base_link"]=T.tolist()
            return value
        previous=folder
        for segment in chain["segments"]:
            if time.monotonic()>=deadline:raise TimeoutError("Finite saved-input diagnostic budget")
            current=folder/"motion_substeps"/f"control_{segment['control_end']:06d}"
            left,right=frame(previous),frame(current)
            for expected_sig,data in ((segment["before"],left),(segment["after"],right)):
                actual=signature(data[0],data[1],model,data[2].q)
                for key in ("rgb_sha256","depth_sha256"):assert actual[key]==expected_sig[key]
                np.testing.assert_allclose(actual["camera_fk"],expected_sig["camera_fk"],atol=1e-12,rtol=0)
            baseline=RGBDMotion("rgbd_joint")
            baseline.observe(left[0],left[1],model,left[2].q)
            old=baseline.observe(right[0],right[1],model,right[2].q)
            for key in ("valid","matches","unique_matches","inliers","reason"):
                assert old.get(key)==segment["measurement"].get(key),(chain_path,key,old.get(key),segment["measurement"].get(key))
            filtered=RGBDMotion("rgbd_joint",exclude_robot=True)
            for data,control in ((left,segment["control_start"]),(right,segment["control_end"])):
                self_frame=make_frame(data[0],data[1],model,data[2],control,boxes(data[2]))
                new=filtered.observe(data[0],data[1],model,data[2].q,
                    robot_frame=self_frame,gripper=data[2].gripper,control=control)
            fields=("valid","reason","matches","unique_matches","inliers","inlier_fraction",
                    "median_reprojection_px","median_reverse_reprojection_px","median_depth_correspondence_m","body_delta")
            compact=lambda v:{k:v.get(k) for k in fields}
            print(json.dumps({"decision":folder.name,"controls":[segment["control_start"],segment["control_end"]],
                "baseline":compact(old),"self_excluded":compact(new),"filter":new["robot_self_filter"]}),flush=True)
            pairs+=1;old_valid+=bool(old["valid"]);new_valid+=bool(new["valid"])
            regressions+=bool(old["valid"] and not new["valid"]);rescued+=bool(not old["valid"] and new["valid"])
            previous=current
    print(json.dumps({"summary":True,"seconds":time.monotonic()-started,"pairs":pairs,"baseline_valid":old_valid,
        "self_excluded_valid":new_valid,"regressions":regressions,"rescued":rescued,"new_physics":0,
        "original_thresholds_unchanged":True,"new_closed_loop_effect_not_proven":True}),flush=True)


if __name__=="__main__":main()
