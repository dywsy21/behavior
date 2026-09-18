"""Single task3 engineering probe; simulator truth stays in an audit-only file.

This does not run a policy. Its source digest includes this probe, so its short
gate can NEVER authorize a standard run_v2 agent. No global pose is persisted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import run_v2
from semantic_robot.og_backend import array
from semantic_robot.v2.protocol import Action, HOLD


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--gpu",type=int,default=3)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise ValueError("Preserve prior probe")
    base=run_v2.CalibratedRobot
    class AuditRobot(base):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.audit_previous=None;self.audit_stream=None

        def state(self):
            state=super().state()
            import omnigibson as og
            t=float(og.sim.current_time)
            position,quat=(array(v).copy() for v in self.robot.get_position_orientation())
            current=(t,position,quat,state.base_velocity.copy())
            if self.audit_stream is None:
                self.audit_stream=(out/"odometry_audit.jsonl").open("x",buffering=1)
            if self.audit_previous is not None and t>self.audit_previous[0]+1e-6:
                t0,p0,q0,v0=self.audit_previous
                R0=Rotation.from_quat(q0);R1=Rotation.from_quat(quat)
                rel=R0.inv()*R1
                dp=R0.inv().apply(position-p0)
                row={"dt":t-t0,"physics_dt":float(og.sim.get_physics_dt()),
                     "rendering_dt":float(og.sim.get_rendering_dt()),"sim_step_dt":float(og.sim.get_sim_step_dt()),
                     "base_velocity_before":v0.tolist(),"base_velocity_after":state.base_velocity.tolist(),
                     "diagnostic_body_delta_m":dp.tolist(),"diagnostic_body_rotvec_rad":rel.as_rotvec().tolist(),
                     "no_global_pose_persisted":True,"never_sent_to_actor":True}
                self.audit_stream.write(json.dumps(row)+"\n")
            if self.audit_previous is None or t>self.audit_previous[0]+1e-6:self.audit_previous=current
            return state

    run_v2.CalibratedRobot=AuditRobot
    actions=[HOLD]+[Action("base","yaw_plus","coarse")]*4+[Action("base","yaw_minus","coarse")]*2+[
        Action("base","forward"),Action("base","back"),HOLD]
    run_v2.bounded_gate=lambda grounded:actions
    original_digest=run_v2.implementation_digest
    run_v2.implementation_digest=lambda:hashlib.sha256(
        original_digest().encode()+Path(__file__).read_bytes()+b"AUDIT_ONLY_NOT_POLICY_GATE").hexdigest()
    sys.argv=[str(Path(run_v2.__file__)),"--output",str(out),"--gpu",str(args.gpu),"--task","3",
              "--mode","gate","--harness","grounded","--max-decisions","10",
              "--max-controls","320","--max-seconds","600"]
    run_v2.main()


if __name__=="__main__":main()
