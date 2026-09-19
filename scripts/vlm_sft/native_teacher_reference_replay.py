"""One separately authorized original TRAIN full-segment reference replay.

Not a deployed actor, not native BC, not task SR. No policy, retry, reset loop,
state restoration or teleport. Every issued/completed control is accounted.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if __name__=="__main__" and sys.platform.startswith("linux"):
    native=Path(sys.prefix)/f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/pymeshlab/lib"
    if (native/"libGLU.so.1").exists() and str(native) not in os.environ.get("LD_LIBRARY_PATH","").split(":"):
        env=dict(os.environ);env["LD_LIBRARY_PATH"]=str(native)+":"+env.get("LD_LIBRARY_PATH","")
        os.execve(sys.executable,[sys.executable,*sys.argv],env)

import numpy as np
from PIL import Image
from native_teacher_collect import (REPO,ADAPTER,CalibratedRobot,OnboardRGBD,native_action,
    capture_snapshot,implementation_digest,preserved_session,report_failure)
from native_teacher_artifacts import ArtifactBudget,write_calibration,json_bytes,capture_upper_bound
from native_teacher_reference_contract import load_reference,check_actual_joint_bounds,source_state_diagnostic
from native_teacher_outcomes import LocalOutcome
from native_teacher_og import PrivilegedReader
from native_teacher_seed import extract_seed,seed_identity
from common import sha
from native_teacher_toggle import verify_installed_dependency,TOGGLE_SHA
from native_reference_profile import validate_execution_location,check_initialization_deadline


def main():
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--authorization",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,choices=(1,3),required=True);a=p.parse_args()
    code=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise ValueError("Clean immutable source required")
    release=json.loads(a.authorization.read_text());counts_path=REPO/"configs/vlm_sft/h09r_train_feasibility_counts.json"
    if sha(counts_path)!=release["h09r_counts_sha256"]:raise ValueError("TRAIN counts identity changed")
    ref,spec,prefix,segment,source_states,binding=load_reference(a.prepared,release,code,implementation_digest(),json.loads(counts_path.read_text()))
    validate_execution_location(release,a.output,a.gpu)
    verify_installed_dependency()
    a.output.mkdir(parents=True,exist_ok=False)
    budget=release["budget"];writer=ArtifactBudget(a.output,release["experiment_root"],budget["run_MiB"]*1024**2,budget["total_MiB"]*1024**2)
    if shutil.disk_usage(a.output).free<80*1024**3:raise RuntimeError("Disk reserve: no reset")
    if release.get("reference_profile") is not None and any(shutil.disk_usage(m).free<80*1024**3 for m in ("/mnt/sdc1","/mnt/nvme_tmp")):
        raise RuntimeError("Both-filesystem reserve: no reset")
    os.environ["OMNIGIBSON_GPU_ID"]=str(a.gpu);os.environ["BEHAVIOR_ACTION_STEPS"]="1"
    os.environ.setdefault("OMNIGIBSON_HEADLESS","1");os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    sys.path.insert(0,str(ADAPTER))
    from native_oracle_low_v1.official_factory import load_official_oracle_window,OfficialEvaluatorSession
    window=load_official_oracle_window(a.prepared/"window.json")
    if (window.task_name!=binding["window"]["task_name"] or window.instance_id!=ref["source"]["instance"] or
            window.official_mode!="train" or window.seed!=0 or not np.array_equal(window.frozen_window().prefix_actions,prefix)):
        raise ValueError("Factory changed the exact prepared reset/prefix")
    # Recheck all input bytes after factory loading, still before reset.
    load_reference(a.prepared,release,code,implementation_digest(),json.loads(counts_path.read_text()))
    identity=seed_identity(ref,spec)
    writer.write_json(a.output/"manifest.json",{"code":code,"authorization":release,"binding":binding,"identity":identity,
                     "models":0,"actor":None,"training_eligible":False,"reference_actions_are_not_native_BC":True,
                     "source_validation":"exact_actions_identity_clock; exported_proprio_error_diagnostic_only",
                     "installed_toggle_source_sha256":TOGGLE_SHA})
    completed=issued=0;terminal=False;primary=None;started=None;kin=reader=None;grips=None
    trace=private=None;rows=[];captures=[];final_hold=False;current_writer=writer
    def error(exc):
        nonlocal primary
        primary=primary or exc
        report_failure(a.output,primary,completed,started)
        try:writer.write_json(a.output/"control_failure_ledger.json",{"issued":issued,"completed":completed,
                      "inflight_action_may_have_partially_executed":issued!=completed,"original_error":repr(primary)},cleanup=True)
        except BaseException:pass
    initialization_started=time.monotonic()
    with preserved_session(lambda:OfficialEvaluatorSession(window,gpu=a.gpu),error,lambda:primary) as session:
        import omnigibson as og
        try:
            check_initialization_deadline(release,time.monotonic()-initialization_started)
            session.reset();started=time.monotonic();env=session.evaluator.env
            kin=CalibratedRobot(env.robots[0]);grips=np.clip(kin.state().gripper/.05*2-1,-1,1)
            # Cooperative boundary checks; constructor/reset are native blocking
            # calls, not asynchronously interrupted. A late completed reset can
            # still receive the existing finally hold, never begin the prefix.
            check_initialization_deadline(release,time.monotonic()-initialization_started)
            writer.write_json(a.output/"initialization.json",{"seconds":time.monotonic()-initialization_started,
                "reference_profile":release.get("reference_profile"),"cooperative_not_async_deadline":True})
            model=kin.calibrate(grounded=True);write_calibration(writer,model);onboard=OnboardRGBD(env)
            if any(e["position_m"]>.003 or e["angle_rad"]>.02 or e.get("jacobian_max_abs",0)>.04 for e in kin.compare(model).values()):
                raise RuntimeError("Reference FK mismatch")
            trace=(a.output/"control_trace.jsonl").open("x",buffering=1)
            private=(a.output/"PRIVATE_reference_trace.jsonl").open("x",buffering=1)
            def state():
                s=kin.state();return model.state(s.q,s.gripper,s.base_velocity)
            def step(command,phase,cleanup=False):
                nonlocal completed,issued,terminal,grips
                if terminal or issued>=budget["max_controls"]-(0 if cleanup else 1):raise RuntimeError("Reference terminal/control cap")
                if not cleanup and time.monotonic()-started>=budget["seconds_after_reset"]:raise TimeoutError("Reference time cap")
                check_actual_joint_bounds(state(),model)
                command=np.asarray(command,dtype=np.float32)
                if command.shape!=(23,) or not np.isfinite(command).all():raise ValueError("Finite native23 command required")
                line=json.dumps({"phase":phase,"issued_control":issued+1,"actual_action23":command.tolist()},allow_nan=False)+"\n"
                if not cleanup and shutil.disk_usage(a.output).free<80*1024**3:raise RuntimeError("Disk reserve")
                if cleanup:writer.check(len(line.encode()),cleanup=True)
                else:current_writer.check(len(line.encode())+16384)
                # Record issuance BEFORE env.step; completion is separate. A
                # thrown simulator step cannot disappear from the control ledger.
                if cleanup:trace.write(line)
                else:current_writer.append_text(trace,line)
                trace.flush();issued+=1;grips=command[[14,22]].copy()
                with og.sim.render_on_step(True):obs,_,terminated,truncated,_=env.step(command,n_render_iterations=1)
                completed+=1;terminal=bool(terminated or truncated)
                line=json.dumps({"completed_control":completed,"issued_control":issued,"terminal":terminal})+"\n"
                if cleanup:writer.check(len(line.encode()),cleanup=True);trace.write(line)
                else:current_writer.append_text(trace,line)
                e=session.evaluator;e.obs=e._preprocess_obs(e._sync_lights_and_get_obs(obs))
                return command
            def capture(name):
                directory=a.output/name
                snapshot=capture_snapshot(directory,model,state,onboard,kin.native_self_boxes,og.sim.render,
                                          {"control":completed},a.output,current_writer)
                captures.append({"control":completed,"files_sha256":{str(p.relative_to(a.output)):sha(p) for p in directory.iterdir()}})
                return snapshot
            for command in prefix:
                step(command,"prefix")
                if terminal:raise RuntimeError("Original reference prefix terminated")
            check_actual_joint_bounds(state(),model)
            before=capture("before")
            reader=PrivilegedReader(env,spec);oracle=LocalOutcome(spec)
            def measure(command,phase,cleanup=False,source_index=None):
                if source_index is not None and completed!=len(prefix)+source_index:
                    raise ValueError("Actual reference/source action clock mismatch")
                current=state();check_actual_joint_bounds(current,model)
                f=reader.read(completed);f["finger_opening"]=dict(zip(("left","right"),current.gripper.tolist()))
                reader.check_local_fk(current,f)
                token=None
                if command is not None:
                    grip=command[14 if spec["hand"]=="left" else 22]
                    token=spec["hand"].upper()+("_CLOSE" if grip<-.5 else "_OPEN") if abs(grip)>.5 else None
                verdict=oracle.update(f,token)
                diagnostic=(None if source_index is None else source_state_diagnostic(
                    current,source_states[source_index],len(prefix)+source_index))
                row={"frame":f,"actual_action23":None if command is None else command.tolist(),"phase":phase,"verdict":verdict,
                     "actual_q":current.q.tolist(),"actual_gripper":current.gripper.tolist(),
                     "source_state_diagnostic":diagnostic}
                line=json.dumps(row,allow_nan=False)+"\n"
                if len(line.encode())>16384:raise RuntimeError("Reference telemetry exceeded bound")
                if cleanup:writer.check(len(line.encode()),cleanup=True);private.write(line)
                else:current_writer.append_text(private,line)
                rows.append(row)
                if verdict["outcome"] in ("UNKNOWN","FAILED"):raise RuntimeError("Reference physical evidence: "+verdict["reason"])
            measure(None,"baseline",source_index=0)
            writer.write_json(a.output/"PRIVATE_initial_contacts.json",reader.baseline_receipt)
            # Reserve the COMPLETE segment evidence before its first action.
            # Three remaining full snapshots + bounded ledger + sparse raw RGB.
            reserve=3*capture_upper_bound(before[-1]["array_layout"])+(len(segment)+13)*20480+((len(segment)+29)//30)*1024**2+512*1024
            if shutil.disk_usage(a.output).free<80*1024**3+reserve:raise RuntimeError("Insufficient disk for complete reference evidence")
            try:
                with writer.transaction(reserve) as reservation:
                    current_writer=reservation
                    for i,command in enumerate(segment):
                        actual=step(command,"expert_segment");measure(actual,"expert_segment",source_index=i+1)
                        if terminal:raise RuntimeError("Reference skill terminated before complete evidence")
                        if (i+1)%30==0:
                            # Render-only sparse three-view RGB for independent
                            # review. No private geometry is painted onto pixels.
                            q_before=state()
                            images,_,_=onboard.read(model,render=og.sim.render)
                            q_after=state()
                            if not np.array_equal(q_before.q,q_after.q) or not np.array_equal(q_before.gripper,q_after.gripper):
                                raise RuntimeError("Render-only reference capture changed state")
                            folder=a.output/f"rgb_{completed:06d}";folder.mkdir()
                            import io
                            for view in ("head","left_wrist","right_wrist"):
                                image=Image.fromarray(images[view+"_rgb"].transpose(1,2,0));image.thumbnail((256,256))
                                b=io.BytesIO();image.save(b,format="PNG");reservation.write_bytes(folder/(view+".png"),b.getvalue())
                            captures.append({"control":completed,"files_sha256":{str(p.relative_to(a.output)):sha(p) for p in folder.iterdir()}})
                    capture("segment_end")
                    fixed=state().q.copy()
                    for _ in range(12):
                        actual=step(native_action(fixed,grips),"stability_tail");measure(actual,"stability_tail")
                        if terminal:raise RuntimeError("Reference terminated during stability tail")
                    capture("after_stability")
            finally:current_writer=writer
        except BaseException as exc:error(exc)
        finally:
            if kin is not None and grips is not None and not terminal and issued<budget["max_controls"]:
                try:
                    # Standalone cleanup: calibration/trace creation may have
                    # failed before the ordinary step closure even exists.
                    actual=np.asarray(native_action(kin.state().q,grips),dtype=np.float32)
                    try:writer.write_json(a.output/"final_hold_attempt.json",{"issued_control":issued+1,"actual_action23":actual.tolist()},cleanup=True)
                    except BaseException as exc:error(exc)
                    issued+=1
                    with og.sim.render_on_step(True):obs,_,terminated,truncated,_=env.step(actual,n_render_iterations=1)
                    completed+=1;terminal=bool(terminated or truncated)
                    final_hold=True
                    try:writer.write_json(a.output/"final_hold.json",{"completed":True,"issued":issued,"completed_controls":completed,"actual_action23":actual.tolist()},cleanup=True)
                    except BaseException as exc:error(exc)
                    if reader is not None:measure(actual,"final_hold",cleanup=True)
                    if primary is None:
                        e=session.evaluator;e.obs=e._preprocess_obs(e._sync_lights_and_get_obs(obs))
                        capture("after_final_hold")
                except BaseException as exc:error(exc)
            if reader is not None:
                try:reader.close()
                except BaseException as exc:error(exc)
            for stream in (trace,private):
                if stream is not None:
                    try:stream.close()
                    except BaseException as exc:error(exc)
        if primary is not None:raise primary
        if not final_hold or completed!=budget["max_controls"]:raise RuntimeError("Incomplete reference final control")
        measured=extract_seed(spec,rows)
        writer.write_json(a.output/"captures.json",captures)
        result={"status":"REFERENCE_LOCAL_SUCCEEDED","identity":identity,"issued_controls":issued,"completed_controls":completed,
                "final_hold_completed":final_hold,"failure":None,"model_calls":0,"official_task_success_claim":False,
                "reference_preparation_sha256":binding["reference_preparation_sha256"]}
        writer.write_json(a.output/"result.json",result)
        writer.write_json(a.output/"QUARANTINED_pose_seed.json",{"schema":"h09u-measured-reference-seed-v1","identity":identity,
                **measured,"training_eligible":False,"reference_preparation_sha256":binding["reference_preparation_sha256"],
                "evidence_sha256":{name:sha(a.output/name) for name in ("PRIVATE_reference_trace.jsonl","result.json","captures.json")}})


if __name__=="__main__":main()
