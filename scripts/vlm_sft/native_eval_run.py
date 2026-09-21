"""One of SIX separately released, fixed-start physical comparisons.

The actor sees only the H09X projection. Physical truth is recorded during
execution and scored only after the policy loop AND final hold have ended.
No teacher, pose seed, oracle early-stop, restore or automatic retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request,urlopen

if __name__=="__main__" and sys.platform.startswith("linux"):
    native=Path(sys.prefix)/f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/pymeshlab/lib"
    if (native/"libGLU.so.1").exists() and str(native) not in os.environ.get("LD_LIBRARY_PATH","").split(":"):
        env=dict(os.environ);env["LD_LIBRARY_PATH"]=str(native)+":"+env.get("LD_LIBRARY_PATH","")
        os.execve(sys.executable,[sys.executable,*sys.argv],env)

import numpy as np
from common import CAMERAS,sha,write_json
from native_actor_protocol import VERSION,actor_input,runtime_proprio
from native_dataset import load_dataset
from native_evaluation import VARIANTS,NearestNeighbor,PublicExecution,choose
from native_eval_prepare import SCHEMA,STARTS,ROOT,verify
from native_train import check_storage
from native_storage import activate as activate_storage
from native_teacher_collect import (REPO,ADAPTER,CalibratedRobot,OnboardRGBD,native_action,
    capture_snapshot,implementation_digest,preserved_session,rest_screen)
from native_teacher_artifacts import ArtifactBudget,write_calibration,action_evidence_bound,json_bytes
from native_teacher_reference_contract import check_actual_joint_bounds
from native_execution import (authorization_profile,metadata as execution_metadata,require_dataset_profile,
    actor_protocol,TIMING_PROFILES,WORKSPACE_PROFILE,workspace_budget_ok)
from native_motion_codec import BODY_TOKENS

BUDGET={"resets":1,"max_decisions":12,"new_controls_including_final_hold":420,
    "seconds_after_reset":1200,"initialization_seconds":900,"run_MiB":384,"total_MiB":6144}


def require_evaluation(auth,code,executor,prepared,variant,output,data_sha):
    execution_profile=authorization_profile(auth)
    if execution_profile is not None and any(k in auth for k in ("seed_profile","capacity_profile","authorize_offline_teacher")):
        raise ValueError("Paired evaluation cannot mix private collection profiles")
    instance=prepared["source"][2]
    if (auth.get("schema")!=SCHEMA or auth.get("authorize_evaluation") is not True or
            auth.get("code_commit")!=code or auth.get("executor_digest")!=executor or
            auth.get("protocol")!=actor_protocol(execution_profile) or auth.get("dataset_sha256")!=data_sha or
            auth.get("source")!=prepared["source"] or auth.get("variant")!=variant or variant not in VARIANTS or
            auth.get("physical_gpu")!=3 or type(auth.get("physical_gpu")) is not int or
            auth.get("specified_hand") is not None or auth.get("oracle_actor_feedback") is not False or
            not isinstance(auth.get("reviewer"),str) or not auth["reviewer"] or
            Path(auth.get("experiment_root","")).resolve()!=ROOT or
            Path(output).resolve()!=ROOT/f"eval_t1_i{instance}_{variant}_v1"):
        raise ValueError("Separate exact six-slot evaluation authorization required")
    budget=auth.get("budget")
    if not isinstance(budget,dict) or set(budget)!=set(BUDGET) or any(type(budget[k]) is not int or budget[k]!=v for k,v in BUDGET.items()):
        raise ValueError("Exact bounded single-reset evaluation profile required")
    gates=[json.loads(Path(p).read_text()) for p in auth["engineering_gate_paths"]]
    if len(gates)!=2 or {g["task"] for g in gates}!={0,3} or any(
            g.get("gate_ok") is not True or g.get("robot_geometry_guards") is not True or
            g.get("gripper_completion_v1",False) is not (execution_profile is not None) or
            g.get("implementation_digest")!=executor for g in gates):
        raise ValueError("Matching reviewed robot execution gates required")


def load_service(auth,code,data_sha):
    from native_execution import require_same_pipeline,PRECLOSE_PROFILE
    path=ROOT/"service_v1/identity.json"
    if sha(path)!=auth.get("service_identity_sha256"):raise ValueError("Released service identity changed")
    identity=json.loads(path.read_text())
    require_same_pipeline(auth,identity)
    if (identity.get("code_commit")!=code or identity.get("protocol")!=actor_protocol(authorization_profile(auth)) or identity.get("dataset_sha256")!=data_sha or
            identity.get("physical_gpu")!=3 or identity.get("port")!=8919 or identity.get("max_calls")!=48):
        raise ValueError("Exact paired service/model/data identity required")
    with urlopen("http://127.0.0.1:8919/health",timeout=10) as response:health=json.load(response)
    if {k:v for k,v in health.items() if k!="calls"}!=identity or not 0<=health.get("calls",-1)<48:
        raise ValueError("Service health/identity/call budget changed")
    def remote(request):
        body=json.dumps(request,allow_nan=False).encode()
        with urlopen(Request("http://127.0.0.1:8919/motion",data=body,headers={"Content-Type":"application/json"}),timeout=60) as response:
            answer=json.load(response)
        if any(answer.get(k)!=identity[k] for k in ("adapter_sha256","dataset_sha256","code_commit","training_result_sha256")):
            raise ValueError("Per-call model/source/dataset identity drift")
        if authorization_profile(auth) in TIMING_PROFILES and (
                authorization_profile(answer)!=authorization_profile(auth) or
                answer.get("storage_profile")!=identity["storage"]["profile"]):
            raise ValueError("Per-call execution/storage profile drift")
        return answer
    return identity,remote


def main():
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--authorization",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variant",choices=VARIANTS,required=True);a=p.parse_args()
    code=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise ValueError("Clean immutable runtime required")
    auth=json.loads(a.authorization.read_text());prepared,prefix=verify(a.prepared,auth["preparation_sha256"])
    execution_profile=authorization_profile(auth)
    data_sha=sha(a.data/"dataset.json");require_evaluation(auth,code,implementation_digest(),prepared,a.variant,a.output,data_sha)
    rows,dataset=load_dataset(a.data);require_dataset_profile(dataset,execution_profile)
    from native_execution import require_pipeline_profile
    require_pipeline_profile(auth,dataset)
    nn=NearestNeighbor(rows) if a.variant=="proprio_history_nn" else None
    identity,remote=(load_service(auth,code,data_sha) if nn is None else (None,None))
    storage=activate_storage(auth,a.output);check_storage(storage);a.output.mkdir(parents=True,exist_ok=False)
    writer=ArtifactBudget(a.output,ROOT,384*1024**2,6144*1024**2);current_writer=writer
    writer.write_json(a.output/"manifest.json",{"code_commit":code,"protocol":auth["protocol"],"authorization":auth,
        "preparation":prepared,"dataset_sha256":data_sha,"service_identity":identity,"budget":BUDGET,
        "oracle_actor_feedback":False,"specified_hand":None,"paid_prefix_not_full_task_SR":True,
        "storage":None if storage is None else storage.check()})
    os.environ["OMNIGIBSON_GPU_ID"]="3";os.environ["BEHAVIOR_ACTION_STEPS"]="1"
    os.environ.setdefault("OMNIGIBSON_HEADLESS","1");os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    sys.path.insert(0,str(ADAPTER))
    from native_oracle_low_v1.official_factory import load_official_oracle_window,OfficialEvaluatorSession
    window=load_official_oracle_window(a.prepared/"window.json")
    if (window.task_name!="picking_up_trash" or window.official_mode!="train" or window.seed!=0 or
            window.instance_id!=prepared["source"][2] or not np.array_equal(window.frozen_window().prefix_actions,prefix)):
        raise ValueError("Factory did not load the exact heldout reset/prefix")
    verify(a.prepared,auth["preparation_sha256"])
    controls=prefix_count=issued_native=issued_prefix=0;terminal=False;first_error=None;started=None;kin=None
    grips=None;policy=None;reader=None;initial=None;final=None;frames=[];tokens={};history=[];decisions=[];last_info={}
    trace=issue_trace=private=None;final_hold=False;stop="DECISION_BUDGET";active_token=None;neural_requests=0
    def error(exc):
        nonlocal first_error
        first_error=first_error or exc
        try:
            failure={"error":repr(first_error),"variant":a.variant,"autonomous_local_policy":True,
                "neural_request_attempts":neural_requests,"completed_native":controls,"issued_native":issued_native,
                "completed_prefix":prefix_count,"issued_prefix":issued_prefix,
                "wall_seconds":None if started is None else time.monotonic()-started}
            if not (a.output/"failure.json").exists():writer.write_json(a.output/"failure.json",failure,cleanup=True)
            # __exit__ can fail AFTER the inner finally wrote a provisional
            # result. Such teardown failure must not leave COMPLETE visible.
            if (a.output/"result.json").exists():
                value=json.loads((a.output/"result.json").read_text());value.update(status="FAILED",error=repr(first_error))
                writer.check(len(json_bytes(value)),cleanup=True);write_json(a.output/"result.json",value)
        except BaseException as recording:
            print(f"Evaluation error {first_error!r}; evidence write error {recording!r}",file=sys.stderr,flush=True)
    initialization=time.monotonic()
    with preserved_session(lambda:OfficialEvaluatorSession(window,gpu=3),error,lambda:first_error) as session:
        import omnigibson as og
        try:
            if time.monotonic()-initialization>=900:raise TimeoutError("Initialization before reset")
            check_storage(storage)
            session.reset();started=time.monotonic();env=session.evaluator.env
            kin=CalibratedRobot(env.robots[0]);grips=np.clip(kin.state().gripper/.05*2-1,-1,1)
            if time.monotonic()-initialization>=900:raise TimeoutError("Initialization after reset")
            writer.write_json(a.output/"initialization.json",{"seconds":time.monotonic()-initialization,"cooperative_not_async":True})
            model=kin.calibrate(grounded=True);write_calibration(writer,model);onboard=OnboardRGBD(env)
            if any(e["position_m"]>.003 or e["angle_rad"]>.02 or e.get("jacobian_max_abs",0)>.04 for e in kin.compare(model).values()):
                raise RuntimeError("FK gate failed")
            def state():
                s=kin.state();return model.state(s.q,s.gripper,s.base_velocity)
            trace=(a.output/"native_trace.jsonl").open("x",buffering=1)
            issue_trace=(a.output/"issued_trace.jsonl").open("x",buffering=1)
            private=(a.output/"PRIVATE_measurements.jsonl").open("x",buffering=1)
            def measure():
                # Recording only: no oracle, held/contact truth branch or
                # success condition can affect action selection or stopping.
                f=reader.read(prefix_count+controls)
                f["finger_opening"]=dict(zip(("left","right"),state().gripper.tolist()))
                line=json.dumps(f,allow_nan=False)+"\n"
                if len(line.encode())>16384:raise RuntimeError("Physical evidence telemetry bound exceeded")
                current_writer.append_text(private,line);frames.append(f)
                if active_token is not None:tokens[f["tick"]]=active_token
                return f
            def step(command,phase):
                nonlocal controls,prefix_count,issued_native,issued_prefix,grips,terminal,last_info
                if terminal or (phase!="prefix" and issued_native>=419):raise RuntimeError("Terminal/control limit; final hold reserved")
                if time.monotonic()-started>=1200:raise TimeoutError("Reset-to-end budget exhausted")
                check_storage(storage);check_actual_joint_bounds(state(),model)
                command=np.asarray(command,np.float32)
                if command.shape!=(23,) or not np.isfinite(command).all():raise ValueError("Finite native23 action required")
                line=json.dumps({"phase":phase,"prefix_control":prefix_count+int(phase=="prefix"),
                    "native_control":controls+int(phase!="prefix"),"action23":command.tolist()},allow_nan=False)+"\n"
                current_writer.check(2*len(line.encode())+(16384 if reader else 0))
                current_writer.append_text(issue_trace,line);grips=command[[14,22]].copy()
                if policy is not None:policy.issued(command)
                if phase=="prefix":issued_prefix+=1
                else:issued_native+=1
                with og.sim.render_on_step(True):obs,_,terminated,truncated,last_info=env.step(command,n_render_iterations=1)
                if phase=="prefix":prefix_count+=1
                else:controls+=1
                terminal=bool(terminated or truncated);current_writer.append_text(trace,line)
                e=session.evaluator;e.obs=e._preprocess_obs(e._sync_lights_and_get_obs(obs))
                if reader is not None:measure()
                if terminal:raise RuntimeError("Official episode terminated; no reset/retry")
            def settle():
                fixed=state().q.copy();recent=[]
                for _ in range(12):step(native_action(fixed,grips),"settle");recent.append(state())
                if not rest_screen(recent):raise RuntimeError("Robot did not pass fixed 12-control rest screen")
            def capture(folder,layout=None):
                return capture_snapshot(folder,model,state,onboard,kin.native_self_boxes,og.sim.render,
                    {"prefix_control":prefix_count,"native_control":controls},a.output,current_writer,expected_layout=layout)
            def preflight_at_execution(token,before,depths,geometry,receipt):
                current=state()
                if not np.array_equal(current.q,before.q) or not np.array_equal(current.gripper,before.gripper):
                    raise RuntimeError("Robot changed while the policy was deciding; reject stale token")
                return policy.preflight(token,current,model,depths,geometry,capture_receipt=receipt,
                    expected_clock={"prefix_control":prefix_count,"native_control":controls})
            for command in prefix:step(command,"prefix")
            # Qualification begins at the pause with actual last issued grips;
            # it is not an assertion that an open hand cannot contact anything.
            policy=PublicExecution(grips,execution_profile=execution_profile);settle()
            from native_teacher_og import PrivilegedReader
            spec=json.loads((a.prepared/"PRIVATE_scoring_spec.json").read_text())
            reader=PrivilegedReader(env,spec);initial=reader.read(prefix_count+controls)
            initial["finger_opening"]=dict(zip(("left","right"),state().gripper.tolist()))
            writer.write_json(a.output/"PRIVATE_initial.json",{"spec":spec,"frame":initial,"baseline_contacts":reader.baseline_receipt})
            for index in range(12):
                if time.monotonic()-started>=1200:raise TimeoutError("No new call after deadline")
                folder=a.output/f"decision_{index:02d}";folder.mkdir()
                before,hashes,depths,geometry,receipt=capture(folder/"before")
                clock={"prefix_control":prefix_count,"native_control":controls}
                actor=actor_input(session.observation()["task"],prepared["active_instruction"],
                    runtime_proprio(model,before,clock=receipt["clock"],expected_clock=clock,calibration_sha256=model.sha),hashes,history[-5:],
                    protocol=auth["protocol"])
                raw={v:(folder/"before"/(v+".png")).read_bytes() for v in CAMERAS}
                if nn is None:neural_requests+=1
                answer=choose(a.variant,actor,raw,nn=nn,remote=remote);token=answer["prediction"]
                writer.write_json(folder/"request.json",{"actor":actor,"response":answer,"capture_sha256":sha(folder/"before/capture.json")})
                decision={"index":index,"token":token,"control_start":controls,"response":answer,
                    **execution_metadata(execution_profile)};decisions.append(decision)
                try:
                    servo,preflight=preflight_at_execution(token,before,depths,geometry,receipt)
                except RuntimeError as exc:
                    decision.update(status="REJECTED_NO_MOTION",reason=str(exc));stop="PUBLIC_PREFLIGHT_STOP"
                    writer.write_json(folder/"execution.json",decision);break
                if controls+servo.total_ticks+12>419 or not workspace_budget_ok(token,servo.total_ticks,420-controls,12-index):
                    decision.update(status="REJECTED_NO_MOTION",reason="Whole macro/settle/final-hold budget")
                    stop="CONTROL_BUDGET";writer.write_json(folder/"execution.json",decision);break
                try:
                    with writer.transaction(action_evidence_bound(receipt["array_layout"])) as reservation:
                        current_writer=reservation;active_token=token;reservation.write_json(folder/"preflight.json",preflight)
                        while not servo.done and servo.ticks<servo.total_ticks:step(servo.next_action(state()),"candidate")
                        feedback=servo.finish(state());active_token=None
                        decision.update(control_end=controls,status=feedback["status"],feedback=feedback)
                        reservation.write_json(folder/"execution.json",decision);capture(folder/"after",receipt["array_layout"])
                        if not policy.completed(token,feedback):raise RuntimeError("Actual servo execution did not complete")
                        history.append(token);settle();capture(folder/"after_settle",receipt["array_layout"])
                finally:current_writer=writer;active_token=None
                writer.write_json(folder/"completed.json",{"clock":{"prefix_control":prefix_count,"native_control":controls},"history":history[-5:]})
                print(json.dumps({"decision":index,"token":token,"controls":controls,"variant":a.variant}),flush=True)
        except BaseException as exc:error(exc);stop="ERROR"
        finally:
            if kin is not None and grips is not None and not terminal and issued_native<420:
                try:
                    command=np.asarray(native_action(kin.state().q,grips),np.float32)
                    writer.write_json(a.output/"final_hold_attempt.json",{"issued_native":issued_native+1,"action23":command.tolist()},cleanup=True)
                    issued_native+=1
                    if policy is not None:policy.issued(command)
                    with og.sim.render_on_step(True):obs,_,terminated,truncated,last_info=env.step(command,n_render_iterations=1)
                    controls+=1;terminal=bool(terminated or truncated);final_hold=True
                    writer.write_json(a.output/"final_hold.json",{"completed":True,"prefix_controls":prefix_count,
                        "native_controls":controls,"action23":command.tolist()},cleanup=True)
                    if reader is not None:
                        final=reader.read(prefix_count+controls);final["finger_opening"]=dict(zip(("left","right"),kin.state().gripper.tolist()))
                        writer.write_json(a.output/"PRIVATE_final_hold.json",final,cleanup=True)
                except BaseException as exc:error(exc)
            if reader is not None:
                try:reader.close()
                except BaseException as exc:error(exc)
            for stream in (trace,issue_trace,private):
                if stream is not None:stream.close()
            # Deliberately AFTER the decision loop and last physical action.
            score={"any_hand_local_success":False,"reason":"INCOMPLETE_PHYSICAL_EVIDENCE","posthoc_only":True}
            if initial is not None and final is not None and issued_native==controls and issued_prefix==prefix_count:
                try:
                    from native_eval_outcomes import score_grasp
                    score=score_grasp(spec,initial,frames,tokens,final,specified_hand=None)
                except BaseException as exc:error(exc)
            done=last_info.get("done",{}) if isinstance(last_info,dict) else {};official=done.get("success") if isinstance(done,dict) else None
            writer.write_json(a.output/"result.json",{"status":"COMPLETE" if first_error is None else "FAILED",
                "error":None if first_error is None else repr(first_error),"variant":a.variant,"instance":prepared["source"][2],
                "stratum":prepared["stratum"],"stop":stop,"decisions":decisions,"prefix_controls":prefix_count,
                "native_controls":controls,"issued_prefix":issued_prefix,"issued_native":issued_native,
                "neural_request_attempts":neural_requests,"confirmed_service_calls":[d["response"]["call"] for d in decisions if nn is None],
                "request_without_confirmed_response":nn is None and neural_requests!=len(decisions),
                "final_hold":final_hold,"score":score,"manifest_sha256":sha(a.output/"manifest.json"),
                "official_success":bool(official) if isinstance(official,(bool,np.bool_)) else None,
                "wall_seconds_after_reset":None if started is None else time.monotonic()-started,
                "fixed_local_skill_only":True,"full_task_success_rate_claim":False},cleanup=True)
    if first_error is not None:raise first_error


if __name__=="__main__":main()
