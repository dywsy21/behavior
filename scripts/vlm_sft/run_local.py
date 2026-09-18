"""Matched, bounded fixed-skill pilot; not the H-08 dynamic grounded policy.

The policy receives only current RGB, robot proprioception, a fixed legitimate
task instruction and executed symbols. Simulator truth is written separately
after actions and is never used to choose actions, targets or local stopping.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import Request,urlopen

if __name__=="__main__" and sys.platform.startswith("linux"):
    native=Path(sys.prefix)/f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/pymeshlab/lib"
    entries=os.environ.get("LD_LIBRARY_PATH","").split(":")
    if (native/"libGLU.so.1").exists() and str(native) not in entries:
        env=dict(os.environ);env["LD_LIBRARY_PATH"]=":".join([str(native),*[x for x in entries if x]])
        os.execve(sys.executable,[sys.executable,*sys.argv],env)

import numpy as np
from PIL import Image,ImageDraw

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/"src"),str(REPO/"scripts/semantic_robot")]
from common import VERSION,CAMERAS,TOKENS,sha,write_json,token_to_action
from live import ACTIVE,request_payload,runtime_proprio
from run_sim import ADAPTER,WINDOWS,WINDOW_NAMES,WINDOW_SHAS,ROBOT_SHA
from semantic_robot.v2.og_calibration import CalibratedRobot
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.onboard import OnboardRGBD
from semantic_robot.v2.grounding import observed_cloud,LocalDepthGuard
from semantic_robot.v2.diagnostics import grasp_audit
from semantic_robot.og_backend import array

GATE=("HOLD","RIGHT_UP","RIGHT_DOWN","LEFT_FORWARD","LEFT_BACK","RIGHT_YAW_PLUS","RIGHT_YAW_MINUS",
      "LEFT_ROLL_PLUS","LEFT_ROLL_MINUS","BASE_BACK","BASE_FORWARD","BASE_YAW_PLUS","BASE_YAW_MINUS",
      "BOTH_UP","BOTH_DOWN","RIGHT_CLOSE","RIGHT_OPEN","LEFT_CLOSE","LEFT_OPEN","HOLD")


def implementation_digest():
    paths=sorted((REPO/"src/semantic_robot/v2").glob("*.py"))+[Path(__file__),Path(__file__).with_name("common.py"),Path(__file__).with_name("live.py")]
    return hashlib.sha256(b"".join(str(p.relative_to(REPO)).encode()+p.read_bytes() for p in paths)).hexdigest()


def guard_release(token,latches):
    return not (token.endswith("_OPEN") and latches.get(token.split("_")[0].lower(),False))


def write_privileged_audit(path,robot):
    """Write-only diagnostic boundary: returns nothing to the actor/loop state."""
    position,quaternion=(array(v).tolist() for v in robot.get_position_orientation())
    write_json(path,{**grasp_audit(robot),"robot_world_position_m":position,"robot_world_quaternion_xyzw":quaternion,
        "world_pose_for_actual_motion_audit_only":True})


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--task",type=int,choices=(0,3),required=True);p.add_argument("--mode",choices=("gate","agent"),required=True)
    p.add_argument("--variant",choices=("base","finetuned"),default="base");p.add_argument("--adapter-sha")
    p.add_argument("--gate-result",type=Path,action="append",default=[])
    args=p.parse_args();out=args.output;digest=implementation_digest()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise RuntimeError("Pinned clean source required")
    if shutil.disk_usage(out.parent).free<80*1024**3:raise RuntimeError("Disk reserve")
    identity=None
    if args.mode=="agent":
        gates=[json.loads(x.read_text()) for x in args.gate_result]
        if len(gates)!=2 or {g["task"] for g in gates}!={0,3} or not all(g["gate_ok"] and g["implementation_digest"]==digest for g in gates):raise ValueError("Both identical-code engineering gates required")
        with urlopen("http://127.0.0.1:8918",timeout=15) as r:identity=json.load(r)
        if identity["protocol"]!=VERSION or identity["adapter_sha256"]!=args.adapter_sha or identity["tokens"]!=list(TOKENS):raise ValueError("Model/codec identity mismatch")
    out.mkdir(exist_ok=False)
    os.environ["OMNIGIBSON_GPU_ID"]="1";os.environ["BEHAVIOR_ACTION_STEPS"]="1"
    os.environ.setdefault("OMNIGIBSON_HEADLESS","1");os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS","1")
    sys.path.insert(0,str(ADAPTER))
    from native_oracle_low_v1.official_factory import load_official_oracle_window,OfficialEvaluatorSession
    import imageio.v2 as imageio
    window_path=WINDOWS/WINDOW_NAMES[args.task]/"window.json"
    if args.task in WINDOW_SHAS and sha(window_path)!=WINDOW_SHAS[args.task]:raise ValueError("Frozen window changed")
    window=load_official_oracle_window(window_path)
    if window.official_mode!="train" or window.seed!=0 or window.robot_config_sha256!=ROBOT_SHA:raise ValueError("Frozen evaluator identity")
    if window.instance_id!={0:138,3:242}[args.task]:raise ValueError("Wrong pre-registered instance")
    prefix_limit=448 if args.task==0 else 0
    manifest={"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
        "implementation_digest":digest,"protocol":VERSION,"task":args.task,"instance":window.instance_id,
        "environment_seed":0,"policy_seed":41,"mode":args.mode,"variant":args.variant,"model_identity":identity,
        "window_sha256":sha(window_path),"robot_sha256":ROBOT_SHA,"prefix_controls_registered":prefix_limit,
        "active_instruction":ACTIVE[args.task],"active_instruction_source":"fixed task-language local skill, never oracle skill labels",
        "budgets":{"decisions":40 if args.mode=="agent" else len(GATE),"new_controls":1280 if args.mode=="agent" else 768,"seconds_after_prefix":1200},
        "harness":"H09 fixed local skill + existing SafeServo and local base-depth veto; NOT H08 GroundedController",
        "evaluator":"v3.9.1 development, not official v3.9.2 submission","actor_scene_truth":False,
        "nonbase_environment_collision_certified":False,"grasp_truth_is_diagnostic_only":True}
    write_json(out/"manifest.json",manifest)
    controls=0;prefix_count=0;terminal=False;info={};decisions=[];checks=[];video=None;servo=None;environment=None
    trace=(out/"steps.jsonl").open("x",buffering=1);phase="initialization"
    limits=manifest["budgets"];stop="DECISION_BUDGET";history=[];latches={"left":False,"right":False}
    with OfficialEvaluatorSession(window,gpu=1) as environment:
        try:
            import omnigibson as og
            phase="reset";environment.reset();env=environment.evaluator.env
            onboard=OnboardRGBD(env);kin=CalibratedRobot(env.robots[0]);model=kin.calibrate(grounded=True)
            write_json(out/"robot_calibration.json",model.spec)
            if max(model.finite_difference_error(model.reference).values())>1e-5:raise RuntimeError("FK Jacobian gate")
            def state_now():
                n=kin.state();return model.state(n.q,n.gripper,n.base_velocity)
            def check_fk(label):
                value={"label":label,"errors":kin.compare(model)};checks.append(value)
                if any(e["position_m"]>.003 or e["angle_rad"]>.02 or e.get("jacobian_max_abs",0)>.04 for e in value["errors"].values()):raise RuntimeError("Robot-only FK drift")
                write_json(out/"fk_checks.json",checks)
            def step(action,render=True):
                nonlocal info,terminal
                with og.sim.render_on_step(render):obs,_,terminated,truncated,info=env.step(action,n_render_iterations=1)
                e=environment.evaluator;e.obs=e._preprocess_obs(e._sync_lights_and_get_obs(obs))
                terminal=bool(terminated or truncated)
            def observe():
                q=kin.state().q.copy();rgb,depth,receipt=onboard.read(model,render=og.sim.render)
                if np.max(np.abs(q-kin.state().q))>1e-5 or receipt["head"]["valid_fraction"]<.1:raise RuntimeError("Fresh read-only RGB-D barrier")
                return {v:Image.fromarray(rgb[v+"_rgb"].transpose(1,2,0)) for v in CAMERAS},depth,receipt
            check_fk("reset");phase="expert_prefix"
            prefix=window.frozen_window().prefix_actions[:prefix_limit]
            for i,a in enumerate(prefix):
                step(a,render=(i%16==15 or i==len(prefix)-1));prefix_count+=1
                if terminal:raise RuntimeError("Terminal inside separately counted expert prefix")
                if i%64==63:check_fk(f"prefix_{i+1}")
            state=state_now();check_fk("after_prefix")
            write_privileged_audit(out/"PRIVILEGED_INITIAL_AUDIT.json",env.robots[0])
            servo=SafeServo(model,state,gripper_command=np.asarray(prefix[-1])[[14,22]] if len(prefix) else None)
            video=imageio.get_writer(str(out/"rollout.mp4"),fps=15,codec="libx264",quality=7,macro_block_size=2)
            started=time.monotonic();phase="policy";failures=[]
            def capture(label):
                images=environment.observation()["images"];canvas=Image.new("RGB",(640,1040),(15,20,28))
                canvas.paste(Image.fromarray(images["head_rgb"].transpose(1,2,0)).resize((640,640)),(0,0))
                for x,v in ((0,"left_wrist"),(320,"right_wrist")):canvas.paste(Image.fromarray(images[v+"_rgb"].transpose(1,2,0)).resize((320,320)),(x,640))
                draw=ImageDraw.Draw(canvas);draw.text((8,970),f"H09 {args.variant} task {args.task} controls {controls} | {label}",fill="white")
                draw.text((8,994),f"FIXED LOCAL SKILL; {prefix_count} expert prefix controls excluded",fill="white")
                draw.text((8,1015),"Command/holding diagnostic is not official task success; inference wait omitted",fill="white")
                video.append_data(np.asarray(canvas))
            for decision in range(limits["decisions"]):
                if terminal:stop="OFFICIAL_EPISODE_TERMINATED";break
                if controls>=limits["new_controls"]-1:stop="CONTROL_BUDGET";break
                if time.monotonic()-started>=1200:stop="WALL_TIME_BUDGET";break
                if shutil.disk_usage(out).free<80*1024**3:stop="DISK_RESERVE";break
                state=state_now();images,depths,sensor_receipt=observe()
                folder=out/f"decision_{decision:03d}";folder.mkdir()
                for view,img in images.items():img.save(folder/(view+".png"))
                write_json(folder/"sensor_receipt.json",sensor_receipt)
                proprio=runtime_proprio(state);write_json(folder/"proprio.json",proprio)
                if args.mode=="gate":response={"prediction":GATE[decision],"model_called":False}
                else:
                    payload=request_payload(environment.observation()["task"],proprio,history,images,args.variant,args.task)
                    write_json(folder/"request_without_images.json",{k:v for k,v in payload.items() if k!="images"})
                    data=json.dumps(payload).encode()
                    with urlopen(Request("http://127.0.0.1:8918/motion",data=data,headers={"Content-Type":"application/json"}),timeout=60) as r:response=json.load(r)
                    if response["adapter_sha256"]!=args.adapter_sha or response["protocol"]!=VERSION or response["variant"]!=args.variant:raise RuntimeError("Per-call identity drift")
                token=response["prediction"];action=token_to_action(token)
                row={"decision":decision,"control_start":controls,"token":token,"action":json.loads(action.text()),"response":response}
                guard=LocalDepthGuard(observed_cloud(depths,model,state.q),model,state.q,depths)
                carry=any(latches.values()) and args.mode=="agent"
                allowed,reason=guard.check(action,carry)
                if args.mode=="agent" and not guard_release(token,latches):allowed,reason=False,"CLOSE_LATCH_PRESERVED_NO_RELEASE_IN_FIXED_SKILL"
                row["preflight"]={"accepted":bool(allowed),"reason":reason,"carry_precaution_not_hold_truth":carry}
                accepted=servo.begin(action,state_now(),carry=carry) if allowed else False
                row["servo_accepted"]=bool(accepted)
                if accepted:
                    while not servo.done and servo.ticks<servo.total_ticks and controls<limits["new_controls"]-1:
                        command=servo.next_action(state_now());step(command,render=(controls%2==1 or servo.ticks==servo.total_ticks or servo.done));controls+=1
                        trace.write(json.dumps({"control":controls,"decision":decision,"action23":command.tolist()})+"\n")
                        if controls%2==0:capture(token)
                        if terminal or time.monotonic()-started>=1200:break
                    feedback=servo.finish(state_now());row["feedback"]=feedback
                    if servo.ticks>0:
                        history.append(token)
                        if token.endswith("_CLOSE"):latches[token.split("_")[0].lower()]=True
                        if token.endswith("_OPEN"):latches[token.split("_")[0].lower()]=False
                    if feedback["status"]!="TARGET_REACHED":stop="EXECUTION_SAFETY_STOP";failures.append(row)
                else:row["feedback"]={"status":"REJECTED_NO_MOTION","reason":reason if not allowed else servo.status}
                # No privileged value from this audit returns to policy/latch/history.
                write_privileged_audit(folder/"PRIVILEGED_POST_ACTION_GRASP_AUDIT.json",env.robots[0])
                check_fk(f"decision_{decision}")
                row["control_end"]=controls;decisions.append(row);trace.write(json.dumps(row)+"\n")
                write_json(out/"progress.json",{"controls":controls,"decisions":len(decisions),"last":row})
                print(json.dumps(row),flush=True)
                if failures:break
                if args.mode=="agent" and len(decisions)>=3 and all(r["feedback"]["status"]=="REJECTED_NO_MOTION" for r in decisions[-3:]):stop="THREE_CONSECUTIVE_REJECTIONS";break
            if not terminal and controls<limits["new_controls"]:
                step(servo.safe_hold(state_now()));controls+=1
                trace.write(json.dumps({"control":controls,"safety_stop":True})+"\n")
            reached={r["token"] for r in decisions if r["feedback"]["status"]=="TARGET_REACHED"}
            required={"BASE_YAW_PLUS","BASE_YAW_MINUS","RIGHT_CLOSE","RIGHT_OPEN","LEFT_CLOSE","LEFT_OPEN","HOLD"}
            gate_ok=args.mode=="gate" and not failures and len(decisions)==len(GATE) and len(reached)>=12 and required<=reached
            done=info.get("done",{}) if isinstance(info,dict) else {}
            result={"status":"complete","task":args.task,"variant":args.variant,"gate_ok":bool(gate_ok),"implementation_digest":digest,
                "controls":controls,"prefix_controls":prefix_count,"decisions":decisions,"stop_reason":stop,"wall_s_after_prefix":time.monotonic()-started,
                "official_success":bool(done.get("success",False)),"official_goal_status_diagnostic":done.get("goal_status",{}),
                "fixed_local_skill_only":True,"full_task_success_rate_claim":False,"terminal":terminal}
            write_json(out/"result.json",result)
        except BaseException as exc:
            # Safe stop is best effort and does not replace the original error.
            safe_stop_error=None
            if servo is not None and not terminal and controls<limits["new_controls"]:
                try:
                    step(servo.safe_hold(state_now()));controls+=1
                    trace.write(json.dumps({"control":controls,"exception_safety_stop":True})+"\n")
                except Exception as stop_exc:safe_stop_error=repr(stop_exc)
            write_json(out/"failure.json",{"error":repr(exc),"phase":phase,"controls":controls,"prefix_controls":prefix_count,
                "decisions":decisions,"best_effort_safe_stop_error":safe_stop_error})
            raise
        finally:
            trace.close()
            if video is not None:video.close()


if __name__=="__main__":main()
