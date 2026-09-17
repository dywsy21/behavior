"""Bounded R1Pro development test: controller gate or local VLM closed loop.

Uses preserved v3.9.1 evaluator initialization, unchanged R1Pro config and 23D
controls. Not an official v3.9.2 challenge submission or full-success benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
ADAPTER = Path("/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1")
WINDOWS = Path("/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/c1_windows_v2_matched")
WINDOW_NAMES = {0: "c1v2-matched-t0-train-e121-f448-grasp", 3: "c1v2-matched-t3-train-e762-f4221-grasp"}
WINDOW_SHAS = {0: "93731a793ed550bae225e148d872b97d42e331a709e2fd0cd33bd1ccf76ccbd8"}
ROBOT_SHA = "a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--task", type=int, choices=[0,3], default=0)
    p.add_argument("--mode", choices=["gate","agent"], default="gate")
    p.add_argument("--prefix", type=int, default=0)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--uri", default="http://127.0.0.1:8897")
    p.add_argument("--expected-revision", help="Required model identity for agent mode")
    p.add_argument("--max-decisions", type=int, default=48)
    p.add_argument("--max-controls", type=int, default=1536)
    p.add_argument("--max-seconds", type=int, default=1200)
    args=p.parse_args()
    if not (0 <= args.prefix <= 448 and 1 <= args.max_decisions <= 64 and 1 <= args.max_controls <= 1536 and 0 < args.max_seconds <= 1200):
        raise ValueError("Outside preregistered pilot budget")
    if args.task == 3 and args.prefix:
        raise ValueError("Task3 pilot starts at reset; no unregistered expert prefix")
    model_identity=None
    if args.mode=="agent":
        with urlopen(args.uri,timeout=10) as reply:
            model_identity=json.load(reply)
        if not args.expected_revision or model_identity.get("revision") != args.expected_revision:
            raise ValueError("Model service revision does not match registered checkpoint")
        if not model_identity.get("constrained_action_grammar"):
            raise ValueError("This pilot requires explicit constrained syntax")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Immutable clean worktree required")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    os.environ["OMNIGIBSON_GPU_ID"]=str(args.gpu)
    os.environ.setdefault("OMNIGIBSON_HEADLESS","1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS","1")
    os.environ["BEHAVIOR_ACTION_STEPS"]="1"
    sys.path.insert(0,str(ADAPTER))
    from native_oracle_low_v1.official_factory import load_official_oracle_window, OfficialEvaluatorSession
    from semantic_robot.actions import parse_action
    from semantic_robot.control import R1ProServo
    from semantic_robot.og_backend import OGKinematics
    from semantic_robot.prompts import user_text, PROMPT_VERSION
    from semantic_robot.client import request_action
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    path=WINDOWS/WINDOW_NAMES[args.task]/"window.json"
    if args.task in WINDOW_SHAS and sha(path) != WINDOW_SHAS[args.task]:
        raise ValueError("Preserved window changed")
    window=load_official_oracle_window(path)
    if window.official_mode != "train" or window.seed != 0 or window.robot_config_sha256 != ROBOT_SHA:
        raise ValueError("Frozen train/config boundary changed")
    receipt={"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
             "args":vars(args),"window_sha256":sha(path),"window":str(path),"robot_sha256":ROBOT_SHA,
             "instance":window.instance_id,"split":"train","environment_seed":0,"policy_seed":17,
             "task_name":window.task_name,"prompt_version":PROMPT_VERSION,"training_updates":0,
             "model_identity":model_identity,
             "evaluator_version":"v3.9.1-development-not-official-v3.9.2","started_unix":time.time(),
             "prefix_is_expert_not_agent":bool(args.prefix),"actor_has_privileged_scene_state":False,
             "runtime_factory_sha256":sha(ADAPTER/"native_oracle_low_v1/official_factory.py")}
    write(out/"manifest.json",receipt)
    controls=0; prefix_count=0; decisions=[]; terminal=False; info={}; video=None
    trace=(out/"steps.jsonl").open("x",buffering=1)
    try:
        with OfficialEvaluatorSession(window,gpu=args.gpu) as harness:
            import omnigibson as og
            harness.reset()
            env=harness.evaluator.env
            kin=OGKinematics(env.robots[0])
            write(out/"kinematics.json",kin.receipt())

            def step(action, render=True):
                nonlocal info, terminal
                with og.sim.render_on_step(render):
                    obs, reward, terminated, truncated, info = env.step(action, n_render_iterations=1)
                e=harness.evaluator
                e.obs=e._preprocess_obs(e._sync_lights_and_get_obs(obs))
                terminal=bool(terminated or truncated)
                return terminal

            prefix=window.frozen_window().prefix_actions[:args.prefix]
            for i,action in enumerate(prefix):
                if step(action, render=(i%16==15 or i==len(prefix)-1)):
                    raise RuntimeError("Environment terminated during expert prefix")
                prefix_count+=1
            state=kin.state()
            previous_grips=None if not prefix_count else np.asarray(prefix[-1])[[14,22]]
            servo=R1ProServo(state,gripper_command=previous_grips)
            video=imageio.get_writer(str(out/"rollout.mp4"),fps=15,codec="libx264",quality=7,macro_block_size=2)
            history=[]; feedback=None; started=time.perf_counter()
            # Independent-axis gate, not an expert task controller. No VLM calls.
            gate=["HOLD","R UP FINE","R DOWN FINE","L UP FINE","L DOWN FINE",
                  "R YAW_POS FINE","R YAW_NEG FINE","L ROLL_POS FINE","L ROLL_NEG FINE",
                  "TORSO UP FINE","TORSO DOWN FINE","BASE BACK FINE","BASE FWD FINE",
                  "BASE YAW_POS FINE","BASE YAW_NEG FINE","BOTH UP FINE","BOTH DOWN FINE",
                  "R CLOSE","R OPEN","L CLOSE","L OPEN","MODE CARRY","BOTH UP FINE","HOLD"]

            def capture(label):
                images=harness.observation()["images"]
                head=Image.fromarray(images["head_rgb"].transpose(1,2,0)).resize((640,640))
                canvas=Image.new("RGB",(640,1088),(15,20,28)); canvas.paste(head,(0,0))
                for x,name in ((0,"left_wrist_rgb"),(320,"right_wrist_rgb")):
                    wrist=Image.fromarray(images[name].transpose(1,2,0)).resize((320,320))
                    canvas.paste(wrist,(x,640))
                draw=ImageDraw.Draw(canvas)
                draw.text((12,645),"LEFT WRIST",fill="white"); draw.text((332,645),"RIGHT WRIST",fill="white")
                draw.text((12,974),f"R1Pro semantic agent | task {args.task} | {args.mode}",fill="white")
                draw.text((12,1004),f"Control {controls} | {label}",fill="white")
                draw.text((12,1034),f"LOCAL PILOT | {prefix_count} expert prefix controls not shown",fill="white")
                draw.text((12,1060),"Command is not verified task success",fill="white")
                video.append_data(np.asarray(canvas))

            for decision in range(args.max_decisions):
                if controls>=args.max_controls or time.perf_counter()-started>=args.max_seconds or terminal:
                    break
                if args.mode=="gate" and decision>=len(gate):
                    break
                state=kin.state(); obs=harness.observation()
                prompt=user_text(args.task,window.task_name,obs["task"],state,history,feedback)
                frame_dir=out/f"decision_{decision:03d}"; frame_dir.mkdir()
                for name,image in obs["images"].items():
                    Image.fromarray(image.transpose(1,2,0)).save(frame_dir/f"{name}.png")
                (frame_dir/"prompt.txt").write_text(prompt)
                write(frame_dir/"proprio.json",{"q":state.q.tolist(),"gripper":state.gripper.tolist(),
                    "poses":{k:[v.tolist() for v in pose] for k,pose in state.poses.items()}})
                response={"action":gate[decision],"model_called":False} if args.mode=="gate" else request_action(args.uri,obs["images"],prompt)
                command=response["action"]
                row={"decision":decision,"control_start":controls,"response":response,"command":command}
                try:
                    units=parse_action(command)
                    if command=="DONE":
                        row["stop_reason"]="MODEL_DONE_NOT_VERIFIED_SUCCESS"; decisions.append(row); break
                    servo.begin(units,state)
                except ValueError as exc:
                    row["rejected"]=str(exc)
                    feedback={"status":"REJECTED_NO_MOTION","error":str(exc)}
                    trace.write(json.dumps(row)+"\n"); decisions.append(row)
                    # Invalid response is NOT replaced with a fake successful action.
                    if sum("rejected" in r for r in decisions[-3:])==3:
                        break
                    continue
                start=time.perf_counter()
                while servo.ticks<servo.total_ticks and controls<args.max_controls:
                    action=servo.next_action(kin.state())
                    controls+=1
                    ended=step(action,render=(controls%2==0 or servo.ticks==servo.total_ticks))
                    trace.write(json.dumps({"control":controls,"decision":decision,"action23":action.tolist(),
                                            "terminal":ended})+"\n")
                    if controls%2==0: capture(command)
                    if ended or time.perf_counter()-started>=args.max_seconds: break
                feedback=servo.feedback(kin.state())
                row.update(feedback=feedback,control_end=controls,execution_wall_s=time.perf_counter()-start)
                trace.write(json.dumps(row)+"\n"); decisions.append(row); history.append(command)
                write(out/"progress.json",{"controls":controls,"decisions":len(decisions),"last":row})
                print(json.dumps(row),flush=True)
            # Evaluator-only result; never included in the next actor prompt.
            done_info=info.get("done",{}) if isinstance(info,dict) else {}
            result={"status":"complete","controls":controls,"prefix_controls":prefix_count,"decisions":decisions,
                    "terminal":terminal,"official_success":bool(done_info.get("success",False)),
                    "wall_s_after_prefix":time.perf_counter()-started,"full_task_success_rate_claim":False,
                    "final_done_info":repr(done_info),
                    "final_goal_status":done_info.get("goal_status",{})}
            write(out/"result.json",result)
    except BaseException as exc:
        write(out/"failure.json",{"error":repr(exc),"controls":controls,"prefix_controls":prefix_count})
        raise
    finally:
        trace.close()
        if video is not None: video.close()


if __name__=="__main__":
    main()
