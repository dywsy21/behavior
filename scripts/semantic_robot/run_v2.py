"""Bounded development gate / VLM rollout. v1 and active teammate code unchanged."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

# The preserved simulator environment supplies libGLU through pymeshlab, as do
# the project's reviewed launchers. The native loader reads LD_LIBRARY_PATH at
# process start: setting it after importing Kit is too late. Re-exec only this
# private process; never modify conda, shared libraries or a teammate's service.
if __name__ == "__main__" and sys.platform.startswith("linux"):
    native_lib = Path(sys.prefix)/f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/pymeshlab/lib"
    entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if (native_lib/"libGLU.so.1").exists() and str(native_lib) not in entries:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join([str(native_lib), *[x for x in entries if x]])
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"src"))
from run_sim import ADAPTER, WINDOWS, WINDOW_NAMES, WINDOW_SHAS, ROBOT_SHA, sha, write
from semantic_robot.v2.harness import TaskHarness
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.og_calibration import CalibratedRobot
from semantic_robot.v2.policy import VLMPolicy
from semantic_robot.v2.protocol import Action, HOLD
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import prepare_views
from semantic_robot.v2.onboard import OnboardRGBD
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.policy import GroundedPolicy


def implementation_digest():
    paths = sorted((REPO/"src/semantic_robot/v2").glob("*.py")) + [Path(__file__)]
    return hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()


def bounded_gate(grounded=False):
    actions = [HOLD, Action("right","up"), Action("right","down"),
            Action("left","forward","micro"), Action("left","back","micro"),
            Action("right","yaw_plus","micro","tool"), Action("right","yaw_minus","micro","tool"),
            Action("left","roll_plus","micro","tool"), Action("left","roll_minus","micro","tool"),
            Action("torso","up","micro"), Action("torso","down","micro"),
            Action("base","back"), Action("base","forward"),
            Action("base","yaw_plus","micro"), Action("base","yaw_minus","micro"),
            Action("right","left","micro","head"), Action("right","right","micro","head"),
            Action("both","up","micro"), Action("both","down","micro"),
            Action("right","close"),Action("right","open"),Action("left","close"),Action("left","open"),HOLD]
    if grounded:
        actions[13:15]=[Action("base","yaw_plus","coarse"),Action("base","yaw_minus","coarse")]
    return actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("gate","agent"), default="gate")
    p.add_argument("--harness", choices=("v2","grounded"), default="v2")
    p.add_argument("--task", type=int, choices=(0,3), default=0)
    p.add_argument("--prefix", type=int, default=0)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--uri", default="http://127.0.0.1:8907")
    p.add_argument("--expected-revision")
    p.add_argument("--gate-result", action="append", default=[])
    p.add_argument("--max-decisions", type=int, default=48)
    p.add_argument("--max-controls", type=int, default=1536)
    p.add_argument("--max-seconds", type=int, default=1200)
    args = p.parse_args()
    if not (0 <= args.prefix <= 448 and args.max_decisions in range(1,49) and 1 <= args.max_controls <= 1536 and 1 <= args.max_seconds <= 1200):
        raise ValueError("Registered pilot budget exceeded")
    if args.task == 3 and args.prefix:
        raise ValueError("Task3 has no registered expert prefix")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Immutable clean source required")
    digest = implementation_digest()
    grounded=args.harness=="grounded"
    if args.mode == "agent":
        if not args.expected_revision or len(args.gate_result) != 2:
            raise ValueError("Model identity and both task pose gates required")
        gates = [json.loads(Path(path).read_text()) for path in args.gate_result]
        if {g["task"] for g in gates} != {0,3} or not all(g["gate_ok"] and g["implementation_digest"] == digest and
                g.get("harness","v2")==args.harness for g in gates):
            raise ValueError("Control/FK gates have not passed for this exact implementation")
    out = Path(args.output); out.mkdir(parents=True,exist_ok=False)
    if grounded and shutil.disk_usage(out).free<80*1024**3:
        raise RuntimeError("H-07 disk reserve below 80GiB; do not start")
    os.environ["OMNIGIBSON_GPU_ID"] = str(args.gpu)
    os.environ.setdefault("OMNIGIBSON_HEADLESS","1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS","1")
    os.environ["BEHAVIOR_ACTION_STEPS"] = "1"
    sys.path.insert(0,str(ADAPTER))
    from native_oracle_low_v1.official_factory import load_official_oracle_window, OfficialEvaluatorSession
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    path = WINDOWS/WINDOW_NAMES[args.task]/"window.json"
    if args.task in WINDOW_SHAS and sha(path) != WINDOW_SHAS[args.task]:
        raise ValueError("Original development window drift")
    window = load_official_oracle_window(path)
    if window.official_mode != "train" or window.seed != 0 or window.robot_config_sha256 != ROBOT_SHA:
        raise ValueError("Frozen task/config identity mismatch")
    policy_class=GroundedPolicy if grounded else VLMPolicy
    policy = policy_class(args.uri,args.expected_revision,max_calls=1+2*args.max_decisions+(2 if grounded else 0)) if args.mode=="agent" else None
    manifest = {"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
                "implementation_digest":digest, "args":vars(args), "instance":window.instance_id,
                "task":args.task,"task_name":window.task_name,"split":"train","seed":0,
                "window_sha":sha(path),"robot_sha":ROBOT_SHA,"model_identity":policy.identity if policy else None,
                "training_updates":0,"evaluator":"v3.9.1-development-not-official-v3.9.2",
                "actor_scene_truth":False,"prefix_is_expert_not_agent":bool(args.prefix),
                "actor_modalities":["rgb","depth_linear","proprio"] if grounded else ["rgb","proprio"],
                "harness":args.harness,"max_strategy_replans":2 if grounded else 0,
                "native_library_path":os.environ.get("LD_LIBRARY_PATH", "")}
    write(out/"manifest.json",manifest)
    controls, prefix_count, terminal = 0,0,False
    info, decisions, checks, failures = {},[],[],[]
    sensor_checks=[]
    controller=None
    video, manager, servo, state = None,None,None,None
    trace = (out/"steps.jsonl").open("x",buffering=1)
    try:
        with OfficialEvaluatorSession(window,gpu=args.gpu) as environment:
            import omnigibson as og
            environment.reset()
            env = environment.evaluator.env
            onboard=OnboardRGBD(env) if grounded else None
            if onboard:
                # Attach the allowed depth annotator and finish its initial
                # render before reading buffers. No control step is hidden here.
                for _ in range(3): og.sim.render()
            kin = CalibratedRobot(env.robots[0])
            model = kin.calibrate(grounded=grounded)
            write(out/"robot_calibration.json",model.spec)
            write(out/"robot_metadata.json",kin.receipt())
            fd = model.finite_difference_error(model.reference)
            write(out/"fk_finite_difference.json",fd)
            if max(fd.values()) > 1e-5:
                raise RuntimeError("Analytical FK/Jacobian finite differences failed")

            def state_now():
                native = kin.state()
                return model.state(native.q,native.gripper,native.base_velocity)

            def observation_now(label):
                if onboard is None:
                    return environment.observation()["images"],{},{}
                images,depths,receipt=onboard.read(model)
                check={"label":label,"cameras":receipt}
                sensor_checks.append(check)
                write(out/"sensor_checks.json",sensor_checks)
                if receipt["head"]["valid_fraction"]<.1:
                    raise RuntimeError("Head depth not initialized/usable; refusing blind grounded run")
                return images,depths,receipt

            def check_fk(label):
                check = {"label":label,"errors":kin.compare(model)}
                checks.append(check)
                write(out/"fk_checks.json",checks)
                for name, errors in check["errors"].items():
                    if errors["position_m"] > .003 or errors["angle_rad"] > .02 or errors.get("jacobian_max_abs",0) > .04:
                        # OG shutdown may terminate before the outer except runs.
                        write(out/"failure.json",{"error":"FK_MISMATCH", "check":check,
                              "controls":controls,"prefix_controls":prefix_count,"decisions":decisions})
                        raise RuntimeError(f"Portable robot/camera FK mismatch {label}/{name}: {errors}")

            def step(action, render=True):
                nonlocal info,terminal
                with og.sim.render_on_step(render):
                    obs, _, terminated, truncated, info = env.step(action,n_render_iterations=1)
                evaluator = environment.evaluator
                evaluator.obs = evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
                terminal = bool(terminated or truncated)
                return terminal

            check_fk("reset")
            prefix = window.frozen_window().prefix_actions[:args.prefix]
            for i, action in enumerate(prefix):
                if step(action,render=(i%16==15 or i==len(prefix)-1)):
                    raise RuntimeError("Terminated in expert prefix")
                prefix_count += 1
                if i%64==63:
                    check_fk(f"prefix_{i+1}")
            check_fk("after_prefix")
            state = state_now()
            previous_grips = None if not prefix_count else np.asarray(prefix[-1])[[14,22]]
            servo = SafeServo(model,state,gripper_command=previous_grips)
            video = imageio.get_writer(str(out/"rollout.mp4"),fps=15,codec="libx264",quality=7,macro_block_size=2)
            previous = None
            started = time.perf_counter()
            first_images,first_depth,first_receipt=observation_now("after_prefix")
            first = prepare_views(first_images,model,state.q,grounded=grounded)
            if grounded: np.savez_compressed(out/"initial_depth.npz",**first_depth)
            for label,img in zip(first.labels,first.images): img.save(out/(label+".png"))

            def save_call(directory, name, call):
                payload = dict(call["request"])
                payload["images"] = [{"label":row["label"]} for row in payload["images"]]
                write(directory/(name+".json"),{"result":call["result"],"request_without_pixel_duplicates":payload})

            if policy:
                goals, call = policy.plan(args.task,environment.observation()["task"],first)
                save_call(out,"planner",call)
                manager = GroundedHarness(goals) if grounded else TaskHarness(goals)
                if grounded: controller=GroundedController(model,servo,manager)
                write(out/"plan.json",[asdict(g) for g in goals])

            def capture(label):
                images = environment.observation()["images"]
                canvas = Image.new("RGB",(640,1088),(15,20,28))
                canvas.paste(Image.fromarray(images["head_rgb"].transpose(1,2,0)).resize((640,640)),(0,0))
                for x,name in ((0,"left_wrist_rgb"),(320,"right_wrist_rgb")):
                    canvas.paste(Image.fromarray(images[name].transpose(1,2,0)).resize((320,320)),(x,640))
                draw = ImageDraw.Draw(canvas)
                draw.text((8,967),f"R1Pro v2 | task {args.task} | {args.mode} | controls {controls}",fill="white")
                draw.text((8,992),label[:98],fill="white")
                draw.text((8,1017),f"Stage: {manager.stage if manager else 'KINEMATIC GATE'} | {servo.status}",fill="white")
                draw.text((8,1042),"Observed/commanded is not official success. Inference wait omitted.",fill="white")
                draw.text((8,1065),f"Expert prefix {prefix_count} not shown. 30Hz sim / 15fps video.",fill="white")
                video.append_data(np.asarray(canvas))

            gate = bounded_gate(grounded)
            for decision in range(args.max_decisions):
                recoverable_stop=bool(controller and manager.stop_reason=="RECOVERY_BUDGET_EXHAUSTED" and manager.replans<2)
                if controls>=args.max_controls or time.perf_counter()-started>=args.max_seconds or terminal or (manager and manager.stop_reason and not recoverable_stop):
                    break
                if not policy and decision >= len(gate): break
                if grounded and shutil.disk_usage(out).free<80*1024**3:
                    if manager: manager.stop_reason="DISK_RESERVE_REACHED"
                    break
                state = state_now()
                images,depths,depth_receipt=observation_now(f"decision_{decision}")
                bundle = prepare_views(images,model,state.q,previous,grounded=grounded)
                directory = out/f"decision_{decision:03d}"; directory.mkdir()
                for label,img in zip(bundle.labels,bundle.images): img.save(directory/(label+".png"))
                write(directory/"proprio.json",{"q":state.q.tolist(),"gripper":state.gripper.tolist(),"geometry":bundle.geometry})
                if grounded:
                    np.savez_compressed(directory/"depth.npz",**depths)
                    write(directory/"depth_receipt.json",depth_receipt)
                row = {"decision":decision,"control_start":controls}
                if policy:
                    observation, call = policy.observe(manager,state,bundle)
                    save_call(directory,"observation",call)
                    if controller:
                        controller.observe(observation,state,depths,depth_receipt)
                        if controller.replan_needed:
                            recovery,call=policy.recover(manager,state,bundle,controller.replan_needed)
                            save_call(directory,"recovery",call)
                            controller.apply_recovery(recovery)
                        write(directory/"grounded_progress.json",controller.progress)
                    else:
                        manager.observe(observation,state,bundle.geometry)
                    write(directory/"harness.json",manager.context())
                    if manager.stop_reason:
                        row["stop_reason"] = manager.stop_reason; decisions.append(row); break
                    if controller and not observation.visible and manager.stage in ("SEARCH","RECOVER"):
                        action,selection=controller.search_action(state)
                        write(directory/"action_selection.json",selection)
                        row["selection_source"]=selection["source"]
                    elif controller:
                        allowed=controller.candidates(state)
                        write(directory/"candidates.json",manager.candidate_receipt)
                        if manager.stop_reason:
                            row["stop_reason"]=manager.stop_reason; decisions.append(row); break
                        action,call=policy.act_feasible(manager,state,bundle,allowed)
                        save_call(directory,"action",call)
                        row["selection_source"]="VLM_among_current_preflighted_actions"
                    else:
                        action, call = policy.act(manager,state,bundle)
                        save_call(directory,"action",call)
                else:
                    action = gate[decision]
                row["action"] = asdict(action)
                wall = time.perf_counter()
                state = state_now()  # recheck actual joints after model latency
                accepted = servo.begin(action,state,carry=bool(manager and manager.carry))
                row["accepted_before_motion"] = accepted
                previous = bundle.current_raw
                if accepted:
                    while not servo.done and servo.ticks<servo.total_ticks and controls<args.max_controls:
                        command = servo.next_action(state_now())
                        ended = step(command,render=(controls%2==1 or servo.ticks==servo.total_ticks or servo.done))
                        controls += 1
                        trace.write(json.dumps({"control":controls,"decision":decision,"action23":command.tolist(),"terminal":ended})+"\n")
                        if controls%2==0: capture(action.text())
                        if ended or time.perf_counter()-started>=args.max_seconds: break
                state = state_now()
                feedback = servo.finish(state)
                row.update(feedback=feedback,control_end=controls,wall_s=time.perf_counter()-wall)
                decisions.append(row); trace.write(json.dumps(row)+"\n")
                if controller: controller.executed(action,feedback)
                elif manager: manager.executed(action,feedback)
                check_fk(f"decision_{decision}")
                write(out/"progress.json",{"controls":controls,"decisions":len(decisions),"last":row})
                print(json.dumps(row),flush=True)
                if not policy and accepted and feedback["status"] not in ("TARGET_REACHED",):
                    failures.append(row); break  # diagnose, do not push further after a failed gate
            # Explicit zero base velocity / preserve grippers on EVERY exit path.
            if not terminal and servo is not None and controls < args.max_controls:
                step(servo.safe_hold(state_now())); controls += 1
                trace.write(json.dumps({"control":controls,"safety_stop":True})+"\n")
            done = info.get("done",{}) if isinstance(info,dict) else {}
            reached = [r for r in decisions if r.get("feedback",{}).get("status")=="TARGET_REACHED"]
            gate_ok = args.mode=="gate" and not failures and len(decisions)==len(gate) and len(reached)>=16
            if grounded and args.mode=="gate":
                required={13,14,19,20,21,22}
                gate_ok=bool(gate_ok and required <= {r["decision"] for r in reached} and sensor_checks and
                             "grasp_centers_eef" in model.spec["metadata"])
            stop_reason=manager.stop_reason if manager else "CONTROL_GATE_COMPLETE" if gate_ok else "CONTROL_GATE_FAILED"
            if policy and stop_reason is None:
                stop_reason=("OFFICIAL_EPISODE_TERMINATED" if terminal else "CONTROL_BUDGET_REACHED" if controls>=args.max_controls
                             else "WALL_TIME_BUDGET_REACHED" if time.perf_counter()-started>=args.max_seconds else "DECISION_BUDGET_REACHED")
            result = {"status":"complete","task":args.task,"controls":controls,"prefix_controls":prefix_count,
                      "decisions":decisions,"gate_ok":gate_ok,"implementation_digest":digest,
                      "calibration_sha":model.sha,"fk_check_count":len(checks),"gate_failures":failures,
                      "official_success":bool(done.get("success",False)),"final_goal_status":done.get("goal_status",{}),
                      "stop_reason":stop_reason,"harness":args.harness,"sensor_check_count":len(sensor_checks),
                      "final_harness":manager.context() if manager else None,
                      "model_calls":policy.calls if policy else 0,"terminal":terminal,"wall_s":time.perf_counter()-started,
                      "full_task_success_rate_claim":False}
            write(out/"result.json",result)
    except BaseException as exc:
        write(out/"failure.json",{"error":repr(exc),"controls":controls,"prefix_controls":prefix_count,"decisions":decisions})
        raise
    finally:
        trace.close()
        if video is not None: video.close()


if __name__=="__main__":
    main()
