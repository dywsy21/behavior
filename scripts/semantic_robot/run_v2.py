"""Bounded development gate / VLM rollout. v1 and active teammate code unchanged."""
import argparse
from contextlib import contextmanager
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
from semantic_robot.v2.protocol import Action, HOLD, TRANSLATIONS, ROTATIONS
from semantic_robot.v2.servo import SafeServo, ServoLimits
from semantic_robot.v2.vision import prepare_views
from semantic_robot.v2.onboard import OnboardRGBD
from semantic_robot.v2.grounding import observed_cloud, LocalDepthGuard
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.policy import GroundedPolicy, RefinedGroundedPolicy
from semantic_robot.v2.diagnostics import grasp_audit
from semantic_robot.v2.multicamera_inspection import inspection_carry, free_observing_hand
from semantic_robot.v2.arm_observation_guard import ObservingArmGuard
from semantic_robot.v2.observer_gate import choose_gate_pair
from semantic_robot.v2.motion_feedback import observed_motion_feedback
from semantic_robot.v2.wall_budget import expired, stop_before_motion
from semantic_robot.v2.run_budget import validate_run_budget


def implementation_digest():
    paths = sorted((REPO/"src/semantic_robot/v2").glob("*.py")) + [Path(__file__)]
    return hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()


def secondary_error_report(message):
    """Cleanup/logging failures must not replace the original control error."""
    try:
        print(message,file=sys.stderr,flush=True)
    except BaseException:
        pass  # Even a broken stderr must not suppress the original exception.


def bounded_gate(grounded=False,multicamera=False):
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
    if multicamera:
        actions[7:9]=[Action("left","roll_plus","coarse","tool"),Action("left","roll_minus","coarse","tool")]
    return actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("gate","agent"), default="gate")
    p.add_argument("--harness", choices=("v2","grounded"), default="v2")
    p.add_argument("--refine-grounding",action="store_true",help="Opt-in bounded crop/surface-choice perception")
    p.add_argument("--visual-odometry",action="store_true",help="Use quality-gated onboard RGB-D motion for coverage")
    p.add_argument("--active-grasp-probe",action="store_true",help="Bounded exploratory close; original grasp verification remains mandatory")
    p.add_argument("--contact-geometry",action="store_true",help="Experimental finger guides/tool translations; not mixed into the feedback-only comparison")
    p.add_argument("--grasp-motion",action="store_true",help="Repeated RGB-D target tracking with actual robot-only finger exclusion")
    p.add_argument("--robot-geometry-guards", action="store_true",
                   help="Opt-in actual robot depth self-exclusion and fully-open hand/body collision envelopes")
    p.add_argument("--approach-reorientation",action="store_true",help="Opt-in unladen wrist candidates and bounded robot-only two-command reach preview")
    p.add_argument("--approach-body-options",action="store_true",help="Opt-in all existing unladen far-pick body directions before individual safety preflight")
    p.add_argument("--odometry-estimator",choices=("pnp","rgbd_rigid","rgbd_joint"),default="pnp")
    p.add_argument("--odometry-substep-controls",type=int,choices=(0,6),default=0,
                   help="Opt-in fixed action-internal RGB-D sampling; zero preserves legacy behavior")
    p.add_argument("--odometry-self-exclusion",action="store_true",
                   help="Exclude current robot-only visual geometry at both RGB-D correspondence endpoints")
    p.add_argument("--search-motion-recovery",action="store_true",
                   help="Opt-in at most two open-hand search reference resets after measured HOLD; never old-pose recovery")
    p.add_argument("--approach-progress",action="store_true",help="Veto repeated near-field base advances without observed contact progress")
    p.add_argument("--held-object-inspection",action="store_true",help="Explicit semantic reference and bounded held-object relative inspection")
    p.add_argument("--persistent-grasp-tracks",action="store_true",help="Identity-bound persistent features and stable-depth corner selection; original grasp evidence thresholds")
    p.add_argument("--spatial-grasp-features",action="store_true",help="Fixed spatial feature quotas avoid global contrast domination; original evidence thresholds")
    p.add_argument("--inspection-budget-aware",action="store_true",help="Existing bounded coarse/fine framing plus non-revisited relative views")
    p.add_argument("--multicamera-inspection",action="store_true",help="Opt-in independent free-wrist observation with visible-depth sweep veto")
    p.add_argument("--task", type=int, choices=(0,3), default=0)
    p.add_argument("--prefix", type=int, default=0)
    p.add_argument("--replay-prefix-spec",help="Hash-pinned saved-action diagnostic warm start, counted separately from policy")
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--uri", default="http://127.0.0.1:8907")
    p.add_argument("--expected-revision")
    p.add_argument("--structured-planning", action="store_true")
    p.add_argument("--gate-result", action="append", default=[])
    p.add_argument("--max-decisions", type=int, default=48)
    p.add_argument("--max-controls", type=int, default=1536)
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--budget-profile", choices=("pilot","fullstart192"), default="pilot",
                   help="Explicitly registered original-start extension; gates and prefixes keep pilot limits")
    args = p.parse_args()
    validate_run_budget(args)
    if args.task == 3 and args.prefix:
        raise ValueError("Task3 has no registered expert prefix")
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Immutable clean source required")
    digest = implementation_digest()
    grounded=args.harness=="grounded"
    if (args.refine_grounding or args.visual_odometry or args.active_grasp_probe or args.contact_geometry or args.grasp_motion) and not grounded:
        raise ValueError("Surface refinement requires the grounded sensor contract")
    if args.grasp_motion and not args.visual_odometry:raise ValueError("Grasp registration requires measured RGB-D body motion")
    if args.robot_geometry_guards and not args.grasp_motion:
        raise ValueError("Robot geometry guards require fresh grounded robot geometry")
    if args.approach_reorientation and not args.robot_geometry_guards:
        raise ValueError("Approach reorientation requires the reviewed robot geometry guards")
    if args.approach_body_options and not args.robot_geometry_guards:
        raise ValueError("Approach body options require the reviewed robot geometry guards")
    if args.held_object_inspection and not args.grasp_motion:raise ValueError("Held inspection requires registered verified anchors")
    if args.persistent_grasp_tracks and not args.grasp_motion:raise ValueError("Persistent tracks require registered verification")
    if args.spatial_grasp_features and not args.persistent_grasp_tracks:raise ValueError("Spatial features require persistent tracks")
    if args.inspection_budget_aware and not args.held_object_inspection:raise ValueError("Inspection budget mode requires held-object inspection")
    if args.multicamera_inspection and not args.inspection_budget_aware:raise ValueError("Multi-camera inspection requires budget-aware held inspection")
    if (args.approach_progress or args.odometry_estimator!="pnp") and not args.visual_odometry:
        raise ValueError("Measured approach progress and explicit estimator require visual odometry")
    if args.odometry_substep_controls and not args.visual_odometry:
        raise ValueError("Action-internal sampling requires visual odometry")
    if args.odometry_self_exclusion and not (grounded and args.grasp_motion and args.visual_odometry and
            args.odometry_estimator=="rgbd_joint" and args.odometry_substep_controls==6):
        raise ValueError("Robot self exclusion requires fresh robot geometry and six-control joint RGB-D")
    if args.search_motion_recovery and not (grounded and args.robot_geometry_guards and
            args.held_object_inspection and args.odometry_estimator=="rgbd_joint" and
            args.odometry_substep_controls==6 and args.prefix==0 and not args.replay_prefix_spec):
        raise ValueError("Search reanchor requires original-start guarded joint RGB-D and semantic reference")
    action_control_limit=args.max_controls-(1 if args.odometry_substep_controls else 0)
    if args.mode == "agent":
        if not args.expected_revision or len(args.gate_result) != 2:
            raise ValueError("Model identity and both task pose gates required")
        gates = [json.loads(Path(path).read_text()) for path in args.gate_result]
        if {g["task"] for g in gates} != {0,3} or not all(g["gate_ok"] and g["implementation_digest"] == digest and
                g.get("harness","v2")==args.harness and
                g.get("refine_grounding",False)==args.refine_grounding and
                g.get("visual_odometry",False)==args.visual_odometry and
                g.get("contact_geometry",False)==args.contact_geometry and
                g.get("grasp_motion",False)==args.grasp_motion and
                g.get("robot_geometry_guards",False)==args.robot_geometry_guards and
                g.get("approach_reorientation",False)==args.approach_reorientation and
                g.get("approach_body_options",False)==args.approach_body_options and
                g.get("odometry_estimator","pnp")==args.odometry_estimator and
                g.get("odometry_substep_controls",0)==args.odometry_substep_controls and
                g.get("odometry_self_exclusion",False)==args.odometry_self_exclusion and
                g.get("search_motion_recovery",False)==args.search_motion_recovery and
                g.get("approach_progress",False)==args.approach_progress and
                g.get("persistent_grasp_tracks",False)==args.persistent_grasp_tracks and
                g.get("spatial_grasp_features",False)==args.spatial_grasp_features and
                g.get("inspection_budget_aware",False)==args.inspection_budget_aware and
                g.get("multicamera_inspection",False)==args.multicamera_inspection and
                g.get("held_object_inspection",False)==args.held_object_inspection for g in gates):
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
    policy_class=RefinedGroundedPolicy if args.refine_grounding else GroundedPolicy if grounded else VLMPolicy
    policy = policy_class(args.uri,args.expected_revision,max_calls=1+2*args.max_decisions+
        (2 if grounded else 0)+(16 if args.refine_grounding else 0)+(4 if args.held_object_inspection else 0),
        structured_planning=args.structured_planning) if args.mode=="agent" else None
    if policy and args.held_object_inspection and "reference" not in policy.identity.get("finite_choice_kinds",[]):
        raise ValueError("Held inspection requires a text-only reference service")
    # B15's tracking-anchor prompt did not improve the two failing states.
    # Retain it for static reproduction, not as the production default.
    manifest = {"code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
                "implementation_digest":digest, "args":vars(args), "instance":window.instance_id,
                "task":args.task,"task_name":window.task_name,"split":"train","seed":0,
                "window_sha":sha(path),"robot_sha":ROBOT_SHA,"model_identity":policy.identity if policy else None,
                "training_updates":0,"evaluator":"v3.9.1-development-not-official-v3.9.2",
                "actor_scene_truth":False,"prefix_is_expert_not_agent":bool(args.prefix),
                "matched_grasp_feedback_diagnostic":False,
                "diagnostic_replay_requested":bool(args.replay_prefix_spec),
                "actor_modalities":["rgb","depth_linear","proprio"] if grounded else ["rgb","proprio"],
                "harness":args.harness,"max_strategy_replans":2 if grounded else 0,
                "refine_grounding":args.refine_grounding,"max_surface_choices":16 if args.refine_grounding else 0,
                "visual_odometry":args.visual_odometry,
                "odometry_substep_controls":args.odometry_substep_controls,
                "odometry_self_exclusion":args.odometry_self_exclusion,
                "search_motion_recovery":args.search_motion_recovery,
                "robot_geometry_guards":args.robot_geometry_guards,
                "approach_reorientation":args.approach_reorientation,
                "approach_body_options":args.approach_body_options,
                "active_grasp_probe":args.active_grasp_probe,"privileged_audit_is_actor_input":False,
                "native_library_path":os.environ.get("LD_LIBRARY_PATH", "")}
    write(out/"manifest.json",manifest)
    replay=None
    if args.replay_prefix_spec:
        if args.mode!="agent" or not grounded or not args.grasp_motion:raise ValueError("Matched prefix is an explicit grasp-feedback agent diagnostic")
        from semantic_robot.v2.saved_prefix import load_saved_prefix
        replay=load_saved_prefix(args.replay_prefix_spec,task=args.task,prefix=args.prefix,window_sha=sha(path),robot_sha=ROBOT_SHA)
        write(out/"replay_prefix_source.json",replay["receipt"])
        manifest["diagnostic_replay_purpose"]=replay["receipt"]["purpose"]
        manifest["matched_grasp_feedback_diagnostic"]=replay["receipt"]["purpose"]=="matched_grasp_feedback_diagnostic"
        write(out/"manifest.json",manifest)
    controls, prefix_count, replay_count, terminal = 0,0,0,False
    last_issued_grips=None
    info, decisions, checks, failures = {},[],[],[]
    sensor_checks=[]
    controller=None
    video, manager, servo, state = None,None,None,None
    trace = (out/"steps.jsonl").open("x",buffering=1)
    phase,reset_completed="SESSION_INITIALIZATION",False

    @contextmanager
    def recorded_session():
        nonlocal controls
        with OfficialEvaluatorSession(window,gpu=args.gpu) as environment:
            try:
                yield environment
            except BaseException as exc:
                # The native Kit shutdown may terminate the interpreter before
                # an outer except runs. Persist the ORIGINAL error inside it.
                try:
                    if policy and policy.last_call is not None:
                        from semantic_robot.v2.structured_planning import call_receipt
                        write(out/"last_policy_call_before_failure.json",call_receipt(policy.last_call))
                    if not (out/"failure.json").exists():
                        write(out/"failure.json",{"error":repr(exc),"phase":phase,"reset_completed":reset_completed,
                            "controls":controls,"prefix_controls":prefix_count,"diagnostic_replay_controls":replay_count,"decisions":decisions,
                            "model_calls":policy.calls if policy else 0,"implementation_digest":digest})
                except BaseException as record_error:
                    secondary_error_report(f"Original failure {exc!r}; recording also failed {record_error!r}")
                # The opt-in path reserves one control slot for this stop. Save
                # the first error BEFORE attempting cleanup; a secondary stop
                # error must never replace the diagnostic that caused it.
                if args.odometry_substep_controls and reset_completed and not terminal and servo is not None:
                    stop_record={"control_before":controls,"attempted":False}
                    try:
                        if controls>=args.max_controls:
                            raise RuntimeError("Reserved safe-hold slot is unavailable")
                        stop_record["attempted"]=True
                        if last_issued_grips is not None:servo.grips[:]=last_issued_grips
                        step(servo.safe_hold(state_now())); controls+=1
                        stop_record.update(completed=True,control_after=controls)
                        trace.write(json.dumps({"control":controls,"safety_stop":True,"after_exception":True})+"\n")
                    except BaseException as stop_error:
                        stop_record.update(completed=stop_record.get("completed",False),error=repr(stop_error))
                    try:
                        write(out/"safety_hold_after_failure.json",stop_record)
                    except BaseException as record_error:
                        secondary_error_report(f"Safety hold receipt {stop_record!r}; recording failed {record_error!r}")
                raise
    try:
        with recorded_session() as environment:
            import omnigibson as og
            phase="RESET"
            environment.reset()
            reset_completed=True
            env = environment.evaluator.env
            phase="READ_ONLY_ONBOARD_ADAPTER"
            onboard=OnboardRGBD(env) if grounded else None
            phase="ROBOT_CALIBRATION"
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
                native_before=kin.state()
                q_before=native_before.q.copy()
                images,depths,receipt=onboard.read(model,render=og.sim.render)
                native_after=kin.state()
                q_after=native_after.q
                if np.max(np.abs(q_before-q_after))>1e-5:
                    raise RuntimeError("Robot moved during render-only observation barrier")
                if args.odometry_self_exclusion and not np.array_equal(native_before.gripper,native_after.gripper):
                    raise RuntimeError("Fingers moved during robot-self RGB-D observation barrier")
                check={"label":label,"cameras":receipt}
                sensor_checks.append(check)
                write(out/"sensor_checks.json",sensor_checks)
                if receipt["head"]["valid_fraction"]<.1:
                    raise RuntimeError("Head depth not initialized/usable; refusing blind grounded run")
                return images,depths,receipt

            def capture_self_frame(control,images,depths,current_state):
                from semantic_robot.v2.self_odometry import make_frame
                before=state_now()
                geometry=kin.native_self_boxes()  # Actual fingers, not transported open-pose boxes.
                after=state_now()
                if any(not np.array_equal(getattr(current_state,key),getattr(s,key))
                       for key in ("q","gripper") for s in (before,after)):
                    raise RuntimeError("Robot changed while binding current self geometry")
                return make_frame(images,depths,model,current_state,control,geometry)

            def check_fk(label):
                check = {"label":label,"errors":kin.compare(model)}
                checks.append(check)
                write(out/"fk_checks.json",checks)
                for name, errors in check["errors"].items():
                    if errors["position_m"] > .003 or errors["angle_rad"] > .02 or errors.get("jacobian_max_abs",0) > .04:
                        # OG shutdown may terminate before the outer except runs.
                        write(out/"failure.json",{"error":"FK_MISMATCH", "check":check,
                              "controls":controls,"prefix_controls":prefix_count,"diagnostic_replay_controls":replay_count,"decisions":decisions})
                        raise RuntimeError(f"Portable robot/camera FK mismatch {label}/{name}: {errors}")

            def step(action, render=True):
                nonlocal info,terminal,last_issued_grips
                # begin() can change the desired grip before any real control.
                # Cleanup must preserve the last ISSUED command, not a newly
                # selected close/open that was cancelled by a deadline.
                last_issued_grips=np.asarray(action)[[14,22]].copy()
                with og.sim.render_on_step(render):
                    obs, _, terminated, truncated, info = env.step(action,n_render_iterations=1)
                evaluator = environment.evaluator
                evaluator.obs = evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
                terminal = bool(terminated or truncated)
                return terminal

            check_fk("reset")
            phase="EXPERT_PREFIX"
            prefix = window.frozen_window().prefix_actions[:args.prefix]
            for i, action in enumerate(prefix):
                if step(action,render=(i%16==15 or i==len(prefix)-1)):
                    raise RuntimeError("Terminated in expert prefix")
                prefix_count += 1
                if i%64==63:
                    check_fk(f"prefix_{i+1}")
            check_fk("after_prefix")
            if replay is not None:
                phase="MATCHED_DIAGNOSTIC_SAVED_ACTION_REPLAY"
                replay_started=time.perf_counter()
                for i,action in enumerate(replay["actions"]):
                    if time.perf_counter()-replay_started>=600:
                        raise TimeoutError("Registered saved-prefix replay 600s budget")
                    if step(action,render=True):raise RuntimeError("Terminated during diagnostic action replay")
                    replay_count+=1
                    if i%64==63:check_fk(f"replay_{i+1}")
                check_fk("after_diagnostic_replay")
                from semantic_robot.v2.saved_prefix import check_endpoint
                matched=check_endpoint(replay,state_now());write(out/"replay_endpoint_check.json",matched)
                write(out/"PRIVILEGED_REPLAY_GRASP_AUDIT.json",grasp_audit(env.robots[0]))
                if not matched["passed"]:raise RuntimeError("Diagnostic replay endpoint mismatch; no new policy actions")
            state = state_now()
            previous_grips = (replay["actions"][-1,[14,22]] if replay is not None else
                              None if not prefix_count else np.asarray(prefix[-1])[[14,22]])
            servo = SafeServo(model,state,gripper_command=previous_grips,
                              limits=ServoLimits(robot_geometry_guards=args.robot_geometry_guards))
            if last_issued_grips is None:last_issued_grips=servo.grips.copy()
            video = imageio.get_writer(str(out/"rollout.mp4"),fps=15,codec="libx264",quality=7,macro_block_size=2)
            previous = None
            started = time.perf_counter()
            deadline = started + args.max_seconds
            if policy:policy.deadline=deadline
            phase="INITIAL_OBSERVATION"
            first_images,first_depth,first_receipt=observation_now("after_prefix")
            first = prepare_views(first_images,model,state.q,grounded=grounded,gripper=state.gripper,show_finger_regions=args.contact_geometry)
            if grounded: np.savez_compressed(out/"initial_depth.npz",**first_depth)
            for label,img in zip(first.labels,first.images): img.save(out/(label+".png"))

            def save_call(directory, name, call):
                payload = dict(call["request"])
                payload["images"] = [{"label":row["label"]} for row in payload["images"]]
                receipt={"result":call["result"],"request_without_pixel_duplicates":payload}
                if "validation" in call: receipt["validation"]=call["validation"]
                write(directory/(name+".json"),receipt)

            if policy:
                phase="INITIAL_PLAN"
                if replay is None:
                    goals, call = policy.plan(args.task,environment.observation()["task"],first)
                    save_call(out,"planner",call)
                else:
                    goals=replay["plan"]
                    write(out/"planner_source.json",{"source":"hash_pinned_saved_plan_for_matched_diagnostic","not_new_model_plan":True})
                manager = GroundedHarness(goals,active_grasp_probe=args.active_grasp_probe,contact_geometry=args.contact_geometry,held_inspection=args.held_object_inspection,reference_from_planner=args.held_object_inspection,inspection_budget_aware=args.inspection_budget_aware,multicamera_inspection=args.multicamera_inspection,approach_reorientation=args.approach_reorientation,approach_body_options=args.approach_body_options) if grounded else TaskHarness(goals)
                if replay is not None:
                    from semantic_robot.v2.saved_prefix import bootstrap_unverified_pick
                    bootstrap_unverified_pick(manager,replay,state)
                if grounded: controller=GroundedController(model,servo,manager,visual_odometry=args.visual_odometry,grasp_motion=args.grasp_motion,
                    odometry_estimator=args.odometry_estimator,approach_progress=args.approach_progress,persistent_grasp_tracks=args.persistent_grasp_tracks,
                    spatial_grasp_features=args.spatial_grasp_features,search_motion_recovery=args.search_motion_recovery,
                    odometry_self_exclusion=args.odometry_self_exclusion)
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
                draw.text((8,1065),f"Expert {prefix_count} + saved-policy {replay_count} prefix not shown. 30Hz sim /15fps.",fill="white")
                video.append_data(np.asarray(canvas))

            gate = bounded_gate(grounded,args.multicamera_inspection)
            gate_motion=None
            if args.visual_odometry and not policy:
                from semantic_robot.v2.odometry import RGBDMotion
                gate_motion=RGBDMotion(args.odometry_estimator,exclude_robot=args.odometry_self_exclusion)
            substep_motion=None
            if args.odometry_substep_controls:
                from semantic_robot.v2.substep_odometry import SubstepMotion
                substep_motion=SubstepMotion(controller.motion if controller else gate_motion,args.odometry_substep_controls)
                if controller:controller.motion=substep_motion
                else:gate_motion=substep_motion
            saved_end_snapshot=None
            phase="BOUNDED_CONTROL_OR_AGENT_LOOP"
            for decision in range(args.max_decisions):
                recoverable_stop=bool(controller and controller.can_replan_stop)
                if controls>=action_control_limit or time.perf_counter()-started>=args.max_seconds or terminal or (manager and manager.stop_reason and not recoverable_stop):
                    break
                if not policy and decision >= len(gate): break
                if grounded and shutil.disk_usage(out).free<80*1024**3:
                    if manager: manager.stop_reason="DISK_RESERVE_REACHED"
                    break
                state = state_now()
                motion_frame=None
                if saved_end_snapshot is not None:
                    previous_control,end_q,end_gripper,images,depths,depth_receipt=saved_end_snapshot[:6]
                    if args.odometry_self_exclusion:motion_frame=saved_end_snapshot[6]
                    if (previous_control!=controls or not np.array_equal(end_q,state.q)
                            or not np.array_equal(end_gripper,state.gripper)):
                        raise RuntimeError("Control/state changed before end-frame reuse")
                    saved_end_snapshot=None
                else:
                    images,depths,depth_receipt=observation_now(f"decision_{decision}")
                bundle = prepare_views(images,model,state.q,previous,grounded=grounded,gripper=state.gripper,show_finger_regions=args.contact_geometry)
                directory = out/f"decision_{decision:03d}"; directory.mkdir()
                if args.odometry_self_exclusion and motion_frame is None:
                    motion_frame=capture_self_frame(controls,images,depths,state)
                self_geometry=(motion_frame["geometry"] if args.odometry_self_exclusion else
                               kin.native_self_boxes() if args.grasp_motion else None)
                if motion_frame is not None:write(directory/"robot_motion_frame.json",motion_frame)
                if self_geometry is not None:write(directory/"robot_self_geometry.json",self_geometry)
                for label,img in zip(bundle.labels,bundle.images): img.save(directory/(label+".png"))
                write(directory/"proprio.json",{"q":state.q.tolist(),"gripper":state.gripper.tolist(),"geometry":bundle.geometry})
                if grounded:
                    np.savez_compressed(directory/"depth.npz",**depths)
                    write(directory/"depth_receipt.json",depth_receipt)
                row = {"decision":decision,"control_start":controls}
                if grounded and not policy:
                    guard=LocalDepthGuard(observed_cloud(depths,model,state.q),model,state.q,depths,
                                          self_geometry=self_geometry if args.robot_geometry_guards else None)
                    audit=guard.receipt()
                    audit["base_preflight_without_execution"]={move:guard.check(Action("base",move))
                        for move in ("forward","back","left","right")}
                    write(directory/"self_depth.json",audit)
                    if not audit["chassis_self_depth"]["available"]:
                        row["error"]="MISSING_ROBOT_SELF_GEOMETRY";failures.append(row);decisions.append(row);break
                if gate_motion is not None:
                    motion_kwargs=(dict(robot_frame=motion_frame,gripper=state.gripper,control=controls)
                                   if args.odometry_self_exclusion else {})
                    receipt=gate_motion.observe(images,depths,model,state.q,**motion_kwargs)
                    write(directory/"visual_odometry.json",receipt)
                    if not receipt["valid"]:
                        row["error"]="VISUAL_ODOMETRY_GATE_FAILED";failures.append(row);decisions.append(row);break
                if policy:
                    if args.visual_odometry:
                        motion_kwargs=(dict(robot_frame=motion_frame,control=controls) if args.odometry_self_exclusion else {})
                        receipt=controller.update_motion(images,depths,state,**motion_kwargs)
                        write(directory/"visual_odometry.json",receipt)
                        # A prior exhausted local-recovery window may enter a
                        # bounded strategy replan. Valid odometry must not turn
                        # that permission into an unconditional early break.
                        if manager.stop_reason and not controller.can_replan_stop:
                            row["stop_reason"]=manager.stop_reason;decisions.append(row);break
                    if args.held_object_inspection:
                        call=policy.resolve_target_reference(manager)
                        if call is not None:save_call(directory,"semantic_reference",call)
                        write(directory/"reference_binding.json",manager.reference_receipts.get(manager.index))
                        if manager.stop_reason:
                            row["stop_reason"]=manager.stop_reason;decisions.append(row);break
                    observation, call = policy.observe(manager,state,bundle)
                    save_call(directory,"observation",call)
                    if args.refine_grounding:
                        observation,call,receipt,detail_views=policy.refine(observation,manager,state,bundle,depths,model)
                        write(directory/"refinement.json",receipt)
                        if call is not None:save_call(directory,"surface_choice",call)
                        if detail_views is not None:
                            for label,img in zip(detail_views.labels,detail_views.images):
                                if "CROP" in label or "CANDIDATES" in label:img.save(directory/(label+".png"))
                    if controller:
                        controller.observe(observation,state,depths,depth_receipt,images=bundle.current_raw,self_geometry=self_geometry)
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
                    if controller and controller.goal_changed:
                        action=HOLD
                        row["selection_source"]="goal_transition_barrier_no_model_call"
                        write(directory/"action_selection.json",{"source":row["selection_source"],"reason":"OLD_TARGET_INVALIDATED_OBSERVE_NEW_GOAL_FIRST"})
                    elif controller and ((not observation.visible and manager.stage in ("SEARCH","RECOVER")) or controller.reposition_left>0):
                        if controller.is_held_search:
                            allowed=controller.inspection_candidates(state)
                            write(directory/"candidates.json",manager.candidate_receipt)
                            if manager.stop_reason:
                                row["stop_reason"]=manager.stop_reason;decisions.append(row);break
                            action,call=policy.act_feasible(manager,state,bundle,allowed)
                            save_call(directory,"action",call)
                            row["selection_source"]="VLM_preflighted_relative_held_inspection"
                        else:
                            action,selection=controller.search_action(state)
                            write(directory/"action_selection.json",selection)
                            row["selection_source"]=selection["source"]
                    elif controller:
                        allowed=controller.candidates(state,deadline=deadline)
                        write(directory/"candidates.json",manager.candidate_receipt)
                        if manager.stop_reason:
                            row["stop_reason"]=manager.stop_reason; decisions.append(row); break
                        if controller.grasp_verifier is not None and manager.stage=="VERIFY_GRASP":
                            action,selection=controller.verification_action(allowed)
                            row["selection_source"]=selection["source"]
                            write(directory/"action_selection.json",selection)
                            if manager.stop_reason:
                                row["stop_reason"]=manager.stop_reason;decisions.append(row);break
                        else:
                            action,call=policy.act_feasible(manager,state,bundle,allowed)
                            save_call(directory,"action",call)
                            row["selection_source"]="VLM_among_current_preflighted_actions"
                    else:
                        action, call = policy.act(manager,state,bundle)
                        save_call(directory,"action",call)
                else:
                    if args.multicamera_inspection and decision==7:
                        forward,back,pair=choose_gate_pair(model,state,servo,depths,self_geometry)
                        write(directory/"observer_gate_pair.json",pair)
                        if forward is None:
                            row["error"]=pair["reason"];failures.append(row);decisions.append(row);break
                        gate[7],gate[8]=forward,back
                    action = gate[decision]
                row["action"] = asdict(action)
                if stop_before_motion(deadline,controls,row,decisions,manager):break
                wall = time.perf_counter()
                state = state_now()  # recheck actual joints after model latency
                carry=(inspection_carry(manager,action) if args.multicamera_inspection and manager else bool(manager and manager.carry))
                accepted = servo.begin(action,state,carry=carry)
                free_actor_motion=bool(controller and controller.is_held_search and action.part==free_observing_hand(manager))
                free_gate_motion=bool(args.mode=="gate" and action.part=="left" and action.move in (*TRANSLATIONS,*ROTATIONS))
                if accepted and args.multicamera_inspection and (free_actor_motion or free_gate_motion):
                    # Re-read RGB-D AFTER model latency. Do not reuse a stale
                    # preflight cloud as a current execution clearance check.
                    _,fresh_depths,fresh_receipt=observation_now(f"preexecute_free_camera_{decision}")
                    fresh_state=state_now()
                    if (np.max(np.abs(fresh_state.q-state.q))>1e-5 or
                            np.max(np.abs(fresh_state.gripper-state.gripper))>1e-5):
                        safe,check=False,{"reason":"ROBOT_CHANGED_DURING_OBSERVER_PREFLIGHT"}
                    else:
                        guard=ObservingArmGuard(model,state.q,observed_cloud(fresh_depths,model,state.q,stride=6),kin.native_self_boxes(),action.part)
                        safe,check=guard.check(servo.joint_plan)
                    write(directory/"observer_execution_check.json",{"safe":safe,"check":check,"sensors":fresh_receipt})
                    np.savez_compressed(directory/"observer_execution_depth.npz",**fresh_depths)
                    if not safe:
                        reason="OBSERVER_EXECUTION_VETO_"+check["reason"]
                        if manager:manager.stop_reason=reason
                        else:failures.append({"decision":decision,"error":reason})
                        row.update(accepted_before_motion=False,stop_reason=reason)
                        decisions.append(row)
                        break
                row["accepted_before_motion"] = accepted
                # IK and fresh sensor preflight also take time. No ordinary
                # control is allowed merely because selection began in budget.
                if stop_before_motion(deadline,controls,row,decisions,manager):break
                previous = bundle.current_raw
                motion_fault=False
                budget_interrupted=False
                substep_result=None
                def sample_substep():
                    nonlocal saved_end_snapshot
                    sample_images,sample_depths,sample_receipts=observation_now(f"motion_{decision}_{controls}")
                    sample_state=state_now()
                    location=directory/"motion_substeps"/f"control_{controls:06d}";location.mkdir(parents=True)
                    Image.fromarray(sample_images["head_rgb"].transpose(1,2,0)).save(location/"CURRENT_HEAD_RAW.png")
                    np.savez_compressed(location/"depth.npz",head=sample_depths["head"])
                    write(location/"depth_receipt.json",{"head":sample_receipts["head"]})
                    write(location/"proprio.json",{"q":sample_state.q.tolist(),"gripper":sample_state.gripper.tolist()})
                    sample_frame=(capture_self_frame(controls,sample_images,sample_depths,sample_state)
                                  if args.odometry_self_exclusion else None)
                    if sample_frame is not None:write(location/"robot_motion_frame.json",sample_frame)
                    motion_kwargs=(dict(robot_frame=sample_frame,gripper=sample_state.gripper) if args.odometry_self_exclusion else {})
                    measured=substep_motion.sample(sample_images,sample_depths,model,sample_state.q,controls,**motion_kwargs)
                    write(location/"visual_odometry.json",measured)
                    saved_end_snapshot=(controls,sample_state.q.copy(),sample_state.gripper.copy(),sample_images,sample_depths,sample_receipts)
                    if args.odometry_self_exclusion:saved_end_snapshot+=(sample_frame,)
                    return measured["valid"]
                if accepted:
                    while not servo.done and servo.ticks<servo.total_ticks and controls<action_control_limit:
                        if expired(deadline):
                            budget_interrupted=True
                            break
                        if substep_motion is not None and controls==row["control_start"]:
                            substep_motion.begin(controls)
                        command = servo.next_action(state_now())
                        # next_action mutates the servo clock; producing and
                        # issuing this ONE control is an atomic budget quantum.
                        # Never produce it if the preceding check has expired.
                        ended = step(command,render=(controls%2==1 or servo.ticks==servo.total_ticks or servo.done))
                        controls += 1
                        trace.write(json.dumps({"control":controls,"decision":decision,"action23":command.tolist(),"terminal":ended})+"\n")
                        if controls%2==0: capture(action.text())
                        if substep_motion is not None and controls-substep_motion.last>=args.odometry_substep_controls:
                            if not sample_substep():
                                motion_fault=True
                                break
                        if ended:break
                        if expired(deadline):
                            budget_interrupted=True
                            break
                    if budget_interrupted and controls==row["control_start"]:
                        # Deadline may cross AFTER the second outer check.
                        # No control means no motion chain to finish or deliver,
                        # and no selected grip becomes an issued command.
                        stop_before_motion(deadline,controls,row,decisions,manager)
                        break
                    if substep_motion is not None:
                        if controls>substep_motion.last and not sample_substep():motion_fault=True
                        substep_result=substep_motion.finish(controls,interrupted=bool(terminal or
                            servo.ticks<servo.total_ticks or servo.status!="RUNNING"))
                        motion_fault=motion_fault or not substep_result["valid"]
                        write(directory/"action_motion.json",substep_result)
                state = state_now()
                feedback = servo.finish(state)
                raw_servo_feedback=feedback  # Later adjudication creates copies; preserve original status.
                if motion_fault:
                    feedback={**feedback,"visual_gate_failure":substep_result}
                    if feedback["status"] in ("TARGET_REACHED","BASE_TRACKING_FAILED"):
                        feedback["status"]="VISUAL_ODOMETRY_GATE_FAILED"
                    if manager:
                        manager.motion_receipt=substep_result
                        manager.stop_reason=("OFFICIAL_EPISODE_TERMINATED" if terminal else
                            feedback["status"] if feedback["status"] not in ("TARGET_REACHED","BASE_TRACKING_FAILED","VISUAL_ODOMETRY_GATE_FAILED","INTERRUPTED") else
                            "ACTION_INTERRUPTED_DURING_MOTION_MEASUREMENT" if substep_result["reason"]=="ACTION_INTERRUPTED_NO_FULL_MOTION_CERTIFICATE" else
                            "VISUAL_ODOMETRY_UNCERTAIN")
                if gate_motion is not None and accepted and action.part=="base":
                    # Judge this action before deciding whether the gate may
                    # continue. The pre-action gate observation is still the
                    # odometer reference; never judge it from the next zero-step
                    # duplicate frame or the unreliable instantaneous qvel sum.
                    if substep_motion is not None:
                        _,_,_,post_images,post_depths,post_receipt=saved_end_snapshot[:6]
                    else:
                        post_images,post_depths,post_receipt=observation_now(f"post_base_gate_{decision}")
                    post_state=state_now()
                    post=directory/"post_base_motion";post.mkdir()
                    for view in ("head","left_wrist","right_wrist"):
                        Image.fromarray(post_images[view+"_rgb"].transpose(1,2,0)).save(post/("CURRENT_"+view.upper()+"_RAW.png"))
                    np.savez_compressed(post/"depth.npz",**post_depths)
                    write(post/"depth_receipt.json",post_receipt)
                    write(post/"proprio.json",{"q":post_state.q.tolist(),"gripper":post_state.gripper.tolist()})
                    motion_kwargs={}
                    if args.odometry_self_exclusion:
                        post_frame=saved_end_snapshot[6]
                        write(post/"robot_motion_frame.json",post_frame)
                        motion_kwargs=dict(robot_frame=post_frame,gripper=post_state.gripper,control=controls)
                    motion_receipt=gate_motion.observe(post_images,post_depths,model,post_state.q,**motion_kwargs)
                    write(post/"visual_odometry.json",motion_receipt)
                    write(post/"raw_servo_feedback.json",raw_servo_feedback)
                    if not motion_receipt["valid"] or motion_receipt.get("initial",False):
                        # No fallback to qvel, and no clearing hard servo stops.
                        feedback={**feedback,"visual_gate_failure":motion_receipt}
                        if feedback["status"] in ("TARGET_REACHED","BASE_TRACKING_FAILED"):
                            feedback["status"]="VISUAL_ODOMETRY_GATE_FAILED"
                    else:
                        feedback=observed_motion_feedback(action,feedback,motion_receipt)
                # Stored separately AFTER motion, NEVER added to feedback,
                # manager context, RGB-D bundle, or neural model requests.
                write(directory/"PRIVILEGED_POST_ACTION_GRASP_AUDIT.json",grasp_audit(env.robots[0]))
                row.update(feedback=feedback,control_end=controls,wall_s=time.perf_counter()-wall,
                           wall_budget_interrupted=budget_interrupted)
                decisions.append(row); trace.write(json.dumps(row)+"\n")
                if controller: controller.executed(action,feedback)
                elif manager: manager.executed(action,feedback)
                if budget_interrupted and manager and not terminal:
                    # Keep incomplete odometry as incomplete, but distinguish
                    # the known budget interruption from a sensor failure.
                    manager.stop_reason="WALL_TIME_BUDGET_REACHED_DURING_ACTION"
                if controller and controller.search_recovery is not None and motion_fault and not budget_interrupted:
                    recovery_dir=directory/"search_reanchor";recovery_dir.mkdir()
                    def save_reanchor_snapshot(control, raw_images, raw_depths, receipts, current_state):
                        location=recovery_dir/f"control_{control:06d}";location.mkdir()
                        for view in ("head","left_wrist","right_wrist"):
                            Image.fromarray(raw_images[view+"_rgb"].transpose(1,2,0)).save(location/("CURRENT_"+view.upper()+"_RAW.png"))
                        np.savez_compressed(location/"depth.npz",**raw_depths)
                        write(location/"depth_receipt.json",receipts)
                        write(location/"proprio.json",{"q":current_state.q.tolist(),"gripper":current_state.gripper.tolist(),"control":control})
                        if controller.odometry_self_exclusion:
                            frame=capture_self_frame(control,raw_images,raw_depths,current_state)
                            write(location/"robot_motion_frame.json",frame)
                            return frame
                    anchor_hold=servo.safe_hold(state_now()).copy()
                    def issue_reanchor_hold():
                        nonlocal controls
                        if controls>=action_control_limit or expired(deadline):
                            raise RuntimeError("Recovery control budget expired before issue")
                        ended=step(anchor_hold,render=True)
                        controls+=1
                        trace.write(json.dumps({"control":controls,"reanchor_after_decision":decision,
                            "action23":anchor_hold.tolist(),"terminal":ended})+"\n")
                        if controls%2==0:capture("SENSING HOLD: NEW LOCAL REFERENCE, NO SUCCESS CLAIM")
                        return ended
                    reanchor, new_snapshot=controller.search_recovery.attempt(controller,substep_result,
                        controls=controls,action_limit=action_control_limit,terminal=terminal,deadline=deadline,
                        observe=observation_now,state_now=state_now,issue_hold=issue_reanchor_hold,
                        save_snapshot=save_reanchor_snapshot)
                    write(recovery_dir/"receipt.json",reanchor)
                    row["search_reanchor"]={k:v for k,v in reanchor.items() if k!="chain"}
                    manager.search_reanchor=controller.search_recovery.context()
                    if reanchor["valid"]:
                        substep_motion=controller.motion
                        saved_end_snapshot=new_snapshot
                        previous=None  # No previous-image comparison across the unknown gap.
                    elif terminal:
                        manager.stop_reason="OFFICIAL_EPISODE_TERMINATED_DURING_SEARCH_REANCHOR"
                    elif expired(deadline):
                        manager.stop_reason="WALL_TIME_BUDGET_REACHED_DURING_SEARCH_REANCHOR"
                check_fk(f"decision_{decision}")
                write(out/"progress.json",{"controls":controls,"decisions":len(decisions),"last":row})
                print(json.dumps(row),flush=True)
                if not policy and accepted and (motion_fault or feedback["status"] not in ("TARGET_REACHED",)):
                    failures.append(row); break  # diagnose, do not push further after a failed gate
            # Explicit zero base velocity / preserve grippers on EVERY exit path.
            if not terminal and servo is not None and controls < args.max_controls:
                if last_issued_grips is not None:servo.grips[:]=last_issued_grips
                step(servo.safe_hold(state_now())); controls += 1
                trace.write(json.dumps({"control":controls,"safety_stop":True})+"\n")
            done = info.get("done",{}) if isinstance(info,dict) else {}
            reached = [r for r in decisions if r.get("feedback",{}).get("status")=="TARGET_REACHED"]
            gate_ok = args.mode=="gate" and not failures and len(decisions)==len(gate) and len(reached)>=16
            if grounded and args.mode=="gate":
                required={13,14,19,20,21,22}
                if args.multicamera_inspection:required|={7,8}
                gate_ok=bool(gate_ok and required <= {r["decision"] for r in reached} and sensor_checks and
                             "grasp_centers_eef" in model.spec["metadata"])
            stop_reason=manager.stop_reason if manager else "CONTROL_GATE_COMPLETE" if gate_ok else "CONTROL_GATE_FAILED"
            if policy and stop_reason is None:
                stop_reason=("OFFICIAL_EPISODE_TERMINATED" if terminal else "CONTROL_BUDGET_REACHED" if controls>=args.max_controls
                             else "WALL_TIME_BUDGET_REACHED" if time.perf_counter()-started>=args.max_seconds else "DECISION_BUDGET_REACHED")
            result = {"status":"complete","task":args.task,"controls":controls,"prefix_controls":prefix_count,
                      "diagnostic_replay_controls":replay_count,
                      "diagnostic_replay_purpose":replay["receipt"]["purpose"] if replay is not None else None,
                      "matched_grasp_feedback_diagnostic":replay is not None and replay["receipt"]["purpose"]=="matched_grasp_feedback_diagnostic",
                      "decisions":decisions,"gate_ok":gate_ok,"implementation_digest":digest,
                      "calibration_sha":model.sha,"fk_check_count":len(checks),"gate_failures":failures,
                      "official_success":bool(done.get("success",False)),"final_goal_status":done.get("goal_status",{}),
                      "stop_reason":stop_reason,"harness":args.harness,"sensor_check_count":len(sensor_checks),
                      "refine_grounding":args.refine_grounding,"surface_choices":getattr(policy,"refinements",0),
                      "visual_odometry":args.visual_odometry,
                      "active_grasp_probe":args.active_grasp_probe,
                      "contact_geometry":args.contact_geometry,"grasp_motion":args.grasp_motion,
                      "robot_geometry_guards":args.robot_geometry_guards,
                      "approach_reorientation":args.approach_reorientation,
                      "approach_body_options":args.approach_body_options,
                      "odometry_estimator":args.odometry_estimator,"approach_progress":args.approach_progress,
                      "odometry_substep_controls":args.odometry_substep_controls,
                      "odometry_self_exclusion":args.odometry_self_exclusion,
                      "search_motion_recovery":args.search_motion_recovery,
                      "held_object_inspection":args.held_object_inspection,
                      "persistent_grasp_tracks":args.persistent_grasp_tracks,
                      "spatial_grasp_features":args.spatial_grasp_features,
                      "inspection_budget_aware":args.inspection_budget_aware,
                      "multicamera_inspection":args.multicamera_inspection,
                      "semantic_reference_calls":getattr(policy,"reference_calls",0),
                      "final_harness":manager.context() if manager else None,
                      "model_calls":policy.calls if policy else 0,"terminal":terminal,"wall_s":time.perf_counter()-started,
                      "full_task_success_rate_claim":False}
            write(out/"result.json",result)
    except BaseException as exc:
        try:
            if not (out/"failure.json").exists():
                write(out/"failure.json",{"error":repr(exc),"phase":phase,"reset_completed":reset_completed,
                    "controls":controls,"prefix_controls":prefix_count,"diagnostic_replay_controls":replay_count,"decisions":decisions,
                    "model_calls":policy.calls if policy else 0,"implementation_digest":digest})
        except BaseException as record_error:
            secondary_error_report(f"Original failure {exc!r}; outer recording also failed {record_error!r}")
        raise
    finally:
        primary_active=sys.exc_info()[1] is not None
        cleanup_errors=[]
        for resource in (trace,video):
            if resource is None:continue
            try:resource.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                secondary_error_report(f"Output cleanup failed: {cleanup_error!r}")
        if cleanup_errors and not primary_active:raise cleanup_errors[0]


if __name__=="__main__":
    main()
