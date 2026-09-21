"""Three-instance H09S manual-teacher pilot, NOT enabled by CPU preparation.

After explicit future authorization: exact expert prefix -> fixed-pose settle
-> export current RGB/reference -> wait WITHOUT physics -> manual decision ->
native primitive -> settle/export. Every record remains quarantined until a
separate post-review. No automatic trajectory continuation, restore or model.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __name__ == "__main__" and sys.platform.startswith("linux"):
    native = Path(sys.prefix)/f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/pymeshlab/lib"
    if (native/"libGLU.so.1").exists() and str(native) not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        env = dict(os.environ); env["LD_LIBRARY_PATH"] = str(native)+":"+env.get("LD_LIBRARY_PATH", "")
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/"src"), str(REPO/"scripts/semantic_robot")]
from common import TOKENS, sha, write_json, token_to_action
from live import runtime_proprio
from native_teacher_contract import (SCHEMA, digest, actor_input, validate_approval,
                                     verify_prepared_source, verify_loaded_window)
from native_teacher_artifacts import (ArtifactBudget, write_calibration, json_bytes,
                                     capture_layout, capture_upper_bound, action_evidence_bound)
from semantic_robot.v2.servo import SafeServo, native_action
from native_execution import authorization_profile, metadata as execution_metadata, servo_limits, completed as execution_completed
from semantic_robot.v2.og_calibration import CalibratedRobot
from semantic_robot.v2.onboard import OnboardRGBD
from semantic_robot.v2.grounding import observed_cloud, LocalDepthGuard
from semantic_robot.v2.grasp_motion import robot_point_mask
from run_sim import ADAPTER
from run_v2 import implementation_digest
from replay_contact_audit import preserved_session, report_failure
from native_teacher_near_grasp import SCHEMA as NEAR_SCHEMA
from native_teacher_reference_contract import check_actual_joint_bounds,source_state_diagnostic
from native_teacher_capacity import capacity_limits,near_artifact_budget,H09Y_PROFILE,validate_collection_location
from native_storage import activate as activate_storage,validate_spec as validate_storage_spec

PILOT_TOKENS = [t for t in TOKENS if not t.startswith("BASE_") and
                not any(axis in t for axis in ("ROLL", "PITCH", "YAW"))]


def rest_screen(states):
    """Robot-only near-rest screen, NOT a certificate of stationary objects."""
    if len(states) < 4:
        return False
    q = np.asarray([s.q for s in states[-4:]])
    g = np.asarray([s.gripper for s in states[-4:]])
    v = np.asarray([s.base_velocity for s in states[-4:]])
    values = np.concatenate([q.ravel(), g.ravel(), v.ravel()])
    if not np.isfinite(values).all(): return False
    for arm in ("left", "right"):
        positions = np.asarray([s.poses[arm][0] for s in states[-4:]])
        rotations = Rotation.from_quat(np.asarray([s.poses[arm][1] for s in states[-4:]]))
        if (not np.isfinite(positions).all() or np.linalg.norm(np.diff(positions, axis=0), axis=1).max()*30 >= .02 or
                np.linalg.norm((rotations[1:]*rotations[:-1].inv()).as_rotvec(), axis=1).max()*30 >= .05):
            return False
    return bool(np.max(abs(np.diff(q, axis=0)))*30 < .03
                and np.max(abs(np.diff(g, axis=0)))*30 < .005
                and np.max(np.linalg.norm(v[:, :2], axis=1)) < .02 and np.max(abs(v[:, 2])) < .03)


def require_release(value, code, executor):
    execution_profile=authorization_profile(value, collection=True)
    near=value.get("schema")==NEAR_SCHEMA
    if "seed_profile" in value:
        from native_teacher_pregrasp_seed import PROFILE as PRECONTACT_PROFILE
        if value["seed_profile"] != PRECONTACT_PROFILE or not near or value.get("capacity_profile") != H09Y_PROFILE:
            raise ValueError("Unknown/mixed precontact seed profile; old releases retain v1")
    if validate_storage_spec(value) and value.get("capacity_profile")!=H09Y_PROFILE:
        raise ValueError("New storage requires the explicit H09Y collection profile")
    if near:
        capacity_limits(value)
        if any(type(value.get(k)) is not int for k in ("max_resets","max_candidates_per_instance",
                "native_controls_max","seconds_after_reset","max_teacher_primitives","model_calls")):
            raise ValueError("Integer near-grasp limits required; bools/floats are not budgets")
    elif any(k in value for k in ("capacity_profile","prior_experiment_roots","combined_total_MiB")):
        raise ValueError("Capacity profiles only apply to the explicit near-grasp protocol")
    if (value.get("schema") not in (SCHEMA,NEAR_SCHEMA) or value.get("authorize_collection") is not True or
            value.get("collector_commit") != code or value.get("executor_digest") != executor or
            value.get("h14_body_and_finger_safety_reviewed") is not True or not value.get("reviewer") or
            value.get("max_resets") != (1 if near else 3) or value.get("max_candidates_per_instance") != (12 if near else 3)):
        raise ValueError("A separately reviewed H14-fixed executor and new pilot authorization are required")
    if near and (value.get("authorize_offline_teacher") is not True or value.get("allow_known_empty_rotation") is not True or
                 value.get("allow_grasp_cell_attempt") is not True or
                 value.get("native_controls_max")!=420 or value.get("seconds_after_reset")!=(1200 if value.get("capacity_profile")==H09Y_PROFILE else 900) or
                 value.get("max_teacher_primitives")!=12 or value.get("model_calls")!=0 or
                 not isinstance(value.get("experiment_root"),str)):
        raise ValueError("Exact one-reset near-grasp budget/rotation authorization required")
    gates = [json.loads(Path(p).read_text()) for p in value["engineering_gate_paths"]]
    if len(gates) != 2 or {g["task"] for g in gates} != {0, 3} or not all(
            g["gate_ok"] is True and g.get("robot_geometry_guards") is True and
            g.get("gripper_completion_v1",False) is (execution_profile is not None) and
            g["implementation_digest"] == executor for g in gates):
        raise ValueError("Both matching engineering gates required; old H13 gates are insufficient")


def capture_snapshot(folder, model, state_now, onboard, geometry_now, render, clock, budget_root=None, writer=None,
                     expected_layout=None):
    """One render-only RGB-D/actual-box snapshot, bound to unchanged local q."""
    folder = Path(folder)
    if folder.exists(): raise FileExistsError(folder)
    before = state_now()
    images, depths, receipts = onboard.read(model, render=render)
    geometry = geometry_now()
    after = state_now()
    if np.max(abs(before.q-after.q)) > 1e-5 or np.max(abs(before.gripper-after.gripper)) > 1e-5:
        raise RuntimeError("Physics changed during render-only capture")
    if robot_point_mask(np.empty((0, 3)), geometry) is None:
        raise ValueError("Fresh actual robot self geometry required")
    writer = writer or ArtifactBudget(budget_root or folder)
    layout = capture_layout(images, depths)
    if expected_layout is not None and layout != expected_layout:
        raise RuntimeError("Sensor layout changed after the action reservation")
    hashes, payloads = {}, {}
    for view in ("head", "left_wrist", "right_wrist"):
        buffer = io.BytesIO()
        Image.fromarray(images[view+"_rgb"].transpose(1, 2, 0)).save(buffer, format="PNG")
        payloads[view+".png"] = buffer.getvalue()
        hashes[view] = hashlib.sha256(payloads[view+".png"]).hexdigest()
    buffer = io.BytesIO(); np.savez_compressed(buffer, **depths)
    payloads["depth.npz"] = buffer.getvalue()
    payloads["robot_self_geometry.json"] = json_bytes(geometry)
    payloads["sensors.json"] = json_bytes(receipts)
    payloads["proprio.json"] = json_bytes(runtime_proprio(after))
    evidence = {"clock": clock, "q": after.q.tolist(), "gripper": after.gripper.tolist(),
                "kinematic_model_sha256": model.sha,
                "source": "render_only_current_onboard_RGBD_and_robot_joint_FK", "scene_truth": False,
                "array_layout": layout,
                "files_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
                "depth_array_sha256": {view: hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
                                       for view, raw in depths.items()}}
    payloads["capture.json"] = json_bytes(evidence)
    if sum(len(value) for name,value in payloads.items() if name.endswith(".json")) > 1024**2:
        raise RuntimeError("Capture metadata exceeds the registered 1 MiB bound")
    size = sum(map(len, payloads.values()))
    if size > capture_upper_bound(layout):
        raise RuntimeError("Encoded capture exceeds its raw-shape bound")
    writer.check(size)  # One WHOLE bundle, before any file appears.
    staging = folder.with_name("."+folder.name+".pending")
    staging.mkdir()
    for name, value in payloads.items():
        writer.write_bytes(staging/name, value)
    staging.rename(folder)  # Complete capture becomes visible only here.
    return after, hashes, depths, geometry, evidence


def native_preflight(model, state, grips, action, depths, geometry, rotation_evidence=None, *, execution_profile=None):
    from semantic_robot.v2.protocol import ROTATIONS
    from native_teacher_policy import empty_hand_rotation_allowed
    carry=True
    if action.move in ROTATIONS:
        if (rotation_evidence is None or not empty_hand_rotation_allowed(action.part,state,
                rotation_evidence["frame"],grips,model,rotation_evidence["close_issued"])):
            raise RuntimeError("Rotation requires current known-empty calibrated-open hand and no CLOSE latch")
        carry=False  # ONLY rotation. Existing fine translations remain 1cm.
    guard = LocalDepthGuard(observed_cloud(depths, model, state.q), model, state.q, depths,
                            self_geometry=geometry)
    allowed, reason = guard.check(action, carry=carry)
    servo = SafeServo(model, state, gripper_command=grips,
                      limits=servo_limits(execution_profile))
    if not allowed or not servo.begin(action, state, carry=carry) or servo.total_ticks > 40:
        raise RuntimeError("Native preflight rejected: "+reason+" / "+servo.status)
    return servo, {**execution_metadata(execution_profile),"robot_geometry_guards": True,"carry":carry,"amount":action.amount(carry), "depth_guard": guard.receipt(),
                   "depth_check": reason, "servo_status": servo.status,
                   "q": state.q.tolist(), "gripper": state.gripper.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--gpu", type=int, choices=(1, 3), required=True)
    p.add_argument("--teacher-config", type=Path, help="Separately authorized H09T private offline teacher; never actor input")
    x = p.parse_args()
    code = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Clean immutable source required")
    release = json.loads(x.authorization.read_text())
    require_release(release, code, implementation_digest())
    execution_profile = authorization_profile(release, collection=True)
    validate_collection_location(release,x.output,x.gpu)
    h09y=release.get("capacity_profile")==H09Y_PROFILE
    wall_limit=1200 if h09y else 900
    near=release.get("schema")==NEAR_SCHEMA
    native_limit=420 if near else 200
    if near and x.teacher_config is None:raise ValueError("Near-grasp release cannot enable the manual pilot")
    ref, expected_prefix, source_binding = verify_prepared_source(x.prepared, release)
    teacher_spec = None
    if x.teacher_config is not None:
        from native_teacher_policy import validate_spec, validate_train_group
        if (release.get("authorize_offline_teacher") is not True or
                sha(x.teacher_config) != release.get("teacher_config_sha256") or
                not 1 <= release.get("max_teacher_primitives", 0) <= (12 if near else 6)):
            raise ValueError("New bounded offline-teacher authorization required BEFORE reset")
        teacher_spec = json.loads(x.teacher_config.read_text())
        validate_spec(teacher_spec, ref)
        counts_path = REPO/"configs/vlm_sft/h09r_train_feasibility_counts.json"
        if sha(counts_path) != release.get("train_counts_sha256"):
            raise ValueError("TRAIN exclusions manifest identity mismatch")
        validate_train_group(ref, json.loads(counts_path.read_text()), release["held_out_instance_groups"])
        seed_path = Path(release["pose_evidence_path"])
        from native_teacher_seed import validate_seed_release
        review_path = Path(release["pose_review_path"])
        if sha(review_path) != release.get("pose_review_sha256"):
            raise ValueError("Unregistered independent pose review")
        validate_seed_release(seed_path, review_path, teacher_spec, ref, release["seed_reference_preparation_sha256"],
                              profile=release.get("seed_profile"))
        if teacher_spec["verb"] == "PRESS":
            from native_teacher_toggle import verify_installed_dependency
            verify_installed_dependency()
        if teacher_spec["verb"] == "PLACE_IN":
            raise ValueError("PLACE_IN all-corners volume adapter not yet independently validated; no reset")
    storage=activate_storage(release,x.output)
    x.output.mkdir(parents=True, exist_ok=False)
    artifacts = (near_artifact_budget(x.output,release) if near else
                 ArtifactBudget(x.output, x.prepared.resolve().parent.parent))
    current_writer = artifacts
    import shutil
    if shutil.disk_usage(x.output).free < 80*1024**3:
        raise RuntimeError("Disk reserve: no reset")
    if storage is None and h09y and any(shutil.disk_usage(m).free<80*1024**3 for m in ("/mnt/sdc1","/mnt/nvme_tmp")):
        raise RuntimeError("Both-filesystem reserve before reset")
    os.environ["OMNIGIBSON_GPU_ID"] = str(x.gpu); os.environ["BEHAVIOR_ACTION_STEPS"] = "1"
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    sys.path.insert(0, str(ADAPTER))
    from native_oracle_low_v1.official_factory import load_official_oracle_window, OfficialEvaluatorSession
    window = load_official_oracle_window(x.prepared/"window.json")
    prefix = np.asarray(window.frozen_window().prefix_actions)
    verify_loaded_window(window, prefix, expected_prefix, source_binding)
    # Recheck the pinned files after factory loading; no mutable input reread
    # is allowed to replace the already verified in-memory prefix.
    _, _, rechecked = verify_prepared_source(x.prepared, release)
    if rechecked != source_binding:
        raise ValueError("Prepared identity changed while loading")
    artifacts.write_json(x.output/"manifest.json", {"schema": release["schema"], "code": code, "authorization": release,
              "source": ref["source"], "prepared_binding": source_binding, "robot_geometry_guards": True,
              "prefix_controls": len(prefix), "native_controls_max": native_limit,
              "is_original_skill_phase_start":not near,"translations_m":.01,"known_empty_rotation_deg":3 if near else None,
              "models": 0, "training_eligible": False, "no_restore_or_automatic_recovery": True})
    controls = 0; prefix_count = 0; issued_native = 0; issued_prefix = 0
    terminal = False; first_error = None; started = None
    kin = None; grips = None; trace = issue_trace = None; history = []; completed = []
    teacher_reader = teacher_outcome = teacher_trace = None
    teacher_token = None; teacher_frame = None
    def record_error(exc):
        nonlocal first_error
        first_error = first_error or exc
        report_failure(x.output, first_error, controls, started)
        if near:
            try:artifacts.write_json(x.output/"PRIVATE_failure_control_ledger.json",
                {"issued_prefix":issued_prefix,"completed_prefix":prefix_count,"issued_native":issued_native,
                 "completed_native":controls,"inflight_may_have_executed":issued_prefix+issued_native!=prefix_count+controls},cleanup=True)
            except BaseException:pass
    initialization_started=time.monotonic()
    with preserved_session(lambda: OfficialEvaluatorSession(window, gpu=x.gpu), record_error, lambda: first_error) as session:
        import omnigibson as og
        try:
            if h09y and time.monotonic()-initialization_started>=900:raise TimeoutError("Initialization cap before reset")
            if storage is not None:storage.check()
            session.reset(); started = time.monotonic(); env = session.evaluator.env
            if h09y:
                # Establish cleanup capability before rejecting a late reset.
                kin=CalibratedRobot(env.robots[0]);grips=np.clip(kin.state().gripper/.05*2-1,-1,1)
                if time.monotonic()-initialization_started>=900:raise TimeoutError("Initialization cap after reset")
            kin = CalibratedRobot(env.robots[0]); model = kin.calibrate(grounded=True)
            onboard = OnboardRGBD(env)
            # Establish the reset grip latch before any artifact-budget failure,
            # so the existing finally block can hold without releasing a hand.
            grips = np.clip(kin.state().gripper/.05*2-1, -1, 1)
            write_calibration(artifacts, model)
            checks = kin.compare(model)
            if any(e["position_m"] > .003 or e["angle_rad"] > .02 or e.get("jacobian_max_abs", 0) > .04 for e in checks.values()):
                raise RuntimeError("FK drift")
            def state():
                n = kin.state(); return model.state(n.q, n.gripper, n.base_velocity)
            grips = np.clip(state().gripper/.05*2-1, -1, 1)
            trace = (x.output/"native_trace.jsonl").open("x", buffering=1)
            if near:issue_trace=(x.output/"issued_trace.jsonl").open("x",buffering=1)
            def step(command, kind):
                nonlocal controls, prefix_count, terminal, grips, teacher_frame,issued_native,issued_prefix
                if terminal or (kind != "prefix" and (issued_native if near else controls) >= native_limit-1):
                    raise RuntimeError("Terminal/control budget; final stop slot reserved")
                if time.monotonic()-started >= wall_limit:
                    raise TimeoutError("Pilot wall budget")
                if shutil.disk_usage(x.output).free < 80*1024**3:
                    raise RuntimeError("Disk reserve exhausted; no additional control")
                if storage is not None:storage.check()
                if near:check_actual_joint_bounds(state(),model)
                line = json.dumps({"phase": kind, "prefix_control": prefix_count+int(kind=="prefix"),
                                   "native_control": controls+int(kind!="prefix"),
                                   "action23": np.asarray(command).tolist()}, allow_nan=False)+"\n"
                current_writer.check(len(line.encode("utf-8")))  # BEFORE env.step, including expert prefix.
                # Once issued, a CLOSE command must also be the cleanup latch
                # if env.step throws after partial physical execution.
                if near:current_writer.append_text(issue_trace,line)
                grips = np.asarray(command)[[14,22]].copy()
                if kind=="prefix":issued_prefix+=1
                else:issued_native+=1
                with og.sim.render_on_step(True):
                    obs, _, terminated, truncated, _ = env.step(command, n_render_iterations=1)
                if kind == "prefix": prefix_count += 1
                else: controls += 1
                terminal = bool(terminated or truncated); grips = np.asarray(command)[[14, 22]].copy()
                e = session.evaluator; e.obs = e._preprocess_obs(e._sync_lights_and_get_obs(obs))
                current_writer.append_text(trace, line)
                if teacher_reader is not None:
                    actual_state=state()
                    if near:check_actual_joint_bounds(actual_state,model)
                    teacher_frame = teacher_reader.read(prefix_count+controls)
                    teacher_frame["finger_opening"] = dict(zip(("left", "right"), actual_state.gripper.tolist()))
                    verdict = teacher_outcome.update(teacher_frame, teacher_token)
                    evidence_line = json.dumps({"frame": teacher_frame, "verdict": verdict,
                        "actual_q":actual_state.q.tolist(),"actual_gripper":actual_state.gripper.tolist()}, allow_nan=False)+"\n"
                    if len(evidence_line.encode()) > 16384:
                        raise RuntimeError("Teacher telemetry exceeds registered bound; quarantine entire trajectory")
                    current_writer.append_text(teacher_trace, evidence_line)
                    if verdict["outcome"] in ("UNKNOWN", "FAILED"):
                        raise RuntimeError("Teacher stopped without positive labels: "+verdict["reason"])
                if terminal: raise RuntimeError("Official episode terminated; no automatic reset")
            for command in prefix: step(command, "prefix")
            source_q = np.r_[ref["source_states"][0][53:57], ref["source_states"][0][3:10], ref["source_states"][0][28:35]]
            if near:
                check_actual_joint_bounds(state(),model)
                artifacts.write_json(x.output/"PRIVATE_source_proprio_diagnostic.json",
                    source_state_diagnostic(state(),ref["source_states"][0],len(prefix)))
            elif np.max(abs(state().q-source_q)) > .02:
                raise RuntimeError("Expert replay diverged before pause")
            def settle():
                fixed = state().q.copy(); recent = []
                for _ in range(12):
                    step(native_action(fixed, grips), "settle"); recent.append(state())
                if not rest_screen(recent): raise RuntimeError("No near-rest after fixed 12 controls; no extra settling")
            def capture(folder, layout=None):
                return capture_snapshot(folder, model, state, onboard, kin.native_self_boxes, og.sim.render,
                                        {"prefix_control": prefix_count, "native_control": controls}, x.output, current_writer,
                                        expected_layout=layout)
            settle()
            if teacher_spec is not None:
                from native_teacher_policy import PoseTeacher
                from native_teacher_outcomes import LocalOutcome
                from native_teacher_og import PrivilegedReader
                teacher_reader = PrivilegedReader(env, teacher_spec)
                teacher_outcome = LocalOutcome(teacher_spec)
                teacher_frame = teacher_reader.read(prefix_count+controls)
                teacher_frame["finger_opening"] = dict(zip(("left", "right"), state().gripper.tolist()))
                initial = teacher_outcome.update(teacher_frame)
                if initial["outcome"] in ("UNKNOWN", "FAILED"):
                    raise RuntimeError("Unknown initial teacher evidence")
                if near and (teacher_spec["verb"]!="GRASP" or any(
                        teacher_frame.get("held_any",{}).get(a) is not False or
                        teacher_frame["held"].get(a) is not False or
                        teacher_frame.get("finger_external_contact",{}).get(a) is not False
                        for a in ("left","right"))):
                    raise RuntimeError("Near-grasp starts with unknown load/contact; no action")
                artifacts.write_json(x.output/"PRIVATE_teacher_initial.json", {"spec": teacher_spec, "frame": teacher_frame,
                     "baseline_contacts": teacher_reader.baseline_receipt, "not_actor_input": True})
                teacher_trace = (x.output/"PRIVATE_teacher_trace.jsonl").open("x", buffering=1)
                teacher = PoseTeacher(teacher_spec,allow_empty_rotation=near,
                    allow_grasp_cell_attempt=near and release.get("allow_grasp_cell_attempt") is True)
                for index in range(release["max_teacher_primitives"]):
                    folder = x.output/f"teacher_{index:02d}"; folder.mkdir()
                    before, hashes, depths, geometry, receipt = capture(folder/"before")
                    teacher_reader.check_local_fk(before, teacher_frame)
                    actor = actor_input(session.observation()["task"], ref["active_instruction"], runtime_proprio(before), hashes, history[-5:])
                    ranked = teacher.ranked(before, teacher_frame, teacher_reader.goal(), teacher_reader.base(),grips,model)
                    artifacts.write_json(folder/"PRIVATE_proposal.json", teacher.proposal_receipt)
                    choice = None; rejected = []
                    for token in ranked:
                        try:
                            candidate, preflight = native_preflight(model, before, grips, token_to_action(token), depths, geometry,
                                {"frame":teacher_frame,"close_issued":teacher.close_issued} if near else None,
                                execution_profile=execution_profile)
                            choice = token, candidate, preflight
                            break
                        except RuntimeError as exc: rejected.append({"token": token, "reason": str(exc)})
                    if choice is None: raise RuntimeError("No safe decreasing teacher proposal; no retry/reset")
                    token, servo, preflight = choice
                    if controls+servo.total_ticks+12 > native_limit-1:
                        raise RuntimeError("Insufficient whole primitive + settle budget BEFORE action")
                    layout = receipt["array_layout"]
                    if shutil.disk_usage(x.output).free < 80*1024**3+action_evidence_bound(layout):
                        raise RuntimeError("Insufficient physical disk reserve for whole action evidence")
                    try:
                        with artifacts.transaction(action_evidence_bound(layout)) as reserved:
                            current_writer = reserved; teacher_token = token
                            reserved.write_json(folder/"request.json", {"actor": actor, "token": token,
                                "private_proposal_order": ranked, "rejected": rejected, "preflight": preflight,
                                "private_proposal_basis": teacher.proposal_receipt,
                                "capture_sha256": sha(folder/"before/capture.json"), "training_eligible": False})
                            start_control = controls
                            while not servo.done and servo.ticks < servo.total_ticks:
                                step(servo.next_action(state()), "candidate")
                            feedback = servo.finish(state())
                            reserved.write_json(folder/"native_execution.json", {"token": token, "control_start": start_control,
                                "control_end": controls, "feedback": feedback, **execution_metadata(execution_profile)})
                            capture(folder/"after", layout)
                            if not execution_completed(token, feedback, profile=execution_profile): raise RuntimeError("Teacher native execution failed")
                            teacher_token = None
                            settle(); capture(folder/"after_settle", layout)
                            teacher.executed(token, teacher_frame); history.append(token)
                            reserved.write_json(folder/"QUARANTINED_record.json", {"actor": actor, "token": token,
                                **execution_metadata(execution_profile),
                                "native_trace_sha256": sha(folder/"native_execution.json"), "local_outcome": teacher_outcome.result,
                                "training_eligible": False, "whole_trajectory_terminal_review_required": True})
                            completed.append(str(folder))
                    finally:
                        current_writer = artifacts; teacher_token = None
                    if teacher_outcome.result["outcome"] == "SUCCEEDED": break
                artifacts.write_json(x.output/"PRIVATE_local_outcome.json", teacher_outcome.result)
                if teacher_outcome.result["outcome"] != "SUCCEEDED":
                    raise RuntimeError("Bounded teacher ended without local completion; zero positive BC labels")
            for index in range(0 if teacher_spec is not None else 3):
                if sum(v.stat().st_size for v in x.output.rglob("*") if v.is_file()) > 30*1024**2 or shutil.disk_usage(x.output).free < 80*1024**3:
                    raise RuntimeError("Pilot disk budget")
                folder = x.output/f"sample_{index:02d}"; folder.mkdir()
                before, hashes, depths, geometry, capture_receipt = capture(folder/"before")
                actor = actor_input(session.observation()["task"], ref["active_instruction"], runtime_proprio(before), hashes, history[-5:])
                request = {"schema": SCHEMA, "actor": actor, "native_control": controls,
                           "source_group": {k: ref["pilot"][k] for k in ("task", "episode", "instance")},
                           "allowed_tokens": PILOT_TOKENS,
                           "execution_constraints": "Unknown load treated conservatively: carry=True; no base/rotation in this first pilot",
                           "source_reference": str(x.prepared/"teacher_reference.json"),
                           "source_reference_sha256": sha(x.prepared/"teacher_reference.json"),
                           "before_capture_sha256": sha(folder/"before/capture.json"),
                           "proposal_aid": ref["proposals"], "must_revalidate_intent_after_pause": True}
                artifacts.write_json(folder/"request.json", request)
                artifacts.write_json(folder/"WAITING_FOR_MANUAL_APPROVAL.json", {"request_sha256": digest(request), "deadline_seconds": 120,
                           "approval_path": str(folder/"approval.json"), "physics_steps_while_waiting": 0})
                wait_start = time.monotonic()
                while not (folder/"approval.json").exists():
                    if time.monotonic()-wait_start > 120 or time.monotonic()-started >= 900:
                        raise TimeoutError("Manual approval not supplied in budget")
                    time.sleep(.2)  # No stepping/rendering/controller update while reviewer reads.
                approval = json.loads((folder/"approval.json").read_text()); token = validate_approval(approval, request)
                if token is None: break
                now = state()
                if np.max(abs(now.q-before.q)) > 1e-5 or np.max(abs(now.gripper-before.gripper)) > 1e-5:
                    raise RuntimeError("Approval is stale")
                action = token_to_action(token); carry = True  # Never infer unloaded from a gripper command/aperture.
                if action.part == "base":
                    raise ValueError("First manipulation pilot excludes BASE pending H14 deployment-safe body guard")
                servo, preflight = native_preflight(model, now, grips, action, depths, geometry,
                                                    execution_profile=execution_profile)
                preflight["before_capture_sha256"] = request["before_capture_sha256"]
                preflight["no_physics_since_capture"] = controls == capture_receipt["clock"]["native_control"]
                if not preflight["no_physics_since_capture"]:
                    raise RuntimeError("Approval crossed a physics step")
                layout = capture_receipt["array_layout"]
                if shutil.disk_usage(x.output).free < 80*1024**3+action_evidence_bound(layout):
                    raise RuntimeError("Insufficient physical disk reserve for whole action evidence")
                if controls+servo.total_ticks+12 > native_limit-1:
                    raise RuntimeError("Whole primitive and settling exceed remaining control budget")
                try:
                    with artifacts.transaction(action_evidence_bound(layout)) as reserved:
                        current_writer = reserved
                        reserved.write_json(folder/"preflight.json", preflight)
                        start_control = controls
                        native_actions = []
                        while not servo.done and servo.ticks < servo.total_ticks:
                            command = servo.next_action(state()); step(command, "candidate")
                            native_actions.append({"native_control": controls, "action23": command.tolist()})
                        feedback = servo.finish(state())
                        reserved.write_json(folder/"native_execution.json", {"control_start": start_control, "control_end": controls,
                                   "actions": native_actions, "feedback": feedback})
                        _, post_hashes, _, _, _ = capture(folder/"after", layout)
                        stable_hashes = None
                        if execution_completed(token, feedback, profile=execution_profile):
                            settle()
                            _, stable_hashes, _, _, _ = capture(folder/"after_settle", layout)
                        record = {"request": request, "approval": approval,
                                  "execution": {"token": token, "action": asdict(action), "status": feedback["status"],
                                                "native_controls": len(native_actions),
                                                "post_action_settle_controls": controls-start_control-len(native_actions),
                                                "interrupted": not execution_completed(token, feedback, profile=execution_profile),
                                                "native_trace_sha256": sha(folder/"native_execution.json")},
                                  "feedback": feedback, "post_observation_sha256": digest({"immediate": post_hashes, "settled": stable_hashes}),
                                  "settle_passed": stable_hashes is not None,
                                  "training_eligible": False, "post_review_required": True}
                        reserved.write_json(folder/"QUARANTINED_record.json", record); completed.append(str(folder))
                finally:
                    current_writer = artifacts
                if not execution_completed(token, feedback, profile=execution_profile): raise RuntimeError("Native action failed; no corrective retry")
                history.append(token)
        except BaseException as exc:
            record_error(exc)
        finally:
            if kin is not None and grips is not None and not terminal and (issued_native if near else controls) < native_limit:
                try:
                    stop = native_action(kin.state().q, grips)
                    if near:
                        try:artifacts.write_json(x.output/"final_hold_attempt.json",
                            {"issued_native":issued_native+1,"action23":stop.tolist()},cleanup=True)
                        except BaseException as exc:record_error(exc)
                    issued_native+=1
                    env.step(stop, n_render_iterations=1); controls += 1
                    artifacts.write_json(x.output/"final_hold.json", {"completed": True, "native_controls": controls,
                                         "prefix_controls": prefix_count, "action23": stop.tolist()}, cleanup=True)
                    if teacher_reader is not None:
                        if near:check_actual_joint_bounds(state(),model)
                        frame = teacher_reader.read(prefix_count+controls)
                        frame["finger_opening"] = dict(zip(("left", "right"), kin.state().gripper.tolist()))
                        verdict = teacher_outcome.update(frame)
                        artifacts.write_json(x.output/"PRIVATE_final_hold_outcome.json", {"frame": frame, "verdict": verdict}, cleanup=True)
                        if verdict["outcome"] != "SUCCEEDED" and first_error is None:
                            raise RuntimeError("Final hold did not preserve successful local outcome")
                except BaseException as exc: record_error(exc)
            if trace is not None:
                try: trace.close()
                except BaseException as exc: record_error(exc)
            if issue_trace is not None:
                try:issue_trace.close()
                except BaseException as exc:record_error(exc)
            if teacher_trace is not None:
                try: teacher_trace.close()
                except BaseException as exc: record_error(exc)
            if teacher_reader is not None:
                try: teacher_reader.close()
                except BaseException as exc: record_error(exc)
        if first_error is not None: raise first_error
        try:
            artifacts.write_json(x.output/"result.json", {"status": "COLLECTED_QUARANTINED_NOT_SFT", "prefix_controls": prefix_count,
                       "native_controls": controls, "samples": completed, "model_calls": 0, "official_success_claim": False})
        except BaseException as exc:
            record_error(exc); raise first_error


if __name__ == "__main__": main()
