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
from semantic_robot.v2.servo import SafeServo, ServoLimits, native_action
from semantic_robot.v2.og_calibration import CalibratedRobot
from semantic_robot.v2.onboard import OnboardRGBD
from semantic_robot.v2.grounding import observed_cloud, LocalDepthGuard
from semantic_robot.v2.grasp_motion import robot_point_mask
from run_sim import ADAPTER
from run_v2 import implementation_digest
from replay_contact_audit import preserved_session, report_failure

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
    if (value.get("schema") != SCHEMA or value.get("authorize_collection") is not True or
            value.get("collector_commit") != code or value.get("executor_digest") != executor or
            value.get("h14_body_and_finger_safety_reviewed") is not True or not value.get("reviewer") or
            value.get("max_resets") != 3 or value.get("max_candidates_per_instance") != 3):
        raise ValueError("A separately reviewed H14-fixed executor and new pilot authorization are required")
    gates = [json.loads(Path(p).read_text()) for p in value["engineering_gate_paths"]]
    if len(gates) != 2 or {g["task"] for g in gates} != {0, 3} or not all(
            g["gate_ok"] is True and g.get("robot_geometry_guards") is True and
            g["implementation_digest"] == executor for g in gates):
        raise ValueError("Both matching engineering gates required; old H13 gates are insufficient")


def capture_snapshot(folder, model, state_now, onboard, geometry_now, render, clock, budget_root=None):
    """One render-only RGB-D/actual-box snapshot, bound to unchanged local q."""
    folder = Path(folder); folder.mkdir()
    before = state_now()
    images, depths, receipts = onboard.read(model, render=render)
    geometry = geometry_now()
    after = state_now()
    if np.max(abs(before.q-after.q)) > 1e-5 or np.max(abs(before.gripper-after.gripper)) > 1e-5:
        raise RuntimeError("Physics changed during render-only capture")
    if robot_point_mask(np.empty((0, 3)), geometry) is None:
        raise ValueError("Fresh actual robot self geometry required")
    def save_bytes(file, content):
        if budget_root is not None:
            used = sum(p.stat().st_size for p in Path(budget_root).rglob("*") if p.is_file())
            if used + len(content) > 30*1024**2:
                raise RuntimeError("Raw evidence exceeds per-run artifact budget; no candidate execution")
        file.write_bytes(content)
    hashes = {}
    for view in ("head", "left_wrist", "right_wrist"):
        file = folder/(view+".png")
        buffer = io.BytesIO()
        Image.fromarray(images[view+"_rgb"].transpose(1, 2, 0)).save(buffer, format="PNG")
        save_bytes(file, buffer.getvalue())
        hashes[view] = sha(file)
    buffer = io.BytesIO(); np.savez_compressed(buffer, **depths)
    save_bytes(folder/"depth.npz", buffer.getvalue())
    write_json(folder/"robot_self_geometry.json", geometry)
    write_json(folder/"sensors.json", receipts)
    write_json(folder/"proprio.json", runtime_proprio(after))
    evidence = {"clock": clock, "q": after.q.tolist(), "gripper": after.gripper.tolist(),
                "source": "render_only_current_onboard_RGBD_and_robot_joint_FK", "scene_truth": False,
                "files_sha256": {name: sha(folder/name) for name in
                    ["depth.npz", "robot_self_geometry.json", "sensors.json", "proprio.json",
                     "head.png", "left_wrist.png", "right_wrist.png"]},
                "depth_array_sha256": {view: hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
                                       for view, raw in depths.items()}}
    write_json(folder/"capture.json", evidence)
    return after, hashes, depths, geometry, evidence


def native_preflight(model, state, grips, action, depths, geometry):
    guard = LocalDepthGuard(observed_cloud(depths, model, state.q), model, state.q, depths,
                            self_geometry=geometry)
    allowed, reason = guard.check(action, carry=True)
    servo = SafeServo(model, state, gripper_command=grips,
                      limits=ServoLimits(robot_geometry_guards=True))
    if not allowed or not servo.begin(action, state, carry=True) or servo.total_ticks > 40:
        raise RuntimeError("Native preflight rejected: "+reason+" / "+servo.status)
    return servo, {"robot_geometry_guards": True, "depth_guard": guard.receipt(),
                   "depth_check": reason, "servo_status": servo.status,
                   "q": state.q.tolist(), "gripper": state.gripper.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--gpu", type=int, choices=(1, 3), required=True)
    x = p.parse_args()
    code = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Clean immutable source required")
    release = json.loads(x.authorization.read_text())
    require_release(release, code, implementation_digest())
    ref, expected_prefix, source_binding = verify_prepared_source(x.prepared, release)
    x.output.mkdir(parents=True, exist_ok=False)
    import shutil
    if shutil.disk_usage(x.output).free < 80*1024**3:
        raise RuntimeError("Disk reserve: no reset")
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
    write_json(x.output/"manifest.json", {"schema": SCHEMA, "code": code, "authorization": release,
              "source": ref["source"], "prepared_binding": source_binding, "robot_geometry_guards": True,
              "prefix_controls": len(prefix), "native_controls_max": 200,
              "models": 0, "training_eligible": False, "no_restore_or_automatic_recovery": True})
    controls = 0; prefix_count = 0; terminal = False; first_error = None; started = None
    kin = None; grips = None; trace = None; history = []; completed = []
    def record_error(exc):
        nonlocal first_error
        first_error = first_error or exc
        report_failure(x.output, first_error, controls, started)
    with preserved_session(lambda: OfficialEvaluatorSession(window, gpu=x.gpu), record_error, lambda: first_error) as session:
        import omnigibson as og
        try:
            session.reset(); started = time.monotonic(); env = session.evaluator.env
            kin = CalibratedRobot(env.robots[0]); model = kin.calibrate(grounded=True)
            onboard = OnboardRGBD(env)
            write_json(x.output/"robot_calibration.json", model.spec)
            checks = kin.compare(model)
            if any(e["position_m"] > .003 or e["angle_rad"] > .02 or e.get("jacobian_max_abs", 0) > .04 for e in checks.values()):
                raise RuntimeError("FK drift")
            def state():
                n = kin.state(); return model.state(n.q, n.gripper, n.base_velocity)
            grips = np.clip(state().gripper/.05*2-1, -1, 1)
            trace = (x.output/"native_trace.jsonl").open("x", buffering=1)
            def step(command, kind):
                nonlocal controls, prefix_count, terminal, grips
                if terminal or (kind != "prefix" and controls >= 199):
                    raise RuntimeError("Terminal/control budget; final stop slot reserved")
                if time.monotonic()-started >= 900:
                    raise TimeoutError("Pilot wall budget")
                with og.sim.render_on_step(True):
                    obs, _, terminated, truncated, _ = env.step(command, n_render_iterations=1)
                if kind == "prefix": prefix_count += 1
                else: controls += 1
                terminal = bool(terminated or truncated); grips = np.asarray(command)[[14, 22]].copy()
                e = session.evaluator; e.obs = e._preprocess_obs(e._sync_lights_and_get_obs(obs))
                trace.write(json.dumps({"phase": kind, "prefix_control": prefix_count, "native_control": controls,
                                       "action23": np.asarray(command).tolist()})+"\n")
                if terminal: raise RuntimeError("Official episode terminated; no automatic reset")
            for command in prefix: step(command, "prefix")
            source_q = np.r_[ref["source_states"][0][53:57], ref["source_states"][0][3:10], ref["source_states"][0][28:35]]
            if np.max(abs(state().q-source_q)) > .02:
                raise RuntimeError("Expert replay diverged before pause")
            def settle():
                fixed = state().q.copy(); recent = []
                for _ in range(12):
                    step(native_action(fixed, grips), "settle"); recent.append(state())
                if not rest_screen(recent): raise RuntimeError("No near-rest after fixed 12 controls; no extra settling")
            def capture(folder):
                return capture_snapshot(folder, model, state, onboard, kin.native_self_boxes, og.sim.render,
                                        {"prefix_control": prefix_count, "native_control": controls}, x.output)
            settle()
            for index in range(3):
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
                write_json(folder/"request.json", request)
                write_json(folder/"WAITING_FOR_MANUAL_APPROVAL.json", {"request_sha256": digest(request), "deadline_seconds": 120,
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
                servo, preflight = native_preflight(model, now, grips, action, depths, geometry)
                preflight["before_capture_sha256"] = request["before_capture_sha256"]
                preflight["no_physics_since_capture"] = controls == capture_receipt["clock"]["native_control"]
                if not preflight["no_physics_since_capture"]:
                    raise RuntimeError("Approval crossed a physics step")
                write_json(folder/"preflight.json", preflight)
                start_control = controls
                native_actions = []
                while not servo.done and servo.ticks < servo.total_ticks:
                    command = servo.next_action(state()); step(command, "candidate")
                    native_actions.append({"native_control": controls, "action23": command.tolist()})
                feedback = servo.finish(state())
                write_json(folder/"native_execution.json", {"control_start": start_control, "control_end": controls,
                           "actions": native_actions})
                _, post_hashes, _, _, _ = capture(folder/"after")
                stable_hashes = None
                if feedback["status"] == "TARGET_REACHED":
                    settle()
                    _, stable_hashes, _, _, _ = capture(folder/"after_settle")
                record = {"request": request, "approval": approval,
                          "execution": {"token": token, "action": asdict(action), "status": feedback["status"],
                                        "native_controls": controls-start_control, "interrupted": feedback["status"] != "TARGET_REACHED",
                                        "native_trace_sha256": sha(folder/"native_execution.json")},
                          "feedback": feedback, "post_observation_sha256": digest({"immediate": post_hashes, "settled": stable_hashes}),
                          "settle_passed": stable_hashes is not None,
                          "training_eligible": False, "post_review_required": True}
                write_json(folder/"QUARANTINED_record.json", record); completed.append(str(folder))
                if feedback["status"] != "TARGET_REACHED": raise RuntimeError("Native action failed; no corrective retry")
                history.append(token)
        except BaseException as exc:
            record_error(exc)
        finally:
            if kin is not None and grips is not None and not terminal and controls < 200:
                try:
                    env.step(native_action(kin.state().q, grips), n_render_iterations=1); controls += 1
                    write_json(x.output/"final_hold.json", {"completed": True, "native_controls": controls})
                except BaseException as exc: record_error(exc)
            if trace is not None:
                try: trace.close()
                except BaseException as exc: record_error(exc)
        if first_error is not None: raise first_error
        try:
            write_json(x.output/"result.json", {"status": "COLLECTED_QUARANTINED_NOT_SFT", "prefix_controls": prefix_count,
                       "native_controls": controls, "samples": completed, "model_calls": 0, "official_success_claim": False})
        except BaseException as exc:
            record_error(exc); raise first_error


if __name__ == "__main__": main()
