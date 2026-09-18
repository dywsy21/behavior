"""Frozen native-action replay; privileged contacts are a write-only audit sink.

No actor, planner, new target or recovery exists here. This is not a success-rate
trial. Matching proprio is necessary, not proof that scene dynamics match H13.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if __name__ == "__main__" and sys.platform.startswith("linux"):
    native = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/pymeshlab/lib"
    entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if (native / "libGLU.so.1").exists() and str(native) not in entries:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join([str(native), *[v for v in entries if v]])
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

import numpy as np
from contact_replay_contract import load_contract
from run_sim import ADAPTER, WINDOWS, WINDOW_NAMES, WINDOW_SHAS, ROBOT_SHA, REPO, sha, write
from semantic_robot.v2.og_calibration import CalibratedRobot
from semantic_robot.v2.servo import native_action
from semantic_robot.v2.onboard import OnboardRGBD


def report_failure(out, first_error, controls, started):
    """Never raise while reporting a primary error, including broken stderr."""
    try:
        write(out / "failure.json", {"error": repr(first_error), "controls": controls,
              "autonomous_policy": False, "model_calls": 0,
              "wall_s": None if started is None else time.monotonic() - started})
    except BaseException as secondary:
        try:
            print(f"Original error {first_error!r}; recording error {secondary!r}", file=sys.stderr, flush=True)
        except BaseException:
            pass


def audit_pairs(api, scene_idx, queried, registered_rows, required_links, registered_cols, current_only):
    forward = api.get_contact_pairs(scene_idx, queried, None, current_only)
    # The installed column lookup raises for unregistered visual-only links.
    columns = required_links & registered_cols
    reverse = api.get_contact_pairs(scene_idx, registered_rows, columns, current_only) if columns else set()
    return sorted({tuple(sorted(pair)) for pair in forward | reverse})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, choices=(3,), default=3)
    p.add_argument("--max-seconds", type=int, default=1200)
    a = p.parse_args()
    if not 1 <= a.max_seconds <= 1200:
        raise ValueError("Bounded registered replay only")
    replay = load_contract(a.spec)
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Clean immutable diagnostic source required")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 80 * 1024**3:
        raise RuntimeError("Keep 80GiB free; no new reset")
    os.environ["OMNIGIBSON_GPU_ID"] = str(a.gpu)
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "1")
    os.environ["BEHAVIOR_ACTION_STEPS"] = "1"
    sys.path.insert(0, str(ADAPTER))
    from native_oracle_low_v1.official_factory import load_official_oracle_window, OfficialEvaluatorSession
    import imageio.v2 as imageio
    from PIL import Image
    window_path = WINDOWS / WINDOW_NAMES[0] / "window.json"
    manifest = replay["manifest"]
    if sha(window_path) != WINDOW_SHAS[0] or manifest["window_sha"] != sha(window_path) or manifest["robot_sha"] != ROBOT_SHA:
        raise ValueError("Source task/window/robot identity drift")
    window = load_official_oracle_window(window_path)
    if window.instance_id != 138 or window.seed != 0 or window.official_mode != "train" or window.robot_config_sha256 != ROBOT_SHA:
        raise ValueError("Original development reset required")
    write(out / "manifest.json", {
        "args": vars(a), "source_spec_sha256": replay["spec_sha256"], "source_spec": replay["spec"],
        "code_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "factory_sha256": sha(ADAPTER / "native_oracle_low_v1/official_factory.py"),
        "model_calls": 0, "autonomous_policy": False, "scene_state_used_to_select_commands": False,
        "render_schedule": "even controls and action tails; initial two barriers and six-tick/tail barriers; cached boundaries",
        "max_artifact_MiB": 200, "final_hold": "reconstructed from current q and last commanded grip; original vector unavailable",
        "contact_semantics": "current cache retains sleeping pairs; accumulated covers preceding cache interval; neither gives force/location"})
    controls, terminal, first_error, started = 0, False, None, None
    kin, video, audit, last_grips = None, None, None, None

    def record_error(exc):
        nonlocal first_error
        if first_error is None:
            first_error = exc
        report_failure(out, first_error, controls, started)

    # Persist completion/failure BEFORE native session teardown, which can quit
    # the interpreter. Cleanup never hides an earlier diagnostic exception.
    with OfficialEvaluatorSession(window, gpu=a.gpu) as session:
        import omnigibson as og
        from omnigibson.utils.usd_utils import RigidContactAPI
        reset_completed = False
        try:
            session.reset()
            reset_completed = True
            started = time.monotonic()
            env = session.evaluator.env
            robot = env.robots[0]
            kin = CalibratedRobot(robot)
            last_grips = np.clip(kin.state().gripper / .05 * 2 - 1, -1, 1)
            model = kin.calibrate(grounded=True)
            write(out / "robot_calibration.json", model.spec)
            check = kin.compare(model)
            write(out / "fk_check.json", check)
            if any(v["position_m"] > .003 or v["angle_rad"] > .02 or v.get("jacobian_max_abs", 0) > .04 for v in check.values()):
                raise RuntimeError("Robot/camera FK mismatch")
            scene_idx = next(i for i, s in enumerate(og.sim.scenes) if s is env.scene)
            task_objects = {name: obj for name, obj in env.task.object_scope.items()
                            if obj is not None and hasattr(obj, "links") and hasattr(obj, "get_position_orientation")}
            queried = set(task_objects.values()) | {robot}
            if not any(obj is not robot for obj in queried):
                raise RuntimeError("No auditable task object scope")
            if scene_idx not in RigidContactAPI._CONTACT_MATRIX or not len(RigidContactAPI.get_contact_row_indices(scene_idx, {robot})):
                raise RuntimeError("Contact cache unavailable; refuse evidence-free replay")
            registered_rows = set(RigidContactAPI._PATH_TO_ROW_IDX[scene_idx])
            registered_cols = set(RigidContactAPI._PATH_TO_COL_IDX[scene_idx])
            registered = registered_rows | registered_cols
            required_links = {link.prim_path for obj in queried for link in obj.links.values()}
            coverage = {obj.name: {"query_row_links": [link.prim_path for link in obj.links.values() if link.prim_path in registered_rows],
                                  "filter_column_links": [link.prim_path for link in obj.links.values() if link.prim_path in registered_cols],
                                  "registered_links": [link.prim_path for link in obj.links.values() if link.prim_path in registered],
                                  "unregistered_links": [link.prim_path for link in obj.links.values() if link.prim_path not in registered]}
                        for obj in queried}
            write(out / "contact_coverage.json", {"scene_idx": scene_idx, "objects": coverage})
            if any(not row["registered_links"] for row in coverage.values()):
                raise RuntimeError("A queried entity has no registered contact links")
            # Map only for the write-only sink, to log table motion even if the
            # contacted support was not part of the task's object scope.
            link_objects = {link.prim_path: obj for obj in env.scene.objects for link in obj.links.values()}
            audit = (out / "PRIVILEGED_CONTACTS.jsonl").open("x", buffering=1)
            onboard = OnboardRGBD(env)
            video = imageio.get_writer(str(out / "rollout.mp4"), fps=15, codec="libx264", quality=7, macro_block_size=2)

            def object_audit(control, phase):
                # No returned data can affect action choice, success, or timing.
                def values(x):
                    return x.detach().cpu().numpy().tolist() if hasattr(x, "detach") else np.asarray(x).tolist()
                def pairs(current_only):
                    # A static support may be a column only. Include contacts
                    # from every dynamic row TO the required task/robot links.
                    return audit_pairs(RigidContactAPI, scene_idx, queried, registered_rows,
                                       required_links, registered_cols, current_only)
                current, recent = pairs(True), pairs(False)
                objects = queried | {link_objects[path] for pair in current + recent for path in pair if path in link_objects}
                poses = {obj.name: {"prim_path": obj.prim_path, "world_pose_xyzw": [values(x) for x in obj.get_position_orientation()]}
                         for obj in objects}
                audit.write(json.dumps({"control": control, "phase": phase, "not_actor_input": True,
                       "current_cached_pairs": current, "preceding_cache_interval_pairs": recent,
                       "task_names": {name: obj.name for name, obj in task_objects.items()},
                       "object_world_poses": poses}, allow_nan=False) + "\n")

            def capture():
                before = kin.state()
                snapshot = onboard.read(model, render=og.sim.render)
                after = kin.state()
                if np.max(np.abs(before.q - after.q)) > 1e-5 or np.max(np.abs(before.gripper - after.gripper)) > 1e-5:
                    raise RuntimeError("State changed across render-only barrier")
                return snapshot

            # Match initial planning capture, then decision_0 capture; later
            # boundaries REUSE the preceding six-tick/tail sample as H13 did.
            capture()
            snapshot = capture()
            object_audit(0, "reset_before_replay")
            for index, command in enumerate(replay["actions"]):
                if time.monotonic() - started >= a.max_seconds:
                    raise TimeoutError("Registered replay wall budget")
                if index in replay["boundaries"]:
                    boundary = replay["boundaries"][index]
                    state = kin.state()
                    receipt = {"decision": boundary["decision"], "control": controls,
                               "q_max_abs_rad": float(np.max(np.abs(state.q - boundary["q"]))),
                               "gripper_max_abs_m": float(np.max(np.abs(state.gripper - boundary["gripper"]))),
                               "not_proof_of_same_scene": True}
                    folder = out / f"decision_{boundary['decision']:03d}"
                    folder.mkdir()
                    write(folder / "boundary.json", receipt)
                    if receipt["q_max_abs_rad"] > .01 or receipt["gripper_max_abs_m"] > .003:
                        raise RuntimeError("Replay proprio diverged; no causal attribution to old run")
                    images, depths, sensors = snapshot
                    for view in ("head", "left_wrist", "right_wrist"):
                        Image.fromarray(images[view + "_rgb"].transpose(1, 2, 0)).save(folder / ("CURRENT_" + view.upper() + "_RAW.png"))
                    write(folder / "proprio.json", {"q": state.q.tolist(), "gripper": state.gripper.tolist()})
                    write(folder / "depth_receipt.json", sensors)
                    if boundary["decision"] >= 38:
                        np.savez_compressed(folder / "depth.npz", **depths)
                        write(folder / "robot_self_geometry.json", kin.native_self_boxes())
                    size = sum(v.stat().st_size for v in out.rglob("*") if v.is_file())
                    if size >= 185 * 1024**2 or shutil.disk_usage(out).free < 80 * 1024**3:
                        raise RuntimeError("Artifact/reserve guard, leaving 15MiB for stop/flush")
                render = (controls + 1) % 2 == 0 or controls + 1 in replay["end_controls"]
                with og.sim.render_on_step(render):
                    obs, _, terminated, truncated, _ = env.step(command, n_render_iterations=1)
                controls += 1
                terminal = bool(terminated or truncated)
                last_grips = command[[14, 22]].copy()
                evaluator = session.evaluator
                evaluator.obs = evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
                object_audit(controls, "after_frozen_command")
                if controls % 2 == 0:
                    video.append_data(np.asarray(session.observation()["images"]["head_rgb"]).transpose(1, 2, 0))
                if controls in replay["sample_controls"]:
                    snapshot = capture()
                if terminal:
                    raise RuntimeError("Episode terminated during diagnostic replay")
        except BaseException as exc:
            record_error(exc)
        finally:
            if reset_completed and kin is not None and not terminal:
                stop = {"attempted": True, "reconstructed_not_copied": True, "control_before": controls}
                try:
                    with og.sim.render_on_step(True):
                        env.step(native_action(kin.state().q, last_grips), n_render_iterations=1)
                    controls += 1
                    stop.update(completed=True, total_controls=controls)
                    if audit is not None:
                        object_audit(controls, "final_safe_hold_not_new_policy")
                except BaseException as exc:
                    stop.update(completed=stop.get("completed", False), error=repr(exc))
                    record_error(exc)
                try:
                    write(out / "final_hold.json", stop)
                except BaseException as exc:
                    record_error(exc)
            for handle in (audit, video):
                if handle is not None:
                    try:
                        handle.close()
                    except BaseException as exc:
                        record_error(exc)
        if first_error is not None:
            raise first_error
        try:
            write(out / "result.json", {"status": "frozen_replay_completed_not_policy_success",
                  "replayed_controls": len(replay["actions"]), "total_controls_including_reconstructed_hold": controls,
                  "model_calls": 0, "autonomous_policy": False, "wall_s": time.monotonic() - started,
                  "same_scene_state_certified": False})
        except BaseException as exc:
            record_error(exc)
            raise first_error


if __name__ == "__main__":
    main()
