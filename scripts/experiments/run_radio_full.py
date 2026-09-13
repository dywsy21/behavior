"""One official-init autonomous A+B episode; UNKNOWN-only feedback variant.

No demonstration prefix, fixed oracle subgoal, or privileged policy input.
Full task success is read from the official environment's done.success, while
all terminations, controller faults, consumed actions and video remain saved.
"""
from __future__ import annotations
import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback
from radio_full_manifest import validate_manifest


class BackgroundPolicyLoop:
    """Keep socket pongs alive while synchronous Kit owns the main thread.

    Kit pumps its own asyncio callbacks during load/step. It must never run
    inside our coroutine, and long scene loads must not starve model sockets.
    No simulator objects are accessed by this worker.
    """
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.closed = False
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="native-ab-policy-io", daemon=True)
        self.thread.start()
        if not self.ready.wait(10):
            raise RuntimeError("Policy communication thread did not start")

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            asyncio.set_event_loop(None)

    def call(self, coroutine, *, timeout=1800):
        if self.closed:
            coroutine.close()
            raise RuntimeError("Policy communication loop is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=timeout)
        except BaseException:
            future.cancel()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=15)
            if self.thread.is_alive():
                raise RuntimeError("Policy communication thread did not stop")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--episode-index", type=int, choices=range(3), required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--high-uri", default="ws://127.0.0.1:8771")
    p.add_argument("--low-uri", default="ws://127.0.0.1:8770")
    p.add_argument("--gpu", type=int, default=3)
    args = p.parse_args()
    sys.path.insert(0, str(args.runtime))
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    episode = manifest["episodes"][args.episode_index]
    if sha(manifest["robot_config_path"]) != manifest["robot_config_sha256"]:
        raise ValueError("robot configuration changed")
    if sha(manifest["tasks_path"]) != manifest["tasks_sha256"]:
        raise ValueError("task instructions changed")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.adapter), str(args.source/"src"), str(args.source)]
    import numpy as np
    import imageio.v2 as imageio
    from native_ab_controller import NativeABController
    from native_oracle_low_v1.official_factory import (
        _lazy_official_imports, load_task_instructions, behavior_obs_to_native_low,
        DEFAULT_OG_ROOT, DEFAULT_EVAL_ROOT,
    )
    instructions = load_task_instructions(manifest["tasks_path"])
    started = time.monotonic()
    actions = []
    controller = None
    policy_loop = None
    policy_closed = False
    video = None
    trace = None
    pose_trace = None
    physical_trace = None
    physical_observer = None
    result = dict(kind="native_ab_autonomous_episode", variant=manifest["variant"],
        status="starting", manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        episode=episode, teacher_prefix_actions=0, oracle_subgoals_used=False,
        physical_feedback_trained=False, policy_input="RGB/proprio/issued-command-memory",
        official_success=None, terminated=False, truncated=False, task_success=False,
        render_every_actions=2, high_cadence_actions=128, low_execute_actions=16,
        full_max_steps=episode["max_steps"], wall_budget_seconds=episode["wall_budget_seconds"],
        diagnostic_observable_snapshots=[], diagnostic_pose_policy_input=False,
        simulator_thread="main", policy_transport="dedicated_asyncio_thread_v1",
        low_num_obs_steps=6, physical_evidence_status='not_started', physical_evidence_rows=0,
        physical_diagnostics_policy_input=False, physical_diagnostics_training_admissible=False)

    def close_policy():
        nonlocal policy_closed
        if policy_loop is not None and not policy_closed:
            policy_closed = True
            try:
                if controller is not None:
                    policy_loop.call(controller.close(), timeout=30)
            except Exception:
                result["policy_close_error"] = traceback.format_exc()
            finally:
                policy_loop.close()

    def save():
        nonlocal video, trace, pose_trace, physical_trace
        if video is not None:
            closing = video
            video = None
            try:
                closing.close()
            except Exception:
                result["video_close_error"] = traceback.format_exc()
        if trace is not None:
            trace.close()
            trace = None
        if pose_trace is not None:
            pose_trace.close()
            pose_trace = None
        if physical_trace is not None:
            physical_trace.close()
            physical_trace = None
        np.save(args.output/"consumed_actions23.npy", np.asarray(actions, dtype=np.float32).reshape(-1,23))
        result["consumed_actions"] = len(actions)
        result["elapsed_seconds"] = time.monotonic()-started
        if controller is not None:
            result["high_calls"] = sum(event["new_high_proposal"] for event in controller.events)
            (args.output/"controller_events.json").write_text(json.dumps(json_value(controller.events), indent=2)+"\n")
            if controller.history.entries:
                request = controller.history.request_observation()
                np.savez_compressed(args.output/"last_planner_observation.npz",
                    **{f"image_{key}": value for key, value in request["images"].items()},
                    **{f"state_{key}": value for key, value in request["state"].items()})
                (args.output/"last_planner_observation_meta.json").write_text(json.dumps(
                    {key: value for key, value in request.items() if key not in {"images", "state"}})+"\n")
        (args.output/"result.json").write_text(json.dumps(json_value(result), indent=2)+"\n")

    @contextmanager
    def official_session():
        imports = _lazy_official_imports(og_root=DEFAULT_OG_ROOT, eval_root=DEFAULT_EVAL_ROOT, gpu=args.gpu)
        imports.gm.HEADLESS = True
        if int(imports.seed_everything(episode["seed"])) != episode["seed"]:
            raise ValueError("official environment seed differs from manifest")
        cfg = imports.OmegaConf.create({
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.RGBDFullResWrapper"},
            "policy_name": "native_" + manifest["variant"],
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True, "partial_scene_load": True, "max_steps": episode["max_steps"],
            "write_video": False, "mode": episode["mode"], "seed": episode["seed"],
            "task": {"name": episode["task_name"]},
            "robot": imports.OmegaConf.load(manifest["robot_config_path"]),
        })
        evaluator = imports.Evaluator(cfg)
        evaluator.__enter__()
        try:
            evaluator.reset()
            evaluator.load_task_instance(episode["instance_id"])
            evaluator.reset()
            yield evaluator
        except BaseException:
            result.update(status="failed", traceback=traceback.format_exc())
            raise
        finally:
            # Kit shutdown can terminate Python: save before __exit__, not
            # only in an outer finally that might never execute.
            close_policy()
            save()
            evaluator.__exit__(*sys.exc_info())

    def run():
        nonlocal controller, video, trace, pose_trace, policy_loop, physical_trace, physical_observer
        from autonomous_physical_observer import AutonomousPhysicalObserver, load_reviewed_oracle
        physical_observer=AutonomousPhysicalObserver(load_reviewed_oracle(manifest['physical_oracle_path']))
        policy_loop = BackgroundPolicyLoop()
        controller = policy_loop.call(NativeABController.connect(high_uri=args.high_uri, low_uri=args.low_uri,
            task_name=instructions[episode["task_index"]], policy_seed=manifest["policy_seed"],
            expected_high_sha=manifest["high_checkpoint_sha256"], expected_low_sha=manifest["low_checkpoint_sha256"],
            expected_lineage_sha=manifest["high_phase_lineage_sha256"]), timeout=180)
        result["high_identity"] = controller.high_identity
        result["low_identity"] = controller.low_identity
        video = imageio.get_writer(str(args.output/"rollout.mp4"), fps=15, codec="libx264", quality=7, macro_block_size=2)
        trace = (args.output/"official_step_trace.jsonl").open("w", buffering=1)
        pose_trace = (args.output/"diagnostic_robot_pose.jsonl").open("w", buffering=1)
        physical_trace = (args.output/'diagnostic_physical_evidence.jsonl').open('x',buffering=1)
        def record_physical(env, *, event=None):
            nonlocal physical_observer
            if physical_observer is None:
                return
            try:
                if event is not None:
                    row=physical_observer.install(env,event['active_subgoal'],consumed_actions=len(actions))
                    row={**row,'held_objects':physical_observer.held_objects(env),'record_phase':'chunk_admission'}
                else:
                    row={**physical_observer.observe(env,consumed_actions=len(actions)),
                        'held_objects':physical_observer.held_objects(env),'record_phase':'actual_physics_step'}
                physical_trace.write(json.dumps(json_value(row))+'\n')
                result['physical_evidence_rows']+=1
                result['physical_evidence_status']='available'
            except Exception:
                # A diagnostic failure must not alter the policy's actions or
                # fabricate outcomes. Preserve the error and explicit gap.
                result['physical_evidence_status']='unavailable_after_error'
                result['physical_evidence_error']=traceback.format_exc()
                physical_trace.write(json.dumps(dict(consumed_actions=len(actions),
                    diagnostic_error=result['physical_evidence_error'],policy_input=False,
                    training_admissible=False))+'\n')
                physical_observer=None
        with official_session() as evaluator:
            import omnigibson as og
            result["status"] = "running"
            obs = behavior_obs_to_native_low(evaluator.obs, instructions)
            video.append_data(np.moveaxis(obs["images"]["head_rgb"], 0, -1))
            try:
                while len(actions) < episode["max_steps"]:
                    if time.monotonic()-started >= episode["wall_budget_seconds"]:
                        result.update(status="incomplete_wall_budget", termination_reason="wall_budget")
                        break
                    # Read-only diagnostics never enter either model request.
                    # Pose disambiguates commanded spin from actual blocked
                    # motion. Wrist/head snapshots preserve failure context.
                    pose = dict(consumed_actions=len(actions), policy_input=False)
                    try:
                        position, orientation = evaluator.env.robots[0].get_position_orientation()
                        pose.update(position=json_value(position), quaternion_xyzw=json_value(orientation))
                    except Exception as error:
                        pose["unavailable"] = str(error)
                    pose_trace.write(json.dumps(pose)+"\n")
                    if len(actions) % 128 == 0:
                        folder = args.output/"observable"
                        folder.mkdir(exist_ok=True)
                        snapshot = folder/f"f{len(actions):08d}.npz"
                        np.savez_compressed(snapshot, consumed_actions=np.asarray([len(actions)], dtype=np.int64),
                            **{f"image_{key}":value for key,value in obs["images"].items()},
                            **{f"state_{key}":value for key,value in obs["state"].items()})
                        result["diagnostic_observable_snapshots"].append(dict(consumed_actions=len(actions),
                            path=str(snapshot), sha256=sha(snapshot), source="actual_pre_request_observation"))
                    prediction, event = policy_loop.call(controller.actions(obs, consumed_actions=len(actions),
                        execute_steps=min(16, episode["max_steps"]-len(actions))))
                    (args.output/"last_controller_event.json").write_text(json.dumps(json_value(event), indent=2)+"\n")
                    record_physical(evaluator.env,event=event)
                    for action in prediction:
                        frame = len(actions)+1
                        render = frame % 2 == 0 or frame == episode["max_steps"]
                        with og.sim.render_on_step(render):
                            raw_obs, reward, terminated, truncated, info = evaluator.env.step(action, n_render_iterations=1)
                        # Same observation refresh and metric callback as the
                        # reviewed official runner; no task predicate rewrite.
                        raw_obs = evaluator._sync_lights_and_get_obs(raw_obs)
                        evaluator.obs = evaluator._preprocess_obs(raw_obs)
                        evaluator.robot_action = action
                        for metric in evaluator.metrics:
                            metric.step(evaluator.env, action, raw_obs, 0.0, terminated, truncated, info)
                        actions.append(action.copy())
                        record_physical(evaluator.env)
                        info_json = json_value(info)
                        terminal_success = info_json.get("done", {}).get("success")
                        if type(terminal_success) is bool:
                            result["official_success"] = terminal_success
                        result.update(terminated=bool(terminated), truncated=bool(truncated), last_official_info=info_json)
                        trace.write(json.dumps({"frame": frame, "terminated": bool(terminated),
                            "truncated": bool(truncated), "action": action.tolist(), "info": info_json})+"\n")
                        if render:
                            obs = behavior_obs_to_native_low(evaluator.obs, instructions)
                            video.append_data(np.moveaxis(obs["images"]["head_rgb"], 0, -1))
                        if terminated or truncated:
                            evaluator.n_trials += 1
                            if terminal_success is True:
                                evaluator.n_success_trials += 1
                            result.update(status="complete", task_success=terminal_success is True,
                                termination_reason="official_success" if terminal_success is True else "official_done_without_success")
                            break
                    if result["terminated"] or result["truncated"]:
                        break
                    if len(actions) % 128 == 0:
                        print(json.dumps({"task": episode["task_name"], "actions": len(actions),
                            "seconds": time.monotonic()-started, "official_success": result["official_success"]}), flush=True)
                else:
                    result.update(status="complete", task_success=result["official_success"] is True,
                                  termination_reason="full_official_max_steps")
            except BaseException:
                result.update(status="failed", termination_reason="controller_or_simulator_error", traceback=traceback.format_exc())
                raise
            finally:
                close_policy()
                save()
    try:
        run()
    except BaseException:
        result.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        close_policy()
        save()


if __name__ == "__main__":
    main()

