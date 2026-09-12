"""Collect real, train-instance rotation recovery with a checked geometric teacher.

Privileged pose/target coordinates are teacher/quality-check inputs ONLY. Model
records contain the official RGB/proprio/task inputs, causal text and actions.
Unsafe or incomplete corrections are retained for diagnosis but never accepted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


SOURCE_TEXT = Path(__file__).read_text()
SOURCE_SHA256 = hashlib.sha256(SOURCE_TEXT.encode()).hexdigest()


TASKS = ["turning_on_radio", "picking_up_trash", "putting_away_Halloween_decorations",
         "cleaning_up_plates_and_food", "can_meat"]
TARGETS = ["radio_89", "trash_can_116", "bottom_cabinet_rhdbzv_0", "fridge_petcxr_0", "top_cabinet_lkxmne_2"]
FEEDBACK = "Repeated rotation without approach progress; pause and recover toward the current target."


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--task", type=int, choices=range(5), required=True)
    ap.add_argument("--instances", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=221917)
    ap.add_argument("--source-index", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "eval"), default="train")
    args = ap.parse_args()
    # Demo episode numbers are NOT scene instance numbers. Verify against the
    # original metadata-derived split, including gaps and IDs above 200.
    source_index = json.loads(args.source_index.read_text())
    allowed = set(source_index[f"{args.split}_source_instances"][str(args.task)])
    if not set(args.instances) <= allowed or len(set(args.instances)) != len(args.instances):
        raise ValueError("Recovery source instances do not belong to the declared original split")
    if any(i < 0 or i > 300 for i in args.instances):
        raise ValueError("Public/hidden-test instances cannot enter recovery collection")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "collector_source.py").write_text(SOURCE_TEXT)
    sys.path.insert(0, "/mnt/sdc1/robodojo/behavior_eval")
    import run_behavior_eval_chunked as runner
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import Evaluator
    from omnigibson.eval.utils.eval_utils import seed_everything
    from omnigibson.macros import gm
    gm.HEADLESS = True
    seed_everything(args.seed)
    cfg = OmegaConf.create({
        "env_wrapper": {"_target_": "omnigibson.eval.wrappers.RGBDFullResWrapper"},
        "policy_name": "memlite_train_recovery_teacher_v1",
        "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
        "headless": True, "partial_scene_load": True, "max_steps": None,
        "write_video": False, "mode": "train", "seed": args.seed,
        "task": {"name": TASKS[args.task]},
        "robot": OmegaConf.load(str(Path(runner.og.__file__).parent / "eval/r1pro.yaml"))})
    accepted = []
    with Evaluator(cfg) as evaluator:
        for instance in args.instances:
            out = args.output_dir / f"task{args.task}_instance{instance:03d}"
            out.mkdir()
            evaluator.reset()
            evaluator.load_task_instance(instance)
            evaluator.reset()
            robot = evaluator.env.env.robots[0]
            target_name = TARGETS[args.task]
            target = evaluator.env.scene.object_registry("name", target_name)
            if target is None:
                raise ValueError(f"Target {target_name} absent from loaded training scene")
            # Full initial simulator state permits reproducible closed-loop checks.
            torch.save(runner.og.sim.dump_state(serialized=False), out / "initial_sim_state.pt")

            def npvalue(value):
                return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()

            def state():
                return npvalue(next(v for k, v in evaluator.obs.items() if k.endswith("::proprio")))

            def geometry():
                position, quat = robot.get_position_orientation()
                p, q = npvalue(position), npvalue(quat)
                delta = npvalue(target.get_position_orientation()[0])[:2] - p[:2]
                x, y, z, w = q
                yaw = math.atan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))
                return float(np.linalg.norm(delta)), wrap(math.atan2(delta[1], delta[0]) - yaw), p, yaw

            initial = state()
            assert initial.shape == (61,)
            hold = np.zeros(23, dtype=np.float32)
            hold[3:7], hold[7:14], hold[15:22] = initial[53:57], initial[3:10], initial[28:35]
            hold[14] = 1 if initial[24] > .025 else -1
            hold[22] = 1 if initial[49] > .025 else -1
            actions, states, rows, frames = [], [], [], []
            stage, stage_start = "perturb", 0
            sign = 1 if instance % 2 == 0 else -1
            initial_distance, _, initial_position, _ = geometry()
            start = time.monotonic()
            abort, brake_velocity, failure_error = None, None, None
            recovery_start = 800 + (instance % 4) * 32
            # Conservative radial approach; collision/stall checks below reject
            # trajectories where the direct local teacher is not reliable.
            try:
                extent = npvalue(target.aabb_extent)
                # The base must stop outside supporting furniture, not drive
                # up to the small target's own footprint (e.g. a radio on a
                # coffee table). This is a conservative approach waypoint;
                # manipulation remains the original policy's next subtask.
                standoff = max(1.15, min(1.5, float(np.linalg.norm(extent[:2])) / 2 + .60))
                if float(np.linalg.norm(extent[:2])) > .9:
                    standoff = max(standoff, 1.5)
            except (AttributeError, TypeError):
                standoff = 1.15
            settle_count = 0
            for step in range(recovery_start + 1800):
                s = state()
                distance, error, position, yaw = geometry()
                if step == recovery_start:
                    stage, stage_start, failure_error = "brake", step, abs(error)
                if stage == "brake" and step - stage_start >= 32:
                    brake_velocity = float(abs(s[2]))
                    if brake_velocity > .05:
                        abort = "failed_to_brake"
                        break
                    stage, stage_start = "align", step
                if stage == "align" and abs(error) < .06 and abs(s[2]) < .04:
                    stage, stage_start = "approach", step
                if stage == "approach" and distance <= standoff:
                    stage, stage_start = "settle", step
                if stage == "settle":
                    settle_count += 1
                    if settle_count >= 96:
                        break
                action = hold.copy()
                if stage == "perturb":
                    action[2] = sign * .28
                elif stage == "align":
                    action[2] = float(np.clip(.7 * error, -.24, .24)) if abs(error) > .025 else 0.
                elif stage == "approach":
                    action[2] = float(np.clip(.5 * error, -.15, .15)) if abs(error) > .035 else 0.
                    if abs(error) < .20:
                        action[0] = float(np.clip((distance - standoff) * .6, .025, .18))
                if not np.isfinite(action).all() or not np.isfinite(s).all():
                    abort = "nonfinite"
                    break
                if np.max(np.abs(s[np.r_[3:10, 28:35]] - initial[np.r_[3:10, 28:35]])) > .35:
                    abort = "arm_hold_deviation_or_collision"
                    break
                if stage == "approach" and step - stage_start > 180 and len(rows) > 90:
                    if rows[-90]["distance"] - distance < .015:
                        abort = "approach_stall_or_collision"
                        break
                if abs(position[2] - initial_position[2]) > .10:
                    abort = "base_height_unstable"
                    break
                # Observation at t is stored BEFORE executing action[t].
                states.append(s)
                actions.append(action)
                rows.append({"step": step, "stage": stage, "distance": distance,
                             "heading_error": error, "position": position.tolist(), "yaw": yaw})
                if step % 16 == 0:
                    official = {k: npvalue(v) for k, v in evaluator.obs.items()
                                if k == "task_id" or k.endswith("::rgb") or k.endswith("::proprio")}
                    assert len(official) == 5
                    filename = f"obs_{step:05d}.npz"
                    np.savez_compressed(out / filename, **official)
                    frames.append({"step": step, "file": filename, "stage": stage})
                a = torch.from_numpy(action)
                with runner.og.sim.render_on_step(step % 16 == 15):
                    obs, _, terminated, truncated, info = evaluator.env.step(a, n_render_iterations=1)
                evaluator.obs = evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
                if terminated or truncated:
                    abort = f"task_terminated_before_recovery_check:{info.get('done')}"
                    break
                if step % 256 == 0:
                    print(json.dumps({"task": args.task, "instance": instance, "step": step,
                                      "stage": stage, "distance": distance, "heading_error": error}), flush=True)
            final_distance, final_error, _, _ = geometry()
            final_velocity = float(np.max(np.abs(state()[:3])))
            yaw_path = np.unwrap([r["yaw"] for r in rows[:recovery_start]])
            perturb_rotation = float(abs(yaw_path[-1] - yaw_path[0])) if len(yaw_path) else 0.
            success = (abort is None and stage == "settle" and settle_count >= 96
                       and final_distance <= standoff + .03 and abs(final_error) < .10
                       and final_velocity < .04 and perturb_rotation > 6.2
                       and brake_velocity is not None and brake_velocity <= .05)
            np.savez_compressed(out / "trajectory.npz", states=np.asarray(states), actions=np.asarray(actions))
            report = {"schema_version": 1, "task_id": args.task, "task": TASKS[args.task],
                "instance_id": instance, "mode": "train", "source_split": args.split, "seed": args.seed,
                "target": target_name, "feedback": FEEDBACK, "recovery_start": recovery_start,
                "accepted": bool(success), "abort": abort, "frames": frames, "geometry_diagnostic_only": rows,
                "quality": {"initial_distance": initial_distance, "final_distance": final_distance,
                    "final_heading_error": final_error, "final_velocity": final_velocity,
                    "brake_velocity": brake_velocity, "perturb_rotation_rad": perturb_rotation,
                    "failure_heading_error": failure_error, "standoff": standoff},
                "seconds": time.monotonic()-start,
                "scope": "Local rotation-stop, target reorientation and approach recovery; not manipulation/task success.",
                "teacher": "bounded geometric feedback with measured brake/heading/approach/stall checks",
                "script_sha256": SOURCE_SHA256,
                "execution_source": str(args.output_dir / "collector_source.py")}
            (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
            accepted.append({"path": str(out), "accepted": bool(success), "quality": report["quality"], "abort": abort})
            print(json.dumps(accepted[-1]), flush=True)
    (args.output_dir / "collection.json").write_text(json.dumps({"complete": True, "trajectories": accepted}, indent=2) + "\n")


if __name__ == "__main__":
    main()
