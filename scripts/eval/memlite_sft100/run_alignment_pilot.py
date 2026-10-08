"""Run one guarded 3-environment public-test alignment pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


SOURCE = Path(__file__).resolve().parent
CONTROLLER = Path("/run/ti/BEHAVIOR2026/eval20x3/selected_eval_task.py")
SIM_ROOT = Path("/run/ti/behavior_stage3_20260930")
G05 = Path("/run/ti/rl_memlite_stage1_20261006/code/g05_sft_6af1ab9")
MODEL_PYTHON = Path("/home/tione/notebook/baselines/GalaxeaVLA/.venv/bin/python")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def stop_child(child, timeout=45):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)


def wait_for(path, process, timeout, label):
    deadline = time.time() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited before publishing {path}")
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {label}: {path}")
        time.sleep(2)


def validate(job, task, worker):
    task_output = job / "tasks" / task
    status = json.loads((task_output / "status.json").read_text())
    attempt = json.loads((task_output / "attempts/batch_00.json").read_text())
    ack = json.loads((worker / "evaluation_ack.json").read_text())
    if status.get("status") != "completed" or attempt.get("status") != "completed":
        raise RuntimeError("Official evaluator did not complete")
    if ack.get("weights_unchanged") is not True or ack.get("optimizer_steps") != 0:
        raise RuntimeError("Model weights changed during trace pilot")
    metrics = sorted((task_output / "json").glob("*.json"))
    videos = sorted((task_output / "videos").glob("*.mp4"))
    trace_root = worker / "alignment_trace" / task
    traces = sorted(trace_root.glob("*/chunks.jsonl"))
    if len(metrics) != 3 or len(videos) != 3 or len(traces) != 3:
        raise RuntimeError("Expected three metrics, videos, and per-instance traces")
    trace_counts = {}
    for trace in traces:
        rows = [json.loads(line) for line in trace.read_text().splitlines() if line]
        if not rows or [row["chunk_id"] for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"Non-contiguous trace: {trace}")
        if any(row["task"] != task or row["instance_id"] != int(trace.parent.name) for row in rows):
            raise RuntimeError(f"Trace identity mixing: {trace}")
        if any(row["high_level"]["context_id"] != row["low_level"]["context_id"] for row in rows):
            raise RuntimeError(f"High/low context mismatch: {trace}")
        trace_counts[trace.parent.name] = len(rows)
    return {
        "metrics": [str(path) for path in metrics],
        "videos": [str(path) for path in videos],
        "traces": [str(path) for path in traces],
        "trace_chunks": trace_counts,
        "weights_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--task", choices=["turning_on_radio", "picking_up_trash"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    job = args.job.resolve()
    job.mkdir(parents=True, exist_ok=True)
    (job / "tasks").mkdir(exist_ok=True)
    (job / "workers").mkdir(exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    worker = job / "workers" / args.task
    worker.mkdir(exist_ok=False)
    state = {"status": "starting", "task": args.task, "gpu": args.gpu,
             "port": args.port, "pid": os.getpid(), "started": time.time()}
    atomic_json(worker / "pilot_status.json", state)

    env = os.environ.copy()
    for key in list(env):
        if key.startswith("RL_"):
            env.pop(key)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=f"{G05 / 'src'}:{SOURCE}",
    )
    policy = None
    simulator = None
    policy_log = (job / "logs" / f"{args.task}.policy.log").open("x")
    simulator_log = (job / "logs" / f"{args.task}.sim.log").open("x")
    try:
        policy = subprocess.Popen(
            [str(MODEL_PYTHON), str(SOURCE / "serve.py"), "--run", str(worker),
             "--port", str(args.port), "--inference-mode", "batch",
             "--capture-alignment-trace"],
            env=env, stdin=subprocess.DEVNULL, stdout=policy_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state.update(status="loading_policy", policy_pid=policy.pid)
        atomic_json(worker / "pilot_status.json", state)
        wait_for(worker / "policy.ready", policy, 1200, "policy")

        state.update(status="running_simulator", policy_ready=time.time())
        sim_env = env | {"EVAL_GPU": str(args.gpu), "EVAL_SOURCE": str(SOURCE)}
        simulator = subprocess.Popen(
            ["bash", str(SOURCE / "launch_sim.sh"), str(CONTROLLER), "--task", args.task,
             "--output", str(job / "tasks" / args.task), "--policy-run", str(worker),
             "--port", str(args.port)],
            env=sim_env, stdin=subprocess.DEVNULL, stdout=simulator_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state.update(simulator_pid=simulator.pid)
        atomic_json(worker / "pilot_status.json", state)
        simulator_code = simulator.wait(timeout=10800)
        if simulator_code != 0:
            raise RuntimeError(f"Simulator exited {simulator_code}")

        (worker / "STOP").touch()
        policy_code = policy.wait(timeout=900)
        if policy_code != 0:
            raise RuntimeError(f"Policy exited {policy_code}")
        result = validate(job, args.task, worker)
        state.update(status="complete", finished=time.time(), result=result)
        atomic_json(worker / "pilot_status.json", state)
    except Exception as error:
        state.update(status="failed", failed=time.time(), error=repr(error))
        atomic_json(worker / "pilot_status.json", state)
        raise
    finally:
        stop_child(simulator)
        if policy is not None and policy.poll() is None:
            (worker / "STOP").touch()
            try:
                policy.wait(timeout=900)
            except subprocess.TimeoutExpired:
                stop_child(policy)
        policy_log.close()
        simulator_log.close()


if __name__ == "__main__":
    main()
