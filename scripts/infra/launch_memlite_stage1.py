"""Budgeted single-node launcher. A stopped run resumes only by explicit request.

Use from a clean frozen worktree with the shared training environment active.
The supervisor ledger, unlike a model checkpoint, accounts for failed work
and initialization as well as successful optimizer steps. No automatic retry.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml

from g05.utils.training.stage1_runtime import atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component", choices=("high", "low"), required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/memlite_stage1/stage1.yaml"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preflight-stop-step", type=int)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean frozen Git worktree")
    cfg = yaml.safe_load(args.config.read_text())
    component = cfg[args.component]
    if component["node_ip"] not in subprocess.check_output(["hostname", "-I"], text=True).split():
        raise RuntimeError("Wrong node: high=lc1, low=lc2")
    if args.preflight_stop_step is not None and not 1 <= args.preflight_stop_step <= 64:
        raise ValueError("Preflight admits at most 64 total updates, not 64 more on resume")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
        raise RuntimeError("GPU is occupied; do not stop or compete with teammate processes")
    args.output = args.output.resolve()
    # Keep supervision separate: the trainer creates its own output atomically.
    control = args.output.with_name(args.output.name + ".supervisor")
    control.mkdir(parents=True, exist_ok=args.resume)
    lock = (control / "lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger_path = control / "ledger.json"
    limit = 3600. if args.preflight_stop_step is not None else component["max_wall_hours"] * 3600.
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.resume:
        ledger = json.loads(ledger_path.read_text())
        if ledger["status"] == "RUNNING":
            raise RuntimeError("Unclosed supervisor ledger: reconcile prior PID/time before resuming; never reset the budget")
        if (ledger["limit_seconds"] != limit or ledger["component"] != args.component
                or ledger["commit"] != commit):
            raise ValueError("Supervisor recipe identity changed")
        previous = ledger["consumed_seconds"]
        attempt = ledger["attempt"] + 1
    else:
        previous, attempt = 0., 1
    if previous + cfg["save_reserve_seconds"] >= limit:
        raise RuntimeError("Cumulative budget exhausted")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(repo / "src"), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
               NCCL_IB_DISABLE="1", NCCL_P2P_DISABLE="1", NCCL_SOCKET_IFNAME="bond0",
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.monotonic()
    deadline = started + limit - previous
    env["STAGE1_SUPERVISOR_DEADLINE"] = str(deadline)
    env["STAGE1_SUPERVISOR_LIMIT"] = str(limit)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
               "--max_restarts=0", "scripts/train_memlite_stage1.py", "--config", str(args.config.resolve()),
               "--component", args.component, "--output", str(args.output)]
    if args.resume:
        command.append("--resume")
    if args.preflight_stop_step is not None:
        command.extend(["--preflight-stop-step", str(args.preflight_stop_step)])
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    with (control / f"attempt_{attempt:03d}.log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ledger = dict(component=args.component, commit=commit, attempt=attempt, limit_seconds=limit,
                      supervisor_pid=os.getpid(), torchrun_pid=child.pid, output=str(args.output))
        stop_started = None
        try:
            while child.poll() is None:
                now = time.monotonic()
                ledger.update(status="RUNNING", consumed_seconds=previous+now-started,
                              heartbeat_utc_seconds=time.time())
                atomic_json(ledger_path, ledger)
                if stopping[0] or now >= deadline:
                    if stop_started is None:
                        stop_started = now
                        os.killpg(child.pid, signal.SIGTERM)
                    elif now - stop_started > 30:
                        os.killpg(child.pid, signal.SIGKILL)
                time.sleep(2)
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            ledger.update(status="EXITED", returncode=child.returncode,
                          consumed_seconds=previous+time.monotonic()-started, heartbeat_utc_seconds=time.time())
            atomic_json(ledger_path, ledger)
    print(json.dumps(ledger), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
