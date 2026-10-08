"""Single collector per rank; synchronous global updates; no shared memory slots."""
# ruff: noqa: E402 -- pin dependency paths before importing runtime modules.
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import traceback

from bootstrap import bootstrap
bootstrap()

from a4_wire import packb, unpackb
from checkpoint_io import atomic_json
from shared_engine import SharedStage1Engine
from synchronous import Collective
import torch
import websockets


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.job / "manifest.json").read_text())
    args.run.mkdir(parents=True, exist_ok=True)
    state = dict(status="loading", pid=os.getpid(), rank=args.rank, started=time.time(), optimizer_updates=0)

    def record(status=None, **fields):
        if status is not None:
            state["status"] = status
        state.update(fields, updated=time.time())
        atomic_json(args.run / "policy_status.json", state)

    record()
    engine = await asyncio.to_thread(SharedStage1Engine, args.run / "checkpoints")
    # Loading admission MUST be released before any all-rank collective.
    (args.run / "model.loaded").touch()
    record("joining_ranks")
    collective = await asyncio.to_thread(Collective.initialize, args.rank,
        manifest["world_size"], manifest["distributed_endpoint"], timeout=1800)
    await asyncio.to_thread(engine.configure_shared, collective, manifest)
    gate = asyncio.Lock()
    stopped = asyncio.Event()
    owner = None

    def stopped_reason():
        if (args.job / "ABORT").exists():
            return "global_abort"
        if (args.job / "STOP_TRAINING").exists():
            return "user_or_monitor_stop"
        if time.time() >= manifest["absolute_deadline"] - 300:
            return "training_budget"
        if engine.trainer.update_count >= manifest["training"]["max_updates"]:
            return "update_budget"
        return None

    def dispatch(request):
        kind = request["kind"]
        if kind == "begin":
            from shared_reward_math import CONTROL_GAMMA
            if request.get("reward_protocol") != engine.reward_protocol or request.get("control_gamma") != CONTROL_GAMMA:
                raise ValueError("Collector reward/discount protocol mismatch")
            engine.begin(request["task"], request["num_envs"], request["seed"], request["episode_metadata"])
            return {}
        if kind == "infer":
            return engine.infer(request["observations"], request["indices"])
        if kind == "value":
            return {"values": engine.value(request["observations"], request["indices"])}
        if kind != "update":
            raise ValueError("Unknown request: " + str(kind))
        report = engine.trainer.update(engine, request["experience_ids"], request["advantages"], request["returns"])
        record("shared_round_committed", optimizer_updates=report["update"], last_update=report,
               gpu_peak_GiB=torch.cuda.max_memory_allocated() / 2**30)
        pause = manifest["training"]["acceptance_pause_updates"]
        if report["update"] == pause and not (args.job / "ACCEPTED").exists():
            record("awaiting_acceptance", optimizer_updates=report["update"])
            pause_deadline = time.time() + 1800
            while not (args.job / "ACCEPTED").exists() and not stopped_reason():
                if time.time() > pause_deadline:
                    (args.job / "STOP_TRAINING").touch()
                    break
                time.sleep(1)
        reasons = collective.objects(stopped_reason())
        reason = next((r for r in reasons if r), None)
        if reason:
            # Rank zero alone writes, but ALL ranks wait for successful publish.
            error, receipt = None, None
            try:
                if args.rank == 0:
                    receipt = engine.trainer._save_checkpoint({"reason": reason})
            except Exception as exception:
                error = exception
            collective.check(error, "final checkpoint")
            receipt = collective.objects(receipt)[0]
            atomic_json(args.run / "save_ack.json", receipt)
            report["stop_after_update"] = reason
        return report

    async def handler(socket):
        nonlocal owner
        if owner is not None:
            await socket.close(code=1013, reason="One collector per rank")
            return
        owner = socket
        await socket.send(packb({"kind": "stage1_rl_v1", "shared": True, "rank": args.rank}))
        try:
            async for payload in socket:
                request = unpackb(payload)
                async with gate:
                    record("updating" if request["kind"] == "update" else "collecting",
                           request_kind=request["kind"], optimizer_updates=engine.trainer.update_count)
                    result = await asyncio.to_thread(dispatch, request)
                    record("training", optimizer_updates=engine.trainer.update_count,
                           inference_requests=engine.requests)
                await socket.send(packb({"ok": True, "response": result}))
                if result.get("stop_after_update"):
                    record("finished", stop_reason=result["stop_after_update"])
                    stopped.set()
                    return
        except websockets.ConnectionClosed:
            # Normal task completion reconnects with a new begin. Disconnect
            # with unconsumed samples is a collection failure, not a reset.
            if engine.trainer.experiences:
                record("failed", error="Collector disconnected with unconsumed rollout")
                (args.job / "ABORT").touch()
                stopped.set()
        except Exception as error:
            record("failed", error=repr(error))
            (args.job / "ABORT").touch()
            traceback.print_exc()
            try:
                await socket.send(packb({"ok": False, "error": repr(error)}))
            except websockets.ConnectionClosed:
                pass
            stopped.set()
        finally:
            owner = None

    async def watcher():
        while not stopped.is_set():
            if (args.job / "ABORT").exists():
                record("aborted", error="Global abort; resume only published shared checkpoint")
                stopped.set()
                return
            await asyncio.sleep(1)

    port = manifest["port_base"] + args.rank
    async with websockets.serve(handler, "127.0.0.1", port, compression=None,
                               max_size=512 << 20, ping_timeout=None, close_timeout=10):
        record("ready", optimizer_updates=engine.trainer.update_count)
        (args.run / "policy.ready").touch()
        watch = asyncio.create_task(watcher())
        await stopped.wait()
        watch.cancel()


if __name__ == "__main__":
    asyncio.run(main())
