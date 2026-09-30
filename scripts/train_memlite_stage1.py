"""Single-node eight-A800 trainer: independent high CE or low SkillFM.

This is deliberately separate from the legacy finetune entry. No task-random
replacement, repeat-filled tail, partial restore, mean-of-means DDP loss, or
implicit reset of the low-level 120-hour budget is permitted.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml

from g05.data.memlite_stage1_dataset import Stage1Dataset, collate_stage1, to_device, worker_init
from g05.utils.training.stage1_model import configuration, restore_model, optimizer_groups
from g05.utils.training.stage1_sampling import MixedTaskPass, CommittedBatchSampler
from g05.utils.training.stage1_runtime import (
    sha256, atomic_json, capture_rng, restore_rng, normalize_ddp_gradients,
    WallBudget, lr_multiplier, save_checkpoint, load_checkpoint, init_wandb,
)


def validate_release(release, preflight):
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["format_version"] != "memlite-stage1-data-v1":
        raise ValueError("Unsupported data release")
    for name, checksum in manifest["files"].items():
        if Path(name).name != name or sha256(release / name) != checksum:
            raise ValueError("Compact release file checksum mismatch: " + name)
    if not preflight:
        accepted = json.loads((release / "acceptance.json").read_text())
        if accepted.get("status") != "ACCEPTED" or accepted["manifest_sha256"] != sha256(manifest_path):
            raise ValueError("Data clock/human/loader acceptance is missing or stale")
        for gate in ("source_identity", "protected_split", "clock_alignment", "human_review", "normalizer", "all_task_loader"):
            if accepted.get("gates", {}).get(gate) is not True:
                raise ValueError("Unaccepted data gate: " + gate)
    return manifest


def clear_loss_cache(model):
    # A detached full-vocabulary logits cache otherwise survives until the
    # NEXT forward, unnecessarily overlapping two large CE allocations.
    model.model.ar_helper._last_ce_cache = None


def evaluate(model, dataset, rank, world, device, config, step, *, limit=None):
    saved_rng = capture_rng()
    was_training = model.training
    indices = np.load(dataset.release / "fixed_eval_indices.npy")
    if limit is not None:
        # Still cover all 100 tasks in a short gate, not only the first tasks
        # in the task-major fixed validation table.
        indices = indices.reshape(100, 32)[:, :max(1, limit // 100)].reshape(-1)
    own = indices[rank::world].tolist()
    micro = 4 if config["component"] == "high" else 32
    batches = [own[i:i + micro] for i in range(0, len(own), micro)]
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=config["workers_per_rank"],
        collate_fn=collate_stage1, worker_init_fn=worker_init, multiprocessing_context="spawn",
        prefetch_factor=config["prefetch_factor"], generator=torch.Generator().manual_seed(109 + rank))
    totals = torch.zeros(100, 3, device=device, dtype=torch.float64)
    model.eval()
    try:
        # Fixed FM time/noise across checkpoints; eval restores training RNG.
        torch.manual_seed(9183 + rank)
        random.seed(9183 + rank)
        np.random.seed(9183 + rank)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in loader:
                loss, metrics = model(to_device(batch, device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite evaluation objective")
                ids = torch.tensor([r["task"] for r in batch["source_identity"]], device=device)
                values = torch.stack((metrics["row_loss_numerator"].double(),
                                      metrics["row_loss_denominator"].double(), torch.ones(len(ids), device=device)), dim=1)
                totals.index_add_(0, ids, values)
                clear_loss_cache(model)
        dist.all_reduce(totals)
    finally:
        model.train(was_training)
        restore_rng(saved_rng)
    values = totals.cpu().tolist()
    task_losses = {str(i): r[0] / r[1] for i, r in enumerate(values) if r[1] > 0}
    denominator = sum(r[1] for r in values)
    return dict(step=step, windows=int(sum(r[2] for r in values)),
                weighted_loss=sum(r[0] for r in values) / denominator,
                macro_task_loss=float(np.mean(list(task_losses.values()))), tasks=task_losses,
                per_task_sufficient_statistics=values, fixed_fm_seed=9183,
                scope="teacher-forced held-out objective; not simulator success rate")


def main():
    launch_started = time.monotonic()
    if "STAGE1_SUPERVISOR_DEADLINE" not in os.environ:
        raise RuntimeError("Launch through scripts/infra/launch_memlite_stage1.py for cumulative failed-job accounting")
    external_deadline = float(os.environ["STAGE1_SUPERVISOR_DEADLINE"])
    external_limit = float(os.environ["STAGE1_SUPERVISOR_LIMIT"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/memlite_stage1/stage1.yaml"))
    parser.add_argument("--component", choices=("high", "low"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-stop-step", type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Freeze a clean committed source before launching")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    full_config = yaml.safe_load(args.config.read_text())
    config = {k: v for k, v in full_config.items() if k not in ("high", "low")}
    config.update(full_config[args.component], component=args.component)
    preflight = args.preflight_stop_step is not None
    if preflight and not 1 <= args.preflight_stop_step <= 64:
        raise ValueError("Preflight is bounded to at most 64 total updates")
    if config["node_ip"] not in subprocess.check_output(["hostname", "-I"], text=True).split():
        raise RuntimeError("Wrong node for this component; lc1=high and lc2=low")
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != 8 or rank != local:
        raise ValueError("One eight-GPU node per component is required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=900))
    own_pids = [None] * world
    dist.all_gather_object(own_pids, os.getpid())
    node_lock = None
    if rank == 0:
        node_lock = (Path(config["root"]) / "runs" / f"stage1-{config['node_ip']}.lock").open("a")
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        foreign = {int(x) for x in active.splitlines() if x.strip().isdigit()} - set(own_pids)
        if foreign:
            raise RuntimeError("Other GPU jobs are present; do not compete with teammate training")
        if args.resume:
            if not args.output.is_dir():
                raise FileNotFoundError("Resume output does not exist")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
        manifest = validate_release(Path(config["release"]), preflight)
    else:
        manifest = None
    box = [manifest]
    dist.broadcast_object_list(box, src=0)
    manifest = box[0]
    model_config = configuration(config["root"], args.component, manifest["task_names"])
    if rank == 0 and sha256(model_config["initial_weights"]) != model_config["initial_weights_sha256"]:
        raise ValueError("Initial weight SHA mismatch")
    dist.barrier()
    recipe = dict(config=config, model=model_config, commit=commit,
                  manifest_sha256=sha256(Path(config["release"]) / "manifest.json"),
                  normalizer_sha256=sha256(model_config["stats_path"]), preflight=preflight)
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    state = dict(step=0, epoch=0, next_update=0, save_sequence=0, consumed_seconds=0.,
                 fingerprint=fingerprint, wandb_run_id=uuid.uuid4().hex[:12] if rank == 0 else None)
    saved = None
    if args.resume:
        saved, receipt = load_checkpoint(args.output / "checkpoints", fingerprint)
        state = saved["state"]
        state["consumed_seconds"] = receipt["consumed_seconds"]
    box = [state]
    dist.broadcast_object_list(box, src=0)
    state = box[0]
    wall = WallBudget(3600 if preflight else config["max_wall_hours"] * 3600,
                      state["consumed_seconds"] + time.monotonic() - launch_started)
    if wall.exhausted(config["save_reserve_seconds"]):
        raise RuntimeError("Cumulative wall budget exhausted before initialization; do not renew")
    torch.manual_seed(config["seed"] + rank)
    random.seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    model, restoration = restore_model(model_config, args.component,
                                       saved["model_state_dict"] if saved else None)
    model.to(device).train()
    groups = optimizer_groups(model, args.component, config)
    base_lrs = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups, betas=tuple(config["betas"]), fused=False)
    if saved:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
    ddp = DDP(model, device_ids=[local], find_unused_parameters=args.component == "low",
              bucket_cap_mb=128, gradient_as_bucket_view=True)
    if saved:
        if len(saved["rng_by_rank"]) != world:
            raise ValueError("Resume world size/RNG mismatch")
        restore_rng(saved["rng_by_rank"][rank])
    del saved
    gc.collect()
    atomic_json(args.output / f"restore_rank{rank}_step{state['step']}.json",
                dict(commit=commit, rank=rank, hostname=socket.gethostname(), **restoration))
    run = None
    if rank == 0:
        wb = dict(config["wandb"], name=f"stage1-{args.component}-{'acceptance' if preflight else 'formal'}")
        run = init_wandb(wb, args.output, run_id=state["wandb_run_id"], resume=args.resume,
            metadata=dict(component=args.component, code_commit=commit, recipe_fingerprint=fingerprint,
                          manifest_sha256=recipe["manifest_sha256"], initialization_sha256=model_config["initial_weights_sha256"],
                          global_batch=256, stride=16, single_frame=True, preflight=preflight,
                          high_passes=1, low_wall_hours=120, normalizer_sha256=recipe["normalizer_sha256"]))
        atomic_json(args.output / "recipe.json", recipe)
        atomic_json(args.output / "wandb.json", dict(id=run.id, url=run.url, entity=run.entity, project=run.project))
    dist.barrier()
    stopping = [False]
    def signal_stop(_signum, _frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    eval_dataset = Stage1Dataset(config["release"], model_config, args.component, "eval")
    parameters = [p for p in model.parameters() if p.requires_grad]
    def accounted_seconds():
        return max(wall.consumed(), external_limit - (external_deadline - time.monotonic()))

    def checkpoint(reason):
        # No live forward/cache references belong in an optimizer checkpoint.
        clear_loss_cache(model)
        local_rng = capture_rng()
        rng_by_rank = [None] * world if rank == 0 else None
        dist.gather_object(local_rng, rng_by_rank, dst=0)
        state["save_sequence"] += 1
        state["consumed_seconds"] = accounted_seconds()
        if rank == 0:
            receipt = save_checkpoint(args.output / "checkpoints", model=model, optimizer=optimizer,
                                      state=dict(state), rng_by_rank=rng_by_rank)
            atomic_json(args.output / "status.json", dict(status=reason, step=state["step"], epoch=state["epoch"],
                next_update=state["next_update"], consumed_seconds=receipt["consumed_seconds"], checkpoint=receipt["path"]))
        dist.barrier()

    def eval_and_log():
        result = evaluate(model, eval_dataset, rank, world, device, config, state["step"], limit=200 if preflight else None)
        if rank == 0:
            atomic_json(args.output / f"eval_{state['step']:08d}.json", result)
            run.log({"train/update": state["step"], "eval/loss": result["weighted_loss"],
                     "eval/macro_task_loss": result["macro_task_loss"],
                     **{f"eval/task_{k}": v for k, v in result["tasks"].items()}})

    if state["step"] == 0:
        eval_and_log()
        checkpoint("INITIALIZED")
    reason = "STOPPED"
    while state["step"] < config["max_updates"]:
        if config["max_passes"] is not None and state["epoch"] >= config["max_passes"]:
            reason = "COMPLETED_ONE_PASS"
            break
        dataset = Stage1Dataset(config["release"], model_config, args.component, "train", epoch=state["epoch"])
        schedule = MixedTaskPass(dataset.task_ids, seed=config["seed"], epoch=state["epoch"],
            micro_batch=config["micro_batch"], accumulation=config["accumulation"])
        if "schedule_sha256" in state and state["next_update"] and state["schedule_sha256"] != schedule.digest:
            raise ValueError("Resume schedule identity mismatch")
        state["schedule_sha256"] = schedule.digest
        horizon = schedule.updates if config["schedule_updates"] == "one_pass" else config["schedule_updates"]
        sampler = CommittedBatchSampler(schedule, rank, state["next_update"])
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["workers_per_rank"],
            collate_fn=collate_stage1, worker_init_fn=worker_init, multiprocessing_context="spawn",
            prefetch_factor=config["prefetch_factor"], pin_memory=True,
            generator=torch.Generator().manual_seed(config["seed"] + rank + state["epoch"] * world))
        iterator = iter(loader)
        epoch_finished = True
        for update in range(state["next_update"], schedule.updates):
            should_stop = (stopping[0] or wall.exhausted(config["save_reserve_seconds"])
                           or time.monotonic() + config["save_reserve_seconds"] >= external_deadline)
            if rank == 0 and state["step"] % 10 == 0:
                active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
                should_stop |= bool({int(x) for x in active.splitlines() if x.strip().isdigit()} - set(own_pids))
            should_stop |= preflight and state["step"] >= args.preflight_stop_step
            should_stop |= state["step"] >= config["max_updates"]
            stop = torch.tensor(int(should_stop), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                reason = "PREFLIGHT_STOPPED" if preflight else "BUDGET_OR_SIGNAL_STOPPED"
                epoch_finished = False
                break
            begin = time.monotonic()
            factor = lr_multiplier(state["step"], config["warmup_updates"], horizon, config["final_lr_ratio"])
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, dtype=torch.float64, device=device)
            for accumulation in range(config["accumulation"]):
                batch = next(iterator)
                expected = schedule.micro_indices(update, accumulation, rank)
                actual = [r["candidate"] for r in batch["source_identity"]]
                if actual != expected or len({r["task"] for r in batch["source_identity"]}) < 2:
                    raise ValueError("Loader changed the committed mixed-task sample identities")
                sync = nullcontext() if accumulation == config["accumulation"] - 1 else ddp.no_sync()
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = ddp(to_device(batch, device))
                    denominator = metrics["loss_denominator"].detach()
                    if not torch.isfinite(loss) or denominator <= 0:
                        raise ValueError("Invalid objective; no optimizer update permitted")
                    if not torch.isclose(loss.detach().float(), metrics["row_loss_numerator"].sum() / denominator,
                                         rtol=1e-4, atol=1e-5):
                        raise ValueError("Per-task statistics do not match the differentiated objective")
                    (loss * denominator).backward()
                totals += torch.stack((loss.detach().double() * denominator,
                                       denominator.double(), torch.tensor(len(actual), device=device)))
                clear_loss_cache(model)
                del loss, metrics, batch
            dist.all_reduce(totals)
            normalize_ddp_gradients(parameters, totals[1].item(), world)
            if state["step"] == (0 if not args.resume else receipt["state"]["step"]):
                groups = model.coordination_trainable_parameter_groups()
                coverage = {k: sum(p.grad is not None for _, p in v) for k, v in groups.items()}
                expected = {"planner_vlm": 326} if args.component == "high" else {"action_expert": 322, "vlm_lora": 182}
                if coverage != expected or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Real first-step trainable/frozen gradient coverage failed: " + str(coverage))
                atomic_json(args.output / f"gradients_rank{rank}_step{state['step']}.json",
                            dict(coverage=coverage, frozen_gradients=0, global_denominator=float(totals[1]),
                                 observed_global_batch=int(totals[2]), statistics_match_objective=True))
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            state["step"] += 1
            state["next_update"] = update + 1  # Commit only after the optimizer succeeded.
            torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            if rank == 0:
                record = dict(step=state["step"], epoch=state["epoch"], next_update=state["next_update"],
                    loss=(totals[0]/totals[1]).item(), denominator=totals[1].item(), observations=int(totals[2].item()),
                    seconds=elapsed, observations_per_second=totals[2].item()/elapsed,
                    grad_norm=float(grad_norm), lr=optimizer.param_groups[0]["lr"], consumed_seconds=accounted_seconds())
                with (args.output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
                if state["step"] % config["log_every"] == 0 or preflight:
                    run.log({"train/update": state["step"], **{f"train/{k}": v for k, v in record.items() if k != "step"}})
            if state["step"] % config["eval_every"] == 0:
                eval_and_log()
            if state["step"] % config["checkpoint_every"] == 0:
                checkpoint("RUNNING")
        del iterator, loader, dataset
        if not epoch_finished:
            break
        state["epoch"] += 1
        state["next_update"] = 0
    eval_and_log()
    checkpoint(reason if state["step"] < config["max_updates"] else "COMPLETED_UPDATE_LIMIT")
    if rank == 0:
        run.finish()
        node_lock.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
