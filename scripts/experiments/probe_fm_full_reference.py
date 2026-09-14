"""One real four-rank initial fixed80 replay, zero training or retries.

Use the same model, DDP initialization, data and evaluator as the failed FM
control. Add tensor fingerprints for BOTH of its already-existing forwards
per window. Do not relax or replace the original A4 reference threshold.
"""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import subprocess

from action_training_runtime import bootstrap, REPO, INPUT, sha
from action_training_data import build_original_pipeline
from probe_action_training_gpu import prepare_action_updates
from probe_ar_execution_codec import publish
from probe_fm_initial_reference import fingerprint, trace_forward
from train_fm_method_probe import BASE, PARENT, PARENT_SHA, verify_parent
import train_action_method_probe as trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output != BASE / "fm_full_reference_probe_v1":
        raise ValueError("Only the declared full initial reference replay is permitted")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean immutable source")
    identity = bootstrap()
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from omegaconf import OmegaConf
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.config.config_resolvers import register_default_resolvers

    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 4 or rank != local or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise RuntimeError("Exactly the original four physical GPU ranks are required")
    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Each rank needs 40 GiB free; do not evict other work")
    torch.cuda.set_per_process_memory_fraction(.4, local)
    random.seed(41 + rank)
    np.random.seed(41 + rank)
    torch.manual_seed(41 + rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    if rank == 0:
        args.output.mkdir(exist_ok=False)
    dist.barrier()
    cfg_path = INPUT / "diagnostic_processor_config.yaml"
    if sha(cfg_path) != trainer.read(INPUT / "result.json")["processor_config_sha256"] or sha(PARENT) != PARENT_SHA:
        raise RuntimeError("Original A4 or data configuration identity changed")
    cfg = OmegaConf.load(cfg_path)
    arch = trainer.configure(cfg, "fm")
    pipeline = build_original_pipeline(cfg, rank=rank, world_size=world, workers=4)
    identity.update(commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)), config_sha256=sha(cfg_path), parent_sha256=PARENT_SHA,
        actual_evaluator_sha256=sha(Path(trainer.__file__)), rank=rank, pipeline=pipeline.identity,
        max_total_forwards=160, actual_train_draws=0, optimizer_updates=0, simulator_controls=0)
    publish(args.output / f"identity_rank{rank}.json", identity)
    model, saved = load_model_from_checkpoint(arch, str(PARENT), device=f"cuda:{local}", eval_mode=False, return_full_checkpoint=True)
    verify_parent(model, saved)
    del saved
    gc.collect()
    contract = prepare_action_updates(model, f"cuda:{local}")
    ddp = DDP(model, device_ids=[local], find_unused_parameters=True, gradient_as_bucket_view=True, bucket_cap_mb=128)
    publish(args.output / f"restoration_rank{rank}.json", dict(passed=True, exact_entries=1138, lora_entries=192, contract=contract))
    current, traces = {}, []
    original_batch = trainer.exact_eval_batch
    original_ref = trainer.reference_fm
    # The evaluator already makes two forwards per window (auxiliary FM and
    # actual route). A hook records both; no additional model call is inserted.
    original_loss = model.model.fm_helper.cal_fm_loss
    def collect_batch(pipe, window):
        batch = original_batch(pipe, window)
        current.clear()
        current.update(window_id=window["window_id"], input=fingerprint(batch), calls=[])
        traces.append(current.copy())
        return batch
    def collect_loss(*values, **kwargs):
        result = original_loss(*values, **kwargs)
        traces[-1]["calls"].append(dict(loss=float(result), fm_loss_inputs=fingerprint(dict(args=values, kwargs=kwargs))))
        return result
    trainer.exact_eval_batch = collect_batch
    model.model.fm_helper.cal_fm_loss = collect_loss
    try:
        result = trainer.evaluate(model, pipeline, "fm", 0, args.output, rank, world, full=True)
    finally:
        trainer.exact_eval_batch = original_batch
        trainer.reference_fm = original_ref
        model.model.fm_helper.cal_fm_loss = original_loss
    if len(traces) != 20 or any(len(r["calls"]) != 2 for r in traces):
        raise RuntimeError("The actual original fixed80/two-forward budget changed")
    publish(args.output / f"trace_rank{rank}.json", dict(rank=rank, windows=traces, model_forwards=40))
    gathered = [None] * world
    dist.all_gather_object(gathered, traces)
    if rank == 0:
        all_traces = [row for subset in gathered for row in subset]
        paired = [{"window_id": r["window_id"], "first": r["calls"][0]["loss"], "second": r["calls"][1]["loss"]}
                  for r in all_traces if r["calls"][0] != r["calls"][1]]
        reference = trainer.read(BASE / "ar_a4_fulltrain_v2/formal/eval_step_0.json")
        old_rows = {r["window_id"]: r for r in reference["rows"]}
        differences = [dict(window_id=r["window_id"], actual=r["reference_fm_loss"],
            reference=old_rows[r["window_id"]]["reference_fm_loss"]) for r in result["rows"]
            if r["reference_fm_loss"] != old_rows[r["window_id"]]["reference_fm_loss"]]
        aggregate = result["aggregate"]["reference_fm_loss"]
        publish(args.output / "result.json", dict(complete=True, actual_forwards=160, actual_windows=80,
            optimizer_updates=0, actual_train_draws=0, simulator_controls=0, aggregate=aggregate,
            reference=.1969439941, original_absolute_tolerance=2e-6,
            original_reference_gate_passed=abs(aggregate - .1969439941) <= 2e-6,
            per_window_differences=differences, same_input_paired_forward_differences=paired,
            original_threshold_unchanged=True, training_admissible=False, success_rate_claim=False))
        print(json.dumps(dict(complete=True, aggregate=aggregate, original_gate_passed=abs(aggregate-.1969439941)<=2e-6,
                             differing_windows=len(differences), differing_pairs=len(paired))), flush=True)
    dist.barrier()
    del ddp
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
