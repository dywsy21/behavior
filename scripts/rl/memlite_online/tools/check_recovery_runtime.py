"""Tiny synthetic DDP/Adam/cursor/RNG proof, NOT a G0.5 SFT or effect test.

CPU: torchrun --standalone --nproc-per-node=2 ... --backend gloo
A800: inherited activation + 8 ranks + --backend nccl, outer timeout 300s.
No parent weights/data/credentials loaded; toy checkpoints stay in this run.
"""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from g05.utils.training.recovery_runtime import (
    global_objective_step, schedule_fingerprint, shard_finite_batch,
)
from g05.utils.training.stage1_runtime import (
    atomic_json, capture_rng, restore_rng, load_checkpoint, save_checkpoint,
)


def same_tree(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right))
    return left == right


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--backend', choices=('gloo', 'nccl'), required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world not in (2, 8) or rank != int(os.environ['LOCAL_RANK']):
        raise ValueError('Bounded single-node proof only')
    if args.backend == 'nccl':
        if world != 8:
            raise ValueError('A800 proof expects eight ranks')
        torch.cuda.set_device(rank)
    device = torch.device('cuda', rank) if args.backend == 'nccl' else torch.device('cpu')
    dist.init_process_group(args.backend, timeout=timedelta(seconds=180))
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        started = time.monotonic()
        torch.manual_seed(713)
        model = torch.nn.Linear(3, 1, dtype=torch.float64, device=device)
        reference = copy.deepcopy(model)
        ddp = DDP(model, device_ids=[rank] if device.type == 'cuda' else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        reference_opt = torch.optim.AdamW(reference.parameters(), lr=.0003)
        # Two accumulation rounds. In round 2 all but rank 0 are dummy-only.
        rows = list(range(world * 2 + 1))
        micros = shard_finite_batch(rows, rank=rank, world_size=world, micro_batch=2)
        binding = dict(component='L0', source_commit='0'*40, parent_sha256='0'*64,
                       admission_sha256='0'*64, recipe_sha256='0'*64,
                       world_size=world, micro_batch=2, synthetic_test_only=True)
        fingerprint = schedule_fingerprint([dict(rows=rows)], binding=binding)
        randomize = [False]

        def load(indices):
            x = torch.tensor([[i/10., (i+2)/11., 1.] for i in indices], device=device, dtype=torch.float64)
            if randomize[0]:
                x = x + torch.randn_like(x) * (random.random() + np.random.rand()) / 100
            weights = torch.tensor([1+(i%4) for i in indices], device=device, dtype=torch.float64)
            return x, weights

        def objective(net, batch):
            x, weights = batch
            return (net(x).flatten().square()*weights).sum()/weights.sum(), weights.sum()

        x, weights = load(rows)
        reference_loss, _ = objective(reference, (x, weights))
        reference_loss.backward()
        reference_opt.step()
        result = global_objective_step(ddp, optimizer, micros, load_micro=load, objective=objective,
                                       device=device, clip_norm=1e6)
        error = max(float((a-b).abs().max().detach()) for a,b in zip(model.parameters(),reference.parameters()))
        if error > 1e-12 or abs(result['loss'] - float(reference_loss.detach())) > 1e-12:
            raise AssertionError('Uneven-rank global objective/Adam differs from concatenated reference')
        if result['real_samples'] != len(rows):
            raise AssertionError('Zero-weight tail was counted as supervision')
        # Save exactly after update 1. Python/NumPy/Torch RNG differs by rank.
        random.seed(61+rank); np.random.seed(71+rank); torch.manual_seed(81+rank)
        rngs = [None] * world if rank == 0 else None
        dist.gather_object(capture_rng(), rngs, dst=0)
        state = dict(step=1, next_update=1, epoch=0, save_sequence=1, consumed_seconds=3.,
                     fingerprint=fingerprint, wandb_run_id=None, synthetic_test_only=True)
        if rank == 0:
            save_checkpoint(args.output/'toy-checkpoints', model=model, optimizer=optimizer,
                            state=state, rng_by_rank=rngs)
        dist.barrier()
        randomize[0] = True
        expected = global_objective_step(ddp, optimizer, micros, load_micro=load, objective=objective,
                                        device=device, clip_norm=1e6)
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        saved, _ = load_checkpoint(args.output/'toy-checkpoints', fingerprint)
        if saved['state']['next_update'] != 1 or len(saved['rng_by_rank']) != world:
            raise AssertionError('Committed cursor or all-rank RNG missing')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        restore_rng(saved['rng_by_rank'][rank])
        resumed = global_objective_step(ddp, optimizer, micros, load_micro=load, objective=objective,
                                       device=device, clip_norm=1e6)
        if (expected != resumed or not same_tree(model.state_dict(), expected_model)
                or not same_tree(optimizer.state_dict(), expected_optimizer)):
            raise AssertionError('Resumed loss, parameters or Adam moments differ from uninterrupted run')
        # One rank fails its reader: every rank must exit the update before Adam.
        def broken_load(indices):
            if rank == world - 1:
                raise OSError('Synthetic reader failure')
            return load(indices)
        try:
            global_objective_step(ddp, optimizer, micros, load_micro=broken_load, objective=objective,
                                  device=device, clip_norm=1e6)
        except RuntimeError as exc:
            if 'data load' not in str(exc):
                raise
        else:
            raise AssertionError('One-rank data error did not abort the global update')
        if not same_tree(model.state_dict(), expected_model) or not same_tree(optimizer.state_dict(), expected_optimizer):
            raise AssertionError('Aborted read updated model or Adam')
        receipt = dict(rank=rank, world=world, backend=args.backend,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
            scope='synthetic runtime proof only; not full G0.5 trainer or data acceptance',
            exact_global_objective=True, parameter_error_vs_reference=error,
            empty_rank_zero_weight=True, real_rows=len(rows), exact_resume_model_adam_rng=True,
            one_rank_data_failure_aborts_all=True, g05_optimizer_steps=0,
            toy_optimizer_executions=3, seconds=time.monotonic()-started,
            max_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
            status='passed')
        atomic_json(args.output/f'rank{rank}.json',receipt)
        print(json.dumps(receipt), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
