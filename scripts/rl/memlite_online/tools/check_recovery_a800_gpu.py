"""Bounded 8-rank BF16/NCCL hardware smoke: no model, optimizer, or simulator.

Run with torchrun --standalone --nproc-per-node=8 and an outer 300s timeout.
This is not an exhaustive memory burn-in or a throughput benchmark.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 8 or not 0 <= rank < world:
        raise ValueError("Eight local ranks are required")
    path = args.output / f"rank{rank}.json"
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    started = time.monotonic()
    try:
        # Different rank values make silent duplicate-device/rank errors visible.
        value = torch.full((2**20,), rank+1., device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        if not torch.equal(value, torch.full_like(value, 36.)):
            raise ValueError("NCCL rank reduction mismatch")
        a = torch.ones((1024, 1024), dtype=torch.bfloat16, device="cuda")
        b = torch.full_like(a, (rank+1)/8.)
        for _ in range(8):
            result = a @ b
            if not torch.equal(result, torch.full_like(result, 128*(rank+1))):
                raise ValueError("BF16 matmul mismatch")
        pattern = torch.arange(2**24, dtype=torch.int32, device="cuda")
        copied = pattern.clone()
        if not torch.equal(copied, pattern):
            raise ValueError("GPU memory copy mismatch")
        # A true backward+allreduce, with a closed-form expected gradient.
        parameter = torch.nn.Parameter(torch.tensor([.25], device="cuda"))
        ((parameter * (rank+1)).square().sum()).backward()
        dist.all_reduce(parameter.grad)
        if not torch.equal(parameter.grad, torch.tensor([102.], device="cuda")):
            raise ValueError("Distributed gradient sum mismatch")
        torch.cuda.synchronize()
        receipt = dict(rank=rank, world=world, gpu_name=torch.cuda.get_device_name(rank),
            bf16_matmul=True, nccl_allreduce=True, backward_allreduce=True,
            memory_copy_bytes=copied.numel()*copied.element_size(),
            seconds=time.monotonic()-started, max_allocated_bytes=torch.cuda.max_memory_allocated(),
            optimizer_steps=0, model_loaded=False, status="passed")
        args.output.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
        dist.barrier()
        print(json.dumps(receipt), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
