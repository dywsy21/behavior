"""Small bounded all-reduce benchmark, launched with torchrun (no model training)."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import statistics
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--iterations", type=int, default=10)
    args = ap.parse_args()
    if not 1 <= args.iterations <= 10 or args.sizes_mib != [1, 16, 64, 256]:
        ap.error("Probe exceeds registered size/iteration budget")
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world not in (8, 16, 24) or not 0 <= rank < world or not 0 <= local < 8:
        raise ValueError("Expected registered 8/16/24-rank probe with eight GPUs per node")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    results = []
    try:
        topology = [None] * world
        dist.all_gather_object(topology, dict(rank=rank, host=socket.gethostname(), local_rank=local))
        hosts = {entry["host"] for entry in topology}
        if len(hosts) != world // 8 or any(
                sorted(entry["local_rank"] for entry in topology if entry["host"] == host)
                != list(range(8)) for host in hosts):
            raise ValueError("Observed rank topology does not match registered node count")
        expected = world * (world + 1) / 2
        for mib in args.sizes_mib:
            count = mib * 2**20 // 4
            buf = torch.empty(count, dtype=torch.float32, device="cuda")
            times = []
            for step in range(3 + args.iterations):
                buf.fill_(rank + 1)
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                dist.all_reduce(buf)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if not torch.all(buf == expected).item():
                    raise RuntimeError("Incorrect all-reduce result")
                longest = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
                dist.all_reduce(longest, op=dist.ReduceOp.MAX)
                if step >= 3:
                    times.append(longest.item())
            median = statistics.median(times)
            results.append(dict(mib=mib, seconds=times, median_ms=median*1000,
                                algorithm_GBps=mib*2**20/median/1e9,
                                bus_GBps=mib*2**20/median/1e9*2*(world-1)/world))
            del buf
        # Actual CUDA BF16 forward/backward smoke, distinct from communication timings.
        x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        y = (x @ x.T).float().square().mean()
        y.backward()
        if not torch.isfinite(y).item() or not torch.isfinite(x.grad).all().item():
            raise RuntimeError("Nonfinite BF16 forward/backward")
        dist.barrier()
        if rank == 0:
            report = dict(host=socket.gethostname(), world_size=world,
                          topology=topology,
                          torch=torch.__version__, cuda=torch.version.cuda,
                          nccl=torch.cuda.nccl.version(), results=results,
                          bf16_forward_backward="all ranks passed",
                          peak_memory_bytes=torch.cuda.max_memory_allocated())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as f:
                json.dump(report, f, indent=2)
            print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
