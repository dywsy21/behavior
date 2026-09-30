"""Eight-rank CPU/Gloo proof of unequal-mask, accumulated global gradients."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from g05.utils.training.stage1_runtime import normalize_ddp_gradients, atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError("This test requires eight ranks")
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(.25)
    ddp = DistributedDataParallel(model)
    reference = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    reference.load_state_dict(model.state_dict())
    local_denominator = 0.
    all_numerators = []
    all_denominators = []
    for r in range(world):
        for micro in range(8):
            count = 2 + (r + micro) % 3  # exact-tail-like uneven microbatches
            x = torch.arange(1, 1+count*2, dtype=torch.float64).reshape(count, 2) / (r + micro + 2)
            weights = torch.arange(1, count+1, dtype=torch.float64) * (r + 1)
            denominator = weights.sum()
            all_numerators.append((reference(x).flatten().square()*weights).sum())
            all_denominators.append(denominator)
            if r == rank:
                with nullcontext() if micro == 7 else ddp.no_sync():
                    numerator = (ddp(x).flatten().square()*weights).sum()
                    numerator.backward()
                local_denominator += denominator
    dist.all_reduce(local_denominator)
    normalize_ddp_gradients(model.parameters(), local_denominator.item(), world)
    (sum(all_numerators)/sum(all_denominators)).backward()
    error = float((model.weight.grad-reference.weight.grad).abs().max())
    if error > 1e-12:
        raise AssertionError(f"Global numerator/denominator gradient mismatch: {error}")
    atomic_json(args.output/f"rank{rank}.json", dict(rank=rank, world=world, max_absolute_error=error,
        accumulation=8, varying_observations_and_weights=True, cuda_initialized=torch.cuda.is_initialized()))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
