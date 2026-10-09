"""Finite-corpus SFT execution primitives; no data admission or launch authority.

Keep the original Stage-1 runtime unchanged. Recovery batches may have fewer
rows than GPUs, and their tails must not silently repeat scarce physical events.
The caller must validate an admitted pool before constructing any CUDA model.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math


def schedule_fingerprint(schedule, *, binding):
    """Pin source/parent/data/world identity together with the exact row order."""
    required = {"component", "source_commit", "parent_sha256", "admission_sha256",
                "recipe_sha256", "world_size", "micro_batch"}
    if not required <= binding.keys() or binding["component"] not in {"H0", "H1", "L0"}:
        raise ValueError("Incomplete recovery identity")
    if any(type(binding[k]) is not int or binding[k] < 1 for k in ("world_size", "micro_batch")):
        raise ValueError("Invalid distributed identity")
    if not schedule or any(not b.get("rows") for b in schedule):
        raise ValueError("Empty finite schedule")
    for key in ("source_commit", "parent_sha256", "admission_sha256", "recipe_sha256"):
        value = binding[key]
        length = 40 if key == "source_commit" else 64
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Unpinned recovery identity: " + key)
    payload = json.dumps(dict(schema="recovery_schedule_v1", binding=binding, schedule=schedule),
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def shard_finite_batch(rows, *, rank, world_size, micro_batch):
    """Same number of forwards on every rank; zero-weight dummy only if empty.

    Partial real microbatches keep their real size. An empty rank uses ONE
    observable row solely to participate in DDP: its objective/denominator and
    sample count are zero, so it is not a supervised repetition or event pass.
    This requires a per-sample model, without cross-row BatchNorm/statistics.
    """
    if (not rows or any(type(x) is not int for x in (rank, world_size, micro_batch))
            or world_size < 1 or micro_batch < 1 or not 0 <= rank < world_size):
        raise ValueError("Invalid finite batch/shard")
    result = []
    stride = world_size * micro_batch
    for start in range(0, len(rows), stride):
        own = rows[start + rank * micro_batch:start + (rank + 1) * micro_batch]
        result.append(dict(rows=own if own else rows[:1], real_count=len(own), dummy=not own))
    return result


def collective_error(local_error, *, device, phase):
    """Fail all ranks together before a later collective/optimizer can commit."""
    import torch
    import torch.distributed as dist
    flag = torch.tensor(int(local_error is not None), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        # Exception type only: never log batch contents, credentials or input
        # payloads that a downstream library might put in its error message.
        kinds = [None] * dist.get_world_size()
        dist.all_gather_object(kinds, type(local_error).__name__ if local_error else None)
        raise RuntimeError(f"Recovery {phase} failed on ranks: {kinds}; resume last committed checkpoint") from local_error


def global_objective_step(ddp, optimizer, micros, *, load_micro, objective, device,
                          clip_norm=1., autocast_factory=nullcontext):
    """One exact globally normalized update, including scarce-data tail ranks.

    objective(ddp, batch) returns (mean_loss_with_graph, valid_denominator).
    Never average local means. Accumulate summed numerators; DDP averages
    gradients, so scale by world/global_denominator BEFORE clipping/Adam.
    Callers advance the committed schedule cursor only after this returns.
    """
    import torch
    import torch.distributed as dist
    from g05.utils.training.stage1_runtime import normalize_ddp_gradients

    world = dist.get_world_size()
    local_error = None
    if not micros or not math.isfinite(clip_norm) or clip_norm <= 0:
        local_error = ValueError("Empty update or invalid clipping")
    collective_error(local_error, device=device, phase="setup")
    counts = [None] * world
    dist.all_gather_object(counts, len(micros))
    if len(set(counts)) != 1:
        raise ValueError("Ranks disagree on the number of accumulated forwards")
    parameters = [p for p in ddp.module.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for i, micro in enumerate(micros):
        batch, error = None, None
        try:
            if (type(micro["real_count"]) is not int or micro["real_count"] < 0
                    or micro["dummy"] != (micro["real_count"] == 0)
                    or len(micro["rows"]) != (micro["real_count"] or 1)):
                raise ValueError("Changed committed microbatch")
            batch = load_micro(micro["rows"])
        except Exception as exc:
            error = exc
        collective_error(error, device=device, phase="data load")
        sync = nullcontext() if i == len(micros) - 1 else ddp.no_sync()
        with sync:
            error = None
            try:
                with autocast_factory():
                    loss, denominator = objective(ddp, batch)
                denominator = torch.as_tensor(denominator, device=device).detach().double()
                if (loss.ndim or denominator.ndim or not loss.requires_grad or not torch.isfinite(loss)
                        or not torch.isfinite(denominator) or denominator <= 0):
                    raise ValueError("Nonfinite, empty, or detached differentiated objective")
                weight = 0. if micro["dummy"] else 1.
                # Never round a large denominator to BF16 just because the
                # model uses mixed precision. Preserve FP64 for proof tests.
                objective_dtype = torch.float64 if loss.dtype == torch.float64 else torch.float32
                numerator = loss.to(objective_dtype) * denominator.to(objective_dtype) * weight
            except Exception as exc:
                error = exc
            collective_error(error, device=device, phase="forward")
            numerator.backward()
        totals += torch.stack((numerator.detach().double(), denominator * weight,
                               torch.tensor(micro["real_count"], device=device, dtype=torch.float64)))
        del batch, loss, numerator
    dist.all_reduce(totals)
    if totals[1] <= 0 or not torch.isfinite(totals).all():
        raise ValueError("Zero or invalid global supervision; no optimizer/weight-decay update")
    normalize_ddp_gradients(parameters, totals[1].item(), world)
    error = None
    try:
        if any(p.grad is not None for p in ddp.module.parameters() if not p.requires_grad):
            raise ValueError("Gradient reached a frozen parameter")
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    except Exception as exc:
        error = exc
    collective_error(error, device=device, phase="gradients")
    error = None
    try:
        optimizer.step()
    except Exception as exc:
        error = exc
    collective_error(error, device=device, phase="optimizer")
    return dict(loss=float(totals[0] / totals[1]), denominator=float(totals[1]),
                real_samples=int(totals[2]), grad_norm=float(norm), world_size=world,
                accumulated_forwards=len(micros), repeated_supervised_tail_rows=0)
