"""Explicit synchronous PPO collectives, also testable with CPU/Gloo.

All ranks enter the same phases, even when one has no local actor advantage.
No optimizer step or gradient clipping happens before the weighted SUM.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import math
import time

import torch
import torch.distributed as dist


class Collective:
    def __init__(self, device, *, group=None, bucket_bytes=32 << 20):
        self.device = torch.device(device)
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.bucket_bytes = bucket_bytes

    @classmethod
    def initialize(cls, rank, world_size, endpoint, *, backend="nccl", timeout=1800):
        device = "cuda:0" if backend == "nccl" else "cpu"
        if backend == "nccl":
            torch.cuda.set_device(0)
        options = {"device_id": torch.device(device)} if backend == "nccl" else {}
        dist.init_process_group(backend, init_method=endpoint, rank=rank,
                                world_size=world_size, timeout=timedelta(seconds=timeout), **options)
        return cls(device)

    def objects(self, value):
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def same(self, value, label):
        values = self.objects(value)
        if any(item != values[0] for item in values):
            raise RuntimeError(f"Cross-rank {label} mismatch: {values}")
        return values[0]

    def check(self, error, phase):
        errors = self.objects(None if error is None else repr(error))
        if any(item is not None for item in errors):
            raise RuntimeError(f"Shared PPO {phase} failed: {errors}")

    def sum(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        return tensor.cpu().tolist()

    def maximum(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.group)
        return tensor.cpu().tolist()

    def broadcast_parameters(self, parameters):
        for parameter in parameters:
            dist.broadcast(parameter.data, src=0, group=self.group)

    def gradients(self, parameters, *, denominator):
        """Local gradients are weighted SUMS, not local sample means."""
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid global loss denominator")
        started = time.monotonic()
        bucket, size = [], 0

        def flush():
            nonlocal bucket, size
            if not bucket:
                return
            flat = torch.cat([p.grad.detach().reshape(-1) for p in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.group)
            flat.div_(denominator)
            offset = 0
            for parameter in bucket:
                count = parameter.numel()
                parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
                offset += count
            bucket, size = [], 0

        for parameter in parameters:
            if parameter.grad is None:
                # A zero-advantage / empty local shard must still participate.
                parameter.grad = torch.zeros_like(parameter)
            needed = parameter.numel() * parameter.element_size()
            if bucket and (size + needed > self.bucket_bytes or
                           bucket[0].dtype != parameter.dtype):
                flush()
            bucket.append(parameter)
            size += needed
        flush()
        return time.monotonic() - started


def state_digest(value):
    """Exact parameters AND optimizer state identity, not a sampled checksum."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def task_weights(groups):
    """Inverse nominal chunk occupancy for round-robin episode collectors.

    Each rank samples the same chunk quota, but a long task occupies more
    rounds. Under full nominal horizons its occupancy is H_t / sum(H_group).
    Weighting by (world / task_count) * sum(H_group) / H_t corrects BOTH this
    length effect and unequal shard sizes. Early successes change occupancy;
    report observed coverage as well, never call this exact achieved balance.
    """
    tasks = [task for group in groups for task in group["tasks"]]
    names = [task["task"] for task in tasks]
    if not tasks or len(set(names)) != len(names) or any(not g["tasks"] for g in groups):
        raise ValueError("Task groups must be a nonempty disjoint partition")
    result = {}
    for group in groups:
        lengths = [int(t["official_horizon_steps"]) + 1 for t in group["tasks"]]
        if min(lengths) <= 1:
            raise ValueError("Invalid task horizon")
        total = sum(lengths)
        for task, length in zip(group["tasks"], lengths, strict=True):
            result[task["task"]] = len(groups) * total / (len(tasks) * length)
    return result


def global_accept(global_metrics, group_metrics, *, target_kl, max_clip_fraction):
    """One shared decision; prevent averaging away a catastrophically bad shard."""
    if not all(math.isfinite(float(value)) for value in global_metrics.values()):
        return False
    if (global_metrics["mean_approx_kl"] > target_kl or
            global_metrics["clip_fraction"] > max_clip_fraction):
        return False
    return all(math.isfinite(row["mean_approx_kl"]) and
               math.isfinite(row["ratio_mean"]) and
               row["mean_approx_kl"] <= 2 * target_kl and
               row["clip_fraction"] <= min(0.5, 2 * max_clip_fraction)
               for row in group_metrics if row["experiences"])
