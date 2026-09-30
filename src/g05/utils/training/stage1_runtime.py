"""Small, explicit runtime primitives for formal MEM-Lite stage-1 jobs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    except BaseException:
        # Keep any partial file for diagnosis; never disturb a prior final.
        raise


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def capture_rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda_current=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda_current"] is not None:
        torch.cuda.set_rng_state(state["cuda_current"])


def normalize_ddp_gradients(parameters, global_denominator, world_size):
    """DDP averaged summed local numerators; convert to one global mean.

    Call AFTER accumulation/all-reduce and BEFORE clipping/AdamW. Never divide
    by accumulation again. A zero objective must not perform an AdamW update.
    """
    if not math.isfinite(global_denominator) or global_denominator <= 0:
        raise ValueError("No globally valid supervision; refusing optimizer/weight-decay step")
    factor = float(world_size) / global_denominator
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


class WallBudget:
    def __init__(self, maximum_seconds, consumed_seconds=0., clock=time.monotonic):
        if maximum_seconds <= 0 or consumed_seconds < 0:
            raise ValueError("Invalid cumulative wall budget")
        self.limit = float(maximum_seconds)
        self.previous = float(consumed_seconds)
        self.clock, self.start = clock, clock()

    def consumed(self):
        return self.previous + self.clock() - self.start

    def exhausted(self, reserve_seconds=0.):
        return self.consumed() + reserve_seconds >= self.limit


def lr_multiplier(completed_steps, warmup, horizon, final_ratio=.1):
    if not 0 <= completed_steps or not 0 < warmup < horizon or not 0 < final_ratio <= 1:
        raise ValueError("Invalid learning-rate schedule")
    if completed_steps < warmup:
        return (completed_steps + 1) / warmup
    progress = min(1., (completed_steps - warmup) / (horizon - warmup))
    return final_ratio + (1. - final_ratio) * .5 * (1. + math.cos(math.pi * progress))


def disk_has_reserve(path, minimum_gib=256., *, free_bytes=None):
    """Stop before the shared disk is full, leaving room for a final save.

    This never deletes checkpoints or assumes ownership of another user's data.
    """
    if minimum_gib < 64:
        raise ValueError("Keep at least 64GiB for safe saving/shared-disk headroom")
    free = shutil.disk_usage(path).free if free_bytes is None else free_bytes
    return free >= minimum_gib * 1024**3


def save_checkpoint(directory, *, model, optimizer, state, rng_by_rank):
    """Publish weights+optimizer and all-rank RNG atomically, then latest.json.

    state carries the *next committed* observation cursor, config/data hashes,
    source commit, W&B run identity and cumulative active wall time. No deletion.
    """
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step_{state['step']:08d}_save_{state['save_sequence']:04d}.pt"
    final = directory / name
    if final.exists():
        raise FileExistsError(final)
    fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".pending", dir=directory)
    started = time.monotonic()
    with os.fdopen(fd, "wb") as stream:
        torch.save(dict(format_version="memlite-stage1-v1", model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), state=state,
                        rng_by_rank=rng_by_rank), stream)
        stream.flush()
        os.fsync(stream.fileno())
    checksum = sha256(temporary)
    os.replace(temporary, final)
    fsync_directory(directory)
    # This receipt also includes saving+hashing wall time, which cannot be
    # known inside the checkpoint before writing its bytes.
    receipt = dict(path=name, sha256=checksum, state=state,
                   consumed_seconds=state["consumed_seconds"] + time.monotonic() - started)
    atomic_json(final.with_suffix(".json"), receipt)
    atomic_json(directory / "latest.json", receipt)
    return receipt


def load_checkpoint(directory, expected_fingerprint):
    import torch
    directory = Path(directory)
    receipt = json.loads((directory / "latest.json").read_text())
    name = receipt["path"]
    if Path(name).name != name:
        raise ValueError("Checkpoint receipt must reference a sibling file")
    path = directory / name
    if sha256(path) != receipt["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if (saved.get("format_version") != "memlite-stage1-v1"
            or saved["state"]["fingerprint"] != expected_fingerprint
            or saved["state"] != receipt["state"]):
        raise ValueError("Resume code/config/data identity mismatch")
    return saved, receipt


def init_wandb(config, output, *, run_id, resume, metadata):
    """Only rank zero calls this. Never upload code, checkpoints, or API keys."""
    import wandb
    if config["mode"] != "online":
        raise ValueError("Formal stage 1 requires explicit online W&B; no silent offline fallback")
    credential = Path(config["credential_file"])
    if credential.stat().st_mode & 0o077:
        raise PermissionError("W&B credential must be mode 0600 or stricter")
    key = credential.read_text().strip()
    if not key:
        raise ValueError("Empty W&B credential")
    # Environment is local to this Python process, not a command-line argument.
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_SILENT"] = "true"
    run = wandb.init(project=config["project"], entity=config.get("entity"),
                     id=run_id, resume="must" if resume else "never", mode="online",
                     name=config["name"], group=config["group"], dir=str(output),
                     config=metadata, save_code=False,
                     settings=wandb.Settings(console="off", init_timeout=90))
    run.define_metric("train/update")
    run.define_metric("train/*", step_metric="train/update")
    run.define_metric("eval/*", step_metric="train/update")
    return run
