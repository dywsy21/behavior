"""Publish only a checkpoint which can be reloaded and has finite tensors."""
from pathlib import Path
import hashlib
import json
import os
import time
import torch


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_checkpoint(path, expected_updates):
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if payload['kind'] != 'a4_direct_flow_ppo_trainable_state_v1':
        raise ValueError('Wrong checkpoint type')
    if int(payload['updates']) != expected_updates:
        raise ValueError('Checkpoint counter differs from completed updates')
    actor_updates = int(payload.get('actor_updates', expected_updates))
    counts = {}
    for name in ['action_expert', 'critic', 'actor_optimizer', 'critic_optimizer']:
        counts[name] = 0
        def visit(value):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite tensor in ' + name)
                counts[name] += 1
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
        visit(payload[name])
        if not counts[name] and not (name=='actor_optimizer' and actor_updates==0):
            raise ValueError('Missing state: ' + name)
    steps = [float(s['step']) for s in payload['actor_optimizer']['state'].values()]
    if (actor_updates > 0 and not steps) or any(step != actor_updates for step in steps):
        raise ValueError('Adam steps disagree with completed update count')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return dict(updates=expected_updates, actor_updates=actor_updates, bytes=Path(path).stat().st_size,
                sha256=digest.hexdigest(), tensors_checked=counts,
                actor_optimizer_states=len(steps), finite=True, load_verified=True)


def publish_checkpoint(output, payload):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'direct_latest.tmp.pt'
    latest = output / 'direct_latest.pt'
    previous = output / 'direct_previous.pt'
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = validate_checkpoint(temporary, int(payload['updates']))
    if latest.exists():
        # Keep the previous immutable inode without duplicating 7GB of disk IO.
        pending = output / 'direct_previous.pending.pt'
        if pending.exists():
            pending.unlink()
        os.link(latest, pending)
        os.replace(pending, previous)
    os.replace(temporary, latest)
    milestone = None
    if payload['updates'] > 0 and payload['updates'] % 5000 == 0:
        milestone = output / f"direct_update_{payload['updates']:06d}.pt"
        if not milestone.exists():
            os.link(latest, milestone)
    receipt.update(latest=str(latest), latest_sha256=receipt['sha256'],
                   milestone=str(milestone) if milestone else None, saved_at=time.time())
    atomic_json(output / 'direct_latest.receipt.json', receipt)
    descriptor = os.open(output, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return receipt
