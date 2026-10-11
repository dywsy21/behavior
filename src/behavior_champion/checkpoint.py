"""Explicit-host full-state checkpoints for the A800 transport workaround.

No pickle, no automatic retention/deletion, no implicit GPU0-to-peer restore.
The rebuilt frozen graph supplies structure and shardings. Every dynamic leaf
(parameters, optimizer, EMA, Intermediate state and RNG) is hash verified.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .release import sha256
from .sampling import digest_json


def _description(path, value):
    import jax
    # An abstract resume template need not allocate the full optimizer first.
    dtype = np.dtype(value.dtype) if hasattr(value, 'dtype') else np.asarray(value).dtype
    shape = tuple(value.shape) if hasattr(value, 'shape') else np.asarray(value).shape
    return dict(path=jax.tree_util.keystr(path), shape=list(shape), dtype=str(dtype),
                bytes=int(np.prod(shape, dtype=np.int64)) * dtype.itemsize)


def tree_fingerprints(tree):
    import jax
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        host = np.ascontiguousarray(np.asarray(jax.device_get(leaf))).reshape(-1).view(np.uint8)
        rows.append(dict(**_description(path, leaf), sha256=hashlib.sha256(memoryview(host)).hexdigest()))
    return dict(sha256=digest_json(rows), leaves=rows)


def save_full(directory, tree, *, completed, contract, cursor, reserve_bytes=256 << 30):
    """Commit a new immutable directory only after all full-state files are durable."""
    import jax
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'step_{completed:08d}'
    if target.exists() or completed < 0 or cursor['completed_updates'] != completed:
        raise ValueError('Fresh checkpoint and matching completed-update cursor required')
    items = jax.tree_util.tree_flatten_with_path(tree)[0]
    needed = sum(_description(path, leaf)['bytes'] for path, leaf in items)
    if shutil.disk_usage(root).free < needed * 1.1 + reserve_bytes:
        raise OSError('Insufficient checkpoint space plus reserve; no existing files were removed')
    temporary = Path(tempfile.mkdtemp(prefix=f'.incomplete-{completed:08d}-', dir=root))
    rows = []
    for index, (path, leaf) in enumerate(items):
        host = np.ascontiguousarray(np.asarray(jax.device_get(leaf))).reshape(-1).view(np.uint8)
        row = dict(**_description(path, leaf), file=f'{index:05d}.bin',
                   sha256=hashlib.sha256(memoryview(host)).hexdigest())
        with (temporary / row['file']).open('xb') as stream:
            host.tofile(stream); stream.flush(); os.fsync(stream.fileno())
        rows.append(row)
    manifest = dict(schema='champion-full-state-v1', completed_updates=completed,
                    contract=contract, contract_sha256=digest_json(contract), sampler_cursor=cursor,
                    leaves=rows, dynamic_bytes=needed,
                    tree_sha256=digest_json([{k: v for k, v in row.items() if k != 'file'} for row in rows]))
    with (temporary / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    # If the process fails earlier, the incomplete directory is preserved and
    # never offered as a resumable checkpoint. No overwrite or cleanup fallback.
    if target.exists():
        raise FileExistsError(target)
    temporary.rename(target)
    return dict(path=str(target), manifest_sha256=sha256(target / 'manifest.json'),
                completed_updates=completed, tree_sha256=manifest['tree_sha256'], bytes=needed)


def restore_full(directory, template, shardings, *, contract, plan, manifest_sha256):
    import jax
    from .jax_transport import place_from_host
    # Dataclass configs contain tuples (e.g. task-stage tables), while JSON
    # faithfully stores the same sequence as a list. Canonicalize metadata;
    # this does not relax any array, source identity or cursor comparison.
    contract = json.loads(json.dumps(contract))
    root = Path(directory).resolve()
    if sha256(root / 'manifest.json') != manifest_sha256:
        raise ValueError('Checkpoint manifest digest differs from receipt')
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest['schema'] != 'champion-full-state-v1' or manifest['contract'] != contract or
            manifest['contract_sha256'] != digest_json(contract)):
        raise ValueError('Frozen training contract changed on resume')
    plan.validate_cursor(manifest['sampler_cursor'], manifest['completed_updates'])
    items, treedef = jax.tree_util.tree_flatten_with_path(template)
    placements, placement_def = jax.tree_util.tree_flatten(shardings)
    if treedef != placement_def or len(items) != len(manifest['leaves']):
        raise ValueError('Checkpoint template/sharding tree changed')
    rows = manifest['leaves']
    expected_tree_sha = digest_json([{k: v for k, v in row.items() if k != 'file'} for row in rows])
    if expected_tree_sha != manifest['tree_sha256']:
        raise ValueError('Checkpoint leaf ledger changed')
    # Check all file metadata/digests BEFORE creating any restored device array.
    for index, ((path, leaf), row) in enumerate(zip(items, rows, strict=True)):
        expected = _description(path, leaf)
        if ({key: row[key] for key in expected} != expected or row['file'] != f'{index:05d}.bin'):
            raise ValueError('Checkpoint path/shape/dtype order mismatch')
        source = root / row['file']
        if source.is_symlink() or source.stat().st_size != row['bytes'] or sha256(source) != row['sha256']:
            raise ValueError('Corrupted checkpoint array: ' + row['path'])
    restored = []
    for (_, leaf), row, placement in zip(items, rows, placements, strict=True):
        dtype = np.dtype(leaf.dtype) if hasattr(leaf, 'dtype') else np.asarray(leaf).dtype
        host = np.fromfile(root / row['file'], dtype=dtype).reshape(row['shape'])
        # Fail closed on any mutation between the validation and consumption.
        if hashlib.sha256(memoryview(host.reshape(-1).view(np.uint8))).hexdigest() != row['sha256']:
            raise ValueError('Checkpoint changed while restoring')
        restored.append(place_from_host(host, placement))
    result = jax.tree_util.tree_unflatten(treedef, restored)
    actual_step = int(np.asarray(result['state'].step)) if 'state' in result else manifest['completed_updates']
    if actual_step != manifest['completed_updates']:
        raise ValueError('Optimizer step and sampler cursor disagree')
    return result
