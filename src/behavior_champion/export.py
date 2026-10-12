"""Immutable EMA export in the author's loadable Orbax params/assets format."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .checkpoint import materialize_host
from .release import sha256
from .sampling import digest_json


def export_ema(directory, state, *, receipt, assets, layout, contract, reserve_bytes=256 << 30,
               asset_files=None):
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict
    from openpi.models.model import restore_params
    if state.ema_params is None or int(state.step) != receipt['completed_updates']:
        raise ValueError('EMA export must match a complete durable checkpoint')
    if sha256(Path(receipt['path']) / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Export source checkpoint manifest changed')
    source_manifest = json.loads((Path(receipt['path']) / 'manifest.json').read_text())
    if (source_manifest['completed_updates'] != int(state.step) or
            source_manifest['contract_sha256'] != digest_json(contract)):
        raise ValueError('Export state/contract differs from the durable source')
    source_ema = {row['path']: row for row in source_manifest['leaves']
                  if row['path'].startswith("['state'].ema_params")}
    ema_leaves = jax.tree_util.tree_flatten_with_path(state.ema_params)[0]
    if len(ema_leaves) != len(source_ema):
        raise ValueError('Incomplete EMA ledger in source checkpoint')
    root = Path(directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    destination = root / f'ema_{int(state.step):08d}'
    if destination.exists():
        raise FileExistsError(destination)
    dynamic_bytes = sum(value.size * np.dtype(value.dtype).itemsize for value in jax.tree.leaves(state.ema_params))
    if shutil.disk_usage(root).free < reserve_bytes + 1.1 * dynamic_bytes:
        raise OSError('Insufficient export space; no checkpoint or prior export removed')
    temporary = Path(tempfile.mkdtemp(prefix='.incomplete-ema-', dir=root))
    # Preserve all source NNX fields just as the author's _split_params does.
    # Pure dict avoids serializing Python graph objects. HOST origin is explicit.
    host = jax.tree.map(materialize_host, state.ema_params.to_pure_dict())
    expected = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                         sha256=hashlib.sha256(value.tobytes()).hexdigest())
                for key, value in flatten_dict(host).items()}
    # Map original NNX leaf paths onto the full-state ledger. Do not merely
    # trust a matching step number while exporting a different live EMA.
    for path, leaf in ema_leaves:
        name = "['state'].ema_params" + jax.tree_util.keystr(path)
        saved = source_ema.get(name)
        if saved is None or hashlib.sha256(materialize_host(leaf).tobytes()).hexdigest() != saved['sha256']:
            raise ValueError('Live EMA does not equal its checkpoint: ' + name)
    with ocp.PyTreeCheckpointer() as writer:
        writer.save(temporary / 'params', {'params': host})
    del host
    restored = restore_params(temporary / 'params', restore_type=np.ndarray)
    actual = flatten_dict(restored)
    if actual.keys() != expected.keys():
        raise ValueError('Exported EMA keys differ after author restore_params')
    for key, value in actual.items():
        if dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest()) != expected[key]:
            raise ValueError('Exported EMA bytes differ: ' + str(key))
    del actual, restored
    asset_target = temporary / 'assets/IliaLarchenko/behavior_224_rgb'
    if asset_files is None:
        shutil.copytree(assets, asset_target, symlinks=False)
        for original in Path(assets).rglob('*'):
            if original.is_file() and sha256(original) != sha256(asset_target / original.relative_to(assets)):
                raise ValueError('Exported normalization/tokenizer assets changed')
    else:
        # A CPU export from a live shared installation must not copy transient
        # tokenizer bytecode caches produced after the verified asset receipt.
        # Keep them in place; copy ONLY the original pinned asset file ledger.
        if not asset_files or 'norm_stats.json' not in asset_files:
            raise ValueError('Incomplete explicitly pinned export asset ledger')
        for name, expected_sha in asset_files.items():
            relative = Path(name); original = Path(assets) / relative
            if (relative.is_absolute() or '..' in relative.parts or original.is_symlink()
                    or not original.resolve().is_relative_to(Path(assets).resolve())
                    or sha256(original) != expected_sha):
                raise ValueError('Unsafe or changed pinned export asset')
            copied = asset_target / relative; copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copied)
            if sha256(copied) != expected_sha:
                raise ValueError('Pinned export asset changed during copying')
    # The inference adapter must use this model-task mapping, not dataset IDs as
    # embedding IDs, nor the old hard-coded 50-task table.
    (temporary / 'task-layout.json').write_text(json.dumps(layout, indent=2) + '\n')
    files = [dict(path=str(path.relative_to(temporary)), size=path.stat().st_size, sha256=sha256(path))
             for path in sorted(temporary.rglob('*')) if path.is_file()]
    manifest = dict(schema='champion-ema-export-v2', weights='EMA', completed_updates=int(state.step),
        source_checkpoint=receipt, training_contract_sha256=digest_json(contract),
        purpose=contract['purpose'], usable_for_deployment=contract['purpose'] == 'CHAMPION100_FORMAL_SFT',
        model=contract['model'], task_layout_sha256=digest_json(layout),
        restored_by_author_loader_exact=True, files=files,
        parameters=[dict(path='/'.join(key), **value) for key, value in expected.items()])
    # Commit only after both the loadable parameters and copied assets are on
    # durable storage. A failed/incomplete directory remains visibly incomplete.
    (temporary / 'export.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for path in temporary.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    temporary.rename(destination)
    return dict(path=str(destination), manifest_sha256=sha256(destination / 'export.json'),
                completed_updates=int(state.step), usable_for_deployment=manifest['usable_for_deployment'])
