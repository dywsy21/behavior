"""Read an immutable saved EMA on CPU, without restoring a training process.

Only the EMA bytes are consumed, not the optimizer or live training parameters.
The complete ledger, step and model shape contract are nevertheless checked.
This is an inference export, never a reconstructed resumable training state.
"""
import ast
import hashlib
import json
from pathlib import Path
import types

import numpy as np

from .release import sha256
from .sampling import digest_json


EMA_PREFIX = "['state'].ema_params"
PARAM_PREFIX = "['state'].params"


def ema_key(path):
    if not path.startswith(EMA_PREFIX) or not path.endswith('.value'):
        raise ValueError('Expected an NNX EMA value path')
    node = ast.parse('root' + path[len(EMA_PREFIX):-len('.value')], mode='eval').body
    keys = []
    while isinstance(node, ast.Subscript):
        key = ast.literal_eval(node.slice)
        if not isinstance(key, str) or not key or '/' in key:
            raise ValueError('EMA keys must be nonempty string mapping keys')
        keys.append(key); node = node.value
    if not isinstance(node, ast.Name) or node.id != 'root' or not keys:
        raise ValueError('Unsupported NNX EMA path; no executable path evaluation')
    return tuple(reversed(keys))


def read_checkpoint_ema(checkpoint, *, manifest_sha256, layout):
    """Return a host-only export view; verify bytes again in export_ema."""
    import flax.nnx as nnx
    from flax.traverse_util import unflatten_dict
    import jax
    import ml_dtypes  # Registers lossless bfloat16 NumPy dtype support.
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('Offline export must select JAX_PLATFORMS=cpu')
    root = Path(checkpoint).resolve()
    manifest_path = root / 'manifest.json'
    if manifest_path.is_symlink() or sha256(manifest_path) != manifest_sha256:
        raise ValueError('Saved checkpoint manifest differs from pinned receipt')
    manifest = json.loads(manifest_path.read_text()); contract = manifest['contract']
    step = manifest['completed_updates']; rows = manifest['leaves']
    if (type(step) is not int or step < 1 or root.name != f'step_{step:08d}'
            or manifest['schema'] != 'champion-full-state-v1'
            or contract['purpose'] != 'CHAMPION100_FORMAL_SFT'
            or digest_json(contract) != manifest['contract_sha256']
            or digest_json(layout) != contract['layout_sha256']
            or manifest['sampler_cursor']['completed_updates'] != step
            or manifest['sampler_cursor']['plan_sha256'] != contract['sampler_plan_sha256']
            or contract['model']['num_tasks'] != 100
            or list(contract['model']['task_num_stages']) != layout['num_stages']):
        raise ValueError('Require complete formal checkpoint, original contract and 100-task layout')
    if digest_json([{k:v for k,v in row.items() if k != 'file'} for row in rows]) != manifest['tree_sha256']:
        raise ValueError('Saved leaf ledger has changed')
    ledger = {}
    for index, row in enumerate(rows):
        if row['file'] != f'{index:05d}.bin' or row['path'] in ledger:
            raise ValueError('Unsafe, duplicate or reordered checkpoint leaf')
        ledger[row['path']] = row
    if sum(row['bytes'] for row in rows) != manifest['dynamic_bytes']:
        raise ValueError('Inconsistent checkpoint byte accounting')

    def load(row):
        shape = row['shape']; dtype = np.dtype(row['dtype'])
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or dtype.hasobject or int(np.prod(shape, dtype=np.int64)) * dtype.itemsize != row['bytes']):
            raise ValueError('Unsafe checkpoint array metadata')
        path = root / row['file']
        if path.is_symlink() or path.stat().st_size != row['bytes']:
            raise ValueError('Changed checkpoint leaf size or symlink')
        value = np.fromfile(path, dtype=dtype).reshape(shape)
        if hashlib.sha256(memoryview(value.reshape(-1).view(np.uint8))).hexdigest() != row['sha256']:
            raise ValueError('Changed saved checkpoint array bytes')
        return value

    if int(load(ledger["['state'].step"])) != step:
        raise ValueError('Saved state step differs from its manifest')
    ema_rows = [row for row in rows if row['path'].startswith(EMA_PREFIX)]
    param_rows = [row for row in rows if row['path'].startswith(PARAM_PREFIX)]
    if not ema_rows or len(ema_rows) != len(param_rows):
        raise ValueError('Incomplete EMA coverage of the model state')
    flat = {}
    for row in ema_rows:
        counterpart = ledger.get(PARAM_PREFIX + row['path'][len(EMA_PREFIX):])
        if counterpart is None or any(row[k] != counterpart[k] for k in ('shape', 'dtype', 'bytes')):
            raise ValueError('Saved EMA shape/dtype differs from the model state')
        key = ema_key(row['path'])
        if key in flat or any(key[:len(old)] == old or old[:len(key)] == key for old in flat):
            raise ValueError('Duplicate or colliding EMA tree path')
        # Param here is only a path-preserving export container. The existing
        # exporter writes a pure dictionary, including Intermediate values.
        flat[key] = nnx.VariableState(nnx.Param, load(row))
    state = types.SimpleNamespace(step=step, ema_params=nnx.State(unflatten_dict(flat)))
    receipt = dict(path=str(root), manifest_sha256=manifest_sha256,
        completed_updates=step, tree_sha256=manifest['tree_sha256'], bytes=manifest['dynamic_bytes'])
    return state, receipt, contract


def verify_parent_assets(parent, receipt_file, *, expected_sha256, with_ledger=False):
    """Check the exact original asset subtree, without rereading parent weights."""
    parent = Path(parent).resolve(); receipt_file = Path(receipt_file)
    if sha256(receipt_file) != expected_sha256:
        raise ValueError('Changed original parent asset receipt')
    receipt = json.loads(receipt_file.read_text())
    if receipt['status'] != 'all_selected_assets_verified':
        raise ValueError('Unverified parent assets')
    prefix = 'assets/IliaLarchenko/behavior_224_rgb/'
    selected = [row for row in receipt['files'] if row['path'].startswith(prefix)]
    if not selected or len({r['path'] for r in selected}) != len(selected):
        raise ValueError('Missing or duplicate parent asset ledger')
    for row in selected:
        raw = parent / row['path']; path = raw.resolve()
        if ('..' in Path(row['path']).parts or not path.is_relative_to(parent)
                or raw.is_symlink() or path.stat().st_size != row['size'] or sha256(path) != row['sha256']):
            raise ValueError('Changed original parent asset: ' + row['path'])
    assets = parent / prefix
    actual = {p.relative_to(parent).as_posix() for p in assets.rglob('*') if p.is_file()}
    registered = {r['path'] for r in selected}
    extra = actual - registered
    if (registered - actual or prefix + 'norm_stats.json' not in registered
            or any('__pycache__' not in Path(name).parts or Path(name).suffix != '.pyc' for name in extra)):
        raise ValueError('Unregistered or missing normalization/tokenizer asset')
    # Generated bytecode is not deleted or copied. Other unregistered files
    # still fail closed; callers can export the exact original seven files.
    return (assets, {r['path'][len(prefix):]:r['sha256'] for r in selected}) if with_ledger else assets
