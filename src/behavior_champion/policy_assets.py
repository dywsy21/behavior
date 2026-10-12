"""Fail-closed asset verification for public parents and our formal EMA exports."""
import json
from pathlib import Path

from .release import sha256
from .sampling import digest_json


def verify_policy_assets(root, repo, config):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    receipt_file = (root / config['verified']).resolve()
    if not receipt_file.is_relative_to(root) or sha256(receipt_file) != config['verified_sha256']:
        raise ValueError('Changed or foreign policy asset receipt')
    receipt = json.loads(receipt_file.read_text())
    kind = config.get('checkpoint_kind', 'public_parent')
    expected_model = None
    if kind == 'public_parent':
        if receipt['revision'] != config['revision'] or receipt['status'] != 'all_selected_assets_verified':
            raise ValueError('Wrong public parent')
        asset_root = (root / config['asset_root']).resolve()
        layout = json.loads((repo / config['layout']).read_text())['layout']
    elif kind == 'formal_ema_v2':
        asset_root = (root / config['checkpoint']).resolve()
        if (receipt_file != asset_root / 'export.json' or receipt.get('schema') != 'champion-ema-export-v2'
                or receipt.get('usable_for_deployment') is not True
                or receipt.get('purpose') != 'CHAMPION100_FORMAL_SFT'
                or receipt.get('weights') != 'EMA' or not receipt.get('restored_by_author_loader_exact')
                or receipt.get('completed_updates') != config['completed_updates']
                or receipt['completed_updates'] <= 0):
            raise ValueError('Not the specified verified formal EMA; engineering exports cannot deploy')
        layout = json.loads((asset_root / 'task-layout.json').read_text())
        expected_model = receipt['model']
        if (digest_json(layout) != receipt['task_layout_sha256'] or expected_model['num_tasks'] != 100
                or expected_model['task_num_stages'] != layout['num_stages']
                or config['model_task_count'] != 100 or set(config['trained_model_tasks']) != set(range(100))):
            raise ValueError('Formal 100-task model/layout mismatch')
    else:
        raise ValueError('Unknown checkpoint kind; no implicit parent fallback')
    if not asset_root.is_relative_to(root):
        raise ValueError('Foreign policy asset root')
    checkpoint = (root / config['checkpoint']).resolve()
    if not checkpoint.is_relative_to(asset_root):
        raise ValueError('Checkpoint is not bound to its verified asset root')
    seen = set()
    for row in receipt['files']:
        raw_path = asset_root / row['path']
        path = raw_path.resolve()
        if (Path(row['path']).is_absolute() or '..' in Path(row['path']).parts
                or row['path'] in seen or not path.is_relative_to(asset_root) or raw_path.is_symlink()
                or path.stat().st_size != row['size'] or sha256(path) != row['sha256']):
            raise ValueError('Changed policy artifact: ' + row['path'])
        seen.add(row['path'])
    if not seen or (kind == 'formal_ema_v2' and
                    (not any(path.startswith('params/') for path in seen) or 'task-layout.json' not in seen
                     or 'assets/IliaLarchenko/behavior_224_rgb/norm_stats.json' not in seen)):
        raise ValueError('Incomplete policy asset ledger')
    parameter_prefix = (checkpoint / 'params').relative_to(asset_root).as_posix() + '/'
    if not any(name.startswith(parameter_prefix) for name in seen):
        raise ValueError('Selected checkpoint parameters are not covered by the receipt')
    if kind == 'formal_ema_v2':
        actual = {path.relative_to(asset_root).as_posix() for path in asset_root.rglob('*')
                  if path.is_file() and path != receipt_file}
        if actual != seen:
            raise ValueError('Formal EMA has missing or unregistered artifact files')
    return dict(receipt=receipt, layout=layout, expected_model=expected_model)
