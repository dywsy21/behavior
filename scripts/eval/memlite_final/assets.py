"""Self-contained configs/assets plus hash-bound exact model-only exports."""
from copy import deepcopy
import json
from pathlib import Path

from recovery_corpus import file_sha
from recovery_calibration_launch import bound, read


def relocate_config(value, source_root=None, target_root=None):
    if (source_root is None) == (target_root is None):
        raise ValueError('Exactly one export/materialization root required')
    if isinstance(value,dict):
        return {k:relocate_config(v,source_root,target_root) for k,v in value.items()}
    if isinstance(value,list):
        return [relocate_config(v,source_root,target_root) for v in value]
    if isinstance(value,str):
        if source_root is not None and value.startswith(str(source_root)+'/'):
            return '@PACK@/'+value[len(str(source_root))+1:]
        if target_root is not None and value.startswith('@PACK@/'):
            part = Path(value[7:])
            if part.is_absolute() or '..' in part.parts:
                raise ValueError('Unsafe portable path')
            return str(Path(target_root)/part)
        if source_root is not None and value.startswith('/'):
            raise ValueError('Unregistered absolute config dependency: '+value)
    return deepcopy(value)


def verify_assets(pack, expected_manifest_sha256, config_path):
    pack=Path(pack).resolve(); cfg=read(config_path)
    manifest=read(bound(pack,dict(path='manifest.json',sha256=expected_manifest_sha256)))
    if (manifest['schema']!='memlite_final_assets_v1' or manifest['config_sha256']!=file_sha(config_path)
            or manifest['status']!='exact_model_exports_complete' or manifest['optimizer_updates']!=0
            or set(manifest['models'])!=set(cfg['models'])):
        raise ValueError('Foreign/incomplete latest-model artifact manifest')
    paths={}
    for row in manifest['files']:
        if row['path'] in paths: raise ValueError('Duplicate asset')
        p=bound(pack,dict(path=row['path'],sha256=row['sha256']))
        if p.stat().st_size!=row['bytes']: raise ValueError('Truncated asset')
        paths[row['path']]=p
    actual={p.relative_to(pack).as_posix() for p in pack.rglob('*') if p.is_file() and p!=pack/'manifest.json'}
    if actual!=set(paths):raise ValueError('Unregistered or missing pack dependency')
    for name, row in manifest['models'].items():
        if row['source_sha256']!=cfg['models'][name]['sha256']:
            raise ValueError('Wrong original model: '+name)
        if name=='planner':
            export=cfg['existing_planner_export']
            path=bound(Path(cfg['target_root']),dict(path=export['target_path'],sha256=export['sha256']))
            if row['export_sha256']!=export['sha256'] or row['exact_model_tensors']!=950:
                raise ValueError('Wrong migrated planner')
            row['runtime_path']=str(path)
        else:
            path=paths[row['path']]
            if row['export_sha256']!=file_sha(path): raise ValueError('Wrong export binding')
            row['runtime_path']=str(path)
    configs={side:relocate_config(read(pack/(side+'-config.json')),target_root=pack) for side in ('high','low')}
    for side, name in [('high','planner'),('low','low')]:
        if file_sha(configs[side]['stats_path'])!=cfg['normalization_sha256'][name]:
            raise ValueError('Changed serving normalization')
    if cfg['normalization_sha256']['observer']!=cfg['normalization_sha256']['planner']:
        raise ValueError('Original separate observer normalization changed')
    return cfg,manifest,configs
