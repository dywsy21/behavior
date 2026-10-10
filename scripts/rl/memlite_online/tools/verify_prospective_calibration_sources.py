"""Verify a fixed observer and new source closure before physical collection.

No prediction, label, optimizer or training approval is produced. Historical
source groups (including unsuccessful/unapproved attempts) remain excluded.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha,group_key,split_group


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    if (cfg.get('schema')!='prospective_observer_calibration_collection_v1'
            or cfg.get('declared_before_physical_collection_and_model_predictions') is not True
            or cfg.get('reserved_test_20_untouched') is not True):
        raise ValueError('Explicit prospective source/model contract required')
    if a.output.exists():raise FileExistsError(a.output)
    model=root/cfg['selected_observer']['path'];fit=model.parent.parent/'result.json'
    if (file_sha(model)!=cfg['selected_observer']['sha256']
            or file_sha(fit)!=cfg['selected_observer']['fit_result_sha256']):
        raise ValueError('Selected observer changed after cohort declaration')
    fitted=json.loads(fit.read_text())
    if (fitted['high_sha256']!=cfg['high_sha256']
            or fitted['selected_checkpoint_sha256']!=cfg['selected_observer']['sha256']):
        raise ValueError('Selected observer backbone mismatch')
    protected_path=root/cfg['protected'];protected=set(json.loads(protected_path.read_text())['groups'])
    existing=set();excluded=[]
    for name in cfg['excluded_inventories']:
        inv=root/name
        excluded.append(dict(path=name,sha256=file_sha(inv/'manifest.json')))
        for entry in json.loads((inv/'manifest.json').read_text())['cases']:
            path=inv/entry['directory']/'manifest.json'
            if file_sha(path)!=entry['manifest_sha256']:raise ValueError('Old inventory identity changed')
            existing.add(json.loads(path.read_text())['source_group'])
    if len(existing)!=cfg['selection']['excluded_unique_groups']:raise ValueError('Exclusion closure changed')
    source=root/cfg['source_output'];inventory=json.loads((source/'manifest.json').read_text())
    if inventory['status']!='complete':raise ValueError('Incomplete source export')
    import numpy as np
    from PIL import Image
    groups=set();tasks=set();bindings=[];files=0;size=0;controls=0;release=None
    for entry in inventory['cases']:
        directory=source/entry['directory'];manifest=directory/'manifest.json'
        if directory.parent!=source or file_sha(manifest)!=entry['manifest_sha256']:
            raise ValueError('Unbound fresh source path/identity')
        row=json.loads(manifest.read_text());group=row['source_group']
        if (row['schema']!='recovery_expert_grasp_proposal_v2'
                or row['source_episode']['split']!='train' or row['recovery_split']!='dev'
                or split_group(row['task'],row['instance_id'])!='dev'
                or group!=group_key(row['task'],row['instance_id'])
                or group in groups|existing|protected
                or row['protected_groups_sha256']!=file_sha(protected_path)
                or row['training_approved'] is not False):
            raise ValueError('Fresh calibration overlaps old/protected groups or changes roles')
        if release is None:release=row['original_release_sha256']
        if release!=row['original_release_sha256']:raise ValueError('Mixed original releases')
        for name,sha in row['files'].items():
            path=directory/name
            if Path(name).name!=name or file_sha(path)!=sha:raise ValueError('Changed source bytes')
            if path.suffix=='.png':
                with Image.open(path) as im:im.load()
            files+=1;size+=path.stat().st_size
        with np.load(directory/'prefix.npz',allow_pickle=False) as arrays:
            if (set(arrays.files)!={'action','state'} or arrays['action'].shape!=(row['controls'],23)
                    or arrays['state'].shape!=(row['controls']+1,61)
                    or any(not np.isfinite(arrays[k]).all() for k in arrays.files)):
                raise ValueError('Corrupt raw23/action and raw61/state source')
        groups.add(group);tasks.add(row['task']);controls+=row['controls']
        bindings.append(dict(case=entry['directory'],source_group=group,
            manifest_sha256=entry['manifest_sha256'],original_split='train',usage_role='prospective_calibration_only'))
    if len(groups)!=cfg['selection']['metadata_fresh_sources'] or len(tasks)!=cfg['selection']['metadata_tasks']:
        raise ValueError('Prospective cohort denominator changed')
    result=dict(status='source_closure_and_group_isolation_passed_not_labels',
        config_sha256=file_sha(a.config),source_inventory_sha256=file_sha(source/'manifest.json'),
        selected_observer_sha256=cfg['selected_observer']['sha256'],high_sha256=cfg['high_sha256'],
        old_inventory_bindings=excluded,excluded_unique_groups=len(existing),groups=bindings,
        source_count=len(groups),tasks=len(tasks),dependency_files=files,dependency_bytes=size,
        actual_reference_controls=controls,original_release_sha256=release,
        physical_collection_completed=False,model_predictions_read=False,training_approved=False,
        reserved_test_20_untouched=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','old_inventory_bindings')}))


if __name__=='__main__':main()
