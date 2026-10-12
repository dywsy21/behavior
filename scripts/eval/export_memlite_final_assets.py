"""CPU-only exact exports for latest full evaluation; no training or scene use."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/rl/memlite_online/code'),str(REPO/'scripts/eval/memlite_final')]
from recovery_corpus import file_sha
from recovery_calibration_launch import bound,read,write_new
from assets import relocate_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=read(a.config);root=Path(cfg['source_root']);out=a.output.resolve()
    if (cfg['schema']!='memlite_final_public100_v1' or out.exists() or not out.is_relative_to(root/'runs')
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('New registered CPU-only pack and frozen source required')
    # Reserve every remaining full champion checkpoint, another in-flight one,
    # terminal EMA and 150GiB safety. Never delete earlier files to pass a gate.
    ckpts=root/'runs/champion_formal_20261012/shared100-v1/checkpoints'
    completed=[int(x.name.split('_')[-1]) for x in ckpts.glob('step_*') if (x/'manifest.json').is_file()]
    completed_count=sum(step in completed for step in range(1000,8001,1000))
    reserve=(8-completed_count+1)*55_000_000_000+(163<<30)
    if shutil.disk_usage(root).free < reserve+(24<<30):
        raise ValueError('Protect all remaining champion saves and free-space safety reserve')
    import torch
    from g05.utils.training.stage1_model import configuration
    torch.set_num_threads(2)
    names=read(root/cfg['expert_release']/'manifest.json')['task_names']
    high,low=(configuration(root,side,names) for side in ('high','low'))
    if file_sha(high['stats_path'])!=cfg['normalization_sha256']['planner'] or file_sha(low['stats_path'])!=cfg['normalization_sha256']['low']:
        raise ValueError('Wrong training normalization')
    out.mkdir(parents=True)
    assets=[root/'models/action_tokenizer.pt',Path(high['stats_path']),Path(low['stats_path'])]
    assets+=sorted((root/'models/qwen3_5_2b_base_processor').iterdir())
    for path in assets:
        if not path.is_file() or path.is_symlink(): raise ValueError('Unexpected asset dependency')
        dst=out/path.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
        if file_sha(dst)!=file_sha(path):raise ValueError('Asset copy changed')
    for side,config in [('high',high),('low',low)]:write_new(out/(side+'-config.json'),relocate_config(config,source_root=root))
    write_new(out/'task_names.json',names)
    for name in ('calibration','observer_fit_result','observer_fit_config'):
        source=bound(REPO if name=='observer_fit_config' else root,cfg[name])
        shutil.copyfile(source,out/(name+'.json'))
    models={}
    for name,spec in cfg['models'].items():
        if name=='planner':
            source=bound(root,dict(path=cfg['existing_planner_export']['source_path'],sha256=cfg['existing_planner_export']['sha256']))
            saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
            if saved['source_sha256']!=spec['sha256'] or saved['state']['step']!=360 or len(saved['model_state_dict'])!=950:
                raise ValueError('Incomplete existing exact planner export')
            models[name]=dict(source_sha256=spec['sha256'],export_sha256=file_sha(source),exact_model_tensors=950,
                transport='existing_separately_verified_export',optimizer_state_exported=False)
            del saved
            continue
        source=bound(root,dict(path=spec['path'],sha256=spec['sha256']));dst=out/(name+'.pt')
        if name=='observer_adapter':
            shutil.copyfile(source,dst)
            if file_sha(dst)!=spec['sha256']:raise ValueError('Observer head copy differs')
            models[name]=dict(source_sha256=spec['sha256'],export_sha256=file_sha(dst),path=dst.name,transport='unchanged_file')
            continue
        saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
        state=saved['model_state_dict']
        if saved['state']['step']!=spec['step'] or len(state)!=spec['tensors']:raise ValueError('Wrong final checkpoint')
        size=sum(v.numel()*v.element_size() for v in state.values())
        if shutil.disk_usage(root).free < reserve+size:raise ValueError('Champion storage reserve changed')
        torch.save(dict(model_state_dict=state,state=dict(step=spec['step']),source_sha256=spec['sha256']),dst)
        new=torch.load(dst,map_location='cpu',mmap=True,weights_only=False)['model_state_dict']
        if set(new)!=set(state):raise ValueError('Missing model tensor')
        for key,v in state.items():
            w=new[key]
            if v.dtype!=w.dtype or v.shape!=w.shape or not torch.equal(v.contiguous().reshape(-1).view(torch.uint8),w.contiguous().reshape(-1).view(torch.uint8)):
                raise ValueError('Nonexact model-only export: '+key)
        models[name]=dict(source_sha256=spec['sha256'],export_sha256=file_sha(dst),path=dst.name,
            exact_model_tensors=len(state),optimizer_state_exported=False)
        del state,new,saved
    files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    write_new(out/'manifest.json',dict(schema='memlite_final_assets_v1',status='exact_model_exports_complete',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        config_sha256=file_sha(a.config),models=models,files=files,optimizer_updates=0,model_forwards=0,
        champion_remaining_save_reservation_bytes=reserve,total_bytes=sum(r['bytes'] for r in files)))
    print(json.dumps(dict(status='complete',manifest_sha256=file_sha(out/'manifest.json'),bytes=sum(r['bytes'] for r in files))))


if __name__=='__main__':main()
