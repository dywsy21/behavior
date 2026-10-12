"""Existing Stage1 parent: two complete forward paths, NOT a final360 comparison.

Only the already-transferred exact inputs/assets are consumed. Incomplete
candidate/control model exports are never opened or treated as verified.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/eval'),
             str(REPO/'scripts/infra'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import read,bound,write_new
from recovery_corpus import file_sha,digest
from recovery_handoff_exposure import ENGINEERING_PARENT_SHA
from recovery_handoff_portable import check_prefix_rows,materialize_arch
from recovery_handoff_inference import tree_sha
from recovery_process_cache import isolated_cache_env


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root=Path('/run/ti/rl_memlite_stage1_20261006')
    pack=root/'runs/handoff_portable_20261012/pack-v1'
    if (not args.output.resolve().is_relative_to(root/'runs/handoff_portable_20261012')
            or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='0'
            or Path(sys.executable).absolute()!=Path('/home/tione/notebook/baselines/GalaxeaVLA/.venv/bin/python')
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('Fresh engineering output, registered env/GPU and frozen source required')
    def idle():
        if subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('Do not use a GPU owned by another process')
    idle()
    manifest=read(bound(root,dict(path='incoming/handoff-portable-manifest-v1.json',
        sha256='e9af5f5b21c3858c9a4a3cf42c9b8723ccee23d8688f23947c4b4a89d4dcf876')))
    accepted=read(bound(root,dict(path='incoming/handoff-cpu-prefix-v3.json',
        sha256='ab24503dfeb8a8cb1673d812e11b462c56a4ea5b1c32d8f83eab046e416b39ed')))
    verified=[]
    for row in manifest['files']:
        if row['path'] in ('candidate-model.pt','control-model.pt','cpu-prefix-audit.json'):
            continue
        actual=bound(pack,dict(path=row['path'],sha256=row['sha256']))
        if actual.stat().st_size!=row['size']:raise ValueError('Incomplete shared input/asset')
        verified.append(row['path'])
    checkpoint=bound(root,dict(path='models/stage1/high/step_00048045_save_0027.pt',sha256=ENGINEERING_PARENT_SHA))
    args.output.mkdir(parents=True);os.environ.update(isolated_cache_env(args.output));started=time.monotonic()
    import torch
    torch.set_num_threads(2);torch.manual_seed(17)
    original=torch.load(pack/'inputs.pt',map_location='cpu',weights_only=False)
    if tree_sha(original)!=manifest['input_tree_sha256']:raise ValueError('Changed transported input tensors')
    check_prefix_rows([r[-1] for r in original],accepted)
    chosen=[next(row for row in original if row[1]==split) for split in ('train_continuation','train_handoff')]
    if any(row[3]['original_identity']['split']!='train' for row in chosen):raise ValueError('No EVAL engineering')
    common=dict(purpose='ENGINEERING_PARENT_ONLY_NOT_FINAL360_COMPARISON',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        checkpoint_sha256=ENGINEERING_PARENT_SHA,checkpoint_step=48045,optimizer_updates=0,
        data_split='train',target_free=True,physical_success_measured=False,
        fixed_selection='first_original_continuation_and_handoff_before_prediction',
        verified_shared_files=verified,full_portable_model_pack_accepted=False,
        schedule_sha256=digest([(r[0],r[1],r[2]) for r in chosen]),runtime_node='RTX10383')
    write_new(args.output/'cpu-prefix-audit.json',dict(common,status='passed',rows=[r[-1] for r in chosen]))
    config=dict(arch=materialize_arch(read(pack/'arch.json'),pack))
    idle()
    from diagnose_memlite_handoff_train import run_diagnostic
    run_diagnostic(args,checkpoint,config,chosen,common,started,expected_checkpoint_step=48045)


if __name__=='__main__':main()
