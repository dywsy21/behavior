"""Real final047 serial/batched generation QA on four predeclared TRAIN prefixes.

This checks GPU batching while the remaining models are being transferred. It
does not substitute for a complete high/low/observer simulator preflight.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/rl/memlite_online/code'),str(REPO/'scripts/eval/memlite_sft100')]
from recovery_calibration_launch import read,bound,write_new
from recovery_corpus import file_sha
from recovery_handoff_portable import materialize_arch
from recovery_process_cache import isolated_cache_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();pack=a.pack.resolve()
    if (a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='0'
            or subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('Fresh frozen engineering run on idle RTX GPU0 required')
    manifest=read(bound(pack,dict(path='manifest.json',sha256='e9af5f5b21c3858c9a4a3cf42c9b8723ccee23d8688f23947c4b4a89d4dcf876')))
    wanted={'arch.json','inputs.pt','candidate-model.pt','cpu-prefix-audit.json'}
    for row in manifest['files']:
        if row['path'] in wanted or row['path'].startswith('assets/'):
            bound(pack,dict(path=row['path'],sha256=row['sha256']))
    if not wanted<={r['path'] for r in manifest['files']}:raise ValueError('Missing engineering dependencies')
    candidate=manifest['models']['candidate']
    if candidate['original']['checkpoint']['sha256']!='047152f1ced65db0ecdf9348ced13980fbae3c512e914e14fccfe871e30ed38d':
        raise ValueError('Not final047')
    a.output.mkdir(parents=True);os.environ.update(isolated_cache_env(a.output));started=time.monotonic()
    import torch
    from recovery_cuda_startup import initialize_cuda_backend
    initialize_cuda_backend(0);torch.set_num_threads(2);torch.manual_seed(17)
    from g05.utils.training.stage1_model import restore_model
    from g05.models.g05.qwen35 import vision
    from g05.models.g05.inferencer import PolicyInferencer
    from g05.models.kv_cache import SparseKVCache
    from sparse_cache import indexed_cache_freeze
    from train_memlite_recovery import parameter_digest
    vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    rows=torch.load(pack/'inputs.pt',map_location='cpu',weights_only=False)
    if len(rows)!=383 or any(rows[i][3]['original_identity']['split']!='train' for i in range(4)):
        raise ValueError('Only fixed original TRAIN prefixes')
    selected=[dict(samples=deepcopy(rows[i][5]),pixel_values=rows[i][6]['pixel_values']) for i in range(4)]
    saved=torch.load(pack/'candidate-model.pt',map_location='cpu',mmap=True,weights_only=False)
    if saved['source_sha256']!=candidate['original']['checkpoint']['sha256'] or saved['state']['step']!=360:
        raise ValueError('Wrong final high export')
    cfg=dict(arch=materialize_arch(read(pack/'arch.json'),pack))
    model,restore=restore_model(cfg,'high',state=saved['model_state_dict']);del saved
    model.requires_grad_(False).eval().cuda();before=parameter_digest(model,frozen=True)
    def generate(indices):
        batch=PolicyInferencer._collate([selected[i] for i in indices],padding_input_id=0)
        pixels={k:v.cuda() for k,v in batch['pixel_values'].items()}
        begin=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),indexed_cache_freeze(model.model.ar_helper,SparseKVCache):
            events=model.generate_high_level(batch['samples'],pixels,temperature=0.)['planner_events']
        torch.cuda.synchronize()
        if len(events)!=len(indices):raise ValueError('Batched output count changed')
        return dict(indices=indices,events=events,seconds=time.monotonic()-begin)
    outputs=[generate([i]) for i in range(4)]
    reference={i:out['events'][0] for i,out in enumerate(outputs)}
    for indices in ([0,1],[0,1,2,3],[3,1,0,2],[2]):
        out=generate(list(indices));out['matches_serial']=[e==reference[i] for i,e in zip(indices,out['events'],strict=True)]
        outputs.append(out)
    after=parameter_digest(model,frozen=True)
    if before!=after or any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Engineering probe modified frozen model')
    write_new(a.output/'result.json',dict(status='completed',purpose='HIGH_BATCH_ENGINEERING_ONLY_NOT_FULL_PREFLIGHT_OR_SR',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        planner_source_sha256=candidate['original']['checkpoint']['sha256'],
        model_export_sha256=candidate['sha256'],positions=[0,1,2,3],split='train',model_forwards=8,
        optimizer_updates=0,simulator_steps=0,restore=restore,parameters_before=before,parameters_after=after,
        all_events_match_serial=all(all(x['matches_serial']) for x in outputs[4:]),outputs=outputs,
        max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started))


if __name__=='__main__':main()
