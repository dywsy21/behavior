"""Eight-rank cold-cache driver + real FLA forward/backward, no model/data.

Use torchrun in a fresh frozen checkout. `shared` is diagnostic only; production
uses `rank`. Each mode needs a new empty output; failures are retained verbatim.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_process_cache import isolated_cache_env,isolated_rank_cache_env
from recovery_cuda_startup import initialize_cuda_backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-mode',choices=('shared','rank'),required=True)
    args=parser.parse_args()
    import torch
    import torch.distributed as dist
    rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if rank!=local or world!=8:
        raise ValueError('This audit requires one node with eight ranks')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Audit requires frozen clean source')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    torch.cuda.set_device(local)
    dist.init_process_group('gloo',timeout=timedelta(seconds=180))
    if rank==0: args.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    cache=isolated_cache_env(args.output)
    if args.cache_mode=='rank': cache=isolated_rank_cache_env(cache,local)
    os.environ.update(cache)
    dist.barrier()
    started=time.monotonic()
    result=dict(rank=rank,world=world,source_commit=commit,cache_mode=args.cache_mode,
                runtime_cache=cache,optimizer_steps=0,model_or_data_loaded=False)
    try:
        result['backend']=initialize_cuda_backend(local)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        torch.manual_seed(4100+rank)
        q,k,v=[torch.randn(1,128,4,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
        g=(-torch.rand(1,128,4,device='cuda',dtype=torch.float32)*.1).requires_grad_()
        beta=torch.rand(1,128,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        out,_=chunk_gated_delta_rule(q,k,v,g,beta,use_qk_l2norm_in_kernel=True)
        loss=out.float().square().mean();loss.backward();torch.cuda.synchronize()
        if not torch.isfinite(out).all() or any(x.grad is None or not torch.isfinite(x.grad).all() for x in (q,k,v,g,beta)):
            raise ValueError('Nonfinite or missing real FLA forward/backward')
        result.update(status='passed',loss=loss.item(),gradient_norms=[x.grad.float().norm().item() for x in (q,k,v,g,beta)])
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc())
    result['elapsed_seconds']=time.monotonic()-started
    with (args.output/f'rank-{rank:03d}.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)
    gathered=[None]*world
    dist.all_gather_object(gathered,result)
    passed=all(row['status']=='passed' for row in gathered)
    if rank==0:
        with (args.output/'result.json').open('x') as stream:
            json.dump(dict(status='passed' if passed else 'failed',source_commit=commit,ranks=gathered),stream,indent=2)
            stream.write('\n')
    dist.destroy_process_group()
    if not passed: raise SystemExit(1)


if __name__=='__main__':main()
