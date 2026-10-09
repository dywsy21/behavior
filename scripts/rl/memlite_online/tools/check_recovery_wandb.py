"""One isolated ONLINE engineering record; no SFT/rollout or shared login edit.

Uses the existing authorized team's project and a distinct engineering group.
Prints/publishes only non-sensitive receipts, never credentials/config secrets.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

from g05.utils.training.stage1_runtime import atomic_json, init_wandb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Use clean frozen code')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    args.output.mkdir(parents=True,exist_ok=False)
    identity = uuid.uuid4().hex[:12]
    wb = dict(recipe['wandb'],name='recovery-engineering-wiring-only',
              group=recipe['wandb']['group']+'-engineering')
    public = dict(id=identity,entity=wb['entity'],project=wb['project'],group=wb['group'],
                  scope='engineering connectivity only; not model training',
                  source_commit=commit,g05_optimizer_steps=0)
    atomic_json(args.output/'receipt.json',dict(public,status='STARTING'))
    run = None
    try:
        run = init_wandb(wb,args.output,run_id=identity,resume=False,
                        metadata=dict(engineering_only=True,model_training=False,code_commit=commit))
        run.log({'engineering/probe':1.,'engineering/g05_optimizer_steps':0})
        url = run.url
        run.finish()
        run = None
        import wandb
        match = False
        for attempt in range(10):
            api = wandb.Api(timeout=15)
            remote = api.run(f"{wb['entity']}/{wb['project']}/{identity}")
            values = remote.history(samples=10,keys=['engineering/probe','engineering/g05_optimizer_steps'],pandas=False)
            match = any(r.get('engineering/probe')==1. and r.get('engineering/g05_optimizer_steps')==0 for r in values)
            if match:
                break
            if attempt < 9:
                time.sleep(2)
        if not match:
            raise RuntimeError('No matching committed online engineering history')
        receipt = dict(public,status='PASSED',url=url,online_history_readback=True)
        atomic_json(args.output/'receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    except Exception as exc:
        if run is not None:
            run.finish(exit_code=1)
        atomic_json(args.output/'receipt.json',dict(public,status='FAILED',error_type=type(exc).__name__))
        raise RuntimeError('Engineering W&B check failed; see non-sensitive receipt') from None


if __name__ == '__main__':
    main()
