"""Unequal per-slot history/replan clock QA using real causal TRAIN snapshots.

This constructs a diagnostic batch, not a trajectory, annotation, or score.
Rows retain their own actual observations, ledger, instance identity and clock.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from common import atomic_json
from batched_engine import BatchedSFT
from audit_batch import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--early',type=Path,required=True)
    p.add_argument('--later',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    early=torch.load(a.early,map_location='cpu',weights_only=False)
    later=torch.load(a.later,map_location='cpu',weights_only=False)
    if early['task']!=later['task'] or early['indices']!=later['indices'] or len(early['indices'])<2:
        raise ValueError('Need aligned TRAIN snapshots at different planner clocks')
    mixed=deepcopy(early)
    for row,index in enumerate(early['indices']):
        source=later if row%2 else early
        mixed['observations'][row]=deepcopy(source['observations'][row])
        mixed['slots'][index]=deepcopy(source['slots'][index])
        mixed['episode_metadata'][index]=deepcopy(source['episode_metadata'][index])
    clocks=[mixed['slots'][i]['chunks'] for i in mixed['indices']]
    if len(set(clocks))<2:raise ValueError('Histories are not different')
    engine=BatchedSFT(a.output/'no_checkpoints',mode='serial')
    engine.task=mixed['task'];engine.slots=deepcopy(mixed['slots'])
    engine.episode_metadata=deepcopy(mixed['episode_metadata']);engine.requests=mixed['requests']
    torch.cuda.set_rng_state(mixed['cuda_rng'])
    mixed['actions'],mixed['contexts']=engine.infer_native(deepcopy(mixed['observations']),mixed['indices'])
    mixed['inference_mode']='serial'
    result=audit(engine,mixed)
    receipt=engine.verify()
    atomic_json(a.output/'audit.json',dict(kind='diagnostic_mixed_causal_train_rows_not_rollout',
        source_inputs=[str(a.early),str(a.later)],clocks=clocks,result=result,verification=receipt,
        complete=True,engineering_gate_passed=result['engineering_gate_passed']))
    print(json.dumps(dict(passed=result['engineering_gate_passed'],clocks=clocks)))


if __name__=='__main__':main()
