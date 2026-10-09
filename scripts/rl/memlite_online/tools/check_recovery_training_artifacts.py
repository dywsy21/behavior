"""Independent CPU re-load of the bounded two-update A800 engineering runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    from g05.utils.training.stage1_runtime import load_checkpoint
    torch.set_num_threads(4)
    result=json.loads((a.run/'result.json').read_text())
    schedule=json.loads((a.run/'schedule.json').read_text())
    ledger=json.loads(Path(str(a.run)+'.supervisor/ledger.json').read_text())
    component=schedule['binding']['component'];expected={'L0':322,'H1':326}[component]
    if (result['status']!='completed_finite_schedule' or result['step']!=2 or not result['engineering_only']
            or ledger['status']!='EXITED' or ledger['returncode']!=0 or ledger['attempt']!=2
            or ledger['consumed_seconds']>1800 or result['frozen_before_sha256']!=result['frozen_after_sha256']):
        raise ValueError('Not a successfully completed bounded saved-and-resumed engineering run')
    saved,receipt=load_checkpoint(a.run/'checkpoints',schedule['fingerprint'])
    states=saved['optimizer_state_dict']['state']
    if (saved['state']['step']!=2 or saved['state']['save_sequence']!=2 or len(states)!=expected
            or len(saved['rng_by_rank'])!=8):raise ValueError('Missing optimizer/cursor/rank state')
    for state in states.values():
        if (set(state)!={'step','exp_avg','exp_avg_sq'} or int(state['step'])!=2
                or any(not torch.isfinite(t).all() for t in state.values())):
            raise ValueError('Nonfinite/incomplete Adam state')
    for rng in saved['rng_by_rank']:
        if (set(rng)!={'python','numpy','torch','cuda_current'} or rng['cuda_current'] is None
                or rng['cuda_current'].numel()==0 or rng['torch'].numel()==0):
            raise ValueError('Incomplete rank RNG')
    updates=[json.loads(x) for x in (a.run/'updates.jsonl').read_text().splitlines()]
    if [r['update'] for r in updates]!=[1,2] or any(r['real_samples']!=16 or r['world_size']!=8 for r in updates):
        raise ValueError('Repeated/skipped cursor or unexpected batch')
    out=dict(schema='recovery_engineering_checkpoint_audit_v1',component=component,run=str(a.run),
        status='passed_cpu_reload',checkpoint_path=str(a.run/'checkpoints'/receipt['path']),
        checkpoint_sha256=receipt['sha256'],checkpoint_bytes=(a.run/'checkpoints'/receipt['path']).stat().st_size,
        model_state_tensors=len(saved['model_state_dict']),adam_parameter_states=len(states),
        adam_step=2,rng_ranks=8,global_updates=2,global_expert_rows=32,
        frozen_unchanged=result['frozen_before_sha256'],source_commit=result['source_commit'],
        parent_sha256=result['parent_sha256'],result_sha256=file_sha(a.run/'result.json'),
        cumulative_seconds=ledger['consumed_seconds'],evaluation=result['evaluation'],wandb_url=result['wandb_url'],
        scientific_effect_measured=False,formal_training=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':main()
