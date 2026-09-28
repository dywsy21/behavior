"""Reconcile stopped physical controls, freeze one saved-rollout PPO continuation."""
import json
import math
from pathlib import Path
from common import PREVIOUS,OUT,PARENT,PARENT_SHA,commit,sha,save


def main():
    previous=json.loads((PREVIOUS/'manifest.json').read_text())
    supervisor=json.loads((PREVIOUS/'supervisor.json').read_text())
    status=json.loads((PREVIOUS/'status.json').read_text())
    pids=json.loads((PREVIOUS/'pids.json').read_text())
    for pid in [supervisor['supervisor'],pids['learner'],*pids['simulators']]:
        path=Path(f'/proc/{pid}/stat')
        if path.exists() and path.read_text().rsplit(')',1)[1].split()[0]!='Z':
            raise ValueError('Previous run still active')
    if supervisor['status']!='failed' or status['actor_updates']!=0 or status['critic_updates']!=4:
        raise ValueError('Expected stopped, actor-rolled-back first batch only')
    counts=[]; success_events=[]
    for w in (0,1):
        steps=[json.loads(line) for line in (PREVIOUS/f'worker_{w}/steps.jsonl').read_text().splitlines()]
        if [r['control'] for r in steps]!=list(range(1,len(steps)+1)):
            raise ValueError('Physical control log has gaps')
        counts.append(len(steps))
        success_events.extend(dict(worker=w,control=r['control'],episode=r['episode'])
                              for r in steps if r['success'] and r['phase']=='policy')
    actual=sum(counts)
    if not status['controls']<=actual<=status['controls']+status['pending_controls']:
        raise ValueError('Unreconciled in-flight controls')
    checkpoint=PREVIOUS/'rl_batch_001_updates_0000.pt'
    receipt=json.loads(checkpoint.with_suffix('.json').read_text())
    if receipt['actor_updates'] or sha(checkpoint)!=receipt['sha256'] or sha(PARENT)!=PARENT_SHA:
        raise ValueError('Changed actor/parent/checkpoint')
    rollout=PREVIOUS/'rollout_000.pt'
    used=previous['prior_controls']+actual
    elapsed=previous['prior_active_seconds']+supervisor['seconds']
    available=7200-math.ceil(elapsed)
    if available<180 or used>10000: raise ValueError('Insufficient original budget')
    OUT.mkdir(exist_ok=False,parents=True)
    previous.update(source_commit=commit(),continued_from=str(PREVIOUS),prior_controls=used,
        prior_active_seconds=elapsed,max_controls=0,remaining_control_budget=10000-used,
        max_active_wall_seconds=min(900,available),entry='offline_update',
        source_checkpoint=dict(path=str(checkpoint),sha256=receipt['sha256']),
        source_rollout=dict(path=str(rollout),sha256=sha(rollout)),
        reconciled_physical_controls=dict(workers=counts,status=status['controls'],
            pending=status['pending_controls'],actual=actual,success_events=success_events),
        update_amendment=dict(reason='first fixed LR candidate full-path KL36.7274 rolled back',
            scales=[1.,.1,.01,.001,.0001],max_trials_per_minibatch=5,max_accepted_updates=10,
            max_kl_every_retained_path=.01,require_minibatch_surrogate_nonworsening=True,
            extra_environment_controls=0,critic_updates_this_continuation=0))
    save(OUT/'manifest.json',previous)
    print(json.dumps(dict(output=str(OUT),prior_controls=used,prior_seconds=elapsed,
                         offline_wall_limit=previous['max_active_wall_seconds'],new_controls=0)))


if __name__=='__main__': main()
