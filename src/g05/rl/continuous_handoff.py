"""Fail-closed dependency checks; a running E3 is never interrupted."""
import json
from pathlib import Path

from g05.rl.protocol import paired_summary

PRIOR_SOURCE = '970896ede236851809573b02b48401f67f5e9dd2'


def dependency_state(directory):
    directory=Path(directory)
    def read(name):
        return json.loads((directory/name).read_text())
    supervisor=read('supervisor.json')
    if supervisor['source_commit']!=PRIOR_SOURCE:
        raise ValueError('Wrong E3 dependency source')
    if supervisor['status'] in ('starting','running'):
        return dict(ready=False,reason='waiting_for_e3_training_and_six_final_episodes')
    if supervisor['status']!='completed' or supervisor.get('exit_code')!=0:
        raise ValueError('E3 failed: do not bypass its final evaluation or auto-restart')
    status=read('status.json'); train=read('training_result.json')
    chosen=read('frozen_final_selection.json'); result=read('result.json')
    evaluations=[json.loads(line) for line in (directory/'evaluations.jsonl').read_text().splitlines()]
    paired=paired_summary(evaluations)
    if (status['phase']!='completed' or status['pending_controls']!=0
            or (directory/'failure.json').exists()
            or chosen['eval_results_used'] is not False
            or chosen['checkpoint']!=train['checkpoint'] or result['checkpoint']!=train['checkpoint']
            or result['checkpoint_sha256']!=chosen['sha256']
            or result['actor_updates']!=train['actor_updates']
            or status['actor_updates']!=train['actor_updates']
            or status['controls']!=result['controls']
            or any(result[k]!=paired[k] for k in ('n','before_successes','after_successes','pairs'))):
        raise ValueError('E3 final selection/result/accounting disagree')
    final=[r for r in evaluations if r['variant']=='rl_fp32']
    if any(r['actor_updates']!=train['actor_updates'] or r['ae_precision']!='float32' for r in final):
        raise ValueError('Final episodes used different weights/precision')
    evaluation_controls=sum(r['controls'] for r in final)
    if train['controls']+evaluation_controls!=result['controls']:
        raise ValueError('TRAIN and final physical counters disagree')
    return dict(ready=True,checkpoint=chosen['checkpoint'],sha256=chosen['sha256'],
                actor_updates=train['actor_updates'],training_controls=train['controls'],
                evaluation_controls=evaluation_controls,final_successes=paired['after_successes'],
                final_episodes=paired['n'])


def owned_pids(directory):
    directory=Path(directory)
    supervisor=json.loads((directory/'supervisor.json').read_text())
    pids=json.loads((directory/'pids.json').read_text())
    owned={supervisor['supervisor'],supervisor['learner']}
    for group in pids['history']:
        owned.update([group['learner'],*group['simulators']])
    return owned


def live_pids(pids, proc=Path('/proc')):
    alive=[]
    for pid in sorted(pids):
        try: stat=(proc/str(pid)/'stat').read_text()
        except FileNotFoundError: continue
        if stat.rsplit(')',1)[1].split()[0]!='Z': alive.append(pid)
    return alive


def check_closed_pool(closed, counts):
    if (closed['clean'] is not True or closed['pending_controls']!=0
            or closed['exits']!=[0,0] or closed['reported_controls']!=counts
            or closed['ledger_controls']!=sum(counts)):
        raise ValueError('Dependency pool did not close with exact physical accounting')
    return sum(counts)
