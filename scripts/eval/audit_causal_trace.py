"""Read-only audit of real per-episode memory, intent and raw23 execution logs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import digest


def audit(rows, *, allow_incomplete=False):
    sessions={};slots={};counts=Counter();tokens=set();models=None
    for r in rows:
        kind=r['kind'];identity=r['identity'];episode=identity['episode'];slot=r['slot'];counts[kind]+=1
        if kind=='begin':
            memory=json.loads(r['memory'])
            if (episode in sessions or slot in slots or r['initial_control_step']!=0 or r['teacher_prefix_controls']!=0
                    or memory!={'issued_command_history':[],'task_name':identity['task'],'verified_world_facts':[]}):
                raise ValueError('Reused episode/slot or carried-over initial memory')
            if models is None:models=identity['models']
            if models!=identity['models']:raise ValueError('Model mixture in a frozen run')
            sessions[episode]=dict(identity=identity,slot=slot,step=0,pending=None,goal=None,revision=0,closed=False)
            slots[slot]=episode
            continue
        if episode not in sessions:raise ValueError('Unknown episode')
        s=sessions[episode]
        if s['closed'] or s['identity']!=identity or s['slot']!=slot or slots.get(slot)!=episode:
            raise ValueError('Cross-episode/task/slot identity')
        if kind=='plan':
            p=r['result'];prefix=p['causal_input'];memory=json.loads(prefix['memory'])
            if (s['pending'] is not None or p['control_step']!=s['step'] or p['revision']!=s['revision']+1
                    or p['goal']['task']!=identity['task'] or prefix['task_name']!=identity['task']
                    or memory['task_name']!=identity['task'] or p['physical_success_asserted'] is not False):
                raise ValueError('Foreign/stale high-level memory or intent')
            s.update(goal=p['goal'],revision=p['revision'])
        elif kind=='action':
            d=r['display'];actions=r['raw_actions'];observer=d['observer']
            if (s['pending'] is not None or r['token'] in tokens or d['control_step']!=s['step']
                    or d['revision']!=s['revision'] or d['goal']!=s['goal']
                    or observer['attempt_generation']!=d['attempt_generation']
                    or observer['control_step']>s['step'] or len(actions)!=16 or any(len(a)!=23 for a in actions)):
                raise ValueError('Low intent, observer, token or action-clock mismatch')
            tokens.add(r['token']);s['pending']=r
        elif kind=='ack':
            p=s['pending'];n=r['consumed_controls']
            if (p is None or r['token']!=p['token'] or not 1<=n<=16
                    or r['control_step']!=s['step']+n or r['unused_controls']!=16-n):
                raise ValueError('Foreign/duplicate/unordered action ACK')
            controls=[dict(control_step=s['step']+i,simulator_apply_ack=True,action_executed_raw23=a)
                for i,a in enumerate(p['raw_actions'][:n])]
            if digest(controls)!=r['ack_sha256']:raise ValueError('Applied raw23 differs from offered prefix')
            s.update(step=r['control_step'],pending=None)
        elif kind=='observer':
            if r['control_step']!=s['step'] or r['revision']!=s['revision']:
                raise ValueError('Observer used another intent or action clock')
        elif kind=='close':
            if s['pending'] is not None or r['episode']!=episode or r['control_step']!=s['step'] or r['optimizer_updates']!=0:
                raise ValueError('Unacknowledged or foreign episode close')
            s['closed']=True;del slots[slot]
        else:raise ValueError('Unknown trace event')
    incomplete = bool(slots) or any(not s['closed'] for s in sessions.values())
    if not sessions or (incomplete and not allow_incomplete):raise ValueError('Incomplete trace')
    if allow_incomplete and any(s['pending'] is not None for s in sessions.values()):
        raise ValueError('Incomplete trace still has unacknowledged actions')
    return dict(status='incomplete_causal_trace_audit' if incomplete else 'complete_causal_trace_audit',
        episodes=len(sessions),closed_episodes=sum(s['closed'] for s in sessions.values()),
        unclosed_episodes=[dict(identity=s['identity'],slot=s['slot'],applied_controls=s['step'])
            for s in sessions.values() if not s['closed']],events=dict(counts),
        applied_controls=sum(s['step'] for s in sessions.values()),models=models,
        tasks=sorted({s['identity']['task'] for s in sessions.values()}),
        checks=['fresh empty per-episode memory','full identity and fixed slot on every event',
            'high-to-low exact current intent','observer attempt and applied clock',
            'unique tokens and exact raw23 physical ACK digest','final partial chunk acknowledged'],
        proves_full_task_success=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    content=a.trace.read_bytes();r=audit([json.loads(l) for l in content.splitlines()])
    r.update(trace_sha256=hashlib.sha256(content).hexdigest(),trace=str(a.trace.resolve()))
    with a.output.open('x') as stream:json.dump(r,stream,indent=2);stream.write('\n')
    print(json.dumps(r))
