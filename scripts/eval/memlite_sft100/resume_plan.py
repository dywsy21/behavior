"""Deterministic remaining-case partitioning, independent of scores/outcomes."""
from collections import defaultdict


def remaining_parts(tasks, completed, horizons, num_envs):
    if num_envs not in (2,4):
        raise ValueError('Only validated 2/4-env capacity choices')
    expected={(task,i,0) for task in tasks for i in range(301,311)}
    keys=[(r['task'],r['instance_id'],r['rollout_id']) for r in completed]
    if len(keys)!=len(set(keys)) or not set(keys)<=expected:
        raise ValueError('Duplicate or foreign inherited results')
    done=set(keys);parts=[]
    for task in tasks:
        remaining=[i-301 for i in range(301,311) if (task,i,0) not in done]
        buckets=defaultdict(list)
        while remaining:
            n=next(n for n in (num_envs,2,1) if n<=len(remaining))
            buckets[n].extend(remaining[:n]);del remaining[:n]
        for n,indices in buckets.items():
            parts.append(dict(id=f'{task}_n{n}',task=task,num_envs=n,indices=indices,
                control_steps=len(indices)*(horizons[task]+1),
                scheduling_weight=len(indices)/n*(horizons[task]+1)))
    # Longest processing time first reduces late-stage GPU idling. Ties stable.
    parts.sort(key=lambda row:(-row['scheduling_weight'],row['id']))
    actual={(p['task'],i+301,0) for p in parts for i in p['indices']}
    if actual != expected-done or sum(len(p['indices']) for p in parts)!=len(actual):
        raise ValueError('Remaining-case coverage is not exact')
    return parts
