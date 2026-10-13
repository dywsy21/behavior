"""Read only original metric files whose physical batch receipt is committed.

The official evaluator writes JSON non-atomically and closes MP4 afterwards.
Never race those writes, and never hide a malformed committed result.
"""
import json
from pathlib import Path


def completed_metrics(part, *, final=False):
    part=Path(part)
    status_path=part/'status.json'
    if not status_path.is_file():
        if final:raise ValueError('Missing task receipt: '+str(part))
        return []
    status=json.loads(status_path.read_text());task=status['task']
    paths=[];seen=set()
    attempts=sorted((part/'attempts').glob('*.json'))
    for attempt in attempts:
        row=json.loads(attempt.read_text())
        if row['status']!='completed':
            if final:raise ValueError('Unfinished physical attempt: '+str(attempt))
            continue
        ids=row['instance_ids'];counts=row['actual_applied_controls']
        if len(ids)!=len(counts):raise ValueError('Incomplete physical clock receipt')
        for instance,steps in zip(ids,counts):
            key=(instance,row['rollout'])
            if key in seen:raise ValueError('Repeated physical instance receipt')
            seen.add(key)
            path=part/'json'/f'{task}_{instance}_{row["rollout"]}.json'
            value=json.loads(path.read_text())
            if (value['task'],value['instance_id'],value['rollout_id'],value['steps'])!=(task,instance,row['rollout'],steps):
                raise ValueError('Committed official metric differs from actual physical receipt')
            paths.append((path,value))
    if final and (status['status']!='completed' or not paths
            or {p for p,_ in paths}!=set((part/'json').glob('*.json'))):
        raise ValueError('Incomplete/extra final task records')
    return paths
