"""Read-only task/skill coverage, including failures and still-running attempts.

Metadata, attempted simulation, physical candidates and signed human approval
are separate counters. No approval or result-head training is triggered here.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha,group_key,split_group
from recovery_coverage import skill_family


def report(catalog,sources,collections,reviews):
    planned={};tasks={t['task']:dict(task=t['task'],task_index=t['task_index'],skills={}) for t in catalog['tasks']}
    signed={(r['case'],r['branch']):r for review in reviews for r in review['branches']}
    for root in sources:
        for entry in json.loads((root/'manifest.json').read_text())['cases']:
            path=root/entry['directory']/'manifest.json';sha=file_sha(path)
            if sha!=entry['manifest_sha256']:raise ValueError('Changed source inventory')
            source=json.loads(path.read_text());task=source['task'].replace('_',' ')
            group=group_key(task,source['instance_id']);skills=json.loads(source['selected_segment']['semantic'])
            if (source['source_episode']['split']!='train' or source['source_group']!=group
                    or source['recovery_split']!=split_group(task,source['instance_id']) or len(skills)!=1):
                raise ValueError('Source identity/split mismatch')
            verb=skills[0]['verb'];key=(group,verb)
            if key in planned:
                if planned[key]['proposal_sha256']!=sha:raise ValueError('Conflicting original source')
                continue
            row=dict(task=task,verb=verb,source_group=group,split=source['recovery_split'],case=entry['directory'],
                proposal_sha256=sha,started=False,closed=False,physical_candidate=False,human_approved=False,status='pending')
            found=[]
            for collection in collections:
                directory=collection/entry['directory'];source_path=directory/'source.json'
                if not source_path.exists():continue
                actual=json.loads(source_path.read_text())
                if actual!=source:continue  # Different skill attempt at same instance is not this source.
                found.append(directory)
            if len(found)>1:raise ValueError('Multiple original attempts: select first-attempt roots explicitly')
            if found:
                directory=found[0];closed=(directory/'result.json').exists()
                path=directory/('result.json' if closed else 'status.json')
                state=json.loads(path.read_text())
                row.update(started=True,closed=closed,status=state['status'],receipt_path=str(path),
                    actual_controls_last_receipt=state.get('actual_controls',0))
                for branch in state.get('branches',[]):
                    if branch['physical_recovery_candidate']:row['physical_candidate']=True
                    approval=signed.get((entry['directory'],branch['kind']))
                    if approval:
                        if file_sha(directory/branch['kind']/'manifest.json')!=approval['manifest_sha256']:
                            raise ValueError('Human review points to changed branch')
                        row['human_approved']=True
            planned[key]=row
    rows=list(planned.values());fields=('started','closed','physical_candidate','human_approved')
    for row in rows:
        stats=tasks[row['task']]['skills'].setdefault(row['verb'],dict(family=skill_family(row['verb']),
            planned=0,started=0,closed=0,physical_candidate=0,human_approved=0,by_split=defaultdict(Counter)))
        stats['planned']+=1;stats['by_split'][row['split']]['planned']+=1
        for field in fields:
            stats[field]+=int(row[field]);stats['by_split'][row['split']][field]+=int(row[field])
    totals=dict(planned_sources=len(rows),planned_tasks=len({r['task'] for r in rows}),
        planned_independent_groups=len({r['source_group'] for r in rows}))
    for field in fields:
        totals[field+'_sources']=sum(r[field] for r in rows)
        totals[field+'_tasks']=len({r['task'] for r in rows if r[field]})
        totals[field+'_independent_groups']=len({r['source_group'] for r in rows if r[field]})
    return dict(schema='recovery100_coverage_receipt_v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        totals=totals,statuses=dict(Counter(r['status'] for r in rows)),tasks=sorted(tasks.values(),key=lambda t:t['task_index']),
        cases=rows,ready_for_joint_head_training=False,
        note='Coverage only. New semantic review, split/class calibration and head admission still required; no actor SR.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--sources',type=Path,nargs='+',required=True)
    p.add_argument('--collections',type=Path,nargs='+',required=True)
    p.add_argument('--reviews',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=report(json.loads(a.catalog.read_text()),a.sources,a.collections,[json.loads(p.read_text()) for p in a.reviews])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['totals'],indent=2))


if __name__=='__main__':main()
