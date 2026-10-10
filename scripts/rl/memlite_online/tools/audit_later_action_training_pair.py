"""Verify the sole data change and matched, event-balanced SFT schedules."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest,file_sha
from recovery_sft_data import require_training_pool,finite_mixture_schedule


def portable_row(item):
    """Only transport paths may differ; all identities, labels and hashes stay."""
    result=deepcopy(item)
    result['candidate']['actor_input']['rgb'].pop('archive')
    for ref in result['approval']['evidence']:
        ref.pop('path')
    result['approval']['label'].pop('verified_plan_path',None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('original','candidate','owner-review','expert-index','recipe','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    recipe=json.loads(a.recipe.read_text())
    old_sha=file_sha(a.original/'admission/admission.json')
    new_sha=file_sha(a.candidate/'admission/admission.json')
    if old_sha!=recipe['comparison']['control_admission_sha256']:
        raise ValueError('Unpinned original comparison release')
    decisions=json.loads(a.owner_review.read_text())
    allowed={(b['case'],t) for b in decisions['branches'] for t in b['action_steps']}
    counts={};training=[];added=[]
    for pool in ('outcome','planner','action'):
        rows=[]
        for directory,sha in ((a.original,old_sha),(a.candidate,new_sha)):
            _,current=require_training_pool(directory/'admission',pool,sha)
            rows.append(current)
        def key(row):return (row['candidate']['sample_id'],row['approval']['label'].get('member_index'))
        old,new=({key(r):r for r in side} for side in rows)
        if len(old)!=len(rows[0]) or len(new)!=len(rows[1]) or not old.keys()<=new.keys():
            raise ValueError('Duplicate, missing or changed original identities')
        for k,r in old.items():
            if portable_row(r)!=portable_row(new[k]):
                raise ValueError('Not merely a transport relocation: '+pool+' '+str(k))
        extra=[new[k] for k in new.keys()-old.keys()]
        if pool!='action' and extra:raise ValueError('Unrequested new high-layer supervision')
        if pool=='action':
            for r in extra:
                c=r['candidate'];ap=r['approval']
                case=c['task']+'_'+c['source_group'].rsplit(':',1)[1]
                if c['split']!='train' or (case,c['control_step']) not in allowed:
                    raise ValueError('Unapproved added action or protected split')
                if not any((o['candidate']['source_group'],o['approval']['event_id'])==
                           (c['source_group'],ap['event_id']) for o in old.values()):
                    raise ValueError('Added an independent source/event')
                added.append(dict(case=case,control_step=c['control_step'],sample_id=c['sample_id']))
            if {(r['case'],r['control_step']) for r in added}!=allowed:
                raise ValueError('Missing approved window')
            training=[[r for r in side if r['candidate']['split']=='train'] for side in rows]
        counts[pool]=[dict(Counter(r['candidate']['split'] for r in side)) for side in rows]
    experts=json.loads(a.expert_index.read_text())
    by_task={k:[r['candidate'] for r in v] for k,v in experts['rows'].items()}
    schedules=[list(finite_mixture_schedule(side,by_task,
        batch_size=recipe['pilot']['H1_L0_global_batch'],
        maximum_event_passes=recipe['pilot']['event_passes'],seed=recipe['seed'],
        allow_extended_event_fit=recipe['extended_event_fit'],
        anchor_selection_protocol=recipe['L0']['anchor_selection_protocol'])) for side in training]
    projected=[];exposures=[]
    for side,schedule in zip(training,schedules):
        projection=[];counter=Counter()
        for batch in schedule:
            order=[]
            for kind,index in batch['rows']:
                if kind=='new':
                    row=side[index];counter[row['candidate']['sample_id']]+=1
                    order.append([kind,row['candidate']['source_group'],row['approval']['event_id']])
                else:order.append([kind,index])
            projection.append(dict(event_pass=batch['event_pass'],rows=order,new_events=batch['new_events']))
        projected.append(projection);exposures.append(counter)
    if projected[0]!=projected[1]:raise ValueError('Expert/event/rank schedule differs between arms')
    if any(exposures[1][r['sample_id']]==0 for r in added):
        raise ValueError('A new approved phase receives no actual supervision')
    result=dict(schema='later_action_matched_training_audit_v1',status='passed',
        original_admission_sha256=old_sha,candidate_admission_sha256=new_sha,
        owner_review_sha256=file_sha(a.owner_review),recipe_sha256=file_sha(a.recipe),
        counts=counts,new_windows=sorted(added,key=lambda r:(r['case'],r['control_step'])),
        optimizer_steps=0,protected_rows_changed=0,new_independent_events=0,
        common_expert_event_rank_schedule_sha256=digest(projected[0]),
        updates_per_arm=[len(s) for s in schedules],
        schedule_sha256=[digest(s) for s in schedules],
        new_window_exposures={r['case']+':'+str(r['control_step']):exposures[1][r['sample_id']] for r in added})
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
