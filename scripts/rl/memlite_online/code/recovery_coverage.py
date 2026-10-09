"""Metadata coverage is not physical recovery coverage or label approval."""
from collections import defaultdict
import json

from recovery_corpus import digest,group_key,split_group


def skill_family(verb):
    if verb=='GRASP':return 'grasp'
    if verb in ('NAVIGATE','TURN_TO'):return 'navigation'
    if verb.startswith('PLACE_') or verb=='RELEASE':return 'placement_release'
    if verb.startswith(('OPEN_','CLOSE_')):return 'articulation'
    if verb in ('TURN_ON_SWITCH','TURN_OFF_SWITCH','PRESS','IGNITE'):return 'toggle_contact'
    if verb in ('HOLD','LIFT'):return 'hold_transport'
    if verb in ('HANDOVER','INSERT','ATTACH','HANG'):return 'transfer_assembly'
    return 'tool_material_operation'


def select_grasp_sources(episodes, protected, tasks, counts, existing_groups=(), verb='GRASP'):
    """Outcome-blind, task-balanced first sweep; existing failures also count.

    Counts are inventory coverage targets, not a global experiment quota. The
    caller can expand this inventory without retrying a failed source for luck.
    """
    existing=set(existing_groups); queues=defaultdict(list); coverage=[]
    candidates=defaultdict(list)
    for ep in episodes:
        task=ep['task_name']; raw=ep['row']; group=group_key(task,raw['task_instance_id'])
        if task not in tasks or ep['split']!='train' or group in protected:continue
        for segment in ep['segments']:
            skills=json.loads(segment['semantic'])
            if (len(skills)==1 and skills[0]['verb']==verb and skills[0].get('target')
                    and not skills[0].get('unbound_relation')
                    and (verb not in ('PLACE_IN','PLACE_ON') or
                         (skills[0].get('destination') and skills[0]['destination']!=skills[0]['target']))):
                split=split_group(task,raw['task_instance_id'])
                candidates[(task,split)].append((ep,segment,split));break
    for task in tasks:
        for split,count in counts.items():
            rows=sorted(candidates[(task,split)],key=lambda r:(r[1]['end'],digest([task,r[0]['row']['task_instance_id']])))
            retained=sum(group_key(task,r[0]['row']['task_instance_id']) in existing for r in rows)
            fresh=[r for r in rows if group_key(task,r[0]['row']['task_instance_id']) not in existing]
            chosen=fresh[:max(0,count-retained)];queues[task].extend(chosen)
            coverage.append(dict(task=task,split=split,eligible=len(rows),retained_sources=retained,
                new_sources=len(chosen),requested_source_coverage=count,missing=max(0,count-retained-len(chosen))))
    # Every task receives its first case before any receives its second.
    selected=[]
    for n in range(max((len(v) for v in queues.values()),default=0)):
        selected.extend(queues[task][n] for task in tasks if len(queues[task])>n)
    return selected,coverage


def build_catalog(episodes,protected):
    tasks={};eligible=defaultdict(dict);seen_groups=set();excluded=defaultdict(int)
    for episode in episodes:
        raw=episode['row'];task=episode['task_name'];index=raw['task_index']
        info=tasks.setdefault(task,dict(task=task,task_index=index,original_episodes=0,eligible_groups=0,verbs={}))
        if info['task_index']!=index:raise ValueError('Conflicting task identity')
        info['original_episodes']+=1;group=group_key(task,raw['task_instance_id'])
        if episode['split']!='train' or group in protected:
            excluded['protected_or_original_nontrain']+=1;continue
        if group in seen_groups:raise ValueError('Duplicate original task-instance source')
        seen_groups.add(group);info['eligible_groups']+=1
        split=split_group(task,raw['task_instance_id'])
        for segment in episode['segments']:
            skills=json.loads(segment['semantic'])
            for member,skill in enumerate(skills):
                verb=skill['verb'];key=(task,verb,split)
                counts=info['verbs'].setdefault(verb,dict(family=skill_family(verb),train_groups=0,dev_groups=0,
                    annotated_members=0,fully_bound_members=0,single_member_groups=0))
                counts['annotated_members']+=1
                if skill.get('unbound_relation') or not skill.get('target'):
                    excluded['unbound_members']+=1;continue
                counts['fully_bound_members']+=1
                row=dict(task=task,task_index=index,instance_id=raw['task_instance_id'],episode_index=raw['episode_index'],
                    source_group=group,split=split,verb=verb,family=skill_family(verb),member_index=member,
                    bundle_size=len(skills),segment_start=segment['start'],segment_end=segment['end'],
                    semantic=segment['semantic'],parent=segment['parent'])
                # Single-member seeds first. Parallel bundles remain in the
                # inventory for their own per-member physical adapter; no flattening.
                rank=(len(skills)!=1,segment['end'],digest([group,segment['semantic'],member]))
                current=eligible[key].get(group)
                if current is None or rank<current[0]:eligible[key][group]=(rank,row)
    queue=[]
    for (task,verb,split),groups in sorted(eligible.items()):
        rows=[r[1] for r in sorted(groups.values(),key=lambda x:x[0])]
        tasks[task]['verbs'][verb][split+'_groups']=len(rows)
        tasks[task]['verbs'][verb]['single_member_groups']+=sum(r['bundle_size']==1 for r in rows)
        for order,row in enumerate(rows):queue.append(dict(row,candidate_order=order))
    return dict(schema='recovery100_metadata_catalog_v1',tasks=sorted(tasks.values(),key=lambda r:r['task_index']),
        excluded=dict(excluded),source_groups=len(seen_groups),queue=queue,
        physical_recovery_verified=False,training_approved=False,
        note='Annotation locations only; no end-of-segment success or failure inferred')
