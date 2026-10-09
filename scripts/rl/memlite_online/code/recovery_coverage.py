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
