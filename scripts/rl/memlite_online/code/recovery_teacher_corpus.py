"""Validation and causal context for the separate OFFLINE teacher corpus.

Never misrepresent privileged collection as on-policy actor trajectories.
No fields from teacher/physics/noise enter the model observation allowlist.
"""
from copy import deepcopy
import json
import numpy as np

from recovery_corpus import canonical,digest


def episode_identity(source, result, manifest, branch_kind):
    """Content identity survives copying evidence to another directory/server."""
    return dict(run='offline_teacher_'+result['source_commit'][:12],
        episode_id=digest([result['proposal_sha256'],result['full_snapshot_sha256'],
                           branch_kind,manifest['transitions_sha256']])[:24],
        task=source['task'],instance_id=source['instance_id'],split='train',
        source_commit=result['source_commit'],teacher_kind='offline_local_measured_joint_servo_v1',
        actor_model_used=False,original_source_group=source['source_group'],original_episode=source['episode_index'])


def validate_branch(rows,manifest,plans):
    if not rows or len(rows)!=manifest['controls'] or digest(plans)==digest([]):
        raise ValueError('Missing physical branch / issued plan history')
    if [r['control_step'] for r in rows]!=list(range(len(rows))):raise ValueError('Noncontiguous teacher controls')
    if [p['control_step'] for p in plans]!=sorted(set(p['control_step'] for p in plans)) or plans[0]['control_step']!=0:
        raise ValueError('Duplicate / missing teacher decisions')
    for i,row in enumerate(rows):
        for key,width in (('proprio_before',61),('proprio_after',61),('a_intended',23),
                          ('a_sampled_noise',23),('a_executed_raw23',23)):
            v=np.asarray(row[key],dtype=np.float32)
            if v.shape!=(width,) or not np.isfinite(v).all():raise ValueError('Invalid actual teacher vector')
        intended=np.asarray(row['a_intended'],dtype=np.float32)
        noise=np.asarray(row['a_sampled_noise'],dtype=np.float32)
        executed=intended+noise;executed[[14,22]]=np.clip(executed[[14,22]],-1,1)
        if (not np.array_equal(executed,np.asarray(row['a_executed_raw23'],dtype=np.float32))
                or row['action_executed_raw23']!=row['a_executed_raw23'] or row['simulator_apply_ack'] is not True):
            raise ValueError('Intended/noise/clipping/applied provenance mismatch')
        if row['label_kind'] not in ('injected_fault_not_BC','same_state_local_teacher_candidate'):
            raise ValueError('Unknown offline teacher action source')
        if row['label_kind']=='same_state_local_teacher_candidate' and np.any(noise):
            raise ValueError('Noisy action disguised as clean teacher')
        if i and rows[i-1]['proprio_after']!=row['proprio_before']:raise ValueError('Broken physical chain')
        if i and any(rows[i-1]['physical_audit'][k]!=v for k,v in row['physical_before'].items()):
            raise ValueError('Broken target evidence clock')
        event=[p for p in plans if p['control_step']<=i][-1]
        if (row['context']['context_id']!=event['event_sha256']
                or row['context']['active_skills_semantic_json']!=event['active_skills_semantic_json']
                or row['context']['parent_goal']!=event['parent_goal']):
            raise ValueError('Different task / plan / corrective context')
        if event['event_sha256']!=digest({k:v for k,v in event.items() if k!='event_sha256'}):
            raise ValueError('Changed issued plan')
    return True


def branch_histories(anchors,plans):
    from g05.utils.memlite_skill_protocol import append_b_memory_idempotent,canonical_json
    task=anchors[0]['task'].replace('_',' ')
    memory=canonical_json(dict(task_name=task,issued_command_history=[],verified_world_facts=[]))
    previous='None';contexts=[];attempt=0
    for event in plans:
        if (event['memory_before']!=memory or event['previous_intent']!=previous
                or event['memory_update']!=append_b_memory_idempotent(memory,previous,task_name=task)):
            raise ValueError('Teacher memory recurrence differs from deployed protocol')
        if event['decision'] not in ('EXECUTE','RETRY'):raise ValueError('Unverified teacher planner command')
        memory=event['memory_update'];previous=event['active_skills_text'];attempt+=1
        contexts.append(dict(control_step=event['control_step'],context_id=event['event_sha256'],
            parent_goal=event['parent_goal'],issued_decision=event['decision'],
            issued_skills_semantic_json=event['active_skills_semantic_json'],memory=memory,previous_intent=previous,
            intent_started_control_step=event['control_step'],repeated_planning_count=0,attempt_number=attempt,
            history_is_partial=False,logged_event_sha256=event['event_sha256']))
    result=[]
    for row in anchors:
        t=row['control_step'];prior=[c for c in contexts if c['control_step']<=t]
        current=deepcopy(prior[-1]);current.update(observation_control_step=t,served_controls=t-current['intent_started_control_step'])
        before=[c for c in contexts if c['control_step']<t];predecision=None
        if current['control_step']==t and before:
            predecision=deepcopy(before[-1]);predecision.update(observation_control_step=t,served_controls=t-predecision['intent_started_control_step'])
        result.append(dict(schema='recovery_causal_history_v1',sample_id=row['sample_id'],source_episode=row['source_episode'],
            source_group=row['source_group'],control_step=t,observable=current,predecision=predecision,
            evidence=dict(kind='actual_offline_teacher_commands',policy_initialized_at_verified_curriculum_start=True,
                          latest_plan_sha256=current['logged_event_sha256']),new_label=False))
    return result


class PhysicalProposalIndex:
    """Prefix counts preserve causal answers without O(anchors * controls).

    Whole immutable logs may be indexed, but each query reads only counts
    ending before the observation. Future successes cannot change a label.
    """
    def __init__(self,rows,arm):
        self.rows=rows;self.arm=arm;self.held=[0];self.unclean=[0]
        for row in rows:
            self.held.append(self.held[-1]+int(row['physical_audit']['grasp'][arm]=='TRUE'))
            self.unclean.append(self.unclean[-1]+int(row['label_kind']!='same_state_local_teacher_candidate'))

    def held_between(self,start,end):
        return self.held[end]>self.held[start]

    def clean_between(self,start,end):
        return self.unclean[end]==self.unclean[start]


def physical_proposal(rows,t,arm,*,attempt_start,index=None):
    """Past/current evidence only. These are candidates pending human review."""
    if index is not None and (index.rows is not rows or index.arm!=arm):
        raise ValueError('Cross-branch physical prefix index')
    if t<1:return None
    prior=rows[max(0,t-6):t];current=rows[t]['physical_before']
    if len(prior)==6 and all(r['physical_audit']['grasp'][arm]=='TRUE' for r in prior):
        return 'SUCCEEDED'
    lost=len(prior)==6 and all(all(v=='FALSE' for v in r['physical_audit']['grasp'].values()) for r in prior)
    end=max(attempt_start,t-6)
    was_held=(attempt_start==0 and rows[0]['physical_before']['grasp'][arm]=='TRUE') or (
        index.held_between(attempt_start,end) if index is not None else any(
            r['physical_audit']['grasp'][arm]=='TRUE' for r in rows[attempt_start:end]))
    if lost and was_held:return 'FAILED'
    # A genuinely executing new correction: clean actions already applied in
    # this attempt and the hand is moving/closing or contact lacks debounce.
    if t>attempt_start and attempt_start>0 and (index.clean_between(attempt_start,t) if index is not None else
            all(r['label_kind']=='same_state_local_teacher_candidate' for r in rows[attempt_start:t])):
        old=rows[attempt_start]['physical_before']['gripper_aperture'][arm]
        if abs(current['gripper_aperture'][arm]-old)>.002 or current['grasp'][arm]=='TRUE':
            return 'IN_PROGRESS'
    return None
