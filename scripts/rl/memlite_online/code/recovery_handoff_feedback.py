"""Causal, observable-only requests for original-TRAIN planner handoffs.

The observer consumes the PREVIOUS issued intent, not the next teacher label.
At a period128 decision the last five cadence16 observations all lie after
the preceding planning call. They suffice for the two fresh four-frame head
checks; this is not five fabricated outcomes or an executed actor rollout.
"""
import json

from recovery_corpus import digest, group_key
from recovery_periodic_teacher import command_change_class

SCHEMA = 'periodic_teacher_frozen_observer_requests_v1'


def requests_for_candidate(row, frame_indices, *, excluded_groups):
    from g05.utils.memlite_skill_protocol import canonical_json, parse_active_skills_text
    identity, p = row['identity'], row['projection']
    if (identity['split'] != 'train' or identity['instance'] >= 301
            or row['teacher_kind'] != 'annotation_periodic_command_trace_not_actor_rollout'
            or row['annotation_trace']['period'] != 128
            or p['known_previous_outcome'] != 'UNKNOWN' or p['outcome_target'] != 'UNKNOWN'
            or p['outcome_supervision_mask'] or p['low_action_supervision_mask']
            or p['next_decision'] != 'EXECUTE'):
        raise ValueError('Only accepted TRAIN high-intent candidates, never outcome/BC labels')
    group = group_key(p['task_name'], identity['instance'])
    if group in excluded_groups:
        return None, 'protected_source'
    if p['previous_intent'] == 'None':
        return None, 'initial_no_previous_intent'
    members = parse_active_skills_text(p['previous_intent'])
    if not members or any(m['verb'] != 'GRASP' for m in members):
        return None, 'outside_grasp_calibration'
    feedback = json.loads(p['execution_feedback'])
    elapsed = feedback['same_intent_controls']
    if (feedback['estimated_bundle_outcome'] != 'UNKNOWN'
            or any(m['estimated_outcome'] != 'UNKNOWN' or m['confidence'] != 0
                   for m in feedback['estimated_member_outcomes'])
            or type(elapsed) is not int or elapsed < 128 or elapsed % 128):
        raise ValueError('Expected original unknown estimates and cadence128 prior clock')
    frame = identity['frame']; first = row['annotation_trace']['first_frame']
    if frame <= first or (frame - first) % 128:
        raise ValueError('Decision not on its original period128 clock')
    frames = list(range(frame - 64, frame + 1, 16))
    if any(t not in frame_indices for t in frames):
        return None, 'missing_original_annotated_observation'
    if frame_indices[frame] != identity['original_candidate_index']:
        raise ValueError('Decision observation changed identity')
    sid = digest([SCHEMA, identity])
    checks = []
    for member in range(len(members)):
        five = [dict(original_candidate_index=frame_indices[t], frame=t,
                     control_step=t-first, task_name=p['task_name'],
                     parent_goal=p['previous_parent_goal'],
                     issued_bundle=canonical_json(members), member_index=member,
                     memory=p['memory'], served_controls=elapsed-(frame-t)) for t in frames]
        for offset in (0, 1):
            window = five[offset:offset+4]
            checks.append(dict(request_id=digest([sid, member, window[-1]['frame']]),
                               member_index=member, control_step=window[-1]['control_step'],
                               checks=window))
    return dict(schema=SCHEMA, sample_id=sid, source_group=group,
                original_identity=identity, annotation_trace=row['annotation_trace'],
                target_change_class=command_change_class(p['previous_intent'], p['active_skills_text']),
                original_projection=p, requests=checks,
                input_feedback_generated=False, new_human_admission=False,
                physical_outcome_asserted=False, optimizer_updates=0), None
