"""Outcome-only final observations; never manufacture a next action row."""
from copy import deepcopy
import json

from recovery_corpus import CAMERAS, canonical, digest, file_sha, group_key
from recovery_local_teacher import validate_terminal_observation


def load_terminal(branch, manifest, rows, plans):
    expected = manifest.get('terminal_observation_sha256')
    if expected is None:
        return None  # Older immutable collections did not record this evidence.
    path = branch / 'terminal-observation.json'
    if file_sha(path) != expected:
        raise ValueError('Changed final observation')
    value = json.loads(path.read_text())
    validate_terminal_observation(value, rows, manifest['arm'])
    if (not rows or value['schema'] != 'recovery_terminal_observation_v1'
            or value['last_applied_control_step'] != rows[-1]['control_step']
            or value['context'] != plans[-1] or plans[-1]['control_step'] > len(rows)
            or set(value['images']) != set(CAMERAS)):
        raise ValueError('Final observation changed actual control/context identity')
    for camera, ref in value['images'].items():
        if (ref['path'] != 'terminal_rgb/' + camera + '.jpg'
                or file_sha(branch / ref['path']) != ref['sha256']):
            raise ValueError('Changed/unsafe original final RGB')
    return value


def normalized_terminal(value, binding):
    value = deepcopy(value)
    audit = value['physical_audit']
    audit['grasp_states'] = {audit['entity']: audit['grasp']}
    audit['entity_bindings'] = binding
    return value


def terminal_anchor(episode, value, image_ref, split, branch):
    t = value['control_step']; context = value['context']
    semantic = canonical(json.loads(context['active_skills_semantic_json']))
    identity = [episode['run'], episode['episode_id']]
    return dict(schema='recovery_anchor_candidate_v1', sample_id=digest([identity, t]),
        source_episode=identity, task=episode['task'], instance_id=episode['instance_id'],
        split=split, source_group=group_key(episode['task'], episode['instance_id']),
        control_step=t, policy_update=0,
        actor_input=dict(rgb=image_ref, proprio_before=value['proprio'],
                         issued_skills_semantic_json=semantic, parent_goal=context['parent_goal'],
                         observed_same_intent_controls=t-context['control_step'],
                         history_is_partial=False, observation_only=True),
        label_audit=dict(outcome=dict(value=value['outcome_candidate'],
                         reason='causal_final_observation_pending_review', members=[], evidence_end_control_step=t),
                         physically_held_at_observation={}, full_executed_32_step_target_available=False,
                         target_availability_is_not_action_quality=True,
                         offline_teacher=dict(kind=branch['kind'], action_label_kind='no_next_action',
                             physical_recovery_candidate=branch['physical_recovery_candidate'], not_on_policy=True,
                             source_event_id=digest([group_key(episode['task'],episode['instance_id']),
                                                     'first_demonstrated_grasp']),
                             branch_failure=branch['failure'])),
        eligibility=dict(outcome=False, planner=False, action=False),
        label_status='candidate_pending_semantic_review', review_status='pending')
