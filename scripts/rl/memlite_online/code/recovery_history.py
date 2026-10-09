"""Reconstruct deployment-observable history from complete issued-plan logs.

Never infer command age from a clipped RGB window. Verify the historical
planner-only memory recurrence AND the exact low-context hash before joining.
No rewards/physics/future plans are returned as actor inputs.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json

from recovery_corpus import canonical, digest, group_key


def replay_episode(events, episode):
    from g05.utils.memlite_skill_protocol import (
        append_b_memory_idempotent, canonical_json, parse_active_skills_semantic_json,
        semantic_active_skills_text, validate_semantic_parent_goal,
    )
    task = episode['task'].replace('_',' ')
    memory = canonical_json(dict(task_name=task,issued_command_history=[],verified_world_facts=[]))
    previous = 'None'
    installed, started, refreshes, attempts = None, 0, 0, {}
    result = []
    for ordinal, row in enumerate(sorted(events,key=lambda x:x['control_step'])):
        meta = row['episode']
        if (any(meta[k] != episode[k] for k in ('run','episode_id','task','instance_id','source_commit',
                                               'high_checkpoint_sha256','low_checkpoint_sha256'))
                or row['task'] != task or row['rank'] != episode['gpu']):
            raise ValueError('Planner log belongs to another task/episode/source/model')
        t = row['control_step']
        if type(t) is not int or t != ordinal*128 or row['chunk'] != ordinal*8:
            raise ValueError('Missing, duplicate or shifted historical planning checkpoint')
        e = row['event']
        if (e.get('validated') is not True or e.get('previous_outcome') != 'UNKNOWN'
                or e.get('task_complete') is not False or e.get('task_complete_claimed') is not False
                or e['decision'] not in ('EXECUTE','RETRY','REPLAN')):
            raise ValueError('Not an admitted historical planner-only command')
        parent = validate_semantic_parent_goal(e['parent_goal'],field='parent_goal')
        members = parse_active_skills_semantic_json(e['active_skills_semantic_json'],allow_empty=False)
        if semantic_active_skills_text(members) != e['active_skills_text']:
            raise ValueError('Changed historical intent text')
        expected = append_b_memory_idempotent(memory,previous,task_name=task)
        if e['memory_update'] != expected:
            raise ValueError('Causal command-memory recurrence is broken')
        # This is exactly the old low projection, including memory BEFORE the
        # current event commits. JSON whitespace is part of that historical hash.
        projection = dict(schema_version=6,memlite_branch='low',task_name=task,
            parent_goal=parent,target_parent_goal='',previous_parent_goal='none',memory=memory,
            previous_intent='None',known_previous_outcome='UNKNOWN',execution_feedback='none',
            active_skills_semantic_json=e['active_skills_semantic_json'],active_skills_text=e['active_skills_text'],
            next_decision='EXECUTE',memory_update='',task_complete=False,outcome_target='UNKNOWN',
            outcome_supervision_mask=False,parent_goal_supervision_mask=False,low_action_supervision_mask=False)
        context_id = hashlib.sha256(json.dumps(projection,sort_keys=True).encode()).hexdigest()
        if context_id != row['context_id']:
            raise ValueError('Replay differs from the actually served low context')
        key = canonical([parent,members])
        if installed == key and e['decision'] != 'RETRY':
            refreshes += 1
        else:
            attempts[key] = attempts.get(key,0)+1
            started, refreshes = t, 0
        installed = key
        memory, previous = expected, e['active_skills_text']
        result.append(dict(control_step=t,context_id=context_id,parent_goal=parent,
            issued_skills_semantic_json=canonical(members),memory=memory,
            previous_intent=previous,intent_started_control_step=started,
            repeated_planning_count=refreshes,attempt_number=attempts[key],
            history_is_partial=False,logged_event_sha256=digest(row)))
    if not result:
        raise ValueError('Empty full-episode planner log')
    return result


def join_causal_history(anchors, inventory, log_rows, reader):
    episodes = {canonical([r['episode']['run'],r['episode']['episode_id']]):r['episode'] for r in inventory}
    by_episode = defaultdict(list)
    for row in log_rows:
        ep = row['episode']
        key = canonical([ep['run'],ep['episode_id']])
        if key in episodes:
            by_episode[key].append(row)
    replayed = {key:replay_episode(by_episode[key],ep) for key,ep in episodes.items()}
    output = []
    for candidate in anchors:
        key = canonical(candidate['source_episode'])
        t = candidate['control_step']
        prior = [row for row in replayed[key] if row['control_step'] <= t]
        if not prior:
            raise ValueError('Observation predates the first issued plan')
        current = prior[-1]
        raw, _ = reader.episode(candidate['source_episode'])
        if (raw[t]['context']['context_id'] != current['context_id']
                or current['issued_skills_semantic_json'] != candidate['actor_input']['issued_skills_semantic_json']
                or current['parent_goal'] != candidate['actor_input']['parent_goal']
                or candidate['source_group'] != group_key(episodes[key]['task'],episodes[key]['instance_id'])):
            raise ValueError('Candidate does not match its causally active recorded command')
        observable = deepcopy(current)
        observable.update(observation_control_step=t,
                          served_controls=t-current['intent_started_control_step'])
        # H1 at an actual decision boundary must NOT read the command that is
        # its own CE target. The predecision view excludes the event at s[t].
        before = [p for p in prior if p['control_step'] < t]
        predecision = None
        if current['control_step'] == t and before:
            predecision = deepcopy(before[-1])
            predecision.update(observation_control_step=t,
                served_controls=t-predecision['intent_started_control_step'])
        output.append(dict(schema='recovery_causal_history_v1',sample_id=candidate['sample_id'],
            source_episode=candidate['source_episode'],source_group=candidate['source_group'],
            control_step=t,observable=observable,predecision=predecision,
            evidence=dict(prior_logged_plans=len(prior),latest_plan_sha256=current['logged_event_sha256']),
            new_label=False))
    return output
