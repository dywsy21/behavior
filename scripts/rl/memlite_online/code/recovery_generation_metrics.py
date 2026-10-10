"""Offline planner diagnostics, not physical success or a deployment gate."""
import json
import math


def select_original_indices(fixed_by_task, previous_intent, task_index, mode):
    """Choose from the frozen heldout schedule by metadata, never prediction.

    Its first row per task is always episode-initial in this release. Such a
    sample cannot test a feedback shortcut on an already executing command.
    Do not silently substitute it when a noninitial control was requested.
    """
    if mode not in ('first_fixed_one_per_task_v1','noninitial_fixed_one_per_task_v1'):
        raise ValueError('Unknown planner generation schedule')
    result=[]
    for task, ids in enumerate(fixed_by_task):
        if not ids or any(type(i) is not int or task_index(i)!=task for i in ids):
            raise ValueError('Frozen schedule lost task binding')
        candidates=ids if mode=='first_fixed_one_per_task_v1' else [
            i for i in ids if previous_intent(i)!='None']
        if not candidates:
            raise ValueError('No noninitial frozen heldout control for task '+str(task))
        result.append(candidates[0])
    if not result or len(set(result))!=len(result):
        raise ValueError('Empty/duplicated original generation schedule')
    return result


def feedback_probe(causal, split, *, unknown_confidence=0.):
    """Counterfactual diagnostic, never real labels, training or runtime input.

    Remove the feedback field from recovery inputs; put an all-UNKNOWN field
    on normal noninitial inputs, with the correct previous-bundle member count.
    This exposes a possible `JSON present => RETRY` shortcut. The synthetic
    counters are explicitly not asserted to be the actual expert clock.
    """
    from g05.utils.memlite_skill_protocol import parse_active_skills_text
    if (isinstance(unknown_confidence,bool) or not isinstance(unknown_confidence,(int,float))
            or not math.isfinite(unknown_confidence) or not 0<=unknown_confidence<=1):
        raise ValueError('Invalid explicitly synthetic probe confidence')
    result=dict(causal)
    if causal['known_previous_outcome']!='UNKNOWN':
        raise ValueError('Probe only the pilot UNKNOWN-only feedback contract')
    if split=='recovery_dev':
        result['execution_feedback']='none';return result,'removed_recovery_feedback'
    if causal['previous_intent']=='None':
        return result,'initial_no_previous_command_unmodified'
    from recovery_observer_training import feedback_text
    members=parse_active_skills_text(causal['previous_intent'],allow_empty=False)
    prior=dict(served_controls=32,repeated_planning_count=0,attempt_number=1)
    result['execution_feedback']=feedback_text(prior,[dict(member=i,estimated_outcome='UNKNOWN',confidence=float(unknown_confidence))
        for i in range(len(members))])
    return result,'synthetic_unknown_feedback_on_normal_state'


def score_event(event, target):
    result = dict(valid=bool(event and event.get('validated')), decision=False, bundle=False,
                  parent=False, memory=False, exact=False, false_retry=False)
    if not result['valid']:
        return result
    result.update(decision=event['decision'] == target['next_decision'],
        bundle=json.loads(event['active_skills_semantic_json']) == json.loads(target['active_skills_semantic_json']),
        parent=event['parent_goal'] == target['target_parent_goal'],
        memory=event['memory_update'] == target['memory_update'],
        false_retry=target['next_decision'] != 'RETRY' and event['decision'] == 'RETRY')
    result['exact'] = all(result[k] for k in ('decision', 'bundle', 'parent', 'memory'))
    return result


def summarize(rows):
    result = {}
    for split in sorted({r['split'] for r in rows}):
        selected = [r for r in rows if r['split'] == split]
        result[split] = dict(rows=len(selected), **{k: sum(r['score'][k] for r in selected)/len(selected)
            for k in ('valid', 'decision', 'bundle', 'parent', 'memory', 'exact', 'false_retry')})
    return result
