"""Batch bookkeeping; no simulator, model, rewards, or privileged state."""
from copy import deepcopy
import hashlib
import json


def validate_indices(observations, indices, num_slots):
    if not observations or len(observations) != len(indices):
        raise ValueError('Nonempty observations and indices must have equal length')
    if any(type(i) is not int or not 0 <= i < num_slots for i in indices):
        raise ValueError('Invalid environment slot')
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate environment slot')


def needs_plan(slot):
    return slot['projection'] is None or (
        slot['chunks'] % 8 == 0 and slot['planned_chunk'] != slot['chunks'])


def stage_plans(slots, indices, events):
    """Prepare independent transactions; a bad row cannot commit another row.

    Keep the *previous* ledger memory in the low projection, exactly as the
    sealed serial Stage1Engine.plan does. This is not a memory-format change.
    """
    if len(events) != len(indices):
        raise ValueError('Planner event count differs from requested slots')
    staged = []
    for index, event in zip(indices, events, strict=True):
        ledger = deepcopy(slots[index]['ledger'])
        proposed = ledger.stage(event)
        goal = proposed['installed_subgoal']
        projection = dict(schema_version=6, memlite_branch='low', task_name=ledger.task_name,
            parent_goal=goal['parent_goal'], target_parent_goal='', previous_parent_goal='none',
            memory=ledger.memory, previous_intent='None', known_previous_outcome='UNKNOWN',
            execution_feedback='none', active_skills_semantic_json=goal['active_skills_semantic_json'],
            active_skills_text=proposed['event']['active_skills_text'], next_decision='EXECUTE',
            memory_update='', task_complete=False, outcome_target='UNKNOWN',
            outcome_supervision_mask=False, parent_goal_supervision_mask=False,
            low_action_supervision_mask=False)
        context_id = hashlib.sha256(json.dumps(projection, sort_keys=True).encode()).hexdigest()
        ledger.commit(proposed['proposal_id'])
        staged.append(dict(index=index, ledger=ledger, projection=projection,
                           context_id=context_id, event=proposed['event']))
    return staged


def commit_plans(slots, staged):
    for row in staged:
        slot = slots[row['index']]
        slot.update(ledger=row['ledger'], projection=row['projection'],
                    context_id=row['context_id'], planned_chunk=slot['chunks'])


def partitions(indices, size):
    if size not in (2, 4) or not indices or len(set(indices)) != len(indices):
        raise ValueError('Expected unique cases and 2/4 environments')
    return [indices[i:i+size] for i in range(0, len(indices), size)]
