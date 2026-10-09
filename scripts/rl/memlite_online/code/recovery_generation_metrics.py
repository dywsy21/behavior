"""Offline planner diagnostics, not physical success or a deployment gate."""
import json


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
