"""Causal member feature requests; metadata/labels never become VLM inputs."""
from collections import defaultdict
import json

from recovery_corpus import canonical, digest


def feature_requests(anchors, history, selections):
    """selections: (sample_id, current|predecision, member index) tuples.

    Four real observations at least 16 controls apart, confined to the same
    actually issued intent attempt. Missing history is masked, never repeated.
    H1's current image is combined ONLY with the pre-decision command memory.
    """
    by_id = {r['sample_id']:r for r in anchors}
    histories = {r['sample_id']:r for r in history}
    if len(by_id) != len(anchors) or len(histories) != len(history):
        raise ValueError('Duplicate observation/history')
    episodes = defaultdict(list)
    for row in anchors: episodes[canonical(row['source_episode'])].append(row)
    for rows in episodes.values(): rows.sort(key=lambda r:r['control_step'])
    output = {}
    for sid,role,member in selections:
        if role not in ('observable','predecision') or sid not in by_id or sid not in histories:
            raise ValueError('Unknown causal feature selection')
        row = by_id[sid]; h = histories[sid]
        if (h['source_episode'] != row['source_episode'] or h['source_group'] != row['source_group']
                or h['control_step'] != row['control_step']):
            raise ValueError('Cross-episode history join')
        context = h[role]
        if context is None or context['history_is_partial']:
            raise ValueError('No complete pre-decision history; do not invent it')
        members = json.loads(context['issued_skills_semantic_json'])
        if type(member) is not int or not 0 <= member < len(members):
            raise ValueError('Wrong issued skill member')
        def bound(observation, ctx):
            t = observation['control_step']
            if (ctx['observation_control_step'] != t or ctx['control_step'] > t
                    or ctx['served_controls'] != t-ctx['intent_started_control_step']):
                raise ValueError('Future/stale context clock')
            # Only these observable values are consumed by the prefix builder.
            return dict(sample_id=observation['sample_id'], control_step=t,
                task_name=observation['task'].replace('_',' '), parent_goal=ctx['parent_goal'],
                issued_bundle=ctx['issued_skills_semantic_json'], member_index=member,
                memory=ctx['memory'], served_controls=ctx['served_controls'])
        checks = [bound(row, context)]
        for earlier in reversed(episodes[canonical(row['source_episode'])]):
            t = earlier['control_step']
            # Low-level execution yields a fresh image every 16 controls.
            # Waiting 128 (the old planner cadence) makes short recovery
            # decisions have only one check, forcing feedback UNKNOWN forever.
            if t > checks[-1]['control_step']-16 or t < context['intent_started_control_step']: continue
            previous = histories[earlier['sample_id']]['observable']
            if (previous['issued_skills_semantic_json'] != context['issued_skills_semantic_json']
                    or previous['parent_goal'] != context['parent_goal']
                    or previous['intent_started_control_step'] != context['intent_started_control_step']):
                continue
            checks.append(bound(earlier,previous))
            if len(checks) == 4: break
        checks.reverse()
        key = digest([sid,role,member])
        output[key] = dict(request_id=key,sample_id=sid,role=role,member_index=member,
            source_group=row['source_group'],split=row['split'],source_episode=row['source_episode'],
            checks=checks)
    return list(output.values())


def balanced_event_schedule(rows, *, batch_size, passes, seed=17):
    """Exactly one sampled member/anchor per physical event per pass."""
    import random
    if not 1 <= passes <= 5 or batch_size < 1: raise ValueError('Invalid finite event schedule')
    events = defaultdict(list)
    for i,row in enumerate(rows):
        if row['candidate']['split'] != 'train': raise ValueError('Only TRAIN event schedule')
        events[(row['candidate']['source_group'],row['approval']['event_id'])].append(i)
    rng = random.Random(seed)
    for epoch in range(passes):
        # Round robin across shuffled task queues prevents task-major runs;
        # no fake guarantee for a corpus containing only a single task.
        tasks = defaultdict(list)
        for _,indices in sorted(events.items()):
            i = rng.choice(indices); tasks[rows[i]['candidate']['task']].append(i)
        for values in tasks.values(): rng.shuffle(values)
        task_order=list(tasks); rng.shuffle(task_order); order=[]
        while any(tasks.values()):
            for task in task_order:
                if tasks[task]: order.append(tasks[task].pop())
        for start in range(0,len(order),batch_size):
            yield dict(event_pass=epoch,rows=[('new',i) for i in order[start:start+batch_size]])


def group_folds(groups, k=3):
    """Balance immutable source-instance groups, not correlated frames."""
    groups = sorted(set(groups),key=lambda x:digest(['recovery-fold17',x]))
    if len(groups) < 2*k or k < 2: raise ValueError('Insufficient independent groups for out-of-fold heads')
    return {g:i%k for i,g in enumerate(groups)}


def validate_prediction_provenance(row, *, group, high_sha256):
    required={'high_sha256','observer_sha256','trained_groups','calibration_groups','target_groups',
              'feature_cache_sha256','fold','calibrated','temperature','calibration_receipt_sha256'}
    if set(row)!=required or row['high_sha256'] != high_sha256 or type(row['calibrated']) is not bool:
        raise ValueError('Invalid prediction provenance or changed high backbone')
    if (group not in row['target_groups'] or group in row['trained_groups']
            or group in row['calibration_groups'] or set(row['trained_groups']) & set(row['calibration_groups'])):
        raise ValueError('Prediction is not out of source-group training AND calibration')
    for name in ('high_sha256','observer_sha256','feature_cache_sha256','calibration_receipt_sha256'):
        value=row[name]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Unpinned prediction artifact')
    if not isinstance(row['temperature'],(int,float)) or not .05 <= row['temperature'] <= 20:
        raise ValueError('Unbounded calibration temperature')
    return True
