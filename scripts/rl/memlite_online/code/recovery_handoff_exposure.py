"""Audit executed TRAIN exposure, not heldout quality or physical success."""
from collections import Counter, defaultdict
import json
import math


def _unique(rows, key):
    result = {}
    for row in rows:
        value = key(row)
        if value in result:
            raise ValueError('Duplicate TRAIN identity')
        result[value] = row
    return result


def training_exposure(schedule, control, candidates, requests, predictions,
                      processor_audit, updates, fit):
    """Use the completed optimizer sequence, not the nominal candidate pool.

    Token fractions are weighted denominator mass; they are NOT gradient
    contributions. Existing audited token spans are reused without new model
    calls. Requests and processor rows are joined by identity, never zip-order
    across independently sorted files.
    """
    batches = schedule['schedule']
    if not batches or batches != control['schedule']:
        raise ValueError('Changed paired control sequence')
    if (fit['status'] != 'completed_finite_schedule' or fit['step'] != len(batches)
            or fit['planned_updates'] != len(batches) or len(updates) != len(batches)
            or fit['frozen_before_sha256'] != fit['frozen_after_sha256']):
        raise ValueError('Audit requires the complete, frozen-component-preserving fit')
    source = _unique(candidates, lambda r: r['identity']['original_candidate_index'])
    requested = _unique(requests, lambda r: r['sample_id'])
    predicted = _unique(predictions, lambda r: r['sample_id'])
    if set(requested) != set(predicted):
        raise ValueError('Missing or foreign feedback')
    for row in candidates:
        projection = row['projection']
        if (row['identity']['split'] != 'train' or projection['outcome_supervision_mask']
                or projection['low_action_supervision_mask']):
            raise ValueError('Only originally admitted TRAIN planner rows')
    by_index = {}
    for sid, request in requested.items():
        identity = request['original_identity']; index = identity['original_candidate_index']
        if (index not in source or index in by_index
                or identity != source[index]['identity']
                or request['original_projection'] != source[index]['projection']
                or predicted[sid]['original_identity'] != identity
                or request['source_group'] != predicted[sid]['source_group']):
            raise ValueError('Feedback source/target identity mismatch')
        by_index[index] = sid
    seen = Counter(); recovery = Counter(); per_task = Counter()
    for step, (batch, update) in enumerate(zip(batches, updates), 1):
        experts = [i for kind, i in batch['rows'] if kind == 'expert']
        new = [i for kind, i in batch['rows'] if kind == 'new']
        if (len(experts) != 6 or len(new) != 2 or len(batch['rows']) != 8
                or update['update'] != step or update['real_samples'] != 8
                or update['expert_examples'] != 6 or update['recovery_examples'] != 2
                or update['world_size'] != 8 or not math.isfinite(update['loss'])
                or not math.isfinite(update['grad_norm'])
                or not math.isfinite(update['denominator']) or update['denominator'] <= 0):
            raise ValueError('Partial, reordered, nonfinite or changed actual batch')
        if any(index not in source for index in experts):
            raise ValueError('Unadmitted expert row in actual optimizer sequence')
        tasks = [source[index]['identity']['task'] for index in experts]
        if len(set(tasks)) != 6:
            raise ValueError('Expert tasks are not mixed within the batch')
        seen.update(experts); recovery.update(new); per_task.update(tasks)
    if (processor_audit['status'] != 'passed'
            or len(processor_audit['processed']) != len(requests)
            or len(processor_audit['token_checks']) != len(requests)
            or not processor_audit['training_serving_prefix_equal']):
        raise ValueError('Incomplete actual processor/token acceptance')
    tokens = {}
    for processed, token in zip(processor_audit['processed'], processor_audit['token_checks']):
        sid = processed['sample_id']
        if (sid not in requested or sid in tokens
                or processed['identity']['candidate'] != requested[sid]['original_identity']['original_candidate_index']
                or token['task_name'] != requested[sid]['original_projection']['task_name']
                or token['memory_weight'] != .25):
            raise ValueError('Processor/token row alignment changed')
        fields = token['supervised_field_tokens']
        if fields['outcome_target'] or fields['task_complete'] or token['supervised_tokens'] < sum(fields.values()):
            raise ValueError('Invalid evidence mask or field denominator')
        tokens[sid] = token
    rows = []; field_mass = defaultdict(Counter); groups = defaultdict(Counter)
    for index, row in source.items():
        p = row['projection']; previous = p['previous_intent']
        kind = 'initial' if previous == 'None' else 'continuation' if previous == p['active_skills_text'] else 'handoff'
        sid = by_index.get(index); feedback = predicted.get(sid)
        estimate = json.loads(feedback['execution_feedback'])['estimated_bundle_outcome'] if feedback else 'unchanged_UNKNOWN'
        count = seen[index]
        for scope in ('all', 'learned' if sid else 'unchanged'):
            group = groups[(scope, kind, estimate)]
            group.update(pool_rows=1, seen_rows=int(count > 0), exposures=count)
        if sid:
            token = tokens[sid]; fields = token['supervised_field_tokens']
            for name, value in fields.items():
                field_mass[kind][name] += count * value * (.25 if name == 'memory_update' else 1.)
            field_mass[kind]['structural'] += count * (token['supervised_tokens'] - sum(fields.values()))
        rows.append(dict(index=index, task=row['identity']['task'], kind=kind,
                         feedback_sample_id=sid, estimated_outcome=estimate, exposures=count))
    summary = dict(pool_rows=len(source), unique_expert_rows=len(seen), expert_exposures=sum(seen.values()),
                   unexposed_expert_rows=len(set(source) - set(seen)), recovery_exposures=sum(recovery.values()),
                   unique_recovery_rows=len(recovery), optimizer_updates_observed=len(updates),
                   all_batches_six_distinct_expert_tasks=True,
                   exposure_frequency=dict(sorted(Counter(seen.values()).items())),
                   task_exposures=dict(sorted(per_task.items())),
                   strata=[dict(scope=s, kind=k, estimate=e, **v) for (s,k,e),v in sorted(groups.items())],
                   learned_feedback_weighted_token_mass={k:dict(v) for k,v in field_mass.items()},
                   token_mass_is_not_gradient_contribution=True,
                   new_optimizer_updates=0, model_forwards=0, physical_success_measured=False)
    return summary, rows


def first_difference(target, repeated):
    """First distinguishable token under their common teacher-forced prefix.

    A strict prefix has no two-token alternative here and must be marked
    unavailable, not assigned a made-up EOS probability.
    """
    for index, (wanted, old) in enumerate(zip(target, repeated)):
        if wanted != old:
            return index, wanted, old
    return None
