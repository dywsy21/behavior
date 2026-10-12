"""Paired TRAIN diagnostics. Token CE/probabilities are not physical success."""
from collections import Counter
import math


SCORES = ('valid', 'decision', 'bundle', 'parent', 'memory', 'exact', 'false_retry')


def summarize_diagnostic_rows(rows):
    if not rows:
        return dict(rows=0)
    fields = {}; choices = {}
    for row in rows:
        if row['split'] not in ('train_handoff', 'train_continuation'):
            raise ValueError('TRAIN diagnosis cannot include another split')
        metrics = row['teacher_forced_fields']
        for key, count in metrics.items():
            if not key.startswith('ce_field_') or not key.endswith('_tokens'):
                continue
            name = key[len('ce_field_'):-len('_tokens')]
            mean = metrics['ce_field_'+name+'_mean']
            if (not math.isfinite(count) or count < 0 or int(count) != count
                    or not math.isfinite(mean) or mean < 0):
                raise ValueError('Invalid field CE or token count')
            accumulator = fields.setdefault(name, dict(tokens=0, ce_sum=0.))
            accumulator['tokens'] += int(count)
            accumulator['ce_sum'] += count * mean
        for name, choice in row['first_choice'].items():
            if not choice['available']:
                continue
            if (not choice['teacher_forced'] or not choice['preceding_fields_are_teacher_forced']
                    or not all(math.isfinite(choice[k]) for k in
                               ('target_log_probability', 'repeated_log_probability', 'target_minus_repeated_logit'))
                    or max(choice['target_log_probability'], choice['repeated_log_probability']) > 1e-6):
                raise ValueError('Invalid conditional first-choice diagnostic')
            accumulator = choices.setdefault(name, dict(rows=0, target_greedy=0,
                target_beats_repeat=0, margins=[], target_probability_sum=0., repeated_probability_sum=0.))
            accumulator['rows'] += 1
            accumulator['target_greedy'] += int(choice['target_is_greedy'])
            accumulator['target_beats_repeat'] += int(choice['target_minus_repeated_logit'] > 0)
            accumulator['margins'].append(choice['target_minus_repeated_logit'])
            accumulator['target_probability_sum'] += math.exp(choice['target_log_probability'])
            accumulator['repeated_probability_sum'] += math.exp(choice['repeated_log_probability'])
    for value in fields.values():
        value['token_weighted_ce'] = value.pop('ce_sum') / value['tokens'] if value['tokens'] else None
    for value in choices.values():
        margins = sorted(value.pop('margins')); n = value['rows']
        value['mean_target_minus_repeat_logit'] = sum(margins) / n
        value['median_target_minus_repeat_logit'] = (margins[(n-1)//2]+margins[n//2])/2
        value['mean_target_probability'] = value.pop('target_probability_sum') / n
        value['mean_repeated_probability'] = value.pop('repeated_probability_sum') / n
    return dict(rows=len(rows), counts={key:sum(bool(r['score'][key]) for r in rows) for key in SCORES},
        field_ce=fields, conditional_first_choice=choices,
        optimizer_exposures=dict(sorted(Counter(r['input_audit']['optimizer_exposures'] for r in rows).items())),
        strict_parse_errors=sum(r['error'] is not None for r in rows))


def paired_diagnostics(control, candidate):
    if not control or len(control) != len(candidate):
        raise ValueError('Incomplete paired diagnosis')
    seen = set()
    for left, right in zip(control, candidate):
        for key in ('global_schedule_index','scheduled_sample_index','split','sample_id','identity','input_audit','expected'):
            if left[key] != right[key]:
                raise ValueError('Changed paired input, order, exposure or target: '+key)
        if left['sample_id'] in seen:
            raise ValueError('Duplicate paired sample')
        seen.add(left['sample_id'])
        for key in SCORES:
            if type(left['score'][key]) is not bool or type(right['score'][key]) is not bool:
                raise ValueError('Scores must be exact Boolean outcomes from the original parser')
    cohorts = {'all': lambda r:True,
               'handoff': lambda r:r['split']=='train_handoff',
               'continuation': lambda r:r['split']=='train_continuation',
               'exposed_handoff': lambda r:r['split']=='train_handoff' and r['input_audit']['optimizer_exposures']>0,
               'unexposed_handoff': lambda r:r['split']=='train_handoff' and r['input_audit']['optimizer_exposures']==0}
    result = {}
    for name, predicate in cohorts.items():
        pairs = [(left,right) for left,right in zip(control,candidate) if predicate(left)]
        result[name] = dict(control=summarize_diagnostic_rows([p[0] for p in pairs]),
            candidate=summarize_diagnostic_rows([p[1] for p in pairs]),
            paired_counts={key:dict(gains=sum(not a['score'][key] and b['score'][key] for a,b in pairs),
                                   losses=sum(a['score'][key] and not b['score'][key] for a,b in pairs)) for key in SCORES})
    return dict(cohorts=result, data_split='train', conditional_probabilities_are_not_free_decoding=True,
                no_generalization_or_physical_success_claim=True, optimizer_updates=0)
