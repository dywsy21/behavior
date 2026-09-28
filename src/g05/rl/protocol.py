"""CPU-only split, curriculum and paired-result contracts for the RL pilot."""
from math import isfinite

EVAL_INSTANCES = (301, 302)
EVAL_SEEDS = (17, 23, 41)
EVAL_LIMIT = 3224


def validate_worker(spec, *, evaluation):
    if spec['task_id']!=0 or spec['task']!='turning_on_radio' or spec['seed']!=0:
        raise ValueError('Outside registered task/environment-seed scope')
    if spec['worker'] not in (0, 1) or spec['gpu'] != spec['worker']+2:
        raise ValueError('Worker/GPU identity mismatch')
    if bool(spec.get('evaluation_only', False)) != evaluation:
        raise ValueError('Explicit evaluation role required')
    if evaluation:
        if (spec['split'] != 'public_test' or spec['instance'] not in EVAL_INSTANCES
                or spec.get('prefix_controls', 0) != 0 or 'actions' in spec):
            raise ValueError('Evaluation cannot carry TRAIN actions or a prefix')
    elif (spec['split'] != 'train' or spec['episode'] % 200 >= 190
          or not 0 <= spec['prefix_controls'] < spec['first_recorded_terminal']
          or not spec.get('actions')):
        raise ValueError('Only nonterminal original TRAIN curriculum allowed')


def validate_phase(spec, phase):
    if spec.get('evaluation_only'):
        if phase not in ('eval_parent_bf16', 'eval_parent_fp32', 'eval_rl_fp32'):
            raise ValueError('No training/expert controls in evaluation pool')
    elif phase not in ('expert_prefix', 'policy'):
        raise ValueError('Training pool cannot supply evaluation results')


def next_prefix(first_terminal, current, recent):
    """Move earlier after three successes at this exact TRAIN difficulty."""
    if not 0 <= current < first_terminal:
        raise ValueError('Terminal or negative curriculum prefix')
    if len(recent) >= 3 and all(recent[-3:]):
        return max(0, first_terminal-768, current-96)
    return current


def paired_summary(rows):
    expected={(i,s) for i in EVAL_INSTANCES for s in EVAL_SEEDS}
    matrices={}
    for name in ('parent_fp32', 'rl_fp32'):
        selected=[r for r in rows if r['variant']==name]
        keys=[(r['instance'],r['policy_seed']) for r in selected]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:
            raise ValueError('Incomplete or duplicated paired evaluation matrix')
        for row in selected:
            if (row['split']!='public_test' or row['expert_prefix_controls']!=0
                    or row['environment_seed']!=0 or row['control_limit']!=EVAL_LIMIT
                    or row.get('invalid') or row['controls']>EVAL_LIMIT
                    or (row['success'] and not row['terminated'])):
                raise ValueError('Invalid or unmatched complete-task evaluation')
            if (not row['success'] and not row['terminated'] and not row['truncated']
                    and row['controls']!=EVAL_LIMIT):
                raise ValueError('Incomplete nonterminal rollout is not a failure-rate sample')
        matrices[name]={(r['instance'],r['policy_seed']):r for r in selected}
    before=matrices['parent_fp32']; after=matrices['rl_fp32']
    pairs=[dict(instance=i,policy_seed=s,before=bool(before[i,s]['success']),
                after=bool(after[i,s]['success'])) for i,s in sorted(expected)]
    n0=sum(p['before'] for p in pairs); n1=sum(p['after'] for p in pairs)
    return dict(n=6,before_successes=n0,after_successes=n1,before_rate=n0/6,after_rate=n1/6,
                difference=(n1-n0)/6,improved_pairs=sum(not p['before'] and p['after'] for p in pairs),
                regressed_pairs=sum(p['before'] and not p['after'] for p in pairs),pairs=pairs,
                scope='radio development matrix; not blind or all-task success rate',
                statistically_significant_claim=False)


def check_reset(reference, actual, tolerance=1e-5):
    if set(reference)!=set(actual): raise ValueError('Reset object identity changed')
    maximum=0.
    for key, old in reference.items():
        new=actual[key]
        if len(old)!=len(new): raise ValueError('Reset state shape changed')
        for a,b in zip(old,new):
            if not isfinite(a) or not isfinite(b): raise ValueError('Nonfinite reset state')
            maximum=max(maximum,abs(a-b))
    if maximum>tolerance: raise ValueError(f'Unmatched official reset state: {maximum}')
    return maximum
