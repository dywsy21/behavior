"""CPU-only accounting for the registered E2 final-evaluation continuation."""
from math import ceil, isfinite
from g05.rl.protocol import EVAL_INSTANCES, EVAL_SEEDS, EVAL_LIMIT


def remaining_evaluation_budget(manifest, status, supervisor, training_controls, baseline_controls):
    if (manifest['max_controls'] != 100000 or manifest['max_active_wall_seconds'] != 43200
            or status['pending_controls'] != 0 or supervisor['status'] != 'failed'
            or status['phase'] != 'failed'):
        raise ValueError('Only the stopped, fully accounted original E2 may continue')
    if min(training_controls, baseline_controls) < 0:
        raise ValueError('Negative prior controls')
    prior_controls = training_controls + baseline_controls
    if prior_controls != status['controls']:
        raise ValueError('Physical logs and prior control ledger disagree')
    elapsed = max(float(status['seconds']), float(supervisor['seconds']))
    if not isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid prior activity time')
    prior_seconds = ceil(elapsed)
    controls = len(EVAL_INSTANCES) * len(EVAL_SEEDS) * EVAL_LIMIT
    seconds = min(10800, 43200 - prior_seconds - 40)  # retain owned-process cleanup reserve
    if prior_controls + controls > 100000 or seconds < 3600:
        raise ValueError('Insufficient original budget for the registered complete matrix')
    return dict(prior_controls=prior_controls, prior_active_seconds=prior_seconds,
                max_controls=controls, max_active_wall_seconds=seconds,
                original_control_limit=100000, original_active_wall_limit=43200,
                cleanup_reserve_seconds=40)


def validate_baseline(rows):
    expected = {('parent_fp32', i, s) for i in EVAL_INSTANCES for s in EVAL_SEEDS}
    expected |= {('parent_bf16', i, 17) for i in EVAL_INSTANCES}
    keys = [(r['variant'], r['instance'], r['policy_seed']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Original baseline must be complete, unique, and contain no final evaluation')
    for r in rows:
        if (r['split'] != 'public_test' or r['environment_seed'] != 0
                or r['expert_prefix_controls'] != 0 or r['control_limit'] != EVAL_LIMIT
                or not 0 < r['controls'] <= EVAL_LIMIT or r.get('invalid')
                or r['actor_updates'] != 0
                or r['ae_precision'] != ('float32' if r['variant'] == 'parent_fp32' else 'bfloat16')
                or (r['success'] and not r['terminated'])
                or (not r['terminated'] and not r['truncated'] and r['controls'] != EVAL_LIMIT)):
            raise ValueError('Invalid original complete-reset baseline')
    return sum(r['controls'] for r in rows)


def count_physical_steps(rows, *, allowed_phases):
    count = 0
    previous_episode = -1
    within = 0
    for r in rows:
        if r['episode'] != previous_episode:
            if r['episode'] != previous_episode + 1:
                raise ValueError('Missing or reordered physical episode')
            previous_episode = r['episode']
            within = 0
        count += 1
        within += 1
        if (r['control'] != count or r['episode_control'] != within
                or r['phase'] not in allowed_phases):
            raise ValueError('Missing, duplicate, or out-of-scope physical control')
    return count
