"""Compare two COMPLETE read-only skill probes, never select/deploy a policy."""
from copy import deepcopy
import math

from skill_training_protocol import read_only_recipe


def compare_probe_pair(configs, services, audits, config_shas):
    if not all(len(x) == 2 for x in (configs, services, audits, config_shas)):
        raise ValueError('Exactly two explicitly ordered arms required')
    normalized = []
    for cfg, arm in zip(configs, ('control', 'later-actions')):
        if (not read_only_recipe(cfg) or cfg['pair']['protocol'] != 'matched_later_action_sft_v1'
                or cfg['pair']['arm'] != arm):
            raise ValueError('Different experiment or reversed arm identity')
        value = deepcopy(cfg)
        del value['model']; del value['pair']['arm']
        normalized.append(value)
    if normalized[0] != normalized[1] or configs[0]['model']['sha256'] == configs[1]['model']['sha256']:
        raise ValueError('The only permitted differences are checkpoint and declared arm')
    cfg = configs[0]
    seeds = cfg['evaluation_seeds']
    if len(seeds) != len(set(seeds)) or not seeds or not cfg['cases']:
        raise ValueError('Unique, nonempty fixed seeds and cases required')
    expected = {(case, seed) for case in cfg['cases'] for seed in seeds}
    by_arm = []
    for config, service, audit, sha in zip(configs, services, audits, config_shas):
        jobs = service['jobs']
        keys = [(j['case'], j['seed']) for j in jobs]
        if (service['status'] != 'completed_read_only_skill_probes' or service['error'] is not None
                or service['evaluation_only'] is not True
                or service['optimizer_updates'] != 0 or service['actor_updates'] != 0
                or service['active_episodes'] != 0
                or service['whole_task_sr_evaluated'] is not False
                or service['config_sha256'] != sha or service['parent_sha256'] != config['model']['sha256']
                or service['policy_identity_sha256'] != config['model']['sha256']
                or service['actor_before_sha256'] != service['actor_after_sha256']
                or service['frozen_before_sha256'] != service['frozen_after_sha256']
                or set(service['finish_acknowledged']) != set(cfg['cases'])
                or len(keys) != len(expected) or set(keys) != expected
                or any(j['status'] != 'done' or j['phase'] != 'evaluation' or j['round'] != 0 for j in jobs)):
            raise ValueError('Incomplete, changed, mixed-policy or non-read-only service')
        if (audit['status'] != 'machine_control_reward_reset_audit_passed'
                or audit['baseline_protocol_checked'] is not True
                or audit['optimizer_authorized'] is not False or audit['whole_task_sr'] is not False
                or audit['config_sha256'] != sha or audit['policy_sha256'] != config['model']['sha256']):
            raise ValueError('Require independent full control/reward/reset/archive audit')
        rows = audit['episodes']; keys = [(r['case'], r['seed']) for r in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Missing, duplicate or additional audited episodes')
        for row in rows:
            case = cfg['cases'][row['case']]
            if (type(row['success']) is not bool
                    or row['success'] != (row['outcome'] == 'SUCCEEDED')
                    or row['outcome'] not in ('SUCCEEDED', 'FAILED', 'UNKNOWN')
                    or type(row['actual_controls']) is not int
                    or not 0 < row['actual_controls'] <= case['end_control']-case['start_control']
                    or 'observation_archive' not in row):
                raise ValueError('Wrong outcome, horizon or missing raw observation audit')
            for name, limit in (('reset_proprio_error', 1e-3), ('reset_target_error', .005)):
                error = row.get(name)
                if error is None and name == 'reset_target_error' and case['sim']['kind'] == 'placement':
                    continue  # Placement poses were checked against its distinct source snapshot.
                if not isinstance(error, (float, int)) or not math.isfinite(error) or not 0 <= error <= limit:
                    raise ValueError('Audited cold restore gate failed')
        by_arm.append(dict(zip(keys, rows)))
    pairs = []
    for case, seed in sorted(expected):
        left, right = [rows[(case, seed)] for rows in by_arm]
        pairs.append(dict(case=case, seed=seed,
            control={k: left[k] for k in ('success', 'outcome', 'actual_controls')},
            later_actions={k: right[k] for k in ('success', 'outcome', 'actual_controls')},
            outcome_delta=int(right['success'])-int(left['success']),
            controls_delta_if_both_success=(right['actual_controls']-left['actual_controls']
                                           if left['success'] and right['success'] else None)))
    by_case = {}
    for case in cfg['cases']:
        rows = [p for p in pairs if p['case'] == case]
        by_case[case] = dict(attempts_per_arm=len(rows),
            control_successes=sum(p['control']['success'] for p in rows),
            later_actions_successes=sum(p['later_actions']['success'] for p in rows),
            gained_seeds=[p['seed'] for p in rows if p['outcome_delta'] == 1],
            lost_seeds=[p['seed'] for p in rows if p['outcome_delta'] == -1])
    return dict(schema='matched_later_action_probe_comparison_v1',
        status='paired_machine_comparison_not_human_acceptance', pairs=pairs, by_case=by_case,
        independently_audited_controls=sum(r['actual_controls'] for rows in by_arm for r in rows.values()),
        same_cases_seeds_original_horizons=True, whole_task_sr=False, independent_task_generalization=False,
        noise_seed_replication_not_new_instances=True, policy_promotion_authorized=False,
        both_success_control_deltas_are_conditional_not_unconditional_speed=True)
