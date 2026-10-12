"""Complete three-arm physical pairing; no model selection or promotion."""
from copy import deepcopy

from recovery_probe_pair import validate_probe_arm
from recovery_probe_restart import validate_probe_restart

ARMS=('parent','control','candidate')


def compare_native_probes(configs, services, audits, episodes, config_shas, *, repo):
    if not all(len(x)==3 for x in (configs,services,audits,episodes,config_shas)):
        raise ValueError('All three predeclared arms required')
    normalized=[]; checked={}
    for arm,cfg,service,audit,logs,sha in zip(ARMS,configs,services,audits,episodes,config_shas):
        validate_probe_restart(cfg,repo)
        if cfg['comparison_arm']!=arm:
            raise ValueError('Wrong native comparison arm order')
        science=deepcopy(cfg);science.pop('model');science.pop('comparison_arm');normalized.append(science)
        rows=validate_probe_arm(cfg,service,audit,sha)
        job_ids={(j['case'],j['seed']):j['id'] for j in service['jobs']}
        log_keys=[(r['job']['case'],r['job']['seed']) for r in logs]
        if len(log_keys)!=len(rows) or set(log_keys)!=set(rows):
            raise ValueError('Incomplete or duplicated per-episode service terminal ledger')
        for log,key in zip(logs,log_keys):
            physical=rows[key];job=log['job'];case=cfg['cases'][key[0]]
            if (job['id']!=job_ids[key] or physical['job_id']!=job_ids[key]
                    or job['phase']!='evaluation' or job['round']!=0
                    or log['whole_task_sr'] is not False or log['policy_version']!=0
                    or log['policy_sha256']!=cfg['model']['sha256']
                    or type(log['success']) is not bool or log['success']!=physical['success']
                    or log['outcome']!=physical['outcome'] or log['actual_controls']!=physical['actual_controls']
                    or log['reference_end_control']!=case['end_control'] or log['policy_end_control']!=case['end_control']
                    or physical['observation_archive'].get('status')!='exact_boundaries_and_lossless_roundtrip_passed'):
                raise ValueError('Service, real controls, cold restore or observation archive disagree')
        checked[arm]=rows
    if any(x!=normalized[0] for x in normalized[1:]) or len({c['model']['sha256'] for c in configs})!=3:
        raise ValueError('Scientific conditions differ beyond the three checkpoints')
    pairs=[]
    for case,seed in sorted(checked['parent']):
        values={arm:{k:checked[arm][case,seed][k] for k in ('success','outcome','actual_controls','job_id')}
                for arm in ARMS}
        pairs.append(dict(case=case,seed=seed,arms=values))
    counts={};contrasts={}
    for case in configs[0]['cases']:
        subset=[r for r in pairs if r['case']==case]
        counts[case]={arm:dict(success=sum(r['arms'][arm]['success'] for r in subset),total=len(subset)) for arm in ARMS}
        contrasts[case]={}
        for left,right in (('parent','control'),('parent','candidate'),('control','candidate')):
            gains=[r['seed'] for r in subset if not r['arms'][left]['success'] and r['arms'][right]['success']]
            losses=[r['seed'] for r in subset if r['arms'][left]['success'] and not r['arms'][right]['success']]
            contrasts[case][left+'_vs_'+right]=dict(gained_seeds=gains,lost_seeds=losses,
                both_success=sum(r['arms'][left]['success'] and r['arms'][right]['success'] for r in subset),
                both_unsuccessful=sum(not r['arms'][left]['success'] and not r['arms'][right]['success'] for r in subset))
    return dict(status='complete_machine_audited_three_arm_pairing_not_policy_promotion',
        by_case=counts,paired_contrasts=contrasts,pairs=pairs,
        total_scored_controls=sum(r['actual_controls'] for rows in checked.values() for r in rows.values()),
        optimizer_updates=0,all_source_cases_train=True,whole_task_sr=False,independent_source_generalization=False,
        policy_promotion_authorized=False,limitation='Eight noise seeds on each of three repeatedly diagnosed TRAIN starts; only one added corrective source event. No claim of independent task coverage or statistically established generalization.')
