"""Prospective adapter-calibration provenance; no model/optimizer in this module."""
from recovery_calibration_selection import preselect_pool_anchors, selected_rows


def pool_sources(pool, cohorts, *, exposed_groups=None):
    """Account for every first attempt before looking at a model prediction.

    cohorts contains (immutable cohort document, actual file SHA, source config
    SHA) tuples. Completed but unsigned groups must be reviewed, not silently
    marked unavailable to manufacture a passing subcohort.
    """
    v3=pool.get('schema')=='prospective_calibration_source_pool_v3'
    if (pool.get('schema') not in ('prospective_calibration_source_pool_v2','prospective_calibration_source_pool_v3')
            or pool.get('declared_before_any_model_predictions') is not True
            or any(pool.get(k) is not True for k in
                   ('training_forbidden','model_selection_forbidden','reserved_test_20_untouched'))
            or pool.get('per_class') != 30 or pool.get('seed') != 17 or len(cohorts) != 2):
        raise ValueError('Unchanged prospective pool-v2 contract required')
    for key,value in dict(per_class_target=30,confidence=.85,temperature_range=[.5,5],
            minimum_independent_actual_events=20,minimum_independent_confident_predictions=20,
            minimum_selective_precision=.9,minimum_precision_wilson95_lower=.8,
            maximum_false_success_wilson95_upper=.1).items():
        if pool.get('unchanged_gates',{}).get(key)!=value:
            raise ValueError('Prospective statistical safety gate changed')
    groups=[];unavailable=[];bindings={};reserved=None
    counts=(pool['expected_first_sources'],pool['expected_supplement_sources'])
    for (cohort,sha,config_sha),count in zip(cohorts,counts):
        if (cohort.get('schema') != 'recovery_independent_calibration_only_cohort_v1'
                or cohort.get('model_predictions_read') is not False
                or cohort.get('declared_before_any_model_predictions') is not True
                or cohort.get('training_forbidden') is not True
                or cohort.get('model_selection_forbidden') is not True
                or cohort['selected_observer_sha256'] != pool['selected_observer_sha256']
                or cohort['prospective_config_sha256'] != config_sha
                or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
                or len(cohort['calibration_groups']) != count or cohort['declared_sources'] != count):
            raise ValueError('Unbound or prediction-exposed prospective source cohort')
        test=set(cohort['external_frozen_test_source_groups'])
        if len(test) != 20 or (reserved is not None and reserved != test):
            raise ValueError('Reserved test metadata changed; never read its observations')
        reserved=test;local_missing=[]
        for row in cohort['calibration_groups']:
            group=row['source_group']
            if group in groups or group in reserved:
                raise ValueError('Duplicated or reserved-test source in prospective pool')
            groups.append(group)
            if 'unavailable_collection' in row:
                missing=row['unavailable_collection'];local_missing.append(group)
                unavailable.append(dict(source_group=group,reason=missing['reason'],
                    receipt_sha256=missing['receipt_sha256']))
            else:bindings[group]=sha
        if set(local_missing) != set(cohort['unavailable_source_groups']):
            raise ValueError('Lost unavailable-source denominator')
    if len(groups) != pool['declared_sources']:
        raise ValueError('Prospective declared pool size changed')
    result=dict(declared_groups=sorted(groups),unavailable_sources=unavailable,
        source_cohort_by_group=bindings,reserved_frozen_test_groups=sorted(reserved))
    if v3:
        if (exposed_groups is None or sorted(set(groups)&set(exposed_groups))!=pool.get('previously_exposed_declared_groups')
                or pool.get('exposure_quarantined_before_any_prediction') is not True
                or len(pool.get('fit_admission_sha256',''))!=64):
            raise ValueError('Exact actual-fit exposure required before prospective v3 selection')
        ineligible=sorted(set(bindings)&set(exposed_groups))
        for group in ineligible:del bindings[group]
        result['ineligible_sources']=[dict(source_group=g,reason='previous_training_or_model_selection_source',
            exposure_admission_sha256=pool['fit_admission_sha256']) for g in ineligible]
        result['previously_exposed_declared_groups']=pool['previously_exposed_declared_groups']
    elif exposed_groups is not None and set(bindings)&set(exposed_groups):
        raise ValueError('Pool-v2 contains actual TRAIN/selection-DEV groups; no prediction permitted')
    return result


def make_pool_selection(pool, provenance, rows, *, pool_sha256, fit_result_sha256):
    """Freeze deterministic signed anchors, not the model's preferred examples."""
    for row in rows:
        c,a=row['candidate'],row['approval']
        if a.get('cohort_sha256') != provenance['source_cohort_by_group'].get(c['source_group']):
            raise ValueError('Original signed cohort identity lost in union')
    result=preselect_pool_anchors(rows,provenance['declared_groups'],provenance['unavailable_sources'],
        per_class=pool['per_class'],seed=pool['seed'],ineligible_sources=provenance.get('ineligible_sources'))
    return dict(result,schema=('independent_calibration_pool_selection_v3'
        if pool['schema'].endswith('_v3') else 'independent_calibration_pool_selection_v2'),
        declared_before_model_predictions=True,per_class=pool['per_class'],seed=pool['seed'],
        cohort_sha256=pool_sha256,source_pool_sha256=pool_sha256,
        selected_observer_sha256=pool['selected_observer_sha256'],high_sha256=pool['high_sha256'],
        fit_result_sha256=fit_result_sha256,mechanism='GRASP',
        training_forbidden=True,model_selection_forbidden=True,
        only_class_balanced_phase_support=True,certifies_other_mechanisms=False)


def require_prospective_selection(cfg, pool, selection, partition, rows, exposed):
    if (cfg.get('schema') != 'prospective_observer_adapter_calibration_v1'
            or selection.get('schema') != ('independent_calibration_pool_selection_v3'
                if pool['schema'].endswith('_v3') else 'independent_calibration_pool_selection_v2')
            or partition.get('schema') != 'recovery_evaluation_partition_v3'
            or selection.get('mechanism') != 'GRASP'
            or selection['per_class'] != pool['per_class'] or selection['seed'] != pool['seed']
            or selection.get('source_pool_sha256') != cfg['source_pool_sha256']
            or partition['cohort_sha256'] != cfg['source_pool_sha256']
            or selection['cohort_sha256'] != cfg['source_pool_sha256']):
        raise ValueError('Pinned prospective adapter/cohort/anchor-selection contract required')
    for key in ('selected_observer_sha256','high_sha256'):
        if not cfg[key] == selection[key] == pool[key]:
            raise ValueError('Model may not be reselected on calibration predictions')
    if cfg['fit_result_sha256'] != selection['fit_result_sha256']:
        raise ValueError('Original completed fit changed')
    groups={r['candidate']['source_group'] for r in rows}
    if groups & set(exposed) or groups != set(partition['calibration_groups']):
        raise ValueError('Missing cohort data or overlap with training/model-selection groups')
    from recovery_evaluation_partition import validate_partition
    validate_partition(partition,rows)
    chosen=selected_rows(rows,selection)
    if len(chosen) != 90 or len({r['candidate']['source_group'] for r in chosen}) != 90:
        raise ValueError('Ninety independent preselected phase anchors required')
    return chosen
