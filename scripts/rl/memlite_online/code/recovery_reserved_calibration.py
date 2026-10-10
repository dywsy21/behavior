"""Reuse only NEVER-predicted calibration reserves, with immutable provenance.

Model predictions never choose the next rows. All previously selected groups
are excluded, whether correct or wrong. Old failed collection denominators,
old source signatures and the untouched external test roles remain explicit.
"""
import json
from pathlib import Path

from recovery_corpus import file_sha


def prediction_exposure(selection, result, predictions, *, selection_sha256, predictions_sha256):
    if (selection.get('schema')!='independent_calibration_pool_selection_v3'
            or selection.get('declared_before_model_predictions') is not True
            or result.get('status')!='completed_prospective_adapter_calibration_not_deployed'
            or result.get('optimizer_updates')!=0 or result.get('frozen_test_not_read') is not True
            or result.get('anchor_selection_sha256')!=selection_sha256
            or result.get('predictions_sha256')!=predictions_sha256
            or result.get('selected_observer_sha256')!=selection['selected_observer_sha256']
            or result.get('source_pool_sha256')!=selection['source_pool_sha256']
            or result.get('reserved_groups_not_predicted')!=selection['reserved_groups']
            or len(selection['selected_groups'])!=90 or len(predictions)!=90):
        raise ValueError('Need the complete original selection/prediction ledger, including every error')
    expected={(r['source_group'],r['sample_id'],r['value']) for r in selection['selected_anchors']}
    got={(r['source_group'],r['sample_id'],r['label']) for r in predictions}
    groups={r['source_group'] for r in predictions}
    if (len(expected)!=90 or got!=expected or len(groups)!=90
            or groups!=set(selection['selected_groups']) or groups&set(selection['reserved_groups'])):
        raise ValueError('Missing, repeated or substituted previous prediction group')
    return groups


def read_bound(root, binding):
    path=Path(root)/binding['path']
    if file_sha(path)!=binding['sha256']:raise ValueError('Changed calibration provenance: '+str(path))
    return json.loads(path.read_text())


def load_prediction_exposure(root, spec):
    if set(spec)!={'selection','result','predictions'}:raise ValueError('Exact prior exposure bindings required')
    documents={k:read_bound(root,v) for k,v in spec.items()}
    groups=prediction_exposure(documents['selection'],documents['result'],documents['predictions'],
        selection_sha256=spec['selection']['sha256'],predictions_sha256=spec['predictions']['sha256'])
    return groups,documents


def reserve_pool_sources(pool, old_pool, old_provenance, old_selection, predicted, new_cohort, *, exposed):
    """Account for old187 + new30, not a hand-picked clean sub-denominator."""
    cohort,cohort_sha,source_config_sha=new_cohort
    old_groups=set(old_provenance['declared_groups']);old_available=set(old_provenance['source_cohort_by_group'])
    reserved=set(old_selection['reserved_groups']);test=set(old_provenance['reserved_frozen_test_groups'])
    if (pool.get('schema')!='prospective_calibration_source_pool_v4'
            or any(pool.get(k) is not True for k in ('declared_before_any_model_predictions',
                'training_forbidden','model_selection_forbidden','reserved_test_20_untouched'))
            or pool.get('per_class')!=30 or pool.get('seed')!=17
            or pool.get('unchanged_gates')!=old_pool['unchanged_gates']
            or old_selection['declared_groups']!=old_provenance['declared_groups']
            or old_selection['unavailable_sources']!=sorted(old_provenance['unavailable_sources'],key=lambda r:r['source_group'])
            or old_selection['ineligible_sources']!=old_provenance['ineligible_sources']
            or old_available!=set(predicted)|reserved or set(predicted)&reserved
            or len(reserved)!=70 or len(predicted)!=90 or len(old_groups)!=187
            or pool.get('expected_unpredicted_reserves')!=70 or pool.get('expected_new_sources')!=30
            or pool.get('declared_sources')!=217 or pool.get('high_sha256')!=old_pool['high_sha256']
            or pool.get('fit_admission_sha256')!=old_pool['fit_admission_sha256']):
        raise ValueError('Reserve pool changed original denominator, unseen roles, fit or safety gates')
    if (cohort.get('schema')!='recovery_independent_calibration_only_cohort_v1'
            or cohort.get('model_predictions_read') is not False
            or any(cohort.get(k) is not True for k in ('declared_before_any_model_predictions',
                'training_forbidden','model_selection_forbidden'))
            or cohort.get('prospective_config_sha256')!=source_config_sha
            or cohort.get('selected_observer_sha256')!=pool['selected_observer_sha256']
            or set(cohort.get('external_frozen_test_source_groups',[]))!=test
            or len(test)!=20 or cohort.get('declared_sources')!=30
            or len(cohort.get('calibration_groups',[]))!=30 or len(cohort_sha)!=64):
        raise ValueError('Unbound fresh cohort or changed external test roles')
    bindings={g:old_provenance['source_cohort_by_group'][g] for g in sorted(reserved)}
    unavailable=list(old_provenance['unavailable_sources']);new_groups=set();new_missing=set()
    for row in cohort['calibration_groups']:
        group=row['source_group']
        if group in old_groups|new_groups|test|set(exposed):raise ValueError('Fresh source is not fresh')
        new_groups.add(group)
        if 'unavailable_collection' in row:
            missing=row['unavailable_collection'];new_missing.add(group)
            unavailable.append(dict(source_group=group,reason=missing['reason'],receipt_sha256=missing['receipt_sha256']))
        else:bindings[group]=cohort_sha
    if (new_missing!=set(cohort['unavailable_source_groups']) or set(bindings)&set(exposed)
            or len(old_groups|new_groups)!=pool['declared_sources']):
        raise ValueError('Lost unavailable-source denominator or actual-fit overlap')
    exposure=pool['prior_prediction_exposure']
    return dict(declared_groups=sorted(old_groups|new_groups),unavailable_sources=unavailable,
        ineligible_sources=old_provenance['ineligible_sources'],
        previously_predicted_sources=[dict(source_group=g,reason='previous_model_calibration_prediction',
            selection_sha256=exposure['selection']['sha256'],predictions_sha256=exposure['predictions']['sha256'])
            for g in sorted(predicted)],source_cohort_by_group=bindings,reserved_frozen_test_groups=sorted(test),
        previously_exposed_declared_groups=old_provenance['previously_exposed_declared_groups'])


def load_reserved_pool(root, repo, pool, cohorts, exposed):
    """Same CPU revalidation at preselection and before the first GPU forward."""
    from recovery_prospective_adapter import pool_sources
    if len(cohorts)!=3:raise ValueError('Two old signed cohorts and one fresh wave required')
    old_pool=read_bound(root,pool['prior_pool'])
    old_provenance=pool_sources(old_pool,cohorts[:2],exposed_groups=exposed)
    predicted,documents=load_prediction_exposure(root,pool['prior_prediction_exposure'])
    if (pool['prior_pool']['sha256']!=documents['selection']['source_pool_sha256']
            or documents['selection']['selected_observer_sha256']!=old_pool['selected_observer_sha256']):
        raise ValueError('Previous model selection belongs to a different source pool')
    return reserve_pool_sources(pool,old_pool,old_provenance,documents['selection'],predicted,cohorts[2],exposed=exposed)
