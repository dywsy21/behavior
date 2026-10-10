"""Fixed source-group roles, declared without reading model predictions."""


def proposed_phase_points(labels, retry):
    points, absent = [], []
    for label in ('FAILED', 'IN_PROGRESS', 'SUCCEEDED'):
        times = [t for t,v in labels.items() if v == label and (t < retry if label == 'FAILED' else t > retry)]
        if times:
            points.append((min(times), label))
        else:
            absent.append(label)
    if not points:
        if not labels:
            raise ValueError('No original anchor available even for unlabelled review')
        # An early teacher failure / intervention escape is not a supported
        # FAILED or UNKNOWN target. Show available imagery, sign nothing.
        points = [(max(labels), 'UNLABELLED')]
    return points, absent


def cohort_sources(queue, spec, role):
    external_test=spec.get('schema')=='recovery_independent_calibration_only_cohort_v1'
    if (spec.get('schema') not in ('recovery_independent_cohort_v1','recovery_independent_calibration_only_cohort_v1')
            or role not in ('calibration', 'frozen_test')
            or any(spec.get(k) is not True for k in (
                'declared_before_any_model_predictions', 'training_forbidden',
                'model_selection_forbidden', 'calibration_must_not_use_frozen_test'))):
        raise ValueError('Explicit independent, non-training cohort required')
    if external_test:
        if role!='calibration' or spec.get('frozen_test_groups'):
            raise ValueError('External reserved test must never be materialized/read by this cohort')
        for name in ('external_frozen_test_cohort_sha256','prospective_config_sha256',
                     'prospective_source_audit_sha256','selected_observer_sha256'):
            value=spec.get(name)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('External-test cohort requires immutable prospective/model bindings')
        protected=spec.get('external_frozen_test_source_groups')
        if (not isinstance(protected,list) or not protected or len(protected)!=len(set(protected))
                or any(not isinstance(g,str) or not g for g in protected)):
            raise ValueError('External reserved source-group identities required')
    declared = {}
    for kind in (('calibration',) if external_test else ('calibration','frozen_test')):
        entries = spec[kind+'_groups']
        if not entries:
            raise ValueError('Both independent roles must be nonempty')
        for entry in entries:
            group = entry['source_group']
            if group in declared or (external_test and group in protected):
                raise ValueError('Duplicate or overlapping independent group roles')
            missing=entry.get('unavailable_collection')
            if missing is not None:
                if (not external_test or not isinstance(missing,dict) or not missing.get('reason')
                        or not missing.get('receipt') or not isinstance(missing.get('receipt_sha256'),str)
                        or len(missing['receipt_sha256'])!=64
                        or any(c not in '0123456789abcdef' for c in missing['receipt_sha256'])):
                    raise ValueError('Missing observations require an explicit bound collection failure, not a result label')
            declared[group] = (kind, entry)
    actual = {}
    for row in queue:
        group = row['source_group']
        if row['split'] != 'dev' or group not in declared:
            raise ValueError('Unassigned or TRAIN source in independent cohort')
        expected = declared[group][1]
        if expected.get('unavailable_collection') is not None:
            raise ValueError('An available candidate cannot be silently marked unavailable')
        if any(row[key] != expected[key] for key in ('case', 'arm')):
            raise ValueError('Independent source identity changed')
        branches = actual.setdefault(group, {})
        if row['branch'] in branches:
            raise ValueError('Duplicated closed branch')
        branches[row['branch']] = row
    unavailable={g for g,(_,entry) in declared.items() if entry.get('unavailable_collection') is not None}
    if set(actual)|unavailable != set(declared):
        raise ValueError('Missing independent source; preserve the declared denominator')
    return [(entry, actual.get(entry['source_group'],{})) for entry in spec[role+'_groups']]
