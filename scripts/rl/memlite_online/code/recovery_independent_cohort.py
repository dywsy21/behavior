"""Fixed source-group roles, declared without reading model predictions."""


def cohort_sources(queue, spec, role):
    if (spec.get('schema') != 'recovery_independent_cohort_v1'
            or role not in ('calibration', 'frozen_test')
            or any(spec.get(k) is not True for k in (
                'declared_before_any_model_predictions', 'training_forbidden',
                'model_selection_forbidden', 'calibration_must_not_use_frozen_test'))):
        raise ValueError('Explicit independent, non-training cohort required')
    declared = {}
    for kind in ('calibration', 'frozen_test'):
        entries = spec[kind+'_groups']
        if not entries:
            raise ValueError('Both independent roles must be nonempty')
        for entry in entries:
            group = entry['source_group']
            if group in declared:
                raise ValueError('Duplicate or overlapping independent group roles')
            declared[group] = (kind, entry)
    actual = {}
    for row in queue:
        group = row['source_group']
        if row['split'] != 'dev' or group not in declared:
            raise ValueError('Unassigned or TRAIN source in independent cohort')
        expected = declared[group][1]
        if any(row[key] != expected[key] for key in ('case', 'arm')):
            raise ValueError('Independent source identity changed')
        branches = actual.setdefault(group, {})
        if row['branch'] in branches:
            raise ValueError('Duplicated closed branch')
        branches[row['branch']] = row
    if set(actual) != set(declared):
        raise ValueError('Missing independent source; preserve the declared denominator')
    return [(entry, actual[entry['source_group']]) for entry in spec[role+'_groups']]
