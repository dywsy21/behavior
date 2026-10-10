"""Group-level calibration/test roles, separate from original TRAIN/DEV.

An old DEV set used to select a model is not a blind test. New independent
DEV groups are preserved in the release but excluded by every training loader.
"""


def _groups(value):
    if (not isinstance(value, list) or any(not isinstance(g, str) or not g for g in value)
            or len(value) != len(set(value))):
        raise ValueError('Invalid or duplicate evaluation groups')
    return set(value)


INDEPENDENT_SCHEMAS = ('recovery_evaluation_partition_v2', 'recovery_evaluation_partition_v3')


def _cohort_bindings(partition):
    """A pool union retains each signed source cohort; never rewrites approval SHA."""
    bindings = partition.get('source_cohort_by_group')
    if partition.get('schema') != 'recovery_evaluation_partition_v3':
        if bindings is not None:
            raise ValueError('Multiple signed cohorts require explicit v3 provenance')
        return None
    expected = (set(partition['calibration_groups']) | set(partition['frozen_test_groups'])) - set(
        partition.get('reserved_frozen_test_groups', []))
    if (not isinstance(bindings, dict) or set(bindings) != expected
            or any(not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
                   for s in bindings.values())):
        raise ValueError('Every admitted independent group needs its original signed cohort SHA')
    return bindings


def _guard_approval_roles(rows, partition):
    bindings = _cohort_bindings(partition) if partition is not None else None
    for row in rows:
        approval = row.get('approval', {})
        role = approval.get('usage_role')
        group = row['candidate']['source_group']
        if bindings is not None and group in bindings and role is None:
            raise ValueError('Pool union cannot erase a signed independent usage role')
        if role is None:
            continue
        if (role not in ('calibration', 'frozen_test') or partition is None
                or partition.get('schema') not in INDEPENDENT_SCHEMAS
                or group not in partition[role+'_groups']
                or approval.get('cohort_sha256') != (bindings.get(group) if bindings is not None
                                                     else partition.get('cohort_sha256'))):
            raise ValueError('Independent approval needs its exact non-training cohort role')


def build_partition(spec, unit_rows):
    if spec is None:
        _guard_approval_roles([r for rows in unit_rows for r in rows], None)
        return None
    if not unit_rows or not spec.get('no_added_training_rows_expected'):
        raise ValueError('This partition only permits a fixed training pool')
    if spec.get('schema') in ('recovery_evaluation_partition_spec_v2', 'recovery_evaluation_partition_spec_v3'):
        calibration = _groups(spec['calibration_groups'])
        frozen = _groups(spec['frozen_test_groups'])
        reserved = _groups(spec.get('reserved_frozen_test_groups', []))
        sha = spec.get('cohort_sha256', '')
        if (not calibration or not frozen or calibration & frozen or not reserved <= frozen
                or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
                or spec.get('calibration_may_not_select_checkpoints') is not True):
            raise ValueError('Invalid independent calibration/test contract')
        base_count = spec.get('base_unit_count', 1)
        if type(base_count) is not int or base_count not in (0, 1):
            raise ValueError('Independent release needs one fixed base or transport-only zero base')
        base = unit_rows[0] if base_count else []
        old = {r['candidate']['source_group'] for r in base}
        new = [r for rows in unit_rows[base_count:] for r in rows]
        if old & (calibration | frozen):
            raise ValueError('Independent cohort overlaps previously exposed groups')
        if (any(r['candidate']['split'] != 'dev' for r in new)
                or {r['candidate']['source_group'] for r in new} != (calibration | frozen)-reserved):
            raise ValueError('New DEV must exactly match admitted independent roles')
        partition = dict(schema=('recovery_evaluation_partition_v3'
            if spec['schema'].endswith('_v3') else 'recovery_evaluation_partition_v2'),
            training_groups=sorted({r['candidate']['source_group'] for r in base if r['candidate']['split']=='train'}),
            selection_dev_groups=sorted({r['candidate']['source_group'] for r in base if r['candidate']['split']=='dev'}),
            calibration_groups=sorted(calibration), frozen_test_groups=sorted(frozen),
            reserved_frozen_test_groups=sorted(reserved), cohort_sha256=sha,
            calibration_may_not_select_checkpoints=True,
            test_may_not_select_checkpoints_or_calibration=True)
        if 'source_cohort_by_group' in spec:
            partition['source_cohort_by_group'] = dict(spec['source_cohort_by_group'])
        validate_partition(partition, base+new)
        return partition
    if spec.get('schema') is not None or 'calibration_groups' in spec:
        raise ValueError('Unknown evaluation partition specification')
    frozen = set(spec['new_independent_groups'])
    if not frozen or len(frozen) != len(spec['new_independent_groups']):
        raise ValueError('Empty or duplicate frozen-test groups')
    base = unit_rows[0]
    old_groups = {r['candidate']['source_group'] for r in base}
    new = [r for rows in unit_rows[1:] for r in rows]
    if frozen & old_groups:
        raise ValueError('Frozen test overlaps previously exposed groups')
    if any(r['candidate']['split'] != 'dev' for r in new):
        raise ValueError('No new TRAIN rows allowed in this evaluation release')
    if {r['candidate']['source_group'] for r in new} != frozen:
        raise ValueError('New DEV groups must exactly match frozen-test declaration')
    partition = dict(schema='recovery_evaluation_partition_v1',
        frozen_test_groups=sorted(frozen),
        selection_dev_groups=sorted({r['candidate']['source_group'] for r in base
                                     if r['candidate']['split'] == 'dev'}),
        training_groups=sorted({r['candidate']['source_group'] for r in base
                                if r['candidate']['split'] == 'train'}),
        test_may_not_select_checkpoints_or_calibration=True,
        status='frozen_cross_task_diagnostic_not_deployment_calibration')
    validate_partition(partition, base + new)
    return partition


def validate_partition(partition, rows):
    if (partition.get('schema') not in ('recovery_evaluation_partition_v1', *INDEPENDENT_SCHEMAS)
            or partition.get('test_may_not_select_checkpoints_or_calibration') is not True):
        raise ValueError('Invalid frozen-test role contract')
    groups = [_groups(partition[k]) for k in
              ('training_groups', 'selection_dev_groups', 'frozen_test_groups')]
    reserved = set()
    if partition['schema'] in INDEPENDENT_SCHEMAS:
        groups.append(_groups(partition['calibration_groups']))
        reserved = _groups(partition.get('reserved_frozen_test_groups', []))
        sha = partition.get('cohort_sha256', '')
        if (partition.get('calibration_may_not_select_checkpoints') is not True
                or not reserved <= groups[2] or len(sha) != 64
                or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('Invalid calibration/reserved-test role contract')
    elif 'calibration_groups' in partition:
        raise ValueError('Legacy partition cannot silently ignore calibration roles')
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i+1, len(groups))):
        raise ValueError('Overlapping evaluation roles')
    dev = set().union(*groups[1:])
    for row in rows:
        group, split = row['candidate']['source_group'], row['candidate']['split']
        if group in reserved:
            raise ValueError('Reserved frozen test may not be admitted/read yet')
        expected = 'train' if group in groups[0] else 'dev' if group in dev else None
        if expected is None or expected != split:
            raise ValueError('Unassigned group or split drift in evaluation partition')
    _guard_approval_roles(rows, partition)


def rows_for_purpose(rows, partition, purpose):
    if purpose not in ('training', 'feature_extraction', 'frozen_evaluation', 'calibration'):
        raise ValueError('Unknown data access purpose')
    if partition is None:
        _guard_approval_roles(rows, None)
        if purpose in ('frozen_evaluation', 'calibration'):
            raise ValueError('Independent evaluation requires explicit group roles')
        return rows
    validate_partition(partition, rows)
    frozen = set(partition['frozen_test_groups'])
    calibration = set(partition.get('calibration_groups', []))
    if purpose == 'training':
        return [r for r in rows if r['candidate']['source_group'] not in frozen | calibration]
    if purpose == 'calibration':
        if partition['schema'] not in INDEPENDENT_SCHEMAS:
            raise ValueError('Independent calibration requires explicit calibration roles')
        return [r for r in rows if r['candidate']['source_group'] in calibration]
    if purpose == 'frozen_evaluation':
        return [r for r in rows if r['candidate']['source_group'] in frozen]
    return rows
