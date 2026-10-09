"""Group-level frozen-test roles, separate from the original TRAIN/DEV split.

An old DEV set used to select a model is not a blind test. New independent
DEV groups are preserved in the release but excluded by every training loader.
"""


def build_partition(spec, unit_rows):
    if spec is None:
        return None
    if not unit_rows or not spec.get('no_added_training_rows_expected'):
        raise ValueError('This partition only permits a fixed training pool')
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
    if (partition.get('schema') != 'recovery_evaluation_partition_v1'
            or partition.get('test_may_not_select_checkpoints_or_calibration') is not True):
        raise ValueError('Invalid frozen-test role contract')
    groups = [set(partition[k]) for k in
              ('training_groups', 'selection_dev_groups', 'frozen_test_groups')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Overlapping evaluation roles')
    for row in rows:
        group, split = row['candidate']['source_group'], row['candidate']['split']
        expected = 'train' if group in groups[0] else 'dev' if group in groups[1] | groups[2] else None
        if expected is None or expected != split:
            raise ValueError('Unassigned group or split drift in evaluation partition')


def rows_for_purpose(rows, partition, purpose):
    if purpose not in ('training', 'feature_extraction', 'frozen_evaluation'):
        raise ValueError('Unknown data access purpose')
    if partition is None:
        if purpose == 'frozen_evaluation':
            raise ValueError('A frozen evaluation requires explicit group roles')
        return rows
    validate_partition(partition, rows)
    frozen = set(partition['frozen_test_groups'])
    if purpose == 'training':
        return [r for r in rows if r['candidate']['source_group'] not in frozen]
    if purpose == 'frozen_evaluation':
        return [r for r in rows if r['candidate']['source_group'] in frozen]
    return rows
