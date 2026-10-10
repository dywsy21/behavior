"""Source-disjoint planner supervision with an already frozen observer.

This is not a new observer fit, a new calibration population or an approval.
Preserve original train/dev roles and every attempted source in the ledger.
"""
from collections import defaultdict
import math

from recovery_corpus import digest


def calibration_gate(calibration):
    """Check the original selective contract; never fit or loosen a threshold."""
    if (calibration.get('schema') != 'recovery_observer_calibration_v1'
            or calibration.get('ready') is not True or calibration.get('blockers') != []
            or calibration.get('mechanism') != 'GRASP'
            or calibration.get('certifies_other_mechanisms') is not False
            or calibration.get('minimum_confidence') != .85
            or not isinstance(calibration.get('temperature'), (int, float))
            or not math.isfinite(calibration['temperature'])
            or not .5 <= calibration['temperature'] <= 5):
        raise ValueError('Require the fixed original GRASP-only calibration contract')
    from recovery_observer_training import wilson
    for name in ('IN_PROGRESS', 'SUCCEEDED', 'FAILED'):
        row = calibration['classes'][name]
        actual, total, correct = (row[k] for k in ('actual_events', 'confident_predictions', 'correct'))
        if (any(type(v) is not int for v in (actual, total, correct))
                or actual != 30 or total < 20 or not 0 <= correct <= total
                or correct / total < .9 or wilson(correct, total)[0] < .8):
            raise ValueError('Original independent class support/precision gate failed')
    if (calibration.get('events') != 90 or len(set(calibration['source_groups'])) != 90
            or type(calibration.get('false_success_count')) is not int
            or not 0 <= calibration['false_success_count'] <= 60
            or wilson(calibration['false_success_count'], 60)[1] > .1):
        raise ValueError('Original false-success or source-independence gate failed')
    return True


def exposure_manifest(fit_rows, selection, calibration, historical_groups, frozen_test_groups, bindings):
    calibration_gate(calibration)
    train = {r['candidate']['source_group'] for r in fit_rows if r['candidate']['split'] == 'train'}
    dev = {r['candidate']['source_group'] for r in fit_rows if r['candidate']['split'] == 'dev'}
    if (not train or not dev or train & dev
            or any(r['candidate']['split'] not in ('train', 'dev') for r in fit_rows)):
        raise ValueError('Require the actual disjoint TRAIN and selection DEV of the fixed fit')
    selected = set(selection['selected_groups'])
    declared = set(selection['declared_groups'])
    if (selection.get('schema') != 'independent_calibration_pool_selection_v4'
            or len(selected) != 90 or selected != set(calibration['source_groups'])
            or not selected <= declared or selected & (train | dev)
            or selection['selected_observer_sha256'] != calibration['observer_sha256']
            or selection['high_sha256'] != calibration['high_sha256']):
        raise ValueError('Calibration/source/model identity differs from the completed fixed fit')
    reserved = declared | set(historical_groups) | set(frozen_test_groups)
    excluded = sorted(train | dev | reserved)
    if any(not isinstance(g, str) or not g for g in excluded):
        raise ValueError('Explicit original source-instance groups required')
    return dict(schema='postfit_observer_exposure_v1',
        observer_sha256=calibration['observer_sha256'], high_sha256=calibration['high_sha256'],
        observer_train_groups=sorted(train), observer_selection_groups=sorted(dev),
        calibration_groups=sorted(selected), all_reserved_groups=sorted(reserved),
        excluded_groups=excluded, bindings=bindings,
        calibrated_mechanisms=['GRASP'], source_labels_approved=False,
        interpretation='Only new H1 targets outside ALL observer fit/selection and prior reserved '
            'populations are eligible. Original train/dev split is immutable. No CAL/test reuse, '
            'observer retraining, threshold change, physical rollout or automatic data approval.')


def select_postfit_sources(queue, exposure):
    """All unseen source groups are retained; first structurally positive branch.

    Positive teacher continuation selection is for SFT, NOT a policy success
    estimate. Unsuccessful/ambiguous candidates stay in the preparation ledger.
    No predicted outcome is available to this selection function.
    """
    if (exposure.get('schema') != 'postfit_observer_exposure_v1'
            or exposure.get('calibrated_mechanisms') != ['GRASP']
            or exposure.get('source_labels_approved') is not False):
        raise ValueError('Pinned post-fit exclusion contract required')
    excluded = set(exposure['excluded_groups'])
    expected = (set(exposure['observer_train_groups']) | set(exposure['observer_selection_groups'])
                | set(exposure['all_reserved_groups']))
    if excluded != expected or not set(exposure['calibration_groups']) <= excluded:
        raise ValueError('Incomplete fit/calibration exclusion')
    groups = defaultdict(dict)
    for row in queue:
        g, branch = row['source_group'], row['branch']
        if row['split'] not in ('train', 'dev') or branch in groups[g]:
            raise ValueError('Invalid original split or repeated branch')
        groups[g][branch] = row
    selected, ledger = [], []
    for group in sorted(groups, key=lambda g: digest(['postfit-planner-v1', g])):
        branches = groups[group]
        if (len({r['split'] for r in branches.values()}) != 1
                or len({r['case'] for r in branches.values()}) != 1):
            raise ValueError('Cross-split or cross-case source group')
        row = next(iter(branches.values()))
        item = dict(source_group=group, case=row['case'], split=row['split'])
        if group in excluded:
            ledger.append(dict(item, status='excluded_observer_fit_or_reserved_source'))
            continue
        candidates = [branches[b] for b in ('open_gripper_joint_jitter', 'open_gripper')
                      if b in branches and branches[b]['physical_recovery_candidate']
                      and branches[b]['failure'] is None]
        if not candidates:
            ledger.append(dict(item, status='no_physically_verified_positive_continuation',
                branches={k:dict(physical_recovery_candidate=v['physical_recovery_candidate'],
                    failure=v['failure']) for k, v in branches.items()}))
            continue
        chosen = candidates[0]
        selected.append(chosen)
        ledger.append(dict(item, status='pending_original_media_and_semantic_review',
                           selected_branch=chosen['branch']))
    if not selected:
        raise ValueError('No source-disjoint positive teacher candidates; do not reclassify exclusions')
    return selected, ledger
