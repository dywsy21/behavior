"""Deterministic, source-disjoint EVAL selection before any model predictions."""
from collections import Counter, defaultdict
import copy

from recovery_corpus import digest
from recovery_handoff_feedback import EVAL_SCHEMA

SELECTION = 'heldout-periodic128-grasp-2-per-task-bucket-v1'


def generation_split(config):
    if config.get('schema') == 'periodic_handoff_observer_generation_v1':
        if config.get('purpose') not in (None, 'train_input_generation'):
            raise ValueError('TRAIN generation cannot change its declared purpose')
        if config.get('human_review_required_before_fit') is not True:
            raise ValueError('TRAIN input generation requires personal admission')
        return 'train'
    if (config.get('schema') == 'periodic_handoff_observer_eval_generation_v1'
            and config.get('purpose') == 'heldout_eval_only'
            and config.get('training_admission_forbidden') is True
            and config.get('human_review_required_before_evaluation') is True):
        return 'eval'
    raise ValueError('Explicit separate TRAIN or heldout-only generation contract required')


def selection_bucket(change_class):
    if change_class == 'identical_command':
        return 'continuation'
    if change_class in ('explicit_verb_or_bundle_change', 'explicit_bound_object_or_goal_change'):
        return 'handoff'
    # An unresolved relation/source-description change is not established as a
    # different commanded skill. Count it as excluded, never invent its roles.
    return None


def select_heldout_requests(pairs, *, excluded_groups):
    """Two distinct sources per task/bucket, no outcome/confidence filtering."""
    by_source = {}; counts = Counter(); seen = set()
    for original, request in pairs:
        identity = original['identity']; group = request['source_group']
        if (identity['split'] != 'eval' or request['schema'] != EVAL_SCHEMA
                or request['original_identity'] != identity
                or request['original_projection'] != original['projection']
                or group in excluded_groups or request['input_feedback_generated'] is not False
                or request['physical_outcome_asserted'] is not False
                or request['optimizer_updates'] != 0
                or request['sample_id'] != digest([EVAL_SCHEMA, identity])):
            raise ValueError('Selection accepts original source-disjoint EVAL requests only')
        if request['sample_id'] in seen:
            raise ValueError('Duplicate heldout decision')
        seen.add(request['sample_id'])
        bucket = selection_bucket(request['target_change_class'])
        counts[request['target_change_class']] += 1
        if bucket is None:
            continue
        key = (identity['task'], bucket, group)
        score = digest([SELECTION, identity])
        if key not in by_source or score < by_source[key][0]:
            by_source[key] = (score, original, request)
    strata = defaultdict(list)
    for (task, bucket, group), value in by_source.items():
        strata[(task, bucket)].append(value)
    result = []
    for key in sorted(strata):
        result.extend((original, request) for _, original, request in sorted(strata[key], key=lambda x:x[0])[:2])
    return result, dict(counts)


def replay_heldout_projection(request, prediction, episode, trace, calibration):
    """Recompute real estimates from the saved logits, never promote to truth."""
    from recovery_handoff_inference import feedback_from_trace
    target = request['original_projection']; identity = request['original_identity']
    if (request['schema'] != EVAL_SCHEMA or identity['split'] != 'eval' or episode['split'] != 'eval'
            or prediction['sample_id'] != request['sample_id']
            or prediction['source_group'] != request['source_group']
            or prediction['original_identity'] != identity
            or prediction['target_projection_sha256'] != digest(target)
            or prediction['original_execution_feedback'] != target['execution_feedback']
            or prediction['known_previous_outcome'] != 'UNKNOWN'
            or prediction['physical_outcome_asserted'] is not False
            or prediction['new_human_admission'] is not False
            or target['known_previous_outcome'] != 'UNKNOWN' or target['outcome_target'] != 'UNKNOWN'
            or target['outcome_supervision_mask'] or target['low_action_supervision_mask']):
        raise ValueError('Mismatched heldout estimate/target or invented outcome supervision')
    feedback = feedback_from_trace(request, episode, trace, calibration)
    if feedback != prediction['execution_feedback']:
        raise ValueError('Estimate differs from actual frozen observer logits/production ledger')
    projection = copy.deepcopy(target); projection['execution_feedback'] = feedback
    return projection


def validate_fixed_fit(result, latest, model):
    """Only the predeclared final360, not a partial/best intermediate model."""
    from pathlib import PurePosixPath
    if (result['status'] != 'completed_finite_schedule' or result['engineering_only'] is not False
            or result['step'] != 360 or result['planned_updates'] != 360
            or result['frozen_before_sha256'] != result['frozen_after_sha256']
            or result['parent_sha256'] != 'a74c5cc147dcd632dfddcef780f2268845b13c9c4169f61d2aa7f90a875d5bf2'
            or latest['state']['step'] != 360 or latest['sha256'] != model['sha256']
            or latest['path'] != PurePosixPath(model['path']).name):
        raise ValueError('Require same-parent fully completed fixed final360 and original save receipt')
