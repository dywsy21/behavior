"""Deterministic, source-disjoint EVAL selection before any model predictions."""
from collections import Counter, defaultdict

from recovery_corpus import digest
from recovery_handoff_feedback import EVAL_SCHEMA

SELECTION = 'heldout-periodic128-grasp-2-per-task-bucket-v1'


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
