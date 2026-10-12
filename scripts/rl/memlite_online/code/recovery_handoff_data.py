"""Reviewed learned feedback INPUT overlay; never new truth or BC targets.

The original period128 pool, source-frame identities, targets and schedule are
unchanged. A SHA-bound frozen prediction may replace only execution_feedback.
Unknowns and disagreements are deliberately retained; there is no filtering by
the desired next skill, confidence or visual outcome.
"""
import copy
import json
import math

from recovery_corpus import digest, file_sha, group_key
from recovery_handoff_feedback import SCHEMA
from recovery_periodic_data import bound, validate_periodic_row

PROTOCOL = 'reviewed_frozen_handoff_input_v1'
STATUS = 'ACCEPTED_RESTRICTED_HIGH_FEEDBACK_INPUT_NOT_A_TRAINING_LAUNCH'


def feedback_overlay(original, request, predicted):
    """Pure source/clock/target gate, also used by the actual raw adapter."""
    index, _ = validate_periodic_row(original)
    identity, projection = original['identity'], original['projection']
    sample_id = digest([SCHEMA, identity])
    group = group_key(projection['task_name'], identity['instance'])
    if (identity['instance'] >= 301 or request['schema'] != SCHEMA
            or request['sample_id'] != sample_id or predicted['sample_id'] != sample_id
            or request['source_group'] != group or predicted['source_group'] != group
            or request['original_identity'] != identity or predicted['original_identity'] != identity
            or request['original_projection'] != projection
            or predicted['target_projection_sha256'] != digest(projection)
            or predicted['original_execution_feedback'] != projection['execution_feedback']
            or predicted['known_previous_outcome'] != 'UNKNOWN'
            or predicted['physical_outcome_asserted'] is not False
            or predicted['new_human_admission'] is not False):
        raise ValueError('Cross-source, changed target or unapproved truth in handoff overlay')
    before = json.loads(projection['execution_feedback'])
    after = json.loads(predicted['execution_feedback'])
    mutable = {'estimated_member_outcomes', 'estimated_bundle_outcome'}
    if set(before) != set(after) or any(before[k] != after[k] for k in set(before) - mutable):
        raise ValueError('Handoff prediction changed causal counters/source')
    members = after['estimated_member_outcomes']
    # This first admitted universe contains exactly one calibrated GRASP member.
    # Expanding scope requires a separately reviewed/calibrated protocol.
    if len(members) != 1 or members[0].keys() != {'confidence', 'estimated_outcome', 'member'}:
        raise ValueError('Changed admitted single-member observer contract')
    member = members[0]; outcome = member['estimated_outcome']; confidence = member['confidence']
    if (type(member['member']) is not int or member['member'] != 0
            or outcome not in ('SUCCEEDED', 'FAILED', 'UNKNOWN')
            or type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1 or after['estimated_bundle_outcome'] != outcome
            or (outcome != 'UNKNOWN' and confidence < .85)):
        raise ValueError('Invalid calibrated estimate or unready confidence')
    # Production retains min(raw confidences) when two threshold-gated UNKNOWN
    # values agree. UNKNOWN therefore need not have confidence0, and this field
    # is not generally P(UNKNOWN). Preserve the serving semantics; the CPU gate
    # separately replays the two actual logits through that same ledger.
    from g05.utils.memlite_skill_protocol import parse_active_skills_text
    previous = parse_active_skills_text(projection['previous_intent'])
    if len(previous) != 1 or previous[0]['verb'] != 'GRASP':
        raise ValueError('Feedback outside the admitted GRASP calibration')
    result = copy.deepcopy(projection)
    result['execution_feedback'] = predicted['execution_feedback']
    # All non-input fields, including memory_update, remain byte-for-byte values.
    if any(result[k] != v for k, v in projection.items() if k != 'execution_feedback'):
        raise ValueError('Handoff input overlay changed a teacher target')
    return index, result


class HandoffFeedbackPool:
    def __init__(self, root, spec, periodic, *, repo):
        if spec.get('protocol') != PROTOCOL:
            raise ValueError('Unrecognized handoff training input')
        self.admission_path = bound(repo, spec['admission'])
        if file_sha(self.admission_path) != spec['admission_sha256']:
            raise ValueError('Changed personal handoff admission')
        admission = json.loads(self.admission_path.read_text())
        scope, source, review = (admission[k] for k in ('admission_scope', 'source', 'human_review'))
        if (admission['status'] != STATUS or admission['training_admitted'] is not True
                or scope['allowed_component'] != 'high_planner_only estimated execution_feedback input'
                or scope['allowed_target_change'] is not False
                or scope['required_known_previous_outcome'] != 'UNKNOWN'
                or scope['required_outcome_supervision_mask'] is not False
                or scope['required_low_action_supervision_mask'] is not False
                or scope['required_task_complete'] is not False
                or scope['physical_outcome_asserted'] is not False
                or scope['observer_refit_allowed'] is not False or scope['low_bc_allowed'] is not False
                or not scope['preserve_all383_estimates_including_unknown_and_failures']
                or not scope['all_other_periodic_candidates_keep_original_feedback']
                or review['personally_reviewed_cases'] != 160 or review['tasks'] != 97
                or source['rows'] != 383 or source['candidate_sha256'] != periodic.spec['candidate_sha256']):
            raise ValueError('Missing restricted handoff personal-review scope')
        self.files = {'handoff_admission': self.admission_path}
        control = bound(root, spec['control_schedule']['path'])
        if file_sha(control) != spec['control_schedule']['sha256']:
            raise ValueError('Changed same-parent UNKNOWN-feedback control schedule')
        self.files['handoff_control_schedule'] = control
        for key in ('requests', 'feedback', 'prediction_result'):
            path = bound(root, source[key])
            if file_sha(path) != source[key + '_sha256']:
                raise ValueError('Changed handoff evidence: ' + key)
            self.files['handoff_' + key] = path
        result = json.loads(self.files['handoff_prediction_result'].read_text())
        if (result['status'] != 'frozen_handoff_predictions_completed_not_admitted'
                or result['optimizer_updates'] != 0 or result['rows'] != 383 or result['head_checks'] != 766
                or result['observer_before_sha256'] != result['observer_after_sha256']
                or not result['observer_head_unchanged'] or not result['all_rows_retained']
                or result['physical_outcome_asserted'] is not False
                or result['requests_sha256'] != source['requests_sha256']
                or result['feedback_sha256'] != source['feedback_sha256']
                or result['cpu_audit_sha256'] != source['cpu_audit_sha256']):
            raise ValueError('Incomplete or changed frozen observer evidence')
        for key in ('trace', 'checks'):
            path = self.files['handoff_prediction_result'].parent / (key + '.jsonl')
            if file_sha(path) != result[key + '_sha256']:
                raise ValueError('Changed handoff raw prediction trace')
            self.files['handoff_' + key] = path
        review_path = bound(root, review['review_manifest'])
        if file_sha(review_path) != review['review_manifest_sha256']:
            raise ValueError('Changed personally reviewed images/row manifest')
        self.files['handoff_review_manifest'] = review_path
        self.admission, self.result = admission, result
        requests = [json.loads(line) for line in self.files['handoff_requests'].read_text().splitlines()]
        predictions = [json.loads(line) for line in self.files['handoff_feedback'].read_text().splitlines()]
        predicted = {p['sample_id']: p for p in predictions}
        if (len(requests) != 383 or len(predictions) != 383 or len(predicted) != 383
                or set(predicted) != {r['sample_id'] for r in requests}):
            raise ValueError('Missing, duplicate or filtered handoff rows')
        self.by_index = {}; groups = set(); tasks = set()
        for request in requests:
            index = request['original_identity']['original_candidate_index']
            if index not in periodic.by_index or index in self.by_index:
                raise ValueError('Foreign or duplicated periodic source anchor')
            _, projection = feedback_overlay(periodic.by_index[index], request, predicted[request['sample_id']])
            self.by_index[index] = projection
            groups.add(request['source_group']); tasks.add(request['original_identity']['task'])
        if len(groups) != source['source_groups'] or len(tasks) != source['tasks']:
            raise ValueError('Changed admitted source-group/task coverage')

    def projection(self, index, original_projection):
        value = self.by_index.get(index)
        if value is None:
            return copy.deepcopy(original_projection)
        if set(value) != set(original_projection) or any(
                value[k] != original_projection[k] for k in value if k != 'execution_feedback'):
            raise ValueError('Actual source projection differs from admitted handoff target')
        return copy.deepcopy(value)


def validate_handoff_audit(audit, ticket, recipe):
    """Current frozen source and actual adapter, not the older UNKNOWN audit."""
    spec = recipe['H1']['periodic_teacher']
    if (audit.get('schema') != 'handoff_planner_processor_audit_v1' or audit.get('status') != 'passed'
            or audit.get('source_commit') != ticket['source_commit']
            or audit.get('recipe_sha256') != ticket['files']['recipe']['sha256']
            or audit.get('handoff_admission_sha256') != spec['learned_handoff']['admission_sha256']
            or audit.get('feedback_sha256') != ticket['files']['handoff_feedback']['sha256']
            or audit.get('candidate_sha256') != spec['candidate_sha256']
            or audit.get('original_targets_reconstructed') != 1800
            or audit.get('learned_feedback_rows') != 383 or audit.get('tokenized_rows') != 383
            or audit.get('reviewed_current_images_matched') != 480
            or audit.get('original_observations_match') is not True
            or audit.get('estimates_recomputed_from_logits') is not True
            or audit.get('training_serving_prefix_equal') is not True
            or audit.get('same_periodic_control_schedule') is not True
            or audit.get('outcome_tokens_supervised') != 0 or audit.get('physical_truth_added') != 0
            or audit.get('low_action_supervision') is not False
            or audit.get('model_forwards') != 0 or audit.get('optimizer_updates') != 0):
        raise ValueError('Missing actual source-bound handoff adapter/token acceptance')


def validate_handoff_launch(ticket, recipe):
    from recovery_periodic_data import PeriodicTeacherPool, REPO
    spec = recipe['H1']['periodic_teacher']
    pool = PeriodicTeacherPool(recipe['root'], spec, repo=REPO)
    if pool.handoff is None or 'handoff_processor_audit' not in ticket['files']:
        raise ValueError('Missing registered learned-feedback launch inputs')
    for key, path in pool.handoff.files.items():
        entry = ticket['files'].get(key)
        if entry is None or entry['sha256'] != file_sha(path) or str(path.resolve()) != entry['path']:
            raise ValueError('Different pinned learned-feedback input: '+key)
    result = pool.handoff.result; high = recipe['H1']
    if (result['observer_backbone_sha256'] != high['observer_backbone_sha256']
            or result['observer_adapter_sha256'] != high['observer_sha256']
            or result['calibration_sha256'] != high['calibration']['sha256']):
        raise ValueError('Handoff training substituted the frozen observer identity')
    import json
    from pathlib import Path
    path = Path(ticket['files']['handoff_processor_audit']['path'])
    validate_handoff_audit(json.loads(path.read_text()), ticket, recipe)
