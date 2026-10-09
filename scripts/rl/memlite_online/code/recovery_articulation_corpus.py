"""Re-derive causal articulation labels from immutable physical measurements.

Old collector outcome proposals are not trusted. In particular a FAILED old
attempt at the instant of RETRY is never the outcome of the new attempt.
Reverse-reference actuator commands remain injected faults, never BC targets.
"""
from copy import deepcopy
import json

import numpy as np

from recovery_articulation_teacher import CausalArticulation, functional_goal, progressing
from recovery_corpus import digest, file_sha
from recovery_reference_binding import ReferenceArticulationBinding


VERBS = {'OPEN_DOOR', 'CLOSE_DOOR', 'OPEN_DRAWER', 'CLOSE_DRAWER', 'OPEN_LID', 'CLOSE_LID'}
CLEAN = 'same_state_articulation_teacher_candidate'
# This exact reviewed collector calls _apply_actions, rejects native terminal,
# then records post-action proprio/physics. Its prefix predates the explicit
# ACK field; correction branches already carry actual ACKs. Never infer this
# legacy proof for arbitrary versions or manufacture ACK fields in old data.
LEGACY_PREFIX_AFTER_APPLY = {'319b2eec8b5d1abfca1f7500f6229e970e72bb24'}


def physical_goal(physical, want_open):
    expected = functional_goal(physical['open'], physical['directed_open_fraction'], want_open)
    if physical['goal_predicate'] is not expected:
        raise ValueError('Recorded goal differs from measured functional clearance')
    if not physical['target_name'] or not physical['entity']:
        raise ValueError('Missing actual articulation identity')
    return expected


def verify_articulation_seed(directory, source, result, prefix, seed):
    skills = json.loads(source['selected_segment']['semantic'])
    if len(skills) != 1 or skills[0]['verb'] not in VERBS:
        raise ValueError('Not a single supported articulation skill')
    path = directory / 'reference-binding.json'
    receipt = None
    if path.exists():
        receipt = json.loads(path.read_text())
        segment = source['selected_segment']
        if (receipt['schema'] != 'original_reference_articulation_binding_v1'
                or file_sha(path) != result.get('reference_binding_sha256')
                or receipt['source_proposal_sha256'] != result['proposal_sha256']
                or receipt['requested_category'] != skills[0]['target']
                or receipt['raw_reference_sha256'] != file_sha(directory / 'binding-reference.jsonl')
                or receipt['normalized_prefix_sha256'] != file_sha(directory / 'prefix.jsonl')):
            raise ValueError('Changed articulation reference binding')
        raw = [json.loads(line) for line in (directory / 'binding-reference.jsonl').read_text().splitlines()]
        resolver = ReferenceArticulationBinding(receipt['candidates'], receipt['requested_category'],
                                               segment['start'], segment['end'])
        normalized = []
        for t, original in enumerate(raw):
            found = resolver.observe(t, original['physical_before']['binding_candidates'],
                                     original['physical_audit']['binding_candidates'])
            if found is not None and (t != len(raw)-1 or found != receipt['selected_target']):
                raise ValueError('Different first physical target resolution')
            row = deepcopy(original)
            for field in ('physical_before', 'physical_audit'):
                row[field] = row[field]['binding_candidates'][receipt['selected_target']]
            row['outcome_after_control_candidate'] = 'SUCCEEDED' if found else 'UNKNOWN'
            normalized.append(row)
        skills[0]['target'] = receipt['selected_target']
        if (not raw or found != receipt['selected_target'] or normalized != prefix
                or receipt['resolved_skills'] != skills
                or receipt['selected_entity'] != receipt['candidates'][receipt['selected_target']]['entity']
                or receipt['evidence_available_control_step'] != len(prefix)):
            raise ValueError('Changed physical source/intent resolution')
    elif result.get('reference_binding_sha256'):
        raise ValueError('Missing declared articulation binding')
    want_open = skills[0]['verb'].startswith('OPEN_')
    for t,row in enumerate(prefix):
        legacy = ('simulator_apply_ack' not in row and result['source_commit'] in LEGACY_PREFIX_AFTER_APPLY)
        if (row['control_step'] != t or row['outcome_evidence_available_control_step'] != t+1
                or (row.get('simulator_apply_ack') is not True and not legacy)
                or (t and (prefix[t-1]['proprio_after'] != row['proprio_before']
                           or prefix[t-1]['physical_audit'] != row['physical_before']))):
            raise ValueError('No explicit ACK or verified legacy after-apply prefix chain')
    if (len(prefix) != seed['source_frame'] or len(prefix) < 12
            or prefix[-1]['physical_audit'] != seed['physical']
            or seed['physical']['target_name'] != skills[0]['target']):
        raise ValueError('Seed is not the exact verified reference endpoint')
    for row in prefix[-12:]:
        physical = row['physical_audit']
        if (physical_goal(physical, want_open) is not True
                or physical['target_name'] != seed['physical']['target_name']
                or physical['entity'] != seed['physical']['entity']):
            raise ValueError('Seed lacks twelve actual stable completion controls')
    return receipt


def validate_articulation_branch(rows, manifest, plans, seed):
    """Return labels at observations, including the separately bound old attempt."""
    if (not rows or len(rows) != manifest['controls'] or not plans
            or [r['control_step'] for r in rows] != list(range(len(rows)))
            or [p['control_step'] for p in plans] != sorted({p['control_step'] for p in plans})
            or plans[0]['control_step'] != 0):
        raise ValueError('Missing/duplicate/noncontiguous articulation controls or plans')
    skill = json.loads(plans[0]['active_skills_semantic_json'])
    if len(skill) != 1 or skill[0]['verb'] not in VERBS or manifest['verb'] != skill[0]['verb']:
        raise ValueError('Wrong articulation verb/bundle')
    want_open = skill[0]['verb'].startswith('OPEN_')
    state = CausalArticulation()
    state.seen_false = state.achieved = True
    state.true_streak = state.stable
    previous = 'SUCCEEDED'
    proposals, predecision = {}, {}
    events = {p['control_step']: p for p in plans}
    current = None
    for t, row in enumerate(rows):
        for key, width in (('proprio_before', 61), ('proprio_after', 61), ('action_executed_raw23', 23)):
            vector = np.asarray(row[key], dtype=np.float32)
            if vector.shape != (width,) or not np.isfinite(vector).all():
                raise ValueError('Invalid real articulation vector')
        if row['simulator_apply_ack'] is not True or row['label_kind'] not in (CLEAN, 'injected_fault_not_BC'):
            raise ValueError('Unknown/unapplied articulation command')
        phase = row['teacher']['phase']
        if (row['label_kind'] == 'injected_fault_not_BC') != (phase == 'fault_reverse'):
            raise ValueError('Reversed fault disguised as a teacher correction')
        if t and (rows[t-1]['proprio_after'] != row['proprio_before']
                  or rows[t-1]['physical_audit'] != row['physical_before']):
            raise ValueError('Broken physical articulation observation clock')
        if not t and row['physical_before'] != seed['physical']:
            raise ValueError('Branch does not start at its actual seed')
        retry = t in events and events[t]['decision'] == 'RETRY'
        if t in events:
            event = events[t]
            if (event['decision'] not in ('EXECUTE', 'RETRY')
                    or event['event_sha256'] != digest({k: v for k, v in event.items() if k != 'event_sha256'})
                    or json.loads(event['active_skills_semantic_json']) != skill):
                raise ValueError('Changed actual articulation command identity')
            if retry:
                if current is None or previous != 'FAILED':
                    raise ValueError('RETRY lacks a causally confirmed old-attempt failure')
                predecision[t] = dict(value=previous, context_id=current['event_sha256'],
                                      evidence_end_control_step=t)
            current = event
        if (row['context']['context_id'] != current['event_sha256']
                or row['context']['parent_goal'] != current['parent_goal']
                or row['context']['active_skills_semantic_json'] != current['active_skills_semantic_json']):
            raise ValueError('Cross-attempt/skill/task command')
        for key in ('physical_before', 'physical_audit'):
            physical = row[key]
            physical_goal(physical, want_open)
            if (physical['target_name'] != skill[0]['target']
                    or physical['entity'] != seed['physical']['entity']):
                raise ValueError('Wrong physical articulation target')
        proposals[t] = 'UNKNOWN' if retry else previous
        moved = phase == 'corrective' and progressing(row['physical_before']['directed_open_fraction'],
                                                      row['physical_audit']['directed_open_fraction'], want_open)
        previous = state.update(t, row['physical_audit']['goal_predicate'], retry=retry, moving=bool(moved))
    return proposals, predecision
