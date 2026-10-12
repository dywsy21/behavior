"""Actual fixed-period ANNOTATION command history, not actor rollout history.

Unlike thinning only refresh counters, this recomputes previous intent, parent
and the command-memory ledger at the same 128-control clock. Missing annotated
calls end a usable prefix: never invent a command across an unlabeled gap.
The returned targets cannot create result-head or low-action supervision.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def original_label_module():
    # This source is deliberately dependency-light. Import by pinned repo file
    # to avoid g05.data.__init__ eagerly importing unrelated training stacks.
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[4] / 'src/g05/data/memlite_stage1_labels.py'
    spec = importlib.util.spec_from_file_location('periodic_original_label_contract', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def periodic_annotation_trace(anchors, segments, *, period=128):
    if period != 128 or not anchors:
        raise ValueError('Expected a nonempty original trace and explicit deployed period128')
    lookup = {}
    previous_frame = -1
    for serial, row in enumerate(anchors):
        frame, segment, *_ = row
        if frame <= previous_frame or frame in lookup or not segments[segment]['start'] <= frame < segments[segment]['end']:
            raise ValueError('Reordered or unbound original annotation clock')
        lookup[frame] = (serial, segment); previous_frame = frame
    first, last = anchors[0][0], anchors[-1][0]
    previous, parent, history = 'None', 'None', []
    records, omitted_after_gap = [], 0
    gap = None
    for frame in range(first, last + 1, period):
        if frame not in lookup:
            gap = frame
            omitted_after_gap = sum(tick in lookup for tick in range(frame + period, last + 1, period))
            break
        serial, segment = lookup[frame]
        records.append(dict(original_anchor=serial, frame=frame, segment=segment,
                            previous_intent=previous, previous_parent=parent, history=list(history)))
        if previous != 'None' and (not history or history[-1] != previous):
            history.append(previous); history = history[-3:]
        previous, parent = segments[segment]['text'], segments[segment]['parent']
    return records, dict(first_frame=first, period=period, calls=len(records),
                         first_unannotated_scheduled_call=gap, omitted_known_calls_after_gap=omitted_after_gap)


def command_kind(record, segment):
    if record['previous_intent'] == 'None': return 'initial'
    return 'transition' if record['previous_intent'] != segment['text'] else 'continuation'


def command_change_class(previous, current):
    """Describe label changes, NEVER physical progress or outcome truth.

    The historical ``transition`` bucket is byte-level inequality. A GRASP
    source disappearing (floors -> NONE) is not a newly learned skill switch.
    Keep unknown relations explicit rather than guessing roles from entities.
    """
    from collections import Counter
    from g05.utils.memlite_skill_protocol import parse_active_skills_text
    right = parse_active_skills_text(current)
    if previous == 'None': return 'initial'
    left = parse_active_skills_text(previous)
    if previous == current: return 'identical_command'
    if Counter(s['verb'] for s in left) != Counter(s['verb'] for s in right):
        return 'explicit_verb_or_bundle_change'
    # Claim an explicit role change only when that role is known for every
    # member on both sides. Unknown -> known is improved description, not proof
    # that the robot changed the manipulated object or destination.
    for field in ('target', 'destination', 'target_part', 'arm'):
        if (all(s[field] not in ('', 'NONE', 'UNSPECIFIED') for s in left + right)
                and Counter((s['verb'], s[field]) for s in left)
                != Counter((s['verb'], s[field]) for s in right)):
            return 'explicit_bound_object_or_goal_change'
    if Counter(s['unbound_relation'] for s in left) != Counter(s['unbound_relation'] for s in right):
        return 'unbound_relation_change_not_verified_roles'
    return 'source_or_binding_description_change'


def unknown_feedback(record_index, records, segments):
    # Reuse exactly the already audited causal-counter serialization, but feed
    # it the reconstructed command sequence, not the original stride16 history.
    from recovery_expert_feedback import expert_unknown_feedback
    anchors = [(row['frame'], row['segment'], row['previous_intent'],
                row['previous_parent'], row['history']) for row in records]
    return expert_unknown_feedback(dict(anchors=anchors, segments=segments), record_index,
                                   repeat_stride_controls=128)


def validate_candidate_projection(row, episode, *, original_index, original_pair, expected_split='train'):
    """Recompute original labels; TRAIN remains the fail-closed default."""
    labels = original_label_module()
    source = episode['row']
    if expected_split not in ('train', 'eval') or episode['split'] != expected_split:
        raise ValueError('Periodic candidate crossed its explicit original split')
    serial, anchor_number = map(int, original_pair)
    anchors = labels.anchor_records(episode['segments'], episode['phase'])
    records, audit = periodic_annotation_trace(anchors, episode['segments'])
    found = [(number, record) for number, record in enumerate(records) if record['original_anchor'] == anchor_number]
    if len(found) != 1: raise ValueError('Candidate not on the known periodic command prefix')
    number, record = found[0]; segment = episode['segments'][record['segment']]
    identity = dict(episode_serial=serial, original_candidate_index=int(original_index),
        episode=int(source['episode_index']), task=int(source['task_index']),
        instance=int(source['task_instance_id']), raw_episode=int(source['raw_episode_id']),
        frame=record['frame'], phase=int(episode['phase']), split=expected_split, kind=command_kind(record, segment))
    target = labels.projection(segment, branch='high', task_name=episode['task_name'],
        previous_intent=record['previous_intent'], previous_parent=record['previous_parent'], history=record['history'])
    target['execution_feedback'] = unknown_feedback(number, records, episode['segments'])
    target['low_action_supervision_mask'] = False
    if (row['identity'] != identity or row['annotation_trace'] != audit or row['projection'] != target
            or row['original_stride16_previous_intent'] != anchors[anchor_number][2]
            or row['human_admission'] is not False
            or row['teacher_kind'] != 'annotation_periodic_command_trace_not_actor_rollout'):
        raise ValueError('Candidate identity, causal history, target or supervision differs from original source')
    return target
