"""Actual fixed-period ANNOTATION command history, not actor rollout history.

Unlike thinning only refresh counters, this recomputes previous intent, parent
and the command-memory ledger at the same 128-control clock. Missing annotated
calls end a usable prefix: never invent a command across an unlabeled gap.
The returned targets cannot create result-head or low-action supervision.
"""


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


def unknown_feedback(record_index, records, segments):
    # Reuse exactly the already audited causal-counter serialization, but feed
    # it the reconstructed command sequence, not the original stride16 history.
    from recovery_expert_feedback import expert_unknown_feedback
    anchors = [(row['frame'], row['segment'], row['previous_intent'],
                row['previous_parent'], row['history']) for row in records]
    return expert_unknown_feedback(dict(anchors=anchors, segments=segments), record_index,
                                   repeat_stride_controls=128)
