"""Deterministic phase-spread ACTION review candidates, never approvals.

One initial 32-control window does not supervise a long recovery. Select
three later windows from the SAME successful correction for human review,
without counting them as three new independent events or changing splits.
Privileged completion selects offline review material only, never actor input.
"""


def later_action_windows(rows, available, *, retry_control, first_success, clean_label):
    if (type(retry_control) is not int or type(first_success) is not int
            or not 0<=retry_control<first_success<len(rows)
            or not isinstance(clean_label,str) or not clean_label):
        raise ValueError('Actual corrective start and later physical success required')
    available=sorted(set(available))
    if any(type(t) is not int or not 0<=t<len(rows) for t in available):
        raise ValueError('Recorded observation clocks required')
    context=rows[retry_control]['context']
    eligible=[]
    for t in available:
        if t<retry_control+32 or t+32>=len(rows):continue
        chunk=rows[t:t+32]
        if any(r['label_kind']!=clean_label or r['simulator_apply_ack'] is not True
               or r['control_step']!=t+i or r['terminated'] is not False or r['truncated'] is not False
               or r['context']!=context for i,r in enumerate(chunk)):
            continue
        endpoints=[s for s in available if s>=t+32]
        if endpoints:eligible.append(t)
    duration=first_success-retry_control
    aims=[retry_control+duration/3,retry_control+duration*2/3,first_success-16]
    selected=[]
    for desired in aims:
        # Require disjoint action targets; nearby additional RGB are only
        # bracketing evidence, not duplicated or independent training events.
        choices=[t for t in eligible if all(abs(t-old)>=32 for old in selected)]
        if not choices:continue
        selected.append(min(choices,key=lambda t:(abs(t-desired),t)))
    if not selected:raise ValueError('No later complete clean same-command window')
    frames=set()
    for t in selected:
        frames.update(s for s in available if t<=s<=t+32)
        frames.add(min(s for s in available if s>=t+32))
    return dict(candidate_action_steps=sorted(selected),frames=sorted(frames),
        independent_new_events=0,existing_event_must_be_reused=True,human_approved=False)
