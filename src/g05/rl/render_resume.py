"""Fail-closed accounting for an interrupted render, not a generic crash waiver."""
from math import isfinite

from g05.rl.recovery import count_physical_steps
from semantic_robot.v2.render_batch import validate_batch, RenderCompletionMismatch

OLD_ERROR = "ValueError('Scheduled and completed render batches differ')"


def audit_render_failure(steps, io_rows, failure):
    count = count_physical_steps(steps, allowed_phases={'expert_prefix', 'policy'})
    controls = [r for r in io_rows if r['kind'] == 'control']
    if not count or len(controls) != count or failure.get('controls') != count:
        raise ValueError('Missing physical action or I/O evidence')
    for step, row in zip(steps, controls):
        action = step['action']
        if (len(action) != 23 or not all(isfinite(v) for v in action)
                or row.get('completed') is not True or row.get('error')
                or row['episode'] != step['episode'] or row['call'] != step['episode_control']
                or not all(isfinite(v) for key in ('before','after') for v in row[key].values())
                or row['after']['physics_index']-row['before']['physics_index'] != 4
                or abs(row['after']['simulation_time']-row['before']['simulation_time']-1/30) > 1e-7):
            raise ValueError('Invalid or unmatched physical control')
    capture, close = io_rows[-2:]
    clock = controls[-1]['after']
    if (failure.get('error') != OLD_ERROR or failure['episode'] != steps[-1]['episode']
            or capture['kind'] != 'capture' or capture['completed'] is not False
            or capture.get('error') != OLD_ERROR or capture['episode'] != failure['episode']
            or capture['before'] != clock or capture['after'] != clock
            or close['kind'] != 'close' or close['completed'] is not True or close.get('error')
            or close['episode'] != failure['episode'] or close['before'] != clock or close['after'] != clock):
        raise ValueError('Not the recorded zero-physics render failure and clean native detach')
    try:
        validate_batch(capture['batch'])
    except RenderCompletionMismatch:
        pass
    else:
        raise ValueError('Expected exactly a native completion mismatch')
    return dict(controls=count, policy_controls=sum(r['phase']=='policy' for r in steps),
                incomplete_episode=failure['episode'],
                incomplete_controls=sum(r['episode']==failure['episode'] for r in steps),
                incomplete_policy_controls=sum(r['episode']==failure['episode'] and r['phase']=='policy'
                                               for r in steps))


def reconcile_render_stop(closed, counts, unresolved, status, inherited_controls):
    """Caller first audits all controls/native detach and verifies all PIDs dead."""
    if (len(counts) != 2 or any(type(n) is not int or n <= 0 for n in counts)
            or closed.get('exits') != [0,0] or closed.get('reported_controls') != []
            or closed.get('clean') is not False or closed.get('pending_controls') != 32
            or unresolved != dict(pending={'0':16,'1':16},budget_pending=32)
            or status.get('phase') != 'failed' or status.get('pending_controls') != 32
            or type(inherited_controls) is not int or inherited_controls < 0
            or closed['ledger_controls']+32 != sum(counts)
            or status['controls']+32 != inherited_controls+sum(counts)):
        raise ValueError('Unreconciled render-failure physical budget')
    return dict(controls=inherited_controls+sum(counts), reconciled_pending_controls=32,
                original_close_clean=False, worker_controls=counts,
                evidence='sequential actions and four-tick I/O; native detach at same clock; both exits0')
