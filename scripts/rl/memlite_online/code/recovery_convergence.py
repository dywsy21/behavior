"""Event-balanced, sufficiently fitted observer diagnostic (not calibration)."""
from collections import Counter


def event_weights(rows, mechanisms=None):
    """One unit of CE mass per physical event, irrespective of review density."""
    if not rows:
        raise ValueError('Empty reviewed outcome pool')
    keys = [(r['candidate']['source_group'], r['approval']['event_id']) for r in rows]
    counts = Counter(keys)
    if mechanisms is not None:
        # Each mechanism and then each physical event gets equal CE mass.
        # Adding GRASP sources must not dilute the few articulation events.
        if len(mechanisms) != len(rows) or any(not isinstance(m, str) or not m for m in mechanisms):
            raise ValueError('Every approved member needs its actual issued mechanism')
        event_mechanisms = {}
        for key, mechanism in zip(keys, mechanisms):
            if key in event_mechanisms and event_mechanisms[key] != mechanism:
                raise ValueError('One outcome event has conflicting mechanism bindings')
            event_mechanisms[key] = mechanism
        events_by_mechanism = Counter(event_mechanisms.values())
        return [1. / (len(events_by_mechanism) * events_by_mechanism[mechanism] * counts[key])
                for key, mechanism in zip(keys, mechanisms)]
    return [1. / (len(counts) * counts[key]) for key in keys]


def check_splits(train, dev):
    a = {r['candidate']['source_group'] for r in train}
    b = {r['candidate']['source_group'] for r in dev}
    if not a or not b or a & b:
        raise ValueError('Fit/model-selection source groups overlap or are empty')
    for rows, split in ((train, 'train'), (dev, 'dev')):
        if any(r['candidate']['split'] != split for r in rows):
            raise ValueError('Wrong outcome split')
        if not {'IN_PROGRESS', 'SUCCEEDED', 'FAILED'} <= {r['approval']['label']['value'] for r in rows}:
            raise ValueError('Collect missing real classes; never invent labels')


def selection_key(metrics):
    # All classes matter. No tuning to aggregate accuracy dominated by success.
    return (metrics['balanced_accuracy'], -metrics['event_weighted_ce'])
