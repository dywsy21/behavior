"""Select calibration anchors from labels/identities BEFORE model predictions.

One source group contributes exactly one event to existing strict calibration.
Balanced phase selection is explicit; it does not estimate deployment priors.
No logit, probability, confidence, checkpoint score or optimizer enters here.
"""
from collections import Counter

from recovery_corpus import digest

CLASSES = ('IN_PROGRESS', 'SUCCEEDED', 'FAILED')


def preselect_anchors(rows, expected_groups, *, per_class=20, seed=17):
    if (type(per_class) is not int or per_class < 20 or len(expected_groups) != len(set(expected_groups))
            or len(expected_groups) != 3*per_class):
        raise ValueError('Need exactly three independent class quotas without dropping sources')
    by_group = {}
    for row in rows:
        c, a = row['candidate'], row['approval']
        if c['split'] != 'dev' or a.get('usage_role') != 'calibration' or a['pool'] != 'outcome':
            raise ValueError('Only independently approved calibration observations')
        group, label = c['source_group'], a['label']['value']
        if label not in (*CLASSES, 'UNKNOWN'):
            raise ValueError('Unrecognized result label')
        by_group.setdefault(group, {}).setdefault(label, []).append(row)
    if set(by_group) != set(expected_groups):
        raise ValueError('Missing/added source; preserve calibration denominator')
    # Deterministic bipartite matching: a source may support several phases
    # but can occupy ONLY ONE slot. A failed match is not repaired by copying
    # frames, selecting model-correct samples, or stealing test groups.
    slots = [(label, i) for label in CLASSES for i in range(per_class)]
    candidates = {label: sorted((g for g,v in by_group.items() if label in v),
        key=lambda g:digest(['calibration-phase', seed, label, g])) for label in CLASSES}
    matched = {}
    def assign(slot, visited):
        for group in candidates[slot[0]]:
            if group in visited:
                continue
            visited.add(group)
            if group not in matched or assign(matched[group], visited):
                matched[group] = slot
                return True
        return False
    for slot in slots:
        if not assign(slot, set()):
            raise ValueError('Insufficient independent phase support; do not fabricate labels')
    selected = []
    for group, (label, _) in sorted(matched.items()):
        row = min(by_group[group][label], key=lambda r:digest([
            'calibration-anchor', seed, r['candidate']['sample_id'], r['approval']['label']['member_index']]))
        selected.append(dict(source_group=group, sample_id=row['candidate']['sample_id'],
            value=label, member_index=row['approval']['label']['member_index'],
            history_role=row['approval']['label'].get('history_role','observable')))
    if set(matched) != set(expected_groups) or Counter(s['value'] for s in selected) != Counter({c:per_class for c in CLASSES}):
        raise AssertionError('Calibration matching lost unique source/class quota')
    return selected


def selected_rows(rows, manifest):
    if (manifest.get('schema') != 'independent_calibration_anchor_selection_v1'
            or manifest.get('declared_before_model_predictions') is not True):
        raise ValueError('Need predeclared calibration selection')
    expected = preselect_anchors(rows, manifest['expected_groups'],
        per_class=manifest['per_class'], seed=manifest['seed'])
    if expected != manifest['selected_anchors']:
        raise ValueError('Selected anchors or actual phase support changed')
    by_key = {(r['candidate']['sample_id'], r['approval']['label']['member_index'],
               r['approval']['label'].get('history_role','observable')):r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError('Duplicate calibration observation approval')
    return [by_key[(e['sample_id'],e['member_index'],e['history_role'])] for e in expected]
