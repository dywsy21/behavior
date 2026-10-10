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
    if manifest.get('schema') == 'independent_calibration_pool_selection_v2':
        if manifest.get('declared_before_model_predictions') is not True:
            raise ValueError('Pool selection must precede model predictions')
        expected = preselect_pool_anchors(rows, manifest['declared_groups'],
            manifest['unavailable_sources'], per_class=manifest['per_class'], seed=manifest['seed'])
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Pool denominator, availability, reserved groups or anchors changed')
        anchors = manifest['selected_anchors']
    else:
        if (manifest.get('schema') != 'independent_calibration_anchor_selection_v1'
                or manifest.get('declared_before_model_predictions') is not True):
            raise ValueError('Need predeclared calibration selection')
        expected = preselect_anchors(rows, manifest['expected_groups'],
            per_class=manifest['per_class'], seed=manifest['seed'])
        if expected != manifest['selected_anchors']:
            raise ValueError('Selected anchors or actual phase support changed')
        anchors = expected
    by_key = {(r['candidate']['sample_id'], r['approval']['label']['member_index'],
               r['approval']['label'].get('history_role','observable')):r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError('Duplicate calibration observation approval')
    return [by_key[(e['sample_id'],e['member_index'],e['history_role'])] for e in anchors]


def preselect_pool_anchors(rows, declared_groups, unavailable_sources, *, per_class=30, seed=17):
    """Outcome-blind sampling from a fully accounted larger prospective pool.

    This is NOT v1's all-source certificate. Physical-data availability and
    conditional observer accuracy have different denominators. Unavailable
    sources need pinned reasons; surplus reviewed sources remain reserved.
    Labels determine phase support, but predictions never enter selection.
    """
    if (type(per_class) is not int or per_class < 30 or type(seed) is not int
            or not isinstance(declared_groups,list) or len(declared_groups)!=len(set(declared_groups))
            or any(not isinstance(g,str) or not g for g in declared_groups)):
        raise ValueError('Explicit unique pool and at least the original30-per-phase target required')
    unavailable={}
    for item in unavailable_sources:
        if (set(item)!={'source_group','reason','receipt_sha256'}
                or item['source_group'] not in declared_groups or item['source_group'] in unavailable
                or not isinstance(item['reason'],str) or not item['reason'].strip()
                or not isinstance(item['receipt_sha256'],str) or len(item['receipt_sha256'])!=64
                or any(c not in '0123456789abcdef' for c in item['receipt_sha256'])):
            raise ValueError('Every unavailable source needs its unique pinned terminal/review receipt')
        unavailable[item['source_group']]=dict(item)
    by_group={};identities=set()
    for row in rows:
        c,a=row['candidate'],row['approval'];group=c['source_group'];label=a['label']['value']
        if (c['split']!='dev' or a.get('usage_role')!='calibration' or a['pool']!='outcome'
                or group not in declared_groups or group in unavailable
                or label not in (*CLASSES,'UNKNOWN')):
            raise ValueError('Undeclared, unavailable or non-calibration source in reviewed pool')
        key=(c['sample_id'],a['label']['member_index'],a['label'].get('history_role','observable'))
        if key in identities:raise ValueError('Duplicate approved phase observation')
        identities.add(key);by_group.setdefault(group,{}).setdefault(label,[]).append(row)
    if set(by_group)|set(unavailable)!=set(declared_groups):
        raise ValueError('Unreviewed/missing source must not vanish from the declared pool')
    candidates={label:sorted((g for g,v in by_group.items() if label in v),
        key=lambda g:digest(['calibration-phase',seed,label,g])) for label in CLASSES}
    matched={}
    def assign(slot,visited):
        for group in candidates[slot[0]]:
            if group in visited:continue
            visited.add(group)
            if group not in matched or assign(matched[group],visited):
                matched[group]=slot;return True
        return False
    for slot in ((label,i) for label in CLASSES for i in range(per_class)):
        if not assign(slot,set()):
            raise ValueError('Insufficient unique manually supported phases; no threshold or quota reduction')
    anchors=[]
    for group,(label,_) in sorted(matched.items()):
        row=min(by_group[group][label],key=lambda r:digest([
            'calibration-anchor',seed,r['candidate']['sample_id'],r['approval']['label']['member_index']]))
        anchors.append(dict(source_group=group,sample_id=row['candidate']['sample_id'],value=label,
            member_index=row['approval']['label']['member_index'],
            history_role=row['approval']['label'].get('history_role','observable')))
    # Reuse v1's strict exact-source check on the selected subcohort, while
    # retaining the full pool and reserved identities explicitly outside it.
    preselect_anchors([r for r in rows if r['candidate']['source_group'] in matched],
        sorted(matched),per_class=per_class,seed=seed)
    return dict(declared_groups=sorted(declared_groups),unavailable_sources=[unavailable[g] for g in sorted(unavailable)],
        selected_anchors=anchors,selected_groups=sorted(matched),reserved_groups=sorted(set(by_group)-set(matched)),
        declared_source_count=len(declared_groups),reviewed_source_count=len(by_group),
        unavailable_source_count=len(unavailable),calibration_source_count=len(matched),
        accuracy_conditioned_on_manually_supported_sources=True,certifies_all_declared_sources=False)
