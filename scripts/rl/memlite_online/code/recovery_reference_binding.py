"""Resolve a category annotation by real original-reference contact, not a guess.

The selected object is label-side evidence only until the independently reset
recovery branch starts. Never pick the nearest object, a numeric suffix, or the
first object in a scene. All competitors and the exact original control clock
must be recorded. Ambiguity is a hard rejection.
"""


class ReferenceGraspBinding:
    def __init__(self, candidates, requested_category, requested_arm, start, end):
        if (not candidates or len(set(candidates)) != len(candidates)
                or any(v['category'] != requested_category for v in candidates.values())):
            raise ValueError('Require actual exact-category candidate inventory')
        if requested_arm not in ('LEFT', 'RIGHT', 'UNSPECIFIED') or not 0 <= start < end:
            raise ValueError('Invalid annotated arm or interval')
        self.candidates = candidates
        self.arms = [requested_arm.lower()] if requested_arm != 'UNSPECIFIED' else ['left', 'right']
        self.start, self.end, self.last = start, end, -1
        self.streak = {(n, a): 0 for n in candidates for a in self.arms}
        self.seen_false = {key: False for key in self.streak}

    def observe(self, step, physical_candidates):
        if type(step) is not int or step != self.last + 1:
            raise ValueError('Reference binding needs a consecutive actual control clock')
        self.last = step
        if set(physical_candidates) != set(self.candidates):
            raise ValueError('Missing/added competitor in reference binding evidence')
        for name, value in physical_candidates.items():
            if (value['target_name'] != name or value['entity'] != self.candidates[name]['entity']
                    or set(value['grasp']) != {'left', 'right'}
                    or any(v not in ('TRUE', 'FALSE', 'UNKNOWN') for v in value['grasp'].values())):
                raise ValueError('Changed physical identity or invalid measured grasp')
        if not self.start <= step < self.end:
            for key in self.streak: self.streak[key] = 0
            return None
        if any(physical_candidates[n]['grasp'][a] == 'UNKNOWN' for n in self.candidates for a in self.arms):
            for key in self.streak: self.streak[key] = 0
            return None
        for key in self.streak:
            name, arm = key
            value = physical_candidates[name]['grasp'][arm]
            self.seen_false[key] |= value == 'FALSE'
            self.streak[key] = self.streak[key] + 1 if value == 'TRUE' else 0
        # A competitor held right now blocks choosing another, even before its
        # own debounce completes. A double-handed contact is not silently split.
        held = [(n, a) for n in self.candidates for a in self.arms
                if physical_candidates[n]['grasp'][a] == 'TRUE']
        stable = [k for k in held if self.streak[k] >= 6 and self.seen_false[k]]
        if stable and len(held) != 1:
            raise ValueError('Ambiguous original-reference grasp binding')
        return stable[0] if stable else None


class ReferenceArticulationBinding:
    """Unique, observed goal transition AND unique moving exact-category joint.

    All same-category competitors must have unambiguous physical contracts.
    This is reference annotation resolution, not an online actor oracle.
    """
    def __init__(self, candidates, category, start, end):
        from recovery_articulation_teacher import CausalArticulation
        if (not candidates or any(v['category'] != category for v in candidates.values())
                or not 0 <= start < end):
            raise ValueError('Invalid exact-category articulation inventory')
        self.candidates, self.start, self.end = candidates, start, end
        self.states = {n: CausalArticulation() for n in candidates}
        self.initial_fraction = {}; self.moved = set()

    def observe(self, step, before, after):
        if set(before) != set(self.candidates) or set(after) != set(self.candidates):
            raise ValueError('Missing articulation competitor')
        import math
        achieved = []
        inside = self.start <= step < self.end
        for name, info in self.candidates.items():
            first, second = before[name], after[name]
            for value in (first, second):
                if (value['target_name'] != name or value['entity'] != info['entity']
                        or not math.isfinite(value['directed_open_fraction'])):
                    raise ValueError('Changed articulation binding identity or measurement')
            if inside:
                initial = self.initial_fraction.setdefault(name, first['directed_open_fraction'])
                if abs(second['directed_open_fraction'] - initial) >= .035: self.moved.add(name)
            outcome = self.states[name].update(step, second['goal_predicate'] if inside else None)
            if outcome == 'SUCCEEDED': achieved.append(name)
        if achieved:
            if len(achieved) != 1 or self.moved != set(achieved):
                raise ValueError('Multiple moving/achieved reference articulation candidates')
            return achieved[0]
        return None


def resolve_prefix_rows(rows, candidates, category, arm, start, end, expected_name, expected_arm):
    """Recheck ALL real competitor observations before normalizing a prefix."""
    from copy import deepcopy
    state = ReferenceGraspBinding(candidates, category, arm, start, end)
    normalized = []
    for i, row in enumerate(rows):
        if row['control_step'] != i:
            raise ValueError('Missing original prefix control')
        chosen = state.observe(i, row['physical_audit']['binding_candidates'])
        if chosen is not None and (i != len(rows)-1 or chosen != (expected_name, expected_arm)):
            raise ValueError('Changed first causal reference binding')
        new = deepcopy(row)
        new['physical_audit'] = new['physical_audit']['binding_candidates'][expected_name]
        normalized.append(new)
    if not rows or chosen != (expected_name, expected_arm):
        raise ValueError('No independently reproducible physical binding')
    return normalized


def verify_saved_binding(directory, source, result, normalized):
    """Offline corpus gate; rederive instead of trusting a resolver receipt."""
    import json
    from recovery_corpus import file_sha
    path = directory / 'reference-binding.json'
    if not path.exists():
        if result.get('reference_binding_sha256'):
            raise ValueError('Missing declared reference binding')
        return None
    receipt = json.loads(path.read_text())
    segment = source['selected_segment']
    original = json.loads(segment['semantic'])
    if (receipt['schema'] != 'original_reference_grasp_binding_v1'
            or file_sha(path) != result.get('reference_binding_sha256')
            or receipt['source_proposal_sha256'] != result['proposal_sha256']
            or receipt['requested_category'] != original[0]['target']
            or receipt['requested_arm'] != original[0].get('arm', 'UNSPECIFIED').upper()
            or receipt['raw_reference_sha256'] != file_sha(directory / 'binding-reference.jsonl')
            or receipt['normalized_prefix_sha256'] != file_sha(directory / 'prefix.jsonl')):
        raise ValueError('Unbound reference-derived target')
    raw = [json.loads(line) for line in (directory / 'binding-reference.jsonl').read_text().splitlines()]
    expected = resolve_prefix_rows(raw, receipt['candidates'], receipt['requested_category'],
                                  receipt['requested_arm'], segment['start'], segment['end'],
                                  receipt['selected_target'], receipt['selected_arm'])
    original[0]['target'] = receipt['selected_target']
    if (expected != normalized or receipt['resolved_skills'] != original
            or receipt['evidence_available_control_step'] != len(expected)
            or receipt['selected_entity'] != receipt['candidates'][receipt['selected_target']]['entity']):
        raise ValueError('Reference binding changed actual observations, intent or identity')
    return receipt
