from copy import deepcopy
from pathlib import Path
import sys
import unittest
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_reference_binding import ReferenceGraspBinding, resolve_prefix_rows, verify_saved_binding
from recovery_corpus import file_sha


def inventory():
    return {n: dict(entity='cup.n.01_'+str(i), category='cup')
            for i, n in enumerate(('cup_20', 'cup_3'))}


def physical(held=None):
    return {n: dict(target_name=n, entity=v['entity'],
                   grasp=dict(left='FALSE', right='TRUE' if held == n else 'FALSE'))
            for n, v in inventory().items()}


class BindingTests(unittest.TestCase):
    def test_no_category_or_nearest_guess_six_real_controls(self):
        state = ReferenceGraspBinding(inventory(), 'cup', 'RIGHT', 0, 20)
        self.assertIsNone(state.observe(0, physical()))
        for i in range(1, 6): self.assertIsNone(state.observe(i, physical('cup_20')))
        self.assertEqual(state.observe(6, physical('cup_20')), ('cup_20', 'right'))

    def test_wrong_arm_preexisting_hold_outside_interval_and_unknown_do_not_bind(self):
        for arm, start, end in [('LEFT', 0, 20), ('RIGHT', 0, 4), ('RIGHT', 10, 20)]:
            state = ReferenceGraspBinding(inventory(), 'cup', arm, start, end)
            for i in range(9): self.assertIsNone(state.observe(i, physical('cup_20')))
        state = ReferenceGraspBinding(inventory(), 'cup', 'RIGHT', 0, 20)
        state.observe(0, physical())
        for i in range(1, 6): state.observe(i, physical('cup_20'))
        ambiguous = physical('cup_20'); ambiguous['cup_3']['grasp']['right'] = 'TRUE'
        with self.assertRaises(ValueError): state.observe(6, ambiguous)

    def test_changed_identity_missing_competitor_clock_and_category_rejected(self):
        with self.assertRaises(ValueError): ReferenceGraspBinding(inventory(), 'bowl', 'RIGHT', 0, 20)
        for change in ('identity', 'competitor', 'clock'):
            state = ReferenceGraspBinding(inventory(), 'cup', 'RIGHT', 0, 20)
            evidence = physical()
            if change == 'identity': evidence['cup_20']['entity'] = 'other'
            if change == 'competitor': del evidence['cup_3']
            with self.assertRaises(ValueError): state.observe(1 if change == 'clock' else 0, evidence)

    def test_unknown_competitor_breaks_contact_debounce(self):
        state = ReferenceGraspBinding(inventory(), 'cup', 'RIGHT', 0, 30)
        state.observe(0, physical())
        for i in range(1, 6): state.observe(i, physical('cup_20'))
        evidence = physical('cup_20'); evidence['cup_3']['grasp']['right'] = 'UNKNOWN'
        self.assertIsNone(state.observe(6, evidence))
        for i in range(7, 12): self.assertIsNone(state.observe(i, physical('cup_20')))
        self.assertEqual(state.observe(12, physical('cup_20')), ('cup_20', 'right'))

    def test_immutable_raw_competitor_evidence_reproduces_binding(self):
        rows = [dict(control_step=i, physical_audit=dict(binding_candidates=physical('cup_20' if i else None)))
                for i in range(7)]
        before = deepcopy(rows)
        resolved = resolve_prefix_rows(rows, inventory(), 'cup', 'RIGHT', 0, 20, 'cup_20', 'right')
        self.assertEqual(rows, before)
        self.assertEqual(resolved[-1]['physical_audit']['target_name'], 'cup_20')
        with self.assertRaises(ValueError): resolve_prefix_rows(rows, inventory(), 'cup', 'RIGHT', 0, 20, 'cup_3', 'right')

    def test_saved_binding_rechecked_without_changing_original_annotation(self):
        rows = [dict(control_step=i, physical_audit=dict(binding_candidates=physical('cup_20' if i else None)))
                for i in range(7)]
        normalized = resolve_prefix_rows(rows, inventory(), 'cup', 'RIGHT', 0, 20, 'cup_20', 'right')
        original = [dict(verb='GRASP', target='cup', arm='RIGHT')]
        source = dict(selected_segment=dict(semantic=json.dumps(original), start=0, end=20))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, content in [('binding-reference.jsonl', rows), ('prefix.jsonl', normalized)]:
                (root/name).write_text(''.join(json.dumps(row)+'\n' for row in content))
            receipt = dict(schema='original_reference_grasp_binding_v1', requested_category='cup', requested_arm='RIGHT',
                candidates=inventory(), selected_target='cup_20', selected_entity='cup.n.01_0', selected_arm='right',
                evidence_available_control_step=7, source_proposal_sha256='a'*64,
                raw_reference_sha256=file_sha(root/'binding-reference.jsonl'), normalized_prefix_sha256=file_sha(root/'prefix.jsonl'),
                resolved_skills=[dict(verb='GRASP', target='cup_20', arm='RIGHT')])
            (root/'reference-binding.json').write_text(json.dumps(receipt))
            result = dict(reference_binding_sha256=file_sha(root/'reference-binding.json'), proposal_sha256='a'*64)
            self.assertEqual(verify_saved_binding(root, source, result, normalized), receipt)
            changed = deepcopy(normalized); changed[-1]['physical_audit']['entity'] = 'wrong'
            with self.assertRaises(ValueError): verify_saved_binding(root, source, result, changed)
            self.assertEqual(json.loads(source['selected_segment']['semantic'])[0]['target'], 'cup')


if __name__ == '__main__': unittest.main()
