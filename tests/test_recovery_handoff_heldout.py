import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_handoff_feedback import requests_for_candidate, requests_for_eval_candidate, EVAL_SCHEMA
from recovery_handoff_heldout import select_heldout_requests, selection_bucket, generation_split
from recovery_periodic_teacher import validate_candidate_projection
import test_recovery_handoff_feedback as fixture
import test_recovery_periodic_teacher as original_fixture


class HeldoutTests(unittest.TestCase):
    def example(self, instance=1, index=108):
        row, frames = fixture.HandoffTests().example()
        row['identity'].update(split='eval', instance=instance, original_candidate_index=index)
        frames = {k:v+index-108 for k,v in frames.items()}
        request, reason = requests_for_eval_candidate(row, frames, excluded_groups=set())
        self.assertIsNone(reason)
        return row, request, frames

    def test_explicit_namespaces_do_not_admit_heldout_to_train(self):
        original, request, frames = self.example()
        self.assertEqual(request['schema'], EVAL_SCHEMA)
        with self.assertRaises(ValueError):
            requests_for_candidate(original, frames, excluded_groups=set())
        original['identity']['split'] = 'train'
        with self.assertRaises(ValueError):
            requests_for_eval_candidate(original, frames, excluded_groups=set())

    def test_projection_requires_true_episode_split_not_a_fake_row_label(self):
        row, episode = original_fixture.PeriodicTeacherTests().candidate()
        row['identity']['split'] = 'eval'
        with self.assertRaises(ValueError):
            validate_candidate_projection(row, episode, original_index=100, original_pair=(7,8), expected_split='eval')
        episode['split'] = 'eval'
        self.assertEqual(validate_candidate_projection(row, episode, original_index=100,
            original_pair=(7,8), expected_split='eval'), row['projection'])
        with self.assertRaises(ValueError):
            validate_candidate_projection(row, episode, original_index=100, original_pair=(7,8))

    def test_selection_is_order_invariant_and_source_unique_within_bucket(self):
        pairs = [self.example(instance, index)[:2] for instance in (1,2,3,4) for index in (108,109)]
        selected, counts = select_heldout_requests(pairs, excluded_groups=set())
        reversed_result, reversed_counts = select_heldout_requests(reversed(pairs), excluded_groups=set())
        self.assertEqual(selected, reversed_result)
        self.assertEqual(counts, reversed_counts)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({r['source_group'] for _,r in selected}), 2)
        self.assertEqual(counts['explicit_verb_or_bundle_change'], 8)

    def test_protected_source_duplicate_or_prediction_never_selected(self):
        original, request, _ = self.example()
        with self.assertRaises(ValueError):
            select_heldout_requests([(original,request)], excluded_groups={request['source_group']})
        with self.assertRaises(ValueError):
            select_heldout_requests([(original,request)]*2, excluded_groups=set())
        for key, value in [('input_feedback_generated',True),('physical_outcome_asserted',True),
                           ('optimizer_updates',1),('schema','train'),('sample_id','foreign')]:
            changed=copy.deepcopy(request); changed[key]=value
            with self.assertRaises(ValueError):
                select_heldout_requests([(original,changed)], excluded_groups=set())

    def test_unknown_relation_is_not_scored_as_verified_handoff(self):
        self.assertEqual(selection_bucket('identical_command'), 'continuation')
        self.assertEqual(selection_bucket('explicit_bound_object_or_goal_change'), 'handoff')
        self.assertIsNone(selection_bucket('source_or_binding_description_change'))
        self.assertIsNone(selection_bucket('unbound_relation_change_not_verified_roles'))

    def test_generation_has_no_implicit_or_relabelled_split(self):
        train=dict(schema='periodic_handoff_observer_generation_v1',human_review_required_before_fit=True)
        heldout=dict(schema='periodic_handoff_observer_eval_generation_v1',purpose='heldout_eval_only',
            training_admission_forbidden=True,human_review_required_before_evaluation=True)
        self.assertEqual(generation_split(train),'train')
        self.assertEqual(generation_split(heldout),'eval')
        for key in ('purpose','training_admission_forbidden','human_review_required_before_evaluation'):
            bad=copy.deepcopy(heldout); bad.pop(key)
            with self.assertRaises(ValueError): generation_split(bad)
        with self.assertRaises(ValueError): generation_split(dict(train,purpose='heldout_eval_only'))


if __name__ == '__main__': unittest.main()
