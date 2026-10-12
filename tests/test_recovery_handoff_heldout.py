import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_handoff_feedback import requests_for_candidate, requests_for_eval_candidate, EVAL_SCHEMA
from recovery_handoff_heldout import (select_heldout_requests, selection_bucket, generation_split,
    replay_heldout_projection, validate_fixed_fit)
from recovery_corpus import digest
from recovery_handoff_inference import feedback_from_trace
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

    def estimate(self):
        original, request, _ = self.example()
        _,ep=original_fixture.PeriodicTeacherTests().candidate(); ep['split']='eval'
        cal=dict(temperature=1.774,high_sha256='a'*64,observer_sha256='b'*64)
        trace=[dict(r,logits=[0.,20.,0.,0.]) for r in request['requests']]
        p=request['original_projection']
        predicted=dict(sample_id=request['sample_id'],source_group=request['source_group'],
            original_identity=request['original_identity'],target_projection_sha256=digest(p),
            original_execution_feedback=p['execution_feedback'],known_previous_outcome='UNKNOWN',
            physical_outcome_asserted=False,new_human_admission=False,
            execution_feedback=feedback_from_trace(request,ep,trace,cal))
        return request,predicted,ep,trace,cal

    def test_real_estimate_replay_preserves_all_targets_and_unknown_truth(self):
        args=self.estimate(); before=copy.deepcopy(args)
        out=replay_heldout_projection(*args)
        for key,value in args[0]['original_projection'].items():
            if key!='execution_feedback': self.assertEqual(out[key],value)
        self.assertEqual(args,before)
        self.assertEqual(out['known_previous_outcome'],'UNKNOWN')
        self.assertFalse(out['outcome_supervision_mask'])

    def test_logits_feedback_source_and_split_tampering_are_rejected(self):
        good=self.estimate()
        for key,value in [('sample_id','foreign'),('new_human_admission',True),
                           ('physical_outcome_asserted',True),('execution_feedback','none')]:
            args=copy.deepcopy(good); args[1][key]=value
            with self.assertRaises(ValueError): replay_heldout_projection(*args)
        args=copy.deepcopy(good); args[3][0]['logits']=[0.,0.,20.,0.]
        with self.assertRaises(ValueError): replay_heldout_projection(*args)
        args=copy.deepcopy(good); args[2]['split']='train'
        with self.assertRaises(ValueError): replay_heldout_projection(*args)

    def test_fixed_final_never_accepts_paused_or_best_intermediate_fit(self):
        result=dict(status='completed_finite_schedule',engineering_only=False,step=360,planned_updates=360,
            frozen_before_sha256='f'*64,frozen_after_sha256='f'*64,
            parent_sha256='a74c5cc147dcd632dfddcef780f2268845b13c9c4169f61d2aa7f90a875d5bf2')
        latest=dict(state=dict(step=360),sha256='c'*64,path='final.pt')
        model=dict(path='run/checkpoints/final.pt',sha256='c'*64)
        validate_fixed_fit(result,latest,model)
        for key,value in [('status','saved_paused'),('step',240),('engineering_only',True),
                           ('frozen_after_sha256','other'),('parent_sha256','other')]:
            with self.assertRaises(ValueError): validate_fixed_fit(dict(result,**{key:value}),latest,model)
        with self.assertRaises(ValueError): validate_fixed_fit(result,latest,dict(model,sha256='other'))


if __name__ == '__main__': unittest.main()
