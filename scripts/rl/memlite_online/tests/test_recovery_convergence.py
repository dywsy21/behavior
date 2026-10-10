import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_convergence import event_weights, check_splits, selection_key, causal_suffix_training_items
from recovery_generation_metrics import score_event, summarize, feedback_probe, select_original_indices


def row(group, label, split='train'):
    return dict(candidate=dict(source_group=group,split=split),approval=dict(event_id='e',label=dict(value=label)))


class ConvergenceTests(unittest.TestCase):
    def test_noninitial_generation_schedule_cannot_silently_test_only_first_frames(self):
        fixed=[[0,1,2],[3,4,5]]
        previous=lambda i:'None' if i in (0,3,4) else 'a real previous command'
        task=lambda i:i//3
        self.assertEqual(select_original_indices(fixed,previous,task,'first_fixed_one_per_task_v1'),[0,3])
        self.assertEqual(select_original_indices(fixed,previous,task,'noninitial_fixed_one_per_task_v1'),[1,5])
        with self.assertRaisesRegex(ValueError,'No noninitial'):
            select_original_indices(fixed,lambda i:'None',task,'noninitial_fixed_one_per_task_v1')
        with self.assertRaisesRegex(ValueError,'task binding'):
            select_original_indices([[0,3],[4,5]],previous,task,'noninitial_fixed_one_per_task_v1')
    def test_suffixes_keep_current_truth_clock_and_event_mass(self):
        import torch
        items=[dict(context=torch.arange(12).reshape(3,4),proprio=torch.arange(81).reshape(3,27),
                    steps=torch.tensor([16,32,48])),
               dict(context=torch.ones(1,4),proprio=torch.ones(1,27),steps=torch.tensor([8]))]
        original=[{k:v.clone() for k,v in item.items()} for item in items]
        result,mass,indices=causal_suffix_training_items(items,[.6,.4])
        self.assertEqual(indices,[0,0,0,1]);self.assertEqual([len(x['steps']) for x in result],[1,2,3,1])
        self.assertAlmostEqual(sum(mass[:3]),.6);self.assertAlmostEqual(sum(mass),1.)
        for item,index in zip(result,indices):
            for key in item:self.assertTrue(torch.equal(item[key][-1],items[index][key][-1]))
            self.assertEqual(item['steps'][-1],items[index]['steps'][-1])
        for old,new in zip(original,items):
            for key in old:self.assertTrue(torch.equal(old[key],new[key]))
        with self.assertRaises(ValueError):causal_suffix_training_items(items,[1.])
        with self.assertRaises(ValueError):causal_suffix_training_items(items,[.6,0])

    def test_probe_is_explicit_and_never_adds_a_success_failure_claim(self):
        import json
        from g05.utils.memlite_skill_protocol import semantic_active_skills_text
        skill=dict(verb='GRASP',target='cup',arm='LEFT',source='',destination='',target_part='',unbound_relation='')
        original=dict(previous_intent=semantic_active_skills_text([skill]),known_previous_outcome='UNKNOWN',execution_feedback='none')
        changed,kind=feedback_probe(original,'original_heldout')
        self.assertEqual(kind,'synthetic_unknown_feedback_on_normal_state')
        j=json.loads(changed['execution_feedback'])
        self.assertEqual(j['estimated_bundle_outcome'],'UNKNOWN')
        self.assertEqual(len(j['estimated_member_outcomes']),1)
        self.assertEqual(original['execution_feedback'],'none')
        matched,_=feedback_probe(original,'original_heldout',unknown_confidence=.5)
        self.assertEqual(json.loads(matched['execution_feedback'])['estimated_member_outcomes'][0]['confidence'],.5)
        with self.assertRaises(ValueError):feedback_probe(original,'original_heldout',unknown_confidence=float('nan'))
        stripped,kind=feedback_probe(changed,'recovery_dev')
        self.assertEqual(stripped,original)
        with self.assertRaises(ValueError):feedback_probe(dict(original,known_previous_outcome='FAILED'),'recovery_dev')

    def test_repeated_frames_cannot_inflate_event_mass(self):
        rows=[row('a','FAILED')]*100+[row('b','SUCCEEDED')]
        w=event_weights(rows)
        self.assertAlmostEqual(sum(w[:100]),.5); self.assertEqual(w[-1],.5)
        self.assertAlmostEqual(sum(w),1.)

    def test_mechanism_balance_preserves_independent_event_mass(self):
        rows = [row('grasp-a','FAILED')]*12+[row('grasp-b','SUCCEEDED')]*3+[row('door','UNKNOWN')]*2
        w = event_weights(rows, ['GRASP']*15+['OPEN_DOOR']*2)
        self.assertAlmostEqual(sum(w[:12]), .25)
        self.assertAlmostEqual(sum(w[12:15]), .25)
        self.assertAlmostEqual(sum(w[15:]), .5)
        self.assertAlmostEqual(sum(w), 1.)
        with self.assertRaisesRegex(ValueError, 'conflicting mechanism'):
            event_weights([row('same','FAILED'),row('same','SUCCEEDED')], ['GRASP','OPEN_DOOR'])
        with self.assertRaises(ValueError):event_weights(rows, ['GRASP'])
        with self.assertRaises(ValueError):event_weights(rows, ['']*len(rows))

    def test_split_and_real_class_checks(self):
        a=[row('a',label) for label in ('FAILED','SUCCEEDED','IN_PROGRESS')]
        b=[row('b',label,'dev') for label in ('FAILED','SUCCEEDED','IN_PROGRESS')]
        check_splits(a,b)
        with self.assertRaises(ValueError):check_splits(a,[row('a',r['approval']['label']['value'],'dev') for r in b])
        with self.assertRaises(ValueError):check_splits(a,b[:2])

    def test_invalid_output_not_excluded_from_denominator(self):
        target=dict(next_decision='EXECUTE',active_skills_semantic_json='[]',target_parent_goal='p',memory_update='m')
        e=dict(validated=True,decision='RETRY',active_skills_semantic_json='[]',parent_goal='p',memory_update='m')
        result=summarize([dict(split='dev',score=score_event(None,target)),dict(split='dev',score=score_event(e,target))])
        self.assertEqual(result['dev']['valid'],.5);self.assertEqual(result['dev']['false_retry'],.5)
        self.assertEqual(result['dev']['exact'],0.)

    def test_selection_prefers_all_classes_before_ce(self):
        self.assertGreater(selection_key(dict(balanced_accuracy=.7,event_weighted_ce=1.)),
                           selection_key(dict(balanced_accuracy=.6,event_weighted_ce=.1)))


if __name__=='__main__':unittest.main()
