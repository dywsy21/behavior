import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_convergence import event_weights, check_splits, selection_key
from recovery_generation_metrics import score_event, summarize, feedback_probe


def row(group, label, split='train'):
    return dict(candidate=dict(source_group=group,split=split),approval=dict(event_id='e',label=dict(value=label)))


class ConvergenceTests(unittest.TestCase):
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
        stripped,kind=feedback_probe(changed,'recovery_dev')
        self.assertEqual(stripped,original)
        with self.assertRaises(ValueError):feedback_probe(dict(original,known_previous_outcome='FAILED'),'recovery_dev')

    def test_repeated_frames_cannot_inflate_event_mass(self):
        rows=[row('a','FAILED')]*100+[row('b','SUCCEEDED')]
        w=event_weights(rows)
        self.assertAlmostEqual(sum(w[:100]),.5); self.assertEqual(w[-1],.5)
        self.assertAlmostEqual(sum(w),1.)

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
