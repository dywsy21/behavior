import json
from pathlib import Path
import sys
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts/infra'),
             str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code')]
from render_memlite_handoff_review import select_review_rows,validate_generation_receipt


class SelectionTests(unittest.TestCase):
    def test_actual_observer_receipt_field_names_and_mutation_rejection(self):
        cfg=dict(requests={'sha256':'a'*64},rows=383,head_checks=766)
        result=dict(status='frozen_handoff_predictions_completed_not_admitted',optimizer_updates=0,
            requests_sha256='a'*64,config_sha256='b'*64,observer_before_sha256='c'*64,
            observer_after_sha256='c'*64,observer_head_unchanged=True,all_rows_retained=True,rows=383,head_checks=766)
        validate_generation_receipt(result,cfg,'b'*64)
        for key,value in [('observer_after_sha256','d'*64),('observer_head_unchanged',False),
                          ('head_checks',765),('all_rows_retained',False),('optimizer_updates',1)]:
            with self.assertRaises(ValueError): validate_generation_receipt(dict(result,**{key:value}),cfg,'b'*64)

    def test_all_tasks_rare_changes_and_available_outcome_strata(self):
        rows=[];feedback={}
        for i in range(240):
            sid=str(i);change='identical_command' if i%2 else 'explicit_verb_or_bundle_change'
            if i in (12,38): change='explicit_bound_object_or_goal_change'
            rows.append(dict(sample_id=sid,original_identity={'task':i%97},target_change_class=change))
            feedback[sid]=dict(execution_feedback=json.dumps({'estimated_bundle_outcome':['SUCCEEDED','FAILED','UNKNOWN'][i%3]}))
        picked=select_review_rows(rows,feedback)
        self.assertGreaterEqual(len(picked),160)
        self.assertEqual(len({r['sample_id'] for r in picked}),len(picked))
        self.assertEqual({r['original_identity']['task'] for r in picked},set(range(97)))
        self.assertTrue({'12','38'}<={r['sample_id'] for r in picked})
        self.assertEqual(picked,select_review_rows(list(reversed(rows)),feedback))
        with self.assertRaises(ValueError): select_review_rows(rows,{})


if __name__=='__main__': unittest.main()
