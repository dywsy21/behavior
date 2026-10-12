import json
from pathlib import Path
import sys
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts/infra'),
             str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code')]
from render_memlite_handoff_review import select_review_rows


class SelectionTests(unittest.TestCase):
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
