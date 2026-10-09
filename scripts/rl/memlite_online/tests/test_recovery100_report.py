import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'code'),str(Path(__file__).resolve().parents[1]/'tools')]
from recovery_corpus import file_sha,split_group
from report_recovery100_coverage import report


class ReportTests(unittest.TestCase):
    def test_physical_candidates_are_not_approved_and_pending_is_not_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';src.mkdir();case=src/'task_1';case.mkdir()
            proposal=dict(task='task',instance_id=1,source_group='task:1',recovery_split=split_group('task',1),
                source_episode=dict(split='train'),selected_segment=dict(semantic=json.dumps([dict(verb='GRASP')])))
            (case/'manifest.json').write_text(json.dumps(proposal))
            (src/'manifest.json').write_text(json.dumps(dict(cases=[dict(directory='task_1',manifest_sha256=file_sha(case/'manifest.json'))])))
            catalog=dict(tasks=[dict(task='task',task_index=0)])
            pending=report(catalog,[src],[root/'collection'],[])
            self.assertEqual(pending['statuses'],{'pending':1});self.assertEqual(pending['totals']['started_tasks'],0)
            out=root/'collection'/'task_1';out.mkdir(parents=True)
            (out/'source.json').write_text(json.dumps(proposal))
            (out/'result.json').write_text(json.dumps(dict(status='completed_candidates_only',branches=[
                dict(kind='fault',physical_recovery_candidate=True)])))
            actual=report(catalog,[src],[root/'collection'],[])
            self.assertEqual(actual['totals']['physical_candidate_tasks'],1)
            self.assertEqual(actual['totals']['human_approved_tasks'],0)
            self.assertFalse(actual['ready_for_joint_head_training'])
            # Copies in two result roots must not masquerade as two events.
            with self.assertRaises(ValueError):report(catalog,[src],[root/'collection',root/'collection'],[])


if __name__=='__main__':unittest.main()
