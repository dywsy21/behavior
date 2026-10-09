"""Synthetic path/provenance tests; not a real-data approval."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_local_recovery_grasp_semantics import audit
from recovery_corpus import file_sha
from test_recovery_teacher_corpus import fixture


class SemanticAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        branch = self.root / 'raw/case/fault'
        branch.mkdir(parents=True)
        rows, manifest, plans = fixture()
        (branch / 'transitions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (branch / 'plans.json').write_text(json.dumps(plans))
        manifest.update(arm='left', transitions_sha256=file_sha(branch / 'transitions.jsonl'),
                        plans_sha256=file_sha(branch / 'plans.json'))
        (branch / 'manifest.json').write_text(json.dumps(manifest))
        self.review_root = self.root / 'reviews'
        review = self.review_root / 'case'
        review.mkdir(parents=True)
        (review / 'sheet.bin').write_bytes(b'synthetic structural proof only')
        (review / 'review.json').write_text(json.dumps(dict(
            source='/old-host/no-longer-mounted/case/fault',
            manifest_sha256=file_sha(branch / 'manifest.json'),
            transitions_sha256=manifest['transitions_sha256'],
            sheets=[dict(path='sheet.bin', sha256=file_sha(review / 'sheet.bin'))])))
        self.decisions = self.root / 'decisions.json'
        self.decisions.write_text(json.dumps(dict(schema='owner_local_recovery_review_v1', branches=[dict(
            case='case', branch='fault', review_directory='case',
            manifest_sha256=file_sha(branch / 'manifest.json'),
            outcomes=[dict(control_step=16, value='FAILED')], planner_steps=[])])))
        self.corpus = self.root / 'corpus'
        self.corpus.mkdir()
        (self.corpus / 'review-queue.json').write_text(json.dumps([
            dict(case='case', branch='fault', source_path='raw/case/fault')] ))

    def test_relocated_source_resolves_by_queue_and_hash_not_stale_path(self):
        result = audit(self.root, [self.decisions], review_root=self.review_root, corpus=self.corpus)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['checked'], 1)
        self.assertEqual(result['new_approvals'], 0)

    def test_changed_review_is_not_accepted_by_relocation(self):
        (self.review_root / 'case/sheet.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'media changed'):
            audit(self.root, [self.decisions], review_root=self.review_root, corpus=self.corpus)

    def test_independent_calibration_is_not_read_by_training_audit(self):
        self.decisions.write_text(json.dumps(dict(schema='owner_local_recovery_review_v1',
                                                  role='calibration_only_never_training_or_selection')))
        with self.assertRaisesRegex(ValueError, 'independent calibration'):
            audit(self.root, [self.decisions])


if __name__ == '__main__':
    unittest.main()
