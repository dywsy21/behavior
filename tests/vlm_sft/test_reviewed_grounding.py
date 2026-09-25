from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import reviewed_grounding as reviewed


class ReviewedGroundingTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(reviewed.REVIEW.read_text())

    def approved(self):
        value = deepcopy(self.review)
        value['status'] = 'FINAL_REVIEW_PASSED'
        value['final_overlay_review']['records_sha256'] = reviewed.records_sha(value['rows'])
        return value

    def test_exact_cohort_and_all_positive_boxes(self):
        rows = reviewed.validate_records(self.review, allow_draft=True)
        self.assertEqual(len(rows), 81)
        self.assertEqual(sum(r['training_eligible'] for r in rows), 73)
        self.assertEqual(sum(len(r['target']['boxes']) for r in rows), 33)
        self.assertEqual(sum(r['raw_native_reviewed'] for r in rows), 44)

    def test_draft_missing_review_and_stale_labels_cannot_release(self):
        for mutation in ('draft', 'missing', 'stale', 'unseen', 'issues', 'missing_page'):
            value = self.approved()
            if mutation == 'draft': value['status'] = 'DRAFT_PENDING_FINAL_OVERLAY_REVIEW'
            if mutation == 'missing': value['final_overlay_review'] = None
            if mutation == 'stale': value['rows'][0]['review_note'] += ' edited'
            if mutation == 'unseen': value['final_overlay_review']['all_pages_inspected_by_parent'] = False
            if mutation == 'issues': value['final_overlay_review']['known_issues_remaining'] = 1
            if mutation == 'missing_page': value['final_overlay_review']['pages'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reviewed.validate_records(value)

    def test_repeat_duplicate_holdout_and_extra_action_field_rejected(self):
        for mutation in ('duplicate', 'repeat', 'holdout', 'action', 'path'):
            value = self.approved()
            if mutation == 'duplicate': value['rows'][1] = deepcopy(value['rows'][0])
            if mutation == 'repeat': value['rows'].append(deepcopy(value['rows'][0]))
            if mutation == 'holdout': value['rows'][0]['split'] = 'visual_validation'
            if mutation == 'action': value['rows'][0]['action'] = 'close_gripper'
            if mutation == 'path': value['rows'][0]['image'] = '../edited_overlay.png'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reviewed.validate_records(value, allow_draft=True)

    def test_uncertain_cannot_be_promoted_or_exported(self):
        for mutation in ('promote', 'eligible', 'no_quarantine'):
            value = self.approved()
            row = next(r for r in value['rows'] if r['target']['visibility'] == 'uncertain')
            if mutation == 'promote': row['target'] = {'visibility': 'absent', 'boxes': []}
            if mutation == 'eligible': row['training_eligible'] = True
            if mutation == 'no_quarantine': row['quarantine_reason'] = None
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reviewed.validate_records(value, allow_draft=True)

    def test_integer_outward_rounding_and_clipping(self):
        self.assertEqual(reviewed.normalized_boxes([[16, 0, 257, 480]], [480, 480]), [[33, 0, 536, 1000]])
        for box in ([False, 0, 20, 20], [0., 0, 20, 20], [-1, 0, 20, 20],
                    [0, 0, 481, 20], [20, 0, 20, 20]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                reviewed.normalized_boxes([box], [480, 480])
        value = self.approved()
        value['rows'][0]['target']['boxes'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'boxes disagree'):
            reviewed.validate_records(value, allow_draft=True)

    def test_export_separates_73_definite_from_8_unknown_and_never_overwrites(self):
        value = self.approved()
        with tempfile.TemporaryDirectory() as tmp, patch.object(reviewed, 'load_review', return_value=value) as audit:
            review_path = Path(tmp) / 'test_review.json'
            review_path.write_text(json.dumps(value))
            output = Path(tmp) / 'new_release'
            bundle = Path(tmp) / 'test_bundle'
            predictions = Path(tmp) / 'predictions.jsonl'
            result = reviewed.export(review_path, bundle, predictions, output)
            self.assertEqual(audit.call_count, 2)
            audit.assert_called_with(review_path, bundle, predictions, overlay_root=None)
            train = [json.loads(x) for x in (output / 'train.jsonl').read_text().splitlines()]
            quarantine = [json.loads(x) for x in (output / 'quarantine.jsonl').read_text().splitlines()]
            self.assertEqual(result['counts'], {'train': 73, 'quarantine': 8})
            self.assertTrue(all(r['training_eligible'] and json.loads(r['target'])['visibility'] != 'uncertain' for r in train))
            self.assertTrue(all(not r['training_eligible'] and json.loads(r['target'])['visibility'] == 'uncertain' for r in quarantine))
            self.assertTrue(all(not r['action_training_eligible'] and not r['independent_eval_eligible'] for r in train + quarantine))
            self.assertTrue(all('source_teacher_target' not in r and 'generated_token_ids' not in r for r in train + quarantine))
            self.assertEqual(result['historical_teacher_correct'], 69)
            with self.assertRaises(FileExistsError):
                reviewed.export(review_path, bundle, predictions, output)
            mismatched = deepcopy(value)
            mismatched['reviewer'] += ' altered'
            review_path.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(ValueError, 'bytes/record mismatch'):
                reviewed.export(review_path, bundle, predictions, Path(tmp) / 'bad')

    def test_missing_source_bundle_cannot_directly_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'release'
            with self.assertRaises(FileNotFoundError):
                reviewed.export(reviewed.REVIEW, Path(tmp) / 'missing', Path(tmp) / 'predictions', output)
            self.assertFalse(output.exists())

    def test_mid_export_review_change_never_seals(self):
        value = self.approved()
        changed = deepcopy(value)
        changed['reviewer'] += ' changed'
        with tempfile.TemporaryDirectory() as tmp, patch.object(reviewed, 'load_review', side_effect=[value, changed]):
            review_path = Path(tmp) / 'review.json'
            review_path.write_text(json.dumps(value))
            output = Path(tmp) / 'release'
            with self.assertRaisesRegex(ValueError, 'release not sealed'):
                reviewed.export(review_path, Path(tmp) / 'bundle', Path(tmp) / 'predictions', output)
            self.assertFalse((output / 'release.json').exists())

    def test_h83_protected_review_has_a_fixed_pin(self):
        value = self.approved()
        value['source_pins']['h83_parent_review_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'review identity'):
            reviewed.validate_records(value)

    def test_missing_or_corrupt_final_sheets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                reviewed.verify_final_sheets(self.approved(), Path(tmp))
            proof = self.approved()['final_overlay_review']
            receipt = {'purpose': 'human_only_final_label_review', 'training_eligible': False,
                       'records_sha256': proof['records_sha256'], 'pages': proof['pages']}
            (Path(tmp) / 'sheets.json').write_text(json.dumps(receipt))
            (Path(tmp) / Path(proof['pages'][0]['path']).name).write_bytes(b'wrong_image')
            with self.assertRaisesRegex(ValueError, 'overlay pixels changed'):
                reviewed.verify_final_sheets(self.approved(), Path(tmp))

    def test_changed_frozen_sources_rejected_before_export(self):
        with patch.object(reviewed, 'sha', return_value='0' * 64), self.assertRaisesRegex(ValueError, 'Frozen source'):
            reviewed.load_review(reviewed.REVIEW, Path('/unused'), Path('/unused/predictions'))

    def test_local_real_81_images_and_pixel_pins(self):
        root = reviewed.REPO / 'artifacts/agentic-vlm-goal-20260918'
        bundle = root / 'h76_raw_review_bundle_v1'
        predictions = root / 'h82_reference_results_v1/calibration/predictions.jsonl'
        if not (bundle / 'review_manifest.json').exists() or not predictions.exists():
            self.skipTest('Pinned image artifacts are not distributed in Git')
        actual = reviewed.load_review(reviewed.REVIEW, bundle, predictions, allow_draft=True)
        self.assertEqual(len(actual['rows']), 81)
        original_sha = reviewed.sha
        def corrupted(path):
            return '0' * 64 if Path(path).suffix == '.png' else original_sha(path)
        with patch.object(reviewed, 'sha', side_effect=corrupted), self.assertRaisesRegex(ValueError, 'PNG changed'):
            reviewed.load_review(reviewed.REVIEW, bundle, predictions, allow_draft=True)


if __name__ == '__main__':
    unittest.main()
