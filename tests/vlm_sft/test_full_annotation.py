from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import prepare_full_annotation as full


class FullAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(full.PLAN.read_bytes())
        self.source = {'task': 0, 'instance': 9, 'episode': 8, 'split': 'train', 'selected_frames': [0]}
        self.original = {'source_identity': {'protected_groups': []}, 'sources': [self.source]}
        self.state = {'id': 't0_i9_e8_f000000', 'task': 0, 'instance': 9, 'episode': 8, 'frame': 0,
                      'split': 'train', 'training_eligible': False,
                      'images': {v: f'images/t0_i9_e8_f000000_{v}.png' for v in full.VIEWS},
                      'image_receipts': {v: {'resolution': [720, 720] if v == 'head' else [480, 480],
                                            'raw_resolution_preserved': True,
                                            'png_sha256': 'a' * 64, 'raw_pixels_sha256': 'b' * 64}
                                         for v in full.VIEWS}}
        self.plan['expected'] = {'states': 1, 'groups': 1, 'images': 3, 'jobs': 3}
        self.pin = 'c' * 64
        self.images, self.jobs = full.make_inventory(self.plan, self.pin, [self.state], set())
        self.jobs = {r['job_id']: r for r in self.jobs}

    def attempt(self, job=None, **kwargs):
        job = job or next(iter(self.jobs.values()))
        return {'schema': 'h84-annotation-attempt-v1', 'attempt_id': 'run-a:0',
                'job_id': job['job_id'], 'queue_sha256': 'd' * 64,
                'png_sha256': job['png_sha256'], 'query': job['query'], 'status': 'PROPOSED',
                'model_identity_sha256': 'e' * 64, 'protocol_sha256': 'f' * 64,
                'response': '{"visibility":"absent","boxes":[]}', 'error': None, **kwargs}

    def test_full_registry_has_15_task_query_slots_and_14_classes(self):
        queries = json.loads(full.PLAN.read_bytes())['queries_by_task']
        self.assertEqual(sum(map(len, queries.values())), 15)
        self.assertEqual(len(set(q for values in queries.values() for q in values)), 14)
        self.assertEqual([len(queries[str(t)]) for t in range(5)], [1, 2, 5, 4, 3])

    def test_all_inputs_are_hash_pinned_without_loading_model(self):
        plan = json.loads(full.PLAN.read_bytes())
        for path, pin in plan['review_files'].items():
            full.checked_bytes(full.REPO / path, pin)
        full.checked_bytes(full.REPO / plan['actor_protocol_file'], plan['actor_protocol_sha256'])

    def test_original_coverage_duplicate_path_split_and_protection(self):
        full.validate_states([self.state], self.original, self.plan['expected'])
        for mutation in ('duplicate', 'path', 'split', 'eligible', 'frame', 'camera', 'protected'):
            states = [deepcopy(self.state)]
            source = deepcopy(self.original)
            if mutation == 'duplicate': states *= 2
            if mutation == 'path': states[0]['images']['head'] = '../wrong.png'
            if mutation == 'split': states[0]['split'] = 'test'
            if mutation == 'eligible': states[0]['training_eligible'] = True
            if mutation == 'frame': states[0]['frame'] = 1
            if mutation == 'camera': states[0]['image_receipts']['head']['resolution'] = [480, 480]
            if mutation == 'protected': source['source_identity']['protected_groups'] = [[0, 9]]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                full.validate_states(states, source, self.plan['expected'])

    def test_calibration_group_is_completely_excluded_not_only_reviewed_frame(self):
        images, jobs = full.make_inventory(self.plan, self.pin, [self.state], {(0, 9)})
        self.assertTrue(all(x['role'] == 'annotation_calibration' and not x['training_eligible'] for x in images + jobs))
        changed = deepcopy(self.state)
        changed['split'] = 'test'
        images, jobs = full.make_inventory(self.plan, self.pin, [changed], set())
        self.assertTrue(all(x['role'] == 'test' and not x['training_eligible'] for x in images + jobs))

    def test_job_ids_deterministic_and_versioned(self):
        _, same = full.make_inventory(self.plan, self.pin, [self.state], set())
        _, changed = full.make_inventory(self.plan, '9' * 64, [self.state], set())
        self.assertEqual(same, list(self.jobs.values()))
        self.assertTrue(set(self.jobs).isdisjoint(x['job_id'] for x in changed))

    def test_no_attempts_all_pending_not_approved(self):
        rows, report = full.resume_status(self.jobs, [], 'd' * 64)
        self.assertEqual(report['counts'], {'PENDING': 3})
        self.assertEqual(report['accepted_labels'], 0)
        self.assertFalse(report['training_eligible'])

    def test_error_then_proposal_can_resume_without_becoming_training_data(self):
        error = self.attempt(status='ERROR', response=None, error='OOM')
        proposal = self.attempt(attempt_id='run-b:0')
        _, report = full.resume_status(self.jobs, [error], 'd' * 64)
        self.assertEqual(report['counts'], {'RETRY_PENDING': 1, 'PENDING': 2})
        _, report = full.resume_status(self.jobs, [error, proposal], 'd' * 64)
        self.assertEqual(report['counts'], {'QUALITY_PENDING': 1, 'PENDING': 2})
        self.assertEqual(report['accepted_labels'], 0)

    def test_uncertain_queued_for_human_review(self):
        attempt = self.attempt(response='{"visibility":"uncertain","boxes":[]}')
        _, report = full.resume_status(self.jobs, [attempt], 'd' * 64)
        self.assertEqual(report['counts'], {'REVIEW_REQUIRED': 1, 'PENDING': 2})

    def test_foreign_tampered_or_fake_approved_attempts_rejected(self):
        for changes in ({'query': 'the cauldron'}, {'png_sha256': '0' * 64}, {'queue_sha256': '0' * 64},
                        {'job_id': 'unknown'}, {'status': 'APPROVED'}, {'training_eligible': True},
                        {'model_identity_sha256': ''}, {'response': '{"visibility":"present","boxes":[]}'},
                        {'response': '{"visibility":"absent","boxes":[],"action":"grasp"}'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                full.resume_status(self.jobs, [self.attempt(**changes)], 'd' * 64)

    def test_duplicate_or_overwrite_of_proposal_rejected(self):
        attempt = self.attempt()
        for second in (attempt, self.attempt(attempt_id='run-b:0')):
            with self.assertRaises(ValueError):
                full.resume_status(self.jobs, [attempt, second], 'd' * 64)

    def test_partial_attempt_file_rejected_without_new_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'queue.json').write_text('{}')
            attempt_path = root / 'partial.jsonl'
            attempt_path.write_bytes(full.packed(self.attempt()))
            with patch.object(full, 'load_queue', return_value=(self.jobs, {'verified_queue_sha256': 'd' * 64})), self.assertRaises(ValueError):
                full.write_resume(root, root, [attempt_path], root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_preparation_seals_no_models_and_is_non_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'queue'
            metadata = Path(folder) / 'metadata'
            values = (self.plan, self.pin, self.images, list(self.jobs.values()), [], [])
            with patch.object(full, 'inventory', return_value=values), \
                    patch.object(full, 'load_inputs', return_value=(self.plan, self.pin)):
                result = full.create_queue(metadata, output)
                self.assertEqual(result['new_model_calls'], 0)
                self.assertEqual(result['accepted_labels'], 0)
                self.assertEqual(result['shards'], 1)
                with self.assertRaises(FileExistsError):
                    full.create_queue(metadata, output)

    def test_duplicate_json_and_nonfinite_evidence_rejected(self):
        for text in ('{"status":"ERROR","status":"PROPOSED"}',
                     '{"outer":{"x":1,"x":2}}', '{"wall_seconds":NaN}', '{"x":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                full.json_lines(text.encode())

    def test_cannot_write_inside_frozen_metadata_or_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'metadata'
            source.mkdir()
            alias = root / 'alias'
            alias.symlink_to(source, target_is_directory=True)
            for output in (source, source / 'new', alias / 'new'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    full.forbid_source_output(output, source)

    def test_complete_queue_seal_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'queue'
            values = (self.plan, self.pin, self.images, list(self.jobs.values()), [], [])
            with patch.object(full, 'inventory', return_value=values), \
                    patch.object(full, 'load_inputs', return_value=(self.plan, self.pin)):
                original = full.create_queue(root / 'metadata', output)
                full.load_queue(output, root / 'metadata')
                for mutation in ('builder_sha256', 'new_model_calls', 'training_updates', 'controls',
                                 'calibration_presence_pairs', 'images_by_role', 'jobs_by_role',
                                 'files_rows', 'files_bytes', 'unknown', 'boolean_zero'):
                    value = deepcopy(original)
                    if mutation == 'unknown': value['approved'] = True
                    elif mutation == 'boolean_zero': value['new_model_calls'] = False
                    elif mutation.startswith('files_'):
                        value['files']['image_index.jsonl'][mutation[6:]] = 999
                    else: value[mutation] = 999
                    (output / 'queue.json').write_bytes(full.packed(value))
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        full.load_queue(output, root / 'metadata')


if __name__ == '__main__':
    unittest.main()
