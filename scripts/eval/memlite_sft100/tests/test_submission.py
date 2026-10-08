import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import aggregate, sha256
from package_submission import package, wait_for_completion


class SubmissionTests(unittest.TestCase):
    def test_wait_never_restarts_or_packages_a_failed_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'status.json').write_text(json.dumps(dict(status='needs_diagnosis')))
            with self.assertRaisesRegex(RuntimeError,'needs diagnosis'):wait_for_completion(root)
            (root/'status.json').write_text(json.dumps(dict(status='completed')))
            self.assertIsNone(wait_for_completion(root))

    def fixture(self, root):
        tasks = [f'task{i}' for i in range(100)]
        bundle = root / 'submission'; bundle.mkdir()
        records, inventory = [], []
        for task in tasks:
            output = root / 'tasks' / task
            (output / 'json').mkdir(parents=True); (output / 'videos').mkdir()
            (output / 'command.json').write_text(json.dumps(dict(argv=['eval', task])))
            for instance in range(301, 311):
                record = dict(task=task,instance_id=instance,rollout_id=0,
                    success=instance == 301,q_score=dict(final=.5))
                path = output / 'json' / f'{task}_{instance}_0.json'
                video = output / 'videos' / (path.stem + '.mp4')
                path.write_text(json.dumps(record)); video.write_bytes(b'original-video-fixture')
                records.append(record)
                inventory.append(dict(metrics=str(path),video=str(video),
                    video_bytes=video.stat().st_size,metrics_sha256=sha256(path),video_sha256=sha256(video)))
        (root / 'status.json').write_text(json.dumps(dict(status='completed')))
        (root / 'summary.json').write_text(json.dumps(aggregate(tasks, records)))
        (root / 'manifest.json').write_text(json.dumps(dict(tasks=tasks,source_commit='test',
            official_tag='pinned-tag',official_commit='pinned',checkpoints={})))
        (bundle / 'rollout_inventory.json').write_text(json.dumps(inventory))
        (bundle / 'rgb_wrapper.py').write_text('# exact wrapper\n')
        (bundle / 'r1pro.yaml').write_text('model: galaxea_r1pro\n')
        (bundle / 'submission_checklist.json').write_text(json.dumps({
            k:dict(path=n,sha256=sha256(bundle/n))
            for k,n in [('wrapper','rgb_wrapper.py'),('robot','r1pro.yaml')]}))
        return inventory

    def test_all_json_are_original_metrics_no_internal_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inventory=self.fixture(root)
            receipt=package(root)
            self.assertEqual(receipt['rollout_count'],1000)
            self.assertFalse(receipt['submission_authorized'])
            with zipfile.ZipFile(receipt['zip']) as archive:
                self.assertEqual(len(archive.namelist()),1003)
                self.assertEqual(sum(n.endswith('.json') for n in archive.namelist()),1000)
                for row in inventory:
                    original=Path(row['metrics'])
                    self.assertEqual(archive.read(original.name),original.read_bytes())
            with self.assertRaises(ValueError):package(root)

    def test_incomplete_or_changed_originals_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inventory=self.fixture(root)
            (root/'status.json').write_text(json.dumps(dict(status='running')))
            with self.assertRaisesRegex(ValueError,'fully completed'):package(root)
            (root/'status.json').write_text(json.dumps(dict(status='completed')))
            Path(inventory[0]['metrics']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'metrics changed'):package(root)
            self.assertFalse((root/'submission/challenge_results_draft.zip').exists())


if __name__ == '__main__':unittest.main()
