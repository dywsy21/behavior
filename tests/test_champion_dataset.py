import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from behavior_champion.dataset import NativeV3Dataset, action_indices


class DatasetTests(unittest.TestCase):
    def make(self, directory, rows, split='train'):
        root = Path(directory)
        payload = json.dumps({'raw_root': str(root / 'raw')}).encode()
        (root / 'manifest.json').write_bytes(payload)
        (root / 'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return NativeV3Dataset(root, split, manifest_sha256=hashlib.sha256(payload).hexdigest())

    @staticmethod
    def ep(eid, length, phase, split='train'):
        return dict(row=dict(episode_index=eid, length=length, task_index=eid),
                    phase=phase, split=split, segments=[{'end': 7}])

    def test_exact_stride_and_no_skill_boundary_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make(directory, [self.ep(0, 40, 3), self.ep(1, 33, 0), self.ep(2, 99, 0, 'eval')])
            self.assertEqual(len(data), 6)
            self.assertEqual([data.locate(i)[1] for i in range(6)], [3, 19, 35, 0, 16, 32])
            self.assertEqual([data.locate(i)[0]['row']['episode_index'] for i in range(6)], [0, 0, 0, 1, 1, 1])
            for index in (-1, 6):
                with self.assertRaises(IndexError): data.locate(index)

    def test_native_chunk_crosses_skills_and_pads_only_episode_end(self):
        np.testing.assert_array_equal(action_indices(3, 40), np.arange(3, 33))
        np.testing.assert_array_equal(action_indices(35, 40), [35, 36, 37, 38] + [39]*26)
        for args in ((40, 40), (-1, 40), (3, 40, 32)):
            with self.assertRaises(ValueError): action_indices(*args)

    def test_release_hash_and_duplicate_episode_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make(directory, [self.ep(0, 40, 3)])
            with self.assertRaises(ValueError): NativeV3Dataset(directory)
            with self.assertRaises(ValueError):
                self.make(directory, [self.ep(0, 40, 3), self.ep(0, 40, 3, 'eval')])

    def test_eval_stays_in_original_split(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make(directory, [self.ep(0, 40, 3), self.ep(1, 20, 8, 'eval')], split='eval')
            self.assertEqual(len(data), 1)
            self.assertEqual(data.locate(0)[0]['split'], 'eval')


if __name__ == '__main__': unittest.main()
