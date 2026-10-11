import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from behavior_champion.dataset import NativeV3Dataset, action_indices
from behavior_champion.release import AcceptedRelease


class DatasetTests(unittest.TestCase):
    def make(self, directory, rows, split='train', exclusions=None):
        root = Path(directory)
        exclusions=exclusions or []
        (root / 'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (root / 'exclusions.json').write_text(json.dumps(exclusions))
        payload = json.dumps(dict(raw_root=str(root / 'raw'),episodes=len(rows),
            excluded_episodes=len(exclusions),files={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('episodes.jsonl','exclusions.json')})).encode()
        (root / 'manifest.json').write_bytes(payload)
        counts={split:sum(bool(r['segments']) and r['split']==split for r in rows) for split in ('train','eval')}
        counts={key:value for key,value in counts.items() if value}
        (root / 'acceptance.json').write_text(json.dumps(dict(status='ACCEPTED',
            manifest_sha256=hashlib.sha256(payload).hexdigest(),eligible_episodes=counts)))
        return NativeV3Dataset(root, split, manifest_sha256=hashlib.sha256(payload).hexdigest())

    @staticmethod
    def ep(eid, length, phase, split='train'):
        return dict(row=dict(episode_index=eid, length=length, task_index=eid,
                             raw_episode_id=eid*10,task_instance_id=10),
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

    def test_whole_source_quarantine_is_not_revived_by_no_skill_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[self.ep(0,40,3),self.ep(1,400,0),self.ep(2,30,0,'eval')]
            rows[1]['segments']=[]
            excluded=[dict(episode_index=1,identity=[1,10,10],split='train',reason='Prior owner quarantine')]
            data=self.make(directory,rows,exclusions=excluded)
            self.assertEqual(len(data),3)
            self.assertEqual({data.locate(i)[0]['row']['episode_index'] for i in range(len(data))},{0})
            self.assertEqual(data.excluded_episode_counts,{'train':1})

    def test_quarantine_omission_wrong_identity_and_metadata_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[self.ep(0,40,3),self.ep(1,40,3)];rows[1]['segments']=[]
            with self.assertRaises(ValueError):self.make(directory,rows)
            wrong=[dict(episode_index=1,identity=[1,999,10],split='train')]
            with self.assertRaises(ValueError):self.make(directory,rows,exclusions=wrong)
            self.make(directory,[self.ep(0,40,3)])
            root=Path(directory);sha=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
            (root/'episodes.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):AcceptedRelease(root,manifest_sha256=sha)

    def test_instance_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[self.ep(0,40,3),self.ep(1,40,3,'eval')]
            rows[1]['row']['task_index']=0
            with self.assertRaises(ValueError):self.make(directory,rows)


if __name__ == '__main__': unittest.main()
