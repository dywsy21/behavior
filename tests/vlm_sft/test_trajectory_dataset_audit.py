from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import audit_trajectory_dataset as audit


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.manifest = {'sources': [], 'quarantined': 1,
            'shard_bytes': 0, 'samples_by_split': dict.fromkeys(('train', 'validation', 'test'), 1),
            'samples_by_task_split': {str(t): dict.fromkeys(('train', 'validation', 'test'), int(t == 0)) for t in range(5)}}
        for episode, split in enumerate(('train', 'validation', 'test')):
            source = {'task': 0, 'instance': episode, 'episode': episode, 'split': split, 'files': []}
            for kind in (('shards', 'quarantine') if episode == 0 else ('shards',)):
                frame = 0 if kind == 'shards' else 16
                target = 4 if kind == 'shards' else 2000
                row = {'id': f't0_i{episode}_e{episode}_f{frame:06d}', 'source_split': split,
                       'task_index': 0, 'task_instance_id': episode, 'episode_index': episode,
                       'frame_index': frame, 'prefix_tokens': 10, 'target_tokens': target, 'total_tokens': 10 + target,
                       'quarantine_reason': '' if kind == 'shards' else 'response_over_budget'}
                relative = f'{kind}/{split}/episode_{episode:06d}.parquet'
                dest = self.root / relative; dest.parent.mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist([row]), dest)
                raw = dest.read_bytes(); self.manifest['shard_bytes'] += len(raw)
                source['files'].append({'kind': kind, 'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(),
                                        'bytes': len(raw), 'samples': 1})
            self.manifest['sources'].append(source)
        for name in ('launch.json', 'manifest.json', 'loader_preflight.json'): (self.root / name).write_text('{}')

    def check(self, manifest=None, wanted=None):
        with patch.object(audit, 'MAX_ROWS', 4):
            return audit.recount(self.root, manifest or self.manifest,
                                 {'t0_i0_e0_f000000'} if wanted is None else wanted, time.monotonic() + 30)

    def test_recount_includes_quarantine_but_train_index_does_not(self):
        report, indices = self.check()
        self.assertEqual(report['unique_ids'], 4)
        self.assertEqual(indices, {'t0_i0_e0_f000000': 0})
        self.assertEqual(report['tokens_by_split']['train'], {'total': 14, 'target': 4})
        self.assertEqual(report['quarantine_by_split']['train'], 1)

    def test_wrong_split_missing_case_or_extra_file_rejected(self):
        bad = deepcopy(self.manifest); bad['sources'][0]['split'] = 'test'
        with self.assertRaises(ValueError): self.check(bad)
        with self.assertRaises(ValueError): self.check(wanted={'missing'})
        (self.root / 'unregistered.parquet').write_bytes(b'junk')
        with self.assertRaises(ValueError): self.check()

    def test_quarantine_reason_must_match_exact_actual_budget_violation(self):
        file = self.manifest['sources'][0]['files'][1]; path = self.root / file['path']
        initial = pq.read_table(path).to_pylist()[0]
        for changes in ({'quarantine_reason': 'unknown'}, {'target_tokens': 4, 'total_tokens': 14}):
            row = {**initial, **changes}; before = file['bytes']
            pq.write_table(pa.Table.from_pylist([row]), path); content = path.read_bytes()
            file.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            self.manifest['shard_bytes'] += len(content) - before
            with self.assertRaises(ValueError): self.check()

    def test_per_task_accounting_is_verified_not_only_reported(self):
        bad = deepcopy(self.manifest); bad['samples_by_task_split']['0']['train'] = 2
        with self.assertRaises(ValueError): self.check(bad)


if __name__ == '__main__': unittest.main()
