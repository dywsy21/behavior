import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import trajectory_dataset as data
import native_trajectory_codec as codec
import trajectory_modeling as modeling
from trajectory_text import ids_digest
from test_trajectory_modeling import Processor
import test_trajectory_sft_export as export_tests


class Reader:
    def read(self, row, source):
        return {v: Image.new('RGB', (720 if v == 'head' else 480,) * 2) for v in modeling.VIEWS}, []
    def close(self): pass


class DatasetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name); self.root = root / 'dataset'; self.root.mkdir()
        corpus = root / 'corpus'; corpus.mkdir(); self.processor = Processor()
        row, source = export_tests.SFTExportTests().row()
        source.update(shard='shards/train/episode_000002.parquet', shard_sha256='sourcehash', samples=1)
        actor = codec.actor_from_state(row['task'], row['active_instruction'], row['observation_state'])
        target = codec.encode(row['expert_action'], row['observation_state'])
        images, _ = Reader().read(row, source)
        prefix = modeling.encode(self.processor, actor, images)
        encoded = modeling.encode(self.processor, actor, images, target=target)
        for image in images.values(): image.close()
        n = prefix['input_ids'].shape[1]
        row.update(actor_json=json.dumps(actor), target_json=target, prefix_tokens=n,
                   target_tokens=encoded['input_ids'].shape[1] - n, total_tokens=encoded['input_ids'].shape[1],
                   prefix_ids_sha256=ids_digest(prefix['input_ids'][0].tolist()),
                   target_ids_sha256=ids_digest(encoded['input_ids'][0, n:].tolist()),
                   target_sha256=hashlib.sha256(target.encode()).hexdigest(), quarantine_reason='')
        dest = self.root / source['shard']; dest.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row]), dest)
        saved = dest.read_bytes()
        file = {'kind': 'shards', 'path': source['shard'], 'sha256': hashlib.sha256(saved).hexdigest(),
                'bytes': len(saved), 'samples': 1}
        raw = json.dumps({'shards': [source]}).encode(); (corpus / 'manifest.json').write_bytes(raw)
        corpus_sha = hashlib.sha256(raw).hexdigest()
        self.addCleanup(patch.stopall)
        patch.object(data, 'CORPUS_SHA', corpus_sha).start()
        patch.object(data, 'CurrentVideoReader', Reader).start()
        m = {'schema': 'h85-composite-action-sft-v1', 'status': 'ACTION_SFT_FORMAT_COMPLETE_CAPACITY_PENDING',
             'protocol': codec.VERSION, 'corpus_manifest_sha256': corpus_sha, 'corpus_root': str(corpus),
             'training_eligible': False, 'samples_by_split': {'train': 1, 'validation': 0, 'test': 0},
             'sources': [{**source, 'source_shard': source['shard'], 'source_sha256': source['shard_sha256'], 'files': [file]}]}
        raw = json.dumps(m).encode(); (self.root / 'manifest.json').write_bytes(raw)
        self.manifest_sha = hashlib.sha256(raw).hexdigest(); self.shard = dest

    def test_prepared_requires_explicit_opt_in_and_returns_only_tensors(self):
        with self.assertRaises(ValueError): data.TrajectoryDataset(self.root, self.manifest_sha, self.processor)
        dataset = data.TrajectoryDataset(self.root, self.manifest_sha, self.processor, allow_prepared=True)
        try:
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(set(item), {'input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw', 'labels'})
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in item.values()))
            with self.assertRaises(IndexError): dataset[1]
        finally: dataset.close()

    def test_missing_requested_split_cannot_fall_back_to_train(self):
        with self.assertRaises(ValueError):
            data.TrajectoryDataset(self.root, self.manifest_sha, self.processor, split='test', allow_prepared=True)

    def test_pending_manifest_cannot_self_promote_or_use_truthy_flags(self):
        for flag in (1, 'true'):
            with self.assertRaises(ValueError):
                data.TrajectoryDataset(self.root, self.manifest_sha, self.processor, allow_prepared=flag)
        path = self.root / 'manifest.json'; manifest = json.loads(path.read_text())
        for flag in (True, 'false'):
            manifest['training_eligible'] = flag; content = json.dumps(manifest).encode(); path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            for opt_in in (False, True):
                with self.assertRaises(ValueError):
                    data.TrajectoryDataset(self.root, digest, self.processor, allow_prepared=opt_in)

    def test_processor_cannot_return_non_tensor_or_non_cpu_items(self):
        dataset = data.TrajectoryDataset(self.root, self.manifest_sha, self.processor, allow_prepared=True)
        try:
            real = dataset[0]
            class PretendGPU(torch.Tensor):
                @property
                def device(self): return torch.device('cuda:0')
            bad_cases = [{**real, 'extra': 'not a tensor'},
                         {**real, 'pixel_values': real['pixel_values'].as_subclass(PretendGPU)}]
            for bad in bad_cases:
                with patch.object(modeling, 'encode', return_value=bad):
                    with self.assertRaisesRegex(ValueError, 'CPU tensors'): dataset[0]
        finally: dataset.close()

    def test_changed_shard_or_failure_receipt_rejects(self):
        dataset = data.TrajectoryDataset(self.root, self.manifest_sha, self.processor, allow_prepared=True)
        self.shard.write_bytes(b'tampered')
        with self.assertRaises(ValueError): dataset[0]
        dataset.close()
        (self.root / 'failure.json').write_text('{}')
        with self.assertRaises(ValueError): data.TrajectoryDataset(self.root, self.manifest_sha, self.processor, allow_prepared=True)


if __name__ == '__main__': unittest.main()
