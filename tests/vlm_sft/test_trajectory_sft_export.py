from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import prepare_trajectory_sft as export
import native_trajectory_codec as codec
from test_trajectory_modeling import Tokenizer
import test_trajectory_text_video as text_video_tests
import test_trajectory_encoding_audit as encoding_tests
from PIL import Image


class SFTExportTests(unittest.TestCase):
    def row(self):
        row, shard = text_video_tests.TextVideoTests().fixture()
        state = np.zeros(61, np.float32); state[23] = state[48] = 1
        actions = np.zeros((16, 23), np.float32); actions[:, [14, 22]] = -1
        row.update(task='Move the object.', active_instruction='verb=GRASP; target=object',
                   observation_state=state.tolist(), expert_action=actions.tolist())
        return row, shard

    def test_full_control_preserved_and_private_identity_not_actor(self):
        row, _ = self.row()
        with patch.object(export, 'prefix_ids', return_value=[7] * 1100):
            out, errors = export.prepare_row(row, Tokenizer(), 'template')
        self.assertEqual(out['expert_action'], row['expert_action'])
        self.assertEqual(out['observation_state'], row['observation_state'])
        self.assertEqual(out['quarantine_reason'], '')
        actor = export.strict_json(out['actor_json']); codec.validate_actor(actor)
        self.assertNotIn('episode_index', actor); self.assertNotIn('expert_action', actor)
        self.assertEqual(out['total_tokens'], out['prefix_tokens'] + out['target_tokens'])
        self.assertEqual(errors, dict.fromkeys(codec.ERROR_LIMITS, 0.))

    def test_overlength_quarantined_not_truncated(self):
        row, _ = self.row(); expected = codec.encode(row['expert_action'], row['observation_state'])
        with patch.object(export, 'prefix_ids', return_value=[7] * 1100), patch.object(export, 'MAX_RESPONSE', 2):
            out, _ = export.prepare_row(row, Tokenizer(), 'template')
        self.assertEqual(out['quarantine_reason'], 'response_over_budget')
        self.assertEqual(out['target_json'], expected)
        self.assertGreater(out['target_tokens'], 2)

    def test_real_parquet_shard_roundtrip_split_and_sha(self):
        row, source = self.row()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); corpus = root / 'corpus'; output = root / 'output'
            source.update(shard='shards/train/episode_000002.parquet', samples=1)
            source_file = corpus / source['shard']; source_file.parent.mkdir(parents=True)
            for view, video in source['videos'].items():
                path = root / f'{view}.mp4'; path.write_bytes(b'test video metadata, not decoded')
                stat = path.stat()
                video.update(path=str(path), resolved_path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
                next(r for r in row['image_references'] if r['view'] == view)['video_path'] = str(path)
            pq.write_table(pa.Table.from_pylist([row]), source_file)
            data = source_file.read_bytes(); source.update(shard_bytes=len(data), shard_sha256=hashlib.sha256(data).hexdigest())
            (output / 'shards/train').mkdir(parents=True); (output / 'quarantine/train').mkdir(parents=True)
            (output / 'launch.json').write_text('{}'); (output / 'loader_preflight.json').write_text('{}')
            with patch.object(export, 'TOKENIZER', Tokenizer(), create=True), patch.object(export, 'TEMPLATE', 'template', create=True), \
                 patch.object(export, 'DEADLINE', time.monotonic() + 60, create=True), \
                 patch.object(export, 'STOPPED', threading.Event(), create=True), \
                 patch.object(export, 'prefix_ids', return_value=[7] * 1100):
                result = export.build_shard(source, corpus, output)
                self.assertEqual((result['samples'], result['quarantined']), (1, 0))
                saved = pq.read_table(output / result['files'][0]['path']).to_pylist()[0]
                self.assertEqual(saved['expert_action'], row['expert_action'])
                self.assertEqual(saved['source_split'], 'train')
                export.verify_export_files(output, [result], [source])
                with self.assertRaises(FileExistsError): export.build_shard(source, corpus, output)
                with self.assertRaises(ValueError): export.build_shard({**source, 'shard_sha256': 'wrong'}, corpus, output)
                target = output / result['files'][0]['path']; content = target.read_bytes()
                target.write_bytes(b'truncated')
                with self.assertRaises(ValueError): export.verify_export_files(output, [result], [source])
                target.write_bytes(content)
                extra = output / 'unexpected.txt'; extra.write_text('unexpected')
                with self.assertRaises(ValueError): export.verify_export_files(output, [result], [source])
                extra.unlink()
                Path(source['videos']['head']['path']).write_bytes(b'changed source video')
                with self.assertRaises(ValueError): export.verify_export_files(output, [result], [source])

    def test_preflight_fifty_join_and_current_pixel_mismatch(self):
        manifest, human = encoding_tests.EncodingAuditTests().fixture()
        original, shard = self.row(); images = {}
        for view in ('head', 'left_wrist', 'right_wrist'):
            image = Image.new('RGB', (720 if view == 'head' else 480,) * 2)
            images[view] = hashlib.sha256(image.tobytes()).hexdigest(); image.close()
        sources = []
        for case in manifest['cases']:
            case['row'] = {**deepcopy(original), **case['row']}
            case['images'] = [{'offset': 0, 'view': v, 'pixels_sha256': digest} for v, digest in images.items()]
            sources.append({**shard, **{k: case[k] for k in ('task', 'instance', 'episode')}})
        ids = [1, 4, 2]
        encoded = [{'id': c['id'], 'prefix_ids_sha256': export.ids_digest(ids)} for c in manifest['cases']]
        values = {'manifest.json': export.packed(manifest), 'h85_parent_action_review_v1.json': export.packed(human),
                  'result.json': export.packed({'status': 'CPU_MULTIMODAL_ENCODING_CHECKED_NOT_TRAIN_RELEASE',
                         'review_manifest_sha256': export.REVIEW_SHA, 'cases': 50, 'rows_sha256': 'rowsdigest'}),
                  'rows.jsonl': b'\n'.join(export.packed(r) for r in encoded), 'chat_template.jinja': b'template'}

        class Reader:
            wrong = False
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, row, source):
                return {v: Image.new('RGB', (720 if v == 'head' else 480,) * 2,
                                    'white' if self.wrong else 'black') for v in images}, []

        with patch.object(export, 'snapshot', side_effect=lambda p, pins, expected=None: values[Path(p).name]), \
             patch.object(export, 'prefix_ids', return_value=ids), patch.object(export, 'CurrentVideoReader', Reader), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
            args = ({'shards': sources}, Path('/review'), Path('/encoded'), Path('/model'), {}, time.monotonic() + 60)
            self.assertEqual(export.preflight(*args)['current_images'], 150)
            Reader.wrong = True
            with self.assertRaisesRegex(ValueError, 'video loader'): export.preflight(*args)
            Reader.wrong = False
            values['rows.jsonl'] = b'\n'.join(export.packed(r) for r in encoded[1:])
            with self.assertRaisesRegex(ValueError, 'cohort'): export.preflight(*args)


if __name__ == '__main__': unittest.main()
