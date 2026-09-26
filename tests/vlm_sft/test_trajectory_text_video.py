from copy import deepcopy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import native_trajectory_codec as codec
from trajectory_text import prefix_ids, ids_digest
from trajectory_video import validate_reference
import trajectory_video as video_module


class TextVideoTests(unittest.TestCase):
    def test_three_placeholder_expansion_and_no_tokenized_paths(self):
        state = np.zeros(61); state[23] = state[48] = 1
        actor = codec.actor_from_state('Move.', 'verb=NAVIGATE; target=table', state)

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                assert kwargs == dict(chat_template='template', tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
                return 'start<|image_pad|>one<|image_pad|>two<|image_pad|>end'
            def encode(self, text, **kwargs): return [1] + [3] * text.count('<|image_pad|>') + [2]
            def convert_tokens_to_ids(self, token): return 3

        t = Tokenizer(); ids = prefix_ids(t, 'template', actor)
        self.assertEqual(len(ids), 398)
        self.assertEqual(len(ids_digest(ids)), 64)
        self.assertEqual([c for c in t.messages[1]['content'] if c['type'] == 'image'], [{'type': 'image'}] * 3)

    def fixture(self):
        videos = {v: {'path': f'/source/{v}.mp4', 'episode_start_timestamp_s': i * 100., 'fps': 30,
                      'resolution': [720, 720] if v == 'head' else [480, 480],
                      'stream_start_s': 0., 'stream_duration_s': 400.}
                  for i, v in enumerate(('head', 'left_wrist', 'right_wrist'))}
        shard = {'task': 0, 'instance': 1, 'episode': 2, 'split': 'train', 'frames': 100, 'videos': videos}
        row = {'id': 't0_i1_e2_f000032', 'task_index': 0, 'task_instance_id': 1, 'episode_index': 2,
               'source_split': 'train', 'frame_index': 32, 'timestamp_s': 32 / 30,
               'image_references': [{'view': v, 'video_path': r['path'],
                                     'episode_start_timestamp_s': r['episode_start_timestamp_s'],
                                     'requested_timestamp_s': r['episode_start_timestamp_s'] + 32 / 30}
                                    for v, r in videos.items()]}
        return row, shard

    def test_independent_camera_clocks_and_identity(self):
        row, shard = self.fixture()
        self.assertEqual(validate_reference(row, shard, 'right_wrist')[1], 200 + 32 / 30)
        bad = deepcopy(row); bad['image_references'][1]['requested_timestamp_s'] += 8 / 30
        with self.assertRaises(ValueError): validate_reference(bad, shard, 'left_wrist')
        for change in ({'source_split': 'test'}, {'episode_index': 3}, {'id': 'other'}):
            with self.assertRaises(ValueError): validate_reference({**row, **change}, shard, 'head')

    def test_cross_view_reference_swap_and_source_bounds(self):
        row, shard = self.fixture()
        bad = deepcopy(row); bad['image_references'][0]['video_path'] = '/source/right_wrist.mp4'
        with self.assertRaises(ValueError): validate_reference(bad, shard, 'head')
        bad = deepcopy(shard); bad['videos']['head']['stream_duration_s'] = .5
        with self.assertRaises(ValueError): validate_reference(row, bad, 'head')

    def test_reader_decodes_actual_clock_and_closes_owned_containers(self):
        row, shard = self.fixture(); containers = []
        for v in shard['videos'].values(): v.update(resolved_path=v['path'], bytes=1, mtime_ns=1)

        class Container:
            def __init__(self, path):
                self.path = path; self.closed = False
                self.streams = SimpleNamespace(video=[SimpleNamespace(average_rate=30, codec_context=SimpleNamespace())])
                containers.append(self)
            def close(self): self.closed = True

        def choose(container, stream, target):
            size = 720 if container.path.endswith('head.mp4') else 480
            return SimpleNamespace(to_image=lambda: Image.new('RGB', (size, size))), target

        with patch.dict(sys.modules, {'av': SimpleNamespace(open=Container)}), \
             patch.object(video_module, 'video_identity', side_effect=lambda v: Path(v['path'])), \
             patch.object(video_module, 'choose_frame', side_effect=choose):
            with video_module.CurrentVideoReader() as reader:
                images, receipts = reader.read(row, shard)
                self.assertEqual(len(images), 3); self.assertEqual(receipts[2]['actual_timestamp_s'], 200 + 32 / 30)
                for image in images.values(): image.close()
                with patch.object(video_module, 'choose_frame', side_effect=lambda c, s, t: (None, t + .1)):
                    with self.assertRaisesRegex(ValueError, 'current observation'): reader.read(row, shard)
            self.assertEqual(len(containers), 3)
            self.assertTrue(all(c.closed for c in containers))


if __name__ == '__main__': unittest.main()
