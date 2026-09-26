from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import audit_trajectory_encoding as audit
import native_trajectory_codec as codec
from test_trajectory_modeling import Processor


class RealShapeProcessor(Processor):
    swap = False

    def image_processor(self, images, **kwargs):
        assert len(images) == 1
        img = images[0]; side = img.width // 16
        return {'image_grid_thw': torch.tensor([[1, side, side]]),
                'pixel_values': torch.full((side * side, 1536), float(img.getpixel((0, 0))[0]))}

    def apply_chat_template(self, messages, **kwargs):
        out = super().apply_chat_template(messages, **kwargs)
        parts = [self.image_processor(images=[messages[1]['content'][i]['image']])['pixel_values'] for i in (1, 3, 5)]
        if self.swap: parts[1], parts[2] = parts[2], parts[1]
        out['pixel_values'] = torch.cat(parts)
        return out


class EncodingAuditTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def fixture(self):
        cases = []
        for task in range(5):
            for i in range(10):
                episode = task * 10 + i
                row = {'id': f't{task}_i{i}_e{episode}_f000032', 'source_split': 'train',
                       'frame_index': 32, 'timestamp_s': 32 / 30, 'task_index': task,
                       'task_instance_id': i, 'episode_index': episode}
                cases.append({'id': row['id'], 'task': task, 'instance': i, 'episode': episode,
                              'frame': 32, 'split': 'train', 'row': row,
                              'images': [{'offset': 0} for _ in range(3)]})
        manifest = {'cases': cases, 'status': 'DECODED_HUMAN_REVIEW_PENDING',
                    'training_eligible': False, 'manifest_sha256': audit.CORPUS_SHA}
        human = {'status': 'SAMPLED_VISUAL_ALIGNMENT_REVIEW_COMPLETED', 'review_manifest_sha256': audit.REVIEW_SHA,
                 'corpus_manifest_sha256': audit.CORPUS_SHA, 'reviewed_cases': 50,
                 'confirmed_visual_alignment_issues_in_sample': 0, 'cases': [{'id': c['id']} for c in cases]}
        return manifest, human

    def test_exact_independent_train_cohort(self):
        manifest, human = self.fixture()
        self.assertEqual(len(audit.checked_cases(manifest, human)), 50)
        for path in ('split', 'id'):
            bad = deepcopy(manifest); bad['cases'][0][path] = 'test'
            with self.assertRaises(ValueError): audit.checked_cases(bad, human)
        bad = deepcopy(manifest); bad['cases'][0] = deepcopy(bad['cases'][1])
        with self.assertRaises(ValueError): audit.checked_cases(bad, human)
        human['review_manifest_sha256'] = 'wrong'
        with self.assertRaises(ValueError): audit.checked_cases(manifest, human)

    def test_future_receipts_not_counted_as_current(self):
        manifest, human = self.fixture(); manifest['cases'][0]['images'][0]['offset'] = 8
        with self.assertRaises(ValueError): audit.checked_cases(manifest, human)

    def test_independent_tensor_order_and_response_supervision(self):
        state = np.zeros(61); state[23] = state[48] = 1
        actor = codec.actor_from_state('Carry the plate.', 'verb=PLACEIN; target=plate', state)
        images = {v: Image.new('RGB', (720 if v == 'head' else 480,) * 2, color)
                  for v, color in zip(('head', 'left_wrist', 'right_wrist'), ('red', 'green', 'white'))}
        target = codec.encode(np.zeros((16, 23)), state); processor = RealShapeProcessor()
        _, stats = audit.check_encoding(processor, actor, images, target)
        self.assertEqual(stats['target_tokens_including_eos'], stats['supervised_tokens'])
        self.assertEqual(stats['total_tokens'], stats['prefix_tokens'] + stats['supervised_tokens'])
        processor.swap = True
        with self.assertRaisesRegex(ValueError, 'ordering/payload'):
            audit.check_encoding(processor, actor, images, target)


if __name__ == '__main__': unittest.main()
