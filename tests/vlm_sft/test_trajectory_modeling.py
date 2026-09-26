from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import native_trajectory_codec as codec
import trajectory_modeling as modeling


class Tokenizer:
    eos_token_id = 3
    unk_token_id = 9
    pad_token_id = 0
    def encode(self, text, **kwargs): return [ord(c) + 1000 for c in text]
    def decode(self, values, **kwargs): return ''.join(chr(v - 1000) for v in values if v != 3)
    def convert_tokens_to_ids(self, token): return {'<|im_end|>': 3, '<|image_pad|>': 4}.get(token, 9)
    def convert_ids_to_tokens(self, token): return {3: '<|im_end|>', 4: '<|image_pad|>'}.get(token, '<unk>')


class Processor:
    tokenizer = Tokenizer()
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_dict=True,
                              return_tensors='pt', enable_thinking=False)
        text = messages[1]['content'][-1]['text']
        ids = [1] + [4] * 396 + self.tokenizer.encode(text)
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones((1, len(ids)), dtype=torch.long),
                'mm_token_type_ids': torch.zeros((1, len(ids)), dtype=torch.long),
                'pixel_values': torch.zeros((1584, 1536)),
                'image_grid_thw': torch.tensor([[1, 28, 28], [1, 20, 20], [1, 20, 20]])}


class ModelingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.processor = Processor()
        self.state = np.zeros(61); self.state[23] = self.state[48] = 1
        self.actor = codec.actor_from_state('Bring the plate to the fridge.', 'verb=PLACEIN; target=plate', self.state)
        self.images = {v: Image.new('RGB', (720 if v == 'head' else 480,) * 2, c)
                       for v, c in zip(modeling.VIEWS, ('red', 'green', 'blue'))}
        self.target = codec.encode(np.zeros((16, 23)), self.state)

    def test_same_inference_prefix_and_response_only_eos(self):
        inference = modeling.encode(self.processor, self.actor, self.images)
        train = modeling.encode(self.processor, self.actor, self.images, target=self.target)
        n = inference['input_ids'].shape[1]
        self.assertTrue(torch.equal(inference['input_ids'], train['input_ids'][:, :n]))
        self.assertTrue(torch.all(train['labels'][:, :n] == -100))
        self.assertEqual(int(train['labels'][0, -1]), 3)
        self.assertEqual(self.processor.tokenizer.decode(train['labels'][0, n:].tolist()), self.target)
        self.assertTrue(torch.all(train['mm_token_type_ids'][:, n:] == 0))

    def test_view_order_and_fixed_resizing(self):
        msg = modeling.messages(self.actor, self.images)[1]['content']
        self.assertEqual([msg[i]['text'] for i in (0, 2, 4)], ['HEAD', 'LEFT_WRIST', 'RIGHT_WRIST'])
        self.assertEqual([msg[i]['image'].size for i in (1, 3, 5)], [(448, 448), (320, 320), (320, 320)])
        self.assertEqual([msg[i]['image'].getpixel((0, 0)) for i in (1, 3, 5)], [(255, 0, 0), (0, 128, 0), (0, 0, 255)])
        with self.assertRaises(ValueError): modeling.messages(self.actor, {**self.images, 'future': self.images['head']})

    def test_context_and_response_never_silently_truncated(self):
        with patch.object(modeling, 'MAX_RESPONSE', 10):
            with self.assertRaises(ValueError): modeling.encode(self.processor, self.actor, self.images, target=self.target)
        with patch.object(modeling, 'MAX_CONTEXT', 500):
            with self.assertRaises(ValueError): modeling.encode(self.processor, self.actor, self.images, target=self.target)

    def test_missing_pixels_or_image_tokens_rejected(self):
        prefix = modeling.encode(self.processor, self.actor, self.images)
        for key in ('pixel_values', 'image_grid_thw'):
            bad = dict(prefix); del bad[key]
            with self.assertRaises(ValueError): modeling.check_visual_tensor(bad, self.processor)
        bad = dict(prefix); bad['input_ids'] = prefix['input_ids'].clone(); bad['input_ids'][bad['input_ids'] == 4] = 5
        with self.assertRaises(ValueError): modeling.check_visual_tensor(bad, self.processor)

    def test_mixed_length_left_padding_never_supervised(self):
        short = modeling.encode(self.processor, self.actor, self.images, target=self.target)
        other = deepcopy(self.actor); other['task'] += ' Keep it balanced.'
        long = modeling.encode(self.processor, other, self.images, target=self.target)
        batch = modeling.collate([short, long], self.processor.tokenizer.pad_token_id)
        self.assertTrue((batch['attention_mask'] == 0).any())
        self.assertTrue(torch.all(batch['labels'][batch['attention_mask'] == 0] == -100))
        self.assertEqual(tuple(batch['image_grid_thw'].shape), (6, 3))
        self.assertEqual(tuple(batch['pixel_values'].shape), (3168, 1536))

    def test_long_variable_json_tail_loss_matches_full_causal_loss(self):
        short = modeling.encode(self.processor, self.actor, self.images, target=self.target)
        moving = np.zeros((16, 23)); moving[:, 7] = np.where(np.arange(16) % 2, .02, -.03)
        target = codec.encode(moving, self.state)
        long = modeling.encode(self.processor, self.actor, self.images, target=target)
        batch = modeling.collate([short, long], 0)

        class Output: pass
        class Model:
            def __call__(self, input_ids, logits_to_keep=None, **kwargs):
                logits = torch.nn.functional.one_hot((input_ids + 1) % 1200, 1200).float()
                result = Output(); result.logits = logits if logits_to_keep is None else logits[:, -logits_to_keep:]
                return result

        model = Model()
        logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 1200),
                    batch['labels'][:, 1:].reshape(-1), ignore_index=-100)
        actual = modeling.supervised_loss(model, batch)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertNotEqual(int((short['labels'] != -100).sum()), int((long['labels'] != -100).sum()))


if __name__ == '__main__': unittest.main()
