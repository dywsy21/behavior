"""CPU-only checks; synthetic fixtures are never the benchmark input file."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from g05.data_processor.processor.memlite_v6_projection import validate_embedded_model_projection
from g05.models.g05.helpers.vlm_lora import VLMloraConfig, inject_vlm_lora, restore_lora_state
from g05.utils.memlite_skill_protocol import V6_MODEL_PROJECTION_FIELDS, canonical_json

entry = Path(__file__).resolve().parents[1] / 'scripts/infra/benchmark_memlite_oneframe.py'
spec = importlib.util.spec_from_file_location('mem1f_bench', entry)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def fixtures():
    skill = dict(verb='GRASP', target='radio', source='table', destination='',
                 target_part='', arm='RIGHT', unbound_relation='')
    rendered = '; '.join(f'{key}={canonical_json(skill[key] or "NONE")}'
                         for key in ('verb', 'target', 'source', 'destination', 'target_part', 'arm'))
    sample = {key: 'none' for key in V6_MODEL_PROJECTION_FIELDS}
    sample.update(schema_version=6, memlite_schema_version=6, memlite_branch='low',
                  active_skills_semantic_json=canonical_json([skill]), active_skills_text='Active skills: ['+rendered+'].',
                  parent_goal='Parent command: description="pick up radio"; targets=["radio"]; sources=["table"]; destinations=[]; target_parts=[]; arms=[]',
                  next_decision='EXECUTE', task_complete=False, low_action_supervision_mask=True,
                  outcome_supervision_mask=False, parent_goal_supervision_mask=True, outcome_target='UNKNOWN',
                  command='test fixture only', embodiment='galaxea_r1pro')
    pad = torch.zeros(27, dtype=torch.bool)
    pad[list(bench.PAD)] = True
    sample['proprio'] = dict(value=torch.arange(6*27).view(6, 27).float(), proprio_dim_is_pad=pad)
    sample['template'] = '<chat_user_prefix>'+''.join(f'<image{i}_image_!>' for i in range(18))+'<bos>Parent goal: <parent_goal_text_!>; <active_skills_text_text_!> State: <proprio_proprio_!>;<EOV><EOC><eos>'
    sample.update({f'image{i}': (256, 256) for i in range(18)})
    pixels = {key: (torch.arange(6).float()+100*k).view(1, 6, 1, 1, 1).expand(2, 6, 3, 256, 256)
              for k, key in enumerate(bench.CAMERAS)}
    batch = dict(samples=[deepcopy(sample), deepcopy(sample)], pixel_values=pixels,
                 action=torch.arange(32*27).view(1, 32, 27).float().expand(2, 32, 27),
                 action_is_pad=torch.zeros(2, 32, dtype=torch.bool), action_dim_is_pad=pad.expand(2, 27),
                 memlite_audit_locator={'do_not_pass_to_model': True})
    return [batch]*5


class OneFrameContract(unittest.TestCase):
    def test_current_frame_and_future_action_zero_are_preserved(self):
        batches = fixtures()
        original = batches[0]['samples'][0]['template']
        rows = bench.single_frame_pool(batches)
        self.assertEqual(len(rows), 10)
        row = rows[0]
        for k, key in enumerate(bench.CAMERAS):
            self.assertTrue(torch.all(row['pixel_values'][key] == 100*k+5))
            self.assertEqual(tuple(row['pixel_values'][key].shape), (1, 3, 256, 256))
        self.assertTrue(torch.equal(row['samples']['proprio']['value'], batches[0]['samples'][0]['proprio']['value'][-1:]))
        self.assertTrue(torch.equal(row['action'], batches[0]['action'][0]))
        self.assertEqual(batches[0]['samples'][0]['template'], original)
        self.assertNotIn('memlite_audit_locator', row)
        self.assertNotIn('image3', row['samples'])
        self.assertEqual(row['samples']['template'].count('_image_!>'), 3)
        out = bench.collate(rows, range(16), 'cpu')
        self.assertEqual(tuple(out['action'].shape), (16, 32, 27))
        self.assertTrue(torch.equal(out['action_dim_is_pad'][0], batches[0]['action_dim_is_pad'][0]))

    def test_real_controls_must_not_be_masked(self):
        batches = fixtures()
        batches[0]['action_dim_is_pad'] = batches[0]['action_dim_is_pad'].clone()
        batches[0]['action_dim_is_pad'][:, 20] = True
        with self.assertRaisesRegex(ValueError, 'base/trunk'):
            bench.single_frame_pool(batches)

    def test_wrong_camera_order_rejected(self):
        batches = fixtures()
        batches[0]['pixel_values'] = dict(reversed(list(batches[0]['pixel_values'].items())))
        with self.assertRaisesRegex(ValueError, 'camera-major'):
            bench.single_frame_pool(batches)

    def test_audit_payload_and_semantic_mismatch_rejected(self):
        sample = fixtures()[0]['samples'][0]
        validate_embedded_model_projection(sample)
        for key, value in [('episode_index', 0), ('active_skills_text', 'unrelated intent')]:
            bad = dict(sample, **{key: value})
            with self.assertRaises(ValueError):
                validate_embedded_model_projection(bad)


class StrictVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, *, inputs_embeds, kv_cache):
        return self.q_proj(inputs_embeds), kv_cache


def toy_policy():
    policy = nn.Module()
    policy.model = nn.Module()
    policy.model.vlm = inject_vlm_lora(StrictVLM(), VLMloraConfig(enabled=True, r=2, alpha=2, target_modules=('q_proj',)))
    return policy


class LoRAContract(unittest.TestCase):
    def test_custom_forward_and_gradient(self):
        policy = toy_policy()
        cache = SimpleNamespace(marker=1)
        output, returned = policy.model.vlm(inputs_embeds=torch.ones(1, 3, 4), kv_cache=cache)
        self.assertIs(returned, cache)
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for n, p in policy.named_parameters() if 'lora_B' in n))
        self.assertTrue(all(p.grad is None for n, p in policy.named_parameters() if 'lora_' not in n))

    def test_adapter_roundtrip_and_partial_restore_rejected(self):
        policy = toy_policy()
        state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        target = toy_policy()
        receipt = restore_lora_state(target, state)
        self.assertEqual(receipt['adapter_load_mode'], 'resume')
        self.assertEqual(receipt['restored'], 2)
        for key, value in target.state_dict().items():
            if 'lora_' in key:
                self.assertTrue(torch.equal(value, state[key]))
        incomplete = {k: v for k, v in state.items() if 'lora_B' not in k}
        with self.assertRaisesRegex(RuntimeError, 'coverage'):
            restore_lora_state(target, incomplete)


if __name__ == '__main__':
    unittest.main()
