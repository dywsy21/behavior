from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from direct_a4_flow import A4DirectPPO


class ReadonlyPrefixTests(unittest.TestCase):
    def make(self):
        trainer = A4DirectPPO.__new__(A4DirectPPO)
        trainer.reward_protocol = 'skill_aligned_v1'
        trainer.policy = SimpleNamespace(prefill=Mock(return_value=SimpleNamespace(
            last_hidden=torch.arange(8, dtype=torch.float32).reshape(1, 8))))
        trainer._ensure_critic = Mock(side_effect=lambda _: torch.rand(3))
        batch = dict(samples=[dict(memlite_branch='low', template='target-free<EOC>')],
                     pixel_values={})
        return trainer, batch

    def test_readonly_does_not_initialize_unused_critic_or_consume_its_rng(self):
        trainer, batch = self.make()
        before = torch.random.get_rng_state().clone()
        state, features = trainer._prefix(batch, initialize_critic=False)
        trainer._ensure_critic.assert_not_called()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(features.requires_grad)
        self.assertEqual(features.shape, (1, 8))

    def test_existing_train_prefix_keeps_critic_initialization(self):
        trainer, batch = self.make()
        trainer._prefix(batch)
        trainer._ensure_critic.assert_called_once()


if __name__ == '__main__':
    unittest.main()
