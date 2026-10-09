import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from direct_a4_flow import ValueHead, A4DirectPPO


class CriticPrecisionTests(unittest.TestCase):
    def test_short_skill_resume_checks_base_identity_before_loading_weights(self):
        from unittest.mock import patch
        trainer=A4DirectPPO.__new__(A4DirectPPO)
        trainer.reward_protocol='skill_aligned_v1';trainer.transition_std=.03;trainer.clip_ratio=.2
        payload=dict(kind='a4_direct_flow_ppo_trainable_state_v1',reward_protocol='skill_aligned_v1',
            transition_std=.03,clip_ratio=.2,parent_sha256='a'*64,updates=3)
        with patch('direct_a4_flow.torch.load',return_value=payload):
            for expected in (dict(parent_sha256='b'*64),dict(updates=4),dict(stats_sha256='c'*64)):
                with self.assertRaisesRegex(ValueError,'Resume base weights'):
                    trainer.load_checkpoint(Path('structural-unit-only.pt'),expected_bindings=expected)

    def test_rollout_and_bootstrap_ignore_outer_autocast(self):
        torch.manual_seed(42);head=ValueHead(32);features=torch.randn(4,32)
        with torch.no_grad():
            with torch.autocast('cpu',dtype=torch.bfloat16):sampled=head(features)
            bootstrap=head(features)
        self.assertEqual(sampled.dtype,torch.float32)
        self.assertTrue(torch.equal(sampled,bootstrap))
        self.assertTrue(all(math.isclose(float(a),float(b),abs_tol=1e-5,rel_tol=1e-5)
                            for a,b in zip(sampled,bootstrap)))
        with torch.autocast('cpu',dtype=torch.bfloat16):loss=head(features).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))


if __name__=='__main__':unittest.main()
