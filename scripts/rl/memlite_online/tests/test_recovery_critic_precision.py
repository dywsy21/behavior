import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from direct_a4_flow import ValueHead


class CriticPrecisionTests(unittest.TestCase):
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
