import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("runtime", Path(__file__).resolve().parents[1] /
    "src/g05/utils/training/stage1_runtime.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RuntimeTests(unittest.TestCase):
    def test_budget_does_not_reset_on_resume(self):
        clock = [10.]
        budget = m.WallBudget(100, 80, clock=lambda: clock[0])
        clock[0] = 31.
        self.assertTrue(budget.exhausted())
        self.assertEqual(budget.consumed(), 101.)

    def test_warmup_cosine_is_resume_invariant(self):
        self.assertEqual(m.lr_multiplier(0, 1000, 50000), .001)
        self.assertEqual(m.lr_multiplier(999, 1000, 50000), 1.)
        self.assertAlmostEqual(m.lr_multiplier(50000, 1000, 50000), .1)

    def test_atomic_json(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "receipt.json"
            m.atomic_json(p, dict(step=1))
            m.atomic_json(p, dict(step=2))
            self.assertEqual(json.loads(p.read_text()), dict(step=2))
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_ddp_global_denominator_not_mean_of_means(self):
        import torch
        # Two rank/micro means would be (10 + 1)/2 = 5.5. Correct weighted
        # objective is (10 + 9*1)/10 = 1.9, including its parameter gradient.
        p = torch.nn.Parameter(torch.tensor(1.))
        p.grad = torch.tensor((10. + 9.) / 2.)
        m.normalize_ddp_gradients([p], 10., 2)
        self.assertAlmostEqual(p.grad.item(), 1.9, places=6)
        with self.assertRaises(ValueError):
            m.normalize_ddp_gradients([p], 0, 2)


if __name__ == "__main__":
    unittest.main()
