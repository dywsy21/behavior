"""CPU tensor regressions for the restored B-only objective; no model allocation."""
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from g05.models.g05.g05_policy_memlite_planner_outcome import G05PolicyMEMLitePlannerOutcome as Policy
from g05.models.g05.helpers.ar_helper import ARHelper


def helper():
    return ARHelper(OmegaConf.create(dict(ce_weight=1., vocab_size=5, use_fused_ce=False,
        ce_z_loss_scale=0., block_wise_autoregressive=False, bos_blk_id=None,
        eos_blk_id=None, block_size=None)))


class HighContract(unittest.TestCase):
    def test_unaudited_history_rejected_before_allocating_policy(self):
        for steps in (6, None, 0, True):
            with self.assertRaisesRegex(ValueError, 'single-frame'):
                Policy(num_obs_steps=steps)

    def test_left_and_right_padding_keep_prefix_boundary(self):
        for full, expected in [([[0, 0, 11, 12, 31, 32], [21, 22, 23, 41, 42, 43]], [3, 2]),
                               ([[11, 12, 31, 32, 0, 0], [21, 22, 23, 41, 42, 43]], [1, 2])]:
            ids = torch.tensor(full)
            context = torch.tensor([[0, 11, 12], [21, 22, 23]])
            self.assertEqual(Policy._context_positions(ids, ids != 0, context, context != 0), expected)
            bad = context.clone()
            bad[0, -1] = 99
            with self.assertRaisesRegex(RuntimeError, 'prefix mismatch'):
                Policy._context_positions(ids, ids != 0, bad, bad != 0)

    def test_weighted_ce_and_gradients_equal_explicit_shift_mask(self):
        torch.manual_seed(73)
        hidden = torch.randn(2, 4, 5, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2], [-100, 3, 4, -100]])
        weights = torch.tensor([[0., 0., 1., .25], [0., .25, 1., 0.]])
        model = SimpleNamespace(vlm=SimpleNamespace(decode=lambda x: x))
        measured, _ = helper().cal_ce_loss(hidden, labels, model, loss_token_weights=weights)
        valid = labels[:, 1:].reshape(-1) != -100
        raw = F.cross_entropy(hidden[:, :-1].reshape(-1, 5)[valid],
                              labels[:, 1:].reshape(-1)[valid], reduction='none')
        selected = weights[:, 1:].reshape(-1)[valid]
        expected = (raw * selected).sum() / selected.sum()
        self.assertTrue(torch.equal(measured, expected))
        self.assertTrue(torch.equal(torch.autograd.grad(measured, hidden, retain_graph=True)[0],
                                    torch.autograd.grad(expected, hidden)[0]))

    def test_all_ones_retain_original_ce_and_masked_tokens_stay_inert(self):
        hidden = torch.randn(1, 3, 5, requires_grad=True)
        labels = torch.tensor([[-100, 2, -100]])
        model = SimpleNamespace(vlm=SimpleNamespace(decode=lambda x: x))
        objective = helper()
        original, _ = objective.cal_ce_loss(hidden, labels, model)
        weighted, _ = objective.cal_ce_loss(hidden, labels, model,
                                             loss_token_weights=torch.tensor([[1e6, 1., 1e6]]))
        self.assertTrue(torch.equal(original, weighted))
        self.assertEqual(objective._last_ce_cache['token_weights'].tolist(), [1.])


if __name__ == '__main__':
    unittest.main()
