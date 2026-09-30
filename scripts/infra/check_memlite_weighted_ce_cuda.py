"""Bounded small-tensor check of raw FLCE versus the exact planner objective.

No policy, checkpoint, dataset, optimizer, or shared environment changes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'src'))
    import torch
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from g05.models.g05.helpers.ar_helper import ARHelper
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(8, 16, device='cuda', requires_grad=True)
    w = torch.randn(37, 16, device='cuda', requires_grad=True)
    y = torch.tensor([1, 3, 7, 9, 2, 4, 6, 8], device='cuda')
    weights = torch.tensor([1., .25, 1., .25, 1., .25, 1., .25], device='cuda')
    def reference(values):
        per_token = torch.nn.functional.cross_entropy(x @ w.t(), y, reduction='none')
        return (per_token * values).sum() / values.sum()
    def error(actual, expected):
        return dict(max_abs=float((actual-expected).abs().max()),
                    relative_l2=float((actual-expected).norm()/expected.norm().clamp_min(1e-12)))
    def compare(kind, values):
        expected = reference(values)
        expected_grad = torch.autograd.grad(expected, (x, w))
        if kind == 'raw_liger':
            raw = LigerFusedLinearCrossEntropyLoss(reduction='none', return_token_accuracy=True)(w, x, y)
            actual = (raw.loss * values).sum()/values.sum()
            backend = 'raw_liger_none'
        else:
            config = OmegaConf.create(dict(ce_weight=1., vocab_size=37, use_fused_ce=True,
                ce_z_loss_scale=0., block_wise_autoregressive=False,
                bos_blk_id=None, eos_blk_id=None, block_size=None))
            helper = ARHelper(config)
            model = SimpleNamespace(vlm=SimpleNamespace(output_proj=SimpleNamespace(weight=w),
                                                        decode=lambda hidden: hidden @ w.t()))
            # Nine hidden positions: target y[i] is predicted from hidden i.
            hidden = torch.cat((x, x.new_zeros(1, 16)), dim=0).unsqueeze(0)
            labels = torch.cat((y.new_tensor([-100]), y)).unsqueeze(0)
            target_weights = torch.cat((values.new_zeros(1), values)).unsqueeze(0)
            actual, _ = helper.cal_ce_loss(hidden, labels, model, loss_token_weights=target_weights)
            backend = helper._last_ce_cache.get('ce_backend')
        actual_grad = torch.autograd.grad(actual, (x, w))
        return dict(kind=kind, uniform=bool((values == values[0]).all()), backend=backend,
                    loss_error=float((actual-expected).abs()),
                    input_gradient=error(actual_grad[0], expected_grad[0]),
                    weight_gradient=error(actual_grad[1], expected_grad[1]),
                    gradients_match=all(torch.allclose(a, b, atol=2e-5, rtol=2e-5)
                                        for a,b in zip(actual_grad, expected_grad)))
    rows = [compare(kind, values) for kind in ('raw_liger', 'policy_helper')
            for values in (torch.ones_like(weights), weights)]
    if not all(row['gradients_match'] for row in rows if row['kind']=='policy_helper'):
        raise RuntimeError('Planner CE helper still has incorrect gradients')
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='complete', rows=rows, actual_policy_updates=0,
                  commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
