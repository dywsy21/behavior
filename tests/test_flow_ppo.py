import math
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from g05.rl.flow_ppo import (gaussian_logp, gaussian_kl, ppo_objective, gae,
                             NoiseHead, ControlBudget, VALID_DIMS)


class FlowPPOTests(unittest.TestCase):
    def test_entire_path_not_only_executed_half_and_padding_excluded(self):
        x = torch.zeros(10,32,27)
        s = torch.ones_like(x)
        expected = -7360*.5*math.log(2*math.pi)
        self.assertAlmostEqual(float(gaussian_logp(x,x,s)), expected, delta=.002)
        padded=x.clone(); padded[...,7]=10000
        self.assertEqual(gaussian_logp(padded,x,s),gaussian_logp(x,x,s))
        tail=x.clone(); tail[:,16:,0]=1
        self.assertAlmostEqual(float(gaussian_logp(tail,x,s)-gaussian_logp(x,x,s)), -80., delta=.002)

    def test_ratio_identity_gradient_and_kl(self):
        x=torch.randn(10,32,27); mean=torch.zeros_like(x,requires_grad=True)
        std=torch.full_like(x,.03)
        old=gaussian_logp(x,mean,std).detach()
        loss=ppo_objective(gaussian_logp(x,mean,std),old,torch.tensor(1.))
        self.assertEqual(float(loss),-1.)
        loss.backward()
        self.assertGreater(float(mean.grad[...,VALID_DIMS].abs().sum()),0)
        self.assertEqual(float(mean.grad[...,7].abs().sum()),0)
        self.assertEqual(float(gaussian_kl(mean.detach(),std,mean,std)),0)

    def test_noise_bounds_initialization_and_public_feature_detach(self):
        head=NoiseHead(8); feat=torch.randn(1,8,requires_grad=True)
        std=head(feat,torch.randn(1,32,27),torch.ones(1))
        self.assertTrue(torch.allclose(std,torch.full_like(std,.01),atol=1e-8))
        std.sum().backward(); self.assertIsNone(feat.grad)
        self.assertGreater(float(head.net[-1].bias.grad.abs().sum()),0)

    def test_terminated_vs_truncated_and_reset_boundary(self):
        a=dict(rewards=[0,1], value=.2,next_value=.9,episode=1,terminated=True,truncated=False)
        b=dict(a,terminated=False,truncated=True,episode=2)
        adv,ret=gae([a,b],gamma=.5)
        self.assertAlmostEqual(float(ret[0]),.5)
        self.assertAlmostEqual(float(ret[1]),.725,places=6)
        self.assertAlmostEqual(float(adv[0]),.3,places=6)

    def test_budget_parallel_reservations_and_early_termination(self):
        b=ControlBudget(32); b.reserve(16); b.reserve(16)
        with self.assertRaises(RuntimeError): b.reserve(1)
        b.finish(16,5); b.finish(16,16)
        self.assertEqual((b.used,b.remaining,b.pending),(21,11,0))


if __name__ == '__main__': unittest.main()
