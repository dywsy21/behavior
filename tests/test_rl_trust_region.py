from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g05.rl.trust_region import bounded_adam_step,cpu_copy


class TrustRegionTests(unittest.TestCase):
    def test_dense_recipe_checks_mean_and_tail_without_relaxing_rollback(self):
        p=torch.nn.Parameter(torch.tensor([1.]))
        opt=torch.optim.AdamW([p],lr=.1,weight_decay=0);p.grad=torch.ones_like(p)
        rejected=bounded_adam_step([p],opt,lambda:dict(mean_kl=.03,max_kl=.06),
                                   limit=.1,mean_limit=.02,scales=(1.,))
        self.assertFalse(rejected['accepted']);self.assertEqual(float(p),1.)
        rejected=bounded_adam_step([p],opt,lambda:dict(mean_kl=.01,max_kl=.11),
                                   limit=.1,mean_limit=.02,scales=(1.,))
        self.assertFalse(rejected['accepted']);self.assertFalse(opt.state)
        accepted=bounded_adam_step([p],opt,lambda:dict(mean_kl=.01,max_kl=.06),
                                   limit=.1,mean_limit=.02,scales=(1.,))
        self.assertTrue(accepted['accepted']);self.assertEqual(float(opt.state[p]['step']),1.)

    def test_retries_same_gradient_not_accumulated_adam_steps(self):
        p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
        opt=torch.optim.AdamW([p],lr=.1,weight_decay=0)
        p.grad=torch.ones_like(p)
        result=bounded_adam_step([p],opt,lambda:dict(max_kl=float((p-1).square().sum())*10000),limit=.02)
        self.assertTrue(result['accepted']); self.assertEqual(result['scale'],.01)
        self.assertAlmostEqual(float(p),.999,places=8)
        self.assertEqual(float(opt.state[p]['step']),1.)
        self.assertAlmostEqual(float(opt.state[p]['exp_avg']),.1)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],.001)

    def test_all_rejected_restores_nonempty_moments_lr_weights_and_gradient(self):
        p=torch.nn.Parameter(torch.tensor([1.]))
        opt=torch.optim.AdamW([p],lr=.1,weight_decay=0)
        p.grad=torch.ones_like(p); opt.step()
        weight=p.detach().clone(); state=cpu_copy(opt.state[p]); p.grad=torch.tensor([3.])
        result=bounded_adam_step([p],opt,lambda:dict(max_kl=1.),scales=(1.,.1))
        self.assertFalse(result['accepted']); self.assertTrue(torch.equal(p,weight))
        for key,value in state.items(): self.assertTrue(torch.equal(opt.state[p][key],value))
        self.assertEqual(opt.param_groups[0]['lr'],.1)
        self.assertTrue(torch.equal(p.grad,torch.tensor([3.])))

    def test_exception_restores_and_bad_surrogate_rejects(self):
        p=torch.nn.Parameter(torch.tensor([1.])); opt=torch.optim.AdamW([p],lr=.1)
        p.grad=torch.ones_like(p)
        result=bounded_adam_step([p],opt,lambda:dict(max_kl=0.,surrogate_improved=False),scales=(1.,))
        self.assertFalse(result['accepted']); self.assertEqual(float(p),1.); self.assertFalse(opt.state)
        def fail(): raise RuntimeError('evaluator failed')
        with self.assertRaisesRegex(RuntimeError,'evaluator failed'):
            bounded_adam_step([p],opt,fail)
        self.assertEqual(float(p),1.); self.assertFalse(opt.state)


if __name__=='__main__': unittest.main()
