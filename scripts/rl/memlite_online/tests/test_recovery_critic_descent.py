from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from critic_descent import guarded_critic_step,save_critic_update_audit,fit_cached_critic


class FlatCritic(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.1]))

    def forward(self,x):return x[:,0]*self.weight[0]


class CriticDescentTests(unittest.TestCase):
    def setUp(self):
        self.head=FlatCritic();self.x=torch.ones(4,1);self.y=torch.zeros(4)

    def test_overshoot_backtracks_parameters_and_adam_counts(self):
        opt=torch.optim.AdamW(self.head.parameters(),lr=1.,weight_decay=0.)
        (.5*(self.head(self.x)-self.y).square().mean()).backward()
        result=guarded_critic_step(self.head,opt,self.x,self.y,learning_rate=1.)
        self.assertTrue(result['critic_updated'])
        self.assertGreater(len(result['critic_backtracking']),1)
        self.assertLess(result['critic_half_mse_after'],result['critic_half_mse_before'])
        self.assertEqual(float(opt.state[self.head.weight]['step']),1.)
        self.assertEqual(opt.param_groups[0]['lr'],1.)

    def test_non_descent_restores_exact_existing_adam_state(self):
        opt=torch.optim.AdamW(self.head.parameters(),lr=.01,weight_decay=0.)
        self.head.weight.grad=torch.ones_like(self.head.weight);opt.step();opt.zero_grad()
        saved=deepcopy(opt.state_dict());before=self.head.weight.detach().clone()
        # Negative gradient moves the positive scalar further from zero.
        opt.state[self.head.weight]['exp_avg'].fill_(-10.)
        saved=deepcopy(opt.state_dict());self.head.weight.grad=-torch.ones_like(self.head.weight)
        result=guarded_critic_step(self.head,opt,self.x,self.y,learning_rate=.01)
        self.assertFalse(result['critic_updated']);self.assertEqual(result['accepted_critic_lr'],0.)
        self.assertTrue(torch.equal(self.head.weight,before))
        for key,value in saved['state'][0].items():self.assertTrue(torch.equal(value,opt.state_dict()['state'][0][key]))
        self.assertEqual(saved['param_groups'],opt.state_dict()['param_groups'])

    def test_gradient_inputs_and_optimizer_boundary_are_checked(self):
        opt=torch.optim.AdamW(self.head.parameters(),lr=.01)
        self.head.weight.grad=torch.ones_like(self.head.weight)
        with self.assertRaises(ValueError):guarded_critic_step(self.head,opt,self.x.requires_grad_(),self.y,learning_rate=.01)
        extra=torch.nn.Parameter(torch.tensor(0.));opt=torch.optim.AdamW([self.head.weight,extra],lr=.01)
        with self.assertRaises(ValueError):guarded_critic_step(self.head,opt,self.x.detach(),self.y,learning_rate=.01)

    def test_optional_restart_recovers_a_real_downhill_gradient_from_stale_momentum(self):
        opt=torch.optim.AdamW(self.head.parameters(),lr=.01,weight_decay=0.)
        self.head.weight.grad=-torch.ones_like(self.head.weight);opt.step();opt.zero_grad()
        opt.state[self.head.weight]['exp_avg'].fill_(-10.)
        old_step=float(opt.state[self.head.weight]['step'])
        (.5*(self.head(self.x)-self.y).square().mean()).backward()
        result=guarded_critic_step(self.head,opt,self.x,self.y,learning_rate=.01,restart_stale_momentum=True)
        self.assertTrue(result['critic_updated']);self.assertTrue(result['critic_first_moment_restarted'])
        self.assertEqual(len(result['critic_backtracking']),8)
        self.assertLess(result['critic_half_mse_after'],result['critic_half_mse_before'])
        self.assertEqual(float(opt.state[self.head.weight]['step']),old_step+1)
        self.assertGreater(float(opt.state[self.head.weight]['exp_avg']),0.)

    def test_failed_restart_rolls_back_moments_and_parameters(self):
        opt=torch.optim.AdamW(self.head.parameters(),lr=.01,weight_decay=0.)
        self.head.weight.grad=-torch.ones_like(self.head.weight);opt.step();opt.zero_grad()
        before=self.head.weight.detach().clone();saved=deepcopy(opt.state_dict())
        # Intentionally invalid uphill gradient: not even a restart may
        # publish a step unless the ACTUAL objective strictly decreases.
        self.head.weight.grad=-torch.ones_like(self.head.weight)
        result=guarded_critic_step(self.head,opt,self.x,self.y,learning_rate=.01,restart_stale_momentum=True)
        self.assertFalse(result['critic_updated']);self.assertFalse(result['critic_first_moment_restarted'])
        self.assertTrue(torch.equal(self.head.weight,before))
        for key,value in saved['state'][0].items():self.assertTrue(torch.equal(value,opt.state_dict()['state'][0][key]))
        self.assertEqual(saved['param_groups'],opt.state_dict()['param_groups'])

    def test_exact_preupdate_replay_evidence_not_an_actor_checkpoint(self):
        opt=torch.optim.AdamW(self.head.parameters(),lr=.01,weight_decay=0.)
        (.5*(self.head(self.x)-self.y).square().mean()).backward()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'critic.pt';ids=[dict(experience_id=i) for i in range(4)]
            save_critic_update_audit(path,self.head,opt,self.x,self.y,ids)
            saved=torch.load(path,weights_only=True)
            self.assertFalse(saved['actor_checkpoint']);self.assertFalse(saved['training_admission'])
            self.assertEqual(saved['identities'],ids)
            torch.testing.assert_close(saved['gradients']['weight'],self.head.weight.grad)
            torch.testing.assert_close(saved['features'],self.x)
            with self.assertRaises(FileExistsError):save_critic_update_audit(path,self.head,opt,self.x,self.y,ids)

    def test_default_cached_fit_is_exact_old_step(self):
        first=deepcopy(self.head);second=deepcopy(self.head)
        a=torch.optim.AdamW(first.parameters(),lr=.01,weight_decay=0.)
        b=torch.optim.AdamW(second.parameters(),lr=.01,weight_decay=0.)
        for head in (first,second):(.25*(head(self.x)-self.y).square().mean()).backward()
        old=guarded_critic_step(first,a,self.x,self.y,learning_rate=.01)
        new=fit_cached_critic(second,b,self.x,self.y,learning_rate=.01)
        self.assertEqual(old,new);self.assertTrue(torch.equal(first.weight,second.weight))
        for key,value in a.state_dict()['state'][0].items():
            self.assertTrue(torch.equal(value,b.state_dict()['state'][0][key]))

    def test_multiple_critic_steps_keep_features_and_targets_detached_unchanged(self):
        opt=torch.optim.AdamW(self.head.parameters(),lr=.01,weight_decay=0.)
        (.25*(self.head(self.x)-self.y).square().mean()).backward()
        before_x=self.x.clone();before_y=self.y.clone()
        result=fit_cached_critic(self.head,opt,self.x,self.y,learning_rate=.01,steps=5,
            restart_stale_momentum=True)
        self.assertEqual(result['critic_fit_optimizer_steps'],5)
        self.assertEqual(float(opt.state[self.head.weight]['step']),5.)
        self.assertEqual(result['actor_steps_during_cached_fit'],0)
        self.assertFalse(result['targets_recomputed_during_cached_fit'])
        self.assertTrue(torch.equal(self.x,before_x));self.assertTrue(torch.equal(self.y,before_y))
        self.assertTrue(all(r['critic_half_mse_after']<r['critic_half_mse_before'] for r in result['critic_fit_curve']))
        for value in (0,True,1.5):
            with self.assertRaises(ValueError):fit_cached_critic(self.head,opt,self.x,self.y,learning_rate=.01,steps=value)

    def test_cached_fit_stops_after_a_skipped_search(self):
        opt=torch.optim.AdamW(self.head.parameters(),lr=.01,weight_decay=0.)
        self.head.weight.grad=-torch.ones_like(self.head.weight)
        before=deepcopy(opt.state_dict());weight=self.head.weight.detach().clone()
        result=fit_cached_critic(self.head,opt,self.x,self.y,learning_rate=.01,steps=5)
        self.assertEqual(result['critic_fit_optimizer_steps'],0)
        self.assertEqual(len(result['critic_fit_curve']),1)
        self.assertTrue(torch.equal(self.head.weight,weight))
        self.assertEqual(before,opt.state_dict())


if __name__=='__main__':unittest.main()
