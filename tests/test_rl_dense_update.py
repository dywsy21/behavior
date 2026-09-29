"""Execute the actual learner.update on a tiny CPU policy, not a config mock."""
import io
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from test_rl_method_lifecycle import base
from g05.rl.flow_ppo import ppo_objective
from g05.rl.trust_region import bounded_adam_step
from g05.rl.dense_recipe import REWARD


class DenseUpdateTests(unittest.TestCase):
    def experiment(self, dense):
        e=base.__new__(base)
        ae=torch.nn.Linear(1,1,bias=False);ae.weight.data.zero_()
        e.policy=SimpleNamespace(model=SimpleNamespace(action_expert=ae))
        e.actor_parameters=list(ae.parameters());e.noise=torch.nn.Linear(1,1,bias=False)
        e.noise.weight.data.zero_();e.critic=torch.nn.Linear(1,1)
        e.actor_optimizer=torch.optim.AdamW([
            dict(params=ae.parameters(),lr=1e-7),dict(params=e.noise.parameters(),lr=1e-6)],weight_decay=0)
        e.critic_optimizer=torch.optim.AdamW(e.critic.parameters(),lr=1e-4,weight_decay=0)
        e.actor_updates=94;e.critic_updates=16;e.batches=0;e.log=io.StringIO();e.rng=random.Random(1)
        e.status=lambda **kw:None;e.timecheck=lambda:None
        e.manifest=dict(entry='method_dense',reward=REWARD if dense else None,clip=.1,
            target_path_kl=.1,target_mean_path_kl=.02,ppo_epochs=4,critic_steps_per_batch=4,
            bounded_backtracking=True,reset_candidate_lr_each_minibatch=True,
            learning_rates=dict(action_expert=1e-7,noise=1e-6),backtracking_scales=[1.,.5],
            max_actor_updates=2094,bc_weight=.1)
        e.adapter=SimpleNamespace(score=lambda context,trace:(
            (ae.weight.sum()+e.noise.weight.sum()*.001)*context['sign'],
            (ae.weight-trace['weight']).square().sum()*.001))
        e.auxiliary_bc=lambda index:(ae.weight.square().sum(),{'fm_loss':float(ae.weight.detach().square().sum())})
        rows=[dict(split='train',context={'feature':torch.zeros(1,1),'sign':float(s)},
                   trace={'old_logp':torch.tensor(0.),'weight':torch.zeros(1,1)},
                   rewards=[.01,-.01]) for s in [-1,1]*4]
        return e,rows

    def execute(self, dense):
        e,rows=self.experiment(dense);clips=[];candidates=[];limits=[]
        original_to=torch.Tensor.to
        def cpu_to(tensor,*args,**kwargs):
            if args and args[0]=='cuda:0':args=('cpu',*args[1:])
            return original_to(tensor,*args,**kwargs)
        def objective(*args,**kwargs):
            clips.append(kwargs['clip']);return ppo_objective(*args,**kwargs)
        def bounded(parameters,optimizer,evaluate,**kwargs):
            candidates.append([g['lr'] for g in optimizer.param_groups])
            limits.append((kwargs['limit'],kwargs['mean_limit']))
            kwargs['scales']=(.5,)  # force a reduction to test NEXT minibatch's reset
            return bounded_adam_step(parameters,optimizer,evaluate,**kwargs)
        with patch.object(torch.Tensor,'to',cpu_to),patch.dict(base.update.__globals__,{
                'ppo_objective':objective,'bounded_adam_step':bounded,'move':lambda x,device:x}):
            e.update(rows,torch.tensor([-1.,1.]*4),torch.zeros(8))
        return e,clips,candidates,limits

    def test_dense_zero_sum_batch_updates_and_uses_all_new_optimizer_settings(self):
        e,clips,candidates,limits=self.execute(True)
        self.assertEqual((e.actor_updates,e.critic_updates),(98,20))
        self.assertTrue(clips and all(x==.1 for x in clips))
        self.assertEqual(candidates,[[1e-7,1e-6]]*4)
        self.assertEqual(limits,[(.1,.02)]*4)
        self.assertGreater(float(e.policy.model.action_expert.weight),0)

    def test_legacy_sparse_zero_sum_batch_still_skips_actor(self):
        e,clips,candidates,limits=self.execute(False)
        self.assertEqual(e.actor_updates,94)
        self.assertEqual(e.critic_updates,20)
        self.assertFalse(clips or candidates or limits)


if __name__=='__main__':unittest.main()
