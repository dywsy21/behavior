"""Bounded continuation of the SAME on-policy batch after an exact rollback.

This is PPO replay of retained old paths, not an offline-RL dataset, no new
samplers and no repeated critic warmup. The original actor never accepted a step.
"""
import json
from pathlib import Path
import signal
import traceback
import numpy as np
import torch
from common import OUT,PARENT_SHA,save,sha
from learner import Experiment
from g05.rl.g05_adapter import move
from g05.rl.flow_ppo import ppo_objective
from g05.rl.trust_region import bounded_adam_step,cpu_copy


class Continuation(Experiment):
    def run(self):
        try:
            self.load(); self.prepare_bc()
            ck=self.manifest['source_checkpoint']; roll=self.manifest['source_rollout']
            if sha(ck['path'])!=ck['sha256'] or sha(roll['path'])!=roll['sha256']:
                raise ValueError('Retained training artifacts changed')
            saved=torch.load(ck['path'],map_location='cpu',weights_only=False)
            if saved['parent_sha256']!=PARENT_SHA or saved['actor_updates']!=0:
                raise ValueError('Retained batch is not on the unchanged parent policy')
            # Confirm full rollback, not just the counter in a JSON receipt.
            for name,value in self.policy.model.action_expert.state_dict().items():
                if not torch.equal(value.detach().cpu(),saved['action_expert'][name]):
                    raise ValueError('Rolled-back AE differs from original 50k: '+name)
            self.noise.load_state_dict(saved['noise'],strict=True)
            self.critic.load_state_dict(saved['critic'],strict=True)
            self.actor_optimizer.load_state_dict(saved['actor_optimizer'])
            self.critic_optimizer.load_state_dict(saved['critic_optimizer'])
            self.critic_updates=saved['critic_updates']
            if self.actor_optimizer.state: raise ValueError('Initial actor Adam rollback was incomplete')
            torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
            data=torch.load(roll['path'],map_location='cpu',weights_only=False)
            rows=data['rows']; advantages=data['advantages']
            if len(rows)!=39 or sum(len(r['rewards']) for r in rows)!=620:
                raise ValueError('Unexpected retained batch')
            if sum(sum(r['rewards']) for r in rows)!=1:
                raise ValueError('Expected one official TRAIN success')
            self.successes=1
            parameters=self.actor_parameters+list(self.noise.parameters())
            with torch.no_grad():
                errors=[]; kls=[]
                for row in rows:
                    self.timecheck()
                    lp,kl=self.adapter.score(move(row['context'],'cuda:0'),move(row['trace'],'cuda:0'))
                    errors.append(float((lp-row['trace']['old_logp'].to('cuda:0')).abs())); kls.append(float(kl))
            if max(errors)>.002 or max(map(abs,kls))>1e-6:
                raise ValueError('Retained old-policy path identity failed')
            save(OUT/'retained_batch_gate.json',dict(samples=39,policy_controls=620,
                old_path_max_logp_error=max(errors),old_path_max_abs_kl=max(map(abs,kls)),
                parent_ae_identical=True,actor_adam_initial_state_empty=True,extra_controls=0))
            del saved
            advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
            self.phase='offline_updating'; self.status()
            stopped=False
            for epoch in range(2):
                indices=list(range(len(rows))); self.rng.shuffle(indices)
                for start in range(0,len(indices),8):
                    self.timecheck(); group=indices[start:start+8]
                    self.actor_optimizer.zero_grad(set_to_none=True); before=[]
                    for i in group:
                        lp,kl=self.adapter.score(move(rows[i]['context'],'cuda:0'),move(rows[i]['trace'],'cuda:0'))
                        loss=ppo_objective(lp,rows[i]['trace']['old_logp'].to('cuda:0'),advantages[i].to('cuda:0'))/len(group)
                        loss.backward(); before.append(float(loss.detach()))
                    bc,details=self.auxiliary_bc(self.actor_updates); (.1*bc).backward()
                    grad=float(torch.nn.utils.clip_grad_norm_(parameters,.5,error_if_nonfinite=True))
                    if any(p.grad is not None for p in self.policy.parameters() if not p.requires_grad):
                        raise ValueError('Frozen gradients in PPO update')
                    original=cpu_copy(self.policy.model.action_expert.state_dict())

                    def evaluate():
                        kls=[]; deltas=[]; surrogate=0.
                        for i,row in enumerate(rows):
                            self.timecheck()
                            lp,kl=self.adapter.score(move(row['context'],'cuda:0'),move(row['trace'],'cuda:0'))
                            delta=float(lp-row['trace']['old_logp'].to('cuda:0'))
                            kls.append(float(kl)); deltas.append(abs(delta))
                            if i in group and abs(delta)<=60:
                                surrogate+=float(ppo_objective(lp,row['trace']['old_logp'].to('cuda:0'),
                                    advantages[i].to('cuda:0')))/len(group)
                        safe=max(deltas)<=60
                        return dict(mean_kl=float(np.mean(kls)),max_kl=max(kls),max_abs_logratio=max(deltas),
                            surrogate=surrogate if safe else None,before_surrogate=sum(before),
                            surrogate_improved=bool(safe and surrogate<=sum(before)+1e-6))

                    def trial(row):
                        self.log.write(json.dumps(dict(event='backtracking_trial',epoch=epoch,
                            update=self.actor_updates,**row),allow_nan=False)+'\n')
                        self.status(last_trial=row)
                    result=bounded_adam_step(parameters,self.actor_optimizer,evaluate,on_trial=trial)
                    if not result['accepted']:
                        stopped=True; break
                    current=self.policy.model.action_expert.state_dict()
                    changes=[(value.detach().cpu()-original[name]).abs() for name,value in current.items()]
                    max_change=max(float(x.max()) for x in changes)
                    changed=sum(int(torch.count_nonzero(x)) for x in changes)
                    if not changed: raise ValueError('Accepted candidate did not change any AE parameter')
                    self.actor_updates+=1
                    self.log.write(json.dumps(dict(event='actor_update',update=self.actor_updates,epoch=epoch,
                        bc=details,
                        gradient_norm=grad,ae_changed_parameters=changed,ae_max_abs_change=max_change,
                        **result['trials'][-1]),allow_nan=False)+'\n')
                    self.status(ae_changed_parameters=changed,ae_max_abs_change=max_change)
                    del original,changes
                    # A safety checkpoint after every accepted step; not a
                    # promise that all ten minibatches will pass the same gate.
                    self.batches+=1; self.checkpoint()
                    if self.actor_updates>=10: stopped=True; break
                if stopped: break
            self.phase='completed'; self.status(extra_environment_controls=0,
                trust_region_exhausted=stopped and self.actor_updates<10,full_task_success_rate_claim=False)
        except BaseException as error:
            self.phase='failed'; self.status(error=repr(error))
            save(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
            raise
        finally: self.log.close()


if __name__=='__main__':
    def stop(signum,frame): raise SystemExit(f'Owned update stopped by signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    Continuation().run()
