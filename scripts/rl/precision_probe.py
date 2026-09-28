"""Counterfactual numerical probe only; restores parent and saves NO actor."""
import json
import signal
import traceback
from collections import Counter
import torch
from common import OUT,save,sha
from learner import Experiment
from g05.rl.g05_adapter import move
from g05.rl.flow_ppo import gaussian_kl,ppo_objective
from g05.rl.trust_region import cpu_copy


class Probe(Experiment):
    def run(self):
        try:
            self.load(); self.prepare_bc()
            spec=self.manifest['source_rollout']
            if sha(spec['path'])!=spec['sha256']: raise ValueError('Changed saved trajectories')
            data=torch.load(spec['path'],map_location='cpu',weights_only=False)
            rows=data['rows']; adv=data['advantages']; adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
            indices=self.manifest['probe']['indices']
            parameters=self.actor_parameters+list(self.noise.parameters())
            weights=[p.detach().cpu().clone() for p in parameters]
            state=cpu_copy(self.actor_optimizer.state_dict())
            self.phase='precision_probe'; self.status()

            def restore():
                with torch.no_grad():
                    for p,w in zip(parameters,weights): p.copy_(w)
                self.actor_optimizer.load_state_dict(cpu_copy(state))

            @torch.no_grad()
            def means(i,full=False):
                context=move(rows[i]['context'],'cuda:0'); trace=move(rows[i]['trace'],'cuda:0')
                if full:
                    context['keys']={k:v.float() for k,v in context['keys'].items()}
                    context['values']={k:v.float() for k,v in context['values'].items()}
                    context['mask']=context['mask'].float() if context['mask'].is_floating_point() else context['mask']
                self.adapter.ae_autocast=not full
                ms=[]; ss=[]
                try:
                    for k in range(10):
                        mean,std=self.adapter.transition(context,trace['xs'][k],trace['times'][k]); ms.append(mean); ss.append(std)
                finally: self.adapter.ae_autocast=True
                return torch.stack(ms),torch.stack(ss)

            reference={i:means(i,True) for i in indices}

            @torch.no_grad()
            def scores():
                records=[]
                for i in indices:
                    self.timecheck(); row=rows[i]
                    lp,kl=self.adapter.score(move(row['context'],'cuda:0'),move(row['trace'],'cuda:0'))
                    fm,fs=means(i,True); base_mean,base_std=reference[i]
                    records.append(dict(index=i,bf16_kl=float(kl),logp_error=float(lp-row['trace']['old_logp'].to('cuda:0')),
                        fp32_change_kl=float(gaussian_kl(base_mean,base_std,fm,fs)),
                        fp32_to_bf16_baseline_kl=float(gaussian_kl(row['trace']['means'].to('cuda:0'),
                            row['trace']['stds'].to('cuda:0'),base_mean,base_std))))
                return records

            initial=scores(); repeated=scores(); restore(); restored=scores()
            if any(abs(r['bf16_kl'])>1e-6 or abs(r['logp_error'])>.002 for r in initial+repeated+restored):
                raise ValueError('No-op baseline identity already unstable')
            save(OUT/'identity.json',dict(initial=initial,repeated=repeated,restored=restored,
                parameter_dtypes=dict(Counter(str(p.dtype) for p in self.actor_parameters))))
            group=list(range(len(rows))); self.rng.shuffle(group); group=group[:8]
            self.actor_optimizer.zero_grad(set_to_none=True)
            for i in group:
                lp,_=self.adapter.score(move(rows[i]['context'],'cuda:0'),move(rows[i]['trace'],'cuda:0'))
                (ppo_objective(lp,rows[i]['trace']['old_logp'].to('cuda:0'),adv[i].to('cuda:0'))/len(group)).backward()
            bc,_=self.auxiliary_bc(0); (.1*bc).backward()
            torch.nn.utils.clip_grad_norm_(parameters,.5,error_if_nonfinite=True)
            gradients=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
            for rate in self.manifest['probe']['learning_rates']:
                self.timecheck(); restore()
                for p,g in zip(parameters,gradients): p.grad=None if g is None else g.clone()
                self.actor_optimizer.param_groups[0]['lr']=rate
                self.actor_optimizer.param_groups[1]['lr']=rate*10
                self.actor_optimizer.step()
                changed=quantized=0; maximum=0.
                for p,w in zip(self.actor_parameters,weights):
                    value=p.detach().cpu(); delta=(value-w).abs()
                    changed+=int(torch.count_nonzero(delta)); maximum=max(maximum,float(delta.max()))
                    quantized+=int(torch.count_nonzero(value.to(torch.bfloat16)!=w.to(torch.bfloat16)))
                row=dict(lr=rate,ae_changed_parameters=changed,ae_max_abs_change=maximum,
                    bf16_cast_changed_parameters=quantized,scores=scores())
                self.log.write(json.dumps(row,allow_nan=False)+'\n'); self.status(last_probe=row)
            restore(); final=scores()
            if any(abs(r['bf16_kl'])>1e-6 or abs(r['logp_error'])>.002 for r in final):
                raise ValueError('Final parent restore changed likelihood')
            save(OUT/'final_restore.json',dict(scores=final,accepted_updates=0,saved_actor=False))
            self.phase='completed'; self.status(accepted_updates=0,extra_environment_controls=0)
        except BaseException as error:
            self.phase='failed'; self.status(error=repr(error))
            save(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc())); raise
        finally: self.log.close()


if __name__=='__main__':
    def stop(signum,frame): raise SystemExit(f'Owned numerical probe stopped by signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    Probe().run()
