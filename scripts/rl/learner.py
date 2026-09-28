"""E1: measured two-simulator collection and bounded full-path Flow-PPO.

This is an integration experiment, NOT an automatic large-training launcher.
Public-test/validation episodes never enter the sampler or auxiliary BC.
"""
from __future__ import annotations
from copy import deepcopy
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from common import (REPO,OUT,PARENT_SHA,MODEL_PY,SIM_PY,commit,save,send,recv,sim_env,sha)
from g05.rl.flow_ppo import (ControlBudget,NoiseHead,ValueHead,gae,ppo_objective)
from g05.rl.g05_adapter import G05FlowAdapter,move


def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_tree(v) for v in x)
    return deepcopy(x)


class Experiment:
    def __init__(self):
        self.started=time.monotonic(); self.manifest=json.loads((OUT/'manifest.json').read_text())
        if self.manifest['source_commit']!=commit(): raise ValueError('Preparation/source mismatch')
        self.budget=ControlBudget(self.manifest['max_controls'])
        self.processes=[]; self.connections={}; self.latest={}; self.actor_updates=0; self.critic_updates=0
        self.successes=0; self.batches=0; self.phase='starting'; self.active=[0,1]
        self.log=(OUT/'learner.jsonl').open('x',buffering=1)
        self.rng=random.Random(1700)
        self.actions={r['worker']:np.load(r['actions']) for r in self.manifest['workers']}
        self.prefix={r['worker']:r['prefix_controls'] for r in self.manifest['workers']}
        self.contexts={}; self.prepared={}; self.policy_controls={0:0,1:0}

    def status(self,**extra):
        row=dict(phase=self.phase,seconds=time.monotonic()-self.started,controls=self.budget.used,
            pending_controls=self.budget.pending,actor_updates=self.actor_updates,critic_updates=self.critic_updates,
            successes=self.successes,batches=self.batches,pid=os.getpid(),**extra)
        save(OUT/'status.json',row,replace=True)
        print('RL_STATUS',json.dumps(row),flush=True)

    def timecheck(self):
        if time.monotonic()-self.started>self.manifest['max_active_wall_seconds']:
            raise TimeoutError('Registered two-hour active-wall budget exhausted')

    def issue(self,worker,actions,phase):
        self.timecheck(); count=len(actions); self.budget.reserve(count)
        send(self.connections[worker],dict(op='step',actions=actions,phase=phase))
        return count

    def collect_reply(self,worker,reserved):
        reply=recv(self.connections[worker])
        self.budget.finish(reserved,int(reply['actual_controls']))
        self.latest[worker]=reply
        return reply

    def reset(self,workers):
        for worker in workers: send(self.connections[worker],dict(op='reset'))
        for worker in workers:
            self.latest[worker]=recv(self.connections[worker]); self.policy_controls[worker]=0
            self.contexts.pop(worker,None); self.prepared.pop(worker,None)

    def replay(self,workers,counts,*,phase,parallel=True):
        progress={w:0 for w in workers}; start=time.monotonic()
        while any(progress[w]<counts[w] for w in workers):
            reserved={}
            for w in workers:
                end=min(progress[w]+16,counts[w])
                if progress[w]>=end: continue
                reserved[w]=self.issue(w,self.actions[w][progress[w]:end],phase)
                if not parallel:
                    row=self.collect_reply(w,reserved[w]); progress[w]+=row['actual_controls']
                    if row['terminal']: raise RuntimeError('Expert prefix became terminal')
            if parallel:
                for w,n in reserved.items():
                    row=self.collect_reply(w,n); progress[w]+=row['actual_controls']
                    if row['terminal']: raise RuntimeError('Expert prefix became terminal')
            self.status(replay_progress=progress)
        return dict(controls=sum(progress.values()),seconds=time.monotonic()-start)

    def spawn(self):
        secret=os.urandom(32)
        self.listener=Listener(('127.0.0.1',0),authkey=secret)
        port=self.listener.address[1]
        for worker in (0,1):
            env=sim_env(worker); env['BEHAVIOR_RL_IPC_KEY']=secret.hex()
            f=(OUT/f'sim_{worker}.stdout.log').open('x')
            p=subprocess.Popen([SIM_PY,str(REPO/'scripts/rl/sim_worker.py'),'--worker',str(worker),'--port',str(port)],
                env=env,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
            f.close(); self.processes.append(p)
        save(OUT/'pids.json',dict(learner=os.getpid(),simulators=[p.pid for p in self.processes]))
        for _ in (0,1):
            conn=self.listener.accept(); hello=recv(conn,30); worker=hello['hello']
            if worker not in (0,1) or worker in self.connections: raise ValueError('Duplicate worker identity')
            self.connections[worker]=conn

    def load(self):
        sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src'),str(REPO)]
        from scripts.experiments import eval_g05_100k as baseline
        baseline.OUT=OUT
        self.cfg,self.policy,self.processor=baseline.load_model(50000)
        torch.manual_seed(1700); torch.cuda.manual_seed_all(1700)
        self.noise=NoiseHead().to('cuda:0'); self.critic=ValueHead().to('cuda:0')
        self.adapter=G05FlowAdapter(self.policy,self.processor,self.noise)
        self.actor_parameters=list(self.policy.model.action_expert.parameters())
        self.actor_optimizer=torch.optim.AdamW([
            dict(params=self.actor_parameters,lr=1e-6),dict(params=self.noise.parameters(),lr=1e-5)],weight_decay=0)
        self.critic_optimizer=torch.optim.AdamW(self.critic.parameters(),lr=1e-4,weight_decay=0)
        save(OUT/'trainable.json',dict(actor_parameters=sum(p.numel() for p in self.actor_parameters),
            frozen_parameters=sum(p.numel() for p in self.policy.parameters() if not p.requires_grad),
            trainable_names=[n for n,p in self.policy.named_parameters() if p.requires_grad],
            policy_training=self.policy.training,noise_initial=.01))

    def benchmark(self):
        self.phase='benchmark'; self.status()
        for w in (0,1): self.latest[w]=recv(self.connections[w],1200)
        # Warm both cameras/physics; then exactly matched 128 controls/worker.
        warm=self.replay([0,1],{0:16,1:16},phase='benchmark_warmup')
        self.reset([0,1])
        serial=self.replay([0,1],{0:128,1:128},phase='benchmark_serial',parallel=False)
        self.reset([0,1])
        parallel=self.replay([0,1],{0:128,1:128},phase='benchmark_parallel')
        speedup=serial['seconds']/parallel['seconds']
        result=dict(warmup=warm,serial=serial,parallel=parallel,speedup=speedup,
            includes_per_chunk_observation=True,excludes_initialization=True,
            definition='same 256 controls, serial dispatch versus concurrent dispatch, both native processes resident')
        save(OUT/'throughput.json',result)
        # Keep both TRAIN instances but fall back to serial step dispatch when
        # concurrent execution does not materially improve measured throughput.
        self.parallel=speedup>1.1
        self.status(throughput=result,use_parallel=self.parallel)
        self.reset([0,1]); self.replay([0,1],self.prefix,phase='expert_prefix',parallel=self.parallel)
        for w in (0,1):
            send(self.connections[w],dict(op='observe',tag='curriculum_start'))
            self.latest[w]=recv(self.connections[w])
            if self.latest[w]['success'] or self.latest[w]['terminal']: raise ValueError('Terminal curriculum start')
        save(OUT/'curriculum_gate.json',dict(workers=[dict(worker=w,episode=self.latest[w]['episode'],
            controls=self.latest[w]['episode_controls'],held=self.latest[w]['held'],success=False,
            image=str(OUT/f'worker_{w}'/f'curriculum_start_{self.latest[w]["episode"]:03d}.png')) for w in (0,1)]))
        self.phase='awaiting_parent_image_review'; self.status()
        # Agent manually views both original camera mosaics and signs their SHA.
        while not (OUT/'human_release.json').is_file():
            self.timecheck(); time.sleep(2)
        release=json.loads((OUT/'human_release.json').read_text())
        if release.get('accepted') is not True or release.get('reviewer')!='parent_agent':
            raise ValueError('Manual curriculum review missing')
        for row in release['images']:
            if sha(row['path'])!=row['sha256']: raise ValueError('Reviewed image changed')
        expected={r['image'] for r in json.loads((OUT/'curriculum_gate.json').read_text())['workers']}
        if {r['path'] for r in release['images']}!=expected: raise ValueError('Both TRAIN start images required')

    def prepare_worker(self,w):
        if w not in self.contexts:
            context,prepared=self.adapter.prepare(self.latest[w]['observation'])
            self.contexts[w]=context; self.prepared[w]=prepared
        return self.contexts[w],self.prepared[w]

    def gates(self):
        self.phase='model_probability_gates'; self.status()
        context,prepared=self.prepare_worker(0)
        result=self.adapter.equivalence_gate(context,prepared)
        _,trace=self.adapter.sample(context)
        lp,_=self.adapter.score(context,move(trace,'cuda:0'))
        grad_mode_error=float((lp.detach()-trace['old_logp'].to('cuda:0')).abs())
        if grad_mode_error>.002: raise ValueError('Gradient-mode old-path identity failed')
        result['grad_mode_old_path_logp_error']=grad_mode_error
        self.actor_optimizer.zero_grad(set_to_none=True)
        # Identity-ratio policy-gradient smoke test, without optimizer update.
        loss=ppo_objective(lp,trace['old_logp'].to('cuda:0'),torch.tensor(1.,device='cuda:0'))
        loss.backward()
        grads=[p.grad for p in self.actor_parameters if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads): raise ValueError('Invalid AE gradients')
        if any(p.grad is not None for p in self.policy.parameters() if not p.requires_grad):
            raise ValueError('Gradient leaked into frozen parent')
        result.update(ae_gradient_norm=float(torch.nn.utils.clip_grad_norm_(self.actor_parameters,.5)),
                      frozen_gradient_leak=False,optimizer_steps=0)
        self.actor_optimizer.zero_grad(set_to_none=True)
        save(OUT/'probability_gate.json',result)
        from g05.utils.data.processor_utils import instantiate_dataset
        self.dataset=instantiate_dataset(self.cfg,is_training_set=True); self.dataset.set_processor(self.processor)
        child=self.dataset.datasets[0]
        episodes=list(map(int,child._active_episode_indices))
        if len(episodes)!=950 or any(e%200>=190 for e in episodes): raise ValueError('Original TRAIN split changed')
        self.bc_indices=[]
        for episode in (0,200,400,600,800):
            local=episodes.index(episode)
            start,end=int(child.episode_data_index['from'][local]),int(child.episode_data_index['to'][local])
            self.bc_indices.append(start+(end-start-32)//2)
        save(OUT/'bc_windows.json',dict(episodes=[0,200,400,600,800],indices=self.bc_indices,
                                       source='original 95% TRAIN only',weight=.1))

    def auxiliary_bc(self,index):
        from g05.utils.data.data_utils import collate_fn_pad_sequences
        sample=self.dataset[self.bc_indices[index%5]]
        batch=collate_fn_pad_sequences([sample],padding_input_id=self.processor.pad_token_id)
        batch=move(batch,'cuda:0')
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,details=self.policy(batch)
        if not torch.isfinite(loss): raise ValueError('Nonfinite original TRAIN FM BC')
        return loss, {k:float(v.detach()) for k,v in details.items()}

    def rollout(self):
        rows={w:[] for w in self.active}; completed=set(); count=0
        while count<128 and self.budget.remaining>=16*len(self.active):
            self.timecheck(); pending={}
            for w in self.active:
                if w in completed: continue
                context,prepared=self.prepare_worker(w)
                with torch.no_grad():
                    x,trace=self.adapter.sample(context); value=float(self.critic(context['feature']).item())
                actions=self.adapter.decode(x,prepared)
                row=dict(context=move(context,'cpu'),trace=trace,value=value,episode=self.latest[w]['episode'],
                         raw_observation=self.latest[w]['observation'])
                n=min(16,512-self.policy_controls[w],self.budget.remaining)
                if n<1: completed.add(w); continue
                reserved=self.issue(w,actions[:n],'policy')
                pending[w]=(row,reserved)
                if not self.parallel:
                    pending[w]=(row,reserved,self.collect_reply(w,reserved))
            for w,item in pending.items():
                row,reserved=item[:2]; reply=item[2] if len(item)==3 else self.collect_reply(w,reserved)
                self.policy_controls[w]+=reply['actual_controls']; self.contexts.pop(w,None); self.prepared.pop(w,None)
                # Final observation is encoded BEFORE any reset, even on truncation.
                context,_=self.prepare_worker(w)
                with torch.no_grad(): next_value=float(self.critic(context['feature']).item())
                row.update(rewards=reply['rewards'],terminated=reply['terminated'],
                    truncated=reply['truncated'] or self.policy_controls[w]>=512,next_value=next_value)
                rows[w].append(row); count+=1
                if reply['success']: self.successes+=1
                if row['terminated'] or row['truncated']: completed.add(w)
            self.status(rollout_chunks=count,completed_workers=sorted(completed))
            # Curriculum reset is costly; flush this on-policy batch at the
            # first completed pair of episodes (128 chunks remains an upper bound).
            if set(self.active)<=completed: break
        flat=[]; advantages=[]; returns=[]
        for w in self.active:
            adv,ret=gae(rows[w]); flat+=rows[w]; advantages+=adv.tolist(); returns+=ret.tolist()
        return flat,torch.tensor(advantages),torch.tensor(returns),completed

    def update(self,rows,advantages,returns):
        self.phase='updating'; self.status(batch_samples=len(rows),batch_reward=sum(sum(r['rewards']) for r in rows))
        if len(rows)<8:
            self.log.write(json.dumps(dict(event='skip_short_batch',n=len(rows)))+'\n'); return
        features=torch.cat([r['context']['feature'] for r in rows]).to('cuda:0')
        targets=returns.to('cuda:0')
        # Warm critic on the first REAL return batch before any AE update.
        for epoch in range(4 if self.batches==0 else 2):
            pred=self.critic(features)
            loss=.5*(pred-targets).square().mean()
            self.critic_optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(),.5,error_if_nonfinite=True)
            self.critic_optimizer.step(); self.critic_updates+=1
        rewards=sum(sum(r['rewards']) for r in rows)
        if rewards==0:
            self.log.write(json.dumps(dict(event='skip_actor_zero_success',samples=len(rows),critic_loss=float(loss)))+'\n')
            return
        # Preserve rollout-time value/GAE. Critic warmup must not retroactively
        # change the old-policy advantage estimator within this PPO batch.
        advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
        for epoch in range(2):
            indices=list(range(len(rows))); self.rng.shuffle(indices)
            for start in range(0,len(indices),8):
                self.timecheck(); group=indices[start:start+8]
                # CPU rollback includes Adam moments and step numbers. No
                # rejected candidate may remain deployed to the simulators.
                backup=dict(ae=cpu_tree(self.policy.model.action_expert.state_dict()),
                    noise=cpu_tree(self.noise.state_dict()),optimizer=cpu_tree(self.actor_optimizer.state_dict()))
                self.actor_optimizer.zero_grad(set_to_none=True); losses=[]; before_kls=[]
                for i in group:
                    context=move(rows[i]['context'],'cuda:0'); trace=move(rows[i]['trace'],'cuda:0')
                    lp,kl=self.adapter.score(context,trace)
                    if float(kl.detach())<-.002: raise ValueError('Negative conditional Gaussian KL')
                    before_kls.append(float(kl.detach()))
                    if before_kls[-1]>.01:
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        self.log.write(json.dumps(dict(event='early_stop_prior_path_kl',kl=before_kls[-1]))+'\n'); return
                    loss=ppo_objective(lp,trace['old_logp'],advantages[i].to('cuda:0'))/len(group)
                    loss.backward(); losses.append(float(loss.detach()))
                bc,bc_details=self.auxiliary_bc(self.actor_updates)
                (.1*bc).backward()
                grad=float(torch.nn.utils.clip_grad_norm_(self.actor_parameters+list(self.noise.parameters()),.5,error_if_nonfinite=True))
                self.actor_optimizer.step()
                with torch.no_grad():
                    after=[float(self.adapter.score(move(rows[i]['context'],'cuda:0'),move(rows[i]['trace'],'cuda:0'))[1]) for i in group]
                mean_kl=float(np.mean(after))
                if not np.isfinite(mean_kl) or mean_kl>.01:
                    self.policy.model.action_expert.load_state_dict(backup['ae'],strict=True)
                    self.noise.load_state_dict(backup['noise'],strict=True)
                    self.actor_optimizer.load_state_dict(backup['optimizer'])
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.log.write(json.dumps(dict(event='rollback_path_kl',kl=mean_kl,epoch=epoch))+'\n')
                    return
                self.actor_updates+=1
                self.log.write(json.dumps(dict(event='actor_update',update=self.actor_updates,ppo=sum(losses),
                    bc=bc_details,path_kl=mean_kl,gradient_norm=grad,samples=len(group)))+'\n')
                self.status(last_path_kl=mean_kl)
                del backup

    def checkpoint(self):
        path=OUT/f'rl_batch_{self.batches:03d}_updates_{self.actor_updates:04d}.pt'
        if path.exists(): raise ValueError('Refuse to overwrite RL checkpoint')
        tmp=path.with_suffix('.tmp')
        torch.save(dict(schema='g05_flow_ppo_e1_delta_v1',parent_sha256=PARENT_SHA,source_commit=commit(),
            action_expert=cpu_tree(self.policy.model.action_expert.state_dict()),noise=cpu_tree(self.noise.state_dict()),
            critic=cpu_tree(self.critic.state_dict()),actor_optimizer=cpu_tree(self.actor_optimizer.state_dict()),
            critic_optimizer=cpu_tree(self.critic_optimizer.state_dict()),actor_updates=self.actor_updates,
            critic_updates=self.critic_updates,controls=self.budget.used,manifest=self.manifest,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=self.rng.getstate()),tmp)
        os.replace(tmp,path)
        save(path.with_suffix('.json'),dict(path=str(path),sha256=sha(path),parent_sha256=PARENT_SHA,
            actor_updates=self.actor_updates,critic_updates=self.critic_updates,controls=self.budget.used,
            deployment='load parent then strict action_expert delta; original ODE, no additional transition noise'))

    def run(self):
        try:
            self.spawn(); self.load(); self.benchmark(); self.gates()
            while self.budget.remaining>=16*len(self.active):
                self.phase='collecting'; self.status()
                rows,advantages,returns,completed=self.rollout()
                if not rows: break
                # Raw public observations, all latent states and old likelihoods
                # are saved before updating; no hidden regenerated actions.
                torch.save(dict(rows=rows,advantages=advantages,returns=returns),OUT/f'rollout_{self.batches:03d}.pt')
                self.update(rows,advantages,returns); self.batches+=1; self.checkpoint()
                del rows
                if completed:
                    required=sum(self.prefix[w]+16 for w in completed)
                    if self.budget.remaining<required: break
                    self.reset(sorted(completed))
                    self.replay(sorted(completed),self.prefix,phase='expert_prefix',parallel=self.parallel)
                    for w in completed:
                        if self.latest[w]['terminal']: raise ValueError('Repeated curriculum replay became terminal')
                if time.monotonic()-self.started>self.manifest['max_active_wall_seconds']-300: break
            self.phase='completed'; self.status(full_task_success_rate_claim=False)
        except BaseException as error:
            self.phase='failed'; self.status(error=repr(error))
            save(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),controls=self.budget.used))
            raise
        finally:
            for conn in self.connections.values():
                try: send(conn,dict(op='close'))
                except BaseException: pass
            for process in self.processes:
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
            for conn in self.connections.values(): conn.close()
            self.log.close()


if __name__=='__main__':
    def stop(signum,frame): raise SystemExit(f'Owned experiment stopped by signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    Experiment().run()
