"""E2: fixed full-reset evaluation, resumed TRAIN PPO, paired final evaluation.

No evaluation transition can reach update(). All stages share one control
ledger and deadline. The final checkpoint is selected BEFORE final evaluation.
"""
from __future__ import annotations
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import numpy as np
import torch

from common import OUT,REPO,SIM_PY,PARENT_SHA,save,sha,send,recv,sim_env
from learner import Experiment,cpu_tree
from g05.rl.g05_adapter import G05FlowAdapter,move
from g05.rl.protocol import EVAL_SEEDS,EVAL_LIMIT,paired_summary,next_prefix,check_reset
from g05.rl.rewards import has_learning_signal
from g05.rl.time_limits import training_deadline


class TrainingCutoff(RuntimeError):
    pass


class MethodExperiment(Experiment):
    def __init__(self):
        super().__init__()
        if self.manifest['entry'] not in ('method','method_dense'): raise ValueError('Wrong experiment role')
        self.pool=None; self.evaluations=[]; self.reset_references={}
        self.parent_ae=None; self.last_checkpoint=None; self.start_updates=10
        self.pool_history=[]; self.training_started=None; self.stop_reason=None
        self.outstanding={}; self.training_start_controls=None
        self.eval_log=(OUT/'evaluations.jsonl').open('x',buffering=1)

    def timecheck(self):
        super().timecheck()
        if self.pool=='final' and time.monotonic()>=getattr(self,'final_deadline',float('inf')):
            raise TimeoutError('Registered final-evaluation timeout')
        if (self.pool=='training' and hasattr(self,'training_deadline')
                and time.monotonic()>=self.training_deadline):
            raise TrainingCutoff('Registered training deadline; preserve final evaluation reserve')

    def issue(self,worker,actions,phase):
        if worker in self.outstanding: raise ValueError('Overlapping action dispatch')
        count=super().issue(worker,actions,phase)
        self.outstanding[worker]=count
        return count

    def collect_reply(self,worker,reserved):
        if self.outstanding.get(worker)!=reserved: raise ValueError('Unmatched pending controls')
        row=super().collect_reply(worker,reserved)
        del self.outstanding[worker]
        return row

    def close_pool(self,strict=True):
        failed=False; closed=[]; pool=self.pool
        # A deadline can occur between two dispatches. Reap actual controls
        # before closing, so sent actions cannot disappear from the ledger.
        for w,n in list(self.outstanding.items()):
            try: self.collect_reply(w,n)
            except BaseException as error:
                failed=True
                self.log.write(json.dumps(dict(event='unresolved_controls',worker=w,reserved=n,error=repr(error)))+'\n')
        for w,conn in self.connections.items():
            try:
                send(conn,dict(op='close'))
                receipt=recv(conn,30)
                save(OUT/self.pool/f'worker_{w}_closed.json',receipt)
                closed.append(receipt['total_controls'])
            except BaseException as error:
                failed=True
                self.log.write(json.dumps(dict(event='pool_close_error',worker=w,error=repr(error)))+'\n')
        exits=[]
        for process in self.processes:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
            if process.returncode: failed=True
            exits.append(process.returncode)
        for conn in self.connections.values(): conn.close()
        if hasattr(self,'listener'): self.listener.close()
        self.processes=[]; self.connections={}; self.contexts={}; self.prepared={}; self.latest={}
        if self.outstanding:
            save(OUT/f'unresolved_{self.pool}.json',dict(pending=self.outstanding,budget_pending=self.budget.pending))
            self.outstanding={}
        self.pool=None
        if pool is not None:
            used=self.budget.used-getattr(self,'pool_start_controls',self.budget.used)
            if len(closed)!=2 or sum(closed)!=used: failed=True
            save(OUT/pool/'closed.json',dict(reported_controls=closed,ledger_controls=used,
                 exits=exits,pending_controls=self.budget.pending,clean=not failed))
        if strict and (failed or self.budget.pending):
            raise RuntimeError('Simulator pool did not close cleanly; final evaluation cannot proceed')

    def spawn_pool(self,pool):
        if self.processes or self.connections: raise ValueError('Previous pool still owned')
        self.pool=pool; self.phase=pool+'_initializing'; self.status()
        if pool=='final' and self.manifest.get('max_active_wall_seconds') is None:
            self.final_deadline=time.monotonic()+self.manifest['final_eval_reserved_seconds']
        self.pool_start_controls=self.budget.used
        (OUT/pool).mkdir(exist_ok=False)
        secret=os.urandom(32); self.listener=Listener(('127.0.0.1',0),authkey=secret)
        self.listener._listener._socket.settimeout(1200)
        for worker in (0,1):
            self.timecheck()
            env=sim_env(worker,pool); env['BEHAVIOR_RL_IPC_KEY']=secret.hex()
            with (OUT/pool/f'sim_{worker}.stdout.log').open('x') as log:
                p=subprocess.Popen([SIM_PY,str(REPO/'scripts/rl/sim_worker.py'),'--worker',str(worker),
                    '--port',str(self.listener.address[1]),'--pool',pool],env=env,cwd=REPO,
                    stdout=log,stderr=subprocess.STDOUT)
            self.processes.append(p)
        pids=dict(pool=pool,learner=os.getpid(),simulators=[p.pid for p in self.processes])
        self.pool_history.append(pids); save(OUT/'pids.json',dict(current=pids,history=self.pool_history),replace=True)
        for _ in (0,1):
            conn=self.listener.accept(); hello=recv(conn,30); w=hello['hello']
            if w not in (0,1) or w in self.connections: raise ValueError('Wrong IPC identity')
            self.connections[w]=conn
        for w in (0,1): self.latest[w]=recv(self.connections[w],1200)
        self.contexts={}; self.prepared={}; self.policy_controls={0:0,1:0}

    def configure_precision(self,precision):
        self.contexts={}; self.prepared={}
        self.adapter=G05FlowAdapter(self.policy,self.processor,self.noise,ae_autocast=precision=='bfloat16')
        torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False

    def restore_delta(self,path,expected_sha,*,optimizer,receipt_name=None):
        if sha(path)!=expected_sha: raise ValueError('Resume checkpoint SHA mismatch')
        delta=torch.load(path,map_location='cpu',weights_only=False)
        if (delta['schema']!='g05_flow_ppo_e1_delta_v1' or delta['parent_sha256']!=PARENT_SHA
                or delta['manifest']['ae_precision']!='float32'):
            raise ValueError('Wrong delta parent/schema/precision')
        for key in ('action_expert','noise','critic'):
            if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in delta[key].values()):
                raise ValueError('Nonfinite or non-FP32 delta')
        self.policy.model.action_expert.load_state_dict(delta['action_expert'],strict=True)
        self.noise.load_state_dict(delta['noise'],strict=True)
        self.critic.load_state_dict(delta['critic'],strict=True)
        if optimizer:
            self.actor_optimizer.load_state_dict(delta['actor_optimizer'])
            self.critic_optimizer.load_state_dict(delta['critic_optimizer'])
            for opt,count in ((self.actor_optimizer,delta['actor_updates']),
                              (self.critic_optimizer,delta['critic_updates'])):
                if any(float(s['step'])!=count for s in opt.state.values()):
                    raise ValueError('Optimizer step count does not match checkpoint')
            torch.set_rng_state(delta['torch_rng']); torch.cuda.set_rng_state_all(delta['cuda_rng'])
            self.rng.setstate(delta['python_rng'])
        self.actor_updates=int(delta['actor_updates']); self.critic_updates=int(delta['critic_updates'])
        self.configure_precision('float32')
        save(OUT/(receipt_name or ('resume_load.json' if optimizer else 'final_deployment_load.json')),dict(
            checkpoint=str(path),sha256=expected_sha,strict=True,parent_sha256=PARENT_SHA,
            actor_updates=self.actor_updates,critic_updates=self.critic_updates,
            optimizer_restored=optimizer,ae_precision='float32'))
        del delta

    def check_eval_resets(self,variant,seed):
        checks=[]
        for w in (0,1):
            row=self.latest[w]; spec=self.manifest['sim_pools'][self.pool][w]
            if row['episode_controls'] or row['success'] or row['terminal']:
                raise ValueError('Evaluation must begin from live official reset with zero prefix')
            state=row['reset_state']; key=spec['instance']
            if key not in self.reset_references: self.reset_references[key]=state
            diff=check_reset(self.reset_references[key],state)
            checks.append(dict(instance=key,max_abs_difference=diff,episode=row['episode']))
        save(OUT/self.pool/f'{variant}_seed{seed}_reset.json',dict(checks=checks))

    def evaluate(self,variants):
        if self.pool not in ('baseline','final'): raise ValueError('Evaluation pool required')
        for variant,seeds,precision in variants:
            self.configure_precision(precision)
            for seed in seeds:
                if self.evaluations or any(r['episode_controls'] for r in self.latest.values()):
                    self.reset([0,1])
                self.check_eval_resets(variant,seed)
                self.phase='evaluating_'+variant; self.status(policy_seed=seed)
                if seed==seeds[0]:
                    context,prepared=self.prepare_worker(0)
                    gate=self.adapter.equivalence_gate(context,prepared)
                    # File -> graph -> native decoder is checked for the final delta too.
                    save(OUT/self.pool/f'{variant}_deployment_gate.json',gate)
                generators={w:torch.Generator(device='cuda:0').manual_seed(seed) for w in (0,1)}
                executed={0:0,1:0}; calls={0:0,1:0}; completed=set()
                while len(completed)<2:
                    self.timecheck(); pending={}
                    for w in (0,1):
                        if w in completed: continue
                        context,prepared=self.prepare_worker(w)
                        x,_=self.adapter.sample(context,stochastic=False,generator=generators[w])
                        actions=self.adapter.decode(x,prepared)
                        count=min(16,EVAL_LIMIT-executed[w])
                        pending[w]=self.issue(w,actions[:count],'eval_'+variant); calls[w]+=1
                    for w,n in pending.items():
                        reply=self.collect_reply(w,n); executed[w]+=reply['actual_controls']
                        self.contexts.pop(w,None); self.prepared.pop(w,None)
                        if reply['terminal'] or executed[w]>=EVAL_LIMIT:
                            completed.add(w); spec=self.manifest['sim_pools'][self.pool][w]
                            row=dict(variant=variant,instance=spec['instance'],policy_seed=seed,environment_seed=0,
                                split='public_test',expert_prefix_controls=0,controls=executed[w],control_limit=EVAL_LIMIT,
                                success=bool(reply['success']),terminated=bool(reply['terminated']),
                                truncated=bool(reply['truncated']),model_calls=calls[w],episode=reply['episode'],
                                actor_updates=self.actor_updates,ae_precision=precision,
                                video=str(OUT/self.pool/f'worker_{w}'/f'episode_{reply["episode"]:03d}.mp4'))
                            self.evaluations.append(row); self.eval_log.write(json.dumps(row)+'\n')
                            save(OUT/self.pool/f'{variant}_instance{spec["instance"]}_seed{seed}.json',row)
                    self.status(variant=variant,policy_seed=seed,episode_controls=executed,completed=sorted(completed))

    def review_curriculum(self,batch):
        images=[]
        for w in (0,1):
            send(self.connections[w],dict(op='observe',tag='curriculum_start'))
            self.latest[w]=recv(self.connections[w])
            row=self.latest[w]
            if row['success'] or row['terminal'] or row['episode_controls']!=self.prefix[w]:
                raise ValueError('Invalid curriculum start')
            file=OUT/'training'/f'worker_{w}'/f'curriculum_start_{row["episode"]:03d}.png'
            images.append(dict(worker=w,path=str(file),sha256=sha(file),prefix=self.prefix[w],
                held=row['held'],success=False,terminal=False,instance=self.manifest['workers'][w]['instance']))
        gate=OUT/f'curriculum_gate_{batch:03d}.json'; save(gate,dict(images=images,batch=batch))
        if self.manifest.get('curriculum_admission')=='automatic':
            from g05.rl.curriculum import check_curriculum
            checks=[check_curriculum(self.manifest['workers'][w],self.latest[w],self.prefix[w]) for w in (0,1)]
            save(OUT/f'curriculum_auto_{batch:03d}.json',dict(batch=batch,checks=checks,
                accepted=True,human_review_required=False))
            self.phase='automatic_curriculum_passed'; self.status(batch=batch)
            return
        self.phase='awaiting_curriculum_review'; self.status(gate=str(gate),images=images)
        release=OUT/f'human_release_{batch:03d}.json'
        while not release.is_file():
            self.timecheck()
            if time.monotonic()>self.training_deadline: raise TrainingCutoff('Training review budget exhausted')
            time.sleep(2)
        signed=json.loads(release.read_text())
        if signed.get('accepted') is not True or signed.get('reviewer')!='parent_agent':
            raise ValueError('Parent image review not accepted')
        if {(r['path'],r['sha256']) for r in signed['images']} != {(r['path'],r['sha256']) for r in images}:
            raise ValueError('Wrong reviewed images')
        if any(sha(r['path'])!=r['sha256'] for r in images): raise ValueError('Reviewed image changed')

    def train(self):
        if self.pool!='training': raise ValueError('Only TRAIN pool may train')
        resume=self.manifest.get('training_resume',{})
        self.training_started=time.monotonic(); start_controls=self.budget.used-resume.get('controls',0)
        self.training_start_controls=start_controls
        self.training_deadline=training_deadline(self.manifest,started=self.started,training_started=self.training_started)
        recent={w:list(resume.get('recent',{}).get(str(w),[])) for w in (0,1)}; reviewed=set()
        zero_rewards=resume.get('zero_rewards',0); zero_updates=resume.get('zero_updates',0); self.parallel=True
        first_batch=self.batches
        for batch in range(first_batch,self.manifest['max_batches']):
            if (time.monotonic()>self.training_deadline-600 or
                    self.actor_updates-self.start_updates>=self.manifest['max_new_actor_updates']):
                self.stop_reason='registered_training_time_or_update_limit'; break
            needed=sum(self.prefix.values())+2*self.manifest['episode_controls']
            if (self.budget.used-start_controls+needed>self.manifest['max_training_controls'] or
                    self.budget.remaining-needed<self.manifest['final_eval_reserved_controls']):
                self.stop_reason='reserve_final_evaluation_controls'; break
            if batch>first_batch: self.reset([0,1])
            self.phase='training_curriculum'; self.status(batch=batch,prefixes=self.prefix)
            self.replay([0,1],self.prefix,phase='expert_prefix',parallel=True)
            pair=tuple(self.prefix[w] for w in (0,1))
            if pair not in reviewed or self.manifest.get('curriculum_admission')=='automatic':
                self.review_curriculum(batch); reviewed.add(pair)
            elif any(r['success'] or r['terminal'] for r in self.latest.values()):
                raise ValueError('Repeated curriculum prefix became terminal')
            if batch==first_batch: self.gates()
            self.phase='collecting'; self.status(batch=batch,prefixes=self.prefix)
            rows,advantages,returns,completed=self.rollout()
            if not rows: self.stop_reason='empty_training_rollout'; break
            if any(r['split']!='train' for r in rows): raise ValueError('Evaluation leakage')
            torch.save(dict(rows=rows,advantages=advantages,returns=returns,prefixes=dict(self.prefix)),
                       OUT/f'rollout_{batch:03d}.pt')
            rewards=sum(sum(r['rewards']) for r in rows); before=self.actor_updates
            signal_present=has_learning_signal(rows,dense=self.manifest.get('reward') is not None)
            for group,key in zip(self.actor_optimizer.param_groups,('action_expert','noise')):
                group['lr']=self.manifest['learning_rates'][key]
            self.update(rows,advantages,returns); self.batches+=1; self.checkpoint()
            self.last_checkpoint=OUT/f'rl_batch_{self.batches:03d}_updates_{self.actor_updates:04d}.pt'
            detail=dict(batch=batch,chunks=len(rows),reward=rewards,new_actor_updates=self.actor_updates-before,
                checkpoint=str(self.last_checkpoint),controls=self.budget.used-start_controls,episodes=[])
            if self.manifest.get('reward') is not None:
                detail.update(official_reward=sum(sum(r['official_rewards']) for r in rows),
                    shaping_reward=sum(sum(r['shaping_rewards']) for r in rows),
                    nonzero_shaping_controls=sum(abs(v)>1e-9 for r in rows for v in r['shaping_rewards']),
                    policy_controls=sum(len(r['rewards']) for r in rows),
                    reward_is_complete_task_success=False)
            for w in (0,1):
                won=bool(self.latest[w]['success']); recent[w].append(won)
                first=self.manifest['workers'][w]['first_recorded_terminal']
                following=next_prefix(first,self.prefix[w],recent[w])
                detail['episodes'].append(dict(worker=w,instance=self.manifest['workers'][w]['instance'],
                    prefix=self.prefix[w],policy_controls=self.policy_controls[w],success=won,next_prefix=following,
                    episode=self.latest[w]['episode']))
                if following!=self.prefix[w]: recent[w]=[]
                self.prefix[w]=following
            save(OUT/f'train_batch_{batch:03d}.json',detail)
            self.log.write(json.dumps(dict(event='training_batch',**detail))+'\n')
            zero_rewards=zero_rewards+1 if not signal_present else 0
            zero_updates=zero_updates+1 if self.actor_updates==before else 0
            del rows
            if zero_rewards>=3 or zero_updates>=3:
                self.stop_reason='three_zero_reward_or_zero_update_batches'; break
        if self.stop_reason is None: self.stop_reason='registered_batch_limit'
        save(OUT/'training_result.json',dict(reason=self.stop_reason,batches=self.batches,
            new_actor_updates=self.actor_updates-self.start_updates,actor_updates=self.actor_updates,
            controls=self.budget.used-start_controls,seconds=time.monotonic()-self.training_started,
            checkpoint=str(self.last_checkpoint or self.manifest['resume_checkpoint'])))

    def run(self):
        try:
            self.load(); self.parent_ae=cpu_tree(self.policy.model.action_expert.state_dict())
            # Fail on real graph/optimizer restore BEFORE expensive evaluation.
            initial_noise=cpu_tree(self.noise.state_dict()); initial_critic=cpu_tree(self.critic.state_dict())
            actor_initial=cpu_tree(self.actor_optimizer.state_dict())
            critic_initial=cpu_tree(self.critic_optimizer.state_dict())
            self.restore_delta(self.manifest['resume_checkpoint'],self.manifest['resume_sha256'],
                               optimizer=True,receipt_name='resume_preflight.json')
            self.policy.model.action_expert.load_state_dict(self.parent_ae,strict=True)
            self.noise.load_state_dict(initial_noise,strict=True); self.critic.load_state_dict(initial_critic,strict=True)
            self.actor_optimizer.load_state_dict(actor_initial); self.critic_optimizer.load_state_dict(critic_initial)
            self.actor_updates=0; self.critic_updates=0
            del initial_noise,initial_critic,actor_initial,critic_initial
            self.prepare_bc()
            self.spawn_pool('baseline')
            self.evaluate([('parent_bf16',[17],'bfloat16'),('parent_fp32',list(EVAL_SEEDS),'float32')])
            self.close_pool()
            self.restore_delta(self.manifest['resume_checkpoint'],self.manifest['resume_sha256'],optimizer=True)
            self.start_updates=self.actor_updates
            self.spawn_pool('training')
            try: self.train()
            except TrainingCutoff:
                # bounded_adam_step has restored any candidate in flight.
                # Preserve already accepted steps and account dispatched actions.
                for w,n in list(self.outstanding.items()): self.collect_reply(w,n)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                self.batches+=1; self.checkpoint()
                self.last_checkpoint=OUT/f'rl_batch_{self.batches:03d}_updates_{self.actor_updates:04d}.pt'
                self.stop_reason='registered_training_deadline'
                save(OUT/'training_result.json',dict(reason=self.stop_reason,batches=self.batches,
                    new_actor_updates=self.actor_updates-self.start_updates,actor_updates=self.actor_updates,
                    controls=self.budget.used-self.training_start_controls,
                    seconds=time.monotonic()-self.training_started,checkpoint=str(self.last_checkpoint),
                    partial_rollout_not_used_for_extra_updates=True))
            self.close_pool()
            chosen=self.last_checkpoint or Path(self.manifest['resume_checkpoint'])
            receipt=json.loads(chosen.with_suffix('.json').read_text())
            save(OUT/'frozen_final_selection.json',dict(checkpoint=str(chosen),sha256=receipt['sha256'],
                selection='last legal training checkpoint; fixed before final evaluation',eval_results_used=False))
            # Deliberately replace the live AE before file reload: final eval
            # must exercise the SAVED checkpoint, not only live training memory.
            self.policy.model.action_expert.load_state_dict(self.parent_ae,strict=True)
            self.restore_delta(chosen,receipt['sha256'],optimizer=False)
            del self.parent_ae; self.parent_ae=None
            self.spawn_pool('final')
            self.evaluate([('rl_fp32',list(EVAL_SEEDS),'float32')]); self.close_pool()
            result=paired_summary(self.evaluations)
            result.update(checkpoint=str(chosen),checkpoint_sha256=receipt['sha256'],
                actor_updates=self.actor_updates,new_actor_updates=self.actor_updates-self.start_updates,
                controls=self.budget.used,seconds=time.monotonic()-self.started,training_stop_reason=self.stop_reason)
            save(OUT/'result.json',result)
            self.phase='completed'; self.status(paired_result=result)
        except BaseException as error:
            self.phase='failed'; self.status(error=repr(error))
            save(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),controls=self.budget.used))
            raise
        finally:
            self.close_pool(strict=False); self.eval_log.close(); self.log.close()


if __name__=='__main__':
    def stop(signum,frame): raise SystemExit(f'Owned E2 stopped by signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    MethodExperiment().run()
