"""A800 shared short-skill PPO with fixed high-level intents and paired probes.

Current RGB/proprio -> frozen VLM -> trainable native FM action expert. No
high-memory, reward truth, outcome labels or simulator state enters the actor.
An update happens only when all registered skill episodes have ended under the
same behavior version; evaluation episodes never become optimization targets.
"""
import argparse
import asyncio
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/eval/memlite_sft100'),str(Path(__file__).resolve().parents[1]/'code')]
from common import atomic_json
from recovery_corpus import digest,file_sha
from skill_training_protocol import SkillTrainingSession
from skill_policy_adapter import SingleFrameSkillAdapter
from skill_rounds import SkillRounds
from skill_cold_worker import validate_baseline_acceptance,finished_workers
from wire import packb,unpackb


async def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--port',type=int,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    if cfg.get('schema')!='short_skill_rl_a800_v1' or cfg.get('user_goal_authorized') is not True:
        raise ValueError('Explicit short-skill goal recipe required')
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Use a new run and clean frozen source')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Select one actually idle A800, do not displace another process')
    cases=cfg['cases']
    if (len({json.loads(v['semantic_bundle'])[0]['verb'] for v in cases.values()})<3
            or any(v['original_split']!='train' or v['recovery_split']!='train' for v in cases.values())):
        raise ValueError('Need three admitted TRAIN skill mechanisms, never DEV/public test')
    if file_sha(root/cfg['model']['path'])!=cfg['model']['sha256']:raise ValueError('Changed SFT parent')
    import numpy as np
    import torch
    import websockets
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.utils.training.stage1_runtime import init_wandb,disk_has_reserve
    from g05.models.g05.qwen35 import vision
    from direct_a4_flow import A4DirectPPO
    vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    a.output.mkdir(parents=True);checkpoints=a.output/'checkpoints';checkpoints.mkdir()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    receipt=dict(status='loading',pid=os.getpid(),source_commit=commit,config_sha256=file_sha(a.config),
        parent_sha256=cfg['model']['sha256'],reward_protocol='skill_aligned_v1',optimizer_updates=0,
        action_format='native_real23_model27_pad_7_8_17_18',high_layer_fixed=True,
        protected_progress_contract_tested=False,whole_task_sr_evaluated=False)
    atomic_json(a.output/'status.json',receipt)
    names=json.loads((root/cfg['expert_release']/'manifest.json').read_text())['task_names']
    config=configuration(root,'low',names)
    if file_sha(config['stats_path'])!=cfg['stats_sha256']:raise ValueError('Normalizer changed')
    saved=torch.load(root/cfg['model']['path'],map_location='cpu',mmap=True,weights_only=False)
    policy,restoration=restore_model(config,'low',state=saved['model_state_dict'])
    del saved;gc.collect();policy.cuda().eval();policy.model.action_expert.float()
    adapter=SingleFrameSkillAdapter(make_processor(config,False),config)
    pad=torch.zeros(27,dtype=torch.bool);pad[[7,8,17,18]]=True
    trainer=A4DirectPPO(policy,pad,checkpoints,reward_protocol='skill_aligned_v1',**cfg['ppo'])
    if len(trainer.actor_parameters)!=322:raise ValueError('Changed actor gradient boundary')
    trainer.checkpoint_extra=dict(parent_sha256=cfg['model']['sha256'],source_commit=commit,
        stats_sha256=cfg['stats_sha256'],case_contracts_sha256=digest(cases),high_layer_fixed=True)
    resumed = None
    if cfg.get('resume') is not None:
        resume = cfg['resume'];resume_path = root/resume['path']
        if file_sha(resume_path)!=resume['sha256']:
            raise ValueError('Changed immutable resume checkpoint')
        resumed = trainer.load_checkpoint(resume_path,expected_bindings=dict(
            parent_sha256=cfg['model']['sha256'],stats_sha256=cfg['stats_sha256'],
            case_contracts_sha256=digest(cases),high_layer_fixed=True,updates=resume['updates']))
        receipt['resume'] = resumed
        trainer.checkpoint_extra['resumed_from_sha256'] = resume['sha256']
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('_short_skill_digests',REPO/'scripts/train_memlite_recovery.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    frozen_before=mod.parameter_digest(policy,frozen=True)
    rounds=SkillRounds(cases,cfg['evaluation_seeds'],rounds=cfg['learning_rounds'],run=str(a.output))
    version=trainer.update_count
    policy_sha=cfg['resume']['sha256'] if resumed else cfg['model']['sha256']
    sessions={};next_eval_id=10**9
    mutex=asyncio.Lock();finished=asyncio.Event();fatal=[None]
    cold_workers=cfg.get('simulator_reset_protocol')=='fresh_process_each_episode_v1'
    finish_acknowledged=set();baseline_approved=not cfg.get('require_baseline_acceptance',False)
    if not baseline_approved and (not cold_workers or len(cfg['evaluation_seeds'])!=2 or len(cases)!=3):
        raise ValueError('The six-start acceptance protocol requires three cold workers and two fixed seeds')
    wb=init_wandb(dict(cfg['wandb'],name=a.output.name),a.output,run_id=digest([str(a.output),commit])[:12],
        resume=False,metadata=dict(receipt,restoration=restoration,case_names=sorted(cases),
            trainable_tensors=322,evaluation_actor_dtype='FP32 deterministic Euler; same solver before/after'))
    def publish():
        receipt.update(status='finished' if rounds.phase=='finished' else 'ready',phase=rounds.phase,
            round=rounds.round,policy_version=version,policy_identity_sha256=policy_sha,
            optimizer_updates=trainer.update_count,active_episodes=len(sessions),
            baseline_approved=baseline_approved,finish_acknowledged=sorted(finish_acknowledged),
            jobs=rounds.jobs,wandb_url=wb.url,updated_unix=time.time())
        atomic_json(a.output/'status.json',receipt)
    def goal(case):return {k:case[k] for k in ('task','parent_goal','semantic_bundle')}

    @torch.no_grad()
    def act(session,job,observation):
        nonlocal next_eval_id
        target=goal(session.case);prepared,anchor=adapter.prepare(observation,target);batch=adapter.collate([prepared])
        chunk=(session.rollout.step-session.case['start_control'])//16
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(job['seed']+1000003*chunk)
            with adapter._branch_context():
                if job['phase']=='train':
                    result=trainer.sample_branch(batch,[observation],[target])
                    actions=result['action'];eid=result['experience_ids'][0];value=float(result['old_value'][0])
                else:
                    state,_=trainer._prefix(batch);fm=trainer.fm
                    dummy=torch.zeros(1,fm.horizon_steps,fm.action_dim,device='cuda',dtype=torch.float32)
                    actions=fm._sample_noise(dummy,torch.float32,[batch['samples'][0].get('embodiment')])
                    actions[...,pad.cuda()]=0;velocity=trainer._velocity_fn(state)
                    pi=str(fm.time_convention)=='pi_convention';n=int(fm.num_inference_steps)
                    for i in range(n):
                        times=torch.full((1,),(n-i)/n if pi else i/n,device='cuda',dtype=torch.float32)
                        actions=actions+(-1. if pi else 1.)/n*velocity(actions,times)
                        actions[...,pad.cuda()]=0
                    eid=next_eval_id;next_eval_id+=1;value=0.
        raw=adapter.postprocess(actions,batch,anchor)
        session.emit(raw,experience_id=eid,old_value=value)
        return dict(action_chunk=raw,experience_id=eid,old_value=value,
            max_controls=session.rollout.pending['max_controls'],identity=asdict(session.identity),
            control_step=session.rollout.step,policy_version=version,policy_sha256=policy_sha)

    @torch.no_grad()
    def next_value(session,job,observation):
        if session.last['terminated'] or job['phase']!='train':return 0.
        batch=adapter.prepare_training_batch([observation],[goal(session.case)])
        with adapter._branch_context():_,features=trainer._prefix(batch)
        return float(trainer.critic(features)[0])

    def shared_update():
        nonlocal version,policy_sha
        if sessions or not rounds.ready or rounds.phase!='train' or not disk_has_reserve(root):
            raise ValueError('Incomplete update barrier or insufficient shared disk reserve')
        targets=rounds.targets
        if set(trainer.experiences)!={r['experience_id'] for r in targets}:
            raise ValueError('Dropped, duplicated or cross-version on-policy experience')
        metrics=trainer.update(adapter,[r['experience_id'] for r in targets],
            [r['advantage'] for r in targets],[r['returns'] for r in targets])
        saved=trainer._save_checkpoint(metrics)
        # Preserve each scientific round checkpoint by hard link, independent
        # of the existing latest/previous publication pointers.
        os.link(saved['latest'],checkpoints/f"round-{rounds.round:04d}.pt")
        version=trainer.update_count;policy_sha=saved['latest_sha256']
        wb.log({'train/update':version,'train/episodes':len(rounds.cases),
            **{f'train/{k}':metrics[k] for k in ('experiences','policy_loss_before','value_loss_before',
                'value_loss_after','accepted_actor_lr','max_reference_tensor_change')},
            'train/post_update_kl':metrics['post_update']['mean_approx_kl']})
        atomic_json(a.output/f'update-{version:04d}.json',dict(metrics,policy_identity_sha256=policy_sha))

    async def handler(socket):
        nonlocal baseline_approved
        current_job=None
        await socket.send(packb(dict(protocol='short_skill_rl_v1',config_sha256=file_sha(a.config))))
        try:
            async for data in socket:
                request=unpackb(data)
                async with mutex:
                    if fatal[0]:raise RuntimeError(fatal[0])
                    op=request.get('op')
                    if op=='job':
                        if set(request)!={'op','case'} or current_job is not None:raise ValueError('Overlapping worker job')
                        if request['case'] not in cases:raise ValueError('Unregistered worker')
                        if rounds.phase=='finished':
                            response=dict(status='finished')
                            finished_workers(cases,finish_acknowledged,request['case'])
                        elif rounds.phase=='train' and rounds.round==1 and not baseline_approved:
                            approval=a.output/'BASELINE_ACCEPTED.json'
                            if approval.exists():
                                validate_baseline_acceptance(json.loads(approval.read_text()),
                                    config_sha256=file_sha(a.config),policy_sha256=policy_sha,
                                    episodes_sha256=file_sha(a.output/'episodes.jsonl'))
                                baseline_approved=True
                            response=dict(status='wait')
                        else:
                            current_job=rounds.take(request['case'])
                            response=(dict(status='wait') if current_job is None else
                                dict(status='job',job=current_job,policy_version=version,policy_sha256=policy_sha,
                                     session=str(a.output),case=cases[current_job['case']]))
                    elif op=='begin':
                        if (set(request)!={'op','job_id','initial_evidence'} or current_job is None
                                or request['job_id']!=current_job['id'] or request['job_id'] in sessions):
                            raise ValueError('Unbound/duplicate real reset')
                        session=SkillTrainingSession(cases[current_job['case']],session=str(a.output),episode=current_job['id'],
                            policy_version=version,policy_sha256=policy_sha,initial_evidence=request['initial_evidence'])
                        sessions[current_job['id']]=session
                        response=dict(status='begun',identity=asdict(session.identity),policy_version=version,policy_sha256=policy_sha)
                    elif op=='action':
                        if current_job is None:raise ValueError('No active job')
                        session=sessions[current_job['id']];observation=session.action_input(request)
                        response=await asyncio.to_thread(act,session,current_job,observation)
                    elif op=='ack':
                        if current_job is None:raise ValueError('No active job')
                        session=sessions[current_job['id']];observation,rewards=session.ack(request)
                        value=await asyncio.to_thread(next_value,session,current_job,observation)
                        transition=session.finish_ack(value)
                        with (a.output/'reward-ledger.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(job=current_job,transition=transition,rewards=rewards))+'\n')
                        response=dict(status='acknowledged',ended=session.rollout.ended,control_step=session.rollout.step,
                            final_reward=rewards[-1],identity=asdict(session.identity),policy_version=version,policy_sha256=policy_sha)
                        if session.rollout.ended:
                            targets=session.training_targets() if current_job['phase']=='train' else []
                            result=dict(job=current_job,success=session.last['skill_success'],outcome=session.last['outcome'],
                                actual_controls=session.rollout.step-session.case['start_control'],
                                policy_version=version,policy_sha256=policy_sha,whole_task_sr=False)
                            with (a.output/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
                            wb.log({'train/update':version,f"{current_job['phase']}/{current_job['case']}/success":int(result['success']),
                                    f"{current_job['phase']}/{current_job['case']}/controls":result['actual_controls']})
                            del sessions[current_job['id']];rounds.complete(current_job['id'],targets);current_job=None
                            if rounds.ready:
                                update=rounds.phase=='train'
                                if update:await asyncio.to_thread(shared_update)
                                rounds.advance(optimizer_completed=update)
                    else:raise ValueError('Unknown short-skill operation')
                    publish();await socket.send(packb(response))
                    # The last simulator must receive its final real-control
                    # ACK before the server closes the listening context.
                    if rounds.phase=='finished' and (not cold_workers or finish_acknowledged==set(cases)):
                        finished.set()
        except websockets.ConnectionClosed:
            if current_job is not None:fatal[0]='Worker disconnected mid-episode';finished.set()
        except BaseException as error:
            fatal[0]=repr(error);finished.set()
            with (a.output/'errors.jsonl').open('a') as stream:stream.write(json.dumps(dict(error=repr(error),job=current_job))+'\n')
            try:await socket.send(packb(dict(error=repr(error))))
            except Exception:pass

    async def stop_watch():
        while not (a.output/'STOP').exists():await asyncio.sleep(2)
        fatal[0]='Owner stop requested; retain completed checkpoint and partial evidence';finished.set()

    async with websockets.serve(handler,'127.0.0.1',a.port,compression=None,max_size=32<<20,ping_timeout=None):
        publish();watch=asyncio.create_task(stop_watch());await finished.wait();watch.cancel()
    frozen_after=mod.parameter_digest(policy,frozen=True)
    if frozen_after!=frozen_before:fatal[0]='Frozen VLM/LoRA changed during short-skill PPO'
    receipt.update(status='failed' if fatal[0] else 'completed_training_window_not_promoted',error=fatal[0],
        frozen_before_sha256=frozen_before,frozen_after_sha256=frozen_after,
        optimizer_updates=trainer.update_count,actor_updates=trainer.actor_update_count)
    atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt);wb.finish()
    if fatal[0]:raise RuntimeError(fatal[0])


if __name__=='__main__':asyncio.run(main())
