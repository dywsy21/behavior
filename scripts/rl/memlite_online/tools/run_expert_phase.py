"""One independent GPU expert for one bounded baseline/train/final phase."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from bootstrap import bootstrap
bootstrap()

ROOT=Path('/run/ti/rl_memlite_stage1_20261006')
SNAP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SNAP/'code'))
from gpu_admission import check_gpu_available
from probe_task_catalog import anonymous_gib,stop_owned


def write(path,state):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2))
    temporary.replace(path)


def stopped(root):
    return any(p.exists() for p in (root/'STOP_TRAINING',ROOT/'STOP_TRAINING',ROOT/'reports/STOP_TRAINING'))


def load_slot(root,deadline):
    """Two simultaneous model/scene loads across the entire formal job."""
    while time.time()<deadline-300 and not stopped(root):
        if anonymous_gib()<160:
            for index in range(2):
                lock=open(root/f'load_slot_{index}.lock','a')
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    return lock
                except BlockingIOError:lock.close()
        time.sleep(2)
    raise TimeoutError('Loading admission deadline or user stop')


def terminate(signum,frame):
    raise RuntimeError('Expert received termination signal')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--phase',choices=['baseline','training','final'],required=True)
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--resume')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,terminate)
    job=Path(args.job);run=Path(args.run);run.mkdir(exist_ok=False)
    manifest=json.loads((job/'manifest.json').read_text())
    group=next(row for row in manifest['groups'] if row['gpu']==args.gpu)
    read_only=args.phase!='training'
    if args.phase=='baseline' and args.resume:raise ValueError('SFT baseline must use original weights')
    if args.phase=='final' and not args.resume:raise ValueError('Final comparison needs this expert checkpoint')
    state=dict(status='starting',started=time.time(),deadline=args.deadline,phase=args.phase,
               gpu=args.gpu,pid=os.getpid(),snapshot=str(SNAP),resume=args.resume,cycles=[])
    write(run/'status.json',state)
    lock=open(f'/run/ti/behavior_stage3_20260930/gpu_{args.gpu}.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state['admission']=check_gpu_available(args.gpu)
    env=os.environ | dict(CUDA_VISIBLE_DEVICES=str(args.gpu),RL_GPU=str(args.gpu),
        RL_JOB_ROOT=str(job),
        RL_POLICY_PORT=str(19400+args.gpu),RL_REWARD_PROTOCOL='shared_terminal_q_v1',
        RL_EVALUATION_ONLY='1' if read_only else '0',
        RL_SIM_DATA_ROOT=str(ROOT/'assets_readiness/sim_data'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.update(RL_TRAINING_CONFIG=str(SNAP/'training.json'),
        RL_SOURCE_COMMIT=manifest['repair_source_commit'],
        RL_RECOVERY_ROOT=str(job/'recovery'/f'gpu_{args.gpu}'))
    for key in ('RL_PREP_BARRIER_DIR','RL_RESUME_CHECKPOINT'):env.pop(key,None)
    if args.resume:env['RL_RESUME_CHECKPOINT']=args.resume
    policy=sim=None;slot=None
    try:
        slot=load_slot(job,args.deadline)
        service='serve_readonly_policy.py' if read_only else 'serve_stage1_rl.py'
        with (run/'policy.log').open('w') as output:
            policy=subprocess.Popen(['timeout','--kill-after=30',str(int(args.deadline-time.time())),
                'bash',str(SNAP/'tools/run_policy_python.sh'),str(SNAP/'tools'/service),
                '--run',str(run),'--port',env['RL_POLICY_PORT'],'--deadline',str(args.deadline)],
                env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        state['policy_pid']=policy.pid;write(run/'status.json',state)
        while not (run/'policy.ready').exists():
            if policy.poll() is not None:raise RuntimeError('Policy load failed')
            if time.time()>args.deadline-400 or stopped(job):raise TimeoutError('Policy load deadline/stop')
            time.sleep(2)
        cycle=int(manifest.get('resume_cursors',{}).get(str(args.gpu),0))
        # Shorter official horizons first ensures broad task coverage early.
        tasks=sorted(group['tasks'],key=lambda row:row['official_horizon_steps'])
        while time.time()<args.deadline-400 and not stopped(job):
            if read_only and cycle==len(tasks):break
            task=tasks[cycle%len(tasks)]
            epoch=cycle//len(tasks)
            if read_only:
                instances=task['development_instances'];seed=task['evaluation_seed_base']
            else:
                pool=task['train_instances'];offset=(2*epoch)%len(pool)
                instances=[pool[offset],pool[(offset+1)%len(pool)]]
                seed=171000+1000*task['task_index']+epoch
            sub=run/f'cycle_{cycle:05d}_{task["task"]}';sub.mkdir()
            if slot is None:slot=load_slot(job,args.deadline)
            cycle_env=env | dict(RL_EPISODE_STEPS=str(task['official_horizon_steps']),
                RL_SEED_BASE=str(seed),RL_TASK_FULL_SCENE='1' if task['template_mode']=='full_scene' else '0')
            episode_metadata=[dict(task=task['task'],task_index=task['task_index'],instance_id=int(instance),
                split='train',source_commit=manifest['repair_source_commit'],
                resume_checkpoint_sha256=manifest['resume_receipts'][str(args.gpu)]['sha256'],
                run=str(sub),gpu=args.gpu,cycle=cycle,seed_base=seed,
                model_parent='fduTristin/memlite-stage1',
                high_checkpoint_sha256=manifest['model']['checkpoints']['high/step_00048045_save_0027.pt']['sha256'],
                low_checkpoint_sha256=manifest['model']['checkpoints']['low/step_00098414_save_0021.pt']['sha256'])
                for instance in instances]
            cycle_env['RL_EPISODE_METADATA']=json.dumps(episode_metadata)
            info=dict(cycle=cycle,task=task['task'],instances=instances,seed_base=seed,
                      horizon=task['official_horizon_steps'],started=time.time(),run=str(sub))
            state.update(status='evaluating' if read_only else 'training',current=info)
            write(run/'status.json',state)
            with (sub/'sim.log').open('w') as output:
                sim=subprocess.Popen(['timeout','--kill-after=20',str(int(args.deadline-time.time()-300)),
                    'bash',str(SNAP/'tools/run_pilot_sim.sh'),str(sub),task['task'],*map(str,instances)],
                    env=cycle_env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            info['sim_pid']=sim.pid
            write(run/'status.json',state)
            while sim.poll() is None:
                if slot is not None and (sub/'scene.ready').exists():slot.close();slot=None
                ps=json.loads((run/'policy_status.json').read_text())
                if ps['status'] in ('failed','save_failed'):raise RuntimeError('Policy protection: '+str(ps))
                if policy.poll() is not None:raise RuntimeError('Policy exited while collector active')
                if anonymous_gib()>210:raise RuntimeError('Anonymous memory protection')
                if time.time()>args.deadline-310 or stopped(job):break
                time.sleep(3)
            if sim.poll() is None:
                info['budget_or_user_interrupted']=True
                stop_owned(sim)
                state['cycles'].append(info)
                break
            if slot is not None:slot.close();slot=None
            if sim.returncode or not (sub/(task['task']+'.completed')).exists():
                raise RuntimeError('Collector failed for '+task['task'])
            records=[json.loads(path.read_text()) for path in sub.glob('eval_*/json/*.json')]
            assert len(records)==2 and {row['instance_id'] for row in records}==set(instances)
            info.update(finished=time.time(),official_episodes=records)
            state['cycles'].append(info);write(run/'status.json',state)
            cycle+=1
        state['status']='completed' if not read_only or cycle==len(tasks) else 'incomplete_evaluation'
        if stopped(job):state['status']='user_stopped'
    except Exception as error:
        state.update(status='failed',error=repr(error))
    finally:
        if slot is not None:slot.close()
        if policy is not None and policy.poll() is None:
            (run/'STOP_SAVE').touch()
            try:policy.wait(timeout=max(1,min(240,args.deadline-time.time())))
            except subprocess.TimeoutExpired:pass
        if sim is not None:stop_owned(sim)
        if policy is not None:stop_owned(policy)
        receipt=run/('evaluation_ack.json' if read_only else 'save_ack.json')
        if receipt.exists():
            state['exit_receipt']=json.loads(receipt.read_text())
            verified=state['exit_receipt']
            if state['status']=='completed':
                valid=(verified.get('weights_unchanged') and verified.get('optimizer_steps')==0) if read_only else (
                    verified.get('updates',0)>0 and verified.get('finite') and verified.get('load_verified'))
                if not valid:state.update(status='failed',error='Exit receipt did not verify weights or saved update')
        elif state['status']=='completed':
            state.update(status='failed',error='Missing final verification/save receipt')
        state['finished']=time.time();write(run/'status.json',state);lock.close()
    if state['status'] not in ('completed','user_stopped'):
        raise RuntimeError(state.get('error',state['status']))


if __name__=='__main__':main()
