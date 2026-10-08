"""One persistent native SFT server and serial official task evaluators per GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from common import ROOT, SOURCE, SIM_ROOT, MODEL_PYTHON, G05, bootstrap, atomic_json
bootstrap()
from probe_task_catalog import anonymous_gib


def stop_child(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)


def acquire_loader(job):
    deadline=time.time()+1800
    while time.time()<deadline and not (job/'STOP').exists():
        if anonymous_gib()<160:
            for i in range(2):
                f=open(job/f'loader_{i}.lock','a')
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
                except BlockingIOError:f.close()
        time.sleep(3)
    raise TimeoutError('Loading admission timeout/stop')


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();job=args.job;gpu=args.gpu
    manifest=json.loads((job/'manifest.json').read_text())
    tasks=['turning_on_radio'] if args.smoke else manifest['groups'][gpu]['tasks']
    run=job/('smoke' if args.smoke else 'workers')/f'gpu_{gpu}'
    run.mkdir(parents=True,exist_ok=False)
    lock=open(SIM_ROOT/f'gpu_{gpu}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='starting',pid=os.getpid(),gpu=gpu,source=manifest['source_commit'],completed_tasks=[])
    def publish():state['updated']=time.time();atomic_json(run/'status.json',state)
    def stopped():return (job/'STOP').exists()
    policy=sim=None;loader=None
    port=19600+gpu
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('RL_'):env.pop(k)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        PYTHONPATH=str(G05/'src')+':'+str(SOURCE))
    try:
        loader=acquire_loader(job)
        with (run/'policy.log').open('x') as f:
            policy=subprocess.Popen([str(MODEL_PYTHON),str(SOURCE/'serve.py'),'--run',str(run),'--port',str(port)],
                env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state['policy_pid']=policy.pid;publish();load_deadline=time.time()+1200
        while not (run/'policy.ready').exists():
            if policy.poll() is not None:raise RuntimeError('SFT policy load failed')
            if time.time()>load_deadline or stopped():raise TimeoutError('Policy load timeout/stop')
            time.sleep(3)
        loader.close();loader=None
        for task in tasks:
            if stopped():raise InterruptedError('Requested evaluation stop')
            if __import__('shutil').disk_usage(job).free<150*2**30:
                raise RuntimeError('Disk safety reserve reached; preserving all evidence')
            output=job/('smoke_tasks' if args.smoke else 'tasks')/task
            output.mkdir(parents=True,exist_ok=False)
            loader=acquire_loader(job)
            sim_env=env|dict(EVAL_GPU=str(gpu),EVAL_SOURCE=str(SOURCE))
            command=['bash',str(SOURCE/'launch_sim.sh'),str(SOURCE/'run_task.py'),
                '--task',task,'--output',str(output),'--policy-run',str(run),'--port',str(port)]
            if args.smoke:command.append('--smoke')
            atomic_json(output/'command.json',dict(argv=command,source_commit=manifest['source_commit'],
                official_commit=manifest['official_commit'],environment_overrides={
                    k:sim_env[k] for k in ['EVAL_GPU','EVAL_SOURCE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
            with (output/'sim.log').open('x') as f:
                sim=subprocess.Popen(command,env=sim_env,stdin=subprocess.DEVNULL,stdout=f,
                    stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='smoke' if args.smoke else 'evaluating',task=task,sim_pid=sim.pid,started_task=time.time())
            publish();last_progress=time.time();previous_request=None
            while sim.poll() is None:
                if loader is not None and (output/'scene.ready').exists():loader.close();loader=None
                if stopped():raise InterruptedError('Requested evaluation stop')
                if policy.poll() is not None:raise RuntimeError('Policy exited while evaluating')
                ps=json.loads((run/'policy_status.json').read_text())
                if ps['status']=='failed':raise RuntimeError('Policy failed: '+str(ps.get('error')))
                if ps.get('requests')!=previous_request:
                    previous_request=ps.get('requests');last_progress=time.time()
                if time.time()-last_progress>1800:raise TimeoutError('No policy progress for 30 minutes')
                if anonymous_gib()>210:raise RuntimeError('Host anonymous RAM safety limit')
                if __import__('shutil').disk_usage(job).free<150*2**30:raise RuntimeError('Disk safety limit')
                time.sleep(5)
            if loader is not None:loader.close();loader=None
            result=json.loads((output/'status.json').read_text())
            records=list((output/'json').glob('*.json'));videos=list((output/'videos').glob('*.mp4'))
            expected=2 if args.smoke else 10
            if sim.returncode or result.get('status')!='completed' or len(records)!=expected or len(videos)!=expected:
                raise RuntimeError('Official task evaluator did not complete with all metrics/videos: '+task)
            for video in videos:
                if video.stat().st_size<1000:raise RuntimeError('Empty official video: '+str(video))
            state['completed_tasks'].append(task);publish()
        state['status']='completed'
    except BaseException as e:
        state.update(status='failed',error=repr(e));raise
    finally:
        if loader is not None:loader.close()
        stop_child(sim)
        if policy is not None and policy.poll() is None:
            (run/'STOP').touch()
            try:policy.wait(timeout=180)
            except subprocess.TimeoutExpired:stop_child(policy)
        receipt=run/'evaluation_ack.json'
        if receipt.exists():state['verification']=json.loads(receipt.read_text())
        if state.get('status')=='completed' and not state.get('verification',{}).get('weights_unchanged'):
            state.update(status='failed',error='Missing final exact weight fingerprint verification')
        state['finished']=time.time();publish();lock.close()
    if state['status']!='completed':raise RuntimeError(state.get('error','Failed verification'))


if __name__=='__main__':main()
