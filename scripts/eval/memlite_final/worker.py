"""One persistent frozen model and official task batches on one owned GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path[:0]=[str(HERE.parent/'memlite_sft100'),str(HERE)]
from common import ROOT,SIM_ROOT,MODEL_PYTHON,atomic_json,sha256
from lifecycle import episode_portions


def stop_child(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=40)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def loader_lock(job):
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline and not (job/'STOP').exists():
        for i in range(2):
            stream=(job/f'loader_{i}.lock').open('a')
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);return stream
            except BlockingIOError:stream.close()
        time.sleep(2)
    raise TimeoutError('Scene/model loading admission timed out or stopped')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--num-envs',type=int,choices=(2,4),required=True);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();job=a.job.resolve();manifest=json.loads((job/'manifest.json').read_text())
    if (job.parent!=ROOT/'runs' or manifest['schema']!='memlite_final_public100_run_v1' or not 0<=a.gpu<8
            or manifest['source_commit']!=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
            or sha256(manifest['config_path'])!=manifest['config_sha256']):
        raise ValueError('Changed or foreign frozen full-evaluation registration')
    if a.preflight:
        path=job/'preflight'/f'batch{a.num_envs}'
        tasks=[dict(task='turning_on_radio'),dict(task='clean_a_keyboard')]
        # Never assume a human-memory task number for a transport identity.
        inverse={c['task']:c['task_id'] for c in manifest['cases']}
        for task in tasks:task['task_id']=inverse[task['task']]
    else:
        accepted=json.loads((job/'accepted-preflight.json').read_text())
        if accepted['source_commit']!=manifest['source_commit'] or accepted['num_envs']!=a.num_envs:
            raise ValueError('No matching completed TRAIN engineering preflight')
        path=job/'workers'/f'gpu_{a.gpu}';tasks=manifest['groups'][a.gpu]['tasks']
    path.mkdir(parents=True,exist_ok=False)
    lock=(SIM_ROOT/f'gpu_{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='loading',pid=os.getpid(),gpu=a.gpu,num_envs=a.num_envs,preflight=a.preflight,
        source_commit=manifest['source_commit'],started=time.time(),completed_parts=[],control_steps=0,run_seconds=0.)
    state.update(config_sha256=manifest['config_sha256'],pack_manifest_sha256=manifest['pack_manifest_sha256'])
    def publish():state['updated']=time.time();atomic_json(path/'status.json',state)
    model=sim=loader=None
    port=19900+a.gpu
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('RL_'):env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        PYTHONPATH=str(REPO/'src'))
    try:
        loader=loader_lock(job)
        with (path/'model.log').open('x') as log:
            model=subprocess.Popen([str(MODEL_PYTHON),str(HERE/'serve.py'),'--config',manifest['config_path'],
                '--pack',manifest['pack'],'--manifest-sha256',manifest['pack_manifest_sha256'],
                '--run',str(path/'model'),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        state['model_pid']=model.pid;publish();deadline=time.monotonic()+1800
        while not (path/'model/READY').exists():
            if model.poll() is not None:raise RuntimeError('Latest model loading failed')
            if time.monotonic()>deadline or (job/'STOP').exists():raise TimeoutError('Model loading timeout/stop')
            time.sleep(3)
        loader.close();loader=None
        for task in tasks:
            portions=episode_portions(a.num_envs,preflight=a.preflight)
            for count,indices in portions:
                if (job/'STOP').exists():raise InterruptedError('Requested administrative stop')
                if shutil.disk_usage(job).free<150<<30:raise RuntimeError('Preserve disk safety reserve')
                part=f"{task['task']}-n{count}-i{indices[0]}"
                output=(path/'parts'/part) if a.preflight else job/'parts'/part
                output.parent.mkdir(parents=True,exist_ok=True)
                loader=loader_lock(job)
                command=['bash',str(HERE.parent/'memlite_sft100/launch_sim.sh'),str(HERE/'run_task.py'),
                    '--task',task['task'],'--task-id',str(task['task_id']),'--output',str(output),
                    '--port',str(port),'--config-sha256',manifest['config_sha256'],'--run-id',job.name+('-preflight'+str(count) if a.preflight else ''),
                    '--num-envs',str(count),'--indices',*map(str,indices)]
                if a.preflight:command+=['--train-smoke','--smoke-steps','256']
                sim_env=env|dict(EVAL_GPU=str(a.gpu),EVAL_SOURCE=str(HERE))
                with (path/(part+'.log')).open('x') as log:
                    sim=subprocess.Popen(command,env=sim_env,stdout=log,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,start_new_session=True)
                atomic_json(path/(part+'-command.json'),dict(argv=command,official_commit=manifest['official_commit']))
                state.update(status='evaluating',sim_pid=sim.pid,part=part);publish()
                last=time.monotonic();previous=None
                while sim.poll() is None:
                    if loader is not None and (output/'scene.ready').exists():loader.close();loader=None
                    if (job/'STOP').exists():raise InterruptedError('Requested stop')
                    if model.poll() is not None:raise RuntimeError('Model service exited')
                    service=json.loads((path/'model/status.json').read_text())
                    if service['status']=='failed':raise RuntimeError('Model error: '+service.get('error',''))
                    if service.get('requests')!=previous:previous=service.get('requests');last=time.monotonic()
                    if time.monotonic()-last>1800:raise TimeoutError('No model progress for 30 minutes')
                    if shutil.disk_usage(job).free<150<<30:raise RuntimeError('Disk safety stop')
                    time.sleep(5)
                if loader is not None:loader.close();loader=None
                result=json.loads((output/'status.json').read_text())
                if sim.returncode or result['status']!='completed':raise RuntimeError('Official task batch incomplete: '+part)
                records=list((output/'json').glob('*.json'));videos=list((output/'videos').glob('*.mp4'))
                if len(records)!=len(indices) or len(videos)!=len(indices) or any(v.stat().st_size<1000 for v in videos):
                    raise RuntimeError('Missing original metrics or videos: '+part)
                for attempt in (output/'attempts').glob('*.json'):
                    r=json.loads(attempt.read_text());state['run_seconds']+=r['seconds'];state['control_steps']+=sum(r['actual_applied_controls'])
                state['completed_parts'].append(str(output));publish()
        state['status']='completed'
    except BaseException as e:
        state.update(status='failed_no_retry',error=repr(e));raise
    finally:
        if loader is not None:loader.close()
        stop_child(sim)
        if model is not None and model.poll() is None:
            (path/'model/STOP').touch()
            deadline=time.monotonic()+180
            while model.poll() is None and time.monotonic()<deadline:time.sleep(3)
            stop_child(model)
        final=path/'model/weights-final.json'
        if final.exists():state['verification']=json.loads(final.read_text())
        if state['status']=='completed' and not state.get('verification',{}).get('weights_unchanged'):
            state.update(status='failed_no_retry',error='Missing exact frozen-model final verification')
        state['control_steps_per_second']=state['control_steps']/state['run_seconds'] if state['run_seconds'] else None
        state['finished']=time.time();publish();lock.close()
    if state['status']!='completed':raise RuntimeError('Worker verification failed')


if __name__=='__main__':main()
