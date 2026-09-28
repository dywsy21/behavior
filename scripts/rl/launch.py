"""One-shot supervisor; owns only this experiment's new process group."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from common import REPO,OUT,MODEL_PY,commit,save


def model_env():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(REPO/'src')+':'+str(REPO))
    npp='/mnt/sdc1/robodojo/GalaxeaVLA/.venv/lib/python3.10/site-packages/nvidia/npp/lib'
    env['LD_LIBRARY_PATH']=':'.join([npp,*[p for p in env.get('LD_LIBRARY_PATH','').split(':') if p and p!=npp]])
    return env


def supervise(entry='learner'):
    start=time.monotonic(); process=None; result=dict(source_commit=commit(),supervisor=os.getpid(),status='starting')
    wall=json.loads((OUT/'manifest.json').read_text())['max_active_wall_seconds']
    try:
        # Recheck immediately before GPU use. Never compete with an observed job.
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        values={int(a):int(b) for a,b in (line.split(',') for line in usage.strip().splitlines())}
        if any(values[gpu]>512 for gpu in (0,2,3)): raise RuntimeError('Registered GPUs are not idle')
        with (OUT/'learner.stdout.log').open('x') as log:
            process=subprocess.Popen([MODEL_PY,str(REPO/f'scripts/rl/{entry}.py')],cwd=REPO,env=model_env(),
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result.update(status='running',learner=process.pid); save(OUT/'supervisor.json',result,replace=True)
        try: code=process.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            result['reason']='registered_wall_limit'; os.killpg(process.pid,signal.SIGTERM)
            try: code=process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); code=process.wait(timeout=10)
        result.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=time.monotonic()-start)
    except BaseException as error:
        result.update(status='failed',error=repr(error),seconds=time.monotonic()-start)
        if process is not None and process.poll() is None: os.killpg(process.pid,signal.SIGTERM)
        raise
    finally:
        save(OUT/'supervisor.json',result,replace=True)


def launch(entry='learner'):
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['source_commit']!=commit(): raise ValueError('Frozen source differs from manifest')
    if manifest.get('entry','learner')!=entry: raise ValueError('Registered experiment entry differs')
    save(OUT/'launch_claim.json',dict(source_commit=commit(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    with (OUT/'supervisor.stdout.log').open('x') as log:
        process=subprocess.Popen([sys.executable,str(REPO/'scripts/rl/launch.py'),'--supervise','--entry',entry],cwd=REPO,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=process.pid,output=str(OUT))))


if __name__=='__main__':
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--launch',action='store_true');g.add_argument('--supervise',action='store_true')
    p.add_argument('--entry',choices=('learner','offline_update','precision_probe'),default='learner')
    args=p.parse_args(); (launch if args.launch else supervise)(args.entry)
