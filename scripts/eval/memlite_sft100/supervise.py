"""Count-bounded 1,000-case evaluation. No automatic repeated public attempts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from common import ROOT,SOURCE,REPO,atomic_json
from summarize import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();job=a.job.resolve();manifest=json.loads((job/'manifest.json').read_text())
    if job.parent!=ROOT/'runs' or manifest['episodes']!=1000:raise ValueError('Unregistered full evaluation')
    actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if actual!=manifest['source_commit']:raise ValueError('Source identity changed')
    if (job/'status.json').exists():raise ValueError('No automatic restart of an existing evaluation')
    smoke=json.loads((job/'smoke/gpu_1/status.json').read_text())
    if smoke['status']!='completed' or not smoke['verification']['weights_unchanged']:
        raise ValueError('Real native-FM protocol smoke has not passed')
    repaired=ROOT/'runs/large_scale73h_repaired_20261008_canary_v1'
    canary=json.loads((repaired/'status.json').read_text())
    if canary['status']!='canary_validated_paused_for_sft_evaluation':
        raise ValueError('RL repair canary has not completed and saved')
    # Admission before creating any policy. No unrelated GPU job may be displaced.
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('GPU processes still active; do not start full evaluation: '+apps)
    lock=open(job/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',pid=os.getpid(),started=time.time(),source_commit=actual,completed_gpus=[])
    children={}
    try:
        for gpu in range(8):
            with (job/f'gpu_{gpu}.log').open('x') as f:
                children[gpu]=subprocess.Popen(['python3',str(SOURCE/'worker.py'),'--job',str(job),'--gpu',str(gpu)],
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state['worker_pids']={str(k):v.pid for k,v in children.items()}
        atomic_json(job/'status.json',state)
        while children:
            if (job/'STOP').exists():raise InterruptedError('User or safety stop')
            for gpu,child in list(children.items()):
                if child.poll() is None:continue
                result=json.loads((job/f'workers/gpu_{gpu}/status.json').read_text())
                if child.returncode or result['status']!='completed':
                    raise RuntimeError('Evaluation worker failed: '+json.dumps(result))
                state['completed_gpus'].append(gpu);del children[gpu]
            summary=summarize(job)
            state.update(updated=time.time(),completed_episodes=summary['completed'])
            atomic_json(job/'status.json',state)
            time.sleep(30)
        report=summarize(job,final=True)
        if report['completed']!=1000:raise RuntimeError('Not all1000 registered cases completed')
        state.update(status='completed',q_score=report['official_q_score'],success_rate=report['official_sr'])
    except BaseException as e:
        state.update(status='needs_diagnosis',error=repr(e));(job/'STOP').touch()
        raise
    finally:
        deadline=time.time()+240
        while any(c.poll() is None for c in children.values()) and time.time()<deadline:time.sleep(3)
        state['finished']=time.time();atomic_json(job/'status.json',state);lock.close()


if __name__=='__main__':main()
