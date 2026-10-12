"""Start eight owned workers only after the latest-model TRAIN preflight."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path[:0]=[str(HERE.parent/'memlite_sft100'),str(HERE)]
from common import ROOT,atomic_json,sha256
from report import report


def select_preflight(job, manifest):
    candidates={}
    for size in (2,4):
        path=job/'preflight'/f'batch{size}'/'status.json'
        if not path.is_file():continue
        value=json.loads(path.read_text())
        if value['status']!='completed':continue
        if (value['source_commit']!=manifest['source_commit'] or value['num_envs']!=size
                or value['config_sha256']!=manifest['config_sha256']
                or value['pack_manifest_sha256']!=manifest['pack_manifest_sha256']
                or value['verification']['config_sha256']!=manifest['config_sha256']
                or value['preflight'] is not True or not value['verification']['weights_unchanged']
                or value['verification']['optimizer_updates']!=0 or len(value['completed_parts'])!=2
                or not value['control_steps_per_second']>0):
            raise ValueError('Mismatched actual latest-model engineering receipt')
        candidates[size]=(value,dict(path=str(path),sha256=sha256(path)))
    if 2 not in candidates:raise ValueError('Latest-model batch2 engineering preflight is required')
    size=4 if 4 in candidates and candidates[4][0]['control_steps_per_second']>candidates[2][0]['control_steps_per_second'] else 2
    return dict(source_commit=manifest['source_commit'],num_envs=size,
        evidence={str(k):v[1] for k,v in candidates.items()},
        selection='Higher total applied-controls/sec over the same two TRAIN tasks; no public tuning')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();job=a.job.resolve();manifest=json.loads((job/'manifest.json').read_text())
    if (job.parent!=ROOT/'runs' or manifest['schema']!='memlite_final_public100_run_v1'
            or (job/'status.json').exists() or (job/'STOP').exists()
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=manifest['source_commit']):
        raise ValueError('New frozen registered run, never implicit restart')
    accepted=select_preflight(job,manifest)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise ValueError('GPU processes active; do not displace any job: '+apps)
    lock=(job/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(job/'accepted-preflight.json',accepted)
    state=dict(status='running',pid=__import__('os').getpid(),source_commit=manifest['source_commit'],
        started=time.time(),num_envs=accepted['num_envs'],completed_gpus=[],failed_gpus={})
    children={}
    try:
        for gpu in range(8):
            with (job/f'gpu_{gpu}.log').open('x') as log:
                children[gpu]=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--job',str(job),'--gpu',str(gpu),
                    '--num-envs',str(accepted['num_envs'])],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['worker_pids']={str(k):c.pid for k,c in children.items()};atomic_json(job/'status.json',state)
        while children:
            if (job/'STOP').exists():raise InterruptedError('Administrative/safety stop')
            for gpu,child in list(children.items()):
                if child.poll() is None:continue
                path=job/f'workers/gpu_{gpu}/status.json'
                row=json.loads(path.read_text()) if path.exists() else dict(status='missing_status')
                if child.returncode or row['status']!='completed':
                    state['failed_gpus'][str(gpu)]=row
                    raise RuntimeError('Worker failed; preserve all evidence, no automatic reroll')
                state['completed_gpus'].append(gpu);del children[gpu]
            r=report(job);state.update(completed_episodes=r['completed'],updated=time.time());atomic_json(job/'status.json',state)
            if children:time.sleep(30)
        r=report(job,final=True)
        state.update(status='completed',q_score=r['official_q_score'],success_rate=r['official_sr'])
    except BaseException as e:
        state.update(status='needs_diagnosis',error=repr(e));(job/'STOP').touch();raise
    finally:
        # Workers own their child process groups and save their final weight
        # receipts. Never kill another user's GPU process from this monitor.
        deadline=time.monotonic()+240
        while any(c.poll() is None for c in children.values()) and time.monotonic()<deadline:time.sleep(3)
        state['finished']=time.time();atomic_json(job/'status.json',state);lock.close()


if __name__=='__main__':main()
