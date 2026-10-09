"""Run an explicit source inventory on idle RTX GPUs; retain every attempt.

No training, arbitrary wall-clock/reset quota, or automatic deletion. The
inventory is a coverage batch, not a global collection limit. Native failures
are recorded and do not silently turn into successful samples or retries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/eval/memlite_sft100'))
from common import atomic_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if len(set(a.gpus))!=len(a.gpus):raise ValueError('Duplicate GPU')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    for gpu in a.gpus:
        if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError(f'GPU {gpu} occupied; do not stop another workload')
    sources=json.loads((a.sources/'manifest.json').read_text())['cases']
    for case in sources:
        if sha256(a.sources/case['directory']/'manifest.json')!=case['manifest_sha256']:raise ValueError('Changed inventory')
    a.output.mkdir(parents=True);jobs=queue.Queue();lock=threading.Lock();loading=threading.Semaphore(2)
    rows=[]
    for case in sources:jobs.put(case)
    def worker(gpu):
        while True:
            try:case=jobs.get_nowait()
            except queue.Empty:return
            name=case['directory'];out=a.output/name
            log=(a.output/(name+'.log')).open('x')
            env=dict(os.environ,EVAL_GPU=str(gpu),EVAL_SOURCE=str(REPO/'scripts/eval/memlite_sft100'))
            loading.acquire();released=False
            try:
                proc=subprocess.Popen(['bash',str(REPO/'scripts/eval/memlite_sft100/launch_sim.sh'),
                    str(REPO/'scripts/rl/memlite_online/tools/collect_local_grasp_recovery.py'),
                    '--proposal',str(a.sources/name),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,env=env,cwd=REPO)
                while proc.poll() is None:
                    if not released and (out/'status.json').exists():
                        state=json.loads((out/'status.json').read_text())
                        if state['status']!='loading':loading.release();released=True
                    time.sleep(1)
                row=dict(case=name,gpu=gpu,pid=proc.pid,returncode=proc.returncode,
                         status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else None)
                with lock:
                    rows.append(row);atomic_json(a.output/'status.json',dict(status='collecting',completed=len(rows),
                        inventory_cases=len(sources),quota=None,results=rows))
            finally:
                if not released:loading.release()
                log.close();jobs.task_done()
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:list(pool.map(worker,a.gpus))
    atomic_json(a.output/'result.json',dict(status='inventory_attempts_finished',results=rows,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_inventory_sha256=sha256(a.sources/'manifest.json'),training_approved=False,quota=None))


if __name__=='__main__':main()
