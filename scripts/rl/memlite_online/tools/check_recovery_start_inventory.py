"""Cold-verify each already reviewed branch once; retain failures, no retries.

This is start-state engineering, not policy evaluation or training. An
incomplete prior attempt is retained as rejected, never silently rerun.
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

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/eval/memlite_sft100'))
from common import atomic_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--collections',type=Path,nargs='+',required=True)
    p.add_argument('--retain',type=Path,nargs='*',default=[])
    p.add_argument('--gpus',type=int,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if len(set(a.gpus))!=len(a.gpus):raise ValueError('Duplicate GPUs')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    for gpu in a.gpus:
        if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
            '--format=csv,noheader'],text=True).strip():raise RuntimeError('Occupied GPU')
    decisions=json.loads(a.decisions.read_text())['branches']
    retained={}
    for directory in a.retain:
        path=directory/'result.json'
        if not path.exists():path=directory/'status.json'
        value=json.loads(path.read_text())
        # A stale loading status from native teardown is still a rejection.
        pid=value.get('pid')
        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Retained attempt is still running')
        key=value['source_branch_sha256']
        if key in retained:raise ValueError('Repeated retained branch')
        retained[key]=dict(directory=str(directory),receipt=str(path),sha256=sha256(path),
            status=value['status'] if path.name=='result.json' else 'rejected_prior_incomplete_attempt',
            source_group=value['source_group'],split=value['split'])
    if not set(retained)<={x['manifest_sha256'] for x in decisions}:raise ValueError('Unreviewed retained source')
    jobs=queue.Queue();rows=[];lock=threading.Lock()
    a.output.mkdir(parents=True)
    for decision in decisions:
        found=[d/decision['case'] for d in a.collections if (d/decision['case']/'result.json').exists()]
        if len(found)!=1:raise ValueError('Ambiguous or missing original case: '+decision['case'])
        case=found[0]
        if sha256(case/decision['branch']/'manifest.json')!=decision['manifest_sha256']:
            raise ValueError('Changed reviewed branch')
        if decision['manifest_sha256'] in retained:
            rows.append(dict(case=decision['case'],retained=True,**retained[decision['manifest_sha256']]))
        else:jobs.put((case,decision))
    def worker(gpu):
        while True:
            try:case,decision=jobs.get_nowait()
            except queue.Empty:return
            out=a.output/decision['case']
            env=dict(os.environ,EVAL_GPU=str(gpu),EVAL_SOURCE=str(REPO/'scripts/eval/memlite_sft100'))
            with (a.output/(decision['case']+'.log')).open('x') as log:
                done=subprocess.run(['bash',str(REPO/'scripts/eval/memlite_sft100/launch_sim.sh'),
                    str(REPO/'scripts/rl/memlite_online/tools/check_recovery_failure_start.py'),
                    '--case-directory',str(case),'--branch',decision['branch'],'--output',str(out)],
                    env=env,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            path=out/'result.json'
            if not path.exists():path=out/'status.json'
            value=json.loads(path.read_text()) if path.exists() else {}
            with lock:
                rows.append(dict(case=decision['case'],retained=False,directory=str(out),gpu=gpu,
                    returncode=done.returncode,receipt=str(path) if path.exists() else None,
                    sha256=sha256(path) if path.exists() else None,
                    status=value.get('status','missing_receipt'),source_group=value.get('source_group'),split=value.get('split')))
                atomic_json(a.output/'status.json',dict(status='checking',results=rows,quota=None))
            jobs.task_done()
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:list(pool.map(worker,a.gpus))
    atomic_json(a.output/'result.json',dict(schema='recovery_cold_start_inventory_v1',status='inventory_checked',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        decisions_sha256=sha256(a.decisions),results=rows,quota=None,formal_training=False))


if __name__=='__main__':main()
