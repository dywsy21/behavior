"""No-retry, cumulative-wall supervisor for a pinned H0/H1/L0 pilot ticket."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_train_contract import validate_launch
from recovery_corpus import file_sha
from g05.utils.training.stage1_runtime import atomic_json,disk_has_reserve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticket',type=Path,required=True);p.add_argument('--component',choices=('H0','H1','L0'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--engineering',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();ticket,recipe=validate_launch(a.ticket,a.component,engineering=a.engineering)
    a.ticket=a.ticket.resolve();a.output=a.output.resolve();os.chdir(REPO)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise ValueError('Clean source required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if ticket['source_commit']!=commit: raise ValueError('Source does not match approved ticket')
    if recipe['preferred_node'] not in subprocess.check_output(['hostname','-I'],text=True).split(): raise ValueError('Wrong node')
    if not disk_has_reserve(recipe['root']): raise RuntimeError('Keep 256GiB shared disk reserve')
    # No interference, even when another job uses only one device of this node.
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Node has another GPU job; do not displace/compete')
    control=a.output.with_name(a.output.name+'.supervisor');control.mkdir(parents=True,exist_ok=a.resume)
    lock=(control/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger_path=control/'ledger.json';identity=file_sha(a.ticket)
    if a.resume:
        ledger=json.loads(ledger_path.read_text())
        if ledger['status']=='RUNNING' or ledger['ticket_sha256']!=identity: raise ValueError('Unclosed/changed attempt; reconcile before resume')
        previous=ledger['consumed_seconds'];attempt=ledger['attempt']+1
    else: previous=0.;attempt=1
    limit=ticket['wall_seconds']
    if previous+180>=limit: raise RuntimeError('Cumulative time budget exhausted')
    started=time.monotonic();deadline=started+limit-previous;stop=control/f'stop_{attempt:03d}.json'
    env=dict(os.environ,PYTHONPATH=str(REPO/'src'),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',RECOVERY_DEADLINE=str(deadline),RECOVERY_STOP_FILE=str(stop),
        RECOVERY_PREVIOUS_SECONDS=str(previous),RECOVERY_STARTED=str(started),RECOVERY_ATTEMPT=f'{attempt:03d}',**recipe['nccl_env'])
    if a.component=='H0':
        env['CUDA_VISIBLE_DEVICES']='0'
        command=[sys.executable,'scripts/train_memlite_recovery_h0.py','--ticket',str(a.ticket),'--output',str(a.output)]
    else:
        env['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in range(ticket['world_size']))
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={ticket["world_size"]}',
            '--max_restarts=0','scripts/train_memlite_recovery.py','--ticket',str(a.ticket),'--component',a.component,'--output',str(a.output)]
    if a.resume: command.append('--resume')
    if a.engineering: command.append('--engineering')
    if a.stop_after is not None: command+=['--stop-after',str(a.stop_after)]
    stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    ledger=dict(ticket_sha256=identity,attempt=attempt,limit_seconds=limit,source_commit=commit,
                component=a.component,engineering_only=a.engineering,supervisor_pid=os.getpid())
    with (control/f'attempt_{attempt:03d}.log').open('x') as log:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ledger['child_pid']=child.pid;term_at=None
        try:
            while child.poll() is None:
                now=time.monotonic();ledger.update(status='RUNNING',consumed_seconds=previous+now-started,updated=time.time())
                atomic_json(ledger_path,ledger)
                if stopping[0] or now+180>=deadline or not disk_has_reserve(recipe['root']):
                    if not stop.exists(): atomic_json(stop,dict(reason='signal_time_or_disk_reserve'))
                if now>=deadline:
                    if term_at is None: os.killpg(child.pid,signal.SIGTERM);term_at=now
                    elif now-term_at>30: os.killpg(child.pid,signal.SIGKILL)
                time.sleep(2)
            code=child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:
            ledger.update(status='EXITED',returncode=child.returncode,consumed_seconds=previous+time.monotonic()-started,updated=time.time())
            atomic_json(ledger_path,ledger)
    print(json.dumps(ledger));raise SystemExit(code)


if __name__=='__main__':main()
