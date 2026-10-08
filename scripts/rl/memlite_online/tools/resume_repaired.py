"""Resume the eight saved experts under the ORIGINAL deadline, canary first.

This does not stop the old run. The operator must first request its existing
save/drain mechanism and verify final save receipts. No unverified fallback to
old hourly weights, no baseline, no extra training window, no automatic retry.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from bootstrap import bootstrap

SOURCE=bootstrap()
ROOT=Path('/run/ti/rl_memlite_stage1_20261006')
sys.path.insert(0,str(SOURCE/'code'))
from training_config import load_training_config


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def owned_live(job):
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().split(b'\0')
            if any(a==str(job).encode() or a.startswith((str(job)+'/training/').encode()) for a in args):
                if int(p.parent.name)!=os.getpid():found.append(int(p.parent.name))
        except (OSError,ValueError):pass
    return found


def prepare(parent,job,commit):
    if parent.parent!=ROOT/'runs' or job.parent!=ROOT/'runs' or parent==job:
        raise ValueError('Expected distinct explicit experiment paths')
    if not job.name.startswith('large_scale73h_repaired_'):raise ValueError('Unexpected repair run name')
    if owned_live(parent):raise RuntimeError('Old run still alive; drain it before preparing continuation')
    clock=json.loads((parent/'window.json').read_text())
    if clock.get('hours')!=73 or clock['deadline']-clock['started']!=73*3600:
        raise ValueError('Invalid original clock')
    if time.time()>clock['deadline']-1800:raise RuntimeError('Insufficient original budget')
    manifest=deepcopy(json.loads((parent/'manifest.json').read_text()))
    manifest.update(status='prepared_repair',source_snapshot=str(SOURCE),repair_source_commit=commit,
        parent_run=str(parent),resume_receipts={},resume_cursors={},interrupted_cycles={},
        evaluation_enabled=False,repair_config=load_training_config())
    for gpu in range(8):
        prior=parent/'training'/f'gpu_{gpu}_attempt_000'
        status=json.loads((prior/'status.json').read_text())
        receipt=json.loads((prior/'save_ack.json').read_text())
        if not receipt.get('finite') or not receipt.get('load_verified') or receipt['updates']<=0:
            raise ValueError('No verified final save for GPU'+str(gpu))
        checkpoint=Path(receipt['latest']).resolve()
        if checkpoint.parent!=prior/'checkpoints' or checkpoint.stat().st_size!=receipt['bytes']:
            raise ValueError('Checkpoint identity/size mismatch')
        # Final publisher already hashes and CPU-reloads the immutable inode.
        # Check its sidecar agrees; production load verifies all model tensors.
        sidecar=json.loads((checkpoint.parent/'direct_latest.receipt.json').read_text())
        if any(receipt[k]!=sidecar[k] for k in ('sha256','updates','bytes','latest')):
            raise ValueError('Final and latest receipts disagree')
        current=status.get('current',{})
        cursor=int(current.get('cycle',0))+int('finished' in current)
        manifest['resume_receipts'][str(gpu)]=receipt
        manifest['resume_cursors'][str(gpu)]=cursor
        manifest['interrupted_cycles'][str(gpu)]=current
    job.mkdir(exist_ok=False);(job/'training').mkdir()
    write(job/'manifest.json',manifest);write(job/'window.json',clock)
    return manifest,clock


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--start',action='store_true')
    parser.add_argument('--canary-only',action='store_true',help='Validate GPU0, save, and pause for SFT evaluation; do not start the other experts')
    args=parser.parse_args();job=Path(args.job).resolve();parent=Path(args.parent).resolve()
    if len(args.source_commit)!=40 or any(c not in '0123456789abcdef' for c in args.source_commit):
        raise ValueError('Full committed source SHA required')
    repo=SOURCE.parents[2]
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--','scripts/rl/memlite_online'],text=True)
    if actual!=args.source_commit or dirty:raise RuntimeError('Actual repair checkout differs from clean committed source')
    manifest,clock=prepare(parent,job,args.source_commit)
    if not args.start:return
    lock=open(job/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',phase='canary',pid=os.getpid(),job=str(job),source_snapshot=str(SOURCE),window=clock)
    children={};runs={}
    def stop_requested():return (job/'STOP_TRAINING').exists()
    def launch(gpu):
        run=job/'training'/f'gpu_{gpu}_attempt_000'
        command=['python3',str(SOURCE/'tools/run_expert_phase.py'),'--job',str(job),'--run',str(run),
            '--gpu',str(gpu),'--phase','training','--deadline',str(clock['deadline']),
            '--resume',manifest['resume_receipts'][str(gpu)]['latest']]
        with (job/'training'/f'gpu_{gpu}_attempt_000.log').open('w') as output:
            children[gpu]=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,
                stderr=subprocess.STDOUT,start_new_session=True)
        runs[gpu]=run
    def publish():
        state['updated']=time.time();state['active_gpus']=list(children)
        write(job/'status.json',state)
        write(job/'training/summary.json',dict(phase='training',status=state['status'],deadline=clock['deadline'],
                                             active_runs={str(g):str(r) for g,r in runs.items()}))
    try:
        launch(0);publish();canary_deadline=min(time.time()+1500,clock['deadline']-600)
        while True:
            if stop_requested() or time.time()>canary_deadline:raise TimeoutError('Canary stop/deadline')
            if children[0].poll() is not None:raise RuntimeError('Canary worker exited')
            f=runs[0]/'policy_status.json';rs=job/'recovery/gpu_0/recorder_status.json'
            if f.exists() and rs.exists():
                ps=json.loads(f.read_text());rec=json.loads(rs.read_text())
                if ps['status'] in ('failed','save_failed'):raise RuntimeError('Canary failed: '+str(ps))
                if ps['optimizer_updates']>=manifest['resume_receipts']['0']['updates']+2 and rec['counts']['saved']>=1:
                    # Validate real image/action alignment before other GPUs.
                    from validate_recovery import validate_archive
                    first=next((job/'recovery/gpu_0').glob('*.zip'))
                    validation=validate_archive(first)
                    write(job/'canary_acceptance.json',dict(policy=ps,recorder=rec,archive=validation,
                        threshold='two_real_updates_and_one_verified_candidate',time=time.time()))
                    break
            time.sleep(3)
        if args.canary_only:
            (job/'STOP_TRAINING').touch()
            state.update(phase='saving_validated_canary');publish()
            children[0].wait(timeout=300)
            receipt=json.loads((runs[0]/'save_ack.json').read_text())
            if not receipt.get('finite') or not receipt.get('load_verified'):
                raise RuntimeError('Validated canary did not publish a verified final save')
            write(job/'canary_resume_receipt.json',receipt)
            children.pop(0)
            state.update(status='canary_validated_paused_for_sft_evaluation',phase='paused')
            return
        for gpu in range(1,8):launch(gpu)
        state['phase']='training';publish()
        pointer=ROOT/'reports/current_pilot.json'
        current=json.loads(pointer.read_text()) if pointer.exists() else {}
        if current.get('run') in (str(parent),str(job)):
            write(pointer,dict(kind='eight_expert_training_only73h',run=str(job),status='running',
                supervisor_pid=os.getpid(),started=clock['started'],deadline=clock['deadline'],source=str(SOURCE)))
        while children:
            if stop_requested() or time.time()>clock['deadline']-15:raise TimeoutError('Original deadline or user stop')
            for gpu,child in list(children.items()):
                if child.poll() is None:continue
                result=json.loads((runs[gpu]/'status.json').read_text())
                if child.returncode or result.get('status')!='completed':
                    raise RuntimeError('Expert failed: '+str(runs[gpu]))
                del children[gpu]
            publish();time.sleep(3)
        state['status']='completed'
    except Exception as error:
        state.update(status='needs_diagnosis',error=repr(error))
        raise
    finally:
        # Only workers this supervisor created. Preserve original deadlines;
        # let their existing save handlers finish before process termination.
        for gpu in children:
            if runs[gpu].exists():(runs[gpu]/'STOP_SAVE').touch()
        deadline=min(time.time()+300,clock['deadline'])
        while any(p.poll() is None for p in children.values()) and time.time()<deadline:
            time.sleep(2)
        for child in children.values():
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        state['finished']=time.time();publish();lock.close()


if __name__=='__main__':main()
