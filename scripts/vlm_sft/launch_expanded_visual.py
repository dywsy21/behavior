"""H80 CPU-only one-shot collector with an independent native-process timeout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

from prepare_expanded_visual import BUDGET, REPO, atomic_json, read_json, sha, source_identity

ROOT=Path('/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h80_expanded_raw_v1')
PYTHON=Path('/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python')


def code_identity():
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean immutable collector source required')
    return subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()


def command():
    # GNU timeout survives supervisor failure. It owns the isolated process
    # group, including native decoders that ignore Python's threading Event.
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(BUDGET['seconds'])+'s',
            str(PYTHON),str(REPO/'scripts/vlm_sft/prepare_expanded_visual.py'),'--output',str(ROOT/'raw')]


def check_seal(root,code):
    raw=root/'raw'
    if (raw/'failure.json').exists(): raise ValueError('Worker recorded a failed collection')
    seal=read_json(raw/'complete.json');manifest=read_json(raw/'manifest.json')
    if (seal!={'status':'RAW_COMPLETE_LABELS_PENDING','manifest_sha256':sha(raw/'manifest.json'),
               'training_eligible':False} or manifest['code_commit']!=code or
            manifest['training_eligible'] is not False or
            manifest['status']!='RAW_COMPLETE_LABELS_PENDING'):
        raise ValueError('Worker did not seal a verified RAW-only collection')
    return seal


def stop_group(child):
    # child was created by this supervisor with SID==PGID==PID. Nothing outside
    # that session is signalled. GNU timeout is a second independent backstop.
    if child.poll() is None and os.getpgid(child.pid)!=child.pid:
        raise RuntimeError('Owned process group changed')
    for sig,wait in ((signal.SIGTERM,3),(signal.SIGKILL,3)):
        try: os.killpg(child.pid,sig)
        except ProcessLookupError: pass
        try: child.wait(timeout=wait)
        except subprocess.TimeoutExpired: continue
        try: os.killpg(child.pid,0)
        except ProcessLookupError: return
    raise RuntimeError('Owned collector process group did not exit')


def supervise(root,cmd,code):
    start=time.monotonic();child=None;error=None
    receipt={'status':'starting','code_commit':code,'budget':BUDGET,'supervisor_pid':os.getpid(),
             'training_eligible':False,'automatic_retry':False,'utc':datetime.now(timezone.utc).isoformat()}
    def interrupted(sig,_frame): raise InterruptedError('Collector supervisor signal '+str(sig))
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
        try:
            with (root/'worker.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_UNBLOCK,set(previous)))
            receipt.update(status='running',worker_group=child.pid,command=cmd)
            atomic_json(root/'supervisor.json',receipt)
        finally: signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
        child.wait(timeout=BUDGET['seconds']+12)
        if child.returncode!=0: raise RuntimeError('Collector failed or hit external timeout: '+str(child.returncode))
        if time.monotonic()-start>BUDGET['seconds']: raise TimeoutError('Collector exceeded total job wall budget')
        receipt['seal']=check_seal(root,code)
        receipt['status']='completed'
    except BaseException as caught:
        error=caught;receipt.update(status='failed',error=repr(caught))
    finally:
        # Ignore subsequent cancellation only while releasing our own group.
        for sig in previous: signal.signal(sig,signal.SIG_IGN)
        cleanup=time.monotonic()
        try:
            if child is not None: stop_group(child)
        except BaseException as caught:
            error=error or caught;receipt.update(status='failed',cleanup_error=repr(caught))
        receipt.update(seconds=time.monotonic()-start,cleanup_seconds=time.monotonic()-cleanup,
                       exit_code=None if child is None else child.returncode)
        if receipt['seconds']>BUDGET['outer_seconds'] or receipt['cleanup_seconds']>BUDGET['cleanup_seconds']:
            error=error or TimeoutError('Collector cleanup allowance exceeded')
            receipt.update(status='failed',outer_budget_exceeded=True)
        try: atomic_json(root/'supervisor.json',receipt)
        except BaseException as caught:
            print(json.dumps({'primary_error':repr(error),'receipt_error':repr(caught)}),file=sys.stderr,flush=True)
            error=error or caught
        for sig,handler in previous.items(): signal.signal(sig,handler)
    if error is not None: raise error
    return receipt


def launch():
    if Path(sys.executable)!=PYTHON: raise ValueError('Use registered data interpreter')
    code=code_identity();sources,_,identity=source_identity()
    if ROOT.exists(): raise FileExistsError('One-shot run already reserved; no implicit retry')
    ROOT.mkdir(parents=True)
    token=secrets.token_hex(24)
    env=dict(os.environ,H80_LAUNCH_TOKEN=token,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',
             PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.pop('PYTHONPATH',None)
    receipt={'status':'reserved','code_commit':code,'source_identity':identity,'sources':sources,
             'budget':BUDGET,'command':command(),'utc':datetime.now(timezone.utc).isoformat(),
             'token_sha256':hashlib.sha256(token.encode()).hexdigest(),'training_eligible':False}
    atomic_json(ROOT/'launch.json',receipt)
    with (ROOT/'supervisor.log').open('x') as log:
        child=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--supervise'],cwd=REPO,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(status='supervisor_started',supervisor_pid=child.pid)
    try: atomic_json(ROOT/'launch.json',receipt)
    except Exception as error:
        # The durable reservation already carries every required binding.
        # Popen succeeded: report that fact, never invite a duplicate launch
        # merely because adding the optional supervisor PID to disk failed.
        receipt['launch_receipt_update_error']=repr(error)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    if args.supervise:
        launch_receipt=read_json(ROOT/'launch.json');code=code_identity()
        if (launch_receipt['code_commit']!=code or launch_receipt['budget']!=BUDGET or
                launch_receipt['command']!=command() or
                hashlib.sha256(os.environ.get('H80_LAUNCH_TOKEN','').encode()).hexdigest()!=launch_receipt['token_sha256']):
            raise ValueError('Collector supervisor identity mismatch')
        print(json.dumps(supervise(ROOT,command(),code)),flush=True)
    else: launch()
