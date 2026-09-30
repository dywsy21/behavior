"""One detached, finite gate0 -> gate3 -> six-case campaign; never retry a slot.

Each stage retains its own GPU/resource/timeout supervisor. The coordinator
stops on infrastructure failure, but continues after valid policy failures or
abstentions. No case replacement and no success-conditioned selection.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import launch_jev_closedloop as launch
from probe_jev_multitask import source_identity, validate_evidence

REPO = Path(__file__).resolve().parents[2]
ROOT = launch.RUN_PARENT/'jev_multi_20260930_campaign_v1'
ORDER = launch.MULTI_STAGES
TOTAL_SECONDS = 8*3600


def write(value):
    tmp = ROOT/'campaign.json.tmp'
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,ROOT/'campaign.json')


def start():
    if 'TYPESAFE_API_KEY' in os.environ: raise ValueError('Only private key file')
    identity = source_identity(REPO)
    validate_evidence(REPO, launch.RUN_PARENT/'jev_multi_20260930_api_v1')
    base = launch.configure('gate0','multitask')
    launch.preflight(base)
    for stage in ORDER:
        folder = launch.stage_root(stage)
        runtime = launch.RUNTIME_PARENT/folder.name
        if folder.exists() or folder.is_symlink() or runtime.exists() or runtime.is_symlink():
            raise FileExistsError('A registered stage already exists; inspect it, never resubmit: '+stage)
    ROOT.mkdir(parents=True,exist_ok=False)
    reservation = dict(status='reserved',identity=identity,stages=list(ORDER),utc=datetime.now(timezone.utc).isoformat(),
                       total_seconds=TOTAL_SECONDS,training_updates=0,retries=0)
    (ROOT/'reservation.json').write_text(json.dumps(reservation,indent=2)+'\n')
    with (ROOT/'coordinator.log').open('x') as log:
        process = subprocess.Popen([str(base.PYTHON),str(Path(__file__).resolve()),'--supervise'],cwd=REPO,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(status='coordinator_started',pid=process.pid,output=str(ROOT),**identity)),flush=True)


def supervise():
    identity = source_identity(REPO)
    reserved = json.loads((ROOT/'reservation.json').read_text())
    if reserved['identity'] != identity or reserved['stages'] != list(ORDER):
        raise ValueError('Campaign source or sequence changed')
    # Atomic ownership claim makes accidental duplicate supervision fail closed.
    (ROOT/'coordinator.claim').open('x').close()
    began = time.monotonic()
    receipt = dict(status='running',identity=identity,pid=os.getpid(),stages=[],retries=0,
                   started_utc=datetime.now(timezone.utc).isoformat(),total_seconds=TOTAL_SECONDS)
    try:
        for stage in ORDER:
            if time.monotonic()-began >= TOTAL_SECONDS: raise TimeoutError('Campaign wall budget')
            base = launch.configure(stage,'multitask')
            row = dict(stage=stage,status='launching',output=str(launch.ROOT))
            receipt['stages'].append(row); write(receipt)
            command = [str(base.PYTHON),str(REPO/'scripts/semantic_robot/launch_jev_closedloop.py'),
                       '--campaign','multitask','--stage',stage,'--launch']
            with (ROOT/(stage+'.launch.log')).open('x') as log:
                done = subprocess.run(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,
                                      stderr=subprocess.STDOUT,timeout=120)
            if done.returncode: raise RuntimeError(stage+' launch failed; no retry')
            submitted = json.loads((launch.ROOT/'launch.json').read_text())
            if (submitted['source_commit'] != identity['code_commit'] or submitted['campaign'] != 'multitask'
                    or submitted['implementation_digest'] != identity['implementation_digest']):
                raise ValueError('Submitted source mismatch')
            row.update(status='running',supervisor_pid=submitted['supervisor_pid']); write(receipt)
            deadline = time.monotonic()+submitted['wall_seconds']+240
            while True:
                path = launch.ROOT/'supervisor.json'
                value = json.loads(path.read_text()) if path.exists() else {}
                if value.get('status') in ('completed','failed') and 'exit_codes' in value:
                    row.update(status=value['status'],official_success=value.get('official_success'),
                        controls=value.get('controls'),stop_reason=value.get('stop_reason'),
                        supervisor_sha256=launch.common.sha(path))
                    write(receipt)
                    if value['status'] != 'completed': raise RuntimeError(stage+' infrastructure/validation failure; no next reset')
                    break
                proc = Path('/proc')/str(row['supervisor_pid'])
                try: cmd = (proc/'cmdline').read_bytes()
                except FileNotFoundError: raise RuntimeError(stage+' supervisor vanished without final receipt')
                if str(REPO/'scripts/semantic_robot/launch_jev_closedloop.py').encode() not in cmd:
                    raise RuntimeError('Supervisor PID identity changed; do not signal it')
                if time.monotonic() > deadline or time.monotonic()-began > TOTAL_SECONDS:
                    raise TimeoutError('Coordinator wait limit; existing stage has independent timeout, do not retry')
                time.sleep(10)
        receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='stopped',error=repr(error)); raise
    finally:
        receipt.update(seconds=time.monotonic()-began,updated_utc=datetime.now(timezone.utc).isoformat())
        write(receipt)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch',action='store_true');mode.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    (start if args.launch else supervise)()
