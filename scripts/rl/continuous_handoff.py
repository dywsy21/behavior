"""One-shot CPU dependency launcher; no polling/changes to the active source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import REPO,OUT,RUNTIME,PREVIOUS_RUNTIME,MODEL_PY,commit,save
from g05.rl.continuous_handoff import dependency_state,owned_pids,live_pids

PRIOR=Path('/mnt/nvme_tmp/robodojo_g05_rl_20260928/e3_render_resume_v1')
HANDOFF=PRIOR.parent/'e4_handoff_v1'


def validate_paths():
    if (OUT!=PRIOR.parent/'e4_continuous_v1'
            or RUNTIME!=Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e4_continuous_v1')
            or PREVIOUS_RUNTIME!=RUNTIME.parent/'rl_g05_50k_e3_render_resume_v1/training'):
        raise ValueError('Unregistered handoff paths')


def watch():
    validate_paths(); source=commit(); previous=None
    def status(phase,**details):
        nonlocal previous
        row=dict(phase=phase,pid=os.getpid(),source_commit=source,**details)
        if row!=previous:
            previous=row
            row=dict(row,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            save(HANDOFF/'status.json',row,replace=True)
            print(json.dumps(row),flush=True)
    try:
        claim=json.loads((HANDOFF/'claim.json').read_text())
        if claim['source_commit']!=source: raise ValueError('Handoff source changed')
        while True:
            ready=dependency_state(PRIOR)
            if not ready['ready']:
                status('waiting_for_e3',reason=ready['reason']); time.sleep(30); continue
            alive=live_pids(owned_pids(PRIOR))
            if alive:
                status('waiting_for_e3_process_exit',pids=alive); time.sleep(10); continue
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                             '--format=csv,noheader,nounits'],text=True)
            usage={int(a):int(b) for a,b in (line.split(',') for line in output.strip().splitlines())}
            busy={str(g):usage[g] for g in (0,2,3) if usage[g]>512}
            if busy:
                status('waiting_for_reserved_gpus',usage_mib=busy); time.sleep(30); continue
            break
        status('preparing',dependency=ready)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
        with (HANDOFF/'prepare.stdout.log').open('x') as log:
            subprocess.run([MODEL_PY,str(REPO/'scripts/rl/prepare_continuous.py')],cwd=REPO,
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if live_pids(owned_pids(PRIOR)): raise ValueError('Old process reappeared; refuse GPU launch')
        status('launching')
        launched=json.loads(subprocess.check_output([MODEL_PY,str(REPO/'scripts/rl/launch.py'),
            '--launch','--entry','method_dense'],cwd=REPO,env=env,text=True))
        # The launch receipt alone is not evidence that the new learner started.
        for _ in range(60):
            path=OUT/'supervisor.json'
            if path.exists():
                supervisor=json.loads(path.read_text())
                if supervisor['status']=='failed': raise RuntimeError('E4 supervisor rejected launch')
                if supervisor['status'] in ('running','completed'):
                    status('launched',launch=launched,supervisor=supervisor,
                           recovery_and_training_not_yet_validated=True); return
            time.sleep(2)
        raise RuntimeError('No E4 supervisor startup receipt; do not retry blindly')
    except BaseException as error:
        status('failed',error=repr(error),traceback=traceback.format_exc()); raise


def arm():
    validate_paths(); source=commit()
    if OUT.exists() or RUNTIME.exists(): raise ValueError('Continuation already exists')
    # Exclusive directory and claim make duplicate requests harmless failures.
    HANDOFF.mkdir(parents=True,exist_ok=False)
    save(HANDOFF/'claim.json',dict(source_commit=source,prior=str(PRIOR),output=str(OUT),
         utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    with (HANDOFF/'watch.stdout.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(REPO/'scripts/rl/continuous_handoff.py'),'--watch'],
            cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    save(HANDOFF/'launcher.json',dict(pid=child.pid,source_commit=source))
    print(json.dumps(dict(pid=child.pid,handoff=str(HANDOFF),state='armed_not_training')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--arm',action='store_true'); group.add_argument('--watch',action='store_true')
    args=parser.parse_args(); (arm if args.arm else watch)()
