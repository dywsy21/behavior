"""One-time user-requested E3 restart, only after a durable batch checkpoint."""
import json
import os
from pathlib import Path
import signal
import time

from common import save
from g05.rl.dense_resume import safe_restart_boundary

PRIOR=Path('/mnt/nvme_tmp/robodojo_g05_rl_20260928/e3_dense_v1')
SOURCE=Path('/mnt/sdc1/robodojo/behavior_dev/git_worktrees/g05_50k_rl_dense_ef99d92')
SOURCE_SHA='ef99d92d0f718c12a1d4f54f54906c627ea0ba59'


def read(path):
    return json.loads(path.read_text())


def owned_identity(pid):
    proc=Path(f'/proc/{pid}')
    cmd=(proc/'cmdline').read_bytes().split(b'\0')
    if (proc/'cwd').resolve()!=SOURCE or str(SOURCE/'scripts/rl/method_dense.py').encode() not in cmd:
        raise ValueError('Refuse to signal any process except the registered E3 learner')
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    if stat[0]=='Z': raise ValueError('Registered learner already ended')
    return stat[19]  # /proc field22: protect against PID reuse.


def main():
    if (PRIOR/'operator_restart_request.json').exists():
        raise ValueError('Restart already requested; do not send a duplicate signal')
    manifest=read(PRIOR/'manifest.json'); supervisor=read(PRIOR/'supervisor.json')
    if manifest['source_commit']!=SOURCE_SHA or supervisor['status']!='running':
        raise ValueError('Expected original running E3')
    pid=supervisor['learner']; identity=owned_identity(pid); start=time.monotonic()
    while time.monotonic()-start<3600:  # Only the operator wait, NOT the training budget.
        if owned_identity(pid)!=identity: raise ValueError('Learner PID identity changed')
        status=read(PRIOR/'status.json'); paths=sorted(PRIOR.glob('train_batch_*.json'))
        if paths:
            batch=read(paths[-1]); receipt=read(Path(batch['checkpoint']).with_suffix('.json'))
            if safe_restart_boundary(status,batch,receipt):
                request=dict(reason='user requested removal of 8h training and enclosing 12h wall limit',
                    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),learner_pid=pid,
                    process_start_ticks=identity,source_commit=SOURCE_SHA,
                    checkpoint=batch['checkpoint'],checkpoint_sha256=receipt['sha256'],
                    actor_updates=receipt['actor_updates'],critic_updates=receipt['critic_updates'],
                    completed_batches=batch['batch']+1,status_before_signal=status,
                    preserves_original_control_update_and_batch_budgets=True)
                if not safe_restart_boundary(read(PRIOR/'status.json'),batch,receipt):
                    continue
                save(PRIOR/'operator_restart_request.json',request)
                os.kill(pid,signal.SIGTERM)  # Handler reaps pending controls and closes owned sims.
                print(json.dumps(request),flush=True)
                return
        time.sleep(1)
    raise TimeoutError('No safe saved-batch boundary observed; original training was not stopped')


if __name__=='__main__': main()
