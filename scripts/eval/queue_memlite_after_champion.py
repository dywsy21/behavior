"""Run the explicitly registered MEM remainder after champion public1000.

User-authorized order change only. No training, policy changes, retries, error
reinterpretation, or takeover of another GPU owner. This queue has no model.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from continue_memlite_final import (REPO, atomic_json, binding, check_binding, frozen_source,
                                   read, registration, require_exited)
from package_final_public_eval import command_inventory
from queue_champion_after_memlite import dependency_ready
from common import aggregate, sha256
from completed_metrics import completed_metrics


def champion_ready(state, summary):
    if state.get('status') in ('needs_diagnosis', 'failed_no_retry', 'failed'):
        raise RuntimeError('Champion needs diagnosis; do not launch MEM or retry champion')
    if state.get('status') != 'completed':
        return False
    if (state.get('failed_gpus') != {} or len(state.get('completed_gpus', [])) != 8
            or set(state.get('completed_gpus', [])) != set(range(8))
            or not state.get('finished') or summary.get('status') != 'complete'
            or type(summary.get('completed')) is not int or summary['completed'] != 1000
            or summary.get('missing') != 0 or summary.get('optimizer_updates') != 0
            or summary.get('original_outputs_unmodified') is not True
            or summary.get('official_q_score') != state.get('q_score')
            or summary.get('official_sr') != state.get('success_rate')):
        raise ValueError('Champion terminal state must contain complete, unmodified public1000')
    return True


def validate_completed_champion(job, manifest):
    """Reconcile the sealed original metrics and eight final model receipts."""
    state, summary = read(job / 'status.json'), read(job / 'summary.json')
    if not champion_ready(state, summary):
        raise ValueError('Champion is not complete')
    require_exited([state['pid'], *state['worker_pids'].values()])
    evidence = [binding(job / n) for n in ('status.json', 'summary.json', 'inventory.json',
                                          'submission/package-status.json')]
    for gpu in range(8):
        path = job / 'workers' / f'gpu_{gpu}' / 'status.json'; row = read(path)
        require_exited([row['pid'], row.get('model_pid'), row.get('sim_pid')])
        verification = row.get('verification', {})
        if (row.get('status') != 'completed' or not row.get('finished')
                or row.get('source_commit') != manifest['source_commit']
                or row.get('export_sha256') != manifest['export_sha256']
                or verification.get('weights_unchanged') is not True
                or verification.get('optimizer_updates') != 0
                or verification.get('source_commit') != manifest['source_commit']
                or verification.get('config_sha256') != manifest['configs'][gpu]['sha256']):
            raise ValueError('Missing exact champion worker/weight terminal receipt')
        evidence.append(binding(path))
    commands = command_inventory(job, manifest)
    records, originals = [], {}
    for part in sorted((job / 'parts').iterdir()):
        for path, row in completed_metrics(part, final=True):
            originals[str(path.resolve())] = row; records.append(row)
    actual = aggregate(manifest['tasks'], records)
    for k in ('status', 'completed', 'missing', 'official_q_score', 'official_sr'):
        if actual[k] != summary[k]:
            raise ValueError('Actual original metrics disagree with champion summary')
    inventory = read(job / 'inventory.json')
    if len(inventory) != 1000 or {r['metrics'] for r in inventory} != set(originals):
        raise ValueError('Champion inventory omitted or duplicated original metrics')
    for row in inventory:
        metric, video = Path(row['metrics']), Path(row['video'])
        if (not video.resolve().is_relative_to(job / 'parts') or not video.is_file()
                or video.stat().st_size != row['video_bytes'] or sha256(metric) != row['metrics_sha256']
                or any(row[k] != originals[str(metric)][k] for k in ('task', 'instance_id', 'rollout_id'))):
            raise ValueError('Champion original inventory/file identity changed')
    sealed = read(job / 'submission/package-status.json'); archive = Path(sealed['zip_path']).resolve()
    if (archive.parent != job / 'submission' or sealed.get('metrics') != 1000
            or sealed.get('videos') != 1000 or sha256(archive) != sealed['zip_sha256']
            or sealed.get('status') != 'evaluation_outputs_only_not_final_model_delivery'):
        raise ValueError('Champion sealed internal archive is incomplete')
    with zipfile.ZipFile(archive) as stream:
        for name in ('manifest.json', 'summary.json', 'inventory.json'):
            if stream.read(name) != (job / name).read_bytes():
                raise ValueError('Original sealed champion provenance differs')
    return dict(evidence=evidence, commands=len(commands), originals=1000, gpu_takeover=False,
                archive=binding(archive), videos_rehashed_by_original_supervisor=True,
                videos_rehashed_by_this_queue=False)


def validate_order(mem, champion, authorization):
    mem, mm = registration(mem)
    champion = Path(champion).resolve(); authorization = Path(authorization).resolve()
    auth = read(authorization); cm = read(champion / 'manifest.json'); seal = read(mm['seal']['path'])
    if (auth.get('schema') != 'user_authorized_eval_priority_stop_v1'
            or mm['prior'] != auth['mem_job'] or auth['champion_job'] != str(champion)
            or auth['mem_manifest_sha256'] != mm['prior_manifest']['sha256']
            or auth['champion_manifest_sha256'] != sha256(champion / 'manifest.json')
            or cm['schema'] != 'champion_final_public100_run_v1'
            or seal.get('priority_authorization') != binding(authorization)
            or mm.get('automatic_retries') is not False or mm.get('policy_changed') is not False
            or mm.get('optimizer_updates') != 0 or cm.get('automatic_retries') is not False
            or cm.get('optimizer_updates') != 0):
        raise ValueError('Explicit user priority authorization and exact registered MEM remainder required')
    return mem, mm, champion, cm, authorization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mem-job', 'champion-job', 'authorization'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    mem, mm, champion, cm, auth = validate_order(args.mem_job, args.champion_job, args.authorization)
    if (mem / 'status.json').exists() or (mem / 'STOP').exists():
        raise ValueError('Never queue a live or previously attempted continuation')
    owner = (mem / 'queue-owner.lock').open('a'); fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue = mem / 'queue-after-champion'; queue.mkdir(exist_ok=False)
    bindings = [binding(mem / 'manifest.json'), binding(champion / 'manifest.json'), binding(auth), mm['seal']]
    state = dict(status='waiting_for_champion', pid=os.getpid(), started=time.time(), phases=[],
                 mem_job=str(mem), champion_job=str(champion), bindings=bindings,
                 queue_source_commit=mm['source_commit'], automatic_retries=False)

    def publish():
        state['updated'] = time.time(); atomic_json(queue / 'status.json', state)

    def unchanged():
        for row in bindings:
            check_binding(row)
        if (queue / 'STOP').exists() or (mem / 'STOP').exists():
            raise InterruptedError('Requested queue stop')

    try:
        publish()
        while True:
            unchanged()
            status = read(champion / 'status.json'); summary = read(champion / 'summary.json')
            ready = champion_ready(status, summary)
            pid = status.get('pid')
            live = bool(pid and Path(f'/proc/{pid}').exists())
            if ready and not live:
                break
            if not ready and not live:
                raise RuntimeError('Champion supervisor disappeared without complete terminal evidence')
            time.sleep(30)
        state['champion_terminal'] = validate_completed_champion(champion, cm); publish()
        frozen_source(REPO, mm['source_commit']); registration(mem); unchanged()
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPUs still owned; do not displace processes')
        argv = [sys.executable, str(REPO / 'scripts/eval/continue_memlite_final.py'), 'supervise', '--job', str(mem)]
        phase = dict(name='remaining_mem', status='running', started=time.time(), argv=argv)
        state['status'] = 'running_mem_remainder'; state['phases'].append(phase); publish()
        with (queue / 'mem-supervisor.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=REPO, env=os.environ | dict(PYTHONDONTWRITEBYTECODE='1'),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            phase['pid'] = process.pid; publish()
            result = process.wait()
        phase.update(returncode=result, finished=time.time(), status='completed' if result == 0 else 'failed')
        publish()
        if result or not dependency_ready(read(mem / 'status.json'), read(mem / 'summary.json'), continuation=True):
            raise RuntimeError('MEM continuation needs diagnosis; no automatic retry')
        state['status'] = ('both_internal_evaluations_finished_with_disclosed_mem_errors'
                           if read(mem / 'summary.json').get('execution_errors') else
                           'both_internal_evaluations_finished_not_submitted')
    except BaseException as error:
        state.update(status='needs_diagnosis', error=repr(error)); raise
    finally:
        state['finished'] = time.time(); publish(); owner.close()


if __name__ == '__main__':
    main()
