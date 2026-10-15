"""Explicit unstarted-suffix recovery of one quarantined champion worker.

The live parent and its healthy workers are never stopped or rewritten. The
recovery is a separate registered run. Its outputs require a later explicit
union with the parent's originals; a recovery is not a new full-model score.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'memlite_sft100')]
import continue_champion_final as c
from common import ROOT, atomic_json

SCHEMA = 'champion_isolated_unstarted_worker_recovery_v1'


def remaining_suffix(parts, completed, failed, present):
    names = [part['id'] for part in parts]
    known = completed + failed
    if (len(known) != len(set(known)) or not set(known) <= set(names)
            or set(known) != set(names[:len(known)])
            or set(present) != set(known)
            or [name for name in names if name in completed] != completed
            or [name for name in names if name in failed] != failed):
        raise ValueError('Recovery must be the exact never-attempted suffix')
    return parts[len(known):]


def audit_terminal_disconnects(path, task_id, num_envs):
    from audit_champion_public_trace import audit_lines
    raw = Path(path).read_bytes(); lines = raw.splitlines(keepends=True)
    positions = [i for i, line in enumerate(lines) if json.loads(line)['kind'] == 'connection_error']
    if not positions:
        return audit_lines(lines, require_closed=True)
    first = positions[0]; report = audit_lines(lines[:first])
    active = {s['connection']: s for s in report['session_summaries'] if not s['closed']}
    tail = [json.loads(line) for line in lines[first:]]
    if len(active) != num_envs or len(tail) != 2 * num_envs:
        raise ValueError('Unknown terminal connection error boundary')
    seen = set()
    for error, closed in zip(tail[::2], tail[1::2]):
        cid = error['connection']; session = active.get(cid)
        if (cid in seen or session is None or session['task'] != task_id
                or error['kind'] != 'connection_error' or error.get('error') != 'ConnectionClosedError(None, None, None)'
                or closed['kind'] != 'closed' or closed['connection'] != cid):
            raise ValueError('Unclassified terminal journal error')
        expected = dict(episode=session['episode'], task=task_id, clock_kind='policy_delivery',
                        delivered_controls=session['deliveries'] * 20, physically_confirmed_controls=None,
                        predictions=session['deliveries'], unused_offered_controls=0, optimizer_updates=0)
        if closed.get('previous') != expected:
            raise ValueError('Terminal disconnect belongs to a different session')
        seen.add(cid)
    if Path(path).read_bytes() != raw:
        raise ValueError('Terminal journal changed while checking')
    return dict(prefix={k: v for k, v in report.items() if k != 'session_summaries'},
                terminal_disconnects=tail, terminal_trace=c.binding(path),
                interpretation='Known failed physical batch closed its four clients; no retries authorized')


def certify(parent, gpu):
    parent = Path(parent).resolve()
    if parent.parent != ROOT / 'runs' or type(gpu) is not int or not 0 <= gpu < 8:
        raise ValueError('Explicit registered parent and GPU required')
    m = c.read(parent / 'manifest.json'); _, original, root_pin = c.root_registration(parent)
    c.frozen_source(m['runtime_source'], original['source_commit'])
    if (m['schema'] != c.SCHEMA or not m.get('isolate_physical_batches')
            or not m.get('continue_on_batch_errors')
            or m['groups'] != c.remaining_groups(original, c.read(m['seal']['path']), True)):
        raise ValueError('Only registered isolated parent schedules can recover')
    state = c.read(parent / 'status.json'); wp = parent / 'workers' / f'gpu_{gpu}'; w = c.read(wp / 'status.json')
    if (state['worker_pids'].get(str(gpu)) != w['pid'] or str(gpu) not in state['failed_gpus']
            or w.get('status') != 'needs_diagnosis' or not w.get('finished')
            or w.get('error') != "RuntimeError('Frozen model loading failed/timed out/stopped')"
            or w.get('final_verification_error') != "FileNotFoundError(2, 'No such file or directory')"):
        raise ValueError('This certificate requires the observed pre-admission reload failure')
    c.require_exited([w['pid'], w.get('model_pid'), w.get('sim_pid')])
    finals = w['model_generations']
    if len(finals) != 1 or len(w['diagnosed_failed_parts']) != 1:
        raise ValueError('Additional model failures require separate classification')
    for pin in finals:
        c.check_binding(pin); value = c.read(pin['path']); c.validate_weights(value, original, gpu)
        c.require_exited([value['pid']])
    rejected = wp / 'model-001'; rejected_log = wp / 'model-001.log'
    log = rejected_log.read_text()
    if (Path(w['model_path']) != rejected or rejected.exists()
            or log.count('Traceback (most recent call last):') != 1
            or not log.rstrip().endswith("ValueError: Registered model GPU must be idle")
            or re.search(r'CUDA error:|CUDA out of memory|device-side assert', log)):
        raise ValueError('Rejected generation may have loaded; cannot waive final verification')
    parts = m['groups'][gpu]['parts']; by_name = {p['id']: p for p in parts}
    present = [p['id'] for p in parts if (parent / 'parts' / p['id']).exists()]
    pending = remaining_suffix(parts, w['completed_parts'], w['diagnosed_failed_parts'], present)
    evidence = [c.binding(wp / 'status.json'), c.binding(parent / f'gpu_{gpu}.log'),
                c.binding(rejected_log), *finals]
    records = []; errors = []
    for name in w['completed_parts'] + w['diagnosed_failed_parts']:
        part = by_name[name]; directory = parent / 'parts' / name; ps = c.read(directory / 'status.json')
        if (any(ps.get(k) != part[k] for k in ('task', 'task_id', 'indices'))
                or ps['source_commit'] != original['source_commit'] or ps['optimizer_updates'] != 0
                or ps['config_sha256'] != original['configs'][gpu]['sha256']):
            raise ValueError('Parent physical identity changed')
        c.shared.require_no_uncommitted_metrics(directory)
        attempts = sorted((directory / 'attempts').glob('*.json'))
        if len(attempts) != 1:
            raise ValueError('Expected one original physical attempt')
        attempt = c.read(attempts[0])
        if (attempt['policy_seed'] != 17 or attempt['simulator_seed'] != 0
                or attempt['max_steps'] != original['nominal_horizons'][part['task']]
                or attempt['indices'] != part['indices']):
            raise ValueError('Original seeds or slots changed')
        evidence += [c.binding(directory / 'status.json'), c.binding(attempts[0])]
        if name in w['completed_parts']:
            if ps['status'] != 'completed':
                raise ValueError('Unfinished output marked complete')
            records += [c.shared.original_record(p, row) for p, row in c.completed_metrics(directory, final=True)]
        else:
            physical_log = wp / (name + '.log'); diagnosis = c.execution_error(physical_log.read_text())
            if diagnosis is None or attempt['status'] != 'running':
                raise ValueError('Only explicitly diagnosed attempted failure can be excluded')
            expected = [dict(task=part['task'], instance_id=i, rollout_id=0, original_attempt=c.binding(attempts[0]),
                             diagnosis=diagnosis, log=c.binding(physical_log), rerun_allowed=False)
                        for i in attempt['instance_ids']]
            ledger = wp / (name + '-execution-errors.json')
            if c.read(ledger) != expected:
                raise ValueError('Original failure ledger changed')
            errors += expected
            evidence += [c.binding(ledger), c.binding(physical_log), c.binding(wp / (name + '-command.json'))]
            evidence += [c.binding(p) for p in sorted((directory / 'videos').glob('*.mp4'))]
    failed = by_name[w['diagnosed_failed_parts'][0]]
    trace = wp / 'model-000/events.jsonl'
    terminal_trace = audit_terminal_disconnects(trace, failed['task_id'], failed['num_envs'])
    evidence += [c.binding(trace), c.binding(wp / 'model-000/status.json')]
    return dict(schema='champion_isolated_worker_failure_certificate_v1', parent=str(parent), gpu=gpu,
                parent_manifest=c.binding(parent / 'manifest.json'), root_manifest=root_pin,
                worker=c.binding(wp / 'status.json'), loaded_models=finals,
                rejected_generation=dict(pid=w['model_pid'], output=str(rejected), log=c.binding(rejected_log),
                                         reason='Original GPU-idle check failed before output mkdir and model loading'),
                previous_records=records, execution_errors=errors, pending=pending,
                pending_cases=sum(len(p['indices']) for p in pending), evidence=evidence,
                terminal_trace=terminal_trace, original_outputs_unmodified=True, rerolls_allowed=False)


def prepare(parent, gpu, job):
    parent, job = Path(parent).resolve(), Path(job).resolve()
    if job.exists() or job.parent != ROOT / 'runs':
        raise ValueError('Fresh isolated recovery directory required')
    certificate = certify(parent, gpu)
    if not certificate['pending']:
        raise ValueError('No never-attempted cases remain on this worker')
    commit = subprocess.check_output(['git', '-C', str(c.REPO), 'rev-parse', 'HEAD'], text=True).strip()
    c.frozen_source(c.REPO, commit)
    pm = c.read(parent / 'manifest.json')
    job.mkdir(); (job / 'parts').mkdir(); (job / 'workers').mkdir()
    atomic_json(job / 'parent-worker-certificate.json', certificate)
    m = dict(schema=SCHEMA, created=time.time(), source_commit=commit, parent=str(parent), gpu=gpu,
             parent_manifest=certificate['parent_manifest'], certificate=c.binding(job / 'parent-worker-certificate.json'),
             runtime_source=pm['runtime_source'], runtime_commit=pm['runtime_commit'],
             groups=[dict(gpu=g, parts=certificate['pending'] if g == gpu else []) for g in range(8)],
             **{key: pm[key] for key in c.IDENTITY},
             isolate_physical_batches=True, continue_on_batch_errors=True, optimizer_updates=0,
             automatic_retries=False, policy_changed=False)
    atomic_json(job / 'manifest.json', m)
    return m


def registration(job, gpu):
    job = Path(job).resolve(); m = c.read(job / 'manifest.json')
    if job.parent != ROOT / 'runs' or m['schema'] != SCHEMA or m['gpu'] != gpu:
        raise ValueError('Exact isolated worker registration required')
    c.frozen_source(c.REPO, m['source_commit']); c.check_binding(m['parent_manifest']); c.check_binding(m['certificate'])
    parent = Path(m['parent']); pm = c.read(parent / 'manifest.json')
    _, original, _ = c.root_registration(parent); cert = c.read(m['certificate']['path'])
    if cert != certify(parent, gpu):
        raise ValueError('Worker certificate differs from actual terminal parent evidence')
    if (m['parent_manifest']['path'] != str(parent / 'manifest.json') or cert['parent_manifest'] != m['parent_manifest']
            or cert['parent'] != str(parent) or cert['gpu'] != gpu or cert['rerolls_allowed'] is not False
            or m['policy_changed'] is not False or m['automatic_retries'] is not False or m['optimizer_updates'] != 0
            or m['runtime_source'] != pm['runtime_source'] or m['runtime_commit'] != original['source_commit']
            or any(m[key] != original[key] for key in c.IDENTITY)
            or m['model_overlay'] != pm['model_overlay'] or not m['isolate_physical_batches'] or not m['continue_on_batch_errors']
            or m['groups'] != [dict(gpu=g, parts=cert['pending'] if g == gpu else []) for g in range(8)]):
        raise ValueError('Recovery changed its certified scope or frozen policy')
    c.frozen_source(m['runtime_source'], m['runtime_commit'])
    for pin in cert['evidence']:
        c.check_binding(pin)
    for row in cert['previous_records']:
        c.check_binding(row['metrics']); c.check_binding(row['video'])
    w = c.read(cert['worker']['path']); c.require_exited([w['pid'], w.get('model_pid'), w.get('sim_pid')])
    for part in cert['pending']:
        if (parent / 'parts' / part['id']).exists():
            raise ValueError('Parent unexpectedly attempted a transferred case')
    return job, m


def check_model_request(job, m, config, output):
    job, output, config = Path(job).resolve(), Path(output).resolve(), Path(config).resolve()
    gpu = m['gpu']; wp = job / 'workers' / f'gpu_{gpu}'; w = c.read(wp / 'status.json')
    if (config != Path(m['configs'][gpu]['path']) or output.parent != wp
            or not re.fullmatch(r'model-\d{3}', output.name) or output.exists()
            or Path(w['model_path']) != output or w['model_pid'] != os.getpid() or w['pid'] != os.getppid()):
        raise ValueError('Model admission is not owned by the registered recovery worker')


def run(job):
    job = Path(job).resolve(); initial = c.read(job / 'manifest.json'); gpu = initial['gpu']
    job, m = registration(job, gpu)
    if (job / 'status.json').exists() or (job / 'STOP').exists():
        raise ValueError('Never automatically restart an attempted recovery')
    state = dict(status='running', pid=os.getpid(), started=time.time(), gpu=gpu,
                 parent=m['parent'], manifest=c.binding(job / 'manifest.json'), automatic_retries=False)
    atomic_json(job / 'status.json', state)
    try:
        c.worker(job, gpu, recovery=True)
        w = c.read(job / 'workers' / f'gpu_{gpu}' / 'status.json')
        if w['status'] not in ('completed', 'completed_with_execution_errors') or w.get('final_verification_error'):
            raise ValueError('Recovered worker did not finish with verified frozen models')
        _, original, _ = c.root_registration(m['parent'])
        for pin in w['model_generations']:
            c.check_binding(pin); value = c.read(pin['path']); c.validate_weights(value, original, gpu)
            if (value.get('launcher_source_commit') != m['source_commit']
                    or value.get('recovery_manifest_sha256') != c.sha256(job / 'manifest.json')):
                raise ValueError('Recovery model entry provenance differs')
        state['status'] = w['status']
    except BaseException as error:
        state.update(status='needs_isolated_worker_recovery', error=repr(error)); raise
    finally:
        state['finished'] = time.time(); atomic_json(job / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--parent', type=Path); parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        m = prepare(args.parent, args.gpu, args.job)
        print(json.dumps(dict(job=str(args.job), gpu=m['gpu'], cases=sum(len(p['indices']) for p in m['groups'][m['gpu']]['parts']))))
    elif args.command == 'verify':
        _, m = registration(args.job, args.gpu); print(json.dumps(dict(status='registered', source_commit=m['source_commit'])))
    else:
        run(args.job)
