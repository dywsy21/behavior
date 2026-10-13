"""Explicit, outcome-independent continuation with immutable policy runtime.

Completed originals are referenced by SHA, never copied over or rerolled. A
diagnosed nonfinite-pose simulator failure consumes its whole physical batch
as execution errors (NOT fabricated official JSON). Only unrelated global-stop
interruptions may restart. New physics failures are isolated to their batch.
All inference and physics still run from the original frozen source checkout.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE / 'memlite_sft100'), str(HERE / 'memlite_final'), str(HERE)]
from common import ROOT, SIM_ROOT, MODEL_PYTHON, aggregate, atomic_json, sha256
from completed_metrics import completed_metrics
from memlite_final.worker import loader_lock, stop_child


def read(path):
    return json.loads(Path(path).read_text())


def key(row):
    return row['task'], row['instance_id'], row['rollout_id']


def binding(path):
    return dict(path=str(path), sha256=sha256(path))


def check_binding(row):
    if sha256(row['path']) != row['sha256']:
        raise ValueError('Changed evidence: ' + row['path'])


def frozen_source(source, commit):
    source = Path(source)
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()):
        raise ValueError('Original inference/simulator source must remain exact and clean')


def physics_error(text):
    """Narrow diagnosis; OOM, timeouts and malformed model output are NOT this."""
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    pattern = (r'AssertionError: /World/scene_(\d+)/controllable__r1pro__robot_r1/'
               r'base_link orientation tensor\(\[nan, nan, nan, nan\]\) is not a unit quaternion\.')
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1 or 'rigid_dynamic_prim.py' not in text or 'Traceback (most recent call last)' not in text:
        return None
    return dict(kind='simulator_nonfinite_robot_pose', scene_slot=int(matches[0][1]),
                exception=matches[0][0], official_score_available=False,
                cause='Undetermined: finite model actions do not prove stable physics or correct action semantics')


def exact_frozen_weights(value):
    return (value.get('weights_unchanged') is True and value.get('optimizer_updates') == 0
            and value.get('observer_head_unchanged') is True
            and isinstance(value.get('observer_before_sha256'), str)
            and value['observer_before_sha256'] == value.get('observer_after_sha256'))


def original_record(metric, row):
    metric = Path(metric)
    video = metric.parent.parent / 'videos' / (metric.stem + '.mp4')
    stream = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames,r_frame_rate,width,height', '-of', 'json', str(video)], text=True))['streams'][0]
    if (int(stream['nb_frames']) != row['steps'] or stream['r_frame_rate'] != '30/1'
            or (stream['width'], stream['height']) != (672, 448)):
        raise ValueError('Original video and physical clock disagree: ' + str(video))
    return dict(task=row['task'], instance_id=row['instance_id'], rollout_id=row['rollout_id'],
                steps=row['steps'], metrics=binding(metric), video=binding(video), video_probe=stream)


def freeze(prior, output):
    prior, output = Path(prior).resolve(), Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('New receipt in an existing evidence directory required')
    m, state = read(prior / 'manifest.json'), read(prior / 'status.json')
    if (m['schema'] != 'memlite_final_public100_run_v1' or state['status'] != 'needs_diagnosis'
            or not (prior / 'STOP').exists() or set(state['failed_gpus']) != {'6'}):
        raise ValueError('This explicit diagnosis applies only to the original single-GPU6 stop')
    workers = {}
    for gpu in range(8):
        path = prior / 'workers' / f'gpu_{gpu}'
        worker = read(path / 'status.json')
        weights = read(path / 'model/weights-final.json')
        if not exact_frozen_weights(weights) or worker['status'] != 'failed_no_retry':
            raise ValueError('Missing exact post-stop frozen-weight receipt')
        for pid in (worker['pid'], worker.get('model_pid'), worker.get('sim_pid')):
            if pid is not None and Path(f'/proc/{pid}').exists():
                raise ValueError('Original process still exists; do not seal live output')
        if gpu != 6 and not any(s in worker.get('error', '') for s in (
                'Requested stop', 'Requested administrative stop', 'timed out or stopped')):
            raise ValueError('Unclassified independent worker failure')
        workers[gpu] = worker
    failed_part = workers[6]['part']
    log = prior / 'workers/gpu_6' / (failed_part + '.log')
    diagnosis = physics_error(log.read_text())
    if diagnosis is None:
        raise ValueError('Original log does not prove the registered physics exception')
    records, interrupted, errors, evidence = [], [], [], []
    for part in sorted((prior / 'parts').iterdir()):
        part_state = read(part / 'status.json')
        for metric, row in completed_metrics(part):
            records.append(original_record(metric, row))
        for attempt in sorted((part / 'attempts').glob('*.json')):
            receipt = read(attempt)
            evidence.append(binding(attempt))
            if receipt['status'] == 'completed':
                continue
            if receipt['status'] != 'running':
                raise ValueError('Unclassified original attempt state')
            for instance in receipt['instance_ids']:
                row = dict(task=part_state['task'], instance_id=instance, rollout_id=receipt['rollout'],
                           original_attempt=binding(attempt))
                if part.name == failed_part:
                    errors.append(row | dict(diagnosis=diagnosis, log=binding(log), rerun_allowed=False))
                else:
                    interrupted.append(row | dict(reason='Other worker triggered global administrative STOP',
                                                   rerun_allowed=True, best_of_n=False))
    if len(errors) != 4 or {r['instance_id'] for r in errors} != {305, 306, 307, 308}:
        raise ValueError('Unexpected failed physical batch; requires another explicit diagnosis')
    known = [key(r) for r in records + interrupted + errors]
    expected = {key(r) for r in m['cases']}
    if len(known) != len(set(known)) or not set(known) <= expected:
        raise ValueError('Duplicate/foreign completed or interrupted original cases')
    for path in [prior / 'manifest.json', prior / 'status.json', prior / 'accepted-preflight.json', log]:
        evidence.append(binding(path))
    for gpu in range(8):
        for name in ('status.json', 'model/weights-final.json', 'model/events.jsonl'):
            evidence.append(binding(prior / 'workers' / f'gpu_{gpu}' / name))
    result = dict(schema='memlite_final_diagnosed_stop_v1', created=time.time(), prior=str(prior),
                  manifest=binding(prior / 'manifest.json'), source_commit=m['source_commit'],
                  records=records, completed=len(records), execution_errors=errors,
                  administrative_interruptions=interrupted, evidence=evidence,
                  original_outputs_unmodified=True, all_eight_models_unchanged=True,
                  policy='Keep every completed result; no reroll of the failed physical batch; '
                         'restart only global-stop interruptions and unstarted cases, never best-of-N')
    atomic_json(output, result)
    return result


def remaining_parts(m, sealed):
    """Preserve original task/slot batches, independent of success or Q."""
    expected = {key(r) for r in m['cases']}
    excluded_list = [key(r) for r in sealed['records'] + sealed['execution_errors']]
    if len(excluded_list) != len(set(excluded_list)) or not set(excluded_list) <= expected:
        raise ValueError('Duplicate or foreign excluded cases')
    excluded = set(excluded_list)
    groups = []
    for group in m['groups']:
        parts = []
        for task in group['tasks']:
            buckets = {4: [], 2: []}
            for indices in (list(range(4)), list(range(4, 8)), [8, 9]):
                keys = {(task['task'], i + 301, 0) for i in indices}
                if keys <= excluded:
                    continue
                if keys & excluded:
                    raise ValueError('Never silently regroup a partially scored physical batch')
                buckets[len(indices)].extend(indices)
            for count, indices in buckets.items():
                if indices:
                    parts.append(dict(**task, num_envs=count, indices=indices,
                                      id=f"{task['task']}-n{count}-i{indices[0]}"))
        groups.append(dict(gpu=group['gpu'], parts=parts))
    planned = [(p['task'], i + 301, 0) for g in groups for p in g['parts'] for i in p['indices']]
    if len(planned) != len(set(planned)) or set(planned) != expected - excluded:
        raise ValueError('Continuation does not cover every remaining case exactly once')
    return groups


def prepare(prior, seal, runtime_source, job):
    prior, seal, runtime_source, job = map(lambda p: Path(p).resolve(), (prior, seal, runtime_source, job))
    m, receipt = read(prior / 'manifest.json'), read(seal)
    if (job.parent != ROOT / 'runs' or job.exists() or receipt['prior'] != str(prior)
            or receipt['manifest'] != binding(prior / 'manifest.json')
            or read(prior / 'accepted-preflight.json')['num_envs'] != 4):
        raise ValueError('New continuation of this exact diagnosed batch4 run required')
    frozen_source(runtime_source, m['source_commit'])
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Freeze the orchestration source first')
    for entry in receipt['evidence']:
        check_binding(entry)
    for row in receipt['records']:
        check_binding(row['metrics']); check_binding(row['video'])
    groups = remaining_parts(m, receipt)
    job.mkdir()
    for name in ('workers', 'parts'):
        (job / name).mkdir()
    manifest = dict(schema='memlite_final_explicit_continuation_v1', created=time.time(),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    runtime_source=str(runtime_source), runtime_commit=m['source_commit'],
                    prior=str(prior), prior_manifest=binding(prior / 'manifest.json'), seal=binding(seal),
                    config_path=m['config_path'], config_sha256=m['config_sha256'],
                    pack=m['pack'], pack_manifest_sha256=m['pack_manifest_sha256'],
                    tasks=m['tasks'], cases=m['cases'], groups=groups,
                    inherited_completed=receipt['completed'], initial_execution_errors=len(receipt['execution_errors']),
                    administrative_restarts=len(receipt['administrative_interruptions']),
                    expected=1000, automatic_retries=False, policy_changed=False, optimizer_updates=0,
                    error_policy='Record diagnosed nonfinite-pose batches as missing official scores; '
                                 'restart model only for different unattempted cases; no score fabrication',
                    numerical_notice='Original model/official source/seed/batching unchanged; administrative '
                                     'process restarts are explicit and not claimed bitwise-equivalent physics')
    atomic_json(job / 'manifest.json', manifest)
    return manifest


def registration(job):
    job = Path(job).resolve(); m = read(job / 'manifest.json')
    if job.parent != ROOT / 'runs' or m['schema'] != 'memlite_final_explicit_continuation_v1':
        raise ValueError('Wrong continuation registration')
    frozen_source(REPO, m['source_commit'])
    frozen_source(m['runtime_source'], m['runtime_commit'])
    for name in ('prior_manifest', 'seal'):
        check_binding(m[name])
    if sha256(m['config_path']) != m['config_sha256']:
        raise ValueError('Policy config changed')
    return job, m


def unstarted_tail(part, attempts):
    """An exception may prevent later batches from starting; never reroll them."""
    issued = [i for _, row in attempts for i in row['indices']]
    if len(issued) != len(set(issued)) or issued != part['indices'][:len(issued)]:
        raise ValueError('Attempted indices are not a unique original prefix')
    for _, row in attempts:
        if row['instance_ids'] != [i + 301 for i in row['indices']] or row['rollout'] != 0:
            raise ValueError('Physical attempt identity changed')
    remaining = part['indices'][len(issued):]
    if not remaining:
        return None
    if len(remaining) % part['num_envs']:
        raise ValueError('Never silently rebatch unstarted instances')
    return part | dict(indices=remaining, id=part['id'] + '-unstarted-i' + str(remaining[0]),
                       predecessor=part['id'])


def require_no_uncommitted_metrics(output):
    """Do not classify an early-finished sibling as an execution error.

    The official evaluator can publish one completed case before the whole
    physical batch finishes. Such a record needs a separate per-case trace/ACK
    audit; this conservative continuation must stop instead of discarding it
    or silently downgrading a real success when another slot later crashes.
    """
    known = {p for p, _ in completed_metrics(output)}
    if set((Path(output) / 'json').glob('*.json')) != known:
        raise ValueError('Uncommitted official case records require independent physical audit; do not label them errors')


def worker(job, gpu):
    job, m = registration(job)
    if not 0 <= gpu < 8:
        raise ValueError('Unknown GPU')
    path = job / 'workers' / f'gpu_{gpu}'; path.mkdir(exist_ok=False)
    lock = (SIM_ROOT / f'gpu_{gpu}.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(m['runtime_source']); scripts = source / 'scripts/eval'; runtime = scripts / 'memlite_final'
    env = {k: v for k, v in os.environ.items() if not k.startswith('RL_')}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=str(source / 'src'))
    state = dict(status='loading', pid=os.getpid(), gpu=gpu, started=time.time(), completed_parts=[],
                 diagnosed_failed_parts=[], model_generations=[], runtime_commit=m['runtime_commit'])
    model = sim = loader = None; model_path = None

    def publish():
        state['updated'] = time.time(); atomic_json(path / 'status.json', state)

    def retire_model():
        nonlocal model
        if model is None:
            return
        if model.poll() is None:
            (model_path / 'STOP').touch(); deadline = time.monotonic() + 180
            while model.poll() is None and time.monotonic() < deadline:
                time.sleep(3)
            stop_child(model)
        final = model_path / 'weights-final.json'
        if not final.exists() or not exact_frozen_weights(read(final)):
            raise ValueError('No exact frozen-weight verification; cannot advance to another case')
        state['model_generations'].append(binding(final)); model = None; publish()

    try:
        publish()
        pending = list(m['groups'][gpu]['parts'])
        while pending:
            part = pending.pop(0)
            if (job / 'STOP').exists():
                raise InterruptedError('Administrative stop')
            if shutil.disk_usage(job).free < 150 << 30:
                raise RuntimeError('Disk safety reserve')
            if model is None:
                loader = loader_lock(job)
                model_path = path / f'model-{len(state["model_generations"]):03d}'
                with (path / (model_path.name + '.log')).open('x') as log:
                    model = subprocess.Popen([str(MODEL_PYTHON), str(runtime / 'serve.py'),
                        '--config', m['config_path'], '--pack', m['pack'], '--manifest-sha256', m['pack_manifest_sha256'],
                        '--run', str(model_path), '--port', str(19900 + gpu)], cwd=source, env=env,
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                state.update(model_pid=model.pid, model_path=str(model_path)); publish()
                deadline = time.monotonic() + 1800
                while not (model_path / 'READY').exists():
                    if model.poll() is not None or time.monotonic() > deadline or (job / 'STOP').exists():
                        raise RuntimeError('Original frozen model loading failed/timed out/stopped')
                    time.sleep(3)
                loader.close(); loader = None
            output = job / 'parts' / part['id']; loader = loader_lock(job)
            command = ['bash', str(scripts / 'memlite_sft100/launch_sim.sh'), str(runtime / 'run_task.py'),
                '--task', part['task'], '--task-id', str(part['task_id']), '--output', str(output),
                '--port', str(19900 + gpu), '--config-sha256', m['config_sha256'], '--run-id', job.name,
                '--num-envs', str(part['num_envs']), '--indices', *map(str, part['indices'])]
            logpath = path / (part['id'] + '.log')
            with logpath.open('x') as log:
                sim = subprocess.Popen(command, cwd=source, env=env | dict(EVAL_GPU=str(gpu), EVAL_SOURCE=str(runtime)),
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            atomic_json(path / (part['id'] + '-command.json'), dict(argv=command, runtime_commit=m['runtime_commit']))
            state.update(status='evaluating', sim_pid=sim.pid, part=part['id']); publish()
            previous = None; last = time.monotonic()
            while sim.poll() is None:
                if loader is not None and (output / 'scene.ready').exists():
                    loader.close(); loader = None
                if (job / 'STOP').exists():
                    raise InterruptedError('Administrative stop')
                service = read(model_path / 'status.json')
                # A broken connection after a physics exception can precede
                # simulator teardown. Wait for that child and its actual log.
                if service.get('status') == 'failed' or model.poll() is not None:
                    sim.wait(timeout=120); break
                if service.get('requests') != previous:
                    previous = service.get('requests'); last = time.monotonic()
                if time.monotonic() - last > 1800 or shutil.disk_usage(job).free < 150 << 30:
                    raise RuntimeError('Progress/disk safety stop')
                time.sleep(5)
            if loader is not None:
                loader.close(); loader = None
            receipt = read(output / 'status.json') if (output / 'status.json').exists() else {}
            if sim.returncode == 0 and receipt.get('status') == 'completed':
                rows = completed_metrics(output, final=True)
                if len(rows) != len(part['indices']):
                    raise ValueError('Missing original case/ACK records')
                for metric, row in rows:
                    video = output / 'videos' / (metric.stem + '.mp4')
                    if not video.is_file() or video.stat().st_size < 1000:
                        raise ValueError('Missing original video')
                state['completed_parts'].append(part['id']); publish(); continue
            diagnosis = physics_error(logpath.read_text())
            if diagnosis is None:
                raise RuntimeError('Unclassified failure; preserve evidence for diagnosis: ' + part['id'])
            require_no_uncommitted_metrics(output)
            attempts = [(p, read(p)) for p in sorted((output / 'attempts').glob('*.json'))]
            unfinished = [(p, r) for p, r in attempts if r['status'] != 'completed']
            if len(unfinished) != 1 or unfinished[0][1]['status'] != 'running':
                raise ValueError('Cannot bind physics failure to one physical batch')
            attempt, failed = unfinished[0]
            tail = unstarted_tail(part, attempts)
            if tail is not None:
                atomic_json(path / (part['id'] + '-unstarted-tail.json'), tail)
                pending.insert(0, tail)
            errors = [dict(task=part['task'], instance_id=i, rollout_id=0, original_attempt=binding(attempt),
                           diagnosis=diagnosis, log=binding(logpath), rerun_allowed=False)
                      for i in failed['instance_ids']]
            atomic_json(path / (part['id'] + '-execution-errors.json'), errors)
            state['diagnosed_failed_parts'].append(part['id']); publish(); retire_model()
        state['status'] = 'completed_with_execution_errors' if state['diagnosed_failed_parts'] else 'completed'
    except BaseException as error:
        state.update(status='needs_diagnosis', error=repr(error)); raise
    finally:
        if loader is not None:
            loader.close()
        stop_child(sim)
        try:
            retire_model()
        except BaseException as error:
            state.update(status='needs_diagnosis', final_verification_error=repr(error))
        state['finished'] = time.time(); publish(); lock.close()
    if state['status'] == 'needs_diagnosis':
        raise RuntimeError('Continuation worker needs diagnosis')


def summary_from_rows(tasks, records, errors):
    summary = aggregate(tasks, records)
    seen = {key(r) for r in records}; failed = [key(r) for r in errors]
    expected = {(t, i, 0) for t in tasks for i in range(301, 311)}
    if len(failed) != len(set(failed)) or seen & set(failed) or not set(failed) <= expected:
        raise ValueError('Duplicate, scored, or foreign execution-error case')
    coverage = len(seen) + len(failed)
    q, sr = summary.pop('official_q_score'), summary.pop('official_sr')
    summary.update(status=('completed_with_execution_errors' if errors else 'completed') if coverage == 1000 else 'running',
                   coverage=coverage, execution_errors=len(errors), unaccounted=1000-coverage,
                   official_full_q_score=q if len(seen) == 1000 else None,
                   official_full_sr=sr if len(seen) == 1000 else None,
                   conservative_q_errors_and_missing_as_zero=q, conservative_sr_errors_and_missing_as_zero=sr,
                   fabricated_official_metrics=0, automatic_rerolls=0)
    return summary


def report(job, *, final=False):
    job = Path(job); m = read(job / 'manifest.json'); sealed = read(m['seal']['path'])
    records = []; inventory = list(sealed['records']); errors = list(sealed['execution_errors'])
    for row in sealed['records']:
        check_binding(row['metrics'])
        if final:
            check_binding(row['video'])
        records.append(read(row['metrics']['path']))
    for part in sorted((job / 'parts').glob('*')):
        for metric, row in completed_metrics(part):
            records.append(row)
            if final:
                inventory.append(original_record(metric, row))
    for path in sorted((job / 'workers').glob('*/*-execution-errors.json')):
        errors.extend(read(path))
    summary = summary_from_rows(m['tasks'], records, errors)
    summary.update(runtime_commit=m['runtime_commit'], orchestrator_commit=m['source_commit'],
                   inherited_completed=m['inherited_completed'], administrative_restarts=m['administrative_restarts'],
                   policy_changed=False, optimizer_updates=0, original_outputs_unmodified=True)
    atomic_json(job / 'summary.json', summary)
    if final:
        if summary['unaccounted']:
            raise ValueError('Cannot claim all cases covered')
        atomic_json(job / 'inventory.json', inventory)
        atomic_json(job / 'execution-errors.json', errors)
    return summary


def supervise(job):
    job, m = registration(job)
    if (job / 'status.json').exists() or (job / 'STOP').exists():
        raise ValueError('New explicitly registered continuation only')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('Do not displace GPU processes')
    lock = (job / 'supervisor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='running', pid=os.getpid(), started=time.time(), completed_gpus=[], failed_gpus={})
    children = {}
    try:
        for gpu in range(8):
            with (job / f'gpu_{gpu}.log').open('x') as log:
                children[gpu] = subprocess.Popen([sys.executable, __file__, 'worker', '--job', str(job), '--gpu', str(gpu)],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        state['worker_pids'] = {str(g): p.pid for g, p in children.items()}
        while children:
            if (job / 'STOP').exists():
                raise InterruptedError('Administrative stop')
            for gpu, process in list(children.items()):
                if process.poll() is None:
                    continue
                path = job / 'workers' / f'gpu_{gpu}' / 'status.json'
                row = read(path) if path.exists() else dict(status='missing')
                if process.returncode or row['status'] not in ('completed', 'completed_with_execution_errors'):
                    state['failed_gpus'][str(gpu)] = row
                    raise RuntimeError('Unclassified worker failure; needs diagnosis, never automatic reroll')
                state['completed_gpus'].append(gpu); del children[gpu]
            summary = report(job)
            state.update(completed=summary['completed'], coverage=summary['coverage'], updated=time.time())
            atomic_json(job / 'status.json', state)
            if children:
                time.sleep(30)
        summary = report(job, final=True); state['status'] = summary['status']
    except BaseException as error:
        state.update(status='needs_diagnosis', error=repr(error)); (job / 'STOP').touch(); raise
    finally:
        deadline = time.monotonic() + 300
        while any(p.poll() is None for p in children.values()) and time.monotonic() < deadline:
            time.sleep(3)
        state['finished'] = time.time(); atomic_json(job / 'status.json', state); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'prepare', 'worker', 'supervise', 'report'))
    for name in ('prior', 'seal', 'runtime-source', 'job', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--gpu', type=int); parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze(args.prior, args.output)
        print(json.dumps(dict(completed=result['completed'], execution_errors=len(result['execution_errors']),
                              administrative_interruptions=len(result['administrative_interruptions']))))
    elif args.command == 'prepare':
        result = prepare(args.prior, args.seal, args.runtime_source, args.job)
        print(json.dumps(dict(job=str(args.job), cases=sum(len(p['indices']) for g in result['groups'] for p in g['parts']))))
    elif args.command == 'worker':
        worker(args.job, args.gpu)
    elif args.command == 'supervise':
        supervise(args.job)
    else:
        result = report(args.job, final=args.final)
        print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}))
