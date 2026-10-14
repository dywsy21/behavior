"""Outcome-independent continuation of the frozen terminal champion evaluator.

Only orchestration changes. Original model service, simulator entry, configs,
seeds and physical batches remain fixed. Diagnosed failures consume their
attempts; unrelated administrative interruptions are explicitly disclosed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'memlite_sft100')]
import continue_memlite_final as shared
from common import ROOT, SIM_ROOT, atomic_json, sha256
from completed_metrics import completed_metrics
from loader_admission import loader_lock

read, key, binding = shared.read, shared.key, shared.binding
check_binding, frozen_source, require_exited = shared.check_binding, shared.frozen_source, shared.require_exited
SCHEMA = 'champion_final_explicit_continuation_v1'
ORIGINAL = 'champion_final_public100_run_v1'
IDENTITY = ('tasks', 'cases', 'configs', 'export_sha256', 'terminal_path', 'terminal_sha256',
            'official_commit', 'nominal_horizons', 'task_csv_sha256', 'model_python',
            'model_overlay', 'checkpoint', 'policy_seed', 'simulator_seed', 'transport')


def reset_object_error(text):
    """One observed official reset traceback, never a blanket AttributeError."""
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    expected = "AttributeError: 'NoneType' object has no attribute 'load_state'"
    errors = re.findall(r'\b[A-Za-z_][\w.]*(?:Error|Exception): [^\r\n]+', text)
    frames = ('omnigibson/eval/evaluator.py', 'in evaluate_instances_batched',
              'load_fn(dict(env_idx_to_instance))', 'in load_fn', 'in load_batch',
              'in _load_instance_state', 'obj.load_state(tro_state, serialized=False)')
    if (text.count('Traceback (most recent call last):') != 1 or errors != [expected]
            or not all(s in text for s in frames)
            or re.search(r'CUDA out of memory|CUDA error:|device-side assert', text, re.I)):
        return None
    return dict(kind='simulator_missing_task_object_on_reset', scene_slot=None,
                exception=expected, failure_phase='official_instance_state_load',
                official_score_available=False, failed_slot_known=False,
                cause='Official object_scope lookup returned None while restoring task-object state. '
                      'The object, slot and reason it became unavailable were not logged.')


def execution_error(text):
    return reset_object_error(text) or shared.physics_error(text) or shared.articulation_error(text)


def validate_weights(value, original, gpu):
    if (value.get('status') != 'closed' or value.get('weights_unchanged') is not True
            or value.get('optimizer_updates') != 0
            or value.get('source_commit') != original['source_commit']
            or value.get('config_sha256') != original['configs'][gpu]['sha256']
            or value.get('asset_receipt_sha256') != original['export_sha256']
            or not re.fullmatch('[0-9a-f]{64}', value.get('parameters_before', ''))
            or value.get('parameters_before') != value.get('parameters_after')):
        raise ValueError('Exact unchanged terminal champion weights required')


def root_registration(job, seen=None):
    job = Path(job).resolve(); seen = set() if seen is None else seen
    if job in seen:
        raise ValueError('Cyclic champion lineage')
    seen.add(job); m = read(job / 'manifest.json')
    if m['schema'] == ORIGINAL:
        if (len(m['cases']) != 1000 or len({key(r) for r in m['cases']}) != 1000
                or len(m['configs']) != 8 or m['optimizer_updates'] != 0
                or m['automatic_retries'] is not False or m['policy_seed'] != 17
                or m['simulator_seed'] != 0):
            raise ValueError('Original public1000 identity required')
        for i, cfg in enumerate(m['configs']):
            if cfg['gpu'] != i:
                raise ValueError('Original GPU config order changed')
            check_binding(cfg)
        if sha256(m['terminal_path']) != m['terminal_sha256']:
            raise ValueError('Terminal training receipt changed')
        return job, m, binding(job / 'manifest.json')
    if m['schema'] != SCHEMA or m['prior_manifest']['path'] != str(Path(m['prior']) / 'manifest.json'):
        raise ValueError('Unknown champion parent')
    check_binding(m['prior_manifest']); check_binding(m['seal'])
    root, original, pin = root_registration(m['prior'], seen)
    if (m['root_manifest'] != pin or m['runtime_commit'] != original['source_commit']
            or m['policy_changed'] is not False or m['optimizer_updates'] != 0
            or m['automatic_retries'] is not False or any(m[k] != original[k] for k in IDENTITY)):
        raise ValueError('Champion identity changed across continuation')
    return root, original, pin


def checked_parts(job, m, original):
    groups = (shared.remaining_parts(original, dict(records=[], execution_errors=[]))
              if m['schema'] == ORIGINAL else m['groups'])
    parts = {p['id']: (g['gpu'], p) for g in groups for p in g['parts']}
    if len(parts) != sum(len(g['parts']) for g in groups) or [g['gpu'] for g in groups] != list(range(8)):
        raise ValueError('Duplicate part or changed GPU schedule')
    for g in groups:
        for path in sorted((job / 'workers' / f"gpu_{g['gpu']}").glob('*-unstarted-tail.json')):
            tail = read(path); predecessor = tail['predecessor']
            if predecessor not in parts or tail['id'] in parts or parts[predecessor][0] != g['gpu']:
                raise ValueError('Unknown/duplicate champion tail')
            attempts = [(p, read(p)) for p in sorted((job / 'parts' / predecessor / 'attempts').glob('*.json'))]
            if tail != shared.unstarted_tail(parts[predecessor][1], attempts):
                raise ValueError('Tail differs from actual unattempted suffix')
            parts[tail['id']] = (g['gpu'], tail)
    return parts


def freeze(prior, runtime_source, output, *, precontrol_reset_certificate=None):
    prior, runtime_source, output = map(lambda p: Path(p).resolve(), (prior, runtime_source, output))
    if output.exists() or not output.parent.is_dir():
        raise ValueError('New stop receipt required')
    m, state = read(prior / 'manifest.json'), read(prior / 'status.json')
    if state['status'] != 'needs_diagnosis' or not state.get('finished') or not (prior / 'STOP').exists():
        raise ValueError('Actual stopped evaluation required')
    require_exited([state['pid'], *state['worker_pids'].values()])
    root, original, pin = root_registration(prior); frozen_source(runtime_source, original['source_commit'])
    if m['schema'] == SCHEMA and str(runtime_source) != m['runtime_source']:
        raise ValueError('Original runtime path changed')
    reset_certificate = None
    if precontrol_reset_certificate is not None:
        from champion_reset_timeout import verify_certificate
        reset_certificate = verify_certificate(precontrol_reset_certificate)
        if reset_certificate['job'] != str(prior):
            raise ValueError('Certificate belongs to another run')
    records, errors, interrupted = [], [], []
    evidence = [binding(prior / n) for n in ('manifest.json', 'status.json', 'STOP')]
    evidence += [pin, binding(root / 'accepted-preflight.json')]
    if reset_certificate:
        evidence += [binding(precontrol_reset_certificate), *reset_certificate['evidence']]
    if m['schema'] == SCHEMA:
        sealed = read(m['seal']['path'])
        if sealed['prior'] != m['prior'] or sealed['manifest'] != m['prior_manifest']:
            raise ValueError('Parent seal identity differs')
        evidence += [m['seal'], *sealed['evidence']]
        records.extend(sealed['records']); errors.extend(sealed['execution_errors'])
    for item in evidence:
        check_binding(item)
    for item in records:
        check_binding(item['metrics']); check_binding(item['video'])
    parts = checked_parts(prior, m, original); workers = {}
    for gpu in range(8):
        wpath = prior / 'workers' / f'gpu_{gpu}'; w = read(wpath / 'status.json'); workers[gpu] = w
        require_exited([w['pid'], w.get('model_pid'), w.get('sim_pid')])
        if (not w.get('finished') or w.get('final_verification_error')
                or w.get('status') not in ('failed_no_retry', 'needs_diagnosis', 'completed', 'completed_with_execution_errors')):
            raise ValueError('Missing worker terminal verification')
        finals = [binding(wpath / 'model/result.json')] if m['schema'] == ORIGINAL else w['model_generations']
        if not finals:
            raise ValueError('Missing model generation; requires separate pre-admission diagnosis')
        if m['schema'] == SCHEMA and finals != [binding(p / 'result.json') for p in sorted(wpath.glob('model-*')) if p.is_dir()]:
            raise ValueError('Unverified model generation')
        for receipt in finals:
            check_binding(receipt); value = read(receipt['path']); validate_weights(value, original, gpu)
            require_exited([value.get('pid')]); evidence.append(receipt)
            model = Path(receipt['path']).parent
            evidence.append(binding(model / 'status.json'))
            if (model / 'events.jsonl').exists():
                evidence.append(binding(model / 'events.jsonl'))
        evidence.append(binding(wpath / 'status.json'))
    diagnosed = set()
    for p in sorted((prior / 'parts').iterdir()):
        if p.name not in parts:
            raise ValueError('Unregistered physical part')
        gpu, part = parts[p.name]; w = workers[gpu]; ps = read(p / 'status.json')
        if (any(ps.get(k) != part[k] for k in ('task', 'task_id', 'indices'))
                or ps.get('source_commit') != original['source_commit'] or ps.get('optimizer_updates') != 0
                or ps.get('config_sha256') != original['configs'][gpu]['sha256']):
            raise ValueError('Original physical identity changed')
        shared.require_no_uncommitted_metrics(p)
        records.extend(shared.original_record(metric, row) for metric, row in completed_metrics(p))
        attempts = [(a, read(a)) for a in sorted((p / 'attempts').glob('*.json'))]
        shared.unstarted_tail(part, attempts)
        if any(a.get('policy_seed') != 17 or a.get('simulator_seed') != 0
               or a.get('max_steps') != original['nominal_horizons'][part['task']] for _, a in attempts):
            raise ValueError('Original seeds changed')
        log = prior / 'workers' / f'gpu_{gpu}' / (p.name + '.log'); text = log.read_text()
        command = log.with_name(p.name + '-command.json')
        evidence += [binding(p / 'status.json'), binding(log), binding(command), *[binding(a) for a, _ in attempts]]
        unfinished = [(a, r) for a, r in attempts if r['status'] != 'completed']
        if not unfinished:
            if (attempts and ps['status'] != 'completed') or (not attempts and 'Traceback (most recent call last)' in text):
                raise ValueError('Unclassified pre-attempt/part failure')
            continue
        if len(unfinished) != 1 or unfinished[0] != attempts[-1] or unfinished[0][1]['status'] != 'running':
            raise ValueError('Unknown incomplete attempt')
        a, attempt = unfinished[0]
        if len(attempt['instance_ids']) != part['num_envs']:
            raise ValueError('Partial physical batch')
        rows = [dict(task=part['task'], instance_id=i, rollout_id=0, original_attempt=binding(a)) for i in attempt['instance_ids']]
        diagnosis = execution_error(text)
        ledger = log.with_name(p.name + '-execution-errors.json')
        if diagnosis is None and m.get('continue_on_batch_errors') is True and ledger.exists():
            entries = read(ledger)
            if entries and entries[0]['diagnosis']['kind'] == 'unclassified_operational_batch_error':
                diagnosis = entries[0]['diagnosis']
                evidence.append(validate_operational_error(diagnosis, log, a, part))
        if diagnosis:
            if diagnosis['scene_slot'] is not None and not 0 <= diagnosis['scene_slot'] < part['num_envs']:
                raise ValueError('Failure slot outside batch')
            entries = [r | dict(diagnosis=diagnosis, log=binding(log), rerun_allowed=False) for r in rows]
            if ledger.exists():
                if read(ledger) != entries:
                    raise ValueError('Original error ledger changed')
                evidence.append(binding(ledger))
            elif w.get('part') != p.name:
                raise ValueError('Unregistered historical execution error')
            errors.extend(entries); diagnosed.add(str(gpu))
        elif (reset_certificate and reset_certificate['gpu'] == gpu
              and reset_certificate['part'] == p.name and reset_certificate['original_attempt'] == binding(a)):
            interrupted.extend(r | dict(log=binding(log), last_physical_status=binding(p/'status.json'),
                reason=reset_certificate['reason'], rerun_allowed=True, best_of_n=False,
                precontrol_reset_certificate=binding(precontrol_reset_certificate)) for r in rows)
            diagnosed.add(str(gpu))
        else:
            ui = shared.nonfatal_property_ui(text, ps, attempt) if 'Traceback (most recent call last)' in text else None
            if (w.get('part') != p.name or w.get('error') != "InterruptedError('Administrative stop')"
                    or ('Traceback (most recent call last)' in text and ui is None)
                    or (re.search(r'\b[A-Za-z_][\w.]*(?:Error|Exception):', text) and ui is None)
                    or re.search(r'CUDA out of memory|CUDA error:|device-side assert', text, re.I)):
                raise ValueError('Unknown failure is not an administrative restart: ' + p.name)
            interrupted.extend(r | dict(log=binding(log), last_physical_status=binding(p / 'status.json'),
                reason='Other worker triggered global STOP', rerun_allowed=True, best_of_n=False,
                reviewed_nonfatal_diagnostic=ui) for r in rows)
    if not set(state['failed_gpus']) <= diagnosed:
        raise ValueError('Supervisor failure still undiagnosed')
    for gpu, w in workers.items():
        if w.get('error') and str(gpu) not in diagnosed and w['error'] != "InterruptedError('Administrative stop')":
            raise ValueError('Additional unknown worker failure')
    known = [key(r) for r in records + errors + interrupted]; expected = {key(r) for r in original['cases']}
    if len(known) != len(set(known)) or not set(known) <= expected:
        raise ValueError('Duplicate or foreign original outcome')
    receipt = dict(schema='champion_final_diagnosed_stop_v1', created=time.time(), prior=str(prior),
        manifest=binding(prior / 'manifest.json'), root_manifest=pin, runtime_source=str(runtime_source),
        source_commit=original['source_commit'], records=records, completed=len(records), execution_errors=errors,
        administrative_interruptions=interrupted, unstarted=1000-len(known), evidence=evidence,
        all_loaded_models_unchanged=True, original_outputs_unmodified=True,
        policy='No reroll of scored or diagnosed failed attempts; explicit administrative restarts '
               'or independently certified pre-control coordinator interruptions only')
    shared.remaining_parts(original, receipt); atomic_json(output, receipt)
    return receipt


def remaining_groups(original, receipt, isolate_batches=False):
    if type(isolate_batches) is not bool:
        raise ValueError('Explicit batch isolation boolean required')
    groups = shared.remaining_parts(original, receipt)
    if not isolate_batches:
        return groups
    return [dict(gpu=g['gpu'], parts=[p | dict(indices=indices,
        id=f"{p['task']}-n{p['num_envs']}-i{indices[0]}")
        for p in g['parts'] for start in range(0, len(p['indices']), p['num_envs'])
        for indices in [p['indices'][start:start+p['num_envs']]]]) for g in groups]


def prepare(prior, seal, runtime_source, job, *, isolate_batches=False, continue_on_errors=False):
    prior, seal, runtime_source, job = map(lambda p: Path(p).resolve(), (prior, seal, runtime_source, job))
    root, original, pin = root_registration(prior); receipt = read(seal)
    if (job.parent != ROOT / 'runs' or job.exists() or receipt['schema'] != 'champion_final_diagnosed_stop_v1'
            or receipt['prior'] != str(prior) or receipt['manifest'] != binding(prior / 'manifest.json')
            or receipt['root_manifest'] != pin or receipt['completed'] != len(receipt['records'])
            or receipt['source_commit'] != original['source_commit'] or receipt['runtime_source'] != str(runtime_source)
            or read(root / 'accepted-preflight.json')['num_envs'] != 4):
        raise ValueError('Exact diagnosed champion continuation required')
    frozen_source(runtime_source, original['source_commit'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    frozen_source(REPO, commit)
    for item in receipt['evidence']:
        check_binding(item)
    for item in receipt['records']:
        check_binding(item['metrics']); check_binding(item['video'])
    for case in original['cases']:
        check_binding(dict(path=case['instance_path'], sha256=case['instance_sha256']))
    if type(continue_on_errors) is not bool or (continue_on_errors and not isolate_batches):
        raise ValueError('Continuous failure isolation requires one physical batch per process')
    groups = remaining_groups(original, receipt, isolate_batches)
    manifest = {k: original[k] for k in IDENTITY}
    manifest.update(schema=SCHEMA, created=time.time(), source_commit=commit,
        runtime_source=str(runtime_source), runtime_commit=original['source_commit'],
        prior=str(prior), prior_manifest=binding(prior / 'manifest.json'), seal=binding(seal), root_manifest=pin,
        groups=groups, inherited_completed=receipt['completed'], initial_execution_errors=len(receipt['execution_errors']),
        administrative_restarts=len(receipt['administrative_interruptions']), expected=1000,
        automatic_retries=False, policy_changed=False, optimizer_updates=0,
        isolate_physical_batches=isolate_batches, continue_on_batch_errors=continue_on_errors,
        loader_admission='original_two_slots_fifo', numerical_notice='Administrative process restarts are explicit; '
        'not claimed bitwise equivalent to uninterrupted physics. Same policy, seeds, batches and official evaluator.')
    job.mkdir()
    for n in ('workers', 'parts'):
        (job / n).mkdir()
    atomic_json(job / 'manifest.json', manifest); return manifest


def registration(job, *, coordinator_source=None):
    job = Path(job).resolve(); m = read(job / 'manifest.json')
    if job.parent != ROOT / 'runs' or m['schema'] != SCHEMA:
        raise ValueError('Explicit champion continuation required')
    frozen_source(REPO if coordinator_source is None else Path(coordinator_source), m['source_commit'])
    frozen_source(m['runtime_source'], m['runtime_commit'])
    _, original, _ = root_registration(job)
    sealed = read(m['seal']['path'])
    isolated = m.get('isolate_physical_batches', False); continuous = m.get('continue_on_batch_errors', False)
    if type(continuous) is not bool or (continuous and not isolated):
        raise ValueError('Invalid failure isolation mode')
    if (m['groups'] != remaining_groups(original, sealed, isolated)
            or m['inherited_completed'] != len(sealed['records'])
            or m['initial_execution_errors'] != len(sealed['execution_errors'])
            or m['administrative_restarts'] != len(sealed['administrative_interruptions'])):
        raise ValueError('Remaining case plan/counts changed')
    return job, m


def operational_batch_error(job, part, error, returncode, logpath):
    """Consume one attempted physical batch, preserving uncertainty and originals.

    Unattempted startup errors and uncommitted metrics still require diagnosis;
    they never silently become scored failures or automatic policy rerolls.
    """
    output = job / 'parts' / part['id']
    shared.require_no_uncommitted_metrics(output)
    attempts = sorted((output/'attempts').glob('*.json'))
    if len(attempts) != 1:
        raise ValueError('Operational isolation needs one original physical attempt')
    attempt = read(attempts[0])
    if (attempt['status'] != 'running' or attempt['indices'] != part['indices']
            or len(part['indices']) != part['num_envs']):
        raise ValueError('Ambiguous operational batch boundary')
    receipt = logpath.with_name(part['id'] + '-operational-error.json')
    value = dict(schema='champion_operational_batch_error_v1', part=part, error=error,
        returncode=returncode, log=binding(logpath), original_attempt=binding(attempts[0]),
        physical_status=binding(output/'status.json'), command=binding(logpath.with_name(part['id']+'-command.json')),
        policy_failure_claimed=False, official_score_available=False, rerun_allowed=False)
    atomic_json(receipt, value)
    return dict(kind='unclassified_operational_batch_error', scene_slot=None,
        failure_phase='runtime_unknown_preserved_for_diagnosis', official_score_available=False,
        policy_failure_claimed=False, operational_receipt=binding(receipt))


def validate_operational_error(diagnosis, log, attempt, plan):
    receipt = diagnosis['operational_receipt']; check_binding(receipt); row = read(receipt['path'])
    for field in ('log', 'original_attempt', 'physical_status', 'command'):
        check_binding(row[field])
    if (row.get('schema') != 'champion_operational_batch_error_v1' or row['part'] != plan
            or row['log'] != binding(log) or row['original_attempt'] != binding(attempt)
            or row['official_score_available'] is not False or row['policy_failure_claimed'] is not False
            or row['rerun_allowed'] is not False or not (row['error'] or row['returncode'])):
        raise ValueError('Changed or unsupported operational error receipt')
    return receipt


def model_environment(m, gpu):
    source = Path(m['runtime_source']); policy = read(m['configs'][gpu]['path'])
    env = {k: v for k, v in os.environ.items() if not k.startswith('RL_') and k != 'JAX_PLATFORMS'}
    paths = [source / 'src', ROOT / policy['reference'] / 'src', ROOT / policy['openpi_reference'] / 'src',
             ROOT / policy['openpi_reference'] / 'packages/openpi-client/src', Path(m['model_overlay'])]
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    return env


def worker(job, gpu):
    job, m = registration(job)
    if type(gpu) is not int or not 0 <= gpu < 8:
        raise ValueError('Registered GPU required')
    _, original, _ = root_registration(job)
    runtime = Path(m['runtime_source']) / 'scripts/eval'; cfg = m['configs'][gpu]
    path = job / 'workers' / f'gpu_{gpu}'; path.mkdir(exist_ok=False)
    lock = (SIM_ROOT / f'gpu_{gpu}.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='loading', pid=os.getpid(), gpu=gpu, started=time.time(), completed_parts=[],
                 diagnosed_failed_parts=[], model_generations=[], runtime_commit=m['runtime_commit'])
    def publish():
        state['updated'] = time.time(); atomic_json(path / 'status.json', state)
    model = sim = loader = model_path = None
    env = model_environment(m, gpu)
    def retire_model():
        nonlocal model, model_path
        if model is None:
            return
        if model.poll() is None:
            os.killpg(model.pid, signal.SIGTERM)
            try:
                model.wait(timeout=240)
            except subprocess.TimeoutExpired:
                shared.stop_child(model)
        receipt = model_path / 'result.json'; validate_weights(read(receipt), original, gpu)
        state['model_generations'].append(binding(receipt)); model = None; model_path = None; publish()
    pending = list(m['groups'][gpu]['parts'])
    try:
        while pending:
            if (job / 'STOP').exists():
                raise InterruptedError('Administrative stop')
            if shutil.disk_usage(job).free < 150 << 30:
                raise RuntimeError('Disk safety reserve')
            if model is None:
                loader = loader_lock(job); model_path = path / f"model-{len(state['model_generations']):03d}"
                with (path / (model_path.name + '.log')).open('x') as log:
                    model = subprocess.Popen([m['model_python'], str(runtime / 'serve_champion_official.py'),
                        '--config', cfg['path'], '--output', str(model_path), '--profile-memory'], env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(model_pid=model.pid, model_path=str(model_path)); publish(); deadline = time.monotonic()+1800
                while True:
                    status = model_path / 'status.json'
                    if status.exists() and read(status)['status'] == 'ready':
                        break
                    if model.poll() is not None or time.monotonic() > deadline or (job / 'STOP').exists():
                        raise RuntimeError('Frozen model loading failed/timed out/stopped')
                    time.sleep(3)
                loader.close(); loader = None
            part = pending.pop(0); output = job / 'parts' / part['id']; loader = loader_lock(job)
            command = ['bash', str(runtime / 'memlite_sft100/launch_sim.sh'), str(runtime / 'run_champion_task.py'),
                '--task', part['task'], '--task-id', str(part['task_id']), '--policy-config', cfg['path'],
                '--output', str(output), '--num-envs', str(part['num_envs']), '--indices', *map(str, part['indices'])]
            sim_env = env | dict(EVAL_GPU=str(gpu), EVAL_SOURCE=str(runtime), PYTHONPATH=str(Path(m['runtime_source']) / 'src'))
            logpath = path / (part['id'] + '.log')
            with logpath.open('x') as log:
                sim = subprocess.Popen(command, env=sim_env, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(path / (part['id'] + '-command.json'), dict(argv=command, official_commit=m['official_commit']))
            state.update(status='evaluating', sim_pid=sim.pid, part=part['id']); publish()
            previous = None; last = time.monotonic(); runtime_error = None
            while sim.poll() is None:
                if loader is not None and (output / 'scene.ready').exists():
                    loader.close(); loader = None
                if (job / 'STOP').exists():
                    raise InterruptedError('Administrative stop')
                if model.poll() is not None or read(model_path / 'status.json')['status'] == 'failed_no_retry':
                    if not m.get('continue_on_batch_errors', False):
                        raise RuntimeError('Frozen model service failed')
                    runtime_error = "RuntimeError('Frozen model service failed')"; break
                events = model_path / 'events.jsonl'; progress = events.stat().st_size if events.exists() else 0
                if progress != previous:
                    previous = progress; last = time.monotonic()
                if time.monotonic()-last > 1800:
                    if not m.get('continue_on_batch_errors', False):
                        raise TimeoutError('No model progress for 30 minutes')
                    runtime_error = "TimeoutError('No model progress for 30 minutes')"; break
                if shutil.disk_usage(job).free < 150 << 30:
                    raise RuntimeError('Disk safety reserve')
                time.sleep(5)
            if runtime_error:
                shared.stop_child(sim)
            if loader is not None:
                loader.close(); loader = None
            status = read(output / 'status.json')
            if sim.returncode == 0 and status['status'] == 'completed':
                rows = completed_metrics(output, final=True)
                videos = list((output / 'videos').glob('*.mp4'))
                if len(rows) != len(part['indices']) or len(videos) != len(rows) or any(v.stat().st_size < 1000 for v in videos):
                    raise ValueError('Missing original completed outputs')
                state['completed_parts'].append(part['id']); publish(); continue
            diagnosis = execution_error(logpath.read_text())
            if diagnosis is None:
                if not m.get('continue_on_batch_errors', False):
                    raise RuntimeError('Unclassified champion failure: ' + part['id'])
                diagnosis = operational_batch_error(job, part, runtime_error or
                    status.get('error') or 'Official physical batch incomplete', sim.returncode, logpath)
            if diagnosis['scene_slot'] is not None and not 0 <= diagnosis['scene_slot'] < part['num_envs']:
                raise ValueError('Error slot outside batch')
            shared.require_no_uncommitted_metrics(output)
            attempts = [(p, read(p)) for p in sorted((output / 'attempts').glob('*.json'))]
            incomplete = [(p, r) for p, r in attempts if r['status'] != 'completed']
            if len(incomplete) != 1 or incomplete[0] != attempts[-1] or incomplete[0][1]['status'] != 'running':
                raise ValueError('Failure is not one physical batch')
            tail = shared.unstarted_tail(part, attempts)
            if tail is not None:
                atomic_json(path / (part['id'] + '-unstarted-tail.json'), tail); pending.insert(0, tail)
            attempt, failed = incomplete[0]
            errors = [dict(task=part['task'], instance_id=i, rollout_id=0, original_attempt=binding(attempt),
                           diagnosis=diagnosis, log=binding(logpath), rerun_allowed=False) for i in failed['instance_ids']]
            atomic_json(path / (part['id'] + '-execution-errors.json'), errors)
            state['diagnosed_failed_parts'].append(part['id']); publish(); retire_model()
        state['status'] = 'completed_with_execution_errors' if state['diagnosed_failed_parts'] else 'completed'
    except BaseException as error:
        state.update(status='needs_diagnosis', error=repr(error)); raise
    finally:
        if loader is not None:
            loader.close()
        shared.stop_child(sim)
        try:
            retire_model()
        except BaseException as error:
            state.update(status='needs_diagnosis', final_verification_error=repr(error))
        state['finished'] = time.time(); publish(); lock.close()


def report(job, *, final=False):
    registration(job)
    return shared.report(job, final=final)


def supervise(job):
    job, m = registration(job)
    if (job / 'status.json').exists() or (job / 'STOP').exists():
        raise ValueError('New continuation only; never implicit retry')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('Do not displace GPU owners')
    lock = (job / 'supervisor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='running', pid=os.getpid(), started=time.time(), completed_gpus=[], failed_gpus={})
    children = {}
    try:
        for gpu in range(8):
            with (job / f'gpu_{gpu}.log').open('x') as log:
                children[gpu] = subprocess.Popen([sys.executable, __file__, 'worker', '--job', str(job), '--gpu', str(gpu)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['worker_pids'] = {str(g): p.pid for g, p in children.items()}
        while children:
            if (job / 'STOP').exists():
                raise InterruptedError('Administrative stop')
            for gpu, p in list(children.items()):
                if p.poll() is None:
                    continue
                path = job / 'workers' / f'gpu_{gpu}' / 'status.json'; row = read(path) if path.exists() else {}
                if p.returncode or row.get('status') not in ('completed', 'completed_with_execution_errors'):
                    state['failed_gpus'][str(gpu)] = row
                    if not m.get('continue_on_batch_errors', False):
                        raise RuntimeError('Unknown worker failure; no automatic reroll')
                    # Quarantine only this worker; never send global STOP to healthy GPUs.
                    del children[gpu]; atomic_json(job/'status.json', state); continue
                state['completed_gpus'].append(gpu); del children[gpu]
            try:
                result = report(job)
                state.update(completed=result['completed'], coverage=result['coverage'], updated=time.time())
                state.pop('report_error', None)
            except Exception as error:
                if not m.get('continue_on_batch_errors', False):
                    raise
                state.update(report_error=repr(error), updated=time.time())
            atomic_json(job / 'status.json', state)
            if children:
                time.sleep(30)
        if state['failed_gpus']:
            state.update(status='needs_isolated_worker_recovery', finished=time.time())
            atomic_json(job/'status.json', state)
            return
        result = report(job, final=True); state['status'] = result['status']
    except BaseException as error:
        state.update(status='needs_diagnosis', error=repr(error)); (job / 'STOP').touch(); raise
    finally:
        deadline = time.monotonic()+300
        while any(p.poll() is None for p in children.values()) and time.monotonic() < deadline:
            time.sleep(3)
        state['finished'] = time.time(); atomic_json(job / 'status.json', state); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'prepare', 'worker', 'supervise', 'report'))
    for name in ('prior', 'runtime-source', 'output', 'seal', 'job', 'precontrol-reset-certificate'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--gpu', type=int); parser.add_argument('--final', action='store_true')
    parser.add_argument('--isolate-batches', action='store_true')
    parser.add_argument('--continue-on-errors', action='store_true')
    a = parser.parse_args()
    if a.command == 'freeze':
        r = freeze(a.prior, a.runtime_source, a.output, precontrol_reset_certificate=a.precontrol_reset_certificate)
        print(json.dumps(dict(completed=r['completed'], errors=len(r['execution_errors']),
                             interrupted=len(r['administrative_interruptions']), unstarted=r['unstarted'])))
    elif a.command == 'prepare':
        r = prepare(a.prior, a.seal, a.runtime_source, a.job,
                    isolate_batches=a.isolate_batches, continue_on_errors=a.continue_on_errors)
        print(json.dumps(dict(job=str(a.job), remaining=sum(len(p['indices']) for g in r['groups'] for p in g['parts']))))
    elif a.command == 'worker':
        worker(a.job, a.gpu)
    elif a.command == 'supervise':
        supervise(a.job)
    else:
        print(json.dumps({k:v for k,v in report(a.job, final=a.final).items() if k != 'tasks'}))
