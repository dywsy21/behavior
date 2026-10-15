"""Audit/collect a terminal champion continuation without rerolls or uploads.

Every scheduled case must be accounted for. Diagnosed execution errors remain
missing official results, not synthetic zero-valued JSON. Run from a separate
checkout; all model and simulator code stays frozen in its original checkout.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continue_champion_final as c
from queue_champion_after_memlite import dependency_ready


def terminal_ready(state, summary):
    if state.get('status') in ('needs_diagnosis', 'failed_no_retry', 'failed'):
        raise RuntimeError('Champion continuation needs diagnosis; do not start MEM or retry')
    if not dependency_ready(state, summary, continuation=True):
        return False
    if (state.get('failed_gpus') != {} or not state.get('finished')
            or sorted(state.get('completed_gpus', [])) != list(range(8))
            or summary.get('original_outputs_unmodified') is not True
            or summary.get('fabricated_official_metrics') != 0
            or summary.get('automatic_rerolls') != 0):
        raise ValueError('Missing champion terminal verification')
    return True


def registered_recoveries(job, recoveries):
    """Explicit sidecars, never inferred from directory names or score quality."""
    from champion_worker_recovery import registration
    job = Path(job).resolve(); registered = {}
    for recovery, source in recoveries:
        recovery, source = Path(recovery).resolve(), Path(source).resolve()
        m = c.read(recovery / 'manifest.json'); gpu = m['gpu']
        if recovery == job or gpu in registered or Path(m['parent']).resolve() != job:
            raise ValueError('Duplicate or foreign isolated recovery')
        recovery, m = registration(recovery, gpu, coordinator_source=source)
        registered[gpu] = dict(job=recovery, source=source, manifest=m,
                              certificate=c.read(m['certificate']['path']))
    return registered


def recovered_terminal_ready(job, recoveries):
    """Lightweight waiting gate; full immutable evidence is checked by the audit."""
    job = Path(job); state = c.read(job / 'status.json')
    if not recoveries:
        return terminal_ready(state, c.read(job / 'summary.json'))
    expected = {str(c.read(Path(p) / 'manifest.json')['gpu']) for p, _ in recoveries}
    if len(expected) != len(recoveries):
        raise ValueError('Duplicate recovered GPU')
    side_states = [c.read(Path(p) / 'status.json') for p, _ in recoveries]
    if any(s.get('status') not in ('running', 'completed', 'completed_with_execution_errors') for s in side_states):
        raise RuntimeError('Isolated recovery needs diagnosis; healthy workers are not stopped')
    if state.get('status') == 'running':
        return False
    if (state.get('status') != 'needs_isolated_worker_recovery' or not state.get('finished')
            or set(state.get('failed_gpus', {})) != expected
            or sorted(state.get('completed_gpus', [])) != sorted(set(range(8)) - {int(g) for g in expected})
            or state.get('error') or state.get('report_error')):
        raise ValueError('Parent terminal quarantine is not exactly covered by registered recoveries')
    if any(s['status'] == 'running' for s in side_states):
        return False
    if any(not s.get('finished') or s.get('error') for s in side_states):
        raise ValueError('Missing recovery terminal verification')
    # A just-written terminal status can precede actual exit. Wait rather than
    # declaring a failure or admitting MEM while any evaluator still owns GPUs.
    pids = [state['pid'], *state['worker_pids'].values(), *[s['pid'] for s in side_states]]
    return not any(pid and Path(f'/proc/{pid}').exists() for pid in pids)


def worker_materials(path, original, *, recovery=None, quarantined=None):
    gpu = int(path.parent.name[4:]); w = c.read(path)
    if quarantined is None:
        if (w['status'] not in ('completed', 'completed_with_execution_errors')
                or not w.get('finished') or w.get('final_verification_error') or w.get('error')):
            raise ValueError('Worker terminal verification missing')
    elif c.binding(path) != quarantined['worker']:
        raise ValueError('Parent quarantine changed after recovery registration')
    c.require_exited([w['pid'], w.get('model_pid'), w.get('sim_pid')])
    finals = [c.binding(p / 'result.json') for p in sorted(path.parent.glob('model-*')) if p.is_dir()]
    if not finals or finals != w['model_generations']:
        raise ValueError('Unverified model generation')
    evidence = [c.binding(path)]; errors = []
    for pinned in finals:
        model = Path(pinned['path']).parent
        for p in (Path(pinned['path']), model / 'status.json'):
            value = c.read(p); c.validate_weights(value, original, gpu)
            c.require_exited([value.get('pid')])
            if recovery is not None and (value.get('launcher_source_commit') != recovery['manifest']['source_commit']
                    or value.get('recovery_manifest_sha256') != c.sha256(recovery['job'] / 'manifest.json')):
                raise ValueError('Recovery model launcher provenance differs')
            evidence.append(c.binding(p))
        if (model / 'events.jsonl').exists():
            evidence.append(c.binding(model / 'events.jsonl'))
    for error in sorted(path.parent.glob('*-execution-errors.json')):
        errors.extend(c.read(error)); evidence.append(c.binding(error))
    return errors, evidence


def accounted_summary(tasks, records, errors):
    result = c.shared.summary_from_rows(tasks, records, errors)
    kinds = {'unclassified_operational_batch_error', 'simulator_nonfinite_robot_pose', 'simulator_articulation_unavailable',
             'simulator_missing_task_object_on_reset'}
    if result['unaccounted'] or not records:
        raise ValueError('All 1000 cases must be accounted for, with original results')
    if any(e.get('rerun_allowed') is not False or e['diagnosis'].get('official_score_available') is not False
           or e['diagnosis']['kind'] not in kinds for e in errors):
        raise ValueError('Only diagnosed champion execution errors may be missing')
    return result


def audited_materials(job, manifest_sha256, coordinator_source, *, recoveries=()):
    job = Path(job).resolve()
    if c.sha256(job / 'manifest.json') != manifest_sha256:
        raise ValueError('Changed champion registration')
    state, saved = c.read(job / 'status.json'), c.read(job / 'summary.json')
    if not recovered_terminal_ready(job, recoveries):
        raise ValueError('Champion is not terminal; no early partial collection')
    c.require_exited([state['pid'], *state['worker_pids'].values()])
    job, m = c.registration(job, coordinator_source=coordinator_source)
    recovered = registered_recoveries(job, recoveries)
    root, original, root_pin = c.root_registration(job)
    source = Path(m['runtime_source']); sealed = c.read(m['seal']['path'])
    lineage, admin_attempts, evidence = [], set(), []
    checked_certificates = set()
    cursor = job
    while True:
        lineage.append(cursor); parent = c.read(cursor / 'manifest.json')
        evidence.append(c.binding(cursor / 'manifest.json'))
        if cursor == root:
            break
        seal = c.read(parent['seal']['path']); evidence.append(parent['seal'])
        for item in seal['evidence']:
            c.check_binding(item)
        for item in seal['administrative_interruptions']:
            c.check_binding(item['original_attempt']); c.check_binding(item['log'])
            if item.get('rerun_allowed') is not True or item.get('best_of_n') is not False:
                raise ValueError('Unregistered administrative restart')
            if item.get('precontrol_reset_certificate'):
                pin = item['precontrol_reset_certificate']; c.check_binding(pin)
                if pin['path'] not in checked_certificates:
                    from champion_reset_timeout import verify_certificate
                    verify_certificate(pin['path']); checked_certificates.add(pin['path'])
                if c.read(pin['path'])['original_attempt'] != item['original_attempt']:
                    raise ValueError('Certificate does not authorize this exact interrupted attempt')
            admin_attempts.add(item['original_attempt']['path'])
        cursor = Path(parent['prior']).resolve()

    expected_inventory, expected_errors = list(sealed['records']), list(sealed['execution_errors'])
    worker_paths = sorted((job / 'workers').glob('gpu_*/status.json'))
    if {p.parent.name for p in worker_paths} != {f'gpu_{i}' for i in range(8)}:
        raise ValueError('Exactly eight final workers required')
    for path in worker_paths:
        gpu = int(path.parent.name[4:])
        rec = recovered.get(gpu)
        errors, pins = worker_materials(path, original, quarantined=rec['certificate'] if rec else None)
        expected_errors.extend(errors); evidence.extend(pins)
    for part in sorted((job / 'parts').iterdir()):
        c.shared.require_no_uncommitted_metrics(part)
        expected_inventory.extend(c.shared.original_record(p, r) for p, r in c.completed_metrics(part))
    if recovered:
        # The untouched parent correctly ended incomplete. Do not invent a
        # completed parent status or write new inventory files into its run.
        partial = c.shared.summary_from_rows(m['tasks'], [c.read(r['metrics']['path']) for r in expected_inventory], expected_errors)
        if any(saved.get(k) != v for k, v in partial.items()):
            raise ValueError('Parent partial summary differs from original bytes')
        inventory, errors = list(expected_inventory), list(expected_errors)
        for gpu, rec in recovered.items():
            run, rm = rec['job'], rec['manifest']; lineage.append(run)
            if {p.parent.name for p in (run / 'workers').glob('gpu_*/status.json')} != {f'gpu_{gpu}'}:
                raise ValueError('Foreign worker in isolated recovery')
            extra_errors, pins = worker_materials(run / 'workers' / f'gpu_{gpu}' / 'status.json', original, recovery=rec)
            errors.extend(extra_errors); evidence.extend(pins)
            evidence.extend([c.binding(run / 'manifest.json'), c.binding(run / 'status.json'), rm['certificate']])
            for part in sorted((run / 'parts').iterdir()):
                c.shared.require_no_uncommitted_metrics(part)
                inventory.extend(c.shared.original_record(p, r) for p, r in c.completed_metrics(part))
    else:
        inventory, errors = c.read(job / 'inventory.json'), c.read(job / 'execution-errors.json')
        if inventory != expected_inventory or errors != expected_errors:
            raise ValueError('Final inventory/error ledger omitted or changed an original')

    originals, records, selected = [], [], {}
    for item in inventory:
        metric = Path(item['metrics']['path']).resolve(); part = metric.parent.parent
        if part.parent.name != 'parts' or part.parent.parent not in lineage or metric in selected:
            raise ValueError('Duplicate original or result outside continuation lineage')
        row = c.read(metric)
        if item != c.shared.original_record(metric, row):
            raise ValueError('Original metric/video/physical receipt changed')
        selected[metric] = row; records.append(row); originals.append((metric.name, metric.read_bytes()))

    commands, actual, running_attempts = [], {}, {}
    for run in lineage:
        rm = c.read(run / 'manifest.json'); parts = c.checked_parts(run, rm, original)
        # The final run must have visited every planned/tail part. Ancestors
        # may contain never-started parts, explicitly recovered by this run.
        transferred = {p['id'] for rec in recovered.values() for p in rec['certificate']['pending']} if run == job else set()
        final_run = run == job or run in {rec['job'] for rec in recovered.values()}
        if final_run and {p.name for p in (run / 'parts').iterdir()} != set(parts) - transferred:
            raise ValueError('Unvisited or foreign final physical part')
        expected_command_paths = set()
        for part in sorted((run / 'parts').iterdir()):
            if part.name not in parts:
                raise ValueError('Unregistered physical part')
            gpu, plan = parts[part.name]; ps = c.read(part / 'status.json')
            if (any(ps.get(k) != plan[k] for k in ('task', 'task_id', 'indices'))
                    or ps.get('source_commit') != original['source_commit']
                    or ps.get('config_sha256') != original['configs'][gpu]['sha256']
                    or ps.get('optimizer_updates') != 0):
                raise ValueError('Physical part identity changed')
            cmd = run / 'workers' / f'gpu_{gpu}' / (part.name + '-command.json')
            expected_command_paths.add(cmd); invocation = c.read(cmd)
            expected_argv = ['bash', str(source / 'scripts/eval/memlite_sft100/launch_sim.sh'),
                str(source / 'scripts/eval/run_champion_task.py'), '--task', plan['task'],
                '--task-id', str(plan['task_id']), '--policy-config', original['configs'][gpu]['path'],
                '--output', str(part), '--num-envs', str(plan['num_envs']), '--indices', *map(str, plan['indices'])]
            if invocation != dict(argv=expected_argv, official_commit=original['official_commit']):
                raise ValueError('Original evaluator invocation differs')
            commands.append(c.binding(cmd) | dict(gpu=str(gpu), argv=expected_argv))
            c.shared.require_no_uncommitted_metrics(part)
            attempts = [(a, c.read(a)) for a in sorted((part / 'attempts').glob('*.json'))]
            tail = c.shared.unstarted_tail(plan, attempts)
            if final_run and tail is not None and tail['id'] not in parts:
                raise ValueError('Unvisited final physical suffix')
            for a, row in attempts:
                if (row.get('policy_seed') != original['policy_seed']
                        or row.get('simulator_seed') != original['simulator_seed']
                        or row.get('max_steps') != original['nominal_horizons'][plan['task']]):
                    raise ValueError('Original seed/horizon changed')
                if row['status'] != 'completed':
                    if row['status'] != 'running':
                        raise ValueError('Unknown physical attempt state')
                    running_attempts[str(a)] = (plan, row)
            for metric, row in c.completed_metrics(part):
                actual[metric.resolve()] = row
        if set((run / 'workers').glob('gpu_*/*-command.json')) != expected_command_paths:
            raise ValueError('Omitted original evaluator invocation')
    if actual != selected:
        raise ValueError('Inventory omitted or changed committed physical results')
    by_attempt = {}
    for error in errors:
        for field in ('original_attempt', 'log'):
            c.check_binding(error[field])
        attempt = error['original_attempt']['path']; plan, row = running_attempts[attempt]
        log = Path(error['log']['path'])
        operational = error['diagnosis']['kind'] == 'unclassified_operational_batch_error'
        if operational:
            run_manifest = c.read(Path(attempt).parent.parent.parent.parent / 'manifest.json')
            if run_manifest.get('continue_on_batch_errors') is not True:
                raise ValueError('Operational continuation was not registered')
            evidence.append(c.validate_operational_error(error['diagnosis'], log, attempt, plan))
        expected_logs = [p for p in (Path(attempt).parent.parent.parent.parent / 'workers').glob(
            'gpu_*/' + Path(attempt).parent.parent.name + '.log')]
        if (expected_logs != [log] or error['task'] != plan['task'] or row['rollout'] != error['rollout_id']
                or error['instance_id'] not in row['instance_ids']
                or (not operational and c.execution_error(log.read_text()) != error['diagnosis'])):
            raise ValueError('Execution error does not match the original physical attempt/log')
        by_attempt.setdefault(attempt, []).append(error['instance_id'])
    for attempt, ids in by_attempt.items():
        if sorted(ids) != sorted(running_attempts[attempt][1]['instance_ids']):
            raise ValueError('Failed physical batch was only partly disclosed')
    if set(running_attempts) != set(by_attempt) | admin_attempts or set(by_attempt) & admin_attempts:
        raise ValueError('Unclassified/overlapping incomplete physical attempt')
    summary = accounted_summary(m['tasks'], records, errors)
    if not recovered and any(saved.get(k) != v for k, v in summary.items()):
        raise ValueError('Final summary differs from original bytes')
    if (saved.get('original_outputs_unmodified') is not True or saved.get('policy_changed') is not False
            or saved.get('optimizer_updates') != 0):
        raise ValueError('Missing immutable parent summary provenance')
    for name in (('status.json', 'summary.json') if recovered else ('status.json', 'summary.json', 'inventory.json', 'execution-errors.json')):
        evidence.append(c.binding(job / name))

    wrapper = source / 'scripts/eval/memlite_sft100/rgb_wrapper.py'
    robot = root / 'submission/r1pro.yaml'
    pin = c.read(source / 'scripts/eval/memlite_sft100/official_manifest.json')
    if ((root / 'submission/rgb_wrapper.py').read_bytes() != wrapper.read_bytes()
            or c.sha256(robot) != pin['files']['omnigibson/eval/r1pro.yaml']):
        raise ValueError('Original evaluation wrapper/robot changed')
    files = {'rgb_wrapper.py': wrapper.read_bytes(), 'r1pro.yaml': robot.read_bytes()}
    for name, relative in [('run_champion_task.py', 'run_champion_task.py'),
            ('champion_batched_client.py', 'champion_batched_client.py'),
            ('wire.py', 'memlite_sft100/wire.py')]:
        files['transport/' + name] = (source / 'scripts/eval' / relative).read_bytes()
    sidecars = {'original-run-manifest.json': (root / 'manifest.json').read_bytes()}
    sidecars.update({f'policy-config-gpu-{i}.json': Path(cfg['path']).read_bytes() for i, cfg in enumerate(m['configs'])})
    for gpu, rec in recovered.items():
        sidecars[f'recovery-gpu-{gpu}-manifest.json'] = (rec['job'] / 'manifest.json').read_bytes()
        sidecars[f'recovery-gpu-{gpu}-certificate.json'] = Path(rec['manifest']['certificate']['path']).read_bytes()
    return dict(originals=originals, inventory=inventory, errors=errors, files=files, evidence=evidence,
                commands=commands, lineage=lineage, summary=summary, manifest=m, root_manifest=root_pin, sidecars=sidecars)


def package(job, manifest_sha256, coordinator_source, output, *, recoveries=()):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError('New output directory required')
    data = audited_materials(job, manifest_sha256, coordinator_source, recoveries=recoveries)
    if any(output.resolve().is_relative_to(p) for p in data['lineage']):
        raise ValueError('Collection must be outside original evaluation directories')
    s, m = data['summary'], data['manifest']
    names = [name for name, _ in data['originals']]
    if len(names) != len(set(names)) or len(names) != s['completed']:
        raise ValueError('Duplicate/missing original filenames')
    readme = (f'# Champion internal result collection — not submitted\n\n'
        f'{s["completed"]} original official results; {s["execution_errors"]} missing results from diagnosed '
        f'execution errors; all 1000 prescribed cases accounted for.\n'
        f'Fixed-1000 Q (missing as zero): {s["conservative_q_errors_and_missing_as_zero"]}; '
        f'SR: {s["conservative_sr_errors_and_missing_as_zero"]}. Not 1000 valid official outcomes.\n'
        'No missing JSON fabricated, no original metric/video edited, no outcome-based rerolls.\n'
        f'Actor: {m["runtime_commit"]}; terminal export SHA256: {m["export_sha256"]}.\n'
        'Wrapper: rgb_wrapper.RGBOnlyFullResWrapper; robot: r1pro.yaml.\n'
        'Original MP4 paths/SHA and diagnosed missing cases are in adjacent inventory/error sidecars.\n'
        'Historical administrative interruptions and partial videos remain in the unchanged lineage.\n'
        f'Explicit isolated-worker recoveries: {len(recoveries)}; their original manifests/certificates are adjacent.\n'
        'Model delivery, video hosting and submission are deferred. This is not a standalone serving image.\n\n'
        '## Historical evaluator commands — do not reroll for score selection\n\n```bash\n'
        + '\n'.join('EVAL_GPU=' + r['gpu'] + ' ' + shlex.join(r['argv']) for r in data['commands']) + '\n```\n')
    files = data['files'] | {'README.md': readme.encode()}
    output.mkdir(); target = output / 'challenge_results_draft.zip'
    partial = output / 'challenge_results_draft.zip.partial'
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in data['originals']:
            archive.writestr('json/' + name, raw)
        for name, raw in files.items():
            archive.writestr(name, raw)
    with zipfile.ZipFile(partial) as archive:
        if (archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist()))
                or {n for n in archive.namelist() if n.endswith('.json')} != {'json/' + n for n in names}
                or any(archive.read('json/' + n) != raw for n, raw in data['originals'])):
            raise ValueError('Archive round-trip mismatch')
    partial.rename(target)
    receipt = dict(schema='champion_continuation_internal_collection_v1', created=time.time(), job=str(Path(job).resolve()),
        output=str(target), sha256=c.sha256(target), manifest_sha256=manifest_sha256, root_manifest=data['root_manifest'],
        metrics=len(names), missing=s['execution_errors'], expected=1000, fabricated_metrics=0,
        originals_modified=False, submitted=False, videos_uploaded=False, model_delivery_ready=False,
        evidence=data['evidence'], commands=data['commands'])
    receipt['recoveries'] = [dict(job=str(Path(p).resolve()), source=str(Path(s).resolve()),
                                 manifest=c.binding(Path(p) / 'manifest.json')) for p, s in recoveries]
    for name, value in [('receipt.json', receipt), ('video-inventory.json', data['inventory']),
            ('execution-errors.json', data['errors']), ('summary.json', s), ('run-manifest.json', m)]:
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
    for name, raw in data['sidecars'].items():
        with (output / name).open('xb') as stream:
            stream.write(raw)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'coordinator-source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--recovery', type=Path, nargs=2, action='append', default=[], metavar=('JOB', 'SOURCE'))
    a = p.parse_args()
    print(json.dumps(package(a.job, a.manifest_sha256, a.coordinator_source, a.output, recoveries=a.recovery), indent=2))
