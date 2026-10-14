"""Read-only result draft after all 1000 cases are accounted for.

Official rules allow missing results, scored as zero. This does NOT authorize
early stopping, rerolls, synthetic JSON, external upload, or model delivery.
Use from a separate checkout; the running evaluation source remains frozen.
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
import continue_memlite_final as c


def accounted_summary(tasks, records, errors):
    result = c.summary_from_rows(tasks, records, errors)
    if result['unaccounted'] or not records:
        raise ValueError('All 1000 cases must be accounted for, with original valid results')
    for error in errors:
        if (error.get('rerun_allowed') is not False
                or error['diagnosis'].get('official_score_available') is not False
                or error['diagnosis']['kind'] not in
                ('simulator_nonfinite_robot_pose', 'planner_memory_recurrence_rejected')):
            raise ValueError('Only explicitly diagnosed original execution errors may be missing')
    return result


def audited_materials(job, manifest_sha):
    job = Path(job).resolve()
    if c.sha256(job / 'manifest.json') != manifest_sha:
        raise ValueError('Changed registration')
    m, state = c.read(job / 'manifest.json'), c.read(job / 'status.json')
    if (m['schema'] != 'memlite_final_explicit_continuation_v1'
            or state['status'] not in ('completed', 'completed_with_execution_errors')
            or not state.get('finished') or state.get('failed_gpus')
            or set(state['completed_gpus']) != set(range(8))):
        raise ValueError('Continuation is not terminal; do not make an early partial draft')
    c.require_exited([state['pid'], *state['worker_pids'].values()])
    root, original, root_pin = c.root_registration(job)
    source = Path(m['runtime_source'])
    c.frozen_source(source, original['source_commit'])
    c.check_binding(dict(path=m['config_path'], sha256=m['config_sha256']))
    sealed = c.read(m['seal']['path'])
    for item in sealed['evidence']:
        c.check_binding(item)
    lineage, cursor = [], job
    while True:
        lineage.append(cursor)
        parent = c.read(cursor / 'manifest.json')
        if cursor == root:
            break
        cursor = Path(parent['prior']).resolve()

    expected_inventory = list(sealed['records'])
    expected_errors = list(sealed['execution_errors'])
    worker_paths = sorted((job / 'workers').glob('gpu_*/status.json'))
    if {p.parent.name for p in worker_paths} != {f'gpu_{i}' for i in range(8)}:
        raise ValueError('Exactly eight terminal workers required')
    for path in worker_paths:
        w = c.read(path)
        if (w['status'] not in ('completed', 'completed_with_execution_errors')
                or not w.get('finished') or w.get('final_verification_error')):
            raise ValueError('Worker/model terminal verification missing')
        c.require_exited([w['pid'], w.get('model_pid'), w.get('sim_pid')])
        finals = [c.binding(p / 'weights-final.json') for p in sorted(path.parent.glob('model-*')) if p.is_dir()]
        if finals != w['model_generations']:
            raise ValueError('Missing model-generation verification')
        for pinned in finals:
            if not c.exact_frozen_weights(c.read(pinned['path'])):
                raise ValueError('Model or observer weights changed')
            ms = c.read(Path(pinned['path']).parent / 'status.json')
            if (ms.get('source_commit') != m['runtime_commit']
                    or ms.get('config_sha256') != m['config_sha256']
                    or ms.get('manifest_sha256') != m['pack_manifest_sha256']):
                raise ValueError('Terminal model is not the registered policy')
        for error_path in sorted(path.parent.glob('*-execution-errors.json')):
            expected_errors.extend(c.read(error_path))
    for part in sorted((job / 'parts').iterdir()):
        c.require_no_uncommitted_metrics(part)
        expected_inventory.extend(c.original_record(path, row) for path, row in c.completed_metrics(part))
    inventory, errors = c.read(job / 'inventory.json'), c.read(job / 'execution-errors.json')
    if inventory != expected_inventory or errors != expected_errors:
        raise ValueError('Final inventory/error ledger omits or changes an original')

    records, originals, commands, visited_parts = [], [], [], set()
    for item in inventory:
        metric = Path(item['metrics']['path']).resolve()
        part = metric.parent.parent
        if part.parent.name != 'parts' or part.parent.parent not in lineage:
            raise ValueError('Result outside the registered continuation lineage')
        row = c.read(metric)
        # Repeat the actual video/physical receipt check, including inherited
        # originals. Never trust only a previous green count or cached probe.
        if item != c.original_record(metric, row):
            raise ValueError('Original metric/video/clock changed')
        originals.append((metric.name, metric.read_bytes()))
        records.append(row)
    # Audit every issued part, not just those with positive/valid scores.
    for run in lineage:
        for part in sorted((run / 'parts').iterdir()):
            ps = c.read(part / 'status.json')
            if (ps['source_commit'] != original['source_commit']
                    or ps['official_commit'] != original['official_commit']
                    or ps['config_sha256'] != original['config_sha256']):
                raise ValueError('Part has foreign model/config/evaluator')
            paths = list((run / 'workers').glob('gpu_*/' + part.name + '-command.json'))
            if len(paths) != 1:
                raise ValueError('Missing/ambiguous original evaluator invocation')
            command = c.read(paths[0]); argv = command['argv']
            def one(flag):
                if argv.count(flag) != 1:
                    raise ValueError('Missing/duplicate evaluator flag')
                return argv[argv.index(flag) + 1]
            if (argv[:3] != ['bash', str(source / 'scripts/eval/memlite_sft100/launch_sim.sh'),
                                         str(source / 'scripts/eval/memlite_final/run_task.py')]
                    or Path(one('--output')).resolve() != part or one('--task') != ps['task']
                    or one('--config-sha256') != original['config_sha256']
                    or int(one('--task-id')) != ps['task_id']
                    or int(one('--num-envs')) != ps['num_envs']
                    or list(map(int, argv[argv.index('--indices') + 1:])) != ps['indices']):
                raise ValueError('Original command and physical part differ')
            c.require_no_uncommitted_metrics(part)
            committed = {c.key(row): (path.resolve(), row) for path, row in c.completed_metrics(part)}
            for item in inventory:
                path = Path(item['metrics']['path']).resolve()
                if path.parent.parent == part:
                    actual = committed.pop(c.key(item), None)
                    if actual is None or actual[0] != path or actual[1] != c.read(path):
                        raise ValueError('Inventory does not match committed physical ACK')
            if committed:
                raise ValueError('Unselected completed original result')
            commands.append(c.binding(paths[0]) | dict(argv=argv, gpu=paths[0].parent.name[4:]))
            visited_parts.add(part)
    for error in errors:
        for field in ('original_attempt', 'log'):
            c.check_binding(error[field])
        a = Path(error['original_attempt']['path']).resolve()
        attempt = c.read(a)
        if (a.parent.parent not in visited_parts or attempt['status'] != 'running'
                or c.read(a.parent.parent / 'status.json')['task'] != error['task']
                or error['instance_id'] not in attempt['instance_ids'] or attempt['rollout'] != error['rollout_id']):
            raise ValueError('Missing result has no matching original failed attempt')
    summary = accounted_summary(m['tasks'], records, errors)
    saved = c.read(job / 'summary.json')
    if any(saved.get(k) != v for k, v in summary.items()):
        raise ValueError('Summary differs from original bytes and fixed-1000 denominator')
    if state['status'] != summary['status']:
        raise ValueError('Supervisor and final result status differ')

    wrapper = source / 'scripts/eval/memlite_sft100/rgb_wrapper.py'
    pin = c.read(source / 'scripts/eval/memlite_sft100/official_manifest.json')
    robot = root / 'submission/r1pro.yaml'
    if ((root / 'submission/rgb_wrapper.py').read_bytes() != wrapper.read_bytes()
            or c.sha256(robot) != pin['files']['omnigibson/eval/r1pro.yaml']):
        raise ValueError('Exact evaluation wrapper/robot differs')
    files = {'rgb_wrapper.py': wrapper.read_bytes(), 'r1pro.yaml': robot.read_bytes()}
    for name in ('client.py', 'lifecycle.py', 'run_task.py'):
        files['transport/' + name] = (source / 'scripts/eval/memlite_final' / name).read_bytes()
    for name, relative in (('full_episode_client.py', 'scripts/eval/full_episode_client.py'),
                           ('wire.py', 'scripts/eval/memlite_sft100/wire.py')):
        files['transport/' + name] = (source / relative).read_bytes()
    return dict(originals=originals, files=files, inventory=inventory, errors=errors,
                commands=commands, summary=summary, manifest=m, root_manifest=root_pin,
                lineage=lineage, sidecars={
                    'run-manifest.json': (job / 'manifest.json').read_bytes(),
                    'original-run-manifest.json': (root / 'manifest.json').read_bytes(),
                    'policy-config.json': Path(m['config_path']).read_bytes(),
                })


def package(job, manifest_sha256, output):
    job, output = Path(job).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError('New output directory required')
    data = audited_materials(job, manifest_sha256)
    if any(output.resolve().is_relative_to(p) for p in data['lineage']):
        raise ValueError('Draft must be outside all original evaluation directories')
    s, m = data['summary'], data['manifest']
    readme = (f'# MEM-Lite result draft — not uploaded or model-delivery ready\n\n'
        f'All 1000 prescribed cases accounted for: {s["completed"]} original results; '
        f'{s["execution_errors"]} missing official results due to diagnosed execution errors.\n'
        f'Fixed-denominator Q (missing as zero): {s["conservative_q_errors_and_missing_as_zero"]}; '
        f'SR: {s["conservative_sr_errors_and_missing_as_zero"]}. These are not 1000 valid official outcomes.\n'
        'No missing JSON was fabricated; no score/MP4 was edited or selected from rerolls.\n'
        'Official partial-result rule: https://behavior.stanford.edu/challenge/submission.html\n'
        f'Actor commit: {m["runtime_commit"]}; policy config SHA: {m["config_sha256"]}; '
        f'model pack SHA: {m["pack_manifest_sha256"]}.\n'
        'Wrapper: rgb_wrapper.RGBOnlyFullResWrapper (rgb_wrapper.py); robot: r1pro.yaml.\n'
        'Use the frozen actor repository and its acknowledged transport; this draft is not a standalone serving image.\n'
        'Original videos remain at paths/SHA in adjacent video-inventory.json; no hosted link yet.\n'
        'Adjacent execution-errors.json records every missing result. Historical administrative interruptions '
        'and their original partial videos remain in the unchanged lineage directories.\n'
        'Docker single-24GB or 50 public ports, video hosting, team declarations and external submission are deferred.\n\n'
        '## Historical original evaluator invocations — do not rerun for score selection\n\n```bash\n'
        + '\n'.join('EVAL_GPU=' + row['gpu'] + ' ' + shlex.join(row['argv']) for row in data['commands'])
        + '\n```\n')
    files = data['files'] | {'README.md': readme.encode()}
    names = [name for name, _ in data['originals']]
    if len(names) != len(set(names)) or len(names) != s['completed']:
        raise ValueError('Duplicate or missing original result names')
    output.mkdir()
    target = output / 'challenge_results_draft.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in data['originals']:
            z.writestr('json/' + name, raw)
        for name, raw in files.items():
            z.writestr(name, raw)
    with zipfile.ZipFile(target) as z:
        if (z.testzip() is not None
                or {n for n in z.namelist() if n.endswith('.json')} != {'json/' + n for n in names}
                or any(z.read('json/' + n) != raw for n, raw in data['originals'])):
            raise ValueError('Draft round-trip failed')
    receipt = dict(schema='memlite_continuation_result_draft_v1', created=time.time(),
        status='result_draft_not_model_delivery_or_submission', job=str(job), output=str(target),
        sha256=c.sha256(target), manifest_sha256=manifest_sha256, root_manifest=data['root_manifest'],
        metrics=len(names), expected=1000, missing=s['execution_errors'],
        submission_q_missing_as_zero=s['conservative_q_errors_and_missing_as_zero'],
        submission_sr_missing_as_zero=s['conservative_sr_errors_and_missing_as_zero'],
        originals_modified=False, fabricated_metrics=0, submitted=False, videos_uploaded=False,
        model_delivery_ready=False, commands=data['commands'])
    for name, value in (('receipt.json', receipt), ('video-inventory.json', data['inventory']),
                        ('execution-errors.json', data['errors']), ('summary.json', s)):
        with (output / name).open('x') as f:
            json.dump(value, f, indent=2); f.write('\n')
    for name, raw in data.get('sidecars', {}).items():
        with (output / name).open('xb') as f:
            f.write(raw)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(package(a.job, a.manifest_sha256, a.output), indent=2))
