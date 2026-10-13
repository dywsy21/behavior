"""Create a portal-only draft from a completed, sealed final-policy evaluation.

Run from a separate checkout. Never import a model, use a GPU, alter the active
evaluator, edit original outputs, upload anything, or declare deployment ready.
The larger evaluation_outputs.zip remains the internal reproducibility bundle.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'memlite_sft100'), str(HERE)]
from common import aggregate, expected_cases, sha256
from completed_metrics import completed_metrics


def read_json(path):
    return json.loads(Path(path).read_text())


def inside_file(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('File outside registered run or missing: ' + str(path))
    return path


def command_inventory(job, manifest):
    """Require exact one-pass public coverage; never silently omit a command."""
    records, seen = [], Counter()
    expected = expected_cases(manifest['tasks'])
    task_ids = {r['task']: r['task_id'] for r in manifest['cases']}
    for path in sorted((job / 'workers').glob('gpu_*/*-command.json')):
        row = read_json(path)
        args = row['argv']
        if (row['official_commit'] != manifest['official_commit']
                or not isinstance(args, list) or not all(isinstance(a, str) for a in args)
                or any('smoke' in a for a in args)):
            raise ValueError('Foreign or non-public evaluator command')
        def one(flag):
            if args.count(flag) != 1:
                raise ValueError('Missing/repeated command flag: ' + flag)
            return args[args.index(flag) + 1]
        task = one('--task')
        if task not in task_ids or int(one('--task-id')) != task_ids[task]:
            raise ValueError('Command task name/ID mismatch')
        if int(one('--num-envs')) not in (2, 4) or args.count('--indices') != 1:
            raise ValueError('Unregistered evaluation batch')
        ids = args[args.index('--indices') + 1:]
        if not ids or any(not i.isdecimal() for i in ids):
            raise ValueError('Missing or unexpected public indices')
        part = Path(one('--output')).resolve()
        if part.parent != (job / 'parts').resolve() or not part.is_dir():
            raise ValueError('Command output outside this run')
        for i in ids:
            key = task, int(i) + 301, 0
            if key not in expected:
                raise ValueError('Unexpected public instance')
            seen[key] += 1
        records.append(dict(path=str(path), sha256=sha256(path), argv=args,
                            gpu=path.parent.name.removeprefix('gpu_')))
    if set(seen) != expected or any(n != 1 for n in seen.values()):
        raise ValueError('Evaluator commands have missing/repeated cases')
    return records


def package(job, manifest_sha256, output):
    job, output = Path(job).resolve(), Path(output).absolute()
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or output.resolve().is_relative_to(job / 'parts')):
        raise ValueError('Need a new output directory outside original trajectories')
    if sha256(job / 'manifest.json') != manifest_sha256:
        raise ValueError('Run registration changed')
    manifest = read_json(job / 'manifest.json')
    if manifest['schema'] not in ('memlite_final_public100_run_v1', 'champion_final_public100_run_v1'):
        raise ValueError('Only registered final-policy evaluations are supported')
    if (read_json(job / 'status.json')['status'] != 'completed'
            or read_json(job / 'summary.json')['status'] != 'complete'):
        raise ValueError('Full evaluation is not completed; no draft was produced')
    expected = expected_cases(manifest['tasks'])
    cases = [(c['task'], c['instance_id'], c['rollout_id']) for c in manifest['cases']]
    if len(cases) != 1000 or set(cases) != expected:
        raise ValueError('Registration must prescribe all 1000 cases exactly once')
    if manifest['optimizer_updates'] != 0 or manifest['automatic_retries'] is not False:
        raise ValueError('Unregistered training/retry semantics')
    summary = read_json(job / 'summary.json')
    sealed = read_json(job / 'submission/package-status.json')
    source_zip = inside_file(sealed['zip_path'], job / 'submission')
    if (sealed['status'] != 'evaluation_outputs_only_not_final_model_delivery'
            or sealed['metrics'] != 1000 or sealed['videos'] != 1000
            or sha256(source_zip) != sealed['zip_sha256']):
        raise ValueError('Original internal archive is not intact and complete')
    physical = {}
    for part in sorted((job / 'parts').iterdir()):
        for path, row in completed_metrics(part, final=True):
            path = path.resolve()
            if path in physical:
                raise ValueError('Duplicate committed physical metric')
            physical[path] = row
    commands = command_inventory(job, manifest)
    inventory = read_json(job / 'inventory.json')
    originals, records, metric_paths, names = [], [], set(), set()
    with zipfile.ZipFile(source_zip) as source:
        if source.testzip() is not None or len(source.namelist()) != len(set(source.namelist())):
            raise ValueError('Corrupted/ambiguous internal archive')
        manifest_member = 'run-manifest.json' if manifest['schema'].startswith('memlite_') else 'manifest.json'
        for member, original in ((manifest_member, 'manifest.json'), ('inventory.json', 'inventory.json'),
                                 ('summary.json', 'summary.json')):
            if source.read(member) != (job / original).read_bytes():
                raise ValueError('Sealed provenance changed: ' + original)
        for item in inventory:
            metric = inside_file(item['metrics'], job / 'parts')
            video = inside_file(item['video'], job / 'parts')
            raw = metric.read_bytes()
            row = json.loads(raw)
            name = f"{row['task']}_{row['instance_id']}_{row['rollout_id']}.json"
            if (name in names or metric.name != name or video.stem != metric.stem
                    or metric not in physical or physical[metric] != row
                    or any(item[k] != row[k] for k in ('task', 'instance_id', 'rollout_id'))
                    or hashlib.sha256(raw).hexdigest() != item['metrics_sha256']
                    or video.stat().st_size != item['video_bytes'] or sha256(video) != item['video_sha256']
                    or int(item['video_probe']['nb_frames']) != row['steps']
                    or item['video_probe']['r_frame_rate'] != '30/1'
                    or source.read('json/' + name) != raw):
                raise ValueError('Original metric/video/physical receipt differs: ' + name)
            originals.append((name, raw)); records.append(row); names.add(name); metric_paths.add(metric)
        if metric_paths != set(physical):
            raise ValueError('Inventory omitted committed physical results')
        result = aggregate(manifest['tasks'], records)
        if result['status'] != 'complete':
            raise ValueError('Full original 1000-case inventory required')
        for key in ('official_q_score', 'official_sr', 'completed', 'missing'):
            if result[key] != summary[key]:
                raise ValueError('Original results disagree with summary: ' + key)
        files = {name: source.read(name) for name in ('rgb_wrapper.py', 'r1pro.yaml', 'README.md')}
        for name, content in files.items():
            if content != (job / 'submission' / name).read_bytes():
                raise ValueError('Original wrapper/robot/README changed')
        # Preserve the exact transport code from the frozen evaluation archive,
        # not a possibly newer local checkout. Do not include metadata JSON.
        for name in source.namelist():
            if name.startswith('transport/') and name.endswith('.py'):
                if '..' in Path(name).parts or Path(name).is_absolute():
                    raise ValueError('Unsafe source archive member')
                files[name] = source.read(name)
    prefix = ('# Portal-only draft — not uploaded or deployment-ready\n\n'
              'Every JSON in this ZIP is an unmodified official rollout result.\n'
              'The separately retained evaluation_outputs.zip contains model/config provenance.\n'
              'Original MP4 videos remain separate; a hosted link still needs authorization.\n'
              'Still required: actual single-24GB Docker acceptance OR 50 reachable policy ports;\n'
              'team/contact/disclosure confirmation; external upload and final submission.\n\n'
              f'Registered run manifest SHA256: {manifest_sha256}\n'
              f'Frozen evaluation source: {manifest["source_commit"]}\n\n'
              '## Exact original evaluator invocations\n\n'
              'These are historical commands, not instructions to reroll these cases.\n'
              'The absolute paths refer to the original host. GPU assignment is shown explicitly;\n'
              'the frozen worker sets its remaining environment.\n\n```bash\n')
    command_text = '\n'.join(f"EVAL_GPU={r['gpu']} " + shlex.join(r['argv']) for r in commands)
    files['README.md'] = (prefix + command_text + '\n```\n\n' + files['README.md'].decode()).encode()
    # Nothing is created before all original files, commands and scores pass.
    output.mkdir(exist_ok=False)
    target = output / 'challenge_results_draft.zip'
    partial = output / 'challenge_results_draft.zip.partial'
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in originals:
            archive.writestr('json/' + name, raw)
        for name, raw in files.items():
            archive.writestr(name, raw)
    with zipfile.ZipFile(partial) as archive:
        if (archive.testzip() is not None
                or {n for n in archive.namelist() if n.endswith('.json')} != {'json/' + n for n in names}
                or any(archive.read('json/' + n) != raw for n, raw in originals)):
            raise ValueError('Draft ZIP failed round-trip integrity check')
    partial.rename(target)
    receipt = dict(schema='final_public_portal_draft_v1', created=time.time(),
        status='draft_materials_not_ready_for_submission', job=str(job),
        manifest_sha256=manifest_sha256, source_archive_sha256=sealed['zip_sha256'],
        source_commit=manifest['source_commit'], output=str(target), sha256=sha256(target),
        metrics=1000, videos=1000, video_inventory_sha256=sha256(job / 'inventory.json'),
        official_q_score=result['official_q_score'], official_sr=result['official_sr'],
        every_zip_json_is_original_rollout=True, original_videos_rehashed=True,
        exact_evaluation_commands=commands, originals_modified=False,
        videos_uploaded=False, docker_24gb_validated=False, hosted_50_ports_validated=False,
        team_declarations_confirmed=False, submitted=False)
    with (output / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.job, args.manifest_sha256, args.output), indent=2))
