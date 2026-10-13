import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/eval/package_final_public_eval.py'
spec = importlib.util.spec_from_file_location('final_portal_package', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def prepared(tmp_path, schema='memlite_final_public100_run_v1'):
    """Synthetic 1000-case packaging fixture, not a video/physics acceptance."""
    job = tmp_path / 'run'
    tasks = [f'task_{i}' for i in range(100)]
    manifest = dict(schema=schema, tasks=tasks, source_commit='frozen', official_commit='official',
                    optimizer_updates=0, automatic_retries=False, cases=[])
    records, inventory = [], []
    for i, task in enumerate(tasks):
        part = job / 'parts' / task
        save(part / 'status.json', dict(status='completed', task=task))
        save(part / 'attempts/00.json', dict(status='completed', instance_ids=list(range(301, 311)),
                                             rollout=0, actual_applied_controls=[2] * 10))
        command = dict(official_commit='official', argv=[
            'bash', '/frozen/launch_sim.sh', '/frozen/run_task.py', '--task', task,
            '--task-id', str(i), '--output', str(part), '--num-envs', '2',
            '--indices', *map(str, range(10))])
        save(job / 'workers/gpu_0' / (task + '-command.json'), command)
        for instance in range(301, 311):
            row = dict(task=task, instance_id=instance, rollout_id=0, steps=2,
                       success=False, q_score=dict(final=.25 if instance == 301 else 0.))
            name = f'{task}_{instance}_0'
            metric, video = part / 'json' / (name + '.json'), part / 'videos' / (name + '.mp4')
            save(metric, row)
            video.parent.mkdir(exist_ok=True)
            video.write_bytes(b'synthetic-video-fixture-not-actual-mp4')
            inventory.append(dict(task=task, instance_id=instance, rollout_id=0,
                metrics=str(metric), video=str(video), metrics_sha256=m.sha256(metric),
                video_sha256=m.sha256(video), video_bytes=video.stat().st_size,
                video_probe=dict(nb_frames='2', r_frame_rate='30/1')))
            records.append(row)
            manifest['cases'].append(dict(task_id=i, task=task, instance_id=instance, rollout_id=0))
    save(job / 'manifest.json', manifest)
    save(job / 'status.json', dict(status='completed'))
    save(job / 'summary.json', m.aggregate(tasks, records))
    save(job / 'inventory.json', inventory)
    dest = job / 'submission'
    dest.mkdir()
    for name in ('README.md', 'rgb_wrapper.py', 'r1pro.yaml'):
        (dest / name).write_text('original ' + name)
    internal = dest / 'evaluation_outputs.zip'
    with zipfile.ZipFile(internal, 'x') as archive:
        for item in inventory:
            f = Path(item['metrics'])
            archive.write(f, 'json/' + f.name)
        for name in ('README.md', 'rgb_wrapper.py', 'r1pro.yaml'):
            archive.write(dest / name, name)
        archive.writestr('transport/exact_runtime.py', '# original transport')
        archive.write(job / 'summary.json', 'summary.json')
        archive.write(job / 'inventory.json', 'inventory.json')
        archive.write(job / 'manifest.json', 'run-manifest.json' if schema.startswith('memlite_') else 'manifest.json')
    save(dest / 'package-status.json', dict(status='evaluation_outputs_only_not_final_model_delivery',
        metrics=1000, videos=1000, zip_path=str(internal), zip_sha256=m.sha256(internal)))
    return job, m.sha256(job / 'manifest.json'), tmp_path / 'portal-draft'


@pytest.mark.parametrize('schema', ['memlite_final_public100_run_v1', 'champion_final_public100_run_v1'])
def test_portal_has_only_original_metrics_json_no_metadata(tmp_path, schema):
    job, manifest_sha, output = prepared(tmp_path, schema)
    original = job / 'parts/task_0/json/task_0_301_0.json'
    before = original.read_bytes()
    receipt = m.package(job, manifest_sha, output)
    with zipfile.ZipFile(receipt['output']) as archive:
        assert len([n for n in archive.namelist() if n.endswith('.json')]) == 1000
        assert 'summary.json' not in archive.namelist()
        assert archive.read('json/task_0_301_0.json') == before
        assert archive.read('transport/exact_runtime.py') == b'# original transport'
        assert b'EVAL_GPU=0 bash /frozen/launch_sim.sh' in archive.read('README.md')
    assert original.read_bytes() == before
    assert receipt['official_q_score'] == .025
    assert receipt['submitted'] is False and receipt['docker_24gb_validated'] is False
    with pytest.raises(ValueError, match='new output'):
        m.package(job, manifest_sha, output)


@pytest.mark.parametrize('bad', ['incomplete', 'registration', 'metric', 'video', 'physical',
                               'summary', 'wrapper', 'duplicate_command', 'missing_command', 'external_path'])
def test_rejects_changed_or_incomplete_originals_without_creating_draft(tmp_path, bad):
    job, manifest_sha, output = prepared(tmp_path)
    if bad == 'incomplete':
        save(job / 'status.json', dict(status='running'))
    elif bad == 'registration':
        manifest_sha = '0' * 64
    elif bad == 'metric':
        (job / 'parts/task_0/json/task_0_301_0.json').write_text('{}')
    elif bad == 'video':
        (job / 'parts/task_0/videos/task_0_301_0.mp4').write_bytes(b'changed')
    elif bad == 'physical':
        path = job / 'parts/task_0/attempts/00.json'; row = m.read_json(path)
        row['actual_applied_controls'][0] = 3; save(path, row)
    elif bad == 'summary':
        path = job / 'summary.json'; row = m.read_json(path)
        row['official_q_score'] = 1.; save(path, row)
    elif bad == 'wrapper':
        (job / 'submission/rgb_wrapper.py').write_text('changed')
    elif bad == 'duplicate_command':
        path = job / 'workers/gpu_0/task_0-command.json'
        (path.parent / 'extra-command.json').write_bytes(path.read_bytes())
    elif bad == 'missing_command':
        (job / 'workers/gpu_0/task_0-command.json').unlink()
    else:
        path = job / 'inventory.json'; row = m.read_json(path)
        row[0]['metrics'] = str(tmp_path / 'foreign.json')
        save(tmp_path / 'foreign.json', {}); save(path, row)
    with pytest.raises((ValueError, KeyError)):
        m.package(job, manifest_sha, output)
    assert not output.exists()
