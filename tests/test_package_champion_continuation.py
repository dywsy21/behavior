import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval'))
import package_champion_continuation as m


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def terminal_fixture(tmp_path, monkeypatch):
    """Synthetic 1000-case accounting with fake ffprobe, not simulator evidence."""
    spec = importlib.util.spec_from_file_location('champion_fixture', Path(__file__).with_name('test_continue_champion_final.py'))
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    prior, runtime, seal, original = base.stopped(tmp_path, monkeypatch)
    tasks = runtime / 'scripts/eval'; (tasks / 'memlite_sft100').mkdir(parents=True)
    for name in ('run_champion_task.py', 'champion_batched_client.py', 'memlite_sft100/wire.py', 'memlite_sft100/rgb_wrapper.py'):
        (tasks / name).write_text('# frozen ' + name)
    (prior / 'submission').mkdir(); robot = prior / 'submission/r1pro.yaml'; robot.write_text('robot: r1pro')
    (prior / 'submission/rgb_wrapper.py').write_bytes((tasks / 'memlite_sft100/rgb_wrapper.py').read_bytes())
    save(tasks / 'memlite_sft100/official_manifest.json', dict(files={'omnigibson/eval/r1pro.yaml': m.c.sha256(robot)}))
    plans = m.c.checked_parts(prior, original, original)
    def command(run, part, gpu, plan):
        argv = ['bash', str(tasks / 'memlite_sft100/launch_sim.sh'), str(tasks / 'run_champion_task.py'),
            '--task', plan['task'], '--task-id', str(plan['task_id']), '--policy-config', original['configs'][gpu]['path'],
            '--output', str(part), '--num-envs', str(plan['num_envs']), '--indices', *map(str, plan['indices'])]
        save(run / 'workers' / f'gpu_{gpu}' / (part.name + '-command.json'), dict(argv=argv, official_commit='official'))
    for part in (prior / 'parts').iterdir():
        gpu, plan = plans[part.name]; command(prior, part, gpu, plan)
    m.c.freeze(prior, runtime, seal)
    job = prior.parent / 'resume'; manifest = m.c.prepare(prior, seal, runtime, job)
    for group in manifest['groups']:
        gpu = group['gpu']; worker = job / 'workers' / f'gpu_{gpu}'
        for plan in group['parts']:
            part = job / 'parts' / plan['id']; n = plan['num_envs']
            save(part / 'status.json', dict(status='completed', task=plan['task'], task_id=plan['task_id'],
                indices=plan['indices'], source_commit='runtime', optimizer_updates=0,
                config_sha256=original['configs'][gpu]['sha256']))
            command(job, part, gpu, plan)
            for offset in range(0, len(plan['indices']), n):
                ids = plan['indices'][offset:offset+n]
                save(part / 'attempts' / f'{ids[0]:02d}.json', dict(status='completed', indices=ids,
                    instance_ids=[i+301 for i in ids], rollout=0, policy_seed=17, simulator_seed=0,
                    max_steps=100, actual_applied_controls=[10]*n))
                for i in ids:
                    name = f'{plan["task"]}_{i+301}_0'
                    save(part / 'json' / (name + '.json'), dict(task=plan['task'], instance_id=i+301,
                        rollout_id=0, success=False, steps=10, q_score={'final': .25}))
                    (part / 'videos').mkdir(exist_ok=True); (part / 'videos' / (name+'.mp4')).write_bytes(b'fake video')
        receipt = m.c.read(prior / f'workers/gpu_{gpu}/model/result.json')
        save(worker / 'model-000/result.json', receipt); save(worker / 'model-000/status.json', receipt)
        save(worker / 'status.json', dict(status='completed', finished=10, pid=99999900+gpu,
            model_pid=receipt['pid'], model_generations=[m.c.binding(worker / 'model-000/result.json')]))
    m.c.shared.report(job, final=True)
    save(job / 'status.json', dict(status='completed_with_execution_errors', finished=20, pid=99999800,
        worker_pids={str(i):99999900+i for i in range(8)}, completed_gpus=list(range(8)), failed_gpus={}))
    return job, runtime, prior


def test_real_audit_keeps_originals_admin_restarts_and_missing_error_batch(tmp_path, monkeypatch):
    job, runtime, prior = terminal_fixture(tmp_path, monkeypatch)
    before = m.c.binding(job / 'summary.json')
    data = m.audited_materials(job, m.c.sha256(job / 'manifest.json'), tmp_path / 'coordinator')
    assert len(data['originals']) == 996 and len(data['errors']) == 4
    assert data['summary']['coverage'] == 1000 and data['summary']['official_full_q_score'] is None
    assert len(data['commands']) == 201
    assert before == m.c.binding(job / 'summary.json')
    out = tmp_path / 'collection'; receipt = m.package(job, m.c.sha256(job / 'manifest.json'), tmp_path / 'coordinator', out)
    assert receipt['metrics'] == 996 and receipt['missing'] == 4 and receipt['submitted'] is False
    assert receipt['model_delivery_ready'] is False and receipt['fabricated_metrics'] == 0
    assert (out/'original-run-manifest.json').read_bytes()==(prior/'manifest.json').read_bytes()
    assert (out/'policy-config-gpu-0.json').read_bytes()==(prior/'configs/gpu_0.json').read_bytes()
    with zipfile.ZipFile(receipt['output']) as z:
        assert len([n for n in z.namelist() if n.endswith('.json')]) == 996
        for name, raw in data['originals']:
            assert z.read('json/' + name) == raw
        assert b'4 missing results' in z.read('README.md')
        assert 'transport/champion_batched_client.py' in z.namelist()
    with pytest.raises(ValueError):
        m.package(job, m.c.sha256(job / 'manifest.json'), tmp_path / 'coordinator', out)


@pytest.mark.parametrize('bad', ['live', 'worker', 'weights', 'generations', 'inventory', 'video',
    'command', 'seed', 'unknown_error', 'summary', 'original', 'wrapper', 'missing_part', 'early'])
def test_audit_refuses_unaccounted_changed_or_live_materials(tmp_path, monkeypatch, bad):
    job, runtime, prior = terminal_fixture(tmp_path, monkeypatch)
    path = None
    if bad in ('live', 'early'):
        path = job/'status.json'; value = m.c.read(path)
        value.update({'pid':os.getpid()} if bad=='live' else {'status':'running'})
    elif bad in ('worker', 'generations'):
        path = job/'workers/gpu_0/status.json'; value = m.c.read(path)
        value.update({'pid':os.getpid()} if bad=='worker' else {'model_generations':[]})
    elif bad=='weights':
        path = job/'workers/gpu_0/model-000/status.json'; value = m.c.read(path); value['parameters_after']='b'*64
    elif bad=='inventory':
        path = job/'inventory.json'; value = m.c.read(path); value.pop()
    elif bad=='video':next((job/'parts').glob('*/videos/*.mp4')).write_bytes(b'changed')
    elif bad=='command':
        path = next((job/'workers').glob('gpu_*/*-command.json')); value = m.c.read(path); value['argv'].insert(3,'--train-smoke')
    elif bad=='seed':
        path = next((job/'parts').glob('*/attempts/*.json')); value = m.c.read(path); value['policy_seed']=18
    elif bad=='unknown_error':
        path=job/'execution-errors.json'; value=m.c.read(path); value[0]['diagnosis']['kind']='unknown'
    elif bad=='summary':
        path=job/'summary.json';value=m.c.read(path);value['completed_only_q']=.99
    elif bad=='original':next((prior/'parts').glob('*/json/*.json')).write_text('{}')
    elif bad=='wrapper':(prior/'submission/rgb_wrapper.py').write_text('# changed')
    else:(job/'parts/foreign').mkdir()
    if path is not None:save(path,value)
    with pytest.raises((ValueError, KeyError)):
        m.audited_materials(job, m.c.sha256(job/'manifest.json'), tmp_path/'coordinator')


def test_state_gate_never_treats_pending_or_failed_as_complete():
    assert not m.terminal_ready({'status':'running'}, {})
    with pytest.raises(RuntimeError):m.terminal_ready({'status':'needs_diagnosis'}, {})


def test_external_coordinator_validation_does_not_repin_actor(tmp_path, monkeypatch):
    job, runtime, _ = terminal_fixture(tmp_path, monkeypatch)
    seen = []; monkeypatch.setattr(m.c, 'frozen_source', lambda path, commit:seen.append((Path(path), commit)))
    _, manifest = m.c.registration(job, coordinator_source=tmp_path/'actual-coordinator')
    assert seen == [(tmp_path/'actual-coordinator', 'coordinator'), (runtime, 'runtime')]
    assert manifest['runtime_source'] == str(runtime)
