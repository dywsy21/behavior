"""Synthetic end-to-end accounting; no simulator performance claims."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval'))
import champion_worker_recovery as r
import package_champion_continuation as p

c = p.c


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def command(run, plan, gpu, runtime, original):
    scripts = runtime / 'scripts/eval'; part = run / 'parts' / plan['id']
    argv = ['bash', str(scripts / 'memlite_sft100/launch_sim.sh'), str(scripts / 'run_champion_task.py'),
            '--task', plan['task'], '--task-id', str(plan['task_id']), '--policy-config', original['configs'][gpu]['path'],
            '--output', str(part), '--num-envs', str(plan['num_envs']), '--indices', *map(str, plan['indices'])]
    save(run / 'workers' / f'gpu_{gpu}' / (plan['id'] + '-command.json'), dict(argv=argv, official_commit='official'))


def setup_original(root, runtime, original):
    scripts = runtime / 'scripts/eval'; (scripts / 'memlite_sft100').mkdir(parents=True)
    for name in ('run_champion_task.py', 'champion_batched_client.py', 'memlite_sft100/wire.py', 'memlite_sft100/rgb_wrapper.py'):
        (scripts / name).write_text('# frozen ' + name)
    (root / 'submission').mkdir(); (root / 'submission/r1pro.yaml').write_text('robot: r1pro')
    (root / 'submission/rgb_wrapper.py').write_bytes((scripts / 'memlite_sft100/rgb_wrapper.py').read_bytes())
    save(scripts / 'memlite_sft100/official_manifest.json', dict(files={
        'omnigibson/eval/r1pro.yaml': c.sha256(root / 'submission/r1pro.yaml')}))
    parts = c.checked_parts(root, original, original)
    for part in (root / 'parts').iterdir():
        gpu, plan = parts[part.name]; command(root, plan, gpu, runtime, original)


def completed_part(run, plan, gpu, runtime, original):
    part = run / 'parts' / plan['id']; indices = plan['indices']
    save(part / 'status.json', dict(status='completed', **{k:plan[k] for k in ('task', 'task_id', 'indices')},
        source_commit='runtime', optimizer_updates=0, config_sha256=original['configs'][gpu]['sha256']))
    save(part / 'attempts' / f'{indices[0]:02d}.json', dict(status='completed', indices=indices,
        instance_ids=[i+301 for i in indices], rollout=0, policy_seed=17, simulator_seed=0,
        max_steps=100, actual_applied_controls=[10]*len(indices)))
    command(run, plan, gpu, runtime, original)
    for i in indices:
        name = f'{plan["task"]}_{i+301}_0'
        save(part / 'json' / (name+'.json'), dict(task=plan['task'], instance_id=i+301, rollout_id=0,
            success=False, steps=10, q_score={'final':.25}))
        (part / 'videos').mkdir(exist_ok=True); (part / 'videos' / (name+'.mp4')).write_bytes(b'fake video')


def completed_worker(run, group, original, runtime, *, recovery=False):
    gpu = group['gpu']; wp = run / 'workers' / f'gpu_{gpu}'
    for plan in group['parts']:
        completed_part(run, plan, gpu, runtime, original)
    model = dict(status='closed', pid=99999100+gpu, weights_unchanged=True, optimizer_updates=0,
        source_commit='runtime', config_sha256=original['configs'][gpu]['sha256'], asset_receipt_sha256='export',
        parameters_before='a'*64, parameters_after='a'*64)
    if recovery:
        model.update(launcher_source_commit=c.read(run/'manifest.json')['source_commit'],
                     recovery_manifest_sha256=c.sha256(run/'manifest.json'))
    save(wp/'model-000/result.json', model); save(wp/'model-000/status.json', model)
    save(wp/'status.json', dict(status='completed', pid=99999000+gpu, model_pid=model['pid'], finished=100,
        completed_parts=[v['id'] for v in group['parts']], diagnosed_failed_parts=[],
        model_generations=[c.binding(wp/'model-000/result.json')]))


def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('recovery_fixture', Path(__file__).with_name('test_champion_worker_recovery.py'))
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    parent, pm, wp = f.recovery_fixture(tmp_path, monkeypatch, setup_original)
    root, original, _ = c.root_registration(parent); runtime = Path(pm['runtime_source'])
    for plan in pm['groups'][0]['parts'][:2]:
        command(parent, plan, 0, runtime, original)
    for group in pm['groups'][1:]:
        completed_worker(parent, group, original, runtime)
    failed = c.read(wp / 'status.json')
    save(parent / 'status.json', dict(status='needs_isolated_worker_recovery', finished=101, pid=99999999,
        worker_pids={str(i):c.read(parent/f'workers/gpu_{i}/status.json')['pid'] for i in range(8)},
        completed_gpus=list(range(1,8)), failed_gpus={'0':failed}))
    c.shared.report(parent)
    child = parent.parent/'recovery'; rm = r.prepare(parent,0,child)
    completed_worker(child,rm['groups'][0],original,runtime,recovery=True)
    save(child/'status.json',dict(status='completed',pid=99999000,finished=102))
    return parent, child, tmp_path/'coordinator', tmp_path/'recovery-source'


def test_terminal_union_keeps_original_parent_and_discloses_all_attempts(tmp_path, monkeypatch):
    parent, child, cs, rs = fixture(tmp_path, monkeypatch)
    before = {str(x):x.read_bytes() for x in parent.rglob('*') if x.is_file()}
    data = p.audited_materials(parent,c.sha256(parent/'manifest.json'),cs,recoveries=[(child,rs)])
    assert len(data['originals']) == 992 and len(data['errors']) == 8
    assert data['summary']['coverage'] == 1000 and data['summary']['official_full_sr'] is None
    assert child in data['lineage'] and 'recovery-gpu-0-certificate.json' in data['sidecars']
    out=tmp_path/'collection'
    receipt=p.package(parent,c.sha256(parent/'manifest.json'),cs,out,recoveries=[(child,rs)])
    assert receipt['metrics']==992 and receipt['missing']==8 and not receipt['submitted']
    assert receipt['recoveries'][0]['manifest']==c.binding(child/'manifest.json')
    assert all(Path(name).read_bytes()==raw for name,raw in before.items())
    assert not (parent/'inventory.json').exists() and not (parent/'STOP').exists()


@pytest.mark.parametrize('bad',['early','live','missing_recovery','extra_recovery','duplicate','foreign_parent',
    'weights','launcher','receipt','summary','parent_reroll','missing_part','seed','foreign_worker','generations','certificate'])
def test_union_refuses_unverified_recoveries_or_changed_materials(tmp_path,monkeypatch,bad):
    parent,child,cs,rs=fixture(tmp_path,monkeypatch);recs=[(child,rs)];path=None
    if bad in ('early','live'):
        path=child/'status.json';row=c.read(path);row.update(status='running') if bad=='early' else row.update(pid=os.getpid())
    elif bad=='missing_recovery':recs=[]
    elif bad=='extra_recovery':
        path=parent/'status.json';row=c.read(path);row['failed_gpus']['1']={}
    elif bad=='duplicate':recs*=2
    elif bad=='foreign_parent':
        path=child/'manifest.json';row=c.read(path);row['parent']=str(parent.parent/'other')
    elif bad in ('weights','launcher','receipt'):
        path=child/'workers/gpu_0/model-000/status.json';row=c.read(path)
        row[{'weights':'parameters_after','launcher':'launcher_source_commit','receipt':'recovery_manifest_sha256'}[bad]]='wrong'
    elif bad=='summary':
        path=parent/'summary.json';row=c.read(path);row['completed']+=1
    elif bad=='parent_reroll':
        name=c.read(child/'manifest.json')['groups'][0]['parts'][0]['id'];(parent/'parts'/name).mkdir()
    elif bad=='missing_part':
        part=next((child/'parts').iterdir());part.rename(part.with_name('foreign'))
    elif bad=='seed':
        path=next((child/'parts').glob('*/attempts/*.json'));row=c.read(path);row['policy_seed']=18
    elif bad=='foreign_worker':save(child/'workers/gpu_1/status.json',{})
    elif bad=='generations':
        path=child/'workers/gpu_0/status.json';row=c.read(path);row['model_generations']=[]
    elif bad=='certificate':
        path=child/'parent-worker-certificate.json';row=c.read(path);row['pending'].pop()
    if path:save(path,row)
    with pytest.raises((ValueError,RuntimeError,KeyError)):
        p.audited_materials(parent,c.sha256(parent/'manifest.json'),cs,recoveries=recs)


def test_wait_gate_tracks_sidecar_after_parent_exits(tmp_path,monkeypatch):
    parent,child,cs,rs=fixture(tmp_path,monkeypatch)
    before=c.read(child/'status.json')
    save(child/'status.json',before|dict(status='running',pid=os.getpid()))
    assert not p.recovered_terminal_ready(parent,[(child,rs)])
    save(child/'status.json',before)
    assert p.recovered_terminal_ready(parent,[(child,rs)])


def test_new_collector_checks_original_recovery_coordinator_not_its_own_source(tmp_path,monkeypatch):
    parent,child,cs,rs=fixture(tmp_path,monkeypatch);seen=[]
    monkeypatch.setattr(c,'frozen_source',lambda path,commit:seen.append((Path(path),commit)))
    p.registered_recoveries(parent,[(child,rs)])
    assert (rs,c.read(child/'manifest.json')['source_commit']) in seen
    assert not any(path==c.REPO for path,_ in seen)
