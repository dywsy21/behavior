import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/eval'))
import champion_worker_recovery as r
import champion_peer_admission as p
import continue_champion_final as c


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_never_attempted_suffix_preserves_order_and_slots():
    parts = [dict(id=str(i), indices=[i]) for i in range(5)]
    assert r.remaining_suffix(parts, ['0'], ['1'], ['0', '1']) == parts[2:]


@pytest.mark.parametrize('completed,failed,present', [
    (['1'], [], ['1']), (['0'], ['0'], ['0']), (['0'], [], ['0', '1']),
    (['0'], ['2'], ['0', '2']), (['1', '0'], [], ['0', '1']), ([], [], ['0'])])
def test_suffix_rejects_rerolls_holes_unregistered_and_partial_outputs(completed, failed, present):
    with pytest.raises(ValueError):
        r.remaining_suffix([dict(id=str(i)) for i in range(4)], completed, failed, present)


def test_exact_auxiliary_owners_are_visible_not_hidden():
    rows = p.parse_compute('123, 174 MiB\n456, 174 MiB\n')
    admitted = p.validate_owners(rows, {123: dict(gpu=0), 456: dict(gpu=7)}, 4)
    assert [row['pid'] for row in admitted] == [123, 456]
    assert p.validate_owners([], {}, 4) == []


@pytest.mark.parametrize('text', ['123, N/A\n', 'x, 174 MiB\n', '123, 174\n', '123, 1 MiB\n123, 1 MiB\n'])
def test_unknown_nvml_ownership_is_not_idle(text):
    with pytest.raises(ValueError): p.parse_compute(text)


@pytest.mark.parametrize('candidates,mem', [({}, 174), ({1: dict(gpu=4)}, 174), ({1: dict(gpu=2)}, 513)])
def test_foreign_same_gpu_or_large_context_is_never_admitted(candidates, mem):
    with pytest.raises(ValueError): p.validate_owners([dict(pid=1, memory_mib=mem)], candidates, 4)


def test_actual_policy_computation_ast_is_unchanged_from_frozen_actor():
    old = subprocess.check_output(['git', 'show', '9f6e75bd7e5efe4ef487c9e7219fcc361e68088b:scripts/eval/serve_champion_official.py'], cwd=REPO, text=True)
    new = (REPO / 'scripts/eval/serve_champion_official.py').read_text()
    def core(source, revised=False):
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run')
        start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'root' for t in n.targets))
        nodes = fn.body[start:]
        if revised:
            nodes = [n for n in nodes if not (isinstance(n, ast.If) and ast.unparse(n.test) == 'recovery_job is not None')]
        return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    assert core(old) == core(new, True)


@pytest.mark.parametrize('bad', [None, 'unknown_pid', 'same_gpu', 'ppid', 'argv', 'gpu_env', 'source_env', 'large', 'low_free', 'changed_process'])
def test_live_peer_process_contract_is_checked(tmp_path, monkeypatch, bad):
    from common import SIM_PYTHON
    parent=tmp_path/'parent';parent.mkdir();runtime=tmp_path/'runtime'
    cfg=parent/'config.json';save(cfg, {'id':'original'})
    plan=dict(id='task0-n4-i0', task='task0', task_id=0, num_envs=4, indices=[0,1,2,3])
    original=dict(source_commit='actor', configs=[dict(path=str(cfg))]*8)
    manifest=dict(runtime_source=str(runtime));save(parent/'manifest.json',manifest)
    save(parent/'status.json',dict(worker_pids={'0':10}))
    save(parent/'workers/gpu_0/status.json',dict(status='evaluating',part=plan['id'],pid=10,sim_pid=20,runtime_commit='actor'))
    monkeypatch.setattr(c,'root_registration',lambda _: (parent,original,{}))
    monkeypatch.setattr(c,'checked_parts',lambda *a:{plan['id']:(0,plan)})
    argv=[str(SIM_PYTHON),str(runtime/'scripts/eval/run_champion_task.py'),'--task','task0','--task-id','0',
          '--policy-config',str(cfg),'--output',str(parent/'parts'/plan['id']),'--num-envs','4','--indices','0','1','2','3']
    peer=dict(pid=20,ppid=10,start_ticks=200,argv=argv,environment=dict(EVAL_GPU='0',EVAL_SOURCE=str(runtime/'scripts/eval'),CUDA_VISIBLE_DEVICES=None))
    if bad=='ppid':peer['ppid']=99
    elif bad=='argv':peer['argv']=argv[:-1]+['9']
    elif bad=='gpu_env':peer['environment']['EVAL_GPU']='7'
    elif bad=='source_env':peer['environment']['EVAL_SOURCE']='foreign'
    counts={}
    def fake_process(pid):
        counts[pid]=counts.get(pid,0)+1
        if pid==10:return dict(pid=10,ppid=1,start_ticks=100,argv=['worker'],environment={})
        row=deepcopy(peer)
        if bad=='changed_process' and counts[pid]>1:row['start_ticks']+=1
        return row
    monkeypatch.setattr(p,'process',fake_process)
    def query(argv,**kwargs):
        if '--query-gpu=memory.free' in argv:return '1000\n' if bad=='low_free' else '60000\n'
        return '99, 174 MiB\n' if bad=='unknown_pid' else f'20, {513 if bad=="large" else 174} MiB\n'
    monkeypatch.setattr(p.subprocess,'check_output',query)
    if bad:
        with pytest.raises(ValueError):p.check(parent,c.sha256(parent/'manifest.json'),0 if bad=='same_gpu' else 4)
    else:
        value=p.check(parent,c.sha256(parent/'manifest.json'),4)
        assert value['owners'][0]['peer']['pid']==20 and not value['peers_stopped']


def terminal_journal():
    spec = importlib.util.spec_from_file_location('recovery_fixture', Path(__file__).with_name('test_champion_official.py'))
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    from behavior_champion.official import OfficialConnection
    rows = []; clients = [OfficialConnection(f.factory, connection_id=str(i)) for i in range(4)]
    for client in clients:
        rows += [dict(kind='connected', connection=client.connection_id),
                 dict(kind='reset', connection=client.connection_id, previous=client.reset())]
        client.prepare(f.observation(0, 20))
        rows.append(dict(kind='delivered', connection=client.connection_id, **client.delivered()))
    for client in clients:
        rows += [dict(kind='connection_error', connection=client.connection_id, error='ConnectionClosedError(None, None, None)'),
                 dict(kind='closed', connection=client.connection_id, previous=client.reset())]
    return rows


@pytest.mark.parametrize('bad', [None, 'error', 'receipt', 'task', 'extra', 'missing'])
def test_only_exact_failed_batch_disconnect_tail_is_accepted(tmp_path, bad):
    rows = terminal_journal()
    if bad == 'error': rows[-2]['error'] = 'ValueError(other)'
    if bad == 'receipt': rows[-1]['previous']['predictions'] = 999
    if bad == 'task': rows[-1]['previous']['task'] = 1
    if bad == 'extra': rows += [deepcopy(rows[-2]), deepcopy(rows[-1])]
    if bad == 'missing': rows.pop()
    path = tmp_path / 'trace.jsonl'; path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    if bad:
        with pytest.raises(ValueError): r.audit_terminal_disconnects(path, 0, 4)
    else:
        report = r.audit_terminal_disconnects(path, 0, 4)
        assert len(report['terminal_disconnects']) == 8
        assert report['prefix']['active_connections'] == 4


def recovery_fixture(tmp_path, monkeypatch, setup_original=None):
    spec = importlib.util.spec_from_file_location('recovery_parent_fixture', Path(__file__).with_name('test_continue_champion_final.py'))
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    old, runtime, seal, original = f.stopped(tmp_path, monkeypatch)
    if setup_original is not None:
        setup_original(old, runtime, original)
    c.freeze(old, runtime, seal)
    parent = old.parent / 'parent'
    pm = c.prepare(old, seal, runtime, parent, isolate_batches=True, continue_on_errors=True)
    monkeypatch.setattr(r, 'ROOT', c.ROOT)
    first, failed = pm['groups'][0]['parts'][:2]
    wp = parent / 'workers/gpu_0'; worker_pid = 99999050
    state = dict(status='needs_diagnosis', pid=worker_pid, finished=100, gpu=0,
                 model_pid=99999150, sim_pid=99999250, model_path=str(wp/'model-001'),
                 completed_parts=[first['id']], diagnosed_failed_parts=[failed['id']],
                 error="RuntimeError('Frozen model loading failed/timed out/stopped')",
                 final_verification_error="FileNotFoundError(2, 'No such file or directory')")
    result = dict(status='closed', pid=99999350, weights_unchanged=True, optimizer_updates=0,
                  source_commit='runtime', config_sha256=original['configs'][0]['sha256'],
                  asset_receipt_sha256='export', parameters_before='a'*64, parameters_after='a'*64)
    save(wp/'model-000/result.json', result); save(wp/'model-000/status.json', result)
    (wp/'model-000/events.jsonl').write_text('fixture separately tested by real-connection tests\n')
    state['model_generations'] = [c.binding(wp/'model-000/result.json')]
    save(wp/'status.json', state)
    save(parent/'status.json', dict(status='running', worker_pids={'0':worker_pid}, failed_gpus={'0':state}))
    (parent/'gpu_0.log').write_text('worker reload rejected\n')
    (wp/'model-001.log').write_text('Traceback (most recent call last):\nValueError: Registered model GPU must be idle\n')
    for plan in (first, failed):
        name = plan['id']; d = parent/'parts'/name
        ps = dict(status='completed' if plan == first else 'failed', **{k:plan[k] for k in ('task','task_id','indices')},
                  source_commit='runtime', config_sha256=original['configs'][0]['sha256'], optimizer_updates=0)
        save(d/'status.json', ps)
        attempt = dict(status='completed' if plan == first else 'running', indices=plan['indices'],
                       instance_ids=[301+i for i in plan['indices']], rollout=0, policy_seed=17, simulator_seed=0,
                       max_steps=100, actual_applied_controls=[10]*len(plan['indices']))
        a = d/'attempts/00.json'; save(a, attempt)
        if plan == first:
            for i in attempt['instance_ids']:
                save(d/'json'/f'{plan["task"]}_{i}_0.json', dict(task=plan['task'],instance_id=i,rollout_id=0,
                    steps=10,success=False,q_score={'final':0.}))
                (d/'videos').mkdir(exist_ok=True); (d/'videos'/f'{plan["task"]}_{i}_0.mp4').write_bytes(b'video')
        else:
            log = wp/(name+'.log'); log.write_text(f.reset_trace())
            diagnosis = c.execution_error(log.read_text())
            errors = [dict(task=plan['task'],instance_id=i,rollout_id=0,original_attempt=c.binding(a),
                           diagnosis=diagnosis,log=c.binding(log),rerun_allowed=False) for i in attempt['instance_ids']]
            save(wp/(name+'-execution-errors.json'),errors); save(wp/(name+'-command.json'),{'argv':['original']})
    monkeypatch.setattr(r, 'audit_terminal_disconnects', lambda *a: {'fixture':'tested separately'})
    return parent, pm, wp


def test_prepare_recovery_never_mutates_live_parent_or_rerolls_errors(tmp_path, monkeypatch):
    parent, pm, wp = recovery_fixture(tmp_path, monkeypatch)
    before = {str(p):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    job = parent.parent/'recovery'; m = r.prepare(parent,0,job)
    assert m['groups'][0]['parts'] == pm['groups'][0]['parts'][2:]
    assert all(g['parts']==[] for g in m['groups'][1:])
    assert all(Path(path).read_bytes()==data for path,data in before.items())
    assert not (parent/'STOP').exists()
    assert r.registration(job,0)[1] == m
    assert m['configs'] == pm['configs'] and m['runtime_commit'] == pm['runtime_commit']


@pytest.mark.parametrize('bad',['loaded','weights','unknown','live','pending_attempted','ledger'])
def test_recovery_certificate_rejects_ambiguous_or_changed_parent(tmp_path, monkeypatch, bad):
    parent, pm, wp = recovery_fixture(tmp_path, monkeypatch)
    if bad == 'loaded': (wp/'model-001').mkdir()
    elif bad == 'weights':
        value=c.read(wp/'model-000/result.json');value['parameters_after']='b'*64;save(wp/'model-000/result.json',value)
    elif bad == 'unknown': (wp/'model-001.log').write_text('RuntimeError: unknown\n')
    elif bad == 'live':
        import os
        value=c.read(wp/'status.json');value['model_pid']=os.getpid();save(wp/'status.json',value)
    elif bad == 'pending_attempted': (parent/'parts'/pm['groups'][0]['parts'][2]['id']).mkdir()
    else:
        name=pm['groups'][0]['parts'][1]['id'];save(wp/(name+'-execution-errors.json'),[])
    with pytest.raises((ValueError,RuntimeError)): r.certify(parent,0)
