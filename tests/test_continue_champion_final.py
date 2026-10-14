import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval'))
import continue_champion_final as m


def reset_trace():
    return ('Traceback (most recent call last):\n'
        'File "/official/omnigibson/eval/evaluator.py", line 116, in evaluate_instances_batched\n'
        'load_fn(dict(env_idx_to_instance))\n'
        'File "/official/omnigibson/eval/evaluator.py", line 654, in load_fn\n'
        'File "/official/omnigibson/eval/evaluator.py", line 549, in load_batch\n'
        'File "/official/omnigibson/eval/evaluator.py", line 507, in _load_instance_state\n'
        'obj.load_state(tro_state, serialized=False)\n'
        "AttributeError: 'NoneType' object has no attribute 'load_state'\n")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_exact_reset_failure_is_not_actor_rejection_or_physical_success():
    r = m.execution_error(reset_trace())
    assert r['kind'] == 'simulator_missing_task_object_on_reset'
    assert r['scene_slot'] is None and r['official_score_available'] is False
    assert r['failure_phase'] == 'official_instance_state_load'


@pytest.mark.parametrize('change', ['frame', 'call', 'error', 'extra', 'cuda', 'twice', 'notrace'])
def test_unknown_or_mixed_errors_stop(change):
    t = reset_trace()
    if change == 'frame': t = t.replace('in _load_instance_state', 'in predict')
    elif change == 'call': t = t.replace('obj.load_state(tro_state, serialized=False)', 'model.load_state(x)')
    elif change == 'error': t = t.replace("attribute 'load_state'", "attribute 'shape'")
    elif change == 'extra': t += 'ValueError: other failure\n'
    elif change == 'cuda': t += 'CUDA out of memory\n'
    elif change == 'twice': t += t
    else: t = t.replace('Traceback (most recent call last):', '')
    assert m.execution_error(t) is None


def stopped(tmp_path, monkeypatch):
    root = tmp_path / 'root'; prior = root / 'runs/original'; runtime = tmp_path / 'runtime'; runtime.mkdir()
    prior.mkdir(parents=True); (prior/'STOP').touch()
    tasks = [f'task{i}' for i in range(100)]
    groups = [dict(gpu=i, tasks=[dict(task=tasks[j],task_id=j) for j in range(i,100,8)]) for i in range(8)]
    configs=[]
    for i in range(8):
        p=prior/f'configs/gpu_{i}.json';save(p,dict(reference='reference',openpi_reference='openpi'))
        configs.append(dict(gpu=i,**m.binding(p)))
    terminal=prior/'terminal.json';save(terminal,{'complete':True})
    instance=prior/'instance.json';save(instance,{'state':True})
    original=dict(schema=m.ORIGINAL,source_commit='runtime',configs=configs,
        tasks=tasks,cases=[dict(task=t,instance_id=i,rollout_id=0,instance_path=str(instance),
                              instance_sha256=m.sha256(instance)) for t in tasks for i in range(301,311)],
        groups=groups,optimizer_updates=0,automatic_retries=False,policy_seed=17,simulator_seed=0,
        export_sha256='export',terminal_path=str(terminal),terminal_sha256=m.sha256(terminal),
        official_commit='official',nominal_horizons={t:100 for t in tasks},task_csv_sha256='csv',
        model_python='/pinned/python',model_overlay='/pinned/overlay',checkpoint='/pinned/ema',transport='original')
    save(prior/'manifest.json',original);save(prior/'accepted-preflight.json',{'num_envs':4})
    save(prior/'status.json',dict(status='needs_diagnosis',pid=99999000,finished=50,
        worker_pids={str(i):99999001+i for i in range(8)},failed_gpus={'0':{}}))
    for i in range(8):
        wp=prior/f'workers/gpu_{i}'
        w=dict(status='failed_no_retry',pid=99999001+i,finished=45,model_pid=99999100+i,
               error="InterruptedError('Administrative stop')")
        if i < 2:
            w['part']=f'task{i}-n4-i0';w['sim_pid']=99999200+i
        if i==0:w['error']="RuntimeError('Official task batch incomplete')"
        save(wp/'status.json',w)
        v=dict(status='closed',pid=99999100+i,weights_unchanged=True,optimizer_updates=0,
            source_commit='runtime',config_sha256=configs[i]['sha256'],asset_receipt_sha256='export',
            parameters_before='a'*64,parameters_after='a'*64)
        save(wp/'model/result.json',v);save(wp/'model/status.json',v)
        (wp/'model/events.jsonl').write_text('original trace\n')
    for i in (0,1):
        p=prior/f'parts/task{i}-n4-i0'
        ps=dict(status='running',task=f'task{i}',task_id=i,indices=list(range(8)),
            source_commit='runtime',config_sha256=configs[i]['sha256'],optimizer_updates=0,
            control_steps=[10]*4,updated=25)
        save(p/'status.json',ps)
        a=dict(status='running',task=f'task{i}',indices=list(range(4)),instance_ids=list(range(301,305)),
               rollout=0,started=10,policy_seed=17,simulator_seed=0,max_steps=100)
        if i==0:
            save(p/'attempts/00.json',a|dict(status='completed',actual_applied_controls=[10]*4))
            for instance_id in range(301,305):
                name=f'task0_{instance_id}_0'
                save(p/'json'/f'{name}.json',dict(task='task0',instance_id=instance_id,rollout_id=0,
                    steps=10,q_score={'final':0.},success=False))
                (p/'videos').mkdir(exist_ok=True);(p/'videos'/f'{name}.mp4').write_bytes(b'video')
            save(p/'attempts/04.json',a|dict(indices=list(range(4,8)),instance_ids=list(range(305,309)),started=20))
        else:save(p/'attempts/00.json',a)
        wp=prior/f'workers/gpu_{i}'
        (wp/f'task{i}-n4-i0.log').write_text(reset_trace() if i==0 else 'normal simulation\n')
        save(wp/f'task{i}-n4-i0-command.json',dict(argv=['original'],official_commit='official'))
    monkeypatch.setattr(m,'ROOT',root)
    monkeypatch.setattr(m,'frozen_source',lambda *a:None)
    real_check=m.subprocess.check_output
    def output(argv,**kw):
        if argv[0]=='ffprobe':return json.dumps({'streams':[dict(nb_frames='10',r_frame_rate='30/1',width=672,height=448)]})
        if argv[:3]==['git','rev-parse','HEAD']:return 'coordinator\n'
        return real_check(argv,**kw)
    monkeypatch.setattr(m.subprocess,'check_output',output)
    return prior,runtime,root/'runs/seal.json',original


def test_freeze_and_prepare_keep_four_completed_four_errors_and_admin_restart(tmp_path,monkeypatch):
    prior,runtime,seal,original=stopped(tmp_path,monkeypatch)
    before=(prior/'manifest.json').read_bytes()
    r=m.freeze(prior,runtime,seal)
    assert r['completed']==4 and len(r['execution_errors'])==4 and len(r['administrative_interruptions'])==4
    assert r['unstarted']==988
    assert {x['instance_id'] for x in r['execution_errors']}==set(range(305,309))
    assert all(x['rerun_allowed'] is False for x in r['execution_errors'])
    job=prior.parent/'resume';manifest=m.prepare(prior,seal,runtime,job)
    planned=[(p['task'],i+301,0) for g in manifest['groups'] for p in g['parts'] for i in p['indices']]
    assert len(planned)==len(set(planned))==992
    assert manifest['configs']==original['configs'] and manifest['runtime_commit']=='runtime'
    assert manifest['groups'][0]['parts'][0]['indices']==[8,9]
    assert (prior/'manifest.json').read_bytes()==before and (prior/'STOP').exists()
    assert m.registration(job)[1]==manifest


@pytest.mark.parametrize('bad',['live','weights','unknown','admin_has_error','uncommitted','seed','horizon','changed_config','missing_receipt'])
def test_freeze_rejects_live_changed_unknown_and_uncommitted_evidence(tmp_path,monkeypatch,bad):
    prior,runtime,seal,_=stopped(tmp_path,monkeypatch)
    if bad=='live':
        p=prior/'status.json';s=m.read(p);s['pid']=os_getpid();save(p,s)
    elif bad=='weights':
        p=prior/'workers/gpu_2/model/result.json';s=m.read(p);s['parameters_after']='b'*64;save(p,s)
    elif bad=='unknown':(prior/'workers/gpu_0/task0-n4-i0.log').write_text('ValueError: unknown\n')
    elif bad=='admin_has_error':(prior/'workers/gpu_1/task1-n4-i0.log').write_text('RuntimeError: independent failure\n')
    elif bad=='uncommitted':save(prior/'parts/task0-n4-i0/json/task0_305_0.json',{'success':True})
    elif bad in ('seed','horizon'):
        p=prior/'parts/task0-n4-i0/attempts/04.json';s=m.read(p);s['policy_seed' if bad=='seed' else 'max_steps']=999;save(p,s)
    elif bad=='changed_config':save(prior/'configs/gpu_0.json',{'changed':True})
    else:(prior/'workers/gpu_2/model/result.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):m.freeze(prior,runtime,seal)
    assert not seal.exists()


def os_getpid():
    import os
    return os.getpid()


@pytest.mark.parametrize('bad',['groups','runtime','config','seal'])
def test_registration_does_not_allow_changed_schedule_or_identity(tmp_path,monkeypatch,bad):
    prior,runtime,seal,_=stopped(tmp_path,monkeypatch);m.freeze(prior,runtime,seal)
    job=prior.parent/'resume';manifest=m.prepare(prior,seal,runtime,job)
    if bad=='groups':manifest['groups'][0]['parts'][0]['indices']=[0,1]
    elif bad=='runtime':manifest['runtime_commit']='other'
    elif bad=='config':manifest['configs'][0]['sha256']='changed'
    else:save(seal,{'tampered':True})
    save(job/'manifest.json',manifest)
    with pytest.raises((ValueError,KeyError)):m.registration(job)


def test_model_and_simulator_paths_use_original_runtime(tmp_path,monkeypatch):
    prior,runtime,seal,_=stopped(tmp_path,monkeypatch);m.freeze(prior,runtime,seal)
    manifest=m.prepare(prior,seal,runtime,prior.parent/'resume')
    env=m.model_environment(manifest,0)
    assert env['PYTHONPATH'].split(':')[0]==str(runtime/'src')
    assert env['CUDA_VISIBLE_DEVICES']=='0' and env['HF_HUB_OFFLINE']=='1'


def test_errors_do_not_create_official_json_or_full_score():
    tasks=[f'task{i}' for i in range(100)]
    row=dict(task='task0',instance_id=301,rollout_id=0,steps=1,success=True,q_score={'final':1.})
    errors=[dict(task=t,instance_id=i,rollout_id=0) for t in tasks for i in range(301,311)
            if (t,i)!=('task0',301)]
    result=m.shared.summary_from_rows(tasks,[row],errors)
    assert result['coverage']==1000 and result['completed']==1 and result['execution_errors']==999
    assert result['official_full_q_score'] is None and result['official_full_sr'] is None
    assert result['conservative_q_errors_and_missing_as_zero']==.001
    assert result['fabricated_official_metrics']==0


@pytest.mark.parametrize('unknown', [False, True])
def test_worker_uses_original_entry_and_never_retries_failed_prefix(tmp_path,monkeypatch,unknown):
    prior,runtime,seal,original=stopped(tmp_path,monkeypatch);m.freeze(prior,runtime,seal)
    job=prior.parent/'resume';manifest=m.prepare(prior,seal,runtime,job)
    # Exercise the coordinator with fake processes, not a claimed physical run.
    parts=[dict(task='task0',task_id=0,num_envs=4,indices=list(range(8)),id='task0-n4-i0'),
           dict(task='task0',task_id=0,num_envs=2,indices=[8,9],id='task0-n2-i8')]
    manifest['groups'][0]['parts']=parts
    monkeypatch.setattr(m,'registration',lambda _: (job,manifest))
    monkeypatch.setattr(m,'root_registration',lambda _: (prior,original,m.binding(prior/'manifest.json')))
    simroot=tmp_path/'simroot';simroot.mkdir();monkeypatch.setattr(m,'SIM_ROOT',simroot)
    class Permit:
        def close(self):pass
    monkeypatch.setattr(m,'loader_lock',lambda _:Permit())
    calls=[];models={}
    class Process:
        def __init__(self,argv,**kw):
            self.argv=argv;self.pid=99000000+len(calls);calls.append((argv,kw))
            self.is_model='serve_champion_official.py' in argv[1]
            if self.is_model:
                self.returncode=None;models[self.pid]=self
                self.output=Path(argv[argv.index('--output')+1]);save(self.output/'status.json',{'status':'ready'})
            else:
                self.returncode=0
                output=Path(argv[argv.index('--output')+1]);ids=list(map(int,argv[argv.index('--indices')+1:]))
                count=int(argv[argv.index('--num-envs')+1]);first=ids[0]==0
                save(output/'status.json',dict(status='running' if first else 'completed',task='task0'))
                attempt=dict(status='running' if first else 'completed',task='task0',indices=ids[:count],
                    instance_ids=[i+301 for i in ids[:count]],rollout=0,started=1,policy_seed=17,simulator_seed=0,max_steps=100)
                if first:
                    kw['stdout'].write('ValueError: unknown\n' if unknown else reset_trace());kw['stdout'].flush()
                else:
                    attempt['actual_applied_controls']=[10]*count
                    for i in ids:
                        stem=f'task0_{i+301}_0'
                        save(output/'json'/f'{stem}.json',dict(task='task0',instance_id=i+301,rollout_id=0,
                            steps=10,q_score={'final':0.},success=False))
                        (output/'videos').mkdir(exist_ok=True);(output/'videos'/f'{stem}.mp4').write_bytes(b'v'*2000)
                save(output/'attempts'/f'{ids[0]:02d}.json',attempt)
        def poll(self):return self.returncode
        def wait(self,timeout=None):return self.returncode
    def terminate(pid,_signal):
        model=models[pid];model.returncode=0
        receipt=m.read(prior/'workers/gpu_0/model/result.json')|dict(pid=pid)
        save(model.output/'result.json',receipt);save(model.output/'status.json',receipt)
    monkeypatch.setattr(m.subprocess,'Popen',Process);monkeypatch.setattr(m.os,'killpg',terminate)
    if unknown:
        with pytest.raises(RuntimeError,match='Unclassified'):m.worker(job,0)
    else:m.worker(job,0)
    simulations=[(a,k) for a,k in calls if a[0]=='bash']
    assert all(a[2]==str(runtime/'scripts/eval/run_champion_task.py') for a,_ in simulations)
    assert all(k['env']['EVAL_SOURCE']==str(runtime/'scripts/eval') for _,k in simulations)
    model_calls=[a for a,_ in calls if a[0]=='/pinned/python']
    assert all(a[1]==str(runtime/'scripts/eval/serve_champion_official.py') for a in model_calls)
    worker=m.read(job/'workers/gpu_0/status.json')
    if unknown:
        assert len(simulations)==len(model_calls)==1 and worker['status']=='needs_diagnosis'
        assert not list((job/'workers/gpu_0').glob('*-execution-errors.json'))
    else:
        assert [list(map(int,a[a.index('--indices')+1:])) for a,_ in simulations]==[list(range(8)),list(range(4,8)),[8,9]]
        assert len(model_calls)==2 and worker['status']=='completed_with_execution_errors'
        errors=m.read(job/'workers/gpu_0/task0-n4-i0-execution-errors.json')
        assert [r['instance_id'] for r in errors]==list(range(301,305))
        assert len(worker['model_generations'])==2
