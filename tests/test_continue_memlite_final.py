import importlib.util
import json
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('continue_final',Path(__file__).resolve().parents[1]/'scripts/eval/continue_memlite_final.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def fixture():
    tasks=[f'task{i}' for i in range(100)]
    groups=[dict(gpu=i,tasks=[]) for i in range(8)]
    for i,t in enumerate(tasks):groups[i%8]['tasks'].append(dict(task=t,task_id=i))
    manifest=dict(tasks=tasks,groups=groups,cases=[dict(task=t,instance_id=i,rollout_id=0)
        for t in tasks for i in range(301,311)])
    seal=dict(records=[dict(task=tasks[0],instance_id=i,rollout_id=0) for i in range(301,305)],
              execution_errors=[dict(task=tasks[0],instance_id=i,rollout_id=0) for i in range(305,309)])
    return manifest,seal


def test_remaining_cases_never_include_completed_or_physics_failed():
    manifest,seal=fixture();groups=m.remaining_parts(manifest,seal)
    rows=[(p['task'],i+301,0) for g in groups for p in g['parts'] for i in p['indices']]
    assert len(rows)==len(set(rows))==992
    assert set(rows)&{m.key(r) for r in seal['records']+seal['execution_errors']}==set()
    assert groups[0]['parts'][0]['indices']==[8,9]
    assert groups[0]['parts'][0]['num_envs']==2
    # Arbitrary outcome changes cannot affect the schedule.
    for row in seal['records']:row.update(success=True,q_score={'final':1.})
    assert groups==m.remaining_parts(manifest,seal)


@pytest.mark.parametrize('bad',['partial','duplicate','foreign','overlap'])
def test_case_exclusion_validation(bad):
    manifest,seal=fixture()
    if bad=='partial':seal['records'].pop()
    elif bad=='duplicate':seal['records'].append(seal['records'][0])
    elif bad=='foreign':seal['records'][0]['task']='other'
    else:seal['execution_errors'].append(seal['records'][0])
    with pytest.raises(ValueError):m.remaining_parts(manifest,seal)


def test_physics_failure_whitelist_requires_exact_actual_signature():
    text=('Traceback (most recent call last):\n  File "rigid_dynamic_prim.py"\n'
          'AssertionError: /World/scene_3/controllable__r1pro__robot_r1/base_link '
          'orientation tensor([nan, nan, nan, nan]) is not a unit quaternion.')
    result=m.physics_error(text)
    assert result['scene_slot']==3 and result['official_score_available'] is False
    assert m.physics_error(text+'\n'+text)['traceback_copies']==2
    for bad in ('CUDA out of memory','connection lost','model malformed JSON',text.replace('nan','0.'),
                text+'\n'+text.replace('scene_3','scene_2'),text+'\nRuntimeError: CUDA out of memory',
                'ValueError: wrong action\n'+text,text+'\nValueError: wrong action'):
        assert m.physics_error(bad) is None


def test_callback_nan_wrapped_repeated_finger_link_is_one_physical_failure():
    text=('Traceback (most recent call last):\n  File "rigid_dynamic_prim.py"\n'
          'AssertionError: /World/scene_1/controllable__r1pro__robot_r1/left_gripper_finger_link1 '
          'orientation tensor([nan, nan, nan, nan]) is not a unit quaternion.')
    cascade='AssertionError: read_from_physx must be called during a physics step'
    wrapper='RuntimeError: Exception occurred during pre-physics step'
    result=m.physics_error('\n'.join([text,cascade,text,cascade,text,wrapper]))
    assert result['scene_slot']==1 and result['robot_link']=='left_gripper_finger_link1'
    assert result['traceback_copies']==3
    for bad in (cascade+'\n'+text,text.replace('robot_r1/','other_robot/'),
                text+'\n'+text.replace('left_gripper','right_gripper'),text.replace('nan, nan','inf, nan'),
                text+'\nRuntimeError: Model parse error'):
        assert m.physics_error(bad) is None


def articulation_failure_fixture():
    return ('PhysX error: NpSceneQueries multiQuery input check: unitDir is not valid.\n'
        'Articulation needs to be initialized.\nTraceback (most recent call last):\n'
        'File "/official/omnigibson/eval/evaluator.py", line 386, in _apply_actions\n'
        'File "/official/omnigibson/envs/env_base.py", line 769, in step\n'
        'File "/official/omnigibson/envs/env_base.py", line 622, in get_obs\n'
        'File "/official/omnigibson/robots/robot.py", line 1536, in get_proprioception\n'
        'File "/official/omnigibson/robots/robot.py", line 1546, in _get_proprioception_dict\n'
        'File "/official/omnigibson/prims/entity_prim.py", line 871, in get_joint_positions\n'
        'joint_positions = self._articulation_view.get_joint_positions().view(self.n_dof)\n'
        "AttributeError: 'NoneType' object has no attribute 'view'\n")


def test_articulation_failure_is_not_a_nan_pose_or_model_failure():
    text=articulation_failure_fixture();r=m.execution_error(text)
    assert r['kind']=='simulator_articulation_unavailable' and r['scene_slot'] is None
    assert r['failed_slot_known'] is False and r['official_score_available'] is False
    assert r['invalid_query_messages']==1 and m.physics_error(text) is None


@pytest.mark.parametrize('bad',['no_physx','no_articulation_warning','foreign_stack','wrong_call',
    'other_exception','extra_error','cuda','duplicate_stack','warning_after_failure'])
def test_articulation_diagnosis_refuses_unrelated_or_incomplete_evidence(bad):
    text=articulation_failure_fixture()
    if bad=='no_physx':text=text.replace('unitDir is not valid.','another warning')
    elif bad=='no_articulation_warning':text=text.replace('Articulation needs to be initialized.','')
    elif bad=='foreign_stack':text=text.replace('omnigibson/prims/entity_prim.py','other/model.py')
    elif bad=='wrong_call':text=text.replace('self._articulation_view.get_joint_positions()','something_else()')
    elif bad=='other_exception':text=text.replace("attribute 'view'","attribute 'shape'")
    elif bad=='extra_error':text+='ValueError: foreign session\n'
    elif bad=='cuda':text+='CUDA out of memory\n'
    elif bad=='duplicate_stack':text+=text
    else:
        text=text.replace('Articulation needs to be initialized.\n','')+'Articulation needs to be initialized.\n'
    assert m.execution_error(text) is None


def test_freeze_articulation_batch_excludes_failed_instances_without_faking_metrics(tmp_path,monkeypatch):
    prior,seal=stopped_continuation(tmp_path,monkeypatch)
    (prior/'workers/gpu_5/task5-n4-i0.log').write_text(articulation_failure_fixture())
    result=m.freeze_continuation(prior,seal)
    rows=[r for r in result['execution_errors'] if r['task']=='task5']
    assert len(rows)==4 and all(r['rerun_allowed'] is False for r in rows)
    assert all(r['diagnosis']['kind']=='simulator_articulation_unavailable' for r in rows)
    assert all(r['task']!='task5' for r in result['administrative_interruptions'])
    original=m.read(tmp_path/'original/manifest.json')
    parts=[p for g in m.remaining_parts(original,result) for p in g['parts'] if p['task']=='task5']
    assert [i for p in parts for i in p['indices']]==list(range(4,10))


def planner_rejection_fixture(path):
    path.mkdir(parents=True, exist_ok=True)
    message='planner generation violates causal K=3 memory recurrence'
    (path/'status.json').write_text(json.dumps(dict(status='failed',error=repr(ValueError(message)))))
    (path.parent/(path.name+'.log')).write_text(
        'Traceback (most recent call last):\n'
        'File "/frozen/scripts/eval/memlite_final/batch_runtime.py", line 39, in plan_many\n'
        'File "/frozen/src/g05/models/g05/g05_policy_memlite_planner_outcome.py", line 388, in _parse_target_free_planner_event\n'
        'ValueError: '+message+'\n')
    return ('Traceback (most recent call last):\n'
        'File "/frozen/scripts/eval/memlite_final/client.py", line 44, in rpc\n'
        'RuntimeError: Model service: '+repr(ValueError(message))+'\n')


def test_planner_rejection_is_not_physics_or_an_administrative_retry(tmp_path):
    model=tmp_path/'model-000';text=planner_rejection_fixture(model)
    r=m.execution_error(text,model)
    assert r['kind']=='planner_memory_recurrence_rejected' and r['scene_slot'] is None
    assert not r['failed_slot_known'] and not r['rejected_text_saved'] and not r['invalid_proposal_executed']
    assert not r['official_score_available'] and m.physics_error(text) is None
    assert r['model_status']==m.binding(model/'status.json')
    assert r['model_log']==m.binding(tmp_path/'model-000.log')


@pytest.mark.parametrize('bad',['no_model','missing_log','ready','wrong_error','extra_model_error',
    'extra_client_error','wrong_stack','foreign_client','cuda','physics_mix'])
def test_planner_rejection_requires_exact_paired_evidence(tmp_path,bad):
    model=tmp_path/'model-000';text=planner_rejection_fixture(model)
    if bad=='no_model':model=None
    elif bad=='missing_log':(tmp_path/'model-000.log').unlink()
    elif bad in ('ready','wrong_error'):
        s=m.read(model/'status.json');s['status' if bad=='ready' else 'error']='unknown'
        (model/'status.json').write_text(json.dumps(s))
    elif bad=='extra_model_error':
        with (tmp_path/'model-000.log').open('a') as f:f.write('\nValueError: wrong session identity')
    elif bad=='extra_client_error':text+='\nRuntimeError: connection lost'
    elif bad=='wrong_stack':
        p=tmp_path/'model-000.log';p.write_text(p.read_text().replace('in _parse_target_free_planner_event','in another_function'))
    elif bad=='foreign_client':text=text.replace('memlite_final/client.py','other/client.py')
    elif bad=='cuda':text+='\nCUDA out of memory'
    elif bad=='physics_mix':text+='\nAssertionError: invalid quaternion'
    assert m.execution_error(text,model) is None


def test_freeze_planner_rejected_batch_is_preserved_and_excluded(tmp_path,monkeypatch):
    prior,seal=stopped_continuation(tmp_path,monkeypatch)
    worker=prior/'workers/gpu_5';model=worker/'model-000'
    text=planner_rejection_fixture(model)
    s=m.read(model/'status.json');s.update(config_sha256='config',manifest_sha256='pack',source_commit='actor')
    (model/'status.json').write_text(json.dumps(s))
    s=m.read(worker/'status.json');s['model_path']=str(model)
    (worker/'status.json').write_text(json.dumps(s))
    (worker/'task5-n4-i0.log').write_text(text)
    result=m.freeze_continuation(prior,seal)
    assert result['completed']==8 and len(result['execution_errors'])==8
    rejected=[r for r in result['execution_errors'] if r['task']=='task5']
    assert len(rejected)==4 and all(not r['rerun_allowed'] for r in rejected)
    assert all(r['diagnosis']['kind']=='planner_memory_recurrence_rejected' for r in rejected)
    assert all(r['task']!='task5' for r in result['administrative_interruptions'])
    original=m.read(tmp_path/'original/manifest.json')
    parts=[p for g in m.remaining_parts(original,result) for p in g['parts'] if p['task']=='task5']
    assert [i for p in parts for i in p['indices']]==list(range(4,10))


def test_freeze_retired_planner_uses_its_bound_generation_not_current_model(tmp_path,monkeypatch):
    prior,first_seal=stopped_continuation(tmp_path,monkeypatch)
    worker=prior/'workers/gpu_5';old=worker/'model-000'
    text=planner_rejection_fixture(old)
    s=m.read(old/'status.json');s.update(config_sha256='config',manifest_sha256='pack',source_commit='actor')
    (old/'status.json').write_text(json.dumps(s))
    ws=m.read(worker/'status.json');ws['model_path']=str(old)
    (worker/'status.json').write_text(json.dumps(ws))
    (worker/'task5-n4-i0.log').write_text(text)
    sealed=m.freeze_continuation(prior,first_seal)
    rows=[r for r in sealed['execution_errors'] if r['task']=='task5']
    ledger=worker/'task5-n4-i0-execution-errors.json';ledger.write_text(json.dumps(rows))
    # Another task/model is now current; its normal state is not evidence of
    # the older rejected batch. GPU1 independently failed with the same NaN.
    other=worker/'model-001';other.mkdir()
    (other/'status.json').write_text(json.dumps(dict(config_sha256='config',manifest_sha256='pack',source_commit='actor')))
    (other/'weights-final.json').write_text((old/'weights-final.json').read_text())
    (other/'events.jsonl').write_text('')
    ws.update(part='another-unstarted-part',model_path=str(other),error="InterruptedError('Administrative stop')",
        diagnosed_failed_parts=['task5-n4-i0'])
    ws['model_generations'].append(m.binding(other/'weights-final.json'))
    (worker/'status.json').write_text(json.dumps(ws))
    state=m.read(prior/'status.json');state['failed_gpus']={'1':{}}
    (prior/'status.json').write_text(json.dumps(state))
    log=prior/'workers/gpu_1/task1-n4-i0.log'
    log.write_text('Traceback (most recent call last):\nFile "rigid_dynamic_prim.py"\n'
        'AssertionError: /World/scene_1/controllable__r1pro__robot_r1/base_link '
        'orientation tensor([nan, nan, nan, nan]) is not a unit quaternion.')
    result=m.freeze_continuation(prior,tmp_path/'second-seal.json')
    assert len(result['execution_errors'])==12
    assert [r for r in result['execution_errors'] if r['task']=='task5']==rows
    rows[0]['diagnosis']['model_status']['sha256']='changed';ledger.write_text(json.dumps(rows))
    with pytest.raises(ValueError):m.freeze_continuation(prior,tmp_path/'bad-seal.json')


def test_summary_cannot_fabricate_official_scores_for_execution_errors():
    manifest,_=fixture()
    records=[r|dict(success=False,q_score={'final':0.2}) for r in manifest['cases'][:-4]]
    errors=manifest['cases'][-4:]
    summary=m.summary_from_rows(manifest['tasks'],records,errors)
    assert summary['coverage']==1000 and summary['completed']==996
    assert summary['execution_errors']==4 and summary['missing']==4
    assert summary['status']=='completed_with_execution_errors'
    assert summary['official_full_q_score'] is None and summary['official_full_sr'] is None
    assert summary['conservative_q_errors_and_missing_as_zero']==pytest.approx(.1992)
    assert summary['fabricated_official_metrics']==0
    with pytest.raises(ValueError):m.summary_from_rows(manifest['tasks'],records,errors+[records[0]])
    with pytest.raises(ValueError):m.summary_from_rows(manifest['tasks'],records,errors*2)


def test_full_official_result_requires_1000_real_results():
    manifest,_=fixture()
    records=[r|dict(success=True,q_score={'final':1.}) for r in manifest['cases']]
    summary=m.summary_from_rows(manifest['tasks'],records,[])
    assert summary['official_full_q_score']==summary['official_full_sr']==1
    assert summary['status']=='completed'
    assert m.summary_from_rows(manifest['tasks'],records[:-1],[])['status']=='running'


def test_unstarted_tail_preserves_batching_without_retrying_failed_prefix():
    part=dict(id='task-n4-i0',task='task',indices=list(range(8)),num_envs=4)
    first=dict(status='running',indices=[0,1,2,3],instance_ids=[301,302,303,304],rollout=0)
    tail=m.unstarted_tail(part,[('00',first)])
    assert tail['indices']==[4,5,6,7] and tail['num_envs']==4
    second=dict(first,indices=[4,5,6,7],instance_ids=[305,306,307,308])
    assert m.unstarted_tail(part,[('00',first),('04',second)]) is None
    for bad in ([('04',second)],[('00',first),('00',first)],
                [('00',dict(first,instance_ids=[302,303,304,305]))]):
        with pytest.raises(ValueError):m.unstarted_tail(part,bad)


def test_exact_model_verification_required():
    row=dict(weights_unchanged=True,optimizer_updates=0,observer_head_unchanged=True,
             observer_before_sha256='abc',observer_after_sha256='abc')
    assert m.exact_frozen_weights(row)
    for k,v in [('weights_unchanged',False),('optimizer_updates',1),('observer_head_unchanged',False),
                ('observer_after_sha256','changed')]:
        assert not m.exact_frozen_weights(row|{k:v})


def test_evidence_mutation_rejected(tmp_path):
    p=tmp_path/'receipt.json';p.write_text('{}')
    pinned=m.binding(p);m.check_binding(pinned)
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError):m.check_binding(pinned)


def test_sibling_physics_failure_cannot_silently_discard_early_finished_case(tmp_path):
    (tmp_path/'json').mkdir();(tmp_path/'attempts').mkdir()
    (tmp_path/'status.json').write_text(json.dumps(dict(status='running',task='radio')))
    (tmp_path/'attempts/00.json').write_text(json.dumps(dict(status='running',instance_ids=[301,302])))
    m.require_no_uncommitted_metrics(tmp_path)
    # Even an unfinished write is kept for diagnosis, not classified away.
    record=tmp_path/'json/radio_301_0.json';record.write_text('{')
    with pytest.raises(ValueError):m.require_no_uncommitted_metrics(tmp_path)
    record.write_text(json.dumps(dict(task='radio',instance_id=301,rollout_id=0,steps=17,success=True)))
    with pytest.raises(ValueError):m.require_no_uncommitted_metrics(tmp_path)


def stopped_continuation(tmp_path, monkeypatch, never_loaded=False):
    """Synthetic metadata only; real ffprobe/hash audit runs on the target."""
    def write(p, v):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(v)); return p
    def record(metric, row):
        video=metric.parent.parent/'videos'/(metric.stem+'.mp4')
        return {k:row[k] for k in ('task','instance_id','rollout_id','steps')}|dict(metrics=m.binding(metric),video=m.binding(video))
    monkeypatch.setattr(m,'frozen_source',lambda *a:None)
    monkeypatch.setattr(m,'original_record',record)
    root=tmp_path/'original';prior=tmp_path/'continuation';(prior/'parts').mkdir(parents=True)
    original,initial=fixture()
    original.update(schema='memlite_final_public100_run_v1',source_commit='actor',official_commit='official',
        config_path='/frozen/config',config_sha256='config',pack='/frozen/pack',pack_manifest_sha256='pack')
    write(root/'manifest.json',original);write(root/'accepted-preflight.json',dict(num_envs=4))
    initial.update(schema='memlite_final_diagnosed_stop_v1',prior=str(root),manifest=m.binding(root/'manifest.json'),
        source_commit='actor',completed=4,evidence=[m.binding(root/'manifest.json')],policy='never reroll',administrative_interruptions=[])
    for row in initial['records']:
        row['steps']=16
        metric=write(root/'parts/task0/json'/f'task0_{row["instance_id"]}_0.json',row)
        write(root/'parts/task0/videos'/(metric.stem+'.mp4'),{})
        row.update(record(metric,row))
    seal=write(tmp_path/'initial-seal.json',initial)
    manifest={k:original[k] for k in ('tasks','cases','config_path','config_sha256','pack','pack_manifest_sha256')}
    manifest.update(schema='memlite_final_explicit_continuation_v1',source_commit='coordinator',
        runtime_source='/frozen/actor',runtime_commit='actor',prior=str(root),prior_manifest=m.binding(root/'manifest.json'),
        seal=m.binding(seal),groups=m.remaining_parts(original,initial),inherited_completed=4,initial_execution_errors=4,
        policy_changed=False,optimizer_updates=0)
    write(prior/'manifest.json',manifest)
    write(prior/'status.json',dict(status='needs_diagnosis',finished=10,pid=98765000,worker_pids={},failed_gpus={'5':{}}))
    (prior/'STOP').touch()
    for group in manifest['groups']:
        gpu=group['gpu'];w=prior/'workers'/f'gpu_{gpu}';md=w/'model-000'
        if never_loaded and gpu==6:
            write(w/'status.json',dict(status='needs_diagnosis',pid=98765007,started=0,finished=1801,
                model_generations=[],completed_parts=[],diagnosed_failed_parts=[],
                error="TimeoutError('Scene/model loading admission timed out or stopped')"))
            (prior/'gpu_6.log').write_text('Traceback (most recent call last):\nloader = loader_lock(job)\n'
                'File "worker.py", line 35, in loader_lock\nTimeoutError: Scene/model loading admission timed out or stopped\n')
            state=m.read(prior/'status.json');state['failed_gpus']['6']={};write(prior/'status.json',state)
            continue
        weights=dict(weights_unchanged=True,observer_head_unchanged=True,optimizer_updates=0,
                     observer_before_sha256='digest',observer_after_sha256='digest')
        write(md/'weights-final.json',weights)
        write(md/'status.json',dict(config_sha256='config',manifest_sha256='pack',source_commit='actor'))
        (md/'events.jsonl').write_text('')
        ws=dict(status='needs_diagnosis',finished=10,pid=98765001+gpu,model_generations=[m.binding(md/'weights-final.json')],
            diagnosed_failed_parts=[],error="InterruptedError('Administrative stop')")
        if gpu in (1,5):
            part=group['parts'][0];out=prior/'parts'/part['id'];ws['part']=part['id']
            write(out/'status.json',dict(part,status='running',config_sha256='config',source_commit='actor',
                                         official_commit='official',optimizer_updates=0))
            if gpu==1:
                write(out/'attempts/00.json',dict(status='completed',indices=[0,1,2,3],instance_ids=[301,302,303,304],
                    rollout=0,actual_applied_controls=[16]*4,simulator_seed=0,policy_seed=17))
                for i in range(301,305):
                    metric=write(out/'json'/f'task1_{i}_0.json',dict(task='task1',instance_id=i,rollout_id=0,steps=16))
                    write(out/'videos'/(metric.stem+'.mp4'),{})
                offset=4;log='Normal output interrupted by global STOP'
            else:
                offset=0;ws['error']="RuntimeError('Unclassified failure')"
                log=('Traceback (most recent call last):\nFile "rigid_dynamic_prim.py"\n'
                    'AssertionError: /World/scene_1/controllable__r1pro__robot_r1/left_gripper_finger_link1 '
                    'orientation tensor([nan, nan, nan, nan]) is not a unit quaternion.')
            write(out/'attempts'/f'{offset:02d}.json',dict(status='running',indices=list(range(offset,offset+4)),
                instance_ids=list(range(301+offset,305+offset)),rollout=0,simulator_seed=0,policy_seed=17))
            (w/(part['id']+'.log')).write_text(log)
        write(w/'status.json',ws)
    return prior,tmp_path/'new-seal.json'


def test_freeze_second_stop_keeps_inherited_and_new_originals(tmp_path,monkeypatch):
    prior,output=stopped_continuation(tmp_path,monkeypatch)
    result=m.freeze_continuation(prior,output)
    assert result['completed']==8 and len(result['execution_errors'])==8
    assert len(result['administrative_interruptions'])==4 and result['unstarted']==980
    assert result['root_manifest']==m.binding(tmp_path/'original/manifest.json')
    assert result['previous_stop_seal']==m.binding(tmp_path/'initial-seal.json')
    assert {r['instance_id'] for r in result['administrative_interruptions']}=={305,306,307,308}
    assert all(not r['rerun_allowed'] for r in result['execution_errors'][4:])
    with pytest.raises(ValueError):m.freeze_continuation(prior,output)


@pytest.mark.parametrize('bad',['live','weights','foreign_runtime','extra_metric','unknown_error','wrong_seed','original_changed'])
def test_freeze_second_stop_rejects_unsafe_or_ambiguous_restarts(tmp_path,monkeypatch,bad):
    import os
    prior,output=stopped_continuation(tmp_path,monkeypatch)
    path=None
    if bad=='live':path=prior/'status.json';field,value='pid',os.getpid()
    elif bad=='weights':path=prior/'workers/gpu_0/model-000/weights-final.json';field,value='optimizer_updates',1
    elif bad=='foreign_runtime':path=prior/'workers/gpu_0/model-000/status.json';field,value='source_commit','another actor'
    elif bad=='extra_metric':
        path=prior/'parts/task5-n4-i0/json/early.json';path.parent.mkdir();path.write_text('{');path=None
    elif bad=='unknown_error':
        with (prior/'workers/gpu_1/task1-n4-i0.log').open('a') as f:f.write('\nTraceback (most recent call last):\nValueError: broken')
    elif bad=='wrong_seed':path=prior/'parts/task1-n4-i0/attempts/04.json';field,value='policy_seed',999
    elif bad=='original_changed':path=tmp_path/'original/manifest.json';field,value='source_commit','other'
    if path is not None:
        row=json.loads(path.read_text());row[field]=value;path.write_text(json.dumps(row))
    with pytest.raises(ValueError):m.freeze_continuation(prior,output)
    assert not output.exists()


def test_second_prepare_uses_root_schedule_not_continuation_coordinator(tmp_path,monkeypatch):
    prior,seal=stopped_continuation(tmp_path,monkeypatch)
    m.freeze_continuation(prior,seal)
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'runs').mkdir()
    def git(args,**kw):return '\n' if 'status' in args else 'new-coordinator\n'
    monkeypatch.setattr(m.subprocess,'check_output',git)
    job=tmp_path/'runs/new-run'
    manifest=m.prepare(prior,seal,'/frozen/actor',job)
    assert manifest['runtime_commit']=='actor' and manifest['source_commit']=='new-coordinator'
    assert manifest['inherited_completed']==8 and manifest['initial_execution_errors']==8
    assert manifest['administrative_restarts']==4
    assert sum(len(p['indices']) for g in manifest['groups'] for p in g['parts'])==984
    assert manifest['prior']==str(prior) and manifest['root_manifest']['path']==str(tmp_path/'original/manifest.json')


def test_independent_nan_during_global_stop_is_not_administratively_rerun(tmp_path,monkeypatch):
    prior,seal=stopped_continuation(tmp_path,monkeypatch)
    log=(prior/'workers/gpu_5/task5-n4-i0.log').read_text()
    (prior/'workers/gpu_1/task1-n4-i0.log').write_text(log)
    result=m.freeze_continuation(prior,seal)
    assert result['completed']==8 and len(result['execution_errors'])==12
    assert result['administrative_interruptions']==[]


def test_property_ui_requires_exact_nonphysics_stack_and_later_applied_steps():
    text=('2026-10-13T11:01:59Z [py stderr]: Traceback (most recent call last):\n'
          'File "/env/omni/kit/property/usd/widgets.py", line 210\n'
          'File "/env/omni/kit/window/property/window.py", line 404\n'
          "AttributeError: 'NoneType' object has no attribute 'scroll_y_max'")
    state=dict(status='running',updated=1791889768,control_steps=[1920]*4)
    attempt=dict(instance_ids=[305,306,307,308],started=1791889301)
    result=m.nonfatal_property_ui(text,state,attempt)
    assert result['last_exception_time']==1791889319
    assert result['later_applied_controls']==[1920]*4
    for bad_text,bad_state in ((text,state|dict(updated=1791889318)),(text,state|dict(control_steps=[])),
            (text,state|dict(status='failed')),(text.replace('widgets.py','physics.py'),state),
            (text+'\nRuntimeError: unknown',state),(text.replace('scroll_y_max','robot_state'),state),
            (text.replace('2026-10-13T11:01:59Z','no_time'),state)):
        assert m.nonfatal_property_ui(bad_text,bad_state,attempt) is None


def test_seal_never_admitted_worker_does_not_fabricate_model_or_episode_failure(tmp_path,monkeypatch):
    prior,seal=stopped_continuation(tmp_path,monkeypatch,never_loaded=True)
    result=m.freeze_continuation(prior,seal)
    assert result['completed']==8 and len(result['execution_errors'])==8
    assert result['model_workers_started']==7 and result['all_loaded_models_unchanged']
    assert result['all_eight_models_unchanged'] is False
    assert result['never_admitted_workers'][0]['gpu']==6
    assert result['never_admitted_workers'][0]['public_attempts_started']==0


@pytest.mark.parametrize('bad',['started_part','wrong_error','short_wait','model_path'])
def test_never_admitted_exception_requires_no_model_no_attempt_and_exact_gate(tmp_path,monkeypatch,bad):
    prior,seal=stopped_continuation(tmp_path,monkeypatch,never_loaded=True)
    path=prior/'workers/gpu_6/status.json';row=m.read(path)
    if bad=='started_part':(prior/'parts/task6-n4-i0').mkdir()
    elif bad=='wrong_error':row['error']='Out of memory'
    elif bad=='short_wait':row['finished']=100
    else:row['model_path']='/maybe-loaded'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):m.freeze_continuation(prior,seal)


def test_ready_unused_model_may_have_no_lazy_trace_but_started_one_may_not(tmp_path,monkeypatch):
    prior,seal=stopped_continuation(tmp_path,monkeypatch,never_loaded=True)
    (prior/'workers/gpu_0/model-000/events.jsonl').unlink()
    result=m.freeze_continuation(prior,seal)
    assert result['models_without_inference_events'][0]['gpu']==0
    assert result['models_without_inference_events'][0]['trace_created'] is False
    path=prior/'workers/gpu_0/model-000/status.json';row=m.read(path);row['requests']=1;path.write_text(json.dumps(row))
    with pytest.raises(ValueError):m.freeze_continuation(prior,tmp_path/'rejected-seal.json')
