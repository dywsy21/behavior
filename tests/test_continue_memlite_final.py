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


def stopped_continuation(tmp_path, monkeypatch):
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
