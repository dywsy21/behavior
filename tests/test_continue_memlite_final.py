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
    for bad in ('CUDA out of memory','connection lost','model malformed JSON',text.replace('nan','0.'),text+text):
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
