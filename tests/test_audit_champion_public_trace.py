from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('champion_public_audit',
    Path(__file__).resolve().parents[1]/'scripts/eval/audit_champion_public_trace.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def journal():
    """Real session/connection bookkeeping with a synthetic (non-model) predictor."""
    spec=importlib.util.spec_from_file_location('champion_official_fixture',
        Path(__file__).with_name('test_champion_official.py'))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    from behavior_champion.official import OfficialConnection
    a,b=[OfficialConnection(helpers.factory,connection_id=c) for c in ('a','b')]
    rows=[]
    for c in (a,b):
        rows.extend([dict(kind='connected',connection=c.connection_id),
            dict(kind='reset',connection=c.connection_id,previous=c.reset())])
    for _ in range(6):
        for c,task in ((a,0),(b,1)):
            c.prepare(helpers.observation(task,20))
            rows.append(dict(kind='delivered',connection=c.connection_id,**c.delivered()))
    rows.append(dict(kind='reset',connection='a',previous=a.reset()))
    for _ in range(3):
        a.prepare(helpers.observation(2,20))
        rows.append(dict(kind='delivered',connection='a',**a.delivered()))
    for c in (a,b):rows.append(dict(kind='closed',connection=c.connection_id,previous=c.reset()))
    return rows


def encoded(rows):
    return [(json.dumps(r)+'\n').encode() for r in rows]


def test_real_connection_interleaving_reset_and_task_change_audit():
    rows=journal();result=m.audit_lines(encoded(rows),allowed_tasks=[0,1,2],require_closed=True)
    assert result['connections']==2 and result['episodes']==result['closed_episodes']==3
    assert result['active_connections']==0 and result['deliveries']==15
    assert result['delivered_controls']==300 and result['physically_confirmed_controls'] is None
    assert result['task_ids']==[0,1,2]


@pytest.mark.parametrize('bad',['foreign_task','foreign_episode','cross_history','clock','inpainting',
    'vote','probability','nan','layout','physical','reset_receipt','missing_reset','duplicate_connection',
    'post_close','connection_error','missing_display','wrong_digest','legacy_change'])
def test_corrupted_or_cross_session_protocol_is_rejected(bad):
    rows=deepcopy(journal());d=rows[4]['generated_display']
    if bad=='foreign_task':rows[4]['task']=99
    elif bad=='foreign_episode':rows[4]['episode']='b/reset-1/task-0'
    elif bad=='cross_history':d['input_stage']=1
    elif bad=='clock':rows[4]['delivered_controls']=40
    elif bad=='inpainting':d['inpainting_input_steps']=4
    elif bad=='vote':d['stage_votes']=[4,4,4]
    elif bad=='probability':d['stage_probability']=0.
    elif bad=='nan':d['stage_probabilities'][0]=float('nan')
    elif bad=='layout':d['stage_probabilities']=[1.]
    elif bad=='physical':rows[4]['physically_confirmed_controls']=20
    elif bad=='reset_receipt':rows[16]['previous']['delivered_controls']=999
    elif bad=='missing_reset':rows.pop(1)
    elif bad=='duplicate_connection':rows.insert(1,rows[0])
    elif bad=='post_close':rows.append(rows[4])
    elif bad=='connection_error':rows.insert(4,dict(kind='connection_error',connection='a',error='broken'))
    elif bad=='missing_display':rows[4]['generated_display']=None
    elif bad=='wrong_digest':rows[4]['action_sha256']='bad'
    else:rows[6]['generated_display']['legacy_corrections']=True
    with pytest.raises(ValueError):m.audit_lines(encoded(rows),allowed_tasks=[0,1,2])


def test_live_prefix_never_invents_close_or_accepts_partial_terminal_line(tmp_path):
    path=tmp_path/'trace.jsonl';rows=journal()
    prefix=b''.join(encoded(rows[:-2]));path.write_bytes(prefix+b'{"kind":')
    report=m.audit_file(path,allowed_tasks=[0,1,2])
    assert report['prefix_bytes']==len(prefix) and report['ignored_incomplete_tail_bytes']==8
    assert report['closed_episodes']==1 and report['active_connections']==2
    with pytest.raises(ValueError):m.audit_file(path,require_closed=True)
    path.write_bytes(b''.join(encoded(rows))+b'{')
    with pytest.raises(ValueError,match='incomplete final line'):m.audit_file(path,require_closed=True)


def test_empty_or_never_connected_trace_is_not_proof(tmp_path):
    with pytest.raises(ValueError):m.audit_lines([])
    with pytest.raises(ValueError):m.audit_lines(encoded([dict(kind='reset',connection='a',previous=None)]))
    path=tmp_path/'empty';path.write_bytes(b'')
    with pytest.raises(ValueError):m.audit_file(path)
