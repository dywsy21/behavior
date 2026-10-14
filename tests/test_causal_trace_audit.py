from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
from audit_causal_trace import audit,digest


def fixture():
    identity=dict(task='radio',instance=1,episode='e',session='s',models=dict(planner='p'))
    memory=json.dumps(dict(issued_command_history=[],task_name='radio',verified_world_facts=[]))
    goal=dict(task='radio',parent_goal='grasp radio',semantic_bundle='bundle')
    raw=[[float(j)/100 for j in range(23)] for _ in range(16)]
    controls=[dict(control_step=0,simulator_apply_ack=True,action_executed_raw23=raw[0])]
    values=[dict(kind='begin',memory=memory,initial_control_step=0,teacher_prefix_controls=0),
        dict(kind='plan',result=dict(control_step=0,revision=1,goal=goal,physical_success_asserted=False,
            causal_input=dict(task_name='radio',memory=memory))),
        dict(kind='observer',control_step=0,revision=1),
        dict(kind='action',token='token',raw_actions=raw,display=dict(control_step=0,revision=1,
            goal=goal,attempt_generation=1,observer=dict(control_step=0,attempt_generation=1))),
        dict(kind='ack',token='token',consumed_controls=1,unused_controls=15,control_step=1,ack_sha256=digest(controls)),
        dict(kind='close',control_step=1,episode='e',optimizer_updates=0)]
    return [dict(deepcopy(v),slot=0,identity=deepcopy(identity)) for v in values]


def test_partial_final_chunk_and_fresh_task_session_pass():
    assert audit(fixture())['applied_controls']==1


def test_stopped_trace_must_explicitly_disclose_unclosed_episodes():
    rows=fixture()[:-1]
    with pytest.raises(ValueError,match='Incomplete trace'):audit(rows)
    result=audit(rows,allow_incomplete=True)
    assert result['status']=='incomplete_causal_trace_audit'
    assert result['episodes']==1 and result['closed_episodes']==0
    assert result['unclosed_episodes'][0]['applied_controls']==1
    assert result['proves_full_task_success'] is False
    with pytest.raises(ValueError,match='unacknowledged'):audit(rows[:-1],allow_incomplete=True)
    rows[3]['raw_actions'][0][0]=999
    with pytest.raises(ValueError):audit(rows,allow_incomplete=True)


@pytest.mark.parametrize('bad',['identity','memory','low_goal','observer','raw_action','unfinished'])
def test_actual_cross_session_and_action_corruption_rejected(bad):
    r=fixture()
    if bad=='identity':r[3]['identity']['episode']='other'
    elif bad=='memory':r[0]['memory']=json.dumps(dict(issued_command_history=['old'],task_name='radio',verified_world_facts=[]))
    elif bad=='low_goal':r[3]['display']['goal']['parent_goal']='another intent'
    elif bad=='observer':r[3]['display']['observer']['attempt_generation']=2
    elif bad=='raw_action':r[3]['raw_actions'][0][0]=999
    else:r.pop()
    with pytest.raises(ValueError):audit(r)
