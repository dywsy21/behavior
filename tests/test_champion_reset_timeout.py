import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
import champion_reset_timeout as m


@pytest.fixture(scope='module')
def terminal_rows():
    spec=importlib.util.spec_from_file_location('reset_helpers',Path(__file__).with_name('test_champion_official.py'))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    from behavior_champion.official import OfficialConnection
    connections=[OfficialConnection(helpers.factory,connection_id=str(i)) for i in range(4)]
    rows=[]
    for c in connections:
        rows += [dict(time=1,kind='connected',connection=c.connection_id),
                 dict(time=2,kind='reset',connection=c.connection_id,previous=c.reset())]
    for step in range(626):
        for c in connections:
            c.prepare(helpers.observation(39,20))
            rows.append(dict(time=3+step,kind='delivered',connection=c.connection_id,**c.delivered()))
    for c in connections:
        rows += [dict(time=900,kind='connection_error',connection=c.connection_id,
                      error='ConnectionClosedError(None, None, None)'),
                 dict(time=901,kind='closed',connection=c.connection_id,previous=c.reset())]
    return rows


@pytest.mark.parametrize('bad',[None,'new_delivery','new_reset','unknown_error','early_disconnect',
    'duplicate_disconnect','wrong_previous','missing_close','partial_line','previous_physics'])
def test_only_exact_shutdown_tail_can_be_projected_without_modifying_original(terminal_rows,bad):
    rows=json.loads(json.dumps(terminal_rows));previous={'actual_applied_controls':[12519]*4}
    if bad=='new_delivery':rows[-9]['time']=801
    elif bad=='new_reset':rows.insert(-8,dict(time=801,kind='reset',connection='0',previous=None))
    elif bad=='unknown_error':rows[-8]['error']='ValueError: unrelated model error'
    elif bad=='early_disconnect':rows[-8]['time']=850
    elif bad=='duplicate_disconnect':rows[-6]['connection']='0'
    elif bad=='wrong_previous':rows[-1]['previous']['delivered_controls']=12540
    elif bad=='missing_close':rows.pop()
    elif bad=='previous_physics':previous['actual_applied_controls'][0]=12000
    raw=b''.join((json.dumps(row)+'\n').encode() for row in rows)
    if bad=='partial_line':raw=raw[:-1]
    args=dict(previous=previous,current={'started':800},stopped_at=910,sampled_at=880)
    if bad:
        with pytest.raises(ValueError):m.terminal_protocol(raw,**args)
    else:
        proof=m.terminal_protocol(raw,**args)
        assert proof['next_batch_policy_deliveries']==0 and proof['original_bytes']==len(raw)
        assert proof['terminal_protocol_projection']['active_connections']==0
        assert len(proof['separately_verified_disconnects'])==4


def test_certificate_verification_recomputes_and_rejects_tampering(tmp_path,monkeypatch):
    import continue_champion_final as c
    observation=tmp_path/'obs.json';observation.write_text('{}')
    value=dict(job='/original',observation=c.binding(observation),proof={'actions':0})
    path=tmp_path/'certificate.json';path.write_text(json.dumps(value))
    monkeypatch.setattr(m,'certify',lambda *a:value)
    assert m.verify_certificate(path)==value
    changed=value|dict(proof={'actions':1});path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):m.verify_certificate(path)
