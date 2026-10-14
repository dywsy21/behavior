import json
import os
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
import pending_mem_upgrade as m


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path,monkeypatch):
    old,new,champ=[tmp_path/n for n in ('old','new','champ')]
    for job in (old,new):
        (job/'parts').mkdir(parents=True);(job/'workers').mkdir()
    part=dict(task='task',task_id=0,num_envs=4,indices=list(range(8)),id='task-n4-i0')
    original={k:'same' for k in m.IDENTITY}
    original['groups']=[dict(gpu=0,parts=[part])]
    newer=original|dict(groups=[dict(gpu=0,parts=[part|dict(indices=list(range(4))),
        part|dict(indices=list(range(4,8)),id='task-n4-i4')])],isolate_physical_batches=True,continue_on_batch_errors=True)
    save(old/'manifest.json',original);save(new/'manifest.json',newer);save(champ/'manifest.json',{})
    auth=tmp_path/'auth.json';save(auth,{})
    queue=old/'queue-old';now=time.time()
    before=dict(status='waiting_for_champion_continuation',pid=99999000,started=now-100,
        phases=[],mem_job=str(old),champion_job=str(champ),bindings=[m.m.binding(old/'manifest.json'),
        m.m.binding(champ/'manifest.json'),m.m.binding(auth)])
    stop=before|dict(status='needs_diagnosis',finished=now+1,error="InterruptedError('Requested queue stop')")
    save(queue/'status.json',stop);(queue/'STOP').touch()
    receipt=tmp_path/'upgrade.json'
    value=dict(schema='pending_mem_isolation_upgrade_v1',created=now-1,old_queue=str(queue),old_mem=str(old),
        new_mem=str(new),champion=str(champ),authorization=str(auth),old_source='/old/source',new_source='/new/source',
        before=before,policy_changed=False,champion_stop_requested=False,scored_cases_replayed=0,
        bindings=[m.m.binding(old/'manifest.json'),m.m.binding(new/'manifest.json'),
                  m.m.binding(champ/'manifest.json'),m.m.binding(auth)])
    save(receipt,value)
    monkeypatch.setattr(m.m,'registration',lambda job,**kw:(job,m.m.read(Path(job)/'manifest.json')))
    return receipt,queue,new,champ,auth


@pytest.mark.parametrize('bad',[None,'started','weights','order','new_binding','phase','live','error','stop_champion'])
def test_only_unused_identical_remainder_and_stopped_waiter_can_be_replaced(tmp_path,monkeypatch,bad):
    args=fixture(tmp_path,monkeypatch);receipt,queue,new,champ,auth=args
    if bad=='started':save(new/'status.json',{'status':'running'})
    elif bad in ('weights','order'):
        value=m.m.read(new/'manifest.json')
        if bad=='weights':value['pack']='different'
        else:value['groups'][0]['parts'].reverse()
        with pytest.raises(ValueError):m.same_remainder(m.m.read(queue.parent/'manifest.json'),value)
        return
    elif bad=='new_binding':save(new/'manifest.json',{})
    elif bad in ('phase','live','error'):
        path=queue/'status.json';value=m.m.read(path)
        if bad=='phase':value['phases']=[{'name':'remaining_mem'}]
        elif bad=='live':
            value['pid']=os.getpid();v=m.m.read(receipt);v['before']['pid']=os.getpid();save(receipt,v)
        else:value['error']='Unknown'
        save(path,value)
    elif bad=='stop_champion':
        v=m.m.read(receipt);v['champion_stop_requested']=True;save(receipt,v)
    if bad:
        with pytest.raises(ValueError):m.verify(*args)
    else:
        assert m.verify(*args)['new_mem']==str(new)
        assert not (champ/'STOP').exists() and not (queue.parent/'STOP').exists()


def test_request_writes_only_waiting_queue_stop_not_champion_or_mem_stop(tmp_path,monkeypatch):
    receipt,queue,new,champ,auth=fixture(tmp_path,monkeypatch)
    # Build a different fresh waiting queue; never delete or overwrite old evidence.
    fresh=queue.parent/'queue-fresh';before=m.m.read(receipt)['before']|dict(pid=os.getpid())
    save(fresh/'status.json',before);save(champ/'status.json',dict(status='running'))
    output=tmp_path/'new-upgrade.json'
    m.request(fresh,new,champ,auth,'/old/source','/new/source',output)
    assert (fresh/'STOP').exists() and output.exists()
    assert not (champ/'STOP').exists() and not (new/'STOP').exists() and not (queue.parent/'STOP').exists()
