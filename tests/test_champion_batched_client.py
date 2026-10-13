import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts/eval'))
from champion_batched_client import ChampionBatchedPolicy,packb,unpackb
from behavior_champion.official import CAMERAS,PROPRIO


class Socket:
    def __init__(self,slot):self.slot=slot;self.requests=[];self.epoch=0;self.calls=0;self.closed=False
    def send(self,data):
        v=unpackb(data);self.requests.append(v)
        if v=={'reset':True}:self.calls=0;self.epoch+=1
    def recv(self,timeout):
        assert 0<timeout<=600
        if not self.requests:
            return packb(dict(policy='champion2026',environment_batch_size=1,action_chunk_size=20,
                action_dimension=23,config_sha256='cfg',source_commit='commit',clock_kind='policy_delivery',allowed_task_ids=[0,1]))
        self.calls+=1
        chunk=np.full((1,20,23),self.slot+100*self.calls,dtype=np.float32)
        return packb(dict(action_chunk=chunk,action=chunk[:,0].copy()))
    def close(self):self.closed=True


def setup_policy():
    sockets=[Socket(i) for i in range(4)];it=iter(sockets)
    policy=ChampionBatchedPolicy(port=1,num_envs=4,task_id=0,config_sha256='cfg',source_commit='commit',socket_factory=lambda:next(it))
    policy.set_timeout(256);policy.reset()
    obs={key:torch.zeros(4,4,5,3,dtype=torch.uint8) for key in CAMERAS.values()}
    obs.update({PROPRIO:torch.zeros(4,61), 'task_id':torch.zeros(4,1,dtype=torch.int64)})
    return policy,sockets,obs


def test_nonrenumbered_shrinking_slots_and_independent_reset():
    p,sockets,obs=setup_policy()
    for step in range(41):
        active=[0,1,2,3] if step<7 else [0,2]
        p.set_active(active);actions=p.forward(obs)
        for i in active:assert actions[i,0]==i+100*(step//20+1)
        for i in set(range(4))-set(active):assert actions[i,0]==0
        p.applied(active)
    result=p.finish()
    assert result['actual_applied_controls']==[41,7,41,7]
    assert result['unused_delivered_controls']=={'0':19,'1':13,'2':19,'3':13}
    assert [s.calls for s in sockets]==[3,1,3,1]
    assert result['physical_ack_sent'] is False
    p.reset();a=p.forward(obs);p.applied([0,1,2,3])
    assert a[:,0].tolist()==[100,101,102,103]
    p.finish();p.close();assert all(s.closed for s in sockets)


def test_no_unapplied_action_or_slot_revival():
    p,_,obs=setup_policy();p.forward(obs)
    with pytest.raises(ValueError):p.forward(obs)
    with pytest.raises(ValueError):p.finish()
    with pytest.raises(ValueError):p.applied([0,1])
    p.applied([0,1,2,3]);p.set_active([2])
    with pytest.raises(ValueError):p.set_active([0,2])
    p.forward(obs);p.applied([2]);p.finish()


def test_foreign_task_privilege_and_model_identity_rejected():
    p,_,obs=setup_policy();obs['reward']=torch.ones(4)
    with pytest.raises(ValueError):p.forward(obs)
    del obs['reward'];obs['task_id'][0]=1
    with pytest.raises(ValueError):p.forward(obs)
    socket=Socket(0)
    with pytest.raises(ValueError):
        ChampionBatchedPolicy(port=1,num_envs=1,task_id=0,config_sha256='wrong',source_commit='commit',socket_factory=lambda:socket)
    assert socket.closed


def test_no_rgb_cpu_copy_during_open_loop_chunk_replay(monkeypatch):
    p,_,obs=setup_policy();calls=[];original=torch.Tensor.cpu
    def counted(t,*a,**kw):
        if t.ndim==4:calls.append(tuple(t.shape))
        return original(t,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'cpu',counted)
    for _ in range(20):p.forward(obs);p.applied([0,1,2,3])
    assert len(calls)==12
    p.forward(obs);p.applied([0,1,2,3])
    assert len(calls)==24
    p.finish()
