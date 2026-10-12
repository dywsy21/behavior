from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/eval/memlite_final'),str(ROOT/'scripts/eval'),str(ROOT/'scripts/eval/memlite_sft100')]
from client import AcknowledgedPolicy
from lifecycle import AppliedBatchClock


def obs():
    value={'task_id':torch.tensor([0,0]),'x::proprio':torch.zeros(2,61),
           'privileged_object_pose':torch.ones(2,3)}
    for cam in ('zed_link','left_realsense_link','right_realsense_link'):
        value[f'robot_r1:{cam}:Camera:0::rgb']=torch.zeros(2,16,16,4,dtype=torch.uint8)
    return value


def policy():
    p=object.__new__(AcknowledgedPolicy)
    p.task_id=0;p.clock=AppliedBatchClock(2);p.active=[0,1];p.closed=False;p.limit=256;p.deadline=None
    p.chunks={};p.receipts={};p.actions=None;calls=[]
    def rpc(op,**values):
        calls.append((op,values))
        for value in values['observations']:
            assert set(value)=={'images','proprio'}
        if op=='finish':return dict(steps=list(p.clock.steps))
        return dict(rows=[dict(slot=i,token=f'call{len(calls)}-slot{i}',
            action_chunk=np.full((16,23),i+1,dtype=np.float32)) for i in values['active']])
    p.rpc=rpc
    return p,calls


def test_two_env_chunk_replay_partial_close_and_no_privileged_inputs():
    p,calls=policy()
    for step in range(19):
        active=[0,1] if step<5 else [1]
        p.set_active(active);actions=p.forward(obs())
        assert tuple(actions.shape)==(2,23)
        np.testing.assert_array_equal(actions[1],2)
        if step>=5:np.testing.assert_array_equal(actions[0],0)
        p.applied(active)
    assert len(calls)==2
    acks=calls[1][1]['acks']
    assert [(r['slot'],len(r['controls'])) for r in acks]==[(0,5),(1,16)]
    assert calls[1][1]['active']==[1]
    assert p.finish(obs())==dict(steps=[5,19])
    assert len(calls[-1][1]['acks'][0]['controls'])==3


def test_inference_does_not_advance_the_applied_clock():
    p,_=policy();p.set_active([0,1]);p.forward(obs())
    assert p.clock.steps==[0,0]
    with pytest.raises(ValueError):p.set_active([0,1])
    with pytest.raises(ValueError):p.finish(obs())
