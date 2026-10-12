from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/rl/memlite_online/code'),str(ROOT/'scripts/eval/memlite_final')]
import batch_runtime
from recovery_calibrated_observer import GraspCalibrationBinding
from test_memlite_causal_session import identity,event


def observation(marker=1):
    return dict(images={k:np.zeros((3,16,16),dtype=np.uint8) for k in
        ('head_rgb','left_wrist_rgb','right_wrist_rgb')},proprio=np.full(61,marker,dtype=np.float32))


def setup(monkeypatch):
    models=identity().models;events=[];calls=[]
    class Observer:
        def __init__(self,session):self.session=session;self.seen=[]
        def check(self,i,obs):
            self.session._check(i,self.session.feedback.control_step)
            self.seen.append((self.session.feedback.control_step,float(obs['proprio'][0])))
            return dict(members=[dict(raw_logits=[1.,0.,0.,0.])],attempt_generation=self.session.attempt_generation)
    def plans(planner,wrappers,observations,adapter):
        result=[]
        for w,obs in zip(wrappers,observations,strict=True):
            s=w.session;token,causal=s.begin_planning(s.identity,s.feedback.control_step)
            e=event(s,target='object_'+str(int(obs['proprio'][0])))
            s.stage(s.identity,token,e);goal=s.commit(s.identity,token)
            result.append(dict(event=e,goal=goal,causal_input=causal,reused=False))
        return result
    monkeypatch.setattr(batch_runtime,'plan_many',plans)
    def sample(observations,goals,steps):
        calls.append(deepcopy((goals,steps)))
        return np.stack([np.full((16,23),obs['proprio'][0]+step/1000,dtype=np.float32) for obs,step in zip(observations,steps)])
    rt=batch_runtime.CausalBatchRuntime(models=models,calibration=GraspCalibrationBinding(models,'f'*64,1.),
        planner=SimpleNamespace(models=models),adapter=SimpleNamespace(prepare=lambda *a:None),
        observer_factory=Observer,sample=sample,emit=lambda k,v:events.append((k,v)),run_id='test-final')
    return rt,events,calls


def metadata(task='turning_on_radio',prefix='batch1'):
    return [dict(task=task,split='public_test',instance_id=301+i,episode=prefix+'-'+str(i)) for i in range(2)]


def acknowledge(rt, result, lengths):
    rows={}
    for slot,n in lengths.items():
        start=rt.sessions[slot].session.feedback.control_step
        rows[slot]=dict(token=result[slot]['token'],controls=[dict(control_step=start+j,simulator_apply_ack=True,
            action_executed_raw23=result[slot]['action_chunk'][j].tolist()) for j in range(n)])
    rt.acknowledge(rows)
    return rows


def test_independent_memory_observer_action_and_partial_clock(monkeypatch):
    rt,events,calls=setup(monkeypatch);rt.begin(metadata(),'public_test')
    obs=[observation(1),observation(2)]
    result,timing=rt.infer(obs,[0,1])
    assert timing['high_batch']==2 and timing['low_batch']==2
    assert rt.sessions[0].session.previous_intent!=rt.sessions[1].session.previous_intent
    assert rt.sessions[0].observer is not rt.sessions[1].observer
    assert result[0]['token']!=result[1]['token']
    with pytest.raises(ValueError):rt.infer(obs,[0,1])
    with pytest.raises(ValueError):
        rt.acknowledge({1:dict(token=result[0]['token'],controls=[dict(control_step=0,simulator_apply_ack=True,
            action_executed_raw23=result[0]['action_chunk'][0].tolist())])})
    assert rt.sessions[1].session.feedback.control_step==0
    acknowledge(rt,result,{0:5,1:16});rt.close([0],obs)
    assert rt.sessions[0].session.feedback.control_step==5
    result2,timing=rt.infer(obs,[1])
    assert set(result2)=={1} and timing['high_batch']==0 and calls[-1][1]==[16]
    np.testing.assert_allclose(result2[1]['action_chunk'],2.016)
    assert rt.sessions[0].session.windows==[]
    assert rt.sessions[1].observer.seen[-1]==(16,2.)
    acknowledge(rt,result2,{1:3});rt.close([1],obs)
    assert rt.sessions[1].session.feedback.control_step==19


def test_reset_requires_clean_end_and_cross_task_starts_empty(monkeypatch):
    rt,_,_=setup(monkeypatch);rt.begin(metadata(),'public_test')
    with pytest.raises(ValueError):rt.begin(metadata('wash_dog_toys','batch2'),'public_test')
    obs=[observation(1),observation(2)];result,_=rt.infer(obs,[0,1]);acknowledge(rt,result,{0:1,1:1});rt.close([0,1],obs)
    with pytest.raises(ValueError):rt.begin(metadata(),'public_test')
    rt.begin(metadata('wash_dog_toys','batch2'),'public_test')
    for w in rt.sessions.values():
        assert w.session.revision==0 and w.session.previous_intent=='None'
        assert w.session.windows==[] and w.observer.seen==[]
        assert json.loads(w.session.memory)==dict(task_name='wash dog toys',issued_command_history=[],verified_world_facts=[])


def test_privileged_actor_payload_and_reactivated_slot_rejected(monkeypatch):
    rt,_,_=setup(monkeypatch);rt.begin(metadata(),'public_test')
    obs=[observation(1),observation(2)]
    with pytest.raises(ValueError):rt.infer([dict(obs[0],success=True),obs[1]],[0,1])
    result,_=rt.infer(obs,[0,1]);acknowledge(rt,result,{0:1,1:16});rt.close([0],obs)
    with pytest.raises(ValueError):rt.infer(obs,[0])
