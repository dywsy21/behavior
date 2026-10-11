"""Official simulator policy transport with a real per-control ACK journal."""
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from websockets.sync.client import connect

sys.path.insert(0,str(Path(__file__).resolve().parent/'memlite_sft100'))
from wire import packb, unpackb


def observable(obs, task_id):
    if int(obs['task_id'].reshape(-1)[0]) != task_id:
        raise ValueError('Wrong task entered policy transport')
    def value(suffix):
        values=[v for k,v in obs.items() if k.endswith(suffix)]
        if len(values)!=1:raise ValueError('Ambiguous observable '+suffix)
        return values[0].detach().cpu().numpy()[0]
    images={}
    for key,suffix in [('head_rgb','robot_r1:zed_link:Camera:0::rgb'),
                       ('left_wrist_rgb','robot_r1:left_realsense_link:Camera:0::rgb'),
                       ('right_wrist_rgb','robot_r1:right_realsense_link:Camera:0::rgb')]:
        raw=value(suffix)
        if raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[-1] not in (3,4):
            raise ValueError('Expected original HWC RGB(A) observation')
        images[key]=np.ascontiguousarray(raw[...,:3].transpose(2,0,1))
    proprio=np.ascontiguousarray(value('::proprio').reshape(-1),dtype=np.float32)
    if proprio.shape!=(61,) or not np.isfinite(proprio).all():raise ValueError('Expected61D proprio')
    return dict(images=images,proprio=proprio)


class FullEpisodePolicy:
    def __init__(self, *, port, task_id, case, config_sha256, output):
        self.output=Path(output);self.task_id=task_id;self.used=False;self.elapsed=0
        self.socket=connect(f'ws://127.0.0.1:{port}',max_size=64<<20,ping_timeout=None,open_timeout=30)
        hello=unpackb(self.socket.recv(timeout=900))
        if (hello['kind']!='fresh_full_episode_v1' or hello['case']!=case or
                hello['config_sha256']!=config_sha256):raise ValueError('Wrong frozen model service')
        self.identity=hello['identity'];self.actions=deque();self.pending=[];self.token=None
        self.last_action=None;self.display=None
        (self.output/'service-hello.json').write_text(json.dumps(hello,indent=2)+'\n')

    def rpc(self,op,**values):
        self.socket.send(packb(dict(op=op,identity=self.identity,**values)))
        result=unpackb(self.socket.recv(timeout=1800))
        if not result['ok']:raise RuntimeError(result['error'])
        return result['result']

    def set_action_dim(self,n):
        if n!=23:raise ValueError('Expected R1Pro23 real controls')

    def reset(self):
        if self.used:raise ValueError('One complete fresh episode, no reset/retry')
        self.rpc('begin');self.used=True

    def ack(self):
        if self.pending:
            result=self.rpc('ack',token=self.token,controls=self.pending)
            if result['control_step']!=self.elapsed:raise ValueError('Actual control clock mismatch')
            self.pending=[];self.token=None

    def forward(self,obs):
        if not self.actions:
            self.ack()
            result=self.rpc('infer',observation=observable(obs,self.task_id))
            action=np.asarray(result['action_chunk'],dtype=np.float32)
            if action.shape!=(16,23) or not np.isfinite(action).all():raise ValueError('Invalid raw23')
            self.token=result['token'];self.display=result['display']
            if self.display['control_step']!=self.elapsed:raise ValueError('Future/stale display context')
            with (self.output/'chunks.jsonl').open('a') as f:
                f.write(json.dumps(dict(token=self.token,display=self.display,actions=action.tolist()),allow_nan=False)+'\n')
            self.actions.extend(action)
        self.last_action=self.actions.popleft().copy()
        return torch.from_numpy(self.last_action[None])

    def applied(self,action,terminated,truncated):
        actual=action.detach().cpu().numpy().reshape(23)
        if not np.array_equal(actual,self.last_action):raise ValueError('Simulator action differs from offered action')
        row=dict(control_step=self.elapsed,simulator_apply_ack=True,action_executed_raw23=actual.tolist())
        self.pending.append(row);self.elapsed+=1
        with (self.output/'controls.jsonl').open('a') as f:
            f.write(json.dumps(dict(**row,token=self.token,terminated=bool(terminated),truncated=bool(truncated)),allow_nan=False)+'\n')

    def finish(self,obs):
        self.ack()
        result=self.rpc('finish',observation=observable(obs,self.task_id))
        if result['control_step']!=self.elapsed:raise ValueError('Incomplete final ACK')
        self.socket.close()
        return result
