"""One original-protocol connection per fixed simulator slot.

The champion server shares frozen weights but never stage/RNG/inpainting state.
Only this client records physical applied counts. The server's counters remain
explicitly policy-delivery counters; no fabricated physical ACK is sent.
"""
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[1]/'src'),str(HERE/'memlite_sft100')]
from behavior_champion.official import parse_observation, CHUNK_KEY, CAMERAS, PROPRIO
from wire import packb, unpackb


class ChampionBatchedPolicy:
    def __init__(self, *, port, num_envs, task_id, config_sha256, source_commit,
                 socket_factory=None):
        if num_envs not in (1,2,4) or type(task_id) is not int or not 0<=task_id<100:
            raise ValueError('Explicit fixed R1Pro slot count and task required')
        if socket_factory is None:
            from websockets.sync.client import connect
            socket_factory=lambda:connect(f'ws://127.0.0.1:{port}',max_size=64<<20,
                compression=None,ping_interval=None,ping_timeout=None,open_timeout=30)
        self.sockets=[]
        try:
            for _ in range(num_envs):
                socket=socket_factory();self.sockets.append(socket)
                meta=unpackb(socket.recv(timeout=600))
                if (meta.get('policy')!='champion2026' or meta.get('environment_batch_size')!=1
                        or meta.get('action_chunk_size')!=20 or meta.get('action_dimension')!=23
                        or meta.get('config_sha256')!=config_sha256 or meta.get('source_commit')!=source_commit
                        or meta.get('clock_kind')!='policy_delivery' or task_id not in meta.get('allowed_task_ids',[])):
                    raise ValueError('Foreign champion model/protocol binding')
        except BaseException:
            self.close();raise
        self.count,self.task_id=num_envs,task_id
        self.closed=True;self.limit=None;self.deadline=None
        self.steps=[0]*num_envs;self.active=set();self.issued=None
        self.chunks={};self.positions={}

    def set_action_dim(self,n):
        if n!=23:raise ValueError('Native R1Pro23 controls required')

    def set_timeout(self,n):
        if type(n) is not int or n<=0:raise ValueError('Official positive max_steps required')
        self.limit=n

    def reset(self):
        if not self.closed:raise ValueError('Previous official episode must finish before reset')
        for socket in self.sockets:socket.send(packb({'reset':True}))
        self.steps=[0]*self.count;self.active=set(range(self.count));self.issued=None
        self.chunks={};self.positions={};self.deadline=None;self.closed=False

    def set_active(self,indices):
        ids=list(indices);active=set(ids)
        if (self.closed or self.issued is not None or not active or len(active)!=len(ids)
                or any(type(i) is not int for i in ids) or not active<=self.active):
            raise ValueError('Foreign, duplicate, revived or unacknowledged slot')
        self.active=active

    def forward(self,obs):
        if self.closed or self.limit is None or self.issued is not None:
            raise ValueError('Ready episode and actual prior step required')
        if self.deadline is None:self.deadline=time.monotonic()+self.limit
        if time.monotonic()>=self.deadline:raise TimeoutError('Official episode wall timeout')
        required={'task_id',PROPRIO,*CAMERAS.values()}
        # The pinned evaluator adds camera extrinsics in _preprocess_obs.
        # They are not an actor input: discard this exact known transport
        # field before touching values, slicing rows or sending a request.
        # Continue rejecting every other unknown/privileged field.
        if not required<=set(obs) or set(obs)-required-{'robot_r1::cam_rel_poses'}:
            raise ValueError('Only original RGB/proprio/task fields are allowed; '
                f'missing={sorted(required-set(obs))}, extra={sorted(set(obs)-required)}')
        obs={k:obs[k] for k in required}
        if any(v.ndim<1 or v.shape[0]!=self.count for v in obs.values()):
            raise ValueError('Observation batch lost original slot dimensions')
        task_ids=obs['task_id']
        task_ids=task_ids.detach().cpu().numpy() if isinstance(task_ids,torch.Tensor) else np.asarray(task_ids)
        if (task_ids.shape not in ((self.count,),(self.count,1)) or task_ids.dtype.kind not in 'iu'
                or not np.all(task_ids==self.task_id)):
            raise ValueError('Task changed without a fresh policy session')
        actions=np.zeros((self.count,23),dtype=np.float32)
        for slot in sorted(self.active):
            if slot not in self.chunks or self.positions[slot]==20:
                # Do not copy full-resolution GPU RGB to CPU during the 19
                # open-loop replay steps: no model consumes those frames.
                request={k:(v[slot:slot+1].detach().cpu().numpy() if isinstance(v,torch.Tensor)
                    else np.asarray(v[slot:slot+1])) for k,v in obs.items()}
                request[CHUNK_KEY]=20
                task,_,_=parse_observation(request)
                if task!=self.task_id:raise ValueError('Foreign actor observation')
                remaining=min(600,self.deadline-time.monotonic())
                if remaining<=0:raise TimeoutError('Official query exceeds remaining wall budget')
                self.sockets[slot].send(packb(request))
                response=unpackb(self.sockets[slot].recv(timeout=remaining))
                chunk=np.asarray(response['action_chunk'])
                first=np.asarray(response['action'])
                if (chunk.shape!=(1,20,23) or first.shape!=(1,23)
                        or not np.isfinite(chunk).all() or chunk.dtype!=np.float32
                        or not np.array_equal(first,chunk[:,0])):
                    raise ValueError('Malformed/changed exact champion chunk20')
                self.chunks[slot]=chunk[0].copy();self.positions[slot]=0
            actions[slot]=self.chunks[slot][self.positions[slot]]
        self.issued=tuple(sorted(self.active))
        return torch.from_numpy(actions)

    def applied(self,indices):
        if self.issued is None or tuple(sorted(indices))!=self.issued:
            raise ValueError('Official applied callback differs from issued slots')
        for slot in self.issued:
            self.positions[slot]+=1;self.steps[slot]+=1
        self.issued=None

    def finish(self):
        if self.closed or self.issued is not None:
            raise ValueError('Cannot finish an unacknowledged/closed physical episode')
        self.closed=True
        return dict(actual_applied_controls=list(self.steps),
            unused_delivered_controls={str(k):20-self.positions[k] for k in self.chunks},
            server_clock_kind='policy_delivery',physical_ack_sent=False)

    def close(self):
        for socket in self.sockets:socket.close()
