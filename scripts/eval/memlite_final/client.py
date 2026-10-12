"""Chunk transport with actual per-environment post-step ACKs.

Only RGB/proprio and declared task identity enter the model service. Active
indices and applied-action receipts are lifecycle control, not actor features.
The evaluator's original step/metrics/video functions remain unmodified.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from websockets.sync.client import connect

from full_episode_client import observable
from lifecycle import AppliedBatchClock, validate_episode_metadata
from wire import packb, unpackb


class AcknowledgedPolicy:
    def __init__(self, *, port, task_id, config_sha256, metadata_path, output):
        self.task_id, self.config_sha256 = task_id, config_sha256
        self.metadata_path, self.output = Path(metadata_path), Path(output)
        self.socket=connect(f'ws://127.0.0.1:{port}',max_size=256<<20,
            ping_interval=None,ping_timeout=None,open_timeout=30)
        hello=unpackb(self.socket.recv(timeout=600))
        if hello.get('kind')!='memlite_final_ack_batch_v1' or hello.get('config_sha256')!=config_sha256:
            raise ValueError('Wrong model service identity')
        self.clock=None;self.chunks={};self.receipts={};self.actions=None
        self.active=None;self.closed=True;self.limit=None;self.deadline=None

    def set_action_dim(self, n):
        if n!=23:raise ValueError('R1Pro real23 required')

    def set_timeout(self, max_steps):
        if type(max_steps) is not int or max_steps<=0:raise ValueError('Official timeout required')
        self.limit=max_steps

    def rpc(self, op, **values):
        remaining=600 if self.deadline is None else min(600,self.deadline-time.monotonic())
        if remaining<=0:raise TimeoutError('Official one-second-per-step episode wall timeout')
        self.socket.send(packb(dict(op=op,config_sha256=self.config_sha256,**values)))
        result=unpackb(self.socket.recv(timeout=remaining))
        if not result['ok']:raise RuntimeError('Model service: '+result['error'])
        return result['result']

    def reset(self):
        if not self.closed:raise ValueError('Do not reset before terminal partial chunks are acknowledged')
        value=json.loads(self.metadata_path.read_text())
        self.metadata=validate_episode_metadata(value['episodes'],allowed_mode=value['mode'])
        self.clock=AppliedBatchClock(len(self.metadata));self.chunks={};self.receipts={}
        self.active=list(range(len(self.metadata)));self.actions=None;self.deadline=None
        self.rpc('begin',metadata=self.metadata,mode=value['mode'])
        self.closed=False

    def set_active(self, indices):
        self.clock.select_active(indices)
        self.active=list(indices)

    def observations(self, obs):
        if np.asarray(obs['task_id'].detach().cpu()).size!=self.clock.count:
            raise ValueError('Observation slot count changed')
        return [observable({k:v[i:i+1] for k,v in obs.items()},self.task_id) for i in range(self.clock.count)]

    def _ack_rows(self):
        pending=self.clock.drain() if self.clock.pending else {}
        result=[]
        for slot,row in pending.items():
            actual=self.receipts.pop(slot)
            if len(actual)!=row['controls'] or actual[0]['control_step']!=row['start']:
                raise ValueError('Applied journal and chunk clock differ')
            result.append(dict(slot=slot,token=row['token'],controls=actual))
        return result

    def forward(self, obs):
        if self.closed or self.limit is None:raise ValueError('Begin and official timeout before actions')
        if self.deadline is None:self.deadline=time.monotonic()+self.limit
        if time.monotonic()>=self.deadline:raise TimeoutError('Official episode wall timeout during chunk replay')
        if self.clock.needs_chunk():
            result=self.rpc('infer',observations=self.observations(obs),active=self.active,acks=self._ack_rows())
            rows=result['rows']
            if len(rows)!=len(self.active) or {r['slot'] for r in rows}!=set(self.active):
                raise ValueError('Server renumbered/dropped/duplicated live slots')
            tokens={};self.chunks={};self.receipts={}
            for row in rows:
                slot=row['slot'];action=np.asarray(row['action_chunk'],dtype=np.float32)
                if action.shape!=(16,23) or not np.isfinite(action).all():raise ValueError('Invalid chunk')
                tokens[slot]=row['token'];self.chunks[slot]=action;self.receipts[slot]=[]
            self.clock.offer(tokens)
        self.clock.before_step(self.active)
        actions=np.zeros((self.clock.count,23),dtype=np.float32)
        for slot in self.active:actions[slot]=self.chunks[slot][self.clock.pending[slot]['controls']]
        self.actions=actions
        return torch.from_numpy(actions.copy())

    def applied(self, active_indices):
        if tuple(active_indices)!=self.clock.issued or self.actions is None:
            raise ValueError('Official step returned for different active actions')
        for slot in active_indices:
            self.receipts[slot].append(dict(control_step=self.clock.steps[slot],simulator_apply_ack=True,
                action_executed_raw23=self.actions[slot].tolist()))
        self.clock.applied(active_indices);self.actions=None

    def finish(self, obs):
        if self.closed:raise ValueError('Duplicate batch finish')
        # Official episode has finished. Final serialization/verification is
        # not a new policy action and does not consume another episode budget.
        self.deadline=None
        result=self.rpc('finish',observations=self.observations(obs),acks=self._ack_rows())
        if result['steps']!=self.clock.steps:raise ValueError('Server and actual simulator clocks differ')
        self.closed=True;return result
