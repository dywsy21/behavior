"""Fresh on-policy chunks with per-environment GAE and explicit reset boundaries."""
from collections import deque
import time,json,os
from pathlib import Path
import numpy as np
import torch
from websockets.sync.client import connect
from a4_wire import packb,unpackb
from a4_observation import behavior_obs_to_native_low
from dense_reward import chunk_targets


class PilotPolicy:
    def __init__(self,num_envs,task,run):
        self.num_envs=num_envs;self.task=task;self.run=Path(run)
        self.tasks={r['task_index']:r['task'] for r in (json.loads(line) for line in Path('/run/ti/behavior_stage3_20260930/tools/a4_tasks.jsonl').read_text().splitlines())}
        self.socket=connect('ws://127.0.0.1:' + os.environ['RL_POLICY_PORT'],max_size=512<<20,ping_timeout=None,open_timeout=30)
        assert unpackb(self.socket.recv(timeout=600))['kind']=='stage1_rl_v1'
        self.reset_count=0;self.total_steps=0;self.active=list(range(num_envs));self.reset()

    def rpc(self,kind,**payload):
        self.socket.send(packb(dict(kind=kind,**payload)))
        response=unpackb(self.socket.recv(timeout=900))
        if not response['ok']:raise RuntimeError(response['error'])
        return response['response']

    def reset(self):
        if hasattr(self,'rows') and any(self.rows):raise RuntimeError('Unconsumed rollouts at reset')
        self.rows=[[] for _ in range(self.num_envs)];self.actions=[deque() for _ in range(self.num_envs)]
        self.rpc('begin',task=self.task,num_envs=self.num_envs,seed=int(os.environ.get("RL_SEED_BASE","17"))+self.reset_count)
        self.reset_count+=1

    def observations(self,obs,indices):
        return [behavior_obs_to_native_low({k:v[i] for k,v in obs.items()},self.tasks) for i in indices]

    def forward(self,obs):
        indices=[i for i in self.active if not self.actions[i]]
        if indices:
            observations=self.observations(obs,indices)
            result=self.rpc('infer',observations=observations,indices=indices)
            for j,i in enumerate(indices):
                value=float(result['values'][j])
                if self.rows[i] and self.rows[i][-1]['next_value'] is None:self.rows[i][-1]['next_value']=value
                self.rows[i].append(dict(experience_id=int(result['experience_ids'][j]),value=value,
                    reward=0.,controls=0,next_value=None,terminated=False,truncated=False))
                self.actions[i].extend(np.asarray(result['actions'][j],dtype=np.float32))
                if getattr(self,'recorder',None) is not None:
                    self.recorder.on_chunk(i,observations[j],result['contexts'][j],
                                           result['experience_ids'][j],result['policy_updates'][j])
        actions=np.zeros((self.num_envs,23),dtype=np.float32)
        for i in self.active:
            actions[i]=self.actions[i].popleft()
            if getattr(self,'recorder',None) is not None:
                self.recorder.before_action(i,{k:v[i] for k,v in obs.items()},actions[i])
        return torch.from_numpy(actions)

    def observe(self,evaluator,active,terminated,truncated):
        for i in active:
            result,audit=evaluator.reward_adapters[i].step(terminated[i],truncated[i])
            row=self.rows[i][-1];row['reward']+=0.999**row['controls']*result.total;row['controls']+=1
            row['terminated']=bool(terminated[i]);row['truncated']=bool(truncated[i])
            if row['terminated']:row['next_value']=0.
            if row['terminated'] or row['truncated']:self.actions[i].clear()
            self.total_steps+=1
            with (self.run/'reward_steps.jsonl').open('a') as f:
                f.write(json.dumps(dict(task=self.task,env=i,step=self.total_steps,reward=result.total,
                    success_bonus=result.success_bonus,shaping=result.shaping,**audit))+'\n')
        # Update at a chunk barrier: never change weights while an old chunk
        # is still being executed by another environment.
        if sum(map(len,self.rows))>=64 and all(not self.actions[i] for i in active):self.flush(evaluator)

    def flush(self,evaluator):
        indices=[i for i,rows in enumerate(self.rows) if rows and rows[-1]['next_value'] is None]
        if indices:
            values=self.rpc('value',observations=self.observations(evaluator._batch_obs(),indices),indices=indices)['values']
            for i,v in zip(indices,values):self.rows[i][-1]['next_value']=float(v)
        ids=[];advantages=[];returns=[]
        for rows in self.rows:
            if not rows:continue
            assert all(row['controls']>0 and row['next_value'] is not None for row in rows)
            adv,ret=chunk_targets(rows)
            ids.extend(row['experience_id'] for row in rows);advantages.extend(adv);returns.extend(ret)
        if ids:
            result=self.rpc('update',experience_ids=ids,advantages=advantages,returns=returns)
            with (self.run/'update_receipts.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            self.rows=[[] for _ in range(self.num_envs)]

    def close(self):self.socket.close()
