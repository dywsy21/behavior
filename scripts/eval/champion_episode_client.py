"""Native23, 20-control champion transport, separate from MEM-Lite16 interface."""
from collections import deque
import json
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

from full_episode_client import FullEpisodePolicy,observable
from wire import unpackb


class ChampionEpisodePolicy(FullEpisodePolicy):
    def __init__(self, *, port, task_id, case, config_sha256, output):
        self.output=Path(output);self.task_id=task_id;self.used=False;self.elapsed=0
        self.socket=connect(f'ws://127.0.0.1:{port}',max_size=64<<20,ping_timeout=None,open_timeout=30)
        hello=unpackb(self.socket.recv(timeout=900))
        if (hello['kind']!='champion2026_episode_v1' or hello['case']!=case or
                hello['config_sha256']!=config_sha256 or hello['chunk_size']!=20):
            raise ValueError('Wrong frozen champion model/episode')
        self.identity=hello['identity'];self.actions=deque();self.pending=[];self.token=None
        self.last_action=None;self.display=None
        (self.output/'service-hello.json').write_text(json.dumps(hello,indent=2)+'\n')

    def forward(self,obs):
        import torch
        if not self.actions:
            self.ack()
            result=self.rpc('infer',observation=observable(obs,self.task_id))
            action=np.asarray(result['action_chunk'],dtype=np.float32)
            if action.shape!=(20,23) or not np.isfinite(action).all():raise ValueError('Invalid champion20/native23')
            self.token=result['token'];self.display=result['display']
            if self.display['control_step']!=self.elapsed or self.display['task']!=self.task_id:
                raise ValueError('Foreign/future/stale stage context')
            with (self.output/'chunks.jsonl').open('a') as f:
                f.write(json.dumps(dict(token=self.token,display=self.display,actions=action.tolist()),allow_nan=False)+'\n')
            self.actions.extend(action)
        self.last_action=self.actions.popleft().copy()
        return torch.from_numpy(self.last_action[None])

    def finish(self,obs):
        observable(obs,self.task_id)  # Still enforce official task identity at terminal.
        self.ack()
        result=self.rpc('finish')
        if result['control_step']!=self.elapsed:raise ValueError('Incomplete final champion ACK')
        self.socket.close()
        return result
