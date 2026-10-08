"""Original SFT weights + native FM ODE; no PPO object, critic or exploration."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from common import bootstrap, atomic_json
bootstrap()
from stage1_engine import Stage1Engine
from g05.models.g05.inferencer import PolicyInferencer


def forbidden(*args, **kwargs):
    raise RuntimeError('SFT evaluation forbids optimizer steps')


class NativeSFT(Stage1Engine):
    def __init__(self, output):
        if os.environ.get('RL_RESUME_CHECKPOINT'):
            raise ValueError('An RL checkpoint cannot enter this SFT evaluation')
        os.environ['RL_REWARD_PROTOCOL'] = 'native_sft_evaluation'
        torch.optim.Adam.step = forbidden
        torch.optim.AdamW.step = forbidden
        super().__init__(output)
        self.initial_hashes = self.hashes()
        self.requests = 0
        atomic_json(self.output.parent / 'weights_initial.json', self.initial_hashes)

    def hashes(self):
        result = {}
        for side, model in self.models.items():
            digest = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                x = tensor.detach().cpu().contiguous()
                digest.update(name.encode())
                digest.update(str((tuple(x.shape), str(x.dtype))).encode())
                digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
            result[side] = digest.hexdigest()
        return result

    @torch.no_grad()
    def infer_native(self, observations, indices):
        # Fixed single-example numerical shape, irrespective of finished envs.
        actions = []
        contexts = []
        for obs, index in zip(observations, indices, strict=True):
            projection = self.ensure_context(obs, index)
            self.slots[index]['chunks'] += 1
            batch = self.prepare('low', [obs], [projection])
            policy = self.models['low']
            if not policy.continuous_action or policy.discrete_action or policy.predict_cot:
                raise ValueError('Not the original SkillFM inference route')
            with self._branch_context():
                result = policy.forward_inference(samples=batch['samples'],
                    pixel_values=batch['pixel_values'], action_dim_is_pad=batch['action_dim_is_pad'])
            if result.get('selected_action_source') != 'fm':
                raise ValueError('Expected native FM output')
            cpu = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            cpu['action'] = result['action'].cpu()
            cpu['selected_action_source'] = 'fm'
            raw = self.raw(obs)
            anchor = {k: raw['state'][k][-1:].clone().unsqueeze(0)
                      for k in ('left_arm', 'right_arm', 'trunk_qpos')}
            groups = PolicyInferencer._postprocess_single(cpu, 0, self.processors['low'], raw_state_anchor=anchor)
            order = [('base_qvel',3), ('trunk_qpos',4), ('left_arm',7), ('left_gripper',1),
                     ('right_arm',7), ('right_gripper',1)]
            action = torch.cat([torch.as_tensor(groups[k]).reshape(32,w) for k,w in order], -1)[:16].float().numpy()
            if action.shape != (16,23) or not np.isfinite(action).all():
                raise ValueError('Invalid raw 23D native FM chunk')
            actions.append(action)
            contexts.append(dict(context_id=self.slots[index]['context_id'],
                                 parent_goal=projection['parent_goal'],
                                 active_skills_semantic_json=projection['active_skills_semantic_json']))
        self.requests += 1
        assert self.trainer is None
        return np.stack(actions), contexts

    def verify(self):
        current = self.hashes()
        if current != self.initial_hashes or self.trainer is not None:
            raise RuntimeError('SFT evaluation modified weights or constructed a trainer')
        return dict(weights_unchanged=True, model_hashes=current, optimizer_steps=0,
                    inference_requests=self.requests, native_fm=True,
                    max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    max_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
