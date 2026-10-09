"""Single-frame MEM-Lite native low FM adapter shared by short-skill PPO.

Only current RGB, proprio and the actual semantic skill enter the frozen VLM.
Unlike the old A4 adapter this cannot silently request six image-history frames.
"""
from contextlib import contextmanager

from fixed_skill_protocol import low_prefix
from skill_training_protocol import actor_observation


class SingleFrameSkillAdapter:
    def __init__(self, processor, config):
        self.processor=processor;self.config=config

    def prepare(self, observation, goal):
        import torch
        obs=actor_observation(observation)
        if set(goal)!={'task','parent_goal','semantic_bundle'}:raise ValueError('Unapproved policy goal fields')
        raw=dict(task=goal['task'].replace('_',' '),idx=0,embodiment='galaxea_r1pro',frequency=30,
            image_is_pad=torch.zeros(1,dtype=torch.bool),state_is_pad=torch.zeros(1,dtype=torch.bool),
            images={k:torch.from_numpy(v.copy())[None] for k,v in obs['images'].items()},
            state={m['key']:torch.from_numpy(obs['proprio'][m['start_index']:m['start_index']+m['raw_shape']].copy())[None]
                   for m in self.config['raw_shape']['state']})
        anchor={k:raw['state'][k][-1:].clone()[None] for k in ('left_arm','right_arm','trunk_qpos')}
        prepared=self.processor._process_tensors(raw)
        prepared['samples']=low_prefix(self.processor.samples_builder,prepared,task=raw['task'],
            parent_goal=goal['parent_goal'],semantic_bundle=goal['semantic_bundle'])
        shape={'action':{k:torch.zeros(1,w) for k,w in
                       {'left_arm':7,'left_gripper':1,'right_arm':7,'right_gripper':1,'lower_body':7}.items()}}
        pad=self.processor.action_state_merger.forward(shape)['action_dim_is_pad']
        if torch.where(pad)[0].tolist()!=[7,8,17,18]:raise ValueError('Lost real base/trunk action dimensions')
        prepared['action_dim_is_pad']=pad
        for key in ('action','gt_action','action_is_pad'):prepared.pop(key,None)
        return prepared,anchor

    def collate(self, prepared):
        import torch
        from g05.models.g05.inferencer import PolicyInferencer
        def move(v):
            if isinstance(v,torch.Tensor):return v.cuda()
            if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
            if isinstance(v,list):return [move(x) for x in v]
            return v
        return move(PolicyInferencer._collate(prepared,padding_input_id=self.processor.pad_token_id))

    def prepare_training_batch(self, observations, goals):
        return self.collate([self.prepare(obs,goal)[0] for obs,goal in zip(observations,goals,strict=True)])

    @contextmanager
    def _branch_context(self):
        import torch
        # The frozen VLM prefill is BF16; A4DirectPPO's velocity explicitly
        # disables autocast and uses the FP32 trainable action expert.
        with torch.autocast('cuda',dtype=torch.bfloat16):yield

    def postprocess(self, action, batch, anchor):
        import numpy as np
        import torch
        from g05.models.g05.inferencer import PolicyInferencer
        cpu={k:v.detach().cpu() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        # Postprocessing is the native FM normalization/unpadding path; PPO
        # likelihoods are always computed on the unclipped latent FM chain.
        cpu.update(action=action.detach().cpu(),selected_action_source='fm')
        groups=PolicyInferencer._postprocess_single(cpu,0,self.processor,raw_state_anchor=anchor)
        order=[('base_qvel',3),('trunk_qpos',4),('left_arm',7),('left_gripper',1),('right_arm',7),('right_gripper',1)]
        result=torch.cat([torch.as_tensor(groups[k]).reshape(32,w) for k,w in order],-1)[:16].float().numpy()
        if result.shape!=(16,23) or not np.isfinite(result).all():raise ValueError('Invalid unpadded raw23 chunk')
        return result
