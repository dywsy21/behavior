"""Stateless model calls; episode RNG/stage/inpainting belong to the caller."""
import dataclasses
import hashlib
from pathlib import Path
import subprocess

import numpy as np


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8<<20):digest.update(block)
    return digest.hexdigest()


def checked_sources(reference,openpi_reference,repo):
    for path,commit,patch in (
        (reference,'ca556f74a455cef7987a2be4537b5ac85cc56dd7','native61-and-task-table-v1.patch'),
        (openpi_reference,'01177e0242a1c7e8fad2547caa0e987def614cda','openpi-lazy-legacy-simulator-v1.patch')):
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()!=commit or
            subprocess.check_output(['git','diff'],cwd=path)!=(repo/'patches/champion2026'/patch).read_bytes()):
            raise ValueError('Changed pinned champion/openpi source')


class ChampionRuntime:
    def __init__(self,checkpoint,num_stages):
        import jax
        from b1k.training import config
        from b1k.policies.policy_config import create_trained_policy
        if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':
            raise ValueError('Use one explicit GPU for this reference runtime')
        base=config.get_config('pi_behavior_b1k_fast')
        cfg=dataclasses.replace(base,model=dataclasses.replace(base.model,
            num_tasks=len(num_stages),task_num_stages=tuple(num_stages)))
        self.policy=create_trained_policy(cfg,checkpoint,sample_kwargs={'num_steps':20})
        self.num_stages=tuple(num_stages)

    def infer(self,observation,task,stage,initial_actions,rng):
        import jax
        import jax.numpy as jnp
        from b1k.models.observation import Observation
        from openpi_client.image_tools import resize_with_pad
        if not 0<=task<len(self.num_stages) or not 0<=stage<self.num_stages[task]:
            raise ValueError('Invalid model task/stage')
        if set(observation['images'])!={'head_rgb','left_wrist_rgb','right_wrist_rgb'}:
            raise ValueError('Exactly three deployed cameras required')
        raw=dict()
        for key,name in [('head_rgb','egocentric_camera'),('left_wrist_rgb','wrist_image_left'),
                         ('right_wrist_rgb','wrist_image_right')]:
            pixels=np.asarray(observation['images'][key])
            if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[0]!=3:
                raise ValueError('Unmodified CHW uint8 camera required')
            # Keep author's inference PIL preprocessing. Do not silently equate
            # PIL uint8 and training JAX resize, or send normalized floats twice.
            raw['observation/'+name]=resize_with_pad(pixels.transpose(1,2,0),224,224)
        raw.update({'observation/state':np.asarray(observation['proprio'],dtype=np.float32),
                    'tokenized_prompt':np.array([task,stage],dtype=np.int32),
                    'tokenized_prompt_mask':np.ones(2,dtype=bool),
                    'subtask_state':np.array(stage,dtype=np.int32)})
        inputs=self.policy._input_transform(dict(raw))
        inputs=jax.tree.map(lambda x:jnp.asarray(x)[None],inputs)
        kwargs=dict(num_steps=20)
        if initial_actions is not None:
            previous=np.asarray(initial_actions)
            if previous.shape!=(4,23) or not np.isfinite(previous).all():
                raise ValueError('Inpainting must be four absolute native controls')
            transformed=self.policy._input_transform(dict(raw,actions=previous))
            kwargs['initial_actions']=jnp.asarray(transformed['actions'])[None]
        # Unlike the upstream Policy.infer, do not mutate policy._rng. A caller
        # owns this uint32 key; interleaved tasks cannot change each other's RNG.
        next_rng,sample_rng=jax.random.split(jnp.asarray(rng,dtype=jnp.uint32))
        obs=Observation.from_dict(inputs)
        actions,logits=self.policy._sample_actions(sample_rng,obs,**kwargs)
        jax.block_until_ready((actions,logits))
        output=self.policy._output_transform(dict(state=np.asarray(inputs['state'][0]),
                    actions=np.asarray(actions[0]),subtask_logits=np.asarray(logits[0])))
        return output,np.asarray(next_rng)
