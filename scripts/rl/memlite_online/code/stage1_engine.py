"""Single-frame Stage1 planner / FM adapter for a bounded local PPO pilot."""
from pathlib import Path
import json
import os
import hashlib
import numpy as np
import torch
from g05.utils.training.stage1_model import make_processor,restore_model
from g05.models.g05.inferencer import PolicyInferencer
from g05.models.g05.qwen35 import vision
from native_b_session import PlannerLedger
from direct_a4_flow import A4DirectPPO
ROOT=Path('/run/ti/rl_memlite_stage1_20261006')


def move(value):
    if isinstance(value,torch.Tensor): return value.cuda()
    if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
    if isinstance(value,list):return [move(v) for v in value]
    return value


class Stage1Engine:
    def __init__(self, output):
        vision._flash_attn_varlen=None;vision._flash_attn_backend=None
        self.models={};self.processors={};self.trainer=None
        self.reward_protocol=os.environ.get('RL_REWARD_PROTOCOL','legacy_two_task_v1')
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        for side,name in [('high','step_00048045_save_0027.pt'),('low','step_00098414_save_0021.pt')]:
            config=json.loads((ROOT/'configs'/(side+'_model.local.json')).read_text())
            original=make_processor(config,False)
            config['processor']['samples_builder']['_target_']='saved_builders.'+('PlannerOutcomeBuilder' if side=='high' else 'SkillFMActionBuilder')
            processor=make_processor(config,False)
            assert processor.samples_builder.template==original.samples_builder.template
            state=torch.load(ROOT/'models/stage1'/side/name,map_location='cpu',mmap=True,weights_only=False)
            policy,_=restore_model(config,side,state=state['model_state_dict'])
            policy.requires_grad_(False).eval().cuda()
            self.models[side]=policy;self.processors[side]=processor
            del state,original
        self.slots=[]
        self.episode_metadata=[]

    @staticmethod
    def raw(observation):
        def frames(value,ndim):
            value=torch.from_numpy(np.asarray(value).copy())
            return value.unsqueeze(0) if value.ndim==ndim else value[-1:]
        return dict(task=observation['task'],idx=0,embodiment='galaxea_r1pro',
            frequency=torch.tensor(30.),image_is_pad=torch.tensor([False]),state_is_pad=torch.tensor([False]),
            images={k:frames(v,3) for k,v in observation['images'].items()},
            state={k:frames(v,1) for k,v in observation['state'].items()})

    def prepare(self,side,observations,projections):
        processor=self.processors[side];samples=[]
        for observation,projection in zip(observations,projections,strict=True):
            sample=processor._process_tensors(self.raw(observation))
            sample['samples']=processor.samples_builder.build_for_inference(projection,sample)
            shape_only={'action':{k:torch.zeros(1,w) for k,w in {'left_arm':7,'left_gripper':1,'right_arm':7,'right_gripper':1,'lower_body':7}.items()}}
            pad=processor.action_state_merger.forward(shape_only)['action_dim_is_pad']
            assert torch.where(pad)[0].tolist()==[7,8,17,18]
            sample['action_dim_is_pad']=pad
            for key in ('action','gt_action','action_is_pad'):sample.pop(key,None)
            samples.append(sample)
        batch=move(PolicyInferencer._collate(samples,padding_input_id=processor.pad_token_id))
        if side=='low' and self.reward_protocol=='shared_terminal_q_v1':
            # Kept outside samples/images, so policy.prefill and the actor do
            # not receive this critic-only finite-horizon feature.
            batch['critic_remaining_fraction']=torch.tensor(
                [observation['critic_remaining_fraction'] for observation in observations],device='cuda')
        return batch

    def prepare_training_batch(self,observations,projections):
        return self.prepare('low',observations,projections)

    def _branch_context(self):return torch.autocast('cuda',dtype=torch.bfloat16)

    def begin(self,task,num_envs,seed,episode_metadata=None):
        if self.trainer is not None and self.trainer.experiences:
            raise RuntimeError('Cannot reset with unconsumed on-policy experience')
        catalog = [json.loads(line)['task_name'] for line in Path('/run/ti/behavior_stage3_20260930/tools/a4_tasks.jsonl').read_text().splitlines() if line.strip()]
        if task not in catalog: raise ValueError(task)
        torch.manual_seed(seed)
        self.task=task.replace('_',' ')
        self.episode_metadata=episode_metadata or [{} for _ in range(num_envs)]
        if len(self.episode_metadata)!=num_envs:raise ValueError('Episode metadata count differs')
        self.slots=[dict(ledger=PlannerLedger(self.task),projection=None,chunks=0,
                         planned_chunk=None,context_id=None) for _ in range(num_envs)]

    def plan(self,observation,index):
        slot=self.slots[index];ledger=slot['ledger']
        batch=self.prepare('high',[observation],[ledger.projection()])
        with torch.no_grad(),self._branch_context():
            result=self.models['high'].generate_high_level(batch['samples'],batch['pixel_values'],temperature=0.)
        proposed=ledger.stage(result['planner_events'][0]);goal=proposed['installed_subgoal']
        projection=dict(schema_version=6,memlite_branch='low',task_name=ledger.task_name,
            parent_goal=goal['parent_goal'],target_parent_goal='',previous_parent_goal='none',memory=ledger.memory,
            previous_intent='None',known_previous_outcome='UNKNOWN',execution_feedback='none',
            active_skills_semantic_json=goal['active_skills_semantic_json'],active_skills_text=proposed['event']['active_skills_text'],
            next_decision='EXECUTE',memory_update='',task_complete=False,outcome_target='UNKNOWN',
            outcome_supervision_mask=False,parent_goal_supervision_mask=False,low_action_supervision_mask=False)
        context_id=hashlib.sha256(json.dumps(projection,sort_keys=True).encode()).hexdigest()
        with (self.output.parent/'planner_events.jsonl').open('a') as log:log.write(json.dumps({'task':self.task,'env':index,'chunk':slot['chunks'],'event':proposed['event'],
            'context_id':context_id,'episode':self.episode_metadata[index],
            'control_step':observation.get('control_step'),
            'policy_update':self.trainer.update_count if self.trainer else 0})+'\n')
        slot['projection']=projection
        slot['planned_chunk']=slot['chunks'];slot['context_id']=context_id
        ledger.commit(proposed['proposal_id'])
        return projection

    def ensure_context(self,observation,index):
        """Admit the next context once, shared by bootstrap and action sampling.

        A value request may precede the optimizer barrier. It must prepare the
        same frozen-planner context the next action will actually use; a second
        infer at that boundary must NOT append memory or generate again.
        """
        observation['task']=self.task
        slot=self.slots[index]
        if slot['projection'] is None or (slot['chunks']%8==0 and slot['planned_chunk']!=slot['chunks']):
            self.plan(observation,index)
        return slot['projection']

    def infer(self,observations,indices):
        # Keep the numerical batch shape identical during sampling and PPO
        # recomputation. BF16 batched prefill changed likelihoods significantly.
        if len(observations)>1:
            rows=[self.infer([obs],[index]) for obs,index in zip(observations,indices,strict=True)]
            return dict(actions=np.concatenate([row['actions'] for row in rows]),
                        experience_ids=[x for row in rows for x in row['experience_ids']],
                        values=np.concatenate([row['values'] for row in rows]),
                        contexts=[x for row in rows for x in row['contexts']],
                        policy_updates=[x for row in rows for x in row['policy_updates']])
        projections=[]
        for observation,index in zip(observations,indices,strict=True):
            slot=self.slots[index]
            projections.append(self.ensure_context(observation,index));slot['chunks']+=1
        batch=self.prepare('low',observations,projections)
        if self.trainer is None:
            from training_config import load_training_config
            settings=load_training_config()
            self.trainer=A4DirectPPO(self.models['low'],batch['action_dim_is_pad'],self.output,
                microbatch_size=1,actor_lr=settings['actor_lr'],critic_lr=settings['critic_lr'],
                target_kl=settings['target_kl'],max_clip_fraction=settings['max_clip_fraction'])
            checkpoint = os.environ.get('RL_RESUME_CHECKPOINT', '')
            if checkpoint:
                receipt = self.trainer.load_checkpoint(Path(checkpoint))
                assert self.trainer.update_count > 0
                assert len(self.trainer.actor_optimizer.state) == (322 if self.trainer.actor_update_count else 0)
                receipt['actor_optimizer_reset'] = False
                (self.output.parent/'resume_receipt.json').write_text(json.dumps(receipt, indent=2))


        with torch.no_grad(),self._branch_context():
            sampled=self.trainer.sample_branch(batch,observations,projections)
        decoded=[]
        cpu={k:v.cpu() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        cpu['action']=sampled['action'].cpu();cpu['selected_action_source']='fm'
        for index,observation in enumerate(observations):
            raw=self.raw(observation)
            anchor={k:raw['state'][k][-1:].clone().unsqueeze(0) for k in ('left_arm','right_arm','trunk_qpos')}
            groups=PolicyInferencer._postprocess_single(cpu,index,self.processors['low'],raw_state_anchor=anchor)
            parts=[torch.as_tensor(groups[k]).reshape(32,width) for k,width in
                [('base_qvel',3),('trunk_qpos',4),('left_arm',7),('left_gripper',1),('right_arm',7),('right_gripper',1)]]
            action=torch.cat(parts,dim=-1)[:16].numpy()
            assert action.shape==(16,23) and np.isfinite(action).all()
            env_index=indices[index]
            with (self.output.parent/'action_audit.jsonl').open('a') as log:log.write(json.dumps({'task':self.task,'env':env_index,
                'episode':self.episode_metadata[env_index],'control_step':observation.get('control_step'),
                'experience_id':sampled['experience_ids'][index],'policy_update':self.trainer.update_count,
                'context_id':self.slots[env_index]['context_id'],
                'min':float(action.min()),'max':float(action.max()),'mean_abs':float(np.abs(action).mean()),'finite':bool(np.isfinite(action).all())})+'\n')
            decoded.append(action)
        return dict(actions=np.stack(decoded),experience_ids=sampled['experience_ids'],
                    values=sampled['old_value'].float().cpu().numpy(),
                    contexts=[dict(context_id=self.slots[i]['context_id'],parent_goal=p['parent_goal'],
                                   active_skills_semantic_json=p['active_skills_semantic_json'],
                                   active_skills_text=p['active_skills_text']) for i,p in zip(indices,projections)],
                    policy_updates=[self.trainer.update_count for _ in indices])

    def value(self,observations,indices):
        if len(observations)>1:
            return np.concatenate([self.value([obs],[index]) for obs,index in zip(observations,indices,strict=True)])
        projections=[self.ensure_context(observation,index) for observation,index in zip(observations,indices,strict=True)]
        batch=self.prepare('low',observations,projections)
        with torch.no_grad(),self._branch_context():
            _,features=self.trainer._prefix(batch)
            return self.trainer.critic(features).float().cpu().numpy()
