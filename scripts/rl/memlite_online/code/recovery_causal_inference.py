"""Opt-in target-free planner/session integration; no shared live engine edits.

Checkpoint loading/hash checks belong to the owning serving process. It must
construct this adapter with the exact loaded model identity and route each
actual ACKed observation to its own CausalPlannerSession. No optimizer is
created here; learned outcome predictions remain shadow-only in that session.
"""
from contextlib import nullcontext
import hashlib
from pathlib import Path

from skill_training_protocol import actor_observation
from g05.utils.memlite_causal_feedback import single_frame_planner_prefix, single_frame_member_prefix


def configuration_stats_sha256(config):
    return hashlib.sha256(Path(config['stats_path']).read_bytes()).hexdigest()


def observable_raw(observation, task_name, config):
    import torch
    obs=actor_observation(observation)
    if not isinstance(task_name,str) or not task_name.strip():
        raise ValueError('Explicit task instruction required')
    state={}
    for item in config['raw_shape']['state']:
        key,start,width=item['key'],item['start_index'],item['raw_shape']
        if (key in state or type(start) is not int or type(width) is not int
                or start<0 or width<=0 or start+width>61):
            raise ValueError('Invalid R1Pro observable state mapping')
        state[key]=torch.from_numpy(obs['proprio'][start:start+width].copy())[None]
    if not state:raise ValueError('Missing proprioception mapping')
    return dict(task=task_name,idx=0,embodiment='galaxea_r1pro',frequency=30,
        image_is_pad=torch.zeros(1,dtype=torch.bool),state_is_pad=torch.zeros(1,dtype=torch.bool),
        images={k:torch.from_numpy(v.copy())[None] for k,v in obs['images'].items()},state=state)


class CausalPlannerInference:
    def __init__(self,policy,processor,config,*,models,loaded_planner_sha256,
                 cache_context=nullcontext,device='cuda'):
        if (loaded_planner_sha256!=models.planner or not policy.planner_only
                or policy.training or any(p.requires_grad for p in policy.parameters())
                or configuration_stats_sha256(config)!=models.planner_normalization):
            raise ValueError('Require the separately SHA-checked frozen planner-only model')
        self.policy,self.processor,self.config=policy,processor,config
        self.models,self.cache_context,self.device=models,cache_context,device

    def ensure_context(self,session,identity,observation,*,validate_low_goal,interval_controls=128):
        """Idempotent serving API for value + action at one observed state.

        Reusing a command does not fabricate a generation, a memory update or
        a planner refresh. Low conditioning is still checked for the current
        observation. The transport owns the actual applied-action ACK clock.
        """
        if identity.models!=self.models or not callable(validate_low_goal):
            raise ValueError('Session/model mismatch or missing low condition check')
        actor_observation(observation)
        if session.planning_due(identity,interval_controls=interval_controls):
            return dict(self.plan(session,identity,observation,validate_low_goal=validate_low_goal),reused=False)
        goal=session.low_goal(identity)
        validate_low_goal(goal)
        return dict(goal=goal,event=None,causal_input=None,reused=True,
            control_step=session.feedback.control_step,revision=session.revision,
            physical_success_asserted=False,observer_feedback_mode=session.observer_feedback_mode)

    def plan(self,session,identity,observation,*,validate_low_goal):
        """Generate, validate downstream conditioning, then atomically issue.

        A failed parser, prefix, model or low-level condition check discards
        the request. It does not append memory, install an intent or reset its
        observer cache. No proposed action is sent to a simulator here.
        """
        import torch
        if identity.models!=self.models or not callable(validate_low_goal):
            raise ValueError('Session/model mismatch or missing low condition check')
        token,causal=session.begin_planning(identity,session.feedback.control_step)
        try:
            prepared=self.processor._process_tensors(observable_raw(observation,identity.task,self.config))
            prefix=single_frame_planner_prefix(self.processor.samples_builder,prepared,causal)
            pixels={k:v.unsqueeze(0).to(self.device) for k,v in prepared['pixel_values'].items()}
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if self.device=='cuda' else nullcontext()
            with torch.inference_mode(),amp,self.cache_context():
                result=self.policy.generate_high_level([prefix],pixels,temperature=0.)
            events=result['planner_events']
            if len(events)!=1:raise ValueError('Planner response count does not match the episode')
            proposed=session.stage(identity,token,events[0])
            validate_low_goal(proposed)
            installed=session.commit(identity,token)
            return dict(goal=installed,event=events[0],causal_input=causal,
                control_step=session.feedback.control_step,revision=session.revision,
                physical_success_asserted=False,observer_feedback_mode=session.observer_feedback_mode)
        except BaseException:
            if session.request is not None and session.request['token']==token:
                session.discard(identity,token)
            raise


def observer_prefix(processor,config,session,identity,member,observation,*,loaded_backbone_sha256):
    """The observer may not consume new-planner features of the same shape."""
    if (loaded_backbone_sha256!=identity.models.observer_backbone
            or configuration_stats_sha256(config)!=identity.models.observer_normalization):
        raise ValueError('Result adapter requires its original trained backbone, not the new planner')
    token,check=session.observer_request(identity,member)
    prepared=processor._process_tensors(observable_raw(observation,identity.task,config))
    prefix=single_frame_member_prefix(processor.samples_builder,prepared,**check)
    return token,prefix,prepared['pixel_values']
