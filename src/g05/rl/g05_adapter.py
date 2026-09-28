"""G05 native preprocessing/AE decoding with explicit stochastic path capture."""
from copy import deepcopy
from contextlib import nullcontext
import torch

from g05.models.kv_cache import SparseKVCache
from g05.models.g05.inferencer import PolicyInferencer
from g05.utils.common.pytorch_utils import dict_apply
from .flow_ppo import PAD_DIMS, gaussian_logp, gaussian_kl
from .native_decode import cpu_postprocess_inputs


def move(value, device):
    return dict_apply(value,lambda t:t.detach().to(device) if isinstance(t,torch.Tensor) else t)


class G05FlowAdapter:
    def __init__(self, policy, processor, noise):
        self.policy,self.processor,self.noise=policy,processor,noise
        self.model=policy.model; self.fm=self.model.fm_helper; self.device='cuda:0'
        self.infer=PolicyInferencer(policy,processor,device=self.device)
        if (self.fm.time_convention!='pi_convention' or self.fm.num_inference_steps!=10 or
                self.fm.horizon_steps!=32 or self.fm.action_dim!=27 or self.fm.zero_pad_action_target or
                self.fm.final_action_clip_value is not None or self.fm.action_causal or
                self.model.cfg.ae_vlm_condition_mode!='cross_attn_only'):
            raise ValueError('Parent FM architecture/configuration changed')
        policy.requires_grad_(False); self.model.action_expert.requires_grad_(True)
        # Eval disables stochastic dropout/augmentation, but permits AE autograd.
        policy.eval(); policy.discrete_action=False

    def context_from_state(self,state):
        dtype=next(iter(state.pixel_values.values())).dtype
        mask,pos=self.model.build_action_mask_and_position_ids(state.attention_mask,action_len=32,
            position_ids_prefix=state.position_ids,split_index=state.attention_mask.size(1),dtype=dtype,action_causal=False)
        kv=self.model._build_prefix_action_kv(state.kv_cache,state.kv_cache.num_items())
        if kv.recurrent_states or kv.conv_states: raise ValueError('Unexpected recurrent AE conditioning')
        return dict(mask=mask,pos=pos,keys=dict(kv.key_cache),values=dict(kv.value_cache),
                    feature=state.last_hidden.float(),dtype=dtype)

    @torch.no_grad()
    def prepare(self,raw):
        from serve_policy import build_obs_dict
        if set(raw)!={'images','state','task','embodiment_type','frequency'}:
            raise ValueError('Privileged field in actor observation')
        p=self.infer._prepare(build_obs_dict(deepcopy(raw),self.processor))
        batch=self.infer._collate([p.sample],padding_input_id=p.sub_processor.pad_token_id)
        batch=move(batch,self.device)
        expected=torch.zeros((1,27),dtype=torch.bool,device=self.device); expected[:,PAD_DIMS]=True
        if not torch.equal(batch['action_dim_is_pad'].bool(),expected): raise ValueError('23/27 action map changed')
        with torch.autocast('cuda',dtype=torch.bfloat16): state=self.policy.prefill(batch['samples'],batch['pixel_values'])
        context=self.context_from_state(state)
        return context,(p,batch,state)

    def velocity(self,context,latent,time):
        ae=self.model.action_expert
        with torch.autocast('cuda',dtype=torch.bfloat16):
            with torch.autocast('cuda',enabled=False):
                embeds=ae.embed(latent.float()); tcond=ae.encode_time(time.float())
            hidden=ae(inputs_embeds=embeds,attention_mask=context['mask'],position_ids=context['pos'],
                time_cond=tcond,attn_implementation=self.model.attn_implementation,mixture_name='action',
                kv_cache=SparseKVCache(context['keys'],context['values']))
            return ae.decode(hidden)

    def transition(self,context,x,t):
        mean=x-.1*self.velocity(context,x,t)
        mean=mean.masked_fill(torch.tensor([i in PAD_DIMS for i in range(27)],device=x.device)[None,None,:],0.)
        # Explicit fp32 std arithmetic, independent of AE autocast.
        with torch.autocast('cuda',enabled=False): std=self.noise(context['feature'],x,t)
        return mean,std

    @torch.no_grad()
    def sample(self,context,*,stochastic=True):
        x=torch.randn(1,32,27,device=self.device,dtype=context['dtype']); x[:,:,PAD_DIMS]=0
        t=torch.ones(1,device=self.device,dtype=context['dtype'])
        xs=[x.clone()]; means=[]; stds=[]; times=[]; lp=torch.tensor(0.,device=self.device)
        for _ in range(10):
            mean,std=self.transition(context,x,t)
            following=mean+std*torch.randn_like(x) if stochastic else mean
            following[:,:,PAD_DIMS]=0
            if stochastic: lp+=gaussian_logp(following,mean,std)
            means.append(mean); stds.append(std); times.append(t.clone()); xs.append(following)
            x=following; t=t-.1
        trace=dict(xs=torch.stack(xs),means=torch.stack(means),stds=torch.stack(stds),times=torch.stack(times),old_logp=lp)
        return x,move(trace,'cpu')

    def score(self,context,trace):
        lp=torch.tensor(0.,device=self.device); kl=torch.tensor(0.,device=self.device)
        for k in range(10):
            mean,std=self.transition(context,trace['xs'][k],trace['times'][k])
            lp=lp+gaussian_logp(trace['xs'][k+1],mean,std)
            kl=kl+gaussian_kl(trace['means'][k],trace['stds'][k],mean,std)
        return lp,kl

    @torch.no_grad()
    def decode(self,x,prepared):
        from scripts.experiments.eval_g05_100k import vector_chunk
        p,batch,_=prepared
        batch=cpu_postprocess_inputs(x,batch)
        action=self.infer._postprocess_single(batch,0,p.sub_processor,
            **({'raw_state_anchor':p.raw_state_anchor} if p.raw_state_anchor is not None else {}))
        return vector_chunk(action)

    @torch.no_grad()
    def equivalence_gate(self,context,prepared):
        p,batch,state=prepared
        rng=torch.cuda.get_rng_state()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            native=self.fm.infer(self.model,state.attention_mask,state.pixel_values,state.kv_cache,
                action_dim_is_pad=batch['action_dim_is_pad'],position_ids_override=state.position_ids)
        torch.cuda.set_rng_state(rng)
        ours,_=self.sample(context,stochastic=False)
        difference=float((native-ours).abs().max())
        if not torch.equal(native,ours): raise ValueError(f'Zero-added-noise decoder differs: {difference}')
        # Validate the real 23D robot boundary as well as normalized latents.
        decoded=self.decode(ours,prepared)
        native_batch=move(dict(batch,action=native,selected_action_source='fm'),'cpu')
        native_action=self.infer._postprocess_single(native_batch,0,p.sub_processor,
            **({'raw_state_anchor':p.raw_state_anchor} if p.raw_state_anchor is not None else {}))
        from scripts.experiments.eval_g05_100k import vector_chunk
        import numpy as np
        if not np.array_equal(decoded,vector_chunk(native_action)):
            raise ValueError('Decoded 23D controls differ from native CPU postprocessing')
        _,trace=self.sample(context)
        lp,kl=self.score(context,move(trace,self.device))
        error=float((lp-trace['old_logp'].to(self.device)).abs())
        if error>.002 or abs(float(kl))>1e-6: raise ValueError('Old-path probability identity failed')
        return dict(zero_noise_max_difference=difference,old_path_logp_error=error,path_kl=float(kl),
                    scored_transitions=10*32*23, frozen_features=True,
                    native_cpu_decode_equal=True,decoded_shape=list(decoded.shape))
