"""Exact restoration of final high, low, and independent calibrated observer."""
import gc
from dataclasses import asdict
from pathlib import Path

from recovery_calibration_launch import read
from recovery_corpus import file_sha
from assets import verify_assets


class LoadedFinalModels:
    def __init__(self, pack, manifest_sha256, config_path):
        self.config,self.manifest,self.configs=verify_assets(pack,manifest_sha256,config_path)
        self.config_sha256=file_sha(config_path)
        self.pack=Path(pack);cfg=self.config;manifest=self.manifest
        from g05.utils.memlite_causal_session import CausalModelIdentity
        from recovery_calibrated_observer import load_grasp_calibration
        self.identity=CausalModelIdentity(**{key:cfg['models'][key]['sha256'] for key in
            ('planner','low','observer_backbone','observer_adapter')},
            **{key+'_normalization':v for key,v in cfg['normalization_sha256'].items()})
        fit_cfg=read(self.pack/'observer_fit_config.json');fit=read(self.pack/'observer_fit_result.json')
        original=cfg['models']['observer_backbone']
        if (file_sha(self.pack/'observer_fit_config.json')!=cfg['observer_fit_config']['sha256']
                or file_sha(self.pack/'observer_fit_result.json')!=cfg['observer_fit_result']['sha256']
                or fit.get('status')!='observer_adapter_fit_complete_not_deployed'
                or fit['config_sha256']!=cfg['observer_fit_config']['sha256']
                or fit['selected_checkpoint_sha256']!=self.identity.observer_adapter
                or fit['frozen_before_sha256']!=fit['frozen_after_sha256']
                or fit_cfg['high']!={k:original[k] for k in ('path','sha256')}
                or fit_cfg['stats_sha256']!=self.identity.observer_normalization
                or fit_cfg.get('history_protocol','cadence16_v1')!='cadence16_v1'
                or fit_cfg.get('include_served_controls',False)):
            raise ValueError('Original observer fit, features or history changed')
        self.calibration=load_grasp_calibration(self.pack/'calibration.json',cfg['calibration']['sha256'],self.identity)
        import torch
        from g05.models.g05.qwen35 import vision
        from g05.utils.training.stage1_model import restore_model,make_processor
        from g05.models.kv_cache import SparseKVCache
        from sparse_cache import indexed_cache_freeze
        from train_memlite_recovery import parameter_digest
        from recovery_causal_inference import CausalPlannerInference
        from recovery_observer_pilot_runtime import LoadedPilotObserver
        from skill_policy_adapter import SingleFrameSkillAdapter
        torch.set_num_threads(2);torch.manual_seed(cfg['policy_seed'])
        vision._flash_attn_varlen=None;vision._flash_attn_backend=None
        def forbidden(*args,**kwargs):raise RuntimeError('Full evaluation forbids optimizer steps')
        torch.optim.Adam.step=torch.optim.AdamW.step=forbidden
        self.models={};self.restores={};self.digest=parameter_digest
        for side,key in [('high','planner'),('low','low')]:
            saved=torch.load(manifest['models'][key]['runtime_path'],map_location='cpu',mmap=True,weights_only=False)
            if saved['source_sha256']!=cfg['models'][key]['sha256'] or saved['state']['step']!=cfg['models'][key]['step']:
                raise ValueError('Wrong exact model export: '+key)
            policy,self.restores[side]=restore_model(self.configs[side],side,state=saved['model_state_dict'])
            del saved;gc.collect()
            self.models[side]=policy.requires_grad_(False).eval().cuda()
        self.models['low'].model.action_expert.float()
        self.adapter=SingleFrameSkillAdapter(make_processor(self.configs['low'],False),self.configs['low'])
        high=self.models['high']
        self.planner=CausalPlannerInference(high,make_processor(self.configs['high'],False),self.configs['high'],
            models=self.identity,loaded_planner_sha256=self.identity.planner,
            cache_context=lambda:indexed_cache_freeze(high.model.ar_helper,SparseKVCache))
        spec=dict(root=str(self.pack),observer_backbone=dict(path='observer_backbone.pt'),observer_adapter=dict(path='observer_adapter.pt'))
        accepted=dict(fit_config=fit_cfg,fit=fit,paths=dict(observer_fit_config=self.pack/'observer_fit_config.json'))
        self.observer=LoadedPilotObserver(dict(models=spec,calibrated=accepted),self.identity,
            read(self.pack/'task_names.json'),parameter_digest=parameter_digest,configuration_override=self.configs['high'])
        self.before={key:parameter_digest(model,frozen=True) for key,model in self.models.items()}

    def sample(self, observations, goals, steps):
        from batch_runtime import sample_many
        return sample_many(self.models['low'],self.adapter,observations,goals,steps,seed=self.config['policy_seed'])

    def finish(self):
        after={key:self.digest(model,frozen=True) for key,model in self.models.items()}
        if after!=self.before or any(p.grad is not None or p.requires_grad for m in self.models.values() for p in m.parameters()):
            raise ValueError('Frozen high/low acquired gradients or changed weights')
        return dict(weights_unchanged=True,model_digests=after,optimizer_updates=0,
            model_identity=asdict(self.identity),config_sha256=self.config_sha256,**self.observer.finish())
