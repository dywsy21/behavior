"""Load the separately trained, fixed observer for an explicit read-only pilot.

Never share the new planner's features with the old observer adapter. Keep
calibration/fit/target-free generation receipts bound before any CUDA load.
"""
import gc
from pathlib import Path

from recovery_calibration_launch import bound,read
from recovery_corpus import file_sha
from recovery_calibrated_observer import load_grasp_calibration,BoundCalibratedGraspInference


def validate_pilot_receipts(models,identities,repo):
    root=Path(models['root']);spec=models['pilot_receipts']
    if set(spec)!={'calibration','observer_fit_config','observer_fit_result','planner_training_result',
                   'planner_initial_generation','planner_noninitial_generation','scope'}:
        raise ValueError('Explicit fit, generation and unchanged calibration receipts required')
    if spec['scope']!='read_only_pilot_not_default_promotion':raise ValueError('No default deployment authority')
    paths={key:bound(Path(repo) if key=='observer_fit_config' else root,value)
        for key,value in spec.items() if key!='scope'}
    binding=load_grasp_calibration(paths['calibration'],spec['calibration']['sha256'],identities)
    fit_cfg,fit=read(paths['observer_fit_config']),read(paths['observer_fit_result'])
    if (fit.get('status')!='observer_adapter_fit_complete_not_deployed'
            or fit['config_sha256']!=spec['observer_fit_config']['sha256']
            or fit['selected_checkpoint_sha256']!=identities.observer_adapter
            or fit['frozen_before_sha256']!=fit['frozen_after_sha256']
            or fit_cfg['high']!=models['observer_backbone']
            or fit_cfg['stats_sha256']!=identities.observer_normalization
            or fit_cfg.get('history_protocol','cadence16_v1')!='cadence16_v1'
            or fit_cfg.get('include_served_controls',False)):
        raise ValueError('Observer architecture/history/normalizer/selected fit changed')
    training=read(paths['planner_training_result'])
    if (training.get('status')!='completed_finite_schedule' or training['step']!=training['planned_updates']
            or training['frozen_before_sha256']!=training['frozen_after_sha256']
            or training['trainable_groups']!={'planner_vlm':326}):
        raise ValueError('Planner incremental training is incomplete or changed frozen parameters')
    for key,schedule in [('planner_initial_generation','first_fixed_one_per_task_v1'),
                         ('planner_noninitial_generation','noninitial_fixed_one_per_task_v1')]:
        generation=read(paths[key])
        if (generation.get('status')!='completed' or generation['checkpoint_sha256']!=identities.planner
                or generation['target_free'] is not True or generation['no_physical_success_measurement'] is not True
                or generation['original_schedule']!=schedule
                or generation['summary']['original_heldout']['rows']!=100
                or generation['summary']['recovery_dev']['rows']!=42):
            raise ValueError('Wrong/incomplete target-free planner check')
    return dict(calibration=binding,fit_config=fit_cfg,fit=fit,paths=paths)


class LoadedPilotObserver:
    def __init__(self,metadata,models,names,*,parameter_digest):
        import torch
        from g05.utils.training.stage1_model import configuration,restore_model,make_processor
        from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
        from g05.models.g05.helpers.vlm_lora import VLMloraConfig,inject_vlm_lora
        from g05.models.g05.helpers.observer_adapter import require_observer_adapter_only
        self.digest=parameter_digest;self.models=models
        spec=metadata['models'];accepted=metadata['calibrated'];cfg=accepted['fit_config'];fit=accepted['fit']
        root=Path(spec['root']);self.config=configuration(root,'high',names)
        if file_sha(self.config['stats_path'])!=models.observer_normalization:raise ValueError('Observer stats drift')
        saved=torch.load(root/spec['observer_adapter']['path'],map_location='cpu',weights_only=False)
        if (saved['schema']!='recovery_observer_adapter_checkpoint_v1'
                or saved['config_sha256']!=file_sha(accepted['paths']['observer_fit_config'])
                or saved['high_sha256']!=models.observer_backbone or saved['stats_sha256']!=models.observer_normalization
                or saved['admission_sha256']!=cfg['admission_sha256'] or saved['source_commit']!=fit['source_commit']
                or saved['epoch']!=fit['selected_epoch'] or saved['adapter_config']!=cfg['adapter']):
            raise ValueError('Wrong frozen observer checkpoint lineage')
        parent=torch.load(root/spec['observer_backbone']['path'],map_location='cpu',mmap=True,weights_only=False)
        self.policy,_=restore_model(self.config,'high',state=parent['model_state_dict']);del parent;gc.collect()
        self.policy.requires_grad_(False).eval().cuda()
        self.policy.model.vlm=inject_vlm_lora(self.policy.model.vlm,VLMloraConfig.from_mapping(cfg['adapter']))
        whitelist=require_observer_adapter_only(self.policy);params=dict(self.policy.named_parameters())
        if len(whitelist)!=192 or set(whitelist)!=set(saved['adapter_state']):raise ValueError('Wrong observer adapter tensors')
        with torch.no_grad():
            for key,value in saved['adapter_state'].items():
                if params[key].shape!=value.shape or not torch.isfinite(value).all():raise ValueError('Invalid observer adapter value')
                params[key].copy_(value)
                if not torch.equal(params[key].detach().cpu(),value):raise ValueError('Adapter reload changed dtype or bytes')
        if self.digest(self.policy,frozen=True)!=fit['frozen_before_sha256']:raise ValueError('Observer backbone drift')
        self.policy.requires_grad_(False).eval();self.before=self.digest(self.policy,frozen=True)
        self.head=TemporalOutcomeObserver(saved['observer_state']['context_projection.0.weight'].numel(),
            include_absolute_proprio=True,include_served_controls=False).cuda()
        self.head.load_state_dict(saved['observer_state'],strict=True);self.head.requires_grad_(False).eval()
        self.head_reference=saved['observer_state'];del saved;gc.collect()
        if any(not torch.isfinite(t).all() for t in self.head.state_dict().values()):raise ValueError('Nonfinite observer head')
        self.processor=make_processor(self.config,False)

    def bind(self,session):
        return BoundCalibratedGraspInference(self.policy,self.processor,self.config,self.head,session,
            loaded_backbone_sha256=self.models.observer_backbone,loaded_adapter_sha256=self.models.observer_adapter)

    def finish(self):
        import torch
        after=self.digest(self.policy,frozen=True)
        if (after!=self.before or any(not torch.equal(self.head.state_dict()[k].detach().cpu(),v)
                for k,v in self.head_reference.items())
                or any(p.grad is not None or p.requires_grad for m in (self.policy,self.head) for p in m.parameters())):
            raise ValueError('Read-only observer acquired gradients or changed weights')
        return dict(observer_before_sha256=self.before,observer_after_sha256=after,observer_head_unchanged=True)
