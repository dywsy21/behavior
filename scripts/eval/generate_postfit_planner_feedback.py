"""Generate immutable, source-disjoint H1 inputs with the fixed calibrated v11.

No optimizer, calibration fit, target-token prefill, physical control or runtime
deployment. Every admitted target remains, including uncertain/wrong estimates.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_corpus import digest,file_sha
from recovery_sft_data import require_training_pool,CandidateArchiveReader,raw_observation
from recovery_calibration_launch import bound,read,write_new
from recovery_postfit import calibration_gate
from recovery_postfit_feedback import clock_requests,predicted_execution_feedback,make_provenance,STATUS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=read(a.config);root=Path(cfg['root'])
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New output and clean frozen source required')
    exposure_path=bound(root,cfg['exposure']);exposure=read(exposure_path)
    bindings=exposure['bindings']
    for b in bindings.values():
        if file_sha(b['path'])!=b['sha256']:raise ValueError('Fixed observer exposure binding changed')
    fit=read(bindings['fit_result']['path']);fit_cfg=read(bindings['fit_config']['path'])
    calibration=read(bindings['calibration']['path']);calibration_gate(calibration)
    high_path=bound(root,cfg['planner_high'])
    if (fit['selected_checkpoint_sha256']!=exposure['observer_sha256']
            or calibration['observer_sha256']!=exposure['observer_sha256']
            or fit_cfg['high']['sha256']!=exposure['high_sha256']
            or calibration['high_sha256']!=exposure['high_sha256']
            or sorted(calibration['source_groups'])!=sorted(exposure['calibration_groups'])
            or fit['frozen_before_sha256']!=fit['frozen_after_sha256']):
        raise ValueError('Observer backbone, adapter or calibration identity drift')
    corpus=root/cfg['corpus'];receipt,rows=require_training_pool(corpus/'admission','planner',cfg['admission_sha256'])
    inventory=read(corpus/'audit/inventory.json');hr=read(corpus/'history/receipt.json')
    if (digest(inventory)!=receipt['inventory_sha256'] or digest(inventory)!=hr['inventory_sha256']
            or file_sha(corpus/'audit/anchors.jsonl')!=hr['anchors_sha256']
            or file_sha(corpus/'history/contexts.jsonl')!=hr['contexts_sha256']):
        raise ValueError('Changed original causal observations/history')
    anchors=[json.loads(x) for x in (corpus/'audit/anchors.jsonl').read_text().splitlines()]
    histories=[json.loads(x) for x in (corpus/'history/contexts.jsonl').read_text().splitlines()]
    selected=[r['candidate'] for r in rows];requests=clock_requests(anchors,histories,selected,exposure)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('One actually idle GPU required')
    import torch
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    from g05.models.g05.helpers.vlm_lora import VLMloraConfig,inject_vlm_lora
    from g05.models.g05.helpers.observer_adapter import require_observer_adapter_only
    from g05.utils.memlite_causal_feedback import single_frame_member_prefix
    from g05.models.g05.qwen35 import vision
    from train_memlite_recovery import parameter_digest
    torch.set_num_threads(2);vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    saved=torch.load(bindings['selected_observer']['path'],map_location='cpu',weights_only=False)
    if (saved['schema']!='recovery_observer_adapter_checkpoint_v1' or saved['config_sha256']!=fit['config_sha256']
            or saved['high_sha256']!=exposure['high_sha256'] or saved['stats_sha256']!=fit_cfg['stats_sha256']
            or saved['admission_sha256']!=fit_cfg['admission_sha256'] or saved['source_commit']!=fit['source_commit']
            or saved['epoch']!=fit['selected_epoch'] or saved['adapter_config']!=fit_cfg['adapter']
            or len(saved['optimizer']['state'])!=206
            or {float(v['step']) for v in saved['optimizer']['state'].values()}!={float(saved['step'])}):
        raise ValueError('Wrong fixed observer checkpoint')
    names=read(root/fit_cfg['expert_release']/'manifest.json')['task_names'];config=configuration(root,'high',names)
    if file_sha(config['stats_path'])!=fit_cfg['stats_sha256']:raise ValueError('Wrong observer normalization')
    parent_path=bound(root,fit_cfg['high'])
    a.output.mkdir(parents=True);started=time.monotonic()
    status=dict(status='loading',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        config_sha256=file_sha(a.config),optimizer_updates=0,runtime_deployed=False,
        planner_high_sha256=cfg['planner_high']['sha256'],observer_backbone_sha256=exposure['high_sha256'],
        observer_sha256=exposure['observer_sha256'],admission_sha256=cfg['admission_sha256'],
        exposure_sha256=file_sha(exposure_path),requests_sha256=digest(requests))
    write_new(a.output/'status.json',status);write_new(a.output/'requests.json',requests)
    parent=torch.load(parent_path,map_location='cpu',mmap=True,weights_only=False)
    policy,_=restore_model(config,'high',state=parent['model_state_dict']);del parent;gc.collect()
    policy.requires_grad_(False).eval().cuda()
    policy.model.vlm=inject_vlm_lora(policy.model.vlm,VLMloraConfig.from_mapping(fit_cfg['adapter']))
    names=require_observer_adapter_only(policy);params=dict(policy.named_parameters())
    if len(names)!=192 or set(names)!=set(saved['adapter_state']):raise ValueError('Wrong observer adapter whitelist')
    with torch.no_grad():
        for name,tensor in saved['adapter_state'].items():
            if params[name].shape!=tensor.shape or not torch.isfinite(tensor).all():raise ValueError('Bad saved tensor')
            params[name].copy_(tensor)
            if not torch.equal(params[name].detach().cpu(),tensor):raise ValueError('Observer reload differs')
    if parameter_digest(policy,frozen=True)!=fit['frozen_before_sha256']:raise ValueError('Observer base changed')
    policy.requires_grad_(False).eval();before=parameter_digest(policy,frozen=True)
    head=TemporalOutcomeObserver(saved['observer_state']['context_projection.0.weight'].numel(),
        include_absolute_proprio=True,include_served_controls=False).cuda()
    if fit_cfg.get('include_served_controls',False):raise ValueError('Unsupported clock-as-head-feature contract')
    head.load_state_dict(saved['observer_state'],strict=True);head.requires_grad_(False).eval()
    processor=make_processor(config,False);reader=CandidateArchiveReader(corpus/'raw',inventory)
    by_id={r['sample_id']:r for r in anchors};history={r['sample_id']:r for r in histories}
    memo={};trace=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for request in requests:
            contexts=[];states=[];steps=[]
            for check in request['checks']:
                key=digest(check)
                if key not in memo:
                    prepared=processor._process_tensors(raw_observation(by_id[check['sample_id']],reader,config))
                    prefix=single_frame_member_prefix(processor.samples_builder,prepared,
                        **{k:check[k] for k in ('task_name','parent_goal','issued_bundle','member_index','memory','served_controls')})
                    pixels={k:v.unsqueeze(0).cuda() for k,v in prepared['pixel_values'].items()}
                    context=policy.outcome_context_from_prefix([prefix],pixels)[0]
                    memo[key]=(context,prefix['proprio']['value'].reshape(27).cuda())
                context,state=memo[key];contexts.append(context);states.append(state);steps.append(check['control_step'])
            values=dict(context=torch.stack(contexts)[None],proprio=torch.stack(states)[None],
                steps=torch.tensor(steps,device='cuda')[None],valid=torch.ones(1,len(steps),dtype=torch.bool,device='cuda'))
            logits=head(**values).float().cpu()[0]
            if logits.shape!=(4,) or not torch.isfinite(logits).all():raise ValueError('Nonfinite observer logits')
            trace.append(dict(request,logits=logits.tolist()))
    if (parameter_digest(policy,frozen=True)!=before or file_sha(high_path)!=cfg['planner_high']['sha256']
            or any(not torch.equal(head.state_dict()[n].detach().cpu(),v) for n,v in saved['observer_state'].items())
            or any(p.requires_grad or p.grad is not None for m in (policy,head) for p in m.parameters())):
        raise ValueError('Read-only observer/model identity mutated')
    write_new(a.output/'prediction-trace.json',trace)
    provenance=make_provenance(exposure,high_sha256=cfg['planner_high']['sha256'],exposure_sha256=file_sha(exposure_path),
        calibration_sha256=bindings['calibration']['sha256'],trace_sha256=file_sha(a.output/'prediction-trace.json'),
        target_groups=[r['source_group'] for r in selected])
    feedback=[dict(sample_id=r['sample_id'],source_group=r['source_group'],control_step=r['control_step'],
        execution_feedback=predicted_execution_feedback(r,history[r['sample_id']],trace,calibration),provenance=provenance)
        for r in selected]
    with (a.output/'feedback.jsonl').open('x') as stream:
        for r in feedback:stream.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    result=dict(status,status=STATUS,diagnostic_only=False,high_sha256=cfg['planner_high']['sha256'],
        observer_weights_updated=False,calibration_refit=False,known_truth_not_added=True,
        feedback_sha256=file_sha(a.output/'feedback.jsonl'),prediction_trace_sha256=file_sha(a.output/'prediction-trace.json'),
        calibration_sha256=bindings['calibration']['sha256'],rows=len(feedback),prefills=len(memo),real_checks=len(trace),
        outcomes_by_split={split:dict(Counter(json.loads(f['execution_feedback'])['estimated_bundle_outcome']
            for f,r in zip(feedback,selected) if r['split']==split)) for split in ('train','dev')},
        frozen_before_sha256=before,frozen_after_sha256=before,seconds=time.monotonic()-started,
        all_admitted_rows_retained=True,scope='GRASP first-attempt intent training input, not outcome labels or policy success')
    write_new(a.output/'feedback_receipt.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
