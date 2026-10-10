"""Read-only prospective calibration of a FIXED result LoRA and temporal head.

Only the 90 source/phase anchors frozen before any prediction are forwarded.
Other supported pool sources and reserved test images are not evaluated. The
temperature fit never updates the observer, planner, low actor or checkpoint.
Even passing GRASP calibration does not authorize other skills or deployment.
"""
import argparse
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
from recovery_features import feature_requests,HISTORY_PROTOCOLS
from recovery_sft_data import require_training_pool,CandidateArchiveReader,raw_observation
from recovery_prospective_adapter import require_prospective_selection,pool_sources
from recovery_observer_training import OUTCOMES,request_key,calibrate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New calibration output and clean frozen source required')
    def read_bound(path,sha):
        path=root/path
        if file_sha(path)!=sha:raise ValueError('Changed immutable artifact: '+str(path))
        return json.loads(path.read_text())
    fit=read_bound(cfg['fit_result'],cfg['fit_result_sha256'])
    fit_cfg=read_bound(cfg['fit_config'],fit['config_sha256'])
    if (fit.get('status')!='observer_adapter_fit_complete_not_deployed'
            or fit['selected_checkpoint_sha256']!=cfg['selected_observer_sha256']
            or fit['frozen_before_sha256']!=fit['frozen_after_sha256']
            or fit_cfg['high']['sha256']!=cfg['high_sha256']):
        raise ValueError('Use the completed preselected adapter with its original high backbone')
    path=root/cfg['selected_observer'];parent=root/fit_cfg['high']['path']
    if file_sha(path)!=cfg['selected_observer_sha256'] or file_sha(parent)!=cfg['high_sha256']:
        raise ValueError('Changed fixed observer/high weights')
    pool=read_bound(cfg['source_pool'],cfg['source_pool_sha256'])
    selection=read_bound(cfg['anchor_selection'],cfg['anchor_selection_sha256'])
    _,old=require_training_pool(root/fit_cfg['admission'],'outcome',fit_cfg['admission_sha256'])
    receipt,rows=require_training_pool(root/cfg['admission'],'outcome',cfg['admission_sha256'],purpose='calibration')
    partition=json.loads((root/cfg['admission']/receipt['evaluation_partition_file']).read_text())
    cohorts=[]
    if len(cfg['cohorts'])!=2:raise ValueError('Two predeclared collection waves required')
    for item,key in zip(cfg['cohorts'],('first_cohort_config','supplement_config')):
        cohorts.append((read_bound(item['path'],item['sha256']),item['sha256'],file_sha(REPO/pool[key])))
    provenance=pool_sources(pool,cohorts)
    if (partition['source_cohort_by_group']!=provenance['source_cohort_by_group']
            or selection['declared_groups']!=provenance['declared_groups']
            or sorted(selection['unavailable_sources'],key=lambda x:x['source_group'])!=sorted(
                provenance['unavailable_sources'],key=lambda x:x['source_group'])
            or selection['admission_sha256']!=cfg['admission_sha256']):
        raise ValueError('Changed original attempts, signed roles or selected release')
    chosen=require_prospective_selection(cfg,pool,selection,partition,rows,
        {r['candidate']['source_group'] for r in old})
    corpus=root/cfg['corpus'];inventory=json.loads((corpus/'audit/inventory.json').read_text())
    history_receipt=json.loads((corpus/'history/receipt.json').read_text())
    if (digest(inventory)!=receipt['inventory_sha256'] or digest(inventory)!=history_receipt['inventory_sha256']
            or file_sha(corpus/'audit/anchors.jsonl')!=history_receipt['anchors_sha256']
            or file_sha(corpus/'history/contexts.jsonl')!=history_receipt['contexts_sha256']):
        raise ValueError('Original image or causal command history changed')
    anchors=[json.loads(x) for x in (corpus/'audit/anchors.jsonl').read_text().splitlines()]
    history=[json.loads(x) for x in (corpus/'history/contexts.jsonl').read_text().splitlines()]
    protocol=fit_cfg.get('history_protocol','cadence16_v1')
    if protocol not in HISTORY_PROTOCOLS:raise ValueError('Unknown fit history protocol')
    requests=feature_requests(anchors,history,[(r['candidate']['sample_id'],
        r['approval']['label'].get('history_role','observable'),r['approval']['label']['member_index'])
        for r in chosen],history_protocol=protocol)
    selected_groups={r['candidate']['source_group'] for r in chosen}
    for request in requests:
        if request['source_group'] not in selected_groups:raise ValueError('Forwarding an unselected group')
        for check in request['checks']:
            if json.loads(check['issued_bundle'])[check['member_index']]['verb']!='GRASP':
                raise ValueError('This prospective calibration only measures GRASP phases')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('One actually idle A800 required; do not displace another job')
    import torch
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.utils.training.stage1_runtime import atomic_json
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    from g05.models.g05.helpers.vlm_lora import VLMloraConfig,inject_vlm_lora
    from g05.models.g05.helpers.observer_adapter import require_observer_adapter_only
    from g05.utils.memlite_causal_feedback import single_frame_member_prefix
    from g05.models.g05.qwen35 import vision
    from train_memlite_recovery import parameter_digest
    torch.set_num_threads(2);vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if (checkpoint['schema']!='recovery_observer_adapter_checkpoint_v1'
            or checkpoint['high_sha256']!=cfg['high_sha256']
            or checkpoint['config_sha256']!=fit['config_sha256']
            or checkpoint['admission_sha256']!=fit_cfg['admission_sha256']
            or checkpoint['source_commit']!=fit['source_commit']
            or checkpoint['stats_sha256']!=fit_cfg['stats_sha256']
            or checkpoint['epoch']!=fit['selected_epoch']
            or checkpoint['adapter_config']!=fit_cfg['adapter']):
        raise ValueError('Checkpoint is not the completed preselected fit')
    adam_steps={float(v['step']) for v in checkpoint['optimizer']['state'].values()}
    if len(checkpoint['optimizer']['state'])!=206 or adam_steps!={float(checkpoint['step'])}:
        raise ValueError('Preselected fit has incomplete adapter/head optimizer provenance')
    names=json.loads((root/fit_cfg['expert_release']/'manifest.json').read_text())['task_names']
    config=configuration(root,'high',names)
    if file_sha(config['stats_path'])!=fit_cfg['stats_sha256']:raise ValueError('State normalization changed')
    a.output.mkdir(parents=True)
    status=dict(status='loading',pid=os.getpid(),optimizer_updates=0,runtime_ready=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        config_sha256=file_sha(a.config),selected_observer_sha256=cfg['selected_observer_sha256'],
        high_sha256=cfg['high_sha256'],anchor_selection_sha256=cfg['anchor_selection_sha256'],
        admission_sha256=cfg['admission_sha256'],source_pool_sha256=cfg['source_pool_sha256'],
        declared_source_count=selection['declared_source_count'],
        unavailable_source_count=selection['unavailable_source_count'],
        reviewed_source_count=selection['reviewed_source_count'],
        reserved_groups_not_predicted=selection['reserved_groups'],frozen_test_not_read=True,
        selected_sources=90,history_protocol=protocol,requests_sha256=digest(requests))
    atomic_json(a.output/'status.json',status);atomic_json(a.output/'requests.json',requests)
    saved=torch.load(parent,map_location='cpu',mmap=True,weights_only=False)
    policy,_=restore_model(config,'high',state=saved['model_state_dict']);del saved;gc.collect()
    policy.requires_grad_(False).eval().cuda()
    policy.model.vlm=inject_vlm_lora(policy.model.vlm,VLMloraConfig.from_mapping(fit_cfg['adapter']))
    adapter_names=require_observer_adapter_only(policy);parameters=dict(policy.named_parameters())
    if set(adapter_names)!=set(checkpoint['adapter_state']) or len(adapter_names)!=192:
        raise ValueError('Result-only adapter tensor whitelist changed')
    with torch.no_grad():
        for name,tensor in checkpoint['adapter_state'].items():
            if parameters[name].shape!=tensor.shape or not torch.isfinite(tensor).all():
                raise ValueError('Malformed saved observer adapter')
            parameters[name].copy_(tensor)
            if not torch.equal(parameters[name].detach().cpu(),tensor):raise ValueError('Adapter reload differs')
    policy.eval();frozen_before=parameter_digest(policy,frozen=True)
    if frozen_before!=fit['frozen_before_sha256']:raise ValueError('Reloaded frozen parent changed')
    hidden=checkpoint['observer_state']['context_projection.0.weight'].numel()
    head=TemporalOutcomeObserver(hidden,include_absolute_proprio=True,
        include_served_controls=fit_cfg.get('include_served_controls',False)).cuda()
    head.load_state_dict(checkpoint['observer_state'],strict=True);head.eval()
    if any(not torch.isfinite(t).all() for t in head.state_dict().values()):raise ValueError('Nonfinite result head')
    by_id={r['sample_id']:r for r in anchors};by_request={r['request_id']:r for r in requests}
    if len(by_request)!=len(chosen):raise ValueError('Duplicate or missing preselected request')
    reader=CandidateArchiveReader(corpus/'raw',inventory);processor=make_processor(config,False)
    memo={};logits=[];started=time.monotonic()
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for row in chosen:
                request=by_request[request_key(row)];contexts=[];states=[];steps=[]
                for check in request['checks']:
                    key=digest(check)
                    if key not in memo:
                        prepared=processor._process_tensors(raw_observation(by_id[check['sample_id']],reader,config))
                        prefix=single_frame_member_prefix(processor.samples_builder,prepared,
                            **{k:check[k] for k in ('task_name','parent_goal','issued_bundle','member_index','memory','served_controls')})
                        pixels={k:v.unsqueeze(0).cuda() for k,v in prepared['pixel_values'].items()}
                        context=policy.outcome_context_for_observer_adapter_training([prefix],pixels)[0]
                        memo[key]=(context,prefix['proprio']['value'].reshape(27).cuda())
                    context,state=memo[key];contexts.append(context);states.append(state);steps.append(check['control_step'])
                values=dict(context=torch.stack(contexts)[None],proprio=torch.stack(states)[None],
                    steps=torch.tensor(steps,device='cuda')[None],valid=torch.ones(1,len(steps),dtype=torch.bool,device='cuda'))
                if head.include_served_controls:
                    values['served_controls']=torch.tensor([c['served_controls'] for c in request['checks']],device='cuda')[None]
                logits.append(head(**values).float().cpu()[0])
        if (parameter_digest(policy,frozen=True)!=frozen_before or file_sha(path)!=cfg['selected_observer_sha256']
                or any(not torch.equal(parameters[n].detach().cpu(),t) for n,t in checkpoint['adapter_state'].items())
                or any(not torch.equal(head.state_dict()[n].detach().cpu(),t) for n,t in checkpoint['observer_state'].items())
                or any(p.grad is not None for p in list(policy.parameters())+list(head.parameters()))):
            raise ValueError('Read-only calibration changed weights or created gradients')
        logits=torch.stack(logits);labels=torch.tensor([OUTCOMES.index(r['approval']['label']['value']) for r in chosen])
        predictions=[dict(sample_id=r['candidate']['sample_id'],source_group=r['candidate']['source_group'],
            label=r['approval']['label']['value'],request_id=request_key(r),logits=v)
            for r,v in zip(chosen,logits.tolist())]
        atomic_json(a.output/'predictions.json',predictions)
        calibration=calibrate(logits,labels,[r['candidate']['source_group'] for r in chosen],minimum_confidence=.85)
        calibration.update(high_sha256=cfg['high_sha256'],observer_sha256=cfg['selected_observer_sha256'],
            anchor_selection_sha256=cfg['anchor_selection_sha256'],cohort_sha256=cfg['source_pool_sha256'],
            mechanism='GRASP',only_class_balanced_phase_support=True,certifies_other_mechanisms=False,
            accuracy_conditioned_on_manually_supported_sources=True,certifies_all_declared_sources=False)
        atomic_json(a.output/'calibration.json',calibration)
        status.update(status='completed_prospective_adapter_calibration_not_deployed',
            calibration_fitted=True,calibration=calibration,calibration_sha256=file_sha(a.output/'calibration.json'),
            predictions_sha256=file_sha(a.output/'predictions.json'),real_unique_prefills=len(memo),
            frozen_before_sha256=frozen_before,frozen_after_sha256=frozen_before,
            uncalibrated_accuracy=float((logits.argmax(-1)==labels).float().mean()),
            seconds=time.monotonic()-started,no_physical_policy_success_measurement=True)
        atomic_json(a.output/'result.json',status);atomic_json(a.output/'status.json',status)
    finally:reader.close()


if __name__=='__main__':main()
