"""Train an isolated outcome-observer LoRA + temporal head on approved rows.

Planner and low controller are not modified. Images/issued-intent history are
re-encoded with gradients; physical labels are CE targets only. This produces
a candidate observer, never runtime permission or a calibrated high planner.
"""
import argparse
from collections import OrderedDict
import gc
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_corpus import file_sha,digest
from recovery_sft_data import require_training_pool,CandidateArchiveReader,raw_observation
from recovery_observer_training import OUTCOMES,request_key
from recovery_convergence import event_weights,check_splits,selection_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    if cfg.get('schema')!='recovery_observer_adapter_training_v1' or cfg.get('user_goal_authorized') is not True:
        raise ValueError('Explicit separate-observer training recipe required')
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New run, clean frozen source required')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('One verified idle A800 required')
    receipt,rows=require_training_pool(root/cfg['admission'],'outcome',cfg['admission_sha256'])
    train=[r for r in rows if r['candidate']['split']=='train'];dev=[r for r in rows if r['candidate']['split']=='dev']
    check_splits(train,dev)
    cache_receipt=json.loads((root/cfg['feature_receipt']).read_text())
    if (file_sha(root/cfg['features'])!=cfg['features_sha256']
            or cache_receipt['features_sha256']!=cfg['features_sha256'] or cache_receipt['diagnostic_only']
            or cache_receipt['high_sha256']!=cfg['high']['sha256']
            or cache_receipt['admission_sha256']!=cfg['admission_sha256']
            or file_sha(root/cfg['high']['path'])!=cfg['high']['sha256']):
        raise ValueError('Unbound parent/admission/cache')
    corpus=root/cfg['corpus'];inventory=json.loads((corpus/'audit/inventory.json').read_text())
    anchors={r['sample_id']:r for r in map(json.loads,(corpus/'audit/anchors.jsonl').read_text().splitlines())}
    history_receipt=json.loads((corpus/'history/receipt.json').read_text())
    if (digest(inventory)!=receipt['inventory_sha256'] or digest(inventory)!=cache_receipt['inventory_sha256']
            or file_sha(corpus/'history/contexts.jsonl')!=cache_receipt['history_sha256']
            or file_sha(corpus/'audit/anchors.jsonl')!=history_receipt['anchors_sha256']):
        raise ValueError('Image/history source changed')
    import torch
    import torch.nn.functional as F
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.utils.training.stage1_runtime import atomic_json,capture_rng,init_wandb,disk_has_reserve
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    from g05.models.g05.helpers.vlm_lora import VLMloraConfig,inject_vlm_lora
    from g05.models.g05.helpers.observer_adapter import require_observer_adapter_only
    from g05.utils.memlite_causal_feedback import single_frame_member_prefix
    from g05.models.g05.qwen35 import vision
    from train_memlite_recovery import parameter_digest
    vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    a.output.mkdir(parents=True);(a.output/'checkpoints').mkdir()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    status=dict(status='loading',pid=os.getpid(),source_commit=commit,config_sha256=file_sha(a.config),
        high_sha256=cfg['high']['sha256'],admission_sha256=cfg['admission_sha256'],optimizer_updates=0,
        runtime_ready=False,planner_and_low_unchanged=True)
    atomic_json(a.output/'status.json',status)
    cache=torch.load(root/cfg['features'],map_location='cpu',weights_only=False)
    if digest(cache['requests'])!=cache_receipt['requests_sha256']:raise ValueError('Changed causal requests')
    requests={r['request_id']:r for r in cache['requests']}
    names=json.loads((root/cfg['expert_release']/'manifest.json').read_text())['task_names']
    config=configuration(root,'high',names)
    if file_sha(config['stats_path'])!=cfg['stats_sha256']:raise ValueError('Changed state normalization')
    saved=torch.load(root/cfg['high']['path'],map_location='cpu',mmap=True,weights_only=False)
    policy,restoration=restore_model(config,'high',state=saved['model_state_dict']);del saved;gc.collect()
    policy.requires_grad_(False).eval().cuda();processor=make_processor(config,False)
    adapter_cfg=VLMloraConfig.from_mapping(cfg['adapter'])
    if not adapter_cfg.enabled or adapter_cfg.adapter_name!='outcome_observer':raise ValueError('Separate named observer adapter required')
    policy.model.vlm=inject_vlm_lora(policy.model.vlm,adapter_cfg)
    adapter_names=require_observer_adapter_only(policy)
    if len(adapter_names)!=192:raise ValueError('Observer adapter module coverage changed')
    policy.eval();frozen_before=parameter_digest(policy,frozen=True)
    # Match the frozen-head experiment's RNG independently of zero-LoRA init.
    torch.manual_seed(cfg['seed'])
    hidden=next(iter(cache['features'].values()))['context'].shape[-1]
    head=TemporalOutcomeObserver(hidden,include_absolute_proprio=True).cuda()
    head.head.load_state_dict(cache['initial_head'],strict=True)
    adapter_parameters=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([dict(params=adapter_parameters,lr=cfg['adapter_learning_rate']),
        dict(params=list(head.parameters()),lr=cfg['head_learning_rate'])],weight_decay=cfg['weight_decay'])
    reader=CandidateArchiveReader(corpus/'raw',inventory);prepared_cache=OrderedDict()
    def feature(row):
        request=requests[request_key(row)];contexts=[];proprio=[];steps=[]
        if request['source_group']!=row['candidate']['source_group']:raise ValueError('Cross-group feature')
        for check in request['checks']:
            sid=check['sample_id']
            if sid not in prepared_cache:
                prepared_cache[sid]=processor._process_tensors(raw_observation(anchors[sid],reader,config))
                if len(prepared_cache)>64:prepared_cache.popitem(last=False)
            prepared=prepared_cache[sid];prepared_cache.move_to_end(sid)
            prefix=single_frame_member_prefix(processor.samples_builder,prepared,
                **{k:check[k] for k in ('task_name','parent_goal','issued_bundle','member_index','memory','served_controls')})
            pixels={k:v.unsqueeze(0).cuda() for k,v in prepared['pixel_values'].items()}
            context=policy.outcome_context_for_observer_adapter_training([prefix],pixels)[0]
            contexts.append(context);proprio.append(prefix['proprio']['value'].reshape(27).cuda());steps.append(check['control_step'])
        return dict(context=torch.stack(contexts)[None],proprio=torch.stack(proprio)[None],
            steps=torch.tensor(steps,device='cuda')[None],valid=torch.ones(1,len(steps),dtype=torch.bool,device='cuda'))
    # Prove zero-adapter equivalence against an already SHA-bound real prefill,
    # then prove real gradients reach only this adapter and the separate head.
    head.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        initial_feature=feature(train[0])
    reference=cache['features'][request_key(train[0])]['context']
    if not torch.equal(initial_feature['context'][0].float().cpu(),reference):
        raise ValueError('Zero observer adapter does not reproduce pinned frozen prefix')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        probe=head(**feature(train[0]),allow_context_grad=True)
        F.cross_entropy(probe.float(),torch.tensor([OUTCOMES.index(train[0]['approval']['label']['value'])],device='cuda')).backward()
    gradients={n:float(p.grad.float().norm()) for n,p in policy.named_parameters() if p.grad is not None}
    if (not gradients or set(gradients)-set(adapter_names) or not any(v>0 for v in gradients.values())
            or not any(p.grad is not None and bool(p.grad.abs().any()) for p in head.parameters())):
        raise ValueError('Actual prefix-gradient insulation probe failed')
    atomic_json(a.output/'gradient-audit.json',dict(zero_adapter_context_bitwise_equal=True,
        adapter_tensors=len(adapter_names),nonzero_adapter_gradients=sum(v>0 for v in gradients.values()),
        adapter_gradient_names=sorted(gradients),all_other_policy_gradients_absent=True,optimizer_steps=0))
    optimizer.zero_grad(set_to_none=True)
    wb=init_wandb(dict(cfg['wandb'],name=a.output.name),a.output,run_id=uuid.uuid4().hex[:12],resume=False,
        metadata=dict(status,train_groups=sorted({r['candidate']['source_group'] for r in train}),
            selection_groups=sorted({r['candidate']['source_group'] for r in dev}),adapter_tensors=192))
    weights=event_weights(train);step=0;started=time.monotonic();stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    def evaluate(epoch):
        head.eval();logits=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for row in dev:logits.append(head(**feature(row)).float().cpu()[0])
        logits=torch.stack(logits);y=torch.tensor([OUTCOMES.index(r['approval']['label']['value']) for r in dev])
        pred=logits.argmax(-1);w=torch.tensor(event_weights(dev));ce=F.cross_entropy(logits,y,reduction='none')
        per={name:dict(rows=int((y==i).sum()),recall=float((pred[y==i]==i).float().mean()))
             for i,name in enumerate(OUTCOMES) if (y==i).any()}
        metrics=dict(event_weighted_ce=float((w*ce).sum()),event_weighted_accuracy=float((w*(pred==y)).sum()),
            balanced_accuracy=sum(v['recall'] for v in per.values())/len(per),per_class=per,
            reviewed_rows=len(dev),source_groups=len({r['candidate']['source_group'] for r in dev}))
        atomic_json(a.output/'progress.json',dict(epoch=epoch,step=step,selection_dev=metrics,runtime_ready=False))
        with (a.output/'evaluations.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,step=step,selection_dev=metrics))+'\n')
        wb.log({'train/update':step,**{f'eval/selection_dev/{k}':metrics[k]
            for k in ('event_weighted_ce','event_weighted_accuracy','balanced_accuracy')}})
        return metrics
    def save(name,epoch,metrics):
        if not disk_has_reserve(root):raise ValueError('Shared disk reserve reached')
        payload=dict(schema='recovery_observer_adapter_checkpoint_v1',step=step,epoch=epoch,
            high_sha256=cfg['high']['sha256'],config_sha256=file_sha(a.config),source_commit=commit,
            stats_sha256=cfg['stats_sha256'],admission_sha256=cfg['admission_sha256'],adapter_config=cfg['adapter'],
            adapter_state={n:p.detach().cpu().clone() for n,p in policy.named_parameters() if n in adapter_names},
            observer_state={n:p.detach().cpu().clone() for n,p in head.state_dict().items()},
            optimizer=optimizer.state_dict(),rng=capture_rng(),metrics=metrics,runtime_ready=False)
        path=a.output/'checkpoints'/name;temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)
        return file_sha(path)
    initial=evaluate(0);best=selection_key(initial);best_epoch=0;best_sha=save('selected.pt',0,initial)
    reason='learning_curve_window';last=initial
    for epoch in range(1,cfg['maximum_epochs']+1):
        order=list(range(len(train)));random.Random(cfg['seed']+epoch).shuffle(order)
        head.train();loss_sum=0.
        for start in range(0,len(order),cfg['global_batch']):
            if stopping[0] or (a.output/'STOP').exists():reason='owner_stop_after_safe_batch';break
            ids=order[start:start+cfg['global_batch']];optimizer.zero_grad(set_to_none=True)
            for i in ids:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=head(**feature(train[i]),allow_context_grad=True)
                    target=torch.tensor([OUTCOMES.index(train[i]['approval']['label']['value'])],device='cuda')
                    loss=F.cross_entropy(output.float(),target)*weights[i]*len(train)/len(ids)
                if not torch.isfinite(loss):raise ValueError('Nonfinite outcome adapter loss')
                loss.backward();loss_sum+=float(loss.detach())
            if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                raise ValueError('Frozen planner/vision/actor received an outcome gradient')
            norm=torch.nn.utils.clip_grad_norm_(adapter_parameters+list(head.parameters()),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            status.update(status='training',optimizer_updates=step,epoch=epoch,wandb_url=wb.url,updated_unix=time.time())
            atomic_json(a.output/'status.json',status)
            wb.log({'train/update':step,'train/gradient_norm':float(norm),'train/epoch':epoch})
        last=evaluate(epoch)
        if selection_key(last)>best:
            best=selection_key(last);best_epoch=epoch;best_sha=save('selected.pt',epoch,last)
        save(f'epoch-{epoch:04d}.pt',epoch,last)
        wb.log({'train/update':step,'train/outcome_ce_estimate':loss_sum/max(1,(len(order)+cfg['global_batch']-1)//cfg['global_batch'])})
        if reason=='owner_stop_after_safe_batch':break
        if epoch>=cfg['minimum_epochs'] and epoch-best_epoch>=cfg['patience_epochs']:
            reason='selection_dev_plateau';break
    frozen_after=parameter_digest(policy,frozen=True)
    if frozen_before!=frozen_after:raise ValueError('Frozen observer base/planner/vision/actor changed')
    status.update(status='observer_adapter_fit_complete_not_deployed',stop_reason=reason,epoch=epoch,
        selected_epoch=best_epoch,selected_checkpoint_sha256=best_sha,initial=initial,last=last,
        frozen_before_sha256=frozen_before,frozen_after_sha256=frozen_after,seconds=time.monotonic()-started,
        original_planner_checkpoint_unchanged=True,independent_calibration_test_not_read=True)
    atomic_json(a.output/'status.json',status);atomic_json(a.output/'result.json',status);reader.close();wb.finish()


if __name__=='__main__':main()
