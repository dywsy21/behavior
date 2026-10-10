"""Independently reload a completed result-only adapter and replay signed rows.

No optimizer, calibration, planner deployment, or new data-role assignment.
The selected checkpoint is fixed by the completed training receipt, never by
this replay. Re-extract real RGB prefixes instead of trusting training memory.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts/rl/memlite_online/code')]
from recovery_corpus import file_sha, digest
from recovery_features import require_history_protocol
from recovery_sft_data import require_training_pool, CandidateArchiveReader, raw_observation
from recovery_observer_training import OUTCOMES, request_key
from recovery_convergence import event_weights, check_splits


def require_diagnostic_groups(items, exposed, expected):
    groups={r['candidate']['source_group'] for r in items}
    if not groups or groups & exposed or groups != set(expected) or len(expected) != len(set(expected)):
        raise ValueError('Diagnostic groups must exactly match a predeclared disjoint cohort')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--additional-diagnostic',type=Path,
        help='Previously inspected calibration cohort: frozen diagnostic only, no new calibration or model selection')
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); root = Path(cfg['root'])
    fit = json.loads((a.fit/'result.json').read_text())
    if (cfg.get('schema') != 'recovery_observer_adapter_training_v1'
            or fit['status'] != 'observer_adapter_fit_complete_not_deployed'
            or fit['config_sha256'] != file_sha(a.config)
            or fit['frozen_before_sha256'] != fit['frozen_after_sha256']
            or a.output.exists()
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('Complete pinned fit, clean source, and new replay output required')
    path = a.fit/'checkpoints/selected.pt'; expected = fit['selected_checkpoint_sha256']
    if file_sha(path) != expected or file_sha(root/cfg['high']['path']) != cfg['high']['sha256']:
        raise ValueError('Changed selected adapter or high parent')
    receipt, rows = require_training_pool(root/cfg['admission'],'outcome',cfg['admission_sha256'])
    train = [r for r in rows if r['candidate']['split']=='train']
    dev = [r for r in rows if r['candidate']['split']=='dev']; check_splits(train,dev)
    feature_receipt = json.loads((root/cfg['feature_receipt']).read_text())
    if (file_sha(root/cfg['features']) != cfg['features_sha256']
            or feature_receipt['features_sha256'] != cfg['features_sha256']
            or feature_receipt['admission_sha256'] != cfg['admission_sha256']
            or feature_receipt['high_sha256'] != cfg['high']['sha256']
            or feature_receipt['diagnostic_only']):
        raise ValueError('Wrong causal cache binding')
    corpus = root/cfg['corpus']; inventory = json.loads((corpus/'audit/inventory.json').read_text())
    history_receipt = json.loads((corpus/'history/receipt.json').read_text())
    if (digest(inventory) != receipt['inventory_sha256']
            or digest(inventory) != feature_receipt['inventory_sha256']
            or file_sha(corpus/'history/contexts.jsonl') != feature_receipt['history_sha256']
            or file_sha(corpus/'audit/anchors.jsonl') != history_receipt['anchors_sha256']):
        raise ValueError('Changed RGB/history source')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('One idle A800 required; do not displace training')
    import torch
    import torch.nn.functional as F
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.utils.training.stage1_runtime import atomic_json
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    from g05.models.g05.helpers.vlm_lora import VLMloraConfig,inject_vlm_lora
    from g05.models.g05.helpers.observer_adapter import require_observer_adapter_only
    from g05.utils.memlite_causal_feedback import single_frame_member_prefix
    from g05.models.g05.qwen35 import vision
    from train_memlite_recovery import parameter_digest
    torch.set_num_threads(2)
    vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    cache = torch.load(root/cfg['features'],map_location='cpu',weights_only=False)
    protocol = require_history_protocol(cfg,feature_receipt,cache)
    if digest(cache['requests']) != feature_receipt['requests_sha256']:
        raise ValueError('Changed causal requests')
    diagnostic=None
    if a.additional_diagnostic:
        dc=json.loads(a.additional_diagnostic.read_text())
        if (dc.get('schema')!='observer_adapter_prior_cohort_diagnostic_v1'
                or dc['selected_checkpoint_sha256']!=expected
                or dc['fit_result_sha256']!=file_sha(a.fit/'result.json')
                or dc['high_sha256']!=cfg['high']['sha256']
                or dc.get('previously_inspected_not_blind') is not True):
            raise ValueError('Diagnostic must pin this completed selected fit; never select on predictions')
        dr,drows=require_training_pool(root/dc['admission'],'outcome',dc['admission_sha256'],purpose='calibration')
        require_diagnostic_groups(drows,{r['candidate']['source_group'] for r in train+dev},dc['expected_groups'])
        dfr=json.loads((root/dc['feature_receipt']).read_text())
        dcache=torch.load(root/dc['features'],map_location='cpu',weights_only=False)
        if (file_sha(root/dc['features'])!=dc['features_sha256']
                or dfr['features_sha256']!=dc['features_sha256'] or dfr['diagnostic_only']
                or dfr['admission_sha256']!=dc['admission_sha256']
                or dfr['high_sha256']!=cfg['high']['sha256']
                or require_history_protocol(cfg,dfr,dcache)!=protocol
                or digest(dcache['requests'])!=dfr['requests_sha256']):
            raise ValueError('Unbound diagnostic RGB request cache')
        dcorpus=root/dc['corpus'];dinv=json.loads((dcorpus/'audit/inventory.json').read_text())
        dh=json.loads((dcorpus/'history/receipt.json').read_text())
        if (digest(dinv)!=dr['inventory_sha256'] or digest(dinv)!=dfr['inventory_sha256']
                or file_sha(dcorpus/'history/contexts.jsonl')!=dfr['history_sha256']
                or file_sha(dcorpus/'audit/anchors.jsonl')!=dh['anchors_sha256']):
            raise ValueError('Diagnostic observations/history changed')
        diagnostic=dict(rows=drows,requests={r['request_id']:r for r in dcache['requests']},
            anchors={r['sample_id']:r for r in map(json.loads,(dcorpus/'audit/anchors.jsonl').read_text().splitlines())},
            inventory=dinv,corpus=dcorpus)
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    if (checkpoint['schema'] != 'recovery_observer_adapter_checkpoint_v1'
            or checkpoint['high_sha256'] != cfg['high']['sha256']
            or checkpoint['config_sha256'] != file_sha(a.config)
            or checkpoint['admission_sha256'] != cfg['admission_sha256']
            or checkpoint['source_commit'] != fit['source_commit']
            or checkpoint['stats_sha256'] != cfg['stats_sha256']
            or checkpoint['epoch'] != fit['selected_epoch']
            or checkpoint['adapter_config'] != cfg['adapter']):
        raise ValueError('Selected checkpoint does not belong to this fit')
    names = json.loads((root/cfg['expert_release']/'manifest.json').read_text())['task_names']
    config = configuration(root,'high',names)
    if file_sha(config['stats_path']) != cfg['stats_sha256']:
        raise ValueError('Changed normalization')
    a.output.mkdir(parents=True)
    status = dict(status='loading',pid=os.getpid(),optimizer_updates=0,runtime_ready=False,
        selected_checkpoint_sha256=expected,source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),history_protocol=protocol)
    atomic_json(a.output/'status.json',status)
    saved = torch.load(root/cfg['high']['path'],map_location='cpu',mmap=True,weights_only=False)
    policy,_ = restore_model(config,'high',state=saved['model_state_dict']);del saved;gc.collect()
    policy.requires_grad_(False).eval().cuda()
    policy.model.vlm = inject_vlm_lora(policy.model.vlm,VLMloraConfig.from_mapping(cfg['adapter']))
    adapter_names = require_observer_adapter_only(policy)
    if set(adapter_names) != set(checkpoint['adapter_state']) or len(adapter_names) != 192:
        raise ValueError('Adapter tensor whitelist mismatch')
    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for name,tensor in checkpoint['adapter_state'].items():
            if tensor.shape != parameters[name].shape or not torch.isfinite(tensor).all():
                raise ValueError('Invalid saved observer adapter')
            parameters[name].copy_(tensor)
            if not torch.equal(parameters[name].detach().cpu(),tensor):
                raise ValueError('Adapter reload is not exact')
    policy.eval(); frozen_before=parameter_digest(policy,frozen=True)
    if frozen_before != fit['frozen_before_sha256']:
        raise ValueError('Reloaded frozen parent does not match training')
    head = TemporalOutcomeObserver(next(iter(cache['features'].values()))['context'].shape[-1],
        include_absolute_proprio=True).cuda()
    head.load_state_dict(checkpoint['observer_state'],strict=True);head.eval()
    if any(not torch.isfinite(t).all() for t in head.state_dict().values()):
        raise ValueError('Nonfinite reloaded observer')
    adam_steps = sorted({float(v['step']) for v in checkpoint['optimizer']['state'].values()})
    if len(checkpoint['optimizer']['state']) != 206 or adam_steps != [float(checkpoint['step'])]:
        raise ValueError('Incomplete saved adapter/head optimizer')
    requests = {r['request_id']:r for r in cache['requests']}
    anchors = {r['sample_id']:r for r in map(json.loads,(corpus/'audit/anchors.jsonl').read_text().splitlines())}
    processor=make_processor(config,False);reader=CandidateArchiveReader(corpus/'raw',inventory)
    memo={};metrics={};started=time.monotonic()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        splits=[('dev',dev),('train',train)]+([('prior_cohort_diagnostic',diagnostic['rows'])] if diagnostic else [])
        for split,items in splits:
            if split=='prior_cohort_diagnostic':
                reader.close()
                reader=CandidateArchiveReader(diagnostic['corpus']/'raw',diagnostic['inventory'])
                anchors=diagnostic['anchors'];requests=diagnostic['requests']
            logits=[]
            for row in items:
                request=requests[request_key(row)];contexts=[];states=[];steps=[]
                if request['source_group'] != row['candidate']['source_group']:
                    raise ValueError('Cross-group feature request')
                for check in request['checks']:
                    key=digest(check)
                    if key not in memo:
                        prepared=processor._process_tensors(raw_observation(anchors[check['sample_id']],reader,config))
                        prefix=single_frame_member_prefix(processor.samples_builder,prepared,
                            **{k:check[k] for k in ('task_name','parent_goal','issued_bundle','member_index','memory','served_controls')})
                        pixels={k:v.unsqueeze(0).cuda() for k,v in prepared['pixel_values'].items()}
                        context=policy.outcome_context_for_observer_adapter_training([prefix],pixels)[0]
                        memo[key]=(context,prefix['proprio']['value'].reshape(27).cuda())
                    context,state=memo[key];contexts.append(context);states.append(state);steps.append(check['control_step'])
                values=dict(context=torch.stack(contexts)[None],proprio=torch.stack(states)[None],
                    steps=torch.tensor(steps,device='cuda')[None],valid=torch.ones(1,len(steps),dtype=torch.bool,device='cuda'))
                logits.append(head(**values).float().cpu()[0])
            logits=torch.stack(logits);y=torch.tensor([OUTCOMES.index(r['approval']['label']['value']) for r in items])
            pred=logits.argmax(-1);w=torch.tensor(event_weights(items));ce=F.cross_entropy(logits,y,reduction='none')
            per={name:dict(rows=int((y==i).sum()),recall=float((pred[y==i]==i).float().mean()))
                for i,name in enumerate(OUTCOMES) if (y==i).any()}
            metrics[split]=dict(event_weighted_ce=float((w*ce).sum()),
                event_weighted_accuracy=float((w*(pred==y)).sum()),
                balanced_accuracy=sum(v['recall'] for v in per.values())/len(per),per_class=per,
                reviewed_rows=len(items),source_groups=len({r['candidate']['source_group'] for r in items}))
            with (a.output/(split+'-predictions.jsonl')).open('x') as stream:
                for row,logit,prob in zip(items,logits.tolist(),logits.softmax(-1).tolist()):
                    stream.write(json.dumps(dict(sample_id=row['candidate']['sample_id'],
                        source_group=row['candidate']['source_group'],label=row['approval']['label']['value'],
                        request_id=request_key(row),logits=logit,probabilities=dict(zip(OUTCOMES,prob))))+'\n')
    for key in ('event_weighted_ce','event_weighted_accuracy','balanced_accuracy'):
        if abs(metrics['dev'][key]-checkpoint['metrics'][key])>1e-6:
            raise ValueError('Independent selected-checkpoint forward differs: '+key)
    if parameter_digest(policy,frozen=True)!=frozen_before or file_sha(path)!=expected:
        raise ValueError('Replay changed parent/checkpoint')
    if any(p.grad is not None for p in policy.parameters()) or any(p.grad is not None for p in head.parameters()):
        raise ValueError('Read-only replay created gradients')
    status.update(status='independent_reload_and_forward_passed_not_deployed',selected_epoch=checkpoint['epoch'],
        selected_step=checkpoint['step'],adapter_tensors=len(adapter_names),head_tensors=len(head.state_dict()),
        adam_states=206,adam_steps=adam_steps,real_unique_prefills=len(memo),metrics=metrics,
        dev_metrics_reproduced=True,calibration_and_test_not_read=diagnostic is None,
        frozen_test_not_read=True,calibration_fitted=False,base_sha256=frozen_before,
        seconds=time.monotonic()-started)
    if diagnostic:
        status.update(prior_cohort_diagnostic_config_sha256=file_sha(a.additional_diagnostic),
            previously_inspected_not_blind=True,diagnostic_not_used_for_checkpoint_selection=True)
    atomic_json(a.output/'result.json',status);atomic_json(a.output/'status.json',status);reader.close()


if __name__=='__main__':
    main()
