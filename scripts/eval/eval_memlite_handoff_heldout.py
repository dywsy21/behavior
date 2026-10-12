"""Target-free planner generation on preregistered original-EVAL real feedback.

Independent shards preserve identical inputs; the existing strict generation
merger can combine them. This measures annotation-command decisions, not robot
success, and neither updates the planner nor executes a single robot action.
"""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/infra'),
               str(REPO/'scripts/eval'),str(REPO/'scripts/eval/memlite_sft100'),
               str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import bound, read, write_new
from recovery_corpus import digest, file_sha
from recovery_handoff_heldout import (generation_split, replay_heldout_projection,
                                     selection_bucket, validate_fixed_fit)
from recovery_handoff_inference import tree_sha, unique_checks
from recovery_generation_metrics import score_event, summarize
from recovery_generation_shards import generation_shard
from recovery_process_cache import isolated_cache_env


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--model',choices=('control','candidate'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--num-shards',type=int,default=1)
    ap.add_argument('--shard-index',type=int,default=0)
    a=ap.parse_args(); cfg=read(a.config); root=Path(cfg['root'])
    if (cfg['schema']!='real_handoff_heldout_planner_generation_v1'
            or cfg['training_admission_forbidden'] is not True or cfg['optimizer_updates']!=0
            or a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('Require read-only registered evaluation, fresh output, clean frozen source')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Select one actually idle registered-node GPU')
    gpath=bound(REPO,cfg['generation_config']); generation=read(gpath)
    if generation_split(generation)!='eval' or generation['root']!=cfg['root']:
        raise ValueError('Only original heldout input generation is eligible')
    prediction_path=bound(root,cfg['prediction']); prediction=read(prediction_path)
    from render_memlite_handoff_review import validate_generation_receipt
    validate_generation_receipt(prediction,generation,file_sha(gpath))
    if (prediction['purpose']!='heldout_eval_only' or prediction['data_split']!='eval'
            or prediction['training_admission_forbidden'] is not True):
        raise ValueError('Wrong real-feedback split')
    run=prediction_path.parent
    for name in ('checks','trace','feedback'):
        if file_sha(run/(name+'.jsonl'))!=prediction[name+'_sha256']:
            raise ValueError('Changed original frozen feedback evidence')
    review=read(bound(root,cfg['owner_review']))
    if (review['status']!='REVIEWED_FOR_HELDOUT_EVALUATION_ONLY'
            or review['generation_sha256']!=cfg['prediction']['sha256']
            or review['training_admission_forbidden'] is not True
            or review['human_review_complete'] is not True or review['personally_reviewed_cases']<160):
        raise ValueError('Root personal image/target review is missing')
    spec=cfg['models'][a.model]; checkpoint=bound(root,spec['checkpoint'])
    fit=read(bound(root,spec['fit_result'])); latest=read(bound(root,spec['latest']))
    validate_fixed_fit(fit,latest,spec['checkpoint'])
    a.output.mkdir(parents=True); os.environ.update(isolated_cache_env(a.output))
    started=time.monotonic()
    import torch
    torch.set_num_threads(2); torch.manual_seed(17)
    from generate_memlite_handoff_feedback import source_inputs
    context=source_inputs(generation)
    requests=context.rows
    feedback_rows=[json.loads(x) for x in (run/'feedback.jsonl').read_text().splitlines()]
    feedback={r['sample_id']:r for r in feedback_rows}; trace=defaultdict(list)
    for r in (json.loads(x) for x in (run/'trace.jsonl').read_text().splitlines()):
        trace[r['decision_sample_id']].append({k:v for k,v in r.items() if k!='decision_sample_id'})
    if (len(feedback_rows)!=len(requests) or len(feedback)!=len(requests)
            or set(feedback)!=set(trace) or set(feedback)!={r['sample_id'] for r in requests}):
        raise ValueError('Missing/duplicate/filtered actual feedback')
    checks={(r['sample_id'],r['check_sha256']):r for r in
            (json.loads(x) for x in (run/'checks.jsonl').read_text().splitlines())}
    projections={}
    for r in requests:
        ep,*_=context.data.locate(r['original_identity']['original_candidate_index'])
        projections[r['sample_id']]=replay_heldout_projection(r,feedback[r['sample_id']],ep,
            trace[r['sample_id']],context.calibration)
    schedule=[('heldout_'+selection_bucket(r['target_change_class']),i) for i,r in enumerate(requests)]
    local,sharding=generation_shard(schedule,a.shard_index,a.num_shards)
    from g05.utils.memlite_skill_protocol import planner_input_projection
    from g05.utils.memlite_causal_feedback import single_frame_planner_prefix
    from recovery_causal_inference import observable_raw
    from check_memlite_periodic_processor import make_token_processor
    tokens=make_token_processor(context.data.config); samples=[]; cpu_audit=[]
    for global_index,split,index in local:
        r=requests[index]; sid=r['sample_id']; i=r['original_identity']
        raw,identity=context.data.raw(i['original_candidate_index'])
        ep,frame,*_=context.data.locate(i['original_candidate_index'])
        observed=dict(images={k:v[0].numpy().copy() for k,v in raw['images'].items()},
            proprio=context.data._raw_arrays(ep['row'])['observation.state'][frame].copy())
        check=unique_checks(r)[-1]; original_check=checks[sid,digest(check)]
        if check['frame']!=frame or tree_sha(observed)!=original_check['observation_sha256'] or identity!=original_check['source_identity']:
            raise ValueError('Planner image/state differs from the evaluated observer observation')
        prepared=context.processor._process_tensors(observable_raw(observed,r['original_projection']['task_name'],context.data.config))
        target=projections[sid]; causal=planner_input_projection(target)
        prefix=single_frame_planner_prefix(context.processor.samples_builder,prepared,causal)
        poisoned=dict(prepared,model_projection={'active_skills_text':'FUTURE_TARGET'},reward=123456,
            action=torch.full((32,27),float('nan')),outcome_target='SUCCEEDED')
        if tree_sha(prefix)!=tree_sha(single_frame_planner_prefix(context.processor.samples_builder,poisoned,causal)):
            raise ValueError('Target or privileged/action field entered high prefix')
        encoded=tokens.encode_inference([prefix],device=torch.device('cpu'),mode='ar',training=False)
        audit=dict(sample_id=sid,observation_sha256=tree_sha(observed),causal_input_sha256=digest(causal),
            prefix_sha256=tree_sha(prefix),encoded_sha256=tree_sha(encoded),pixels_sha256=tree_sha(prepared['pixel_values']))
        cpu_audit.append(audit)
        samples.append((global_index,split,index,r,target,causal,prefix,prepared['pixel_values'],audit))
    write_new(a.output/'cpu-prefix-audit.json',dict(rows=cpu_audit,optimizer_updates=0,targets_not_in_prefix=True))
    del tokens; gc.collect()
    from recovery_cuda_startup import initialize_cuda_backend
    from g05.models.g05.qwen35 import vision
    from g05.utils.training.stage1_model import restore_model
    from sparse_cache import indexed_cache_freeze
    from g05.models.kv_cache import SparseKVCache
    from train_memlite_recovery import parameter_digest
    if subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid',
            '--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU claimed during CPU validation; preserve other jobs')
    initialize_cuda_backend(0); vision._flash_attn_varlen=None; vision._flash_attn_backend=None
    def forbidden(*args,**kwargs): raise RuntimeError('Optimizer forbidden in heldout generation')
    torch.optim.Adam.step=torch.optim.AdamW.step=forbidden
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if saved['state']['step']!=360: raise ValueError('Saved tensor payload is not fixed final360')
    model,restore=restore_model(context.data.config,'high',state=saved['model_state_dict'])
    del saved; gc.collect()
    model.requires_grad_(False).eval().cuda(); before=parameter_digest(model,frozen=True)
    receipt=dict(config_sha256=file_sha(a.config),checkpoint_sha256=spec['checkpoint']['sha256'],
        fit_result_sha256=spec['fit_result']['sha256'],source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),restoration=restore,
        observer_prediction_sha256=cfg['prediction']['sha256'],requests_sha256=generation['requests']['sha256'],
        owner_review_sha256=cfg['owner_review']['sha256'],schedule_sha256=digest(schedule),sharding=sharding,
        target_free=True,no_physical_success_measurement=True,training_admission_forbidden=True,
        optimizer_updates=0,model_before_sha256=before)
    write_new(a.output/'manifest.json',receipt); rows=[]
    for global_index,split,index,r,target,causal,prefix,pixels,audit in samples:
        actual=model.processor.encode_inference([prefix],device=torch.device('cpu'),mode='ar',training=False)
        if tree_sha(actual)!=audit['encoded_sha256']: raise ValueError('Loaded high model prefix tokens changed')
        event=error=text=None; tick=time.monotonic()
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),indexed_cache_freeze(model.model.ar_helper,SparseKVCache):
                result=model.generate_high_level([prefix],{k:v[None].cuda() for k,v in pixels.items()},temperature=0.)
            event=result['planner_events'][0]; text=result.get('high_level_text')
        except (ValueError,RuntimeError) as exc:
            if isinstance(exc,torch.cuda.OutOfMemoryError): raise
            error=str(exc); text=getattr(exc,'planner_texts',None)
        row=dict(split=split,global_schedule_index=global_index,scheduled_sample_index=index,
            identity=r['original_identity'],sample_id=r['sample_id'],source_group=r['source_group'],
            probe_kind='real_frozen_feedback_heldout',actual_causal_input_sha256=digest(causal),
            original_causal_input_sha256=digest(planner_input_projection(r['original_projection'])),
            input_audit=audit,previous_intent=causal['previous_intent'],
            estimated_bundle_outcome=json.loads(causal['execution_feedback'])['estimated_bundle_outcome'],
            target_change_class=r['target_change_class'],score=score_event(event,target),event=event,error=error,
            text=text,seconds=time.monotonic()-tick,expected={k:target[k] for k in
                ('next_decision','target_parent_goal','active_skills_semantic_json','memory_update')})
        rows.append(row)
        with (a.output/'rows.jsonl').open('a') as stream: stream.write(json.dumps(row,sort_keys=True)+'\n')
        print(json.dumps(dict(completed=len(rows),total=len(samples),split=split,score=row['score'])),flush=True)
    after=parameter_digest(model,frozen=True)
    if after!=before: raise ValueError('Read-only high model parameters changed')
    write_new(a.output/'result.json',dict(receipt,status='completed',rows=len(rows),summary=summarize(rows),
        seconds=time.monotonic()-started,model_after_sha256=after,cpu_prefix_audit_sha256=file_sha(a.output/'cpu-prefix-audit.json')))


if __name__=='__main__': main()
