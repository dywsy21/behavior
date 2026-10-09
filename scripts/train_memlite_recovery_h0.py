"""H0 cached-feature heads -> held-group calibration -> causal OOF feedback.

An actual training program, NOT run by preparation. Three fold heads get one
event pass each; the final head gets three. An event participates in at most
two folds, so aggregate exposure never exceeds five. All stages share one
1000-update / four-hour (or smaller approved) budget and checkpoint cursor.
"""
import argparse
from collections import Counter
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
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import digest,file_sha
from recovery_features import balanced_event_schedule,group_folds,validate_prediction_provenance
from recovery_sft_data import require_training_pool
from recovery_train_contract import validate_launch,validate_h0_event_budget


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ticket',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();ticket,recipe=validate_launch(a.ticket,'H0')
    if 'RECOVERY_DEADLINE' not in os.environ: raise RuntimeError('Use cumulative-budget launcher')
    deadline=float(os.environ['RECOVERY_DEADLINE']);stop=Path(os.environ['RECOVERY_STOP_FILE'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if commit!=ticket['source_commit']: raise ValueError('Changed training source')
    _,rows=require_training_pool(ticket['admission'],'outcome',ticket['files']['admission']['sha256'])
    train=[r for r in rows if r['candidate']['split']=='train'];dev=[r for r in rows if r['candidate']['split']=='dev']
    feature_receipt=json.loads(Path(ticket['files']['feature_receipt']['path']).read_text())
    if (feature_receipt['diagnostic_only'] or feature_receipt['high_sha256']!=recipe['parents']['high']['sha256']
            or feature_receipt['features_sha256']!=ticket['files']['features']['sha256']
            or feature_receipt['admission_sha256']!=ticket['files']['admission']['sha256']):
        raise ValueError('Wrong/non-admitted feature cache')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    from g05.utils.training.stage1_runtime import (atomic_json,capture_rng,restore_rng,save_checkpoint,load_checkpoint,init_wandb)
    from recovery_observer_training import (OUTCOMES,request_key,temporal_batch,evaluate_head,calibrate,
                                            predicted_feedback,feedback_text)
    torch.set_num_threads(2);torch.manual_seed(recipe['seed']);random.seed(recipe['seed']);np.random.seed(recipe['seed'])
    cache=torch.load(ticket['files']['features']['path'],map_location='cpu',weights_only=False)
    if cache['schema']!='recovery_member_feature_cache_v1' or digest(cache['requests'])!=feature_receipt['requests_sha256']:
        raise ValueError('Changed feature request binding')
    features=cache['features'];requests=cache['requests']
    if set(features)!={r['request_id'] for r in requests}: raise ValueError('Incomplete cached features')
    for row in rows:
        if request_key(row) not in features: raise ValueError('Missing member-specific approved feature')
    all_train_groups={r['candidate']['source_group'] for r in train}|{r['source_group'] for r in requests if r['split']=='train'}
    fold=group_folds(all_train_groups,k=3)
    jobs=[];schedule=[]
    for i in range(3):
        target={g for g,v in fold.items() if v==i}
        other=sorted(all_train_groups-target,key=lambda x:digest(['h0-cal17',x]))
        calibration=set(other[:max(2,len(other)//4)])
        training=set(other)-calibration
        if len(training)<2: raise ValueError('Need independent groups for fit/calibration/OOF targets')
        selected=[r for r in train if r['candidate']['source_group'] in training]
        calibration_rows=[r for r in train if r['candidate']['source_group'] in calibration]
        jobs.append(dict(name=f'fold{i}',training_groups=sorted(training),calibration_groups=sorted(calibration),
                         target_groups=sorted(target),rows=selected,calibration_rows=calibration_rows,passes=1))
    jobs.append(dict(name='final',training_groups=sorted({r['candidate']['source_group'] for r in train}),
        calibration_groups=sorted({r['candidate']['source_group'] for r in dev}),target_groups=[],rows=train,calibration_rows=dev,passes=3))
    for j,job in enumerate(jobs):
        if not {'IN_PROGRESS','SUCCEEDED','FAILED'}<={r['approval']['label']['value'] for r in job['rows']}:
            raise ValueError('A training fold lacks actual known classes; collect data instead of inventing targets')
        for batch in balanced_event_schedule(job['rows'],batch_size=ticket['global_batch'],passes=job['passes'],seed=recipe['seed']+j):
            schedule.append(dict(job=j,**batch))
    counts=Counter((jobs[b['job']]['rows'][i]['candidate']['source_group'],jobs[b['job']]['rows'][i]['approval']['event_id'])
                   for b in schedule for _,i in b['rows'])
    validate_h0_event_budget(counts,ticket)
    if not schedule or len(schedule)>ticket['maximum_updates']:
        raise ValueError('Finite H0 folds exceed the common approved budget; do not silently truncate OOF')
    fingerprint=digest(dict(ticket=file_sha(a.ticket),schedule=schedule,fold=fold,source_commit=commit))
    device=torch.device('cuda',0);hidden=next(iter(features.values()))['context'].shape[-1]
    model=torch.nn.ModuleDict({j['name']:TemporalOutcomeObserver(hidden) for j in jobs}).to(device)
    for head in model.values(): head.head.load_state_dict(cache['initial_head'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['H0']['learning_rate'],weight_decay=recipe['H0']['weight_decay'])
    state=dict(step=0,save_sequence=0,fingerprint=fingerprint,consumed_seconds=0.,run_id=uuid.uuid4().hex[:12])
    if a.resume:
        saved,_=load_checkpoint(a.output/'checkpoints',fingerprint)
        model.load_state_dict(saved['model_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer_state_dict'])
        state=dict(saved['state']);restore_rng(saved['rng_by_rank'][0]);del saved;gc.collect()
    else: a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'schedule.json',dict(fingerprint=fingerprint,schedule=schedule,fold=fold,
        total_event_passes_max=max(counts.values()),jobs=[{k:v for k,v in j.items() if k not in ('rows','calibration_rows')} for j in jobs]))
    wb=init_wandb(dict(recipe['wandb'],name=a.output.name),a.output,run_id=state['run_id'],resume=a.resume,
        metadata=dict(component='H0',feature_sha256=feature_receipt['features_sha256'],source_commit=commit,
            high_sha256=recipe['parents']['high']['sha256'],schedule_fingerprint=fingerprint,trained_backbone=False))
    stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    while state['step']<len(schedule):
        if stopping[0] or stop.exists() or time.monotonic()+180>=deadline: break
        batch=schedule[state['step']];job=jobs[batch['job']];head=model[job['name']]
        selected=[job['rows'][i] for _,i in batch['rows']]
        values=temporal_batch([features[request_key(r)] for r in selected],device)
        labels=torch.tensor([OUTCOMES.index(r['approval']['label']['value']) for r in selected],device=device)
        optimizer.zero_grad(set_to_none=True);head.train();logits=head(**values);loss=F.cross_entropy(logits,labels)
        if not torch.isfinite(loss): raise ValueError('Nonfinite true outcome objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        state['step']+=1;state['save_sequence']+=1
        state['consumed_seconds']=float(os.environ['RECOVERY_PREVIOUS_SECONDS'])+time.monotonic()-float(os.environ['RECOVERY_STARTED'])
        save_checkpoint(a.output/'checkpoints',model=model,optimizer=optimizer,state=dict(state),rng_by_rank=[capture_rng()])
        wb.log({'train/update':state['step'],'train/outcome_ce':float(loss.detach()),'train/head':job['name'],'train/independent_events':len(selected)})
    if state['step']!=len(schedule):
        atomic_json(a.output/'result.json',dict(status='saved_paused',step=state['step'],planned_updates=len(schedule),feedback_ready=False))
        wb.finish();return
    predictions={};calibrations={};model.eval()
    # Preserve any failed earlier calibration/export attempt on resume.
    post=a.output/('postprocess-'+os.environ['RECOVERY_ATTEMPT'])
    post.mkdir(exist_ok=False)
    histories={r['sample_id']:r for r in (json.loads(x) for x in Path(ticket['files']['history']['path']).read_text().splitlines())}
    for j,job in enumerate(jobs):
        if time.monotonic()+60>=deadline: raise RuntimeError('Calibration exceeded common budget; no feedback release')
        head=model[job['name']]
        logits,labels,cal_rows=evaluate_head(head,job['calibration_rows'],features,device)
        calibration=calibrate(logits,labels,[r['candidate']['source_group'] for r in cal_rows])
        weight=post/(job['name']+'-observer.pt');torch.save(head.state_dict(),weight)
        calibration.update(high_sha256=recipe['parents']['high']['sha256'],observer_sha256=file_sha(weight),
                           feature_cache_sha256=feature_receipt['features_sha256'],fit_groups=job['training_groups'])
        path=post/(job['name']+'-calibration.json');atomic_json(path,calibration);calibrations[job['name']]=calibration
        if job['name']=='final': continue
        provenance=dict(high_sha256=recipe['parents']['high']['sha256'],observer_sha256=file_sha(weight),
            trained_groups=job['training_groups'],calibration_groups=job['calibration_groups'],target_groups=job['target_groups'],
            feature_cache_sha256=feature_receipt['features_sha256'],fold=j,calibrated=calibration['ready'],
            temperature=calibration['temperature'],calibration_receipt_sha256=file_sha(path))
        for request in requests:
            if request['role']!='predecision' or request['source_group'] not in job['target_groups']: continue
            validate_prediction_provenance(provenance,group=request['source_group'],high_sha256=recipe['parents']['high']['sha256'])
            sid=request['sample_id'];prior=histories[sid]['predecision']
            member=predicted_feedback(head,request,features,prior,calibration,device)
            entry=predictions.setdefault(sid,dict(sample_id=sid,source_group=request['source_group'],
                control_step=histories[sid]['control_step'],provenance=provenance,members=[]))
            entry['members'].append(member)
    # H1 dev must also be outside the observer's fit AND calibration groups.
    # Final dev calibration is not valid H1 input. Use a fold head whose fit
    # and calibration are both subsets of TRAIN, evaluated on all DEV requests.
    job=jobs[0];head=model[job['name']];calibration=calibrations[job['name']]
    dev_groups=sorted({r['source_group'] for r in requests if r['split']=='dev'})
    provenance=dict(high_sha256=recipe['parents']['high']['sha256'],observer_sha256=file_sha(post/'fold0-observer.pt'),
        trained_groups=job['training_groups'],calibration_groups=job['calibration_groups'],target_groups=dev_groups,
        feature_cache_sha256=feature_receipt['features_sha256'],fold='independent_dev',calibrated=calibration['ready'],
        temperature=calibration['temperature'],calibration_receipt_sha256=file_sha(post/'fold0-calibration.json'))
    for request in requests:
        if request['role']!='predecision' or request['split']!='dev': continue
        validate_prediction_provenance(provenance,group=request['source_group'],high_sha256=recipe['parents']['high']['sha256'])
        sid=request['sample_id'];prior=histories[sid]['predecision']
        entry=predictions.setdefault(sid,dict(sample_id=sid,source_group=request['source_group'],
            control_step=histories[sid]['control_step'],provenance=provenance,members=[]))
        entry['members'].append(predicted_feedback(head,request,features,prior,calibration,device))
    output=post/'feedback.jsonl'
    with output.open('x') as stream:
        for sid,entry in sorted(predictions.items()):
            members=sorted(entry.pop('members'),key=lambda r:r['member'])
            entry['execution_feedback']=feedback_text(histories[sid]['predecision'],members)
            stream.write(json.dumps(entry,sort_keys=True)+'\n')
    result=dict(schema='recovery_oof_feedback_receipt_v1',status='trained_calibration_and_oof_completed',
        high_sha256=recipe['parents']['high']['sha256'],feedback_sha256=file_sha(output),rows=len(predictions),
        step=state['step'],maximum_event_exposure=max(counts.values()),diagnostic_only=False,
        final_observer_ready=calibrations['final']['ready'],calibration_status={k:v['ready'] for k,v in calibrations.items()},
        new_high_hash_invalidates_all_features_and_calibration=True)
    result['artifacts_directory']=str(post)
    atomic_json(post/'feedback_receipt.json',result);atomic_json(a.output/'result.json',result);wb.finish()


if __name__=='__main__':main()
