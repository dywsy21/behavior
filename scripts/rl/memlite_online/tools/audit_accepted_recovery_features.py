"""Independently reload accepted causal features and exercise the real H0 head."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest,file_sha
from recovery_features import group_folds
from recovery_observer_training import temporal_batch,request_key,one_per_event
from recovery_sft_data import require_training_pool


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','admission','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--admission-sha256',required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    torch.set_num_threads(2);torch.manual_seed(17)
    receipt=json.loads((a.cache/'receipt.json').read_text())
    if (receipt['diagnostic_only'] or receipt['optimizer_steps']!=0 or receipt['admission_sha256']!=a.admission_sha256
            or file_sha(a.cache/'features.pt')!=receipt['features_sha256']):raise ValueError('Unverified actual feature cache')
    _,rows=require_training_pool(a.admission,'outcome',a.admission_sha256)
    cache=torch.load(a.cache/'features.pt',map_location='cpu',weights_only=False)
    if digest(cache['requests'])!=receipt['requests_sha256']:raise ValueError('Changed requests')
    features=cache['features'];requests=cache['requests']
    if set(features)!={r['request_id'] for r in requests}:raise ValueError('Missing request feature')
    for row in rows:
        if request_key(row) not in features:raise ValueError('Missing approved outcome/member')
    for req in requests:
        value=features[req['request_id']]
        if value['steps'].tolist()!=[c['control_step'] for c in req['checks']]:raise ValueError('Feature clock drift')
        if any(v.requires_grad for v in value.values()):raise ValueError('Retained high graph')
    hidden=next(iter(features.values()))['context'].shape[-1]
    model=TemporalOutcomeObserver(hidden);model.head.load_state_dict(cache['initial_head'],strict=True)
    values=temporal_batch(list(features.values()),'cpu')
    values['context'].requires_grad_(True);values['proprio'].requires_grad_(True)
    logits=model(**values)
    if logits.shape!=(len(requests),4) or not torch.isfinite(logits).all():raise ValueError('Invalid actual head consumption')
    logits.square().mean().backward()
    if (values['context'].grad is not None or values['proprio'].grad is not None
            or model.head.classifier.weight.grad is None):raise ValueError('H0 gradient insulation failed')
    train=[r for r in rows if r['candidate']['split']=='train'];dev=[r for r in rows if r['candidate']['split']=='dev']
    folds=group_folds([r['candidate']['source_group'] for r in train])
    result=dict(schema='accepted_feature_reload_audit_v1',status='passed',admission_sha256=a.admission_sha256,
        features_sha256=receipt['features_sha256'],high_sha256=receipt['high_sha256'],requests=len(requests),
        temporal_lengths=dict(Counter(len(v['steps']) for v in features.values())),logits_shape=list(logits.shape),
        head_consumption_and_insulated_backward=True,optimizer_steps=0,fold_groups=folds,
        independent_dev_events=len(one_per_event(dev)),
        calibration_sampled_dev_classes=dict(Counter(r['approval']['label']['value'] for r in one_per_event(dev))),
        runtime_ready=False,calibration_not_performed=True,
        limitation='Pilot data, not adequate evidence for 20 independent events per class and runtime selective-precision readiness.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
