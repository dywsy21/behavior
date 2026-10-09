"""Finite H0 fitting, held-group calibration and explicitly uncertain feedback.

This module does not launch jobs or grant training permission. Every tensor
comes from a SHA-bound frozen-backbone cache; labels stay out of features.
"""
from collections import defaultdict
import math

from recovery_corpus import digest

OUTCOMES = ('IN_PROGRESS','SUCCEEDED','FAILED','UNKNOWN')


def temporal_batch(items, device):
    import torch
    if not items: raise ValueError('Empty temporal batch')
    lengths=[len(r['steps']) for r in items]; length=max(lengths)
    hidden=items[0]['context'].shape[-1]
    result=dict(context=torch.zeros(len(items),length,hidden,device=device),
        proprio=torch.zeros(len(items),length,27,device=device),
        steps=torch.zeros(len(items),length,dtype=torch.long,device=device),
        valid=torch.zeros(len(items),length,dtype=torch.bool,device=device))
    for i,(row,n) in enumerate(zip(items,lengths)):
        if not 1 <= n <= 4 or row['context'].shape!=(n,hidden) or row['proprio'].shape!=(n,27):
            raise ValueError('Malformed member feature sequence')
        for key in ('context','proprio','steps'): result[key][i,:n]=row[key].to(device)
        result['valid'][i,:n]=True
    return result


def one_per_event(rows):
    events=defaultdict(list)
    for row in rows: events[(row['candidate']['source_group'],row['approval']['event_id'])].append(row)
    return [min(values,key=lambda r:digest([r['candidate']['sample_id'],r['approval']['label']['member_index']]))
            for _,values in sorted(events.items())]


def request_key(row):
    return digest([row['candidate']['sample_id'],'observable',row['approval']['label']['member_index']])


def evaluate_head(model,rows,features,device):
    import torch
    selected=one_per_event(rows)
    if not selected: return torch.empty(0,4),torch.empty(0,dtype=torch.long),[]
    was=model.training; model.eval(); logits=[]
    with torch.no_grad():
        for start in range(0,len(selected),64):
            items=selected[start:start+64]
            logits.append(model(**temporal_batch([features[request_key(r)] for r in items],device)).cpu())
    model.train(was)
    return torch.cat(logits),torch.tensor([OUTCOMES.index(r['approval']['label']['value']) for r in selected]),selected


def outcome_metrics(model, rows, features, device):
    """All reviewed labels, equal total weight per physical source event.

    These are diagnostics, not independent-event counts for calibration.
    Evaluating a single random anchor/event could omit a whole result class.
    """
    import torch
    import torch.nn.functional as F
    events=defaultdict(int)
    for row in rows: events[(row['candidate']['source_group'],row['approval']['event_id'])]+=1
    if not rows: raise ValueError('Empty outcome diagnostic split')
    was=model.training;model.eval();logits=[]
    with torch.no_grad():
        for start in range(0,len(rows),64):
            logits.append(model(**temporal_batch([features[request_key(r)] for r in rows[start:start+64]],device)).cpu())
    model.train(was);logits=torch.cat(logits)
    y=torch.tensor([OUTCOMES.index(r['approval']['label']['value']) for r in rows])
    w=torch.tensor([1/events[(r['candidate']['source_group'],r['approval']['event_id'])] for r in rows])
    prediction=logits.argmax(-1);loss=F.cross_entropy(logits,y,reduction='none')
    per_class={name:dict(rows=int((y==i).sum()),recall=float((prediction[y==i]==i).float().mean()))
               for i,name in enumerate(OUTCOMES) if (y==i).any()}
    return dict(event_weighted_ce=float((w*loss).sum()/w.sum()),
        event_weighted_accuracy=float((w*(prediction==y)).sum()/w.sum()),
        balanced_accuracy=sum(v['recall'] for v in per_class.values())/len(per_class),
        per_class=per_class,reviewed_rows=len(rows),independent_events=len(events),
        calibration_ready_not_inferred=True)


def wilson(successes,n,z=1.96):
    if not n: return 0.,1.
    p=successes/n;den=1+z*z/n; center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-radius,center+radius


def calibrate(logits,labels,groups,*,minimum_confidence=.85):
    """Temperature from calibration groups ONLY; 20 independent events/class.

    A fit may succeed numerically yet remain unready. No threshold is chosen
    on the future target/dev instances or to force a desired coverage.
    """
    import torch
    import torch.nn.functional as F
    if logits.shape!=(len(labels),4) or not torch.isfinite(logits).all(): raise ValueError('Bad calibration logits')
    if len(labels):
        temperatures=torch.logspace(math.log10(.5),math.log10(5.),41)
        losses=torch.stack([F.cross_entropy(logits/t,labels) for t in temperatures])
        temp=float(temperatures[losses.argmin()])
        confidence,prediction=(logits/temp).softmax(-1).max(-1)
    else:
        temp=1.;confidence=torch.empty(0);prediction=torch.empty(0,dtype=torch.long)
    blockers=[];classes={}
    if len(set(groups))<2: blockers.append('fewer_than_two_independent_calibration_groups')
    for c,name in enumerate(OUTCOMES[:3]):
        accepted=(prediction==c)&(confidence>=minimum_confidence)
        n=int(accepted.sum());correct=int(((labels==c)&accepted).sum())
        lo,hi=wilson(correct,n)
        classes[name]=dict(actual_events=int((labels==c).sum()),confident_predictions=n,correct=correct,
                          precision=correct/n if n else None,precision_wilson95=[lo,hi])
        if int((labels==c).sum())<20 or n<20 or correct/max(n,1)<.9 or lo<.8:
            blockers.append(name+': insufficient independent support or selective precision')
    negative=labels!=1
    false_success=int(((prediction==1)&(confidence>=minimum_confidence)&negative).sum())
    upper=wilson(false_success,int(negative.sum()))[1]
    if upper>.1: blockers.append('false_success_rate_upper95_exceeds_0.10')
    return dict(schema='recovery_observer_calibration_v1',temperature=temp,minimum_confidence=minimum_confidence,
        ready=not blockers,blockers=blockers,events=len(labels),source_groups=sorted(set(groups)),classes=classes,
        false_success_count=false_success,false_success_upper95=upper,
        status='calibrated_and_selectively_validated' if not blockers else 'fit_only_not_runtime_ready')


def predicted_feedback(model,request,features,history,calibration,device):
    """One member, two *different past/current* checks, no oracle substitution."""
    import torch
    feature=features[request['request_id']];n=len(feature['steps']);result=[]
    model.eval()
    with torch.no_grad():
        for end in range(max(1,n-1),n+1):
            prefix={k:v[:end] for k,v in feature.items()}
            logits=model(**temporal_batch([prefix],device))[0].float().cpu()/calibration['temperature']
            confidence,index=logits.softmax(-1).max(-1)
            value=OUTCOMES[int(index)] if calibration['ready'] and confidence>=calibration['minimum_confidence'] else 'UNKNOWN'
            result.append(dict(control_step=int(feature['steps'][end-1]),outcome=value,confidence=float(confidence)))
    same=(len(result)==2 and result[0]['control_step']<result[1]['control_step']
          and result[0]['outcome']==result[1]['outcome'])
    return dict(member=request['member_index'],estimated_outcome=result[-1]['outcome'] if same else 'UNKNOWN',
                confidence=min(r['confidence'] for r in result) if same else 0.)


def feedback_text(prior,members):
    """Same runtime text contract; only counters and learned estimates enter."""
    from g05.utils.memlite_skill_protocol import canonical_json
    if not members or [x['member'] for x in members]!=list(range(len(members))): raise ValueError('Incomplete bundle predictions')
    values=[x['estimated_outcome'] for x in members]
    aggregate=('SUCCEEDED' if all(v=='SUCCEEDED' for v in values) else 'FAILED' if 'FAILED' in values else
               'UNKNOWN' if 'UNKNOWN' in values else 'IN_PROGRESS')
    return canonical_json(dict(schema='causal_execution_feedback_v1',same_intent_controls=prior['served_controls'],
        same_intent_planner_refreshes=prior['repeated_planning_count'],attempt_index=prior['attempt_number'],
        estimated_member_outcomes=members,attempt_count_scope='last_64_distinct_intents_this_episode',
        estimated_bundle_outcome=aggregate,source='observable_counter_and_learned_observer',stalled_is_not_failed=True))
