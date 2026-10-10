"""Fixed source-disjoint observer feedback, on the actual cadence16 clock.

This explicit protocol is NOT OOF refitting. Observer backbone and the planner
being trained are separate immutable models. Only observable history is used.
"""
from collections import defaultdict
import json
import math
from pathlib import Path

from recovery_corpus import canonical, digest, file_sha
from recovery_postfit import calibration_gate

SCHEMA = 'frozen_source_disjoint_observer_feedback_v1'
STATUS = 'frozen_source_disjoint_feedback_completed'
OUTCOMES = ('IN_PROGRESS', 'SUCCEEDED', 'FAILED', 'UNKNOWN')


def validate_exposure(exposure):
    if (exposure.get('schema') != 'postfit_observer_exposure_v1'
            or exposure.get('calibrated_mechanisms') != ['GRASP']):
        raise ValueError('Unregistered frozen observer exposure')
    expected = set(exposure['observer_train_groups']) | set(exposure['observer_selection_groups']) | set(exposure['all_reserved_groups'])
    if (set(exposure['excluded_groups']) != expected
            or not set(exposure['calibration_groups']) <= expected):
        raise ValueError('Missing observer fit, selection or reserved sources')
    return expected


def clock_requests(anchors, histories, selected, exposure):
    """All real 0,16,... checks preceding first local RETRY, not backward strides.

    At RETRY24, current24 is NOT a new observer check: only 0/16 were seen.
    At64, the preceding48 prediction needs [0,16,32,48], NOT [16,32,48].
    Current approved corpus has first-attempt GRASP recovery; other histories
    must get a separately audited replay instead of invented prior counters.
    """
    excluded = validate_exposure(exposure)
    by_id = {r['sample_id']:r for r in anchors}
    history = {r['sample_id']:r for r in histories}
    if len(by_id) != len(anchors) or len(history) != len(histories):
        raise ValueError('Duplicate anchor or history')
    episodes = defaultdict(dict)
    for row in anchors:
        ep = canonical(row['source_episode']); t = row['control_step']
        if t in episodes[ep]: raise ValueError('Duplicate episode observation clock')
        episodes[ep][t] = row
    output = []
    if len({r['sample_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate target decisions')
    for row in selected:
        sid = row['sample_id']; end = row['control_step']; h = history[sid]; prior = h['predecision']
        if (by_id.get(sid) != row or row['source_group'] in excluded or row['split'] not in ('train', 'dev')
                or h['source_episode'] != row['source_episode'] or h['source_group'] != row['source_group']
                or h['control_step'] != end or prior is None or prior['history_is_partial']
                or prior['intent_started_control_step'] != 0 or prior['control_step'] != 0
                or prior['attempt_number'] != 1 or prior['repeated_planning_count'] != 0
                or prior['served_controls'] != end or prior['observation_control_step'] != end):
            raise ValueError('Unseen, complete first-attempt predecision history required')
        members = json.loads(prior['issued_skills_semantic_json'])
        if not members or any(m['verb'] != 'GRASP' for m in members):
            raise ValueError('Frozen calibration certifies GRASP only')
        for member in range(len(members)):
            checks = []
            for t in range(0, end+1, 16):
                observation = episodes[canonical(row['source_episode'])].get(t)
                if observation is None: raise ValueError('Missing actual cadence16 original observation')
                old = history[observation['sample_id']]
                ctx = old['predecision'] if t == end else old['observable']
                if (old['source_episode'] != row['source_episode'] or old['source_group'] != row['source_group']
                        or old['control_step'] != t or observation['source_group'] != row['source_group']
                        or observation['split'] != row['split'] or observation['task'] != row['task']
                        or ctx is None or ctx['history_is_partial'] or ctx['observation_control_step'] != t
                        or ctx['served_controls'] != t
                        or any(ctx[k] != prior[k] for k in ('context_id','intent_started_control_step',
                            'issued_skills_semantic_json','parent_goal','memory','attempt_number','repeated_planning_count'))):
                    raise ValueError('Cross-task, cross-attempt or future observer context')
                checks.append(dict(sample_id=observation['sample_id'], control_step=t,
                    task_name=row['task'].replace('_',' '), parent_goal=ctx['parent_goal'],
                    issued_bundle=ctx['issued_skills_semantic_json'],member_index=member,
                    memory=ctx['memory'],served_controls=t))
                output.append(dict(request_id=digest([SCHEMA,sid,member,t]),decision_sample_id=sid,
                    source_group=row['source_group'],source_episode=row['source_episode'],split=row['split'],
                    member_index=member,control_step=t,checks=checks[-4:]))
    return output


def predicted_execution_feedback(candidate, history, trace, calibration):
    """Use the same freshness, threshold and two-confirmation ledger as serving."""
    from g05.utils.memlite_causal_feedback import CausalFeedbackLedger, FeedbackIdentity
    calibration_gate(calibration)
    prior = history['predecision']; sid = candidate['sample_id']; end = candidate['control_step']
    members = json.loads(prior['issued_skills_semantic_json'])
    relevant = [r for r in trace if r['decision_sample_id'] == sid]
    expected = {(m,t) for m in range(len(members)) for t in range(0,end+1,16)}
    keys = [(r['member_index'],r['control_step']) for r in relevant]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing/duplicate real observer checks')
    identity = FeedbackIdentity(SCHEMA,candidate['task'],candidate['instance_id'],
        canonical(candidate['source_episode']),calibration['high_sha256'],calibration['observer_sha256'])
    ledger = CausalFeedbackLedger(identity,minimum_confidence=.85,confirmations=2,
        uncertainty_protocol='unready_zero_confidence_v1')
    ledger.issued(identity,0,prior['issued_skills_semantic_json'],prior['parent_goal'],decision='EXECUTE')
    indexed = {(r['member_index'],r['control_step']):r for r in relevant}
    for t in range(0,end+1,16):
        predictions = []
        for m in range(len(members)):
            r = indexed[m,t]
            if (r['source_group'] != candidate['source_group'] or r['source_episode'] != candidate['source_episode']
                    or r['split'] != candidate['split']
                    or [c['control_step'] for c in r['checks']] != list(range(max(0,t-48),t+1,16))):
                raise ValueError('Mixed source or incorrect temporal prediction window')
            v = r['logits']
            if len(v) != 4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in v):
                raise ValueError('Four finite raw model logits required')
            z = [x/calibration['temperature'] for x in v]; top = max(z)
            p = [math.exp(x-top) for x in z]; total = sum(p); i = max(range(4),key=p.__getitem__)
            predictions.append(dict(outcome=OUTCOMES[i],confidence=p[i]/total))
        ledger.observe_clock(identity,t); ledger.estimated(identity,t,predictions,calibrated=True)
    return ledger.projection(identity,end)


def make_provenance(exposure, *, high_sha256, exposure_sha256, calibration_sha256, trace_sha256, target_groups):
    validate_exposure(exposure)
    return dict(schema=SCHEMA,high_sha256=high_sha256,observer_backbone_sha256=exposure['high_sha256'],
        observer_sha256=exposure['observer_sha256'],trained_groups=exposure['observer_train_groups'],
        selection_groups=exposure['observer_selection_groups'],calibration_groups=exposure['calibration_groups'],
        reserved_groups=exposure['all_reserved_groups'],target_groups=sorted(set(target_groups)),
        exposure_sha256=exposure_sha256,calibration_receipt_sha256=calibration_sha256,
        prediction_trace_sha256=trace_sha256,calibrated=True,history_protocol='actual_cadence16_fresh_two_checks_v1')


def validate_provenance(row, *, group, high_sha256):
    required={'schema','high_sha256','observer_backbone_sha256','observer_sha256','trained_groups',
        'selection_groups','calibration_groups','reserved_groups','target_groups','exposure_sha256',
        'calibration_receipt_sha256','prediction_trace_sha256','calibrated','history_protocol'}
    if (set(row) != required or row['schema'] != SCHEMA or row['high_sha256'] != high_sha256
            or row['calibrated'] is not True or row['history_protocol'] != 'actual_cadence16_fresh_two_checks_v1'):
        raise ValueError('Invalid separate-model fixed observer provenance')
    for key in ('high_sha256','observer_backbone_sha256','observer_sha256','exposure_sha256',
                'calibration_receipt_sha256','prediction_trace_sha256'):
        s=row[key]
        if not isinstance(s,str) or len(s)!=64 or any(c not in '0123456789abcdef' for c in s):
            raise ValueError('Unpinned fixed observer artifact')
    excluded=set(row['trained_groups'])|set(row['selection_groups'])|set(row['reserved_groups'])
    if (group not in row['target_groups'] or excluded & set(row['target_groups'])
            or not set(row['calibration_groups']) <= set(row['reserved_groups'])):
        raise ValueError('Observer fit, selection, calibration or reserved leakage')
    return True


def validate_training_release(ticket, recipe):
    """Recompute every feedback input from pinned logits before admitting H1.

    This does not fabricate a legacy OOF-completed receipt. An independently
    frozen observer retains its own backbone as the planner starts from a74.
    """
    from recovery_sft_data import require_training_pool
    files=ticket['files']; high=recipe['parents']['high']['sha256']; h1=recipe['H1']
    required={'feedback','feedback_receipt','observer_exposure','observer_calibration',
              'observer_prediction_trace','history','inventory','admission'}
    if not required<=files.keys() or h1.get('feedback')!='frozen_source_disjoint_observer_v1':
        raise ValueError('Require an explicit pinned fixed-observer training contract')
    for key in required:
        if file_sha(files[key]['path'])!=files[key]['sha256']:raise ValueError('Changed feedback input: '+key)
    def read_key(k):return json.loads(Path(files[k]['path']).read_text())
    receipt=read_key('feedback_receipt');exposure=read_key('observer_exposure');cal=read_key('observer_calibration')
    calibration_gate(cal);validate_exposure(exposure)
    if (receipt.get('status')!=STATUS or receipt.get('diagnostic_only') is not False
            or receipt.get('optimizer_updates')!=0 or receipt.get('calibration_refit') is not False
            or receipt.get('observer_weights_updated') is not False or receipt.get('all_admitted_rows_retained') is not True
            or receipt.get('frozen_before_sha256')!=receipt.get('frozen_after_sha256')
            or receipt.get('high_sha256')!=high or receipt.get('planner_high_sha256')!=high
            or receipt.get('admission_sha256')!=files['admission']['sha256']
            or receipt.get('feedback_sha256')!=files['feedback']['sha256']
            or receipt.get('exposure_sha256')!=files['observer_exposure']['sha256']
            or receipt.get('calibration_sha256')!=files['observer_calibration']['sha256']
            or receipt.get('prediction_trace_sha256')!=files['observer_prediction_trace']['sha256']
            or exposure['high_sha256']!=h1['observer_backbone_sha256']
            or exposure['observer_sha256']!=h1['observer_sha256']
            or exposure['bindings']['calibration']['sha256']!=files['observer_calibration']['sha256']
            or cal['high_sha256']!=exposure['high_sha256'] or cal['observer_sha256']!=exposure['observer_sha256']
            or sorted(cal['source_groups'])!=sorted(exposure['calibration_groups'])):
        raise ValueError('Frozen observer/planner/calibration/data receipt mismatch')
    for key,name in [('observer_exposure','exposure'),('observer_calibration','calibration')]:
        if files[key]['sha256']!=h1[name]['sha256']:
            raise ValueError('Recipe does not pin this exact feedback observer')
    _,rows=require_training_pool(ticket['admission'],'planner',files['admission']['sha256'])
    corpus=Path(ticket['admission']).parent
    hr=json.loads((corpus/'history/receipt.json').read_text())
    if (hr['contexts_sha256']!=files['history']['sha256']
            or hr['anchors_sha256']!=file_sha(corpus/'audit/anchors.jsonl')
            or hr['inventory_sha256']!=digest(read_key('inventory'))):
        raise ValueError('Changed actual prediction-clock evidence')
    anchors=[json.loads(x) for x in (corpus/'audit/anchors.jsonl').read_text().splitlines()]
    histories=[json.loads(x) for x in Path(files['history']['path']).read_text().splitlines()]
    selected=[r['candidate'] for r in rows];requests=clock_requests(anchors,histories,selected,exposure)
    trace=read_key('observer_prediction_trace')
    if ([{k:v for k,v in r.items() if k!='logits'} for r in trace]!=requests
            or receipt.get('requests_sha256')!=digest(requests)):
        raise ValueError('Incomplete or altered real prediction request windows')
    feedback=[json.loads(x) for x in Path(files['feedback']['path']).read_text().splitlines()]
    expected=make_provenance(exposure,high_sha256=high,exposure_sha256=files['observer_exposure']['sha256'],
        calibration_sha256=files['observer_calibration']['sha256'],trace_sha256=files['observer_prediction_trace']['sha256'],
        target_groups=[r['source_group'] for r in selected])
    if len(feedback)!=len(selected) or receipt.get('rows')!=len(selected):raise ValueError('Prediction selection/filtering')
    by_id={r['sample_id']:r for r in histories}
    for f,r in zip(feedback,selected):
        validate_provenance(f['provenance'],group=r['source_group'],high_sha256=high)
        if (f['provenance']!=expected or f['sample_id']!=r['sample_id'] or f['source_group']!=r['source_group']
                or f['control_step']!=r['control_step'] or f['execution_feedback']!=
                    predicted_execution_feedback(r,by_id[r['sample_id']],trace,cal)):
            raise ValueError('Feedback differs from the actual fixed logits and serving freshness contract')
    return dict(status='verified_frozen_source_disjoint_feedback',rows=len(feedback),real_checks=len(trace))
