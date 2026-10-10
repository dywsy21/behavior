"""Replay issued-command and actual-action correspondence without model code.

Physical success is checked separately by the per-control skill auditor. This
audit binds every high input to the real RGB/proprio archive and every emitted
command/action to the simulator's independently saved applied-control stream.
"""
from dataclasses import asdict
import json

import numpy as np

from causal_skill_probe import validate_probe_histories
from recovery_causal_handover import restore_teacher_prefix,AppliedActionClock
from g05.utils.memlite_causal_session import CausalSessionIdentity


def replay_observer_check(session,entry,*,calibration,state,observation_hashes):
    """Recompute ledger from logged raw logits, not model-reported success.

    Context clock/token and current-image identities are independently checked;
    this does not claim to independently recompute the learned GPU logits.
    """
    from recovery_calibrated_observer import calibrated_prediction
    identity=session.identity;tick=session.feedback.control_step;report=entry['result']
    if (entry['control_step']!=tick or entry['revision']!=session.revision
            or entry['attempt_generation']!=session.attempt_generation
            or entry['observation_sha256']!=observation_hashes.get(tick)
            or report['control_step']!=tick or report['observer_feedback_mode']!=session.observer_feedback_mode
            or report['entered_planner'] is not False):
        raise ValueError('Foreign/stale observer input, attempt, clock or mode')
    if state.get('generation')!=session.attempt_generation:
        state.clear();state.update(generation=session.attempt_generation,steps=[])
    last=state['steps'][-1] if state['steps'] else None
    expected=('duplicate_check_not_recomputed' if last==tick else
        'partial_chunk_skipped_by_original_cadence' if last is not None and tick-last<16 else
        'calibrated_grasp_pilot_observer_check')
    if report['status']!=expected:raise ValueError('Observer cadence or duplicate suppression drift')
    count=0
    if expected=='calibrated_grasp_pilot_observer_check':
        members=json.loads(session.installed['semantic_bundle']);rows=report['members'];predictions=[]
        steps=(state['steps']+[tick])[-4:]
        if (report['calibration_sha256']!=calibration.calibration_sha256
                or report['calibration_applied'] is not True or report['physical_success_asserted'] is not False
                or report['known_previous_outcome']!='UNKNOWN' or report['optimizer_updates']!=0
                or report['attempt_generation']!=session.attempt_generation
                or report['observation_sha256']!=entry['observation_sha256'] or len(rows)!=len(members)):
            raise ValueError('Unbound calibration/physical claim/member feedback')
        for i,(member,row) in enumerate(zip(members,rows)):
            if (row['member']!=i or row['context_steps']!=steps
                    or row['request_token']!=session.observer_request(identity,i)[0]):
                raise ValueError('Observer mixed member, context history or attempt token')
            predictions.append(calibrated_prediction(row['raw_logits'],temperature=calibration.temperature,verb=member['verb']))
        session.calibrated_outcomes(identity,tick,predictions,calibration=calibration)
        state['steps']=steps;count=len(rows)
    elif 'members' in report:raise ValueError('Skipped observer check invented predictions')
    if entry['estimated_feedback']!=session.feedback.projection(identity.feedback_identity(),tick):
        raise ValueError('Feedback did not recompute from calibrated raw logits and freshness')
    return count


def audit_joint_episode(events, *, models, history_row, case, job, controls, observation_hashes,calibration=None):
    if not events or events[0]['kind']!='handover' or events[-1]['kind']!='close':
        raise ValueError('Complete handover-to-close causal journal required')
    if (job['phase']!='evaluation' or job['round']!=0
            or any(e['job']!=job for e in events)):
        raise ValueError('Cross-job journal or non-evaluation trajectory')
    key=job['case']
    projection=validate_probe_histories(dict(schema='actual_local_causal_handover_inputs_v1',
        role='TRAIN_engineering_only_never_SFT_calibration_or_test',rows=[history_row]),{key:case})[key]
    identity_data=events[0]['receipt']['destination_identity']
    identity=CausalSessionIdentity(identity_data['session'],case['task'].replace('_',' '),case['instance_id'],job['id'],models)
    if asdict(identity)!=identity_data:
        raise ValueError('Foreign task/episode/model handover identity')
    options={}
    if calibration is not None:
        from recovery_calibrated_observer import CalibratedGraspPilotSession
        options['session_factory']=lambda i:CalibratedGraspPilotSession(i,calibration=calibration)
    session,receipt=restore_teacher_prefix(identity,projection,source_task=case['task'],source_instance=case['instance_id'],**options)
    if receipt!=events[0]['receipt'] or events[0]['observer_predictions']!=0:
        raise ValueError('Handover differs from actual past-only issued prefix')
    clock=AppliedActionClock(session);next_kind='plan';consumed=generations=reuses=chunks=predictions=0
    observer_state={};last_plan=None
    for index,entry in enumerate(events[1:],1):
        kind=entry['kind'];tick=session.feedback.control_step
        previous=events[index-1]
        if kind=='observer':
            phase=entry['phase']
            if calibration is None or not (
                    phase=='before_plan' and next_kind=='plan' and previous['kind'] in ('handover','ack')
                    or phase=='after_plan' and next_kind=='action' and previous['kind']=='plan' and not last_plan['reused']
                    or phase=='final_observation' and next_kind=='plan' and previous['kind']=='ack' and consumed==len(controls)):
                raise ValueError('Unexpected observer transaction order or uncalibrated pilot')
            predictions+=replay_observer_check(session,entry,calibration=calibration,state=observer_state,
                observation_hashes=observation_hashes)
            continue
        if kind=='close':
            if (index!=len(events)-1 or next_kind!='plan' or consumed!=len(controls)
                    or entry['control_step']!=tick or entry['revision']!=session.revision
                    or entry['observer_predictions']!=predictions or consumed<=0
                    or calibration is not None and (previous['kind']!='observer' or previous['phase']!='final_observation')):
                raise ValueError('Early, duplicated, incomplete or wrong-clock close')
            session.close(identity,tick)
            continue
        if kind!=next_kind:
            raise ValueError('Each real chunk needs exactly one plan, offer, and ACK in order')
        if kind=='plan':
            result=entry['result'];due=session.planning_due(identity,interval_controls=128)
            if (type(result['reused']) is not bool or result['reused']==due
                    or entry['observation_sha256']!=observation_hashes.get(tick)
                    or entry['observer_predictions']!=predictions
                    or calibration is not None and (previous['kind']!='observer' or previous['phase']!='before_plan')
                    or entry['scored_original_skill_not_forced_into_actor'] is not True):
                raise ValueError('Stale/cross-observation plan, wrong cadence, or invented result')
            if due:
                token,causal=session.begin_planning(identity,tick)
                if causal!=result['causal_input']:
                    raise ValueError('Generated command consumed the wrong memory/feedback')
                session.stage(identity,token,result['event']);goal=session.commit(identity,token)
                generations+=1
            else:
                if result['event'] is not None or result['causal_input'] is not None:
                    raise ValueError('Reuse invented a new model generation')
                goal=session.low_goal(identity);reuses+=1
            if (goal!=result['goal'] or result['control_step']!=tick or result['revision']!=session.revision
                    or result['physical_success_asserted'] is not False
                    or result['observer_feedback_mode']!=session.observer_feedback_mode):
                raise ValueError('Planner event, installed low condition, or physical claims disagree')
            next_kind='action';last_plan=result
        elif kind=='action':
            if calibration is not None and not last_plan['reused'] and (
                    previous['kind']!='observer' or previous['phase']!='after_plan'):
                raise ValueError('New attempt must receive its own current observation, never previous cached feedback')
            actions=np.asarray(entry['offered_actions_raw23'],dtype=np.float32)
            if (entry['control_step']!=tick or entry['revision']!=session.revision
                    or entry['goal']!=session.low_goal(identity)
                    or entry['max_controls']!=min(16,case['end_control']-tick)
                    or actions.shape!=(entry['max_controls'],23)):
                raise ValueError('Action was not issued under the actual installed goal/horizon')
            token=clock.offer(identity,actions)
            if token!=entry['token']:raise ValueError('Altered offered raw23 action/identity token')
            chunks+=1;next_kind='ack'
        else:
            n=entry['consumed_controls']
            if type(n) is not int or not 1<=n<=16 or consumed+n>len(controls):
                raise ValueError('Missing or extra applied-control prefix')
            part=controls[consumed:consumed+n]
            ack=[dict(control_step=r['control_step']-1,simulator_apply_ack=r['simulator_apply_ack'],
                      action_executed_raw23=r['action_executed_raw23']) for r in part]
            expected=clock.acknowledge(identity,clock.pending['token'],ack)
            if (any(entry[k]!=v for k,v in expected.items())
                    or entry['physical_result_entered_actor'] is not False):
                raise ValueError('Causal ACK does not match the independent simulator stream')
            consumed+=n;next_kind='plan'
    if not session.closed or consumed!=len(controls):
        raise ValueError('Unclosed causal session or unaccounted simulator controls')
    return dict(case=key,seed=job['seed'],job_id=job['id'],generations=generations,reuses=reuses,
        applied_controls=consumed,chunks=chunks,final_control_step=session.feedback.control_step,
        final_revision=session.revision,observer_predictions=predictions,all_current_rgb_proprio_inputs_bound=True,
        all_actual_actions_bound_to_installed_intents=True,physical_success_asserted_by_planner=False)
