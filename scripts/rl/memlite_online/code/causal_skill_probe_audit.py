"""Replay issued-command and actual-action correspondence without model code.

Physical success is checked separately by the per-control skill auditor. This
audit binds every high input to the real RGB/proprio archive and every emitted
command/action to the simulator's independently saved applied-control stream.
"""
from dataclasses import asdict

import numpy as np

from causal_skill_probe import validate_probe_histories
from recovery_causal_handover import restore_teacher_prefix,AppliedActionClock
from g05.utils.memlite_causal_session import CausalSessionIdentity


def audit_joint_episode(events, *, models, history_row, case, job, controls, observation_hashes):
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
    session,receipt=restore_teacher_prefix(identity,projection,source_task=case['task'],source_instance=case['instance_id'])
    if receipt!=events[0]['receipt'] or events[0]['observer_predictions']!=0:
        raise ValueError('Handover differs from actual past-only issued prefix')
    clock=AppliedActionClock(session);next_kind='plan';consumed=generations=reuses=chunks=0
    for index,entry in enumerate(events[1:],1):
        kind=entry['kind'];tick=session.feedback.control_step
        if kind=='close':
            if (index!=len(events)-1 or next_kind!='plan' or consumed!=len(controls)
                    or entry['control_step']!=tick or entry['revision']!=session.revision
                    or entry['observer_predictions']!=0 or consumed<=0):
                raise ValueError('Early, duplicated, incomplete or wrong-clock close')
            session.close(identity,tick)
            continue
        if kind!=next_kind:
            raise ValueError('Each real chunk needs exactly one plan, offer, and ACK in order')
        if kind=='plan':
            result=entry['result'];due=session.planning_due(identity,interval_controls=128)
            if (type(result['reused']) is not bool or result['reused']==due
                    or entry['observation_sha256']!=observation_hashes.get(tick)
                    or entry['observer_predictions']!=0
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
                    or result['observer_feedback_mode']!='shadow_unknown_v1'):
                raise ValueError('Planner event, installed low condition, or physical claims disagree')
            next_kind='action'
        elif kind=='action':
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
        final_revision=session.revision,observer_predictions=0,all_current_rgb_proprio_inputs_bound=True,
        all_actual_actions_bound_to_installed_intents=True,physical_success_asserted_by_planner=False)
