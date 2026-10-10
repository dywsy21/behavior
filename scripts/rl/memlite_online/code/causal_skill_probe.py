"""Read-only joint planner/FM probe with an actually consumed teacher prefix.

The original skill defines the scored physical endpoint, not an actor command
or an RL reward. The new planner may change its command. No privileged result,
reference future, or uncalibrated observer prediction enters either model.
"""
from copy import deepcopy
import json
from pathlib import Path

from recovery_corpus import digest,file_sha
from recovery_causal_handover import restore_teacher_prefix, AppliedActionClock
from skill_training_protocol import observation_hash
from g05.utils.memlite_causal_session import CausalSessionIdentity,CausalModelIdentity


def load_causal_probe_metadata(cfg, repo):
    """Explicit opt-in; it cannot relax the three-mechanism RL requirement."""
    if 'causal_planner' not in cfg:
        return None
    from skill_training_protocol import read_only_recipe
    spec=cfg['causal_planner'];root=Path(cfg['root'])
    calibrated=isinstance(spec,dict) and spec.get('protocol')=='recorded_handover_calibrated_grasp_v1'
    if (not read_only_recipe(cfg) or not isinstance(spec,dict) or set(spec)!={'protocol','models_config','models_config_sha256',
            'inputs_manifest','inputs_manifest_sha256','planning_interval_controls','observer_predictions_enabled','scoring'}
            or spec['protocol'] not in ('recorded_handover_shadow_v1','recorded_handover_calibrated_grasp_v1')
            or type(spec['planning_interval_controls']) is not int or spec['planning_interval_controls']!=128
            or spec['observer_predictions_enabled'] is not calibrated
            or spec['scoring']!='original_restored_skill_endpoint_not_RL_reward'):
        raise ValueError('Require explicitly read-only shadow or separately calibrated GRASP pilot')
    model_path=Path(repo)/spec['models_config'];inputs=root/spec['inputs_manifest']
    if file_sha(model_path)!=spec['models_config_sha256'] or file_sha(inputs)!=spec['inputs_manifest_sha256']:
        raise ValueError('Unbound joint model identities or consumed-command history')
    models=json.loads(model_path.read_text());fixture=json.loads(inputs.read_text())
    mode='calibrated_grasp_estimate_pilot_v1' if calibrated else 'shadow_unknown_v1'
    schema='causal_planner_calibrated_grasp_pilot_v1' if calibrated else 'causal_planner_shadow_runtime_qa_v1'
    if (models['schema']!=schema or models['root']!=str(root)
            or models['low']!=cfg['model'] or models['expert_release']!=cfg['expert_release']
            or models['normalization_sha256']['low']!=cfg['stats_sha256']
            or models['observer_runtime_mode']!=mode):
        raise ValueError('Joint planner/low/normalizer model-binding mismatch')
    for key in ('planner','low','observer_backbone','observer_adapter'):
        if file_sha(root/models[key]['path'])!=models[key]['sha256']:
            raise ValueError('Changed immutable joint '+key)
    if file_sha(root/models['observer_normalization_path'])!=models['normalization_sha256']['observer']:
        raise ValueError('Changed original observer normalization')
    identities=CausalModelIdentity(**{k:models[k]['sha256'] for k in
        ('planner','low','observer_backbone','observer_adapter')},
        **{k+'_normalization':v for k,v in models['normalization_sha256'].items()})
    validate_probe_histories(fixture,cfg['cases'])
    result=dict(models=models,identities=identities,fixture=fixture,feedback_mode=mode)
    if calibrated:
        from recovery_observer_pilot_runtime import validate_pilot_receipts
        result['calibrated']=validate_pilot_receipts(models,identities,repo)
    return result


def validate_probe_histories(fixture, cases):
    if (fixture.get('schema') != 'actual_local_causal_handover_inputs_v1'
            or fixture.get('role') != 'TRAIN_engineering_only_never_SFT_calibration_or_test'):
        raise ValueError('Require recorded, immutable TRAIN handover histories')
    rows=fixture['rows']
    if len(rows) != len(cases) or {r['case'] for r in rows} != set(cases):
        raise ValueError('Exactly one recorded history per physical probe case')
    output={}
    for row in rows:
        case=cases[row['case']];prefix=row['issued_prefix']
        if (case['original_split'] != 'train' or case['recovery_split'] != 'train'
                or case['sim']['kind'] != 'recovery'
                or row['task'] != case['task'].replace('_',' ')
                or row['instance'] != case['instance_id']
                or row['source_branch_manifest_sha256'] != case['manifest_sha256']
                or row['source_replay_proof_sha256'] != case['sim']['proof_sha256']
                or row['record']['control_step'] != case['start_control']
                or prefix['control_step'] != case['start_control']
                or digest(prefix) != row['issued_prefix_sha256']):
            raise ValueError('History differs from the physically restored source/task/clock')
        # Only this observable projection is retained; not the old RGB,
        # result, current teacher proposal, or future reference trajectory.
        output[row['case']]=deepcopy(prefix)
    return output


class CausalSkillProbe:
    """One isolated high session per already bound, read-only physical job."""
    def __init__(self, cases, fixture, models, planner, low, *, log_event,calibration=None,observer_factory=None):
        self.cases=deepcopy(cases)
        self.prefixes=validate_probe_histories(fixture,cases)
        self.models,self.planner,self.low=models,planner,low
        self.log_event=log_event
        self.active={};self.completed=set();self.generations=self.reuses=self.applied_controls=0
        if (calibration is None)!=(observer_factory is None):
            raise ValueError('Calibrated pilot requires both the bound receipt and actual observer')
        if calibration is not None and calibration.models!=models:raise ValueError('Foreign calibrated models')
        self.calibration,self.observer_factory=calibration,observer_factory
        self.observers={};self.observer_counts={};self.observer_predictions=0
        self.feedback_mode='shadow_unknown_v1' if calibration is None else 'calibrated_grasp_estimate_pilot_v1'

    def _job(self, physical, job):
        if (job['case'] not in self.cases or job['phase'] != 'evaluation' or job['round'] != 0
                or physical.case != self.cases[job['case']]
                or physical.identity.episode != job['id'] or physical.version != 0
                or physical.sha != self.models.low):
            raise ValueError('Foreign case, training job, episode or low checkpoint')
        return job['id']

    def begin(self, physical, job):
        key=self._job(physical,job)
        if key in self.active or key in self.completed:
            raise ValueError('Duplicate/closed causal handover')
        case=self.cases[job['case']]
        if physical.rollout.step != case['start_control']:
            raise ValueError('Handover must precede the first new policy control')
        identity=CausalSessionIdentity(physical.identity.session,case['task'].replace('_',' '),
            case['instance_id'],key,self.models)
        options={}
        if self.calibration is not None:
            from recovery_calibrated_observer import CalibratedGraspPilotSession
            options['session_factory']=lambda i:CalibratedGraspPilotSession(i,calibration=self.calibration)
        session,receipt=restore_teacher_prefix(identity,self.prefixes[job['case']],
            source_task=case['task'],source_instance=case['instance_id'],**options)
        if self.observer_factory is not None:
            observer=self.observer_factory(session)
            if observer.session is not session or observer.identity!=identity:
                raise ValueError('Each episode must own a fresh bound observer window')
            self.observers[key]=observer;self.observer_counts[key]=0
        self.active[key]=(session,AppliedActionClock(session))
        self.log_event('handover',dict(job=job,receipt=receipt,observer_predictions=0))

    def observe(self,physical,job,observation,phase):
        key=self._job(physical,job);session,_=self.active[key]
        if self.calibration is None:return
        if session.feedback.control_step!=physical.rollout.step:raise ValueError('Observer needs the actual ACK clock')
        result=self.observers[key].check(session.identity,observation)
        n=len(result.get('members',[]));self.observer_predictions+=n;self.observer_counts[key]+=n
        self.log_event('observer',dict(job=job,phase=phase,result=result,
            observation_sha256=observation_hash(observation),revision=session.revision,
            attempt_generation=session.attempt_generation,control_step=session.feedback.control_step,
            estimated_feedback=session.feedback.projection(session.identity.feedback_identity(),session.feedback.control_step)))

    def goal(self, physical, job, observation):
        key=self._job(physical,job);session,_=self.active[key]
        if session.feedback.control_step != physical.rollout.step:
            raise ValueError('High/low physical control clocks diverged')
        self.observe(physical,job,observation,'before_plan')
        result=self.planner.ensure_context(session,session.identity,observation,
            validate_low_goal=lambda goal:self.low.prepare(observation,goal),interval_controls=128)
        self.generations+=not result['reused'];self.reuses+=result['reused']
        self.log_event('plan',dict(job=job,result=result,observer_predictions=self.observer_counts.get(key,0),
            observation_sha256=observation_hash(observation),
            scored_original_skill_not_forced_into_actor=True))
        if not result['reused']:self.observe(physical,job,observation,'after_plan')
        return result['goal']

    def offer(self, physical, job, raw_actions, max_controls):
        key=self._job(physical,job);session,clock=self.active[key]
        if (type(max_controls) is not int or not 1 <= max_controls <= 16
                or getattr(raw_actions,'shape',None) != (16,23)):
            raise ValueError('Exact emitted physical prefix length required')
        token=clock.offer(session.identity,raw_actions[:max_controls])
        self.log_event('action',dict(job=job,token=token,control_step=session.feedback.control_step,
            revision=session.revision,goal=session.low_goal(session.identity),max_controls=max_controls,
            offered_actions_raw23=deepcopy(clock.pending['actions'])))

    def acknowledge(self, physical, job, controls):
        key=self._job(physical,job);session,clock=self.active[key]
        if not isinstance(controls,list) or not controls:
            raise ValueError('Nonempty real post-control ACKs required')
        # The simulator names ACKs by the post-action state s[t+1]. The
        # planner clock helper names a consumed action by its index t.
        # Drop ALL physical evidence/reward/terminal labels here.
        observable=[]
        for row in controls:
            if type(row['control_step']) is not int:
                raise ValueError('Integer post-action control clock required')
            observable.append(dict(control_step=row['control_step']-1,
                simulator_apply_ack=row['simulator_apply_ack'],
                action_executed_raw23=row['action_executed_raw23']))
        if clock.pending is None:
            raise ValueError('No causal action chunk awaiting real ACKs')
        if controls[-1]['control_step'] != physical.rollout.step:
            raise ValueError('Physical ledger must validate/advance before causal projection')
        result=clock.acknowledge(session.identity,clock.pending['token'],observable)
        if result['control_step'] != physical.rollout.step:
            raise AssertionError('Post-action clock conversion failed')
        self.applied_controls+=result['consumed_controls']
        self.log_event('ack',dict(job=job,**result,physical_result_entered_actor=False))

    def close(self, physical, job,observation=None):
        key=self._job(physical,job);session,_=self.active[key]
        if not physical.rollout.ended:
            raise ValueError('Do not close an unfinished physical episode')
        if self.calibration is not None:
            if observation is None:raise ValueError('Final calibrated check needs the real final observation')
            self.observe(physical,job,observation,'final_observation')
        session.close(session.identity,physical.rollout.step)
        self.log_event('close',dict(job=job,control_step=session.feedback.control_step,
            revision=session.revision,observer_predictions=self.observer_counts.get(key,0)))
        self.observers.pop(key,None);self.observer_counts.pop(key,None)
        del self.active[key];self.completed.add(key)

    def audit(self):
        return dict(generations=self.generations,reuses=self.reuses,applied_controls=self.applied_controls,
            active_episodes=len(self.active),completed_episodes=len(self.completed),
            observer_predictions=self.observer_predictions,observer_feedback_mode=self.feedback_mode,
            scoring='original_restored_skill_endpoint_not_RL_reward',whole_task_sr=False)
