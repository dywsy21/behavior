"""Strict short-skill on-policy control/reward boundary, without model imports.

The actor receives only the observation whitelist. Physical evidence is used
by this ledger to recompute rewards, not copied into a policy batch. A complete
same-version episode is the unit handed to the PPO update barrier.
"""
from dataclasses import asdict
import json

import numpy as np

from fixed_skill_protocol import validate_rgb_proprio
from recovery_corpus import digest
from skill_aligned_reward import SkillIdentity, SkillReward, skill_measurement
from skill_rollout import SkillRollout


def actor_observation(value):
    if set(value) != {'images', 'proprio'}:
        raise ValueError('Actor observations may not include reward, labels, clocks or physics')
    validate_rgb_proprio(value['images'], value['proprio'])
    return dict(images={k:v.copy() for k,v in value['images'].items()}, proprio=value['proprio'].copy())


class SkillTrainingSession:
    def __init__(self, case, *, session, episode, policy_version, policy_sha256, initial_evidence):
        self.case=case; self.version=policy_version; self.sha=policy_sha256
        bundle=json.loads(case['semantic_bundle'])
        if len(bundle)!=1 or case['end_control']<=case['start_control']:
            raise ValueError('Need one issued skill and its finite reference interval')
        self.skill=bundle[0]
        self.identity=SkillIdentity(session,case['task'],case['instance_id'],episode,
            case['context_id'],digest(bundle),0)
        measurement=skill_measurement(self.skill,initial_evidence)
        self.reward=SkillReward(self.identity,measurement,control_step=case['start_control'])
        self.rollout=SkillRollout(self.identity,policy_version=policy_version,
            policy_sha256=policy_sha256,control_step=case['start_control'])
        self.emitted=None; self.last=None

    def check_request(self, request, operation):
        if (request.get('op')!=operation or request.get('identity')!=asdict(self.identity)
                or request.get('policy_version')!=self.version or request.get('policy_sha256')!=self.sha
                or type(request.get('control_step')) is not int or request['control_step']!=self.rollout.step):
            raise ValueError('Cross-session/task/episode/context/policy or stale control request')

    def action_input(self, request):
        if set(request)!={'op','identity','policy_version','policy_sha256','control_step','observation'}:
            raise ValueError('Malformed action message')
        self.check_request(request,'action')
        if self.emitted is not None or self.rollout.ended:
            raise ValueError('Must ACK a chunk before requesting another; no action after end')
        return actor_observation(request['observation'])

    def emit(self, actions, *, experience_id, old_value):
        if (not isinstance(actions,np.ndarray) or actions.shape!=(16,23)
                or actions.dtype!=np.float32 or not np.isfinite(actions).all() or self.emitted is not None):
            raise ValueError('Expected one finite native raw23 action chunk')
        remaining=self.case['end_control']-self.rollout.step
        if remaining<=0:raise ValueError('Completed reference interval')
        self.rollout.begin_chunk(experience_id=experience_id,old_value=old_value,
            policy_version=self.version,policy_sha256=self.sha,control_step=self.rollout.step,
            max_controls=min(16,remaining))
        self.emitted=actions.copy()

    def ack(self, request):
        if set(request)!={'op','identity','policy_version','policy_sha256','control_step','controls','observation'}:
            raise ValueError('Malformed applied-control message')
        self.check_request(request,'ack')
        if self.emitted is None or not isinstance(request['controls'],list) or not request['controls']:
            raise ValueError('No emitted physical controls to acknowledge')
        observation=actor_observation(request['observation'])
        if len(request['controls'])>self.rollout.pending['max_controls']:
            raise ValueError('Extra controls past the emitted chunk or reference horizon')
        rewards=[]
        for index,row in enumerate(request['controls']):
            if (set(row)!={'control_step','simulator_apply_ack','action_executed_raw23','physical_evidence',
                           'official_terminal','official_truncated'}
                    or row['simulator_apply_ack'] is not True or type(row['control_step']) is not int
                    or row['control_step']!=self.rollout.step+1
                    or type(row['official_terminal']) is not bool or type(row['official_truncated']) is not bool
                    or not np.array_equal(np.asarray(row['action_executed_raw23'],dtype=np.float32),self.emitted[index])):
                raise ValueError('Missing/mismatched actual action ACK')
            measured=skill_measurement(self.skill,row['physical_evidence'])
            last=self.reward.advance(self.identity,row['control_step'],measured,protected_values={},
                official_terminal=row['official_terminal'],
                time_limit=row['official_truncated'] or row['control_step']==self.case['end_control'])
            self.rollout.acknowledge(last);rewards.append(dict(last,identity=asdict(self.identity)))
            self.last=last
        if (not self.last['terminated'] and not self.last['truncated']
                and len(request['controls'])!=self.rollout.pending['max_controls']):
            raise ValueError('Partial active chunk cannot drop unexecuted controls')
        return observation,rewards

    def finish_ack(self, next_value):
        result=self.rollout.finish_chunk(next_value=next_value,observation_control_step=self.rollout.step)
        self.emitted=None
        return result

    def training_targets(self):
        if not self.rollout.ended or self.emitted is not None:
            raise ValueError('An incomplete attempt cannot enter the optimizer')
        return self.rollout.advantages()
