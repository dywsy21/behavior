"""Shared short-skill reward contracts, independent of task-specific goals.

Privileged measurements stay in this reward module, never in actor inputs.
The simulator adapter must bind exact object/part IDs and emit one measurement
per ACKed physical control. No label from elapsed time, gripper closure, final
task Q or an ungrounded model claim is accepted as skill success/failure.
"""
from dataclasses import dataclass
import math

from recovery_corpus import digest


SUPPORTED = {'GRASP', 'NAVIGATE', 'PLACE_IN', 'PLACE_ON', 'OPEN_DOOR', 'OPEN_DRAWER',
             'OPEN_LID', 'CLOSE_DOOR', 'CLOSE_DRAWER', 'CLOSE_LID'}


def finite(value, name, minimum=0., maximum=None):
    if (isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError('Invalid measured ' + name)
    return float(value)


@dataclass(frozen=True)
class SkillIdentity:
    session: str
    task: str
    instance: int
    episode: str
    context_id: str
    bundle_sha256: str
    member: int

    def __post_init__(self):
        if (not all(isinstance(x,str) and x for x in (self.session,self.task,self.episode,self.context_id))
                or type(self.instance) is not int or self.instance < 0 or type(self.member) is not int or self.member < 0
                or len(self.bundle_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.bundle_sha256)):
            raise ValueError('Unbound skill reward identity')


def skill_measurement(skill, evidence):
    """Directed potential plus a separate physical completion measurement.

    The entity/part binding is a simulator-side prerequisite. Missing evidence
    is an exception, not a negative label or an invitation to use another task.
    Units/scales are shared across tasks. Navigation is a local reachable pose
    contract, not evidence that the complete BEHAVIOR task succeeded.
    """
    verb=skill['verb']
    if verb not in SUPPORTED:
        raise ValueError('Unsupported skill reward contract: '+verb)
    if evidence.get('semantic_skill_sha256') != digest(skill) or evidence.get('exact_binding_verified') is not True:
        raise ValueError('Measurement does not belong to this exact skill/entity/part')
    def boolean(key):
        value=evidence[key]
        if type(value) is not bool:
            raise ValueError('Unknown physical predicate: '+key)
        return value
    if verb=='GRASP':
        arm=skill['arm']
        if arm not in ('LEFT','RIGHT') or evidence['arm']!=arm:
            raise ValueError('GRASP requires the requested exact hand')
        held=boolean('target_held_by_requested_arm')
        distance=finite(evidence['requested_eef_target_distance_m'],'eef-target distance')
        # Wrong hand or closed-empty fingers never satisfy held.
        return dict(potential=.25*math.exp(-distance/.10)+.75*held, achieved=held)
    if verb=='NAVIGATE':
        distance=finite(evidence['reachable_pose_distance_m'],'reachable-pose distance')
        angle=finite(evidence['heading_error_rad'],'heading error',maximum=math.pi)
        reachable=boolean('reachable_pose_verified')
        collision=boolean('collision_free')
        if not reachable:
            raise ValueError('AABB proximity is not a reachable navigation endpoint')
        return dict(potential=math.exp(-distance/.5-angle/.5),
                    achieved=bool(distance<=.08 and angle<=.15 and collision))
    if verb in ('PLACE_IN','PLACE_ON'):
        relation='inside' if verb=='PLACE_IN' else 'ontop'
        if evidence['relation']!=relation:
            raise ValueError('Placement evidence describes a different relation')
        satisfied=boolean('official_relation')
        released=boolean('released_from_all_hands')
        stable=finite(evidence['object_speed_mps'],'object speed')<=.05
        distance=finite(evidence['target_destination_distance_m'],'placement distance')
        # No reward for merely carrying near a destination or releasing above it.
        return dict(potential=.3*math.exp(-distance/.25)+.7*satisfied,
                    achieved=bool(satisfied and released and stable))
    from recovery_articulation_teacher import functional_goal
    want_open=verb.startswith('OPEN_')
    fraction=finite(evidence['directed_open_fraction'],'directed open fraction',minimum=-.02,maximum=1.02)
    goal=functional_goal(boolean('official_open'),fraction,want_open)
    if goal is None:
        # Hysteresis uncertainty is NOT failure; it is merely not complete.
        achieved=False
    else:
        achieved=goal
    progress=(fraction/.35 if want_open else (1.-fraction)/.975)
    return dict(potential=min(1.,max(0.,progress)),achieved=achieved)


class SkillReward:
    """Potential-based per-control reward, one terminal bonus per attempt.

    Terminal potential is zero. Time-limit truncation retains bootstrap and
    potential; it is not a physical FAILED label. Context switches may not
    splice two potentials or two GAE streams together. Unsupported/unknown
    measurements fail closed before returning a usable RL transition.
    """
    def __init__(self,identity,initial,*,control_step,protected_facts=(),gamma_per_control=.999,
                 potential_weight=.2,stable_controls=6):
        if (not isinstance(identity,SkillIdentity) or type(control_step) is not int or control_step<0
                or type(stable_controls) is not int or stable_controls<2):
            raise ValueError('Invalid reward start/physical debounce')
        self.identity=identity;self.step=control_step;self.stable=stable_controls
        self.gamma=finite(gamma_per_control,'gamma',minimum=0.00001,maximum=1.)
        self.weight=finite(potential_weight,'potential weight',maximum=1.)
        self.phi=self._validate(initial)
        if initial['achieved']:
            raise ValueError('Curriculum must start before skill success, not harvest solved seeds')
        if len(set(protected_facts))!=len(protected_facts):
            raise ValueError('Duplicate preserved progress facts')
        self.protected=tuple(protected_facts)
        self.success_streak=0;self.loss_streak={k:0 for k in protected_facts};self.ended=False

    @staticmethod
    def _validate(measurement):
        if set(measurement)!={'potential','achieved'} or type(measurement['achieved']) is not bool:
            raise ValueError('Missing/unknown physical measurement')
        return finite(measurement['potential'],'potential',maximum=1.)

    def advance(self,identity,control_step,measurement,*,protected_values,physical_failure=False,
                time_limit=False,official_terminal=False):
        if self.ended or identity!=self.identity or type(control_step) is not int or control_step!=self.step+1:
            raise ValueError('Cross-attempt/task, nonconsecutive or post-terminal reward')
        if (set(protected_values)!=set(self.protected) or any(type(x) is not bool for x in protected_values.values())
                or any(type(x) is not bool for x in (physical_failure,time_limit,official_terminal))):
            raise ValueError('Unknown preserved physical progress / boundary')
        next_phi=self._validate(measurement)
        self.success_streak=self.success_streak+1 if measurement['achieved'] else 0
        for key,value in protected_values.items():
            self.loss_streak[key]=0 if value else self.loss_streak[key]+1
        failure=physical_failure or any(n>=self.stable for n in self.loss_streak.values())
        success=self.success_streak>=self.stable and not failure
        terminal=bool(success or failure or official_terminal)
        shaping=self.weight*(self.gamma*(0. if terminal else next_phi)-self.phi)
        reward=shaping+float(success)-float(failure)
        self.step=control_step;self.phi=next_phi;self.ended=terminal or time_limit
        return dict(reward=reward,shaping=shaping,sparse=float(success)-float(failure),
            discount=0. if terminal else self.gamma,bootstrap=not terminal,terminated=terminal,truncated=time_limit and not terminal,
            skill_success=success,physical_failure=failure,
            outcome='SUCCEEDED' if success else 'FAILED' if failure else 'UNKNOWN' if official_terminal or time_limit else 'IN_PROGRESS',
            identity=self.identity,control_step=control_step)
