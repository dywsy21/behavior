"""Read-only OmniGibson measurements for shared, intent-aligned skill rewards.

Exact native object/scope identities only. Category-only labels need an already
verified reference binding before reaching this adapter. Never guess the first
or nearest object. This module is simulator/reward-side, not an actor feature.
"""
import math
import numpy as np

from recovery_corpus import digest
from skill_aligned_reward import SkillIdentity, skill_measurement


def vector(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Invalid measured world-space vector')
    return result


def measured_bool(value):
    if hasattr(value, 'item'):
        value = value.item()
    if type(value) is not bool:
        raise ValueError('Unknown physical predicate is not False')
    return value


def object_state(obj, name):
    values = [value for kind, value in obj.states.items() if kind.__name__ == name]
    if len(values) != 1:
        raise ValueError('Missing/ambiguous actual object state: ' + name)
    return values[0]


def grasped(robot, obj, arm):
    value = robot.is_grasping(arm=arm, candidate_obj=obj).name
    if value not in ('TRUE', 'FALSE'):
        raise ValueError('Unknown contact cannot label a grasp or release')
    return value == 'TRUE'


def distance_to_box(point, obj):
    low, high = map(vector, obj.aabb)
    if np.any(high < low):
        raise ValueError('Invalid measured object collision bounds')
    return float(np.linalg.norm(np.maximum(np.maximum(low-vector(point), vector(point)-high), 0.)))


def articulation_contract(obj, part):
    # This matches the pinned collector's functional-clearance measurement.
    # Multi-part objects require a verified part-aware contract; they must not
    # quietly fall back to the first joint or the whole-object 5% Open flag.
    if part:
        raise ValueError('Part-specific articulation not yet validated')
    from omnigibson.object_states.open_state import _get_relevant_joints, _compute_joint_threshold
    two_sided, joints, directions = _get_relevant_joints(obj)
    if two_sided or len(joints) != 1:
        raise ValueError('Ambiguous articulation needs a validated exact-part adapter')
    _, opened, closed = _compute_joint_threshold(joints[0], directions[0])
    opened, closed = float(opened), float(closed)
    if not all(map(math.isfinite, (opened, closed))) or abs(opened-closed) < 1e-5:
        raise ValueError('Invalid measured joint travel')
    return object_state(obj, 'Open'), joints[0], opened, closed


class OmniSkillMeasurements:
    """One isolated session/episode/member with no simulator state setters.

    Supported live measurement families: GRASP, PLACE_IN/ON, single-joint
    OPEN/CLOSE. NAV remains explicitly blocked until its reachable-pose and
    collision contract is validated; object proximity is not navigation.
    """
    def __init__(self, accessor, identity, skill, *, bundle=None):
        if not isinstance(identity, SkillIdentity):
            raise ValueError('Missing isolated skill identity')
        bundle = [skill] if bundle is None else bundle
        if (not isinstance(bundle, list) or not 0 <= identity.member < len(bundle)
                or digest(bundle) != identity.bundle_sha256 or bundle[identity.member] != skill):
            raise ValueError('Reward member does not match the exact current semantic bundle')
        self.accessor, self.identity, self.skill = accessor, identity, dict(skill)
        self.robot = accessor.robot
        self.target = self.resolve(skill.get('target'))
        self.destination, self.articulation = None, None
        verb = skill['verb']
        if verb == 'GRASP':
            if (skill.get('arm') not in ('LEFT', 'RIGHT','UNSPECIFIED')
                    or (skill['arm']!='UNSPECIFIED' and skill['arm'].lower() not in self.robot.arm_names)):
                raise ValueError('Unknown requested GRASP arm')
        elif verb in ('PLACE_IN', 'PLACE_ON'):
            self.destination = self.resolve(skill.get('destination'))
            if self.destination is self.target:
                raise ValueError('An object cannot be placed onto/into itself')
            object_state(self.target, 'Inside' if verb == 'PLACE_IN' else 'OnTop')
        elif verb in ('OPEN_DOOR','OPEN_DRAWER','OPEN_LID','CLOSE_DOOR','CLOSE_DRAWER','CLOSE_LID'):
            self.articulation = articulation_contract(self.target, skill.get('target_part', ''))
        else:
            raise ValueError('Live measurement not validated for skill: ' + verb)

    def resolve(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError('Missing exact target binding')
        scope = {key:getattr(obj, 'wrapped_obj', obj) for key,obj in self.accessor.object_scope.items()}
        matches = [obj for key,obj in scope.items() if obj is not None and (key == name or obj.name == name)]
        matches.extend(obj for obj in self.robot.scene.objects if obj.name == name)
        unique = {id(obj):obj for obj in matches}
        if len(unique) != 1 or next(iter(unique.values())) is self.robot:
            raise ValueError('Missing/ambiguous exact native target, never use category or nearest: ' + name)
        return next(iter(unique.values()))

    def read(self, identity):
        if (identity != self.identity or self.robot is not self.accessor.robot
                or self.resolve(self.skill['target']) is not self.target):
            raise ValueError('Cross-session/task/episode or stale simulator binding')
        evidence = dict(semantic_skill_sha256=digest(self.skill), exact_binding_verified=True,
                        target_native_name=self.target.name, actor_input=False)
        verb = self.skill['verb']
        if verb == 'GRASP':
            arms = list(self.robot.arm_names) if self.skill['arm']=='UNSPECIFIED' else [self.skill['arm'].lower()]
            if not arms:
                raise ValueError('Robot has no eligible grasp arm')
            contacts = {arm:grasped(self.robot,self.target,arm) for arm in arms}
            evidence.update(arm=self.skill['arm'], eligible_hand_contacts=contacts,
                target_held_by_requested_arm=any(contacts.values()),
                requested_eef_target_distance_m=min(distance_to_box(self.robot.get_eef_position(arm=arm),self.target)
                                                    for arm in arms))
        elif self.destination is not None:
            if self.resolve(self.skill['destination']) is not self.destination:
                raise ValueError('Stale placement destination')
            state_name = 'Inside' if verb == 'PLACE_IN' else 'OnTop'
            contacts = [grasped(self.robot,self.target,arm) for arm in self.robot.arm_names]
            evidence.update(destination_native_name=self.destination.name,
                relation='inside' if verb == 'PLACE_IN' else 'ontop',
                official_relation=measured_bool(object_state(self.target,state_name).get_value(self.destination)),
                released_from_all_hands=not any(contacts),
                object_speed_mps=float(np.linalg.norm(vector(self.target.get_linear_velocity()))),
                target_destination_distance_m=distance_to_box(self.target.get_position_orientation()[0],self.destination))
        else:
            state, joint, opened, closed = self.articulation
            evidence.update(official_open=measured_bool(state.get_value()),
                directed_open_fraction=(float(joint.get_state()[0])-closed)/(opened-closed))
        return skill_measurement(self.skill,evidence), evidence
