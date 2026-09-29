"""Small TRAIN-only potential shaping; no task-name or object-ID policy rules.

All geometry/goal facts stay in the simulator process. A potential is NOT a
success predicate: only the official task termination supplies task reward.
"""
from math import isfinite, sqrt


def finite_scalar(value, name):
    value = float(value)
    if not isfinite(value):
        raise ValueError(f'Nonfinite {name}')
    return value


def goal_fraction(status, count):
    """Use complete compiled conditions, not flattened atoms of OR/NOT goals."""
    if type(count) is not int or count < 1 or set(status) != {'satisfied', 'unsatisfied'}:
        raise ValueError('Nonempty compiled goal status required')
    yes, no = list(status['satisfied']), list(status['unsatisfied'])
    indices = yes + no
    if (any(type(i) is not int for i in indices) or len(indices) != count
            or set(indices) != set(range(count))):
        raise ValueError('Incomplete, duplicate, or invalid goal indices')
    return len(yes) / count


def positive_unary_targets(options, predicate):
    """Ground alternatives are supplied by BDDL; negated literals are skipped."""
    found = set()
    for option in options:
        for atom in option:
            if (isinstance(atom, (list, tuple)) and len(atom) == 2
                    and atom[0] == predicate and isinstance(atom[1], str)):
                found.add(atom[1].lstrip('?'))
    return found


def point_distance(point, low, high):
    if len(point) != 3 or len(low) != 3 or len(high) != 3:
        raise ValueError('3D geometry required')
    point = [finite_scalar(v, 'hand point') for v in point]
    low = [finite_scalar(v, 'goal bound') for v in low]
    high = [finite_scalar(v, 'goal bound') for v in high]
    if any(a > b for a, b in zip(low, high)):
        raise ValueError('Inverted goal bounds')
    return sqrt(sum(max(a - x, 0., x - b) ** 2 for x, a, b in zip(point, low, high)))


def approach_potential(hand_distances, distance_scale=.25):
    scale = finite_scalar(distance_scale, 'distance scale')
    if scale <= 0 or not hand_distances:
        raise ValueError('Hands and positive distance scale required')
    values = []
    for distance in hand_distances:
        distance = finite_scalar(distance, 'target distance')
        if distance < 0:
            raise ValueError('Negative target distance')
        values.append(1. / (1. + distance / scale))
    # Average arms: a holding hand must not hide the other hand's progress.
    return sum(values) / len(values)


def potential(progress, approach, goal_weight=.7):
    progress = finite_scalar(progress, 'goal progress')
    approach = finite_scalar(approach, 'approach')
    weight = finite_scalar(goal_weight, 'goal weight')
    if not all(0 <= x <= 1 for x in (progress, approach, weight)):
        raise ValueError('Potential components must lie in [0,1]')
    return weight * progress + (1 - weight) * approach


def shaped_reward(official, before, after, *, terminated, gamma=.9998, weight=.2):
    """One REAL control transition; truncation deliberately keeps after-Phi.

    Terminal potential is zero, including unsuccessful true termination.
    Signed differences telescope, so repeatedly approaching/retreating cannot
    repeatedly earn an unopposed bonus. Do not clip away negative shaping.
    """
    official, before, after, gamma, weight = [finite_scalar(x, 'reward input')
                                            for x in (official, before, after, gamma, weight)]
    if (official not in (0., 1.) or not 0 <= before <= 1 or not 0 <= after <= 1
            or not 0 < gamma <= 1 or not 0 <= weight <= .2
            or (official == 1 and not terminated)):
        raise ValueError('Invalid official reward, potential, discount, or shaping scale')
    successor = 0. if terminated else after
    dense = weight * (gamma * successor - before)
    return dict(official=official, shaping=dense, total=official + dense,
                potential_before=before, potential_after=after,
                effective_successor_potential=successor)


def has_learning_signal(rows, *, dense=False, threshold=1e-9):
    if dense:
        return any(abs(finite_scalar(v, 'sampled reward')) > threshold
                   for row in rows for v in row['rewards'])
    return sum(sum(row['rewards']) for row in rows) != 0


class GoalGeometryPotential:
    """Read-only OG adapter. Construct ONLY for a registered TRAIN worker.

    Goal object bindings come from the compiled BDDL scope. Positive toggle
    goals use the simulator asset's own button marker; other physical targets
    use their AABB. Systems without geometry are reported, not fabricated.
    """
    def __init__(self, env, config):
        if config != dict(kind='goal_geometry_potential_v1', weight=.2,
                          gamma=.9998, goal_weight=.7, distance_scale=.25):
            raise ValueError('Unregistered shaping configuration')
        self.env, self.config = env, dict(config)
        self.task, self.robot = env.task, env.robots[0]
        self.conditions = list(self.task.activity_goal_conditions)
        self.count = len(self.conditions)
        names, toggle_names = set(), set()
        for condition in self.conditions:
            names.update(condition.get_relevant_objects())
            toggle_names.update(positive_unary_targets(condition.flattened_condition_options, 'toggled_on'))
        self.targets, self.skipped = [], []
        for name in sorted(names):
            if name not in self.task.object_scope:
                raise ValueError('Goal object is absent from task scope')
            obj = self.task.object_scope[name]
            if obj is self.robot:
                self.skipped.append(dict(binding=name, reason='agent is not an external target'))
                continue
            if obj is None:
                self.skipped.append(dict(binding=name, reason='uninstantiated goal entity'))
                continue
            marker = None
            if name in toggle_names:
                from omnigibson.object_states import ToggledOn
                if ToggledOn not in obj.states or obj.states[ToggledOn].visual_marker is None:
                    raise ValueError('Positive toggle goal has no native toggle marker')
                marker = obj.states[ToggledOn].visual_marker
            elif not hasattr(type(obj), 'aabb') and not hasattr(obj, 'aabb'):
                self.skipped.append(dict(binding=name, reason='goal entity has no physical AABB'))
                continue
            self.targets.append((name, obj, marker))
        if not self.count or not self.targets or not self.robot.arm_names:
            raise ValueError('This pilot requires a compiled goal, physical target and hands')
        self.identity = dict(kind=config['kind'], goal_count=self.count,
            targets=[dict(binding=n, object=o.name, geometry='toggle_marker' if m is not None else 'aabb')
                     for n, o, m in self.targets], skipped=self.skipped,
            actor_inputs_changed=False, official_success_unchanged=True)

    @staticmethod
    def vector(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().tolist()
        return list(value)

    def snapshot(self, *, official_success=None):
        won, status = self.task.compiled_task.check_goal(self.task._evaluate_predicate)
        progress = goal_fraction(status, self.count)
        if bool(won) != (progress == 1.) or (official_success is not None and bool(won) != official_success):
            raise ValueError('Shaping goal status disagrees with official success')
        bounds = []
        for _, obj, marker in self.targets:
            if marker is not None:
                point = self.vector(marker.get_position_orientation()[0])
                bounds.append((point, point))
            else:
                a, b = obj.aabb
                bounds.append((self.vector(a), self.vector(b)))
        distances = {}
        for arm in self.robot.arm_names:
            fingers = self.robot.finger_links[arm]
            points = [self.vector(link.get_position_orientation()[0]) for link in fingers]
            if not points:
                raise ValueError('Hand has no valid finger geometry')
            distances[arm] = min(point_distance(p, low, high) for p in points for low, high in bounds)
        approach = approach_potential(list(distances.values()), self.config['distance_scale'])
        value = potential(progress, approach, self.config['goal_weight'])
        return dict(potential=value, goal_fraction=progress, approach=approach,
                    hand_distances=distances, goal_status=status, goal_success=bool(won),
                    geometry_targets=len(self.targets))
