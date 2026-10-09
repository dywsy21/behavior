"""Supplement world/controller snapshots with official post2 rollout clocks.

Narrow, single-environment engineering adapter, not a policy/RL checkpoint.
Only the pinned BehaviorTask + Timeout/PredicateGoal/PotentialReward layout is
supported. Unknown task/metric layouts fail closed instead of losing state.
World, RNG, sensors and any model/session memory remain caller responsibilities.
"""
from copy import deepcopy
from behavior_light_state import capture_lights,validate_light_restore,restore_lights

ENV_FIELDS = ('_current_steps', '_current_episodes')
TASK_FIELDS = ('_reward', '_done', '_success', '_info')
TERMINATIONS = {'timeout': ('Timeout', ('_done',)),
                'predicate': ('PredicateGoal', ('_done', '_goal_status'))}
REWARDS = {'potential': ('PotentialReward', ('_reward', '_info', '_potential'))}
METRICS = {'TaskMetric': ('timesteps', 'render_timestep', 'initial_predicate_states'),
           'AgentMetric': ('initialized', 'next_state_cache', 'state_cache', 'delta_agent_distance')}


def _environment(evaluator):
    env = evaluator.env
    for _ in range(5):
        if type(env).__name__ == 'Environment':
            break
        env = env.env
    if type(env).__name__ != 'Environment' or env.num_envs != 1 or len(evaluator.instance_eval_states) != 1:
        raise ValueError('Only the registered single-env prefix diagnostic is supported')
    task = env.task
    if (type(task).__name__ != 'BehaviorTask'
            or set(task._termination_conditions) != set(TERMINATIONS)
            or set(task._reward_functions) != set(REWARDS)):
        raise ValueError('Unregistered task state topology')
    inst = evaluator.instance_eval_states[0]
    if inst.env_idx != 0:
        raise ValueError('Additional environment state needs its own adapter')
    return env, task, inst


def _fields(obj, names):
    return {k: deepcopy(getattr(obj, k)) for k in names}


def capture_branch_metadata(evaluator):
    env, task, inst = _environment(evaluator)
    lighting = capture_lights(inst)
    components = {}
    for name, group, spec in (('termination', task._termination_conditions, TERMINATIONS),
                              ('reward', task._reward_functions, REWARDS)):
        components[name] = {}
        for key, (cls, fields) in spec.items():
            if type(group[key]).__name__ != cls:
                raise ValueError('Unknown reward or termination state')
            components[name][key] = _fields(group[key], fields)
    metrics = []
    for metric in inst.metrics:
        cls = type(metric).__name__
        if cls not in METRICS or set(metric.state) != {inst.env_accessor.scene}:
            raise ValueError('Unknown or multi-scene metric state')
        # Capture values only: deepcopy of a Scene key could duplicate a live
        # simulator handle. Keep the original scene binding during restore.
        metrics.append(dict(kind=cls, fields=_fields(metric, METRICS[cls]),
                            state=deepcopy(metric.state[inst.env_accessor.scene])))
    if sorted(m['kind'] for m in metrics) != ['AgentMetric', 'TaskMetric']:
        raise ValueError('Both official metric adapters required')
    result=dict(schema='behavior_prefix_branch_metadata_v1', task=task.activity_name, instance=inst.instance_id,
        env=_fields(env, ENV_FIELDS), task_state=_fields(task, TASK_FIELDS), **components, metrics=metrics,
        active=inst.active, evaluator=_fields(evaluator, ('n_trials', 'n_success_trials', 'total_time')),
        policy_state_included=False, scope='single-env engineering; no automatic legal-start approval')
    if lighting is not None: result.update(schema='behavior_prefix_branch_metadata_v2', light_synchronizer=lighting)
    return result


def restore_branch_metadata(evaluator, saved):
    env, task, inst = _environment(evaluator)
    # A freshly reset AgentMetric has initialized=False and intentionally has
    # no lazy step caches yet. Validate topology without trying to capture
    # those absent current values; the complete saved snapshot supplies them.
    kinds=[type(metric).__name__ for metric in inst.metrics]
    if (saved['schema'] not in ('behavior_prefix_branch_metadata_v1','behavior_prefix_branch_metadata_v2') or saved['task'] != task.activity_name
            or saved['instance'] != inst.instance_id or saved['policy_state_included'] is not False
            or [m['kind'] for m in saved['metrics']] != kinds
            or sorted(kinds)!=['AgentMetric','TaskMetric']
            or any(set(metric.state)!={inst.env_accessor.scene} for metric in inst.metrics)):
        raise ValueError('Cross-task/instance or unregistered branch restore')
    if (saved['schema']=='behavior_prefix_branch_metadata_v2') != (saved.get('light_synchronizer') is not None):
        raise ValueError('Versioned light metadata is missing or disguised as legacy state')
    validate_light_restore(inst,saved.get('light_synchronizer'))
    updates = [(env, saved['env'], ENV_FIELDS), (task, saved['task_state'], TASK_FIELDS),
               (evaluator, saved['evaluator'], ('n_trials', 'n_success_trials', 'total_time'))]
    for group, name, spec in ((task._termination_conditions, 'termination', TERMINATIONS),
                              (task._reward_functions, 'reward', REWARDS)):
        if set(saved[name]) != set(spec):
            raise ValueError('Missing task component state')
        for key, (cls, names) in spec.items():
            if type(group[key]).__name__!=cls:raise ValueError('Unknown reward or termination state')
            updates.append((group[key], saved[name][key], names))
    for metric, state in zip(inst.metrics, saved['metrics']):
        updates.append((metric, state['fields'], METRICS[state['kind']]))
    # Validate all schemas BEFORE mutating any state.
    if any(set(values) != set(names) for _, values, names in updates):
        raise ValueError('Incomplete branch metadata')
    for obj,values,_ in updates:
        for key in values:
            if not hasattr(obj,key) and not (type(obj).__name__=='AgentMetric'
                    and getattr(obj,'initialized',None) is False
                    and key in {'next_state_cache','state_cache','delta_agent_distance'}):
                raise ValueError('Missing non-lazy current field: '+key)
    for obj, values, _ in updates:
        for key, value in values.items():
            existing = getattr(obj, key, None)
            if hasattr(existing, 'copy_') and hasattr(value, 'shape'):
                existing.copy_(value)
            else:
                setattr(obj, key, deepcopy(value))
    for metric, state in zip(inst.metrics, saved['metrics']):
        metric.state[inst.env_accessor.scene] = deepcopy(state['state'])
    inst.active = saved['active']
    restore_lights(inst,saved.get('light_synchronizer'))
