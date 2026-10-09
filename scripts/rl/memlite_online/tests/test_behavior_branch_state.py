from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from behavior_branch_state import capture_branch_metadata, restore_branch_metadata


def object_of(name, **kwargs):
    obj=type(name, (), {})();obj.__dict__.update(kwargs);return obj


def fixture():
    scene=object()
    task=object_of('BehaviorTask',activity_name='test',_reward=[1],_done=[False],_success=[False],_info=[{}],
        _termination_conditions=dict(timeout=object_of('Timeout',_done=[False]),
            predicate=object_of('PredicateGoal',_done=[False],_goal_status=[dict(satisfied=[1])])),
        _reward_functions=dict(potential=object_of('PotentialReward',_reward=[1],_info=[{}],_potential=[.5])))
    metrics=[object_of('TaskMetric',timesteps=352,render_timestep=1/30,initial_predicate_states=[0],state={scene:{'k':[1]}}),
        object_of('AgentMetric',initialized=True,next_state_cache={'x':[1]},state_cache={'x':[1]},
                  delta_agent_distance={'base':[.1]},state={scene:{'k':[2]}})]
    inst=SimpleNamespace(instance_id=42,env_idx=0,light_synchronizer=None,active=True,metrics=metrics,
                         env_accessor=SimpleNamespace(scene=scene))
    env=object_of('Environment',num_envs=1,task=task,_current_steps=[352],_current_episodes=[2])
    return SimpleNamespace(env=SimpleNamespace(env=env),instance_eval_states=[inst],n_trials=0,n_success_trials=0,total_time=11)


class BranchStateTests(unittest.TestCase):
    def test_clock_reward_metrics_restored_without_copying_scene(self):
        ev=fixture();saved=capture_branch_metadata(ev);scene=ev.instance_eval_states[0].env_accessor.scene
        ev.env.env._current_steps=[999];ev.env.env.task._reward_functions['potential']._potential=[8.]
        ev.instance_eval_states[0].metrics[0].timesteps=999
        ev.instance_eval_states[0].metrics[0].state[scene]['k'].append(3)
        restore_branch_metadata(ev,saved)
        self.assertEqual(capture_branch_metadata(ev),saved)
        self.assertIs(next(iter(ev.instance_eval_states[0].metrics[0].state)),scene)

    def test_cross_instance_or_missing_component_reject_before_mutation(self):
        ev=fixture();saved=capture_branch_metadata(ev)
        for field in ('instance','reward'):
            bad=deepcopy(saved);bad[field]=43 if field=='instance' else {}
            with self.assertRaises(ValueError):restore_branch_metadata(ev,bad)
            self.assertEqual(capture_branch_metadata(ev),saved)

    def test_unhandled_state_never_silently_accepted(self):
        ev=fixture();ev.instance_eval_states[0].light_synchronizer=object()
        with self.assertRaises(ValueError):capture_branch_metadata(ev)
        ev=fixture();ev.env.env.task._termination_conditions['other']=object()
        with self.assertRaises(ValueError):capture_branch_metadata(ev)
        ev=fixture();ev.env.env.num_envs=2
        with self.assertRaises(ValueError):capture_branch_metadata(ev)


if __name__ == '__main__':unittest.main()
