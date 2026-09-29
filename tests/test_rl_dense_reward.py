from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from g05.rl.rewards import (goal_fraction, positive_unary_targets, point_distance,
    approach_potential, potential, shaped_reward, has_learning_signal, GoalGeometryPotential)
from g05.rl.dense_recipe import REWARD, reward_batch_audit


class DenseRewardTests(unittest.TestCase):
    def test_compiled_condition_indices_not_atom_counts(self):
        # An OR and a NOT are each one compiled goal, not three positive atoms.
        self.assertEqual(goal_fraction(dict(satisfied=[0], unsatisfied=[1]), 2), .5)
        for status in (dict(satisfied=[0], unsatisfied=[0]), dict(satisfied=[0], unsatisfied=[]),
                       dict(satisfied=[0], unsatisfied=[float('nan')])):
            with self.assertRaises(ValueError): goal_fraction(status, 2)

    def test_toggle_binding_excludes_negated_and_unrelated_predicates(self):
        options = [[['not', ['toggled_on', 'wrong']], ['inside', 'a', 'b']],
                   [['toggled_on', '?appliance'], ['toggled_on', 'other']]]
        self.assertEqual(positive_unary_targets(options, 'toggled_on'), {'appliance', 'other'})

    def test_bounded_geometry_and_both_hands(self):
        self.assertEqual(point_distance([2, 0, 0], [0, 0, 0], [1, 1, 1]), 1.)
        self.assertEqual(point_distance([.5, .5, .5], [0, 0, 0], [1, 1, 1]), 0.)
        self.assertGreater(approach_potential([0, .1]), approach_potential([0, .5]))
        self.assertLessEqual(potential(1, 1), 1)
        with self.assertRaises(ValueError): approach_potential([float('nan')])
        with self.assertRaises(ValueError): point_distance([0, 0, 0], [2, 0, 0], [1, 1, 1])

    def test_approach_and_retreat_cannot_repeat_unopposed_bonus(self):
        gamma=.9998
        forward=shaped_reward(0, .2, .8, terminated=False, gamma=gamma)['shaping']
        backward=shaped_reward(0, .8, .2, terminated=False, gamma=gamma)['shaping']
        self.assertGreater(forward, 0); self.assertLess(backward, 0)
        self.assertAlmostEqual(forward+gamma*backward, .2*(gamma**2*.2-.2))
        self.assertLessEqual(forward+gamma*backward, 0)

    def test_terminal_and_timeout_are_not_interchangeable(self):
        success=shaped_reward(1, .3, 1, terminated=True)
        failure=shaped_reward(0, .3, .8, terminated=True)
        timeout=shaped_reward(0, .3, .8, terminated=False)
        self.assertAlmostEqual(success['total'], .94)
        self.assertAlmostEqual(failure['total'], -.06)
        self.assertGreater(timeout['shaping'], 0)
        self.assertEqual(success['effective_successor_potential'], 0.)
        self.assertEqual(timeout['effective_successor_potential'], .8)
        with self.assertRaises(ValueError): shaped_reward(1, .2, .5, terminated=False)

    def test_discounted_telescoping_over_control_chunks(self):
        gamma=.9
        potentials=[.1, .4, .3, .7]
        rows=[shaped_reward(0, a, b, terminated=False, gamma=gamma)['shaping']
              for a,b in zip(potentials, potentials[1:])]
        flat=sum(gamma**i*r for i,r in enumerate(rows))
        chunks=(rows[0]+gamma*rows[1])+gamma**2*rows[2]
        self.assertAlmostEqual(flat, chunks)
        self.assertAlmostEqual(flat, .2*(gamma**3*.7-.1))

    def test_zero_success_and_signed_zero_sum_still_train(self):
        rows=[dict(rewards=[.01, -.01])]
        self.assertFalse(has_learning_signal(rows, dense=False))
        self.assertTrue(has_learning_signal(rows, dense=True))
        self.assertFalse(has_learning_signal([dict(rewards=[0, 0])], dense=True))
        with self.assertRaises(ValueError): has_learning_signal([dict(rewards=[float('nan')])], dense=True)

    def test_real_reward_receipt_is_checked_before_update(self):
        d=shaped_reward(0, .2, .3, terminated=False)
        row=dict(split='train', rewards=[d['total']], official_rewards=[0.],
                 shaping_rewards=[d['shaping']], reward_details=[d])
        receipt=reward_batch_audit([row])
        self.assertEqual(receipt['official_reward'], 0)
        self.assertEqual(receipt['state_changing_potential_controls'], 1)
        for bad in (dict(row, split='public_test'), dict(row, official_rewards=[1.])):
            with self.assertRaises(ValueError): reward_batch_audit([bad])

    def test_geometry_adapter_uses_goal_binding_and_read_only_marker(self):
        state_type=type('ToggleState', (), {})
        state_module=ModuleType('omnigibson.object_states'); state_module.ToggledOn=state_type
        def point(position):
            return SimpleNamespace(get_position_orientation=lambda:(position, [0,0,0,1]))
        target=SimpleNamespace(name='asset_42', states={state_type:SimpleNamespace(visual_marker=point([0,0,0]))})
        unrelated=SimpleNamespace(name='decoy', aabb=([-1,-1,-1],[1,1,1]))
        condition=SimpleNamespace(get_relevant_objects=lambda:['device.n.01_1'],
            flattened_condition_options=[[['toggled_on','device.n.01_1']]])
        task=SimpleNamespace(activity_goal_conditions=[condition], object_scope={'device.n.01_1':target,'decoy':unrelated},
            _evaluate_predicate=lambda *args:False,
            compiled_task=SimpleNamespace(check_goal=lambda fn:(False,dict(satisfied=[],unsatisfied=[0]))))
        robot=SimpleNamespace(arm_names=['left','right'],
            finger_links={'left':[point([.5,0,0])],'right':[point([.1,0,0])]})
        with patch.dict(sys.modules, {'omnigibson.object_states':state_module}):
            provider=GoalGeometryPotential(SimpleNamespace(task=task,robots=[robot]), REWARD)
            snapshot=provider.snapshot(official_success=False)
        self.assertEqual([x['object'] for x in provider.identity['targets']], ['asset_42'])
        self.assertEqual(snapshot['hand_distances'], {'left':.5,'right':.1})
        self.assertFalse(provider.identity['actor_inputs_changed'])
        with self.assertRaises(ValueError): provider.snapshot(official_success=True)


if __name__ == '__main__': unittest.main()
