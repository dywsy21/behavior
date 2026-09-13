"""CPU/stdlib checks for the bounded experiment recipe, no robot/GPU imports."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/experiments/train_fm_method_probe.py'
SPEC = importlib.util.spec_from_file_location('fm_method_probe_recipe_test', PATH)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def trial(name='control'):
    return dict(trial=name, settings=dict(recipe.TRIALS[name]),
        output=str(recipe.BASE / 'unit-test-only'), seed=41, max_updates=500, smoke_updates=5,
        parent_sha256=recipe.PARENT_SHA, methods_sha256='methods', entry_sha256='entry', commit='commit')


def values(name='control', phase='formal'):
    return dict(item.lstrip('+').split('=', 1) for item in recipe.overrides(trial(name), phase, 'digest'))


class RecipeTests(unittest.TestCase):
    def test_budget_data_batch_and_identity(self):
        for name in recipe.TRIALS:
            got = values(name)
            self.assertEqual(got['model.max_steps'], '500')
            self.assertEqual(got['model.batch_size'], '2')
            self.assertEqual(got['model.grad_accumulation_steps'], '2')
            self.assertEqual(got['checkpointing_steps'], '500')
            self.assertEqual(got['eval_steps'], '100')
            self.assertEqual(got['fm_method_recipe_sha256'], 'digest')
            self.assertEqual(got['resume_ckpt'], 'null')
            self.assertEqual(values(name, 'smoke')['model.max_steps'], '5')
            self.assertFalse(any(key.startswith('data.') for key in got))

    def test_lr_arm_changes_only_expert_rate(self):
        control, candidate = recipe.TRIALS['control'], recipe.TRIALS['ae_lr2x']
        self.assertEqual({k for k in control if control[k] != candidate[k]}, {'action_lr'})
        self.assertEqual(candidate['action_lr'], 2 * control['action_lr'])
        for name in ('control', 'ae_lr2x'):
            got = values(name)
            self.assertAlmostEqual(float(got['model.learning_rate']) *
                float(got['model.backbone_lr_multiplier']), 1e-5)

    def test_method_arms_one_factor_each(self):
        baseline = recipe.TRIALS['control']
        for name, expected in [('beta_stratified', 'time_sampling'), ('exec_weight2', 'execution_weight')]:
            self.assertEqual({key for key in baseline if baseline[key] != recipe.TRIALS[name][key]}, {expected})

    def test_invalid_phase_refused(self):
        with self.assertRaises(ValueError):
            recipe.overrides(trial(), 'another_5000', 'digest')

    def test_matching_pinned_spec_passes(self):
        def digest(path):
            return 'methods' if path == recipe.METHODS else 'entry'
        with patch.object(recipe, 'sha', side_effect=digest), patch.object(
                recipe.subprocess, 'check_output', side_effect=['commit\n', '']):
            recipe.validate_spec(trial())

    def test_changed_budget_settings_source_or_output_refused(self):
        with patch.object(recipe, 'sha', side_effect=lambda p: 'methods' if p == recipe.METHODS else 'entry'):
            for change in [dict(max_updates=5000), dict(seed=42), dict(parent_sha256='other'),
                           dict(output='/mnt/sdc1/robodojo'), dict(methods_sha256='changed'),
                           dict(settings=recipe.TRIALS['ae_lr2x'])]:
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    recipe.validate_spec({**trial(), **change})

    def test_dirty_or_moved_worktree_refused(self):
        with patch.object(recipe, 'sha', side_effect=lambda p: 'methods' if p == recipe.METHODS else 'entry'):
            for outputs in [['newcommit\n'], ['commit\n', ' M src/model.py\n']]:
                with patch.object(recipe.subprocess, 'check_output', side_effect=outputs), self.assertRaises(RuntimeError):
                    recipe.validate_spec(trial())

    def test_queued_fm_still_initializes_from_a4_and_pins_dependency_helper(self):
        spec = dict(**trial('beta_stratified'), initialization='a4', parent_path=str(recipe.PARENT),
            after_action={'root': 'native'}, after_fm=None, dependency_helper_sha256='helper')
        def digest(path):
            return 'methods' if path == recipe.METHODS else 'helper' if path == recipe.DEPENDENCY_HELPER else 'entry'
        with patch.object(recipe, 'sha', side_effect=digest):
            for damage in [None, {'initialization': 'native'}, {'parent_path': '/native/model.pt'},
                           {'dependency_helper_sha256': 'changed'}, {'after_fm': {'root': 'other'}}]:
                with patch.object(recipe.subprocess, 'check_output', side_effect=['commit\n', '']):
                    if damage is None:
                        recipe.validate_spec(spec)
                    else:
                        with self.subTest(damage=damage), self.assertRaises(RuntimeError):
                            recipe.validate_spec({**spec, **damage})


if __name__ == '__main__':
    unittest.main()
