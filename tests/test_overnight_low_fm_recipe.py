"""CPU-only tests of the overnight recipe's budgets and fail-closed supervisor."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

RECIPE = Path(__file__).resolve().parents[1] / 'scripts/experiments/overnight_low_fm.py'
spec = importlib.util.spec_from_file_location('overnight_recipe', RECIPE)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class RecipeTests(unittest.TestCase):
    def test_budget_and_separate_parent(self):
        args = recipe.arguments('formal')
        self.assertIn('model.max_steps=2500', args.override)
        self.assertIn('checkpointing_steps=500', args.override)
        self.assertIn('resume_ckpt=null', args.override)
        self.assertIsNone(args.resume_ckpt)
        self.assertEqual(args.low_initial_checkpoint, str(recipe.PARENT))
        self.assertEqual((args.nproc_per_node, args.num_obs_steps, args.seed), (4, 6, 29))
        self.assertEqual(recipe.WALL_SECONDS, 28800)
        self.assertNotEqual(args.run_dir, str(recipe.PARENT.parent.parent))

    def test_smoke_is_not_formal_initial_checkpoint(self):
        smoke, formal = recipe.arguments('smoke'), recipe.arguments('formal')
        self.assertIn('model.max_steps=5', smoke.override)
        self.assertEqual(smoke.low_initial_checkpoint, formal.low_initial_checkpoint)
        self.assertNotEqual(smoke.run_dir, formal.run_dir)

    def test_unknown_phase_refused(self):
        with self.assertRaises(ValueError):
            recipe.arguments('retry')

    def test_environment_ignores_old_dry_run_and_path(self):
        with patch.dict(recipe.os.environ, {'DRY_RUN': '1', 'PYTHONPATH': '/wrong',
                                           'MEMLITE_COORDINATION_CONFIG_ONLY': '1'}):
            env = recipe.environment()
        self.assertNotIn('DRY_RUN', env)
        self.assertNotIn('MEMLITE_COORDINATION_CONFIG_ONLY', env)
        self.assertNotIn('/wrong', env['PYTHONPATH'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3')

    def test_gpu_headroom_fail_closed(self):
        with patch.object(recipe.subprocess, 'check_output', return_value='0, 33999, 47000, 0\n'):
            with self.assertRaises(RuntimeError):
                recipe.gpu_budget([0])
        with patch.object(recipe.subprocess, 'check_output', return_value='0, 50000, 30000, 0\n'):
            with self.assertRaises(RuntimeError):
                recipe.gpu_budget([1])

    def test_disk_reserve_fail_closed(self):
        with patch.object(recipe.subprocess, 'check_output', return_value='0, 50000, 30000, 0\n'), \
             patch.object(recipe.shutil, 'disk_usage', return_value=Mock(free=recipe.MIN_FREE_DISK - 1)):
            with self.assertRaises(RuntimeError):
                recipe.gpu_budget([0])

    def test_finished_child_is_never_signalled(self):
        child = Mock()
        child.poll.return_value = 0
        with patch.object(recipe.os, 'killpg') as kill:
            recipe.stop_owned_child(child)
        kill.assert_not_called()

    def test_only_owned_child_session_is_signalled(self):
        child = Mock(pid=123456)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('owned', 45), 0]
        with patch.object(recipe.os, 'killpg') as kill:
            recipe.stop_owned_child(child)
        self.assertEqual([c.args for c in kill.call_args_list],
                         [(123456, recipe.signal.SIGTERM), (123456, recipe.signal.SIGKILL)])

    def test_receipt_is_valid_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'status.json'
            recipe.publish(destination, {'state': 'running'})
            recipe.publish(destination, {'state': 'failed'})
            self.assertEqual(recipe.read(destination), {'state': 'failed'})
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_subprocess_wall_cap_stops_its_own_child(self):
        child = Mock(pid=123456)
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(recipe.subprocess, 'Popen', return_value=child), \
             patch.object(recipe.time, 'monotonic', side_effect=[0, 2]), \
             patch.object(recipe, 'stop_owned_child') as stop:
            with self.assertRaises(TimeoutError):
                recipe.execute(['owned-child'], {}, Path(directory) / 'log', 1)
        stop.assert_called_once_with(child)


if __name__ == '__main__':
    unittest.main()
