"""CPU gates for the step-explicit, unchanged-control-path L1 recipe."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'scripts/experiments'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


load('paired_a3_a4_actions')
pilot = load('a4_prefix_pilot')
service = load('serve_low_fm_prefix')


class PrefixPilotTests(unittest.TestCase):
    def test_step_2500_has_its_own_real_path(self):
        service.validate_checkpoint_path('/tmp/approved_run', '/tmp/approved_run/checkpoints/step_2500.pt', 2500)
        with self.assertRaises(ValueError):
            service.validate_checkpoint_path('/tmp/approved_run', '/tmp/approved_run/checkpoints/step_5000.pt', 2500)
        for value in (0, -1, True, '2500'):
            with self.assertRaises(ValueError):
                service.validate_checkpoint_path('/tmp/approved_run', '/tmp/approved_run/checkpoints/step_2500.pt', value)

    def test_live_service_control_class_is_unchanged(self):
        text = (ROOT / 'serve_low_fm_prefix.py').read_text()
        tree = ast.parse(text)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'FormalALowService')
        # ast.dump includes new grammar fields in Python >=3.12. Pin the exact
        # original class source bytes, located by AST, for cross-version checks.
        self.assertEqual(hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest(),
                         '7d5e0f17daf3c16a3557d8b2c36f7d794c3da49797f75634bc99549b39f4551f')

    def test_same_window_and_control_budget(self):
        manifest = dict(source_split='train', source_episode_index=121, policy_seed=17)
        context = SimpleNamespace(split='train', task_id=0, instance_id=138)
        window = SimpleNamespace(official_mode='train', execute_steps=16, max_chunks=80,
                                 frozen_window=lambda: SimpleNamespace(prefix_actions=[0]*448))
        pilot.validate_window(manifest, window, context)
        window.max_chunks = 81
        with self.assertRaises(ValueError):
            pilot.validate_window(manifest, window, context)
        window.max_chunks = 80
        context.split = 'eval'
        with self.assertRaises(ValueError):
            pilot.validate_window(manifest, window, context)

    def test_new_output_port_and_checkpoint_identity(self):
        self.assertNotEqual(pilot.OUTPUT, pilot.OLD)
        self.assertEqual(pilot.PORT, 8782)
        self.assertEqual(pilot.CHECKPOINT_SHA, pilot.CHECKPOINTS['A4'][1])


if __name__ == '__main__':
    unittest.main()
