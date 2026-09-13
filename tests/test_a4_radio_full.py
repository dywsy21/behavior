"""CPU safeguards for the predeclared autonomous full-task cohort."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/experiments'
sys.path.insert(0, str(SCRIPTS))
import radio_full_manifest as manifest
from a4_radio_full_campaign import summarize, LOW_PORT, HIGH_PORT
from serve_low_fm_full import validate_checkpoint_path


class RadioFullTests(unittest.TestCase):
    def fixture(self):
        base = dict(immutable=True, teacher_prefix_actions=0, physical_feedback_trained=False,
            policy_seed=17, low_num_obs_steps=6, low_trainability_profile='low_ae_lora_history',
            high_phase_steps=1500, high_total_B_ancestry_steps=5000,
            physical_diagnostics_policy_input=False, reinject=False, teacher_eligible=False,
            high_checkpoint_sha256=manifest.HIGH_SHA)
        def read(path):
            return deepcopy(base) if path == manifest.BASE else dict(state='complete', returncode=0, source_root='/frozen')
        with patch.object(manifest, 'require_hash'), patch.object(manifest, 'read', side_effect=read), \
                patch.object(manifest, 'sha', return_value='a' * 64), \
                patch.object(manifest, 'validate_manifest', side_effect=lambda value: value):
            return manifest.build_manifest('b' * 40)

    def test_fixed_three_official_initial_states_and_full_budget(self):
        value = self.fixture()
        manifest.validate_manifest(value, verify_files=False)
        self.assertEqual([row['instance_id'] for row in value['episodes']], [301, 302, 303])
        self.assertEqual(sum(row['max_steps'] for row in value['episodes']), 9672)
        self.assertTrue(all(row['task_index'] == 0 for row in value['episodes']))

    def test_reject_prefix_oracle_budget_seed_or_checkpoint_changes(self):
        for key, wrong in [('teacher_prefix_actions', 448), ('oracle_subgoals_used', True),
                           ('physical_diagnostics_policy_input', True), ('policy_seed', 29),
                           ('low_checkpoint_step', 5000), ('action_execution_start_index', 5),
                           ('training_admissible', True)]:
            value = self.fixture()
            value[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                manifest.validate_manifest(value, verify_files=False)
        value = self.fixture()
        value['episodes'][0]['max_steps'] = 1280
        with self.assertRaises(ValueError):
            manifest.validate_manifest(value, verify_files=False)

    def test_explicit_checkpoint_step(self):
        run = Path('/run')
        validate_checkpoint_path(run, run / 'checkpoints/step_2500.pt', 2500)
        with self.assertRaises(ValueError):
            validate_checkpoint_path(run, run / 'checkpoints/step_5000.pt', 2500)

    def test_infrastructure_failure_never_becomes_policy_failure(self):
        rows = [dict(status='complete', returncode=0, task_success=True),
                dict(status='complete', returncode=0, task_success=False),
                dict(status='incomplete_wall_budget', returncode=0, task_success=False)]
        value = summarize(rows)
        self.assertEqual((value['successes'], value['completed_episodes']), (1, 2))
        self.assertEqual(value['success_rate'], .5)
        self.assertFalse(value['all_predeclared_episodes_complete'])
        self.assertIsNone(summarize([])['success_rate'])

    def test_distinct_owned_ports(self):
        self.assertEqual((LOW_PORT, HIGH_PORT), (8783, 8784))

    def test_original_physics_and_success_runner_unchanged(self):
        text = (SCRIPTS / 'run_radio_full.py').read_text()
        text = text.replace('from radio_full_manifest import validate_manifest',
                            'from native_ab_manifest import TASK_NAMES, MAX_STEPS, validate_manifest')
        text = text.replace('p.add_argument("--episode-index", type=int, choices=range(3), required=True)\n    p.add_argument("--runtime", type=Path, required=True)',
                            'p.add_argument("--task-index", type=int, choices=range(5), required=True)')
        text = text.replace('    sys.path.insert(0, str(args.runtime))\n', '')
        text = text.replace('    episode = manifest["episodes"][args.episode_index]\n',
            '    episode = manifest["episodes"][args.task_index]\n'
            '    if (episode["task_index"] != args.task_index or episode["task_name"] != TASK_NAMES[args.task_index]\n'
            '            or episode["max_steps"] != MAX_STEPS[args.task_index]\n'
            '            or episode["mode"] != "public_test" or episode["instance_id"] != 301):\n'
            '        raise ValueError("episode is not the pinned full-budget public-test task")\n')
        text = text.replace('instructions[episode["task_index"]]', 'instructions[args.task_index]')
        self.assertEqual(hashlib.sha256((text.rstrip('\n') + '\n').encode()).hexdigest(),
            'b2f69e5cbd956defb2a7f4769ff8b6b4a191e9aaabfdfc819c9562eca2cc465a')

    def test_full_service_behavior_remains_the_corrected_original(self):
        current = (SCRIPTS / 'serve_low_fm_full.py').read_text()
        tree = ast.parse(current)
        cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                   and node.name == 'FormalALowService')
        block = '\n'.join(current.splitlines()[cls.lineno - 1:cls.end_lineno])
        # The complete constructor/reset/infer class is byte-identical to the
        # earlier validated full runner; only loading/bootstrap is generalized.
        old = Path(__file__).resolve().parents[1] / 'artifacts/local-archive-20260912/root/memlite-resume.L6rqZ6/serve_a3_aligned_full_v2.py'
        if old.is_file():
            text = old.read_text()
            old_cls = next(node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.ClassDef)
                           and node.name == 'FormalALowService')
            self.assertEqual(block, '\n'.join(text.splitlines()[old_cls.lineno - 1:old_cls.end_lineno]))
        self.assertIn('LowHistoryIngress()', block)
        self.assertNotIn('initial_action_count', block)


if __name__ == '__main__':
    unittest.main()
