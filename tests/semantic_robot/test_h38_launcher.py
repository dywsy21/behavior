import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('h38_launcher', Path(__file__).resolve().parents[2] / 'scripts/semantic_robot/launch_h38.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class H38LauncherTests(unittest.TestCase):
    def test_frozen_source_and_new_root(self):
        self.assertEqual(module.SOURCE.name, 'semantic_appearance_6f528b5')
        self.assertEqual(module.COMMIT, '6f528b5e6d49950331e08f11217d7f5659038b45')
        self.assertEqual(module.ROOT.name, 'h38_appearance')

    def test_gate_and_agent_exact_flags(self):
        for stage in ('radio', 'plates', 'policy'):
            with self.subTest(stage=stage):
                output, command = module.command_for(stage)
                self.assertTrue(output.is_relative_to(module.ROOT))
                for flag in module.FLAGS:
                    self.assertEqual(command.count('--' + flag), 1)
                self.assertEqual(command[command.index('--gpu') + 1], '2')
                self.assertEqual(command[command.index('--prefix') + 1], '0')
                self.assertNotIn('--replay-prefix-spec', command)
                self.assertEqual(command[command.index('--task') + 1], '3' if stage == 'plates' else '0')
                policy = stage == 'policy'
                for key, expected in (('--max-decisions', '192' if policy else '24'),
                                      ('--max-controls', '6144' if policy else '1536'),
                                      ('--max-seconds', '7200' if policy else '1200')):
                    self.assertEqual(command[command.index(key) + 1], expected)
                self.assertEqual(command.count('--gate-result'), 2 if policy else 0)

    def test_model_matches_policy_port_and_revision(self):
        _, command = module.command_for('model')
        self.assertEqual(command[command.index('--port') + 1], '8930')
        self.assertEqual(command[command.index('--revision') + 1], module.REVISION)
        self.assertEqual(command[command.index('--max-calls') + 1], '431')
        self.assertIn('http://127.0.0.1:8930', module.command_for('policy')[1])

    def test_source_and_launcher_review_both_required(self):
        receipt = {'independent_review_passed': True, 'source_commit': module.COMMIT,
                   'implementation_digest': module.DIGEST, 'launcher_sha256': 'exact',
                   'reviewer': 'Astra-max-vlm_sft_resume_20260921', 'blocking_findings': []}
        module.check_review(receipt, 'exact')
        for field, value in (('independent_review_passed', False), ('source_commit', 'old'),
                             ('implementation_digest', 'old'), ('launcher_sha256', 'old'),
                             ('reviewer', 'Codex-parent'), ('blocking_findings', ['open'])):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                module.check_review({**receipt, field: value}, 'exact')


if __name__ == '__main__':
    unittest.main()
