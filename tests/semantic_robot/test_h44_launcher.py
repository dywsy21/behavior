import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('h44_launcher', Path(__file__).resolve().parents[2] / 'scripts/semantic_robot/launch_h44.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class H44LauncherTests(unittest.TestCase):
    def test_frozen_source_and_new_root(self):
        self.assertEqual(module.SOURCE.name, 'semantic_nearpose_197c2ba')
        self.assertEqual(module.COMMIT, '197c2ba7bc941c9850e71217164f1fd9538b03cb')
        self.assertEqual(module.ROOT.name, 'h44_reachability')

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
        self.assertEqual(command[command.index('--port') + 1], '8932')
        self.assertEqual(command[command.index('--revision') + 1], module.REVISION)
        self.assertEqual(command[command.index('--max-calls') + 1], '431')
        self.assertIn('http://127.0.0.1:8932', module.command_for('policy')[1])

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

    def test_only_exact_child_evaluation_context_is_eligible(self):
        receipt = {'reviewer':'Codex-parent', 'pid':77, 'main_gpu':3, 'max_auxiliary_MiB':512,
                   'source':str(module.CHILD_SOURCE), 'code_commit':module.CHILD_COMMIT,
                   'output':str(module.CHILD_ROOT / 'eval_t1_i1_base_v1'), 'launch_sha256':'a'*64}
        row = [module.GPU_UUID, '77', '209']
        module.check_child_binding(receipt, row)
        for key, value in (('reviewer','someone'), ('pid',78), ('pid',True), ('main_gpu',2),
                           ('max_auxiliary_MiB',1024), ('source','/tmp'), ('code_commit','old'),
                           ('output',str(module.CHILD_ROOT / 'training_v1')), ('launch_sha256','')):
            with self.subTest(key=key,value=value), self.assertRaises(RuntimeError):
                module.check_child_binding({**receipt,key:value}, row)
        for changed in ([module.GPU_UUID,'78','209'], [module.CHILD_GPU,'77','209'],
                        [module.GPU_UUID,'77','513'], [module.GPU_UUID,'77','0']):
            with self.subTest(row=changed), self.assertRaises(RuntimeError):
                module.check_child_binding(receipt, changed)

    def test_three_reviewed_repairs_enabled_and_old_run_preserved(self):
        self.assertEqual(module.OLD_RUN.name, 'h38_appearance')
        for flag in ('joint-boundary-start-v1','approach-translation-preview','near-pose-gap'):
            self.assertIn(flag,module.FLAGS)
        self.assertNotIn('completion-request',module.FLAGS)

    def test_gate_missing_any_repair_flag_blocks_next_stage(self):
        result={'gate_ok':True,'gate_failures':[],'implementation_digest':module.DIGEST}
        flags=('gripper_completion_v1','odometry_match_refinement','workspace_posture','near_contact_review',
               'appearance_memory','carry_duration_v1','joint_boundary_start_v1','approach_translation_preview','near_pose_gap')
        result.update({k:True for k in flags})
        review={'reviewer':'Codex-parent','full_numeric_and_RAW_review_passed':True,
                'video_review_passed':True,'result_sha256':'exact'}
        def invoke(r):
            with patch.object(module,'read',side_effect=[{'pid':999999999999},r,review]),patch.object(module,'sha',return_value='exact'):
                module.gate_review('radio')
        invoke(result)
        for flag in flags:
            with self.subTest(flag=flag),self.assertRaises(RuntimeError):invoke({**result,flag:False})


if __name__ == '__main__':
    unittest.main()
