from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/rl'), str(ROOT/'scripts/semantic_robot'), str(ROOT/'src')]
import rtx_paths as paths
import rtx_profile as profile


class RTXProfileTests(unittest.TestCase):
    def test_explicit_host_user_and_profile_required(self):
        with patch.object(paths.socket, 'gethostname', return_value='teai-g1'), \
                patch.object(paths.os, 'getuid', return_value=1000), \
                patch.dict(paths.os.environ, {'BEHAVIOR_RL_SIM_PROFILE': paths.PROFILE}, clear=True):
            paths.host_guard()
            with patch.dict(paths.os.environ, CUDA_VISIBLE_DEVICES='0'):
                with self.assertRaises(ValueError): paths.host_guard()
            with patch.object(paths.socket, 'gethostname', return_value='robo'):
                with self.assertRaises(ValueError): paths.host_guard()

    def test_worker_identity_and_cpu_partition(self):
        for w, instance, episode in [(0, 1, 0), (1, 138, 121)]:
            spec = dict(worker=w, instance=instance, episode=episode, task='turning_on_radio',
                task_id=0, split='train', seed=0, gpu=0, evaluation_only=False,
                actions=str(paths.ROOT/'fixtures'/f'demo_{episode}.npy'))
            paths.validate_worker(spec, evaluation=False)
            self.assertEqual(paths.cores(w), set(range(w*8, (w+1)*8)))
            for change in ({'instance': 301}, {'split': 'public_test'}, {'gpu': 2}, {'seed': 17},
                           {'actions': '/tmp/other.npy'}, {'episode': 190}, {'task_id': 1}):
                with self.assertRaises(ValueError): paths.validate_worker(dict(spec, **change), evaluation=False)
            with self.assertRaises(ValueError): paths.validate_worker(spec, evaluation=True)

    def test_speed_probe_cannot_train(self):
        for phase in ('benchmark_warmup', 'benchmark_serial', 'benchmark_parallel'):
            paths.validate_phase(phase)
        for phase in ('policy', 'expert_prefix', 'eval_rl_fp32', None):
            with self.assertRaises(ValueError): paths.validate_phase(phase)

    def test_relocation_preserves_suffix_and_refuses_unknown_paths(self):
        mappings = [(Path('/old/env'), Path('/new/env'))]
        self.assertEqual(profile.relocate('/old/env/lib/file', mappings, {}), Path('/new/env/lib/file'))
        self.assertEqual(profile.relocate('/old/window', mappings, {Path('/old/window'):Path('/new/window')}), Path('/new/window'))
        with self.assertRaises(ValueError): profile.relocate('/old/env_other/file', mappings, {})


if __name__ == '__main__': unittest.main()
