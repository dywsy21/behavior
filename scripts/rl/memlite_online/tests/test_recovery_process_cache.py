from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_process_cache import isolated_cache_env


class ProcessCacheTests(unittest.TestCase):
    def test_only_run_local_paths_and_no_home_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=isolated_cache_env(root)
            self.assertEqual(len(env),6)
            self.assertNotIn('HOME',env)
            for path in env.values():
                self.assertTrue(Path(path).is_dir())
                self.assertTrue(Path(path).is_relative_to(root))
            self.assertEqual(env,isolated_cache_env(root))

    def test_missing_control_or_escape_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):isolated_cache_env(root/'absent')
            control=root/'run';control.mkdir()
            (control/'runtime-cache').symlink_to(root,target_is_directory=True)
            with self.assertRaises(ValueError):isolated_cache_env(control)


if __name__=='__main__':unittest.main()
