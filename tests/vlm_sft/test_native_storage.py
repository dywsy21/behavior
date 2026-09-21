import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/vlm_sft"))
import native_storage as storage
from native_reference_profile import validate_profile


class NativeStorageTests(unittest.TestCase):
    def test_explicit_exact_profile_and_no_legacy_mixing(self):
        self.assertFalse(storage.validate_spec({}))
        self.assertTrue(storage.validate_spec({"storage":storage.SPEC}))
        for value in (None,{},"new",{**storage.SPEC,"profile":"unknown"},{**storage.SPEC,"sda_min_free_GiB":True},
                      {**storage.SPEC,"runtime_max_MiB":16385},{**storage.SPEC,"extra":0},
                      {**storage.SPEC,"runtime_root":"/mnt/sdc1/cache"}):
            with self.subTest(value=value),self.assertRaises(ValueError):storage.validate_spec({"storage":value})
        with self.assertRaises(ValueError):validate_profile({"storage":storage.SPEC},{},{})
        with self.assertRaises(ValueError):storage.runtime_environment("/tmp/unregistered")

    def test_actual_paths_environment_reserve_and_cache_size(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);nvme=root/"nvme";sda=root/"sda";nvme.mkdir();sda.mkdir()
            exp=nvme/"experiment";runtime=nvme/"runtime";out=exp/"reference_train114_v1"
            with patch.multiple(storage,NVME=nvme,SDA=sda,EXPERIMENT_ROOT=exp,RUNTIME_ROOT=runtime):
                expected=storage.runtime_environment(out)
                before_home=os.environ.get("HOME");before_codex=os.environ.get("CODEX_HOME")
                with patch.dict(os.environ,expected),patch.object(storage.os.path,"ismount",return_value=True),patch.object(
                        storage.shutil,"disk_usage",return_value=SimpleNamespace(free=100*1024**3)):
                    guard=storage.RuntimeStorage({"storage":storage.SPEC},out);receipt=guard.create()
                    self.assertEqual(receipt["runtime_bytes"],0);self.assertFalse(receipt["source_environment_fully_readonly"])
                    self.assertEqual(os.environ.get("HOME"),before_home);self.assertEqual(os.environ.get("CODEX_HOME"),before_codex)
                    for key in storage.CACHE_SUBDIRS:self.assertTrue(Path(expected[key]).is_dir())
                    for key in expected:
                        with patch.dict(os.environ,{key:"changed"}),self.assertRaises(ValueError):guard.check()
                    with patch.object(storage.shutil,"disk_usage",side_effect=lambda p:SimpleNamespace(
                            free=(31 if p==sda else 100)*1024**3)),self.assertRaises(RuntimeError):guard.check()
                    with patch.object(storage.shutil,"disk_usage",side_effect=lambda p:SimpleNamespace(
                            free=(79 if p==nvme else 100)*1024**3)),self.assertRaises(RuntimeError):guard.check()
                    file=guard.root/"tmp/file";file.write_bytes(b"123")
                    self.assertEqual(guard.check()["runtime_bytes"],3)
                    with patch.object(storage,"tree_bytes",return_value=16384*1024**2),self.assertRaises(RuntimeError):guard.check()
                    with self.assertRaises(ValueError):storage.tree_bytes(runtime,nvme.stat().st_dev+1)
                    with patch.object(storage.os,"scandir",side_effect=PermissionError("unreadable")),self.assertRaises(PermissionError):guard.check()
                    unreadable=guard.root/"mode000";unreadable.mkdir();unreadable.chmod(0)
                    try:
                        with self.assertRaises(PermissionError):guard.check()
                    finally:unreadable.chmod(0o700)
                    link=guard.root/"tmp/outside";link.symlink_to(sda,target_is_directory=True)
                    with self.assertRaises(ValueError):guard.check()

    def test_symlink_escape_and_nonmount_fail_before_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);nvme=root/"nvme";sda=root/"sda";nvme.mkdir();sda.mkdir()
            runtime=nvme/"runtime";runtime.symlink_to(sda,target_is_directory=True)
            with patch.multiple(storage,NVME=nvme,SDA=sda,EXPERIMENT_ROOT=nvme/"experiment",RUNTIME_ROOT=runtime):
                out=nvme/"experiment/run";env=storage.runtime_environment(out)
                with patch.dict(os.environ,env),patch.object(storage.os.path,"ismount",return_value=True):
                    with self.assertRaises(ValueError):storage.RuntimeStorage({"storage":storage.SPEC},out).create()
                with patch.object(storage.os.path,"ismount",return_value=False):
                    with self.assertRaises(ValueError):storage.RuntimeStorage({"storage":storage.SPEC},out).check()
                self.assertFalse((sda/"run").exists())


if __name__=="__main__":unittest.main()
