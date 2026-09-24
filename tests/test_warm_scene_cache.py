import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import shared_cached_runtime as caches
import probe_scene_warm_cache as warm
import probe_scene_reduced_cameras as reduced
import probe_scene_startup as scene
from test_scene_startup_probe import profile


def fixture(root):
    for index,rel in enumerate(caches.CACHE_TREES):
        folder = root/rel; folder.mkdir(parents=True)
        (folder/'test.bin').write_bytes(bytes([index])*9)
    (root/'data').mkdir(); (root/'data/not_a_cache').write_text('do not copy')
    return caches.manifest(root)


class WarmCacheTests(unittest.TestCase):
    def test_import_is_inert_and_cpu_only(self):
        code = '''import sys
import probe_scene_startup as s
before=(s.WORKER_PREPARE,s.supervisor.ENTRYPOINT)
import probe_scene_warm_cache
assert before==(s.WORKER_PREPARE,s.supervisor.ENTRYPOINT)
assert not any(k in sys.modules for k in ('torch','isaacsim','omnigibson'))
'''
        subprocess.run([sys.executable,'-c',code],cwd=scene.supervisor.REPO/'scripts/semantic_robot',check=True)

    def test_profile_changes_only_cache_seed_and_traceback_not_physics_or_resource_budget(self):
        with profile():
            reduced.configure_profile()
            base = scene.supervisor
            settings, dependencies, success = base.RUNTIME_SETTINGS.copy(),base.DEPENDENCIES.copy(),base.SUCCESS_FIELDS.copy()
            warm.configure_profile()
            self.assertIs(scene.WORKER_PREPARE,warm.prepare_cache)
            self.assertEqual(base.RUNTIME_SETTINGS,settings);self.assertEqual(base.DEPENDENCIES,dependencies)
            self.assertEqual(base.WALL_SECONDS,600);self.assertEqual(base.AUXILIARY_MIB,512)
            self.assertEqual(scene.CAMERA_RESOLUTION_PROFILE,'shared_v1')
            self.assertEqual({k:base.SUCCESS_FIELDS[k] for k in success},success)
            self.assertIs(base.SUCCESS_FIELDS['cache_seed_verified'],True)
            self.assertEqual(base.BUDGET_DETAILS['cache_seed_bytes'],7715899668)
            self.assertIs(base.BUDGET_DETAILS['cache_copy_in_600s_activity_budget'],True)
            reduced.configure_profile()
            self.assertIsNone(scene.WORKER_PREPARE)
            self.assertNotIn('cache_seed_verified',base.SUCCESS_FIELDS)

    def test_copy_is_verified_independent_and_only_four_allowed_trees(self):
        with tempfile.TemporaryDirectory() as folder:
            source, dest = Path(folder)/'source',Path(folder)/'dest'; dest.mkdir()
            rows=fixture(source)
            with patch.object(caches.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                result=caches.copy_frozen(source,dest,expected_sha=caches.digest_manifest(rows),
                                         expected_count=4,expected_bytes=36)
            self.assertEqual(result,rows);self.assertFalse((dest/'data').exists())
            for row in rows:
                self.assertNotEqual((source/row['path']).stat().st_ino,(dest/row['path']).stat().st_ino)
            (dest/rows[0]['path']).write_bytes(b'new owned cache')
            self.assertEqual(caches.manifest(source),rows)

    def test_changed_source_wrong_totals_existing_destination_and_overlap_rejected(self):
        for mode in ('hash','count','bytes','occupied','same','nested'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                source,dest=Path(folder)/'source',Path(folder)/'dest';dest.mkdir();rows=fixture(source)
                if mode=='occupied':(dest/'cuda').mkdir();(dest/'cuda/user.bin').write_text('preserve')
                if mode=='same':dest=source
                if mode=='nested':dest=source/'other';dest.mkdir()
                with self.assertRaises(ValueError):
                    caches.copy_frozen(source,dest,expected_sha='0'*64 if mode=='hash' else caches.digest_manifest(rows),
                                       expected_count=5 if mode=='count' else 4,expected_bytes=37 if mode=='bytes' else 36)
                self.assertEqual(caches.manifest(source),rows)

    def test_symlinks_special_files_and_size_limits_rejected(self):
        for mode in ('link','fifo','bytes','files'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                source=Path(folder)/'source';fixture(source)
                if mode=='link':(source/'cuda/redirect').symlink_to(source/'data/not_a_cache')
                if mode=='fifo':
                    import os
                    os.mkfifo(source/'cuda/fifo')
                with patch.object(caches,'MAX_BYTES',1 if mode=='bytes' else caches.MAX_BYTES), \
                     patch.object(caches,'MAX_FILES',1 if mode=='files' else caches.MAX_FILES), self.assertRaises(ValueError):
                    caches.manifest(source)

    def test_copy_requires_headroom_and_never_overwrites_original(self):
        with tempfile.TemporaryDirectory() as folder:
            source,dest=Path(folder)/'source',Path(folder)/'dest';dest.mkdir();rows=fixture(source)
            with patch.object(caches.shutil,'disk_usage',return_value=SimpleNamespace(free=1)),self.assertRaises(ValueError):
                caches.copy_frozen(source,dest,expected_sha=caches.digest_manifest(rows),expected_count=4,expected_bytes=36)
            self.assertEqual(caches.manifest(source),rows);self.assertFalse(any(dest.iterdir()))

    def test_stopped_receipts_and_original_pid_checks_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            documents={'launch.json':{'source_commit':warm.OLD_COMMIT,'runtime':str(warm.SOURCE)},
                'worker.json':{},'supervisor.json':{'status':'failed','exit_code':-15,
                                                   'supervisor_pid':3503152,'worker_pid':3503159}}
            hashes={}
            for name,value in documents.items():
                (root/name).write_text(json.dumps(value));hashes[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            with patch.object(warm,'OLD_RUN',root),patch.object(warm,'RECEIPTS',hashes):
                warm.source_stopped()
                original=Path.exists
                with patch.object(Path,'exists',lambda p:True if str(p)=='/proc/3503159' else original(p)),self.assertRaises(ValueError):
                    warm.source_stopped()
                (root/'worker.json').write_text('{} changed')
                with self.assertRaises(ValueError):warm.source_stopped()

    def test_prepare_sets_success_only_after_copy_and_source_reverification(self):
        with profile(),tempfile.TemporaryDirectory() as folder:
            warm.configure_profile();base=scene.supervisor
            source,dest=Path(folder)/'source',Path(folder)/'dest';dest.mkdir();rows=fixture(source)
            record={}
            with patch.object(warm,'source_stopped') as stopped,patch.object(warm,'copy_frozen',return_value=rows) as copy, \
                 patch.object(base,'write') as write:
                warm.prepare_cache(record)
                self.assertEqual(stopped.call_count,2);copy.assert_called_once()
                self.assertTrue(record['cache_seed_verified'])
                self.assertTrue(record['cache_seed']['independent_copies'])
                self.assertTrue(record['cache_seed']['cache_hit_not_established'])
                self.assertIn('cache_seed_manifest.json',[c.args[0] for c in write.call_args_list])
            failed={}
            with patch.object(warm,'source_stopped'),patch.object(warm,'copy_frozen',side_effect=ValueError('changed')), \
                 patch.object(base,'write'),self.assertRaises(ValueError):warm.prepare_cache(failed)
            self.assertIs(failed['cache_seed_verified'],False)


if __name__=='__main__':unittest.main()
