from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import probe_scene_compatible_cameras as variant
import probe_scene_preconfigured_cameras as camera_profile
import probe_scene_startup as scene
import shared_pathtracing as renderer
from test_scene_startup_probe import profile


class CompatibleCameraTests(unittest.TestCase):
    def test_import_never_enables_combined_profile_or_gpu_import(self):
        code = '''import sys
import probe_scene_startup as s
before=(s.PRECONFIGURE_CAMERAS,s.PATH_TRACING,s.CAMERA_PATH_TRACING_ALLOWED,s.supervisor.ENTRYPOINT)
import probe_scene_compatible_cameras
assert before==(s.PRECONFIGURE_CAMERAS,s.PATH_TRACING,s.CAMERA_PATH_TRACING_ALLOWED,s.supervisor.ENTRYPOINT)
assert not any(k in sys.modules for k in ('torch','isaacsim','omnigibson','carb'))
'''
        subprocess.run([sys.executable,'-c',code],cwd=scene.supervisor.REPO/'scripts/semantic_robot',check=True)

    def test_explicit_profile_preserves_h55_cameras_budget_and_adds_strict_pt(self):
        with profile():
            camera_profile.configure_profile()
            base = scene.supervisor
            before, success, dependencies = base.BUDGET_DETAILS.copy(), base.SUCCESS_FIELDS.copy(), base.DEPENDENCIES.copy()
            variant.configure_profile()
            self.assertTrue(scene.PRECONFIGURE_CAMERAS)
            self.assertTrue(scene.PATH_TRACING); self.assertTrue(scene.CAMERA_PATH_TRACING_ALLOWED)
            self.assertTrue(scene.DISABLE_VIEWER)
            self.assertEqual(base.ENTRYPOINT, Path(variant.__file__).resolve())
            self.assertEqual(base.WALL_SECONDS, 600); self.assertEqual(base.AUXILIARY_MIB, 512)
            for key, value in before.items():
                if key != 'renderer_profile_changed': self.assertEqual(base.BUDGET_DETAILS[key], value)
            for key, value in success.items(): self.assertEqual(base.SUCCESS_FIELDS[key], value)
            for key, value in dependencies.items(): self.assertEqual(base.DEPENDENCIES[key], value)
            self.assertIs(base.SUCCESS_FIELDS['pathtracing_profile_verified'], True)
            self.assertIs(base.SUCCESS_FIELDS['preconfigured_cameras_verified'], True)
            self.assertEqual(base.PROFILE_SETTINGS, renderer.SETTINGS)
            self.assertEqual(base.PROFILE_APP_CONFIG, renderer.APP_CONFIG)
            for key, value in renderer.SETTINGS.items(): self.assertEqual(base.RUNTIME_SETTINGS[key], value)
            self.assertEqual(len(base.DEPENDENCIES), 24)

    def test_original_h55_h54_profiles_clear_combination_permission(self):
        import probe_scene_pathtracing as pt
        for restore in (scene.configure_profile, camera_profile.configure_profile, pt.configure_profile):
            with self.subTest(restore=restore), profile():
                variant.configure_profile(); restore()
                self.assertFalse(scene.CAMERA_PATH_TRACING_ALLOWED)
                self.assertFalse(scene.PATH_TRACING and scene.PRECONFIGURE_CAMERAS)


if __name__ == '__main__': unittest.main()
