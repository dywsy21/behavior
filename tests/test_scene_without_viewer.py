from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import probe_scene_without_viewer as variant
import probe_scene_startup as scene
from test_scene_startup_probe import profile


class Macros:
    RENDER_VIEWER_CAMERA = True
    @contextmanager
    def unlocked(self): yield


class NoViewerTests(unittest.TestCase):
    def test_import_does_not_opt_in_or_import_gpu_stack(self):
        script='''import sys
import probe_scene_startup as scene
before=(scene.DISABLE_VIEWER,scene.supervisor.ENTRYPOINT,scene.supervisor.OUTPUT)
import probe_scene_without_viewer
assert before==(scene.DISABLE_VIEWER,scene.supervisor.ENTRYPOINT,scene.supervisor.OUTPUT)
assert not any(x in sys.modules for x in ('torch','omnigibson','isaacsim'))
'''
        subprocess.run([sys.executable,'-c',script],cwd=scene.supervisor.REPO/'scripts/semantic_robot',check=True)

    def test_variant_changes_only_registered_viewer_paths_and_adds_checks(self):
        base=scene.supervisor
        with profile(),patch.object(scene,'DISABLE_VIEWER',False):
            budget=base.BUDGET_DETAILS.copy(); success=base.SUCCESS_FIELDS.copy()
            settings=base.RUNTIME_SETTINGS.copy(); config=base.app_configuration()
            variant.configure_profile()
            self.assertTrue(scene.DISABLE_VIEWER)
            self.assertEqual(base.ENTRYPOINT,Path(variant.__file__).resolve())
            self.assertEqual(base.WALL_SECONDS,600); self.assertEqual(base.AUXILIARY_MIB,512)
            self.assertEqual(base.RUNTIME_SETTINGS,settings)
            for key,value in budget.items(): self.assertEqual(base.BUDGET_DETAILS[key],value)
            for key,value in success.items(): self.assertEqual(base.SUCCESS_FIELDS[key],value)
            # Only private log/cache paths change in the app configuration.
            for key,value in config.items():
                if key!='extra_args': self.assertEqual(base.app_configuration()[key],value)
            self.assertIs(base.SUCCESS_FIELDS['render_viewer_camera'],False)
            self.assertIs(base.SUCCESS_FIELDS['viewer_camera_absent'],True)

    def test_macro_is_only_set_before_app_and_robot_camera_settings_untouched(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(scene.supervisor,'OUTPUT',Path(folder)), \
             patch.object(scene,'DISABLE_VIEWER',True):
            gm=Macros(); gm.OTHER_RENDER_SETTING='unchanged'; record={}
            scene.configure_viewer_before_launch(gm,SimpleNamespace(app=None,sim=None),record)
            self.assertFalse(gm.RENDER_VIEWER_CAMERA); self.assertEqual(gm.OTHER_RENDER_SETTING,'unchanged')
            self.assertEqual(record,{'render_viewer_camera':False})
            for app,sim in ((object(),None),(None,object())):
                with self.assertRaises(ValueError): scene.configure_viewer_before_launch(Macros(),SimpleNamespace(app=app,sim=sim),{})

    def test_actual_viewer_absence_is_required_not_only_macro_claim(self):
        with patch.object(scene,'DISABLE_VIEWER',True):
            gm=Macros(); gm.RENDER_VIEWER_CAMERA=False; record={}
            scene.validate_viewer_after_scene(gm,SimpleNamespace(viewer_camera=None),record)
            self.assertEqual(record,{'viewer_camera_absent':True})
            with self.assertRaises(ValueError): scene.validate_viewer_after_scene(gm,SimpleNamespace(viewer_camera=object()),{})
            with self.assertRaises(ValueError): scene.validate_viewer_after_scene(Macros(),SimpleNamespace(viewer_camera=None),{})

    def test_original_scene_profile_does_not_touch_viewer_or_require_viewer_attributes(self):
        with patch.object(scene,'DISABLE_VIEWER',False):
            record={}
            scene.configure_viewer_before_launch(object(),object(),record)
            scene.validate_viewer_after_scene(object(),object(),record)
            self.assertEqual(record,{})


if __name__=='__main__': unittest.main()
