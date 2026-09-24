from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import probe_scene_pathtracing as variant
import probe_scene_startup as scene
import shared_pathtracing as renderer
from test_scene_startup_probe import profile


class Settings:
    def __init__(self): self.values = {'/rtx/rendermode': 'RealTimePathTracing', '/physics/unchanged': 42}
    def get(self, key): return self.values.get(key)
    def set(self, key, value): self.values[key] = value


@contextmanager
def fake_route():
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder)/'simulator.py'; source.write_text('frozen simulator')
        og = SimpleNamespace(app=None, sim=None)
        simulator = SimpleNamespace(__file__=str(source))
        def original(*args, **kwargs):
            og.app = object()
            og.sim = SimpleNamespace(scenes=[], viewer_camera=None)
            return og.sim
        og.launch = simulator._launch_simulator = Mock(side_effect=original)
        settings, record, write = Settings(), {}, Mock()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with patch.object(renderer, 'get_settings', return_value=settings):
            yield og, simulator, settings, record, write, digest


class PathTracingTests(unittest.TestCase):
    def test_import_has_no_profile_mutation_or_gpu_import(self):
        code = '''import sys
import probe_scene_startup as s
before=(s.PATH_TRACING,s.supervisor.PROFILE_SETTINGS.copy(),s.supervisor.app_configuration())
import probe_scene_pathtracing
assert before==(s.PATH_TRACING,s.supervisor.PROFILE_SETTINGS,s.supervisor.app_configuration())
assert not any(k in sys.modules for k in ('torch','isaacsim','omnigibson','carb'))
'''
        subprocess.run([sys.executable,'-c',code],cwd=scene.supervisor.REPO/'scripts/semantic_robot',check=True)

    def test_profile_keeps_caps_and_original_sensor_budget_and_resets(self):
        base=scene.supervisor
        with profile():
            original=base.app_configuration(); budget=base.BUDGET_DETAILS.copy()
            variant.configure_profile()
            self.assertTrue(scene.PATH_TRACING); self.assertTrue(scene.DISABLE_VIEWER)
            self.assertEqual(base.ENTRYPOINT,Path(variant.__file__).resolve())
            self.assertEqual(base.WALL_SECONDS,600); self.assertEqual(base.AUXILIARY_MIB,512)
            for key,value in budget.items(): self.assertEqual(base.BUDGET_DETAILS[key],value)
            actual=base.app_configuration()
            for key,value in original.items():
                if key!='extra_args': self.assertEqual(actual[key],value)
            for key,value in renderer.APP_CONFIG.items(): self.assertEqual(actual[key],value)
            self.assertIn('--/rtx/rendermode=PathTracing',actual['extra_args'])
            self.assertIn('--/rtx/pathtracing/dlss/enabled=false',actual['extra_args'])
            base.validate_settings(base.RUNTIME_SETTINGS.copy())
            changed=base.RUNTIME_SETTINGS.copy(); changed['/rtx/pathtracing/spp']=16
            with self.assertRaises(ValueError): base.validate_settings(changed)
            self.assertIs(base.SUCCESS_FIELDS['pathtracing_profile_verified'],True)
            scene.configure_profile()
            self.assertFalse(scene.PATH_TRACING); self.assertFalse(scene.DISABLE_VIEWER)
            self.assertEqual(base.app_configuration(),original)

    def test_original_constructor_then_empty_scene_switch_preserves_args_and_physics(self):
        with fake_route() as (og,sim,settings,record,write,digest):
            original=og.launch
            with renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write):
                returned=og.launch(1,device='cuda:3',physics_dt=.008)
                self.assertIs(returned,og.sim)
                original.assert_called_once_with(1,device='cuda:3',physics_dt=.008)
                self.assertEqual(settings.get('/physics/unchanged'),42)
                self.assertEqual(renderer.read_checked(settings),renderer.SETTINGS)
                self.assertTrue(record['pathtracing']['applied_before_scene'])
                renderer.validate_after_scene(og,record)
                self.assertTrue(record['pathtracing_profile_verified'])
            self.assertIs(og.launch,original)
            self.assertEqual(write.call_count,2)

    def test_native_app_total_spp_override_is_reapplied_before_validation(self):
        settings=Settings(); settings.values.update(renderer.SETTINGS)
        settings.set('/rtx/pathtracing/totalSpp',4)  # Actual installed native constructor behavior.
        with self.assertRaises(ValueError): renderer.read_checked(settings)
        self.assertEqual(renderer.apply_settings(settings),renderer.SETTINGS)
        self.assertEqual(settings.get('/physics/unchanged'),42)

    def test_source_launch_alias_or_live_app_scene_rejected_before_attempt(self):
        for bad in ('source','alias','app','sim'):
            with self.subTest(bad=bad),fake_route() as (og,sim,settings,record,write,digest):
                if bad=='source': digest='0'*64
                elif bad=='alias': og.launch=Mock()
                else: setattr(og,bad,object())
                with self.assertRaises(ValueError),renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write):
                    self.fail('entered unsafe context')
                self.assertEqual(settings.values,Settings().values)

    def test_occupied_scene_or_viewer_or_wrong_result_never_touches_renderer(self):
        for bad in ('scene','viewer','result','app'):
            with self.subTest(bad=bad),fake_route() as (og,sim,settings,record,write,digest):
                def wrong():
                    og.app=object() if bad!='app' else None
                    og.sim=SimpleNamespace(scenes=[object()] if bad=='scene' else [],
                                           viewer_camera=object() if bad=='viewer' else None)
                    return object() if bad=='result' else og.sim
                original=og.launch; original.side_effect=wrong
                with renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write):
                    with self.assertRaises(ValueError): og.launch()
                self.assertIs(og.launch,original)
                self.assertFalse(record['pathtracing']['applied_before_scene'])
                self.assertEqual(settings.values,Settings().values)

    def test_changed_original_mode_is_not_silently_overridden(self):
        with fake_route() as (og,sim,settings,record,write,digest):
            settings.set('/rtx/rendermode','unknown')
            with renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write):
                with self.assertRaises(ValueError): og.launch()
            self.assertEqual(settings.get('/rtx/rendermode'),'unknown')

    def test_observed_native_legacy_mode_is_accepted_only_before_profile_not_after(self):
        with fake_route() as (og,sim,settings,record,write,digest):
            settings.set('/rtx/rendermode','RaytracedLighting')
            with renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write): og.launch()
            self.assertEqual(record['pathtracing']['original_render_mode'],'RaytracedLighting')
            renderer.validate_after_scene(og,record)
            settings.set('/rtx/rendermode','RaytracedLighting')
            with self.assertRaises(ValueError): renderer.validate_after_scene(og,record)

    def test_setter_ignored_or_late_settings_drift_fails(self):
        with fake_route() as (og,sim,settings,record,write,digest):
            settings.set=Mock()
            with renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write):
                with self.assertRaises(ValueError): og.launch()
            self.assertFalse(record['pathtracing']['applied_before_scene'])
        with fake_route() as (og,sim,settings,record,write,digest):
            with renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write): og.launch()
            for key,value in renderer.SETTINGS.items():
                settings.values=dict(renderer.SETTINGS)
                settings.set(key,None)
                with self.subTest(key=key),self.assertRaises(ValueError): renderer.validate_after_scene(og,record)

    def test_repeat_failed_nested_escaped_and_cross_context_attempts_rejected(self):
        for failure in (False,True):
            with self.subTest(failure=failure),fake_route() as (og,sim,settings,record,write,digest):
                original=og.launch
                if failure: original.side_effect=RuntimeError('native failure')
                def context(): return renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write)
                with context():
                    escaped=og.launch
                    with self.assertRaises((ValueError,RuntimeError)),context(): pass
                    if failure:
                        with self.assertRaises(RuntimeError): escaped()
                    else: escaped()
                    with self.assertRaises(RuntimeError): escaped()
                with self.assertRaises(RuntimeError): escaped()
                og.app=og.sim=None
                with self.assertRaises(RuntimeError),context(): pass
                self.assertIs(og.launch,original); self.assertEqual(original.call_count,1)

    def test_unused_escaped_context_cannot_launch_but_fresh_context_can(self):
        with fake_route() as (og,sim,settings,record,write,digest):
            with renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write): escaped=og.launch
            with self.assertRaises(RuntimeError): escaped()
            with renderer.before_scene(og,sim,source_sha256=digest,record=record,write=write): og.launch()


if __name__=='__main__': unittest.main()
