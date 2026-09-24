from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import probe_scene_compatible_cameras as compatible
import probe_scene_reduced_cameras as reduced
import probe_scene_startup as scene
import shared_camera_config as cameras
from semantic_robot.v2.onboard import OnboardRGBD
from test_preconfigured_cameras import config_fixture, environment_fixture, wrapper_module
from test_scene_startup_probe import profile, capture_fixture


class ReducedCameraTests(unittest.TestCase):
    def test_import_does_not_change_profiles_or_load_gpu(self):
        code = '''import sys
import probe_scene_startup as scene
before=(scene.CAMERA_RESOLUTION_PROFILE,scene.PRECONFIGURE_CAMERAS,scene.supervisor.ENTRYPOINT)
import probe_scene_reduced_cameras
assert before==(scene.CAMERA_RESOLUTION_PROFILE,scene.PRECONFIGURE_CAMERAS,scene.supervisor.ENTRYPOINT)
assert not any(m in sys.modules for m in ('torch','omnigibson','isaacsim','carb'))
'''
        subprocess.run([sys.executable, '-c', code], cwd=scene.supervisor.REPO/'scripts/semantic_robot', check=True)

    def test_only_dimensions_change_with_original_resource_renderer_limits(self):
        with profile():
            compatible.configure_profile()
            base = scene.supervisor
            renderer, dependencies = deepcopy(base.RUNTIME_SETTINGS), deepcopy(base.DEPENDENCIES)
            reduced.configure_profile()
            self.assertEqual(scene.CAMERA_RESOLUTION_PROFILE, 'shared_v1')
            self.assertTrue(scene.PRECONFIGURE_CAMERAS and scene.PATH_TRACING and scene.DISABLE_VIEWER)
            self.assertEqual(base.ENTRYPOINT, Path(reduced.__file__).resolve())
            self.assertEqual(base.RUNTIME_SETTINGS, renderer)
            self.assertEqual(base.DEPENDENCIES, dependencies)
            self.assertEqual(base.WALL_SECONDS, 600); self.assertEqual(base.AUXILIARY_MIB, 512)
            self.assertTrue(base.SUCCESS_FIELDS['input_resolution_changed'])
            self.assertTrue(base.SUCCESS_FIELDS['camera_intrinsics_verified'])
            self.assertEqual(base.BUDGET_DETAILS['camera_views_removed'], 0)
            for key in ('robot_controls','model_calls','training_steps','expert_prefix_controls','old_policy_prefix_controls'):
                self.assertEqual(base.BUDGET_DETAILS[key], 0)
            compatible.configure_profile()
            self.assertEqual(scene.CAMERA_RESOLUTION_PROFILE, 'full_v1')
            self.assertNotIn('camera_intrinsics_verified', base.SUCCESS_FIELDS)

    def test_config_preserves_every_non_resolution_camera_field_and_all_physics(self):
        original = config_fixture()
        original['robot']['sensor_config']['VisionSensor']['sensor_kwargs'].update(
            focal_length=22.0, horizontal_aperture=20.0, clipping_range=[.01, 10.0])
        full, _ = cameras.prepare_config(original)
        low, receipt = cameras.prepare_config(original, 'shared_v1')
        self.assertEqual(receipt['configured_camera_pixels'], 466944)
        self.assertEqual(low['env_wrapper']['resolution_profile'], 'shared_v1')
        restored = deepcopy(low)
        restored['env_wrapper'] = deepcopy(full['env_wrapper'])
        for role, spec in receipt['cameras'].items():
            kwargs = restored['robot']['sensor_config'][spec['config_key']]['sensor_kwargs']
            kwargs['image_height'] = kwargs['image_width'] = cameras.RESOLUTIONS[role]
        self.assertEqual(restored, full)
        self.assertEqual(original['robot']['sensor_config']['VisionSensor']['sensor_kwargs']['image_width'], 1080)

    def test_explicit_registered_resolution_only(self):
        for value in (None, True, {}, 'anything', 'shared_v2'):
            with self.subTest(value=value), self.assertRaises(ValueError): cameras.resolutions(value)
        sizes = cameras.resolutions('shared_v1'); sizes['head'] = 1
        self.assertEqual(cameras.resolutions('shared_v1')['head'], 512)

    def test_wrapper_binds_profile_and_rejects_old_or_missing_profile(self):
        env, expected = environment_fixture('shared_v1')
        with wrapper_module():
            with self.assertRaises(ValueError): cameras.wrap_preconfigured(env)
            wrapper = cameras.wrap_preconfigured(env, 'shared_v1')
            self.assertEqual(cameras.validate_wrapper(wrapper, expected)['head']['initial_height'], 512)
            _, wrong = environment_fixture()
            with self.assertRaises(ValueError): cameras.validate_wrapper(wrapper, wrong)
            wrapper._behavior_camera_resolution_profile = None
            with self.assertRaises(ValueError): cameras.validate_wrapper(wrapper, expected)

    def test_capture_shape_and_pixel_hash_checks_use_registered_dimensions(self):
        q = np.zeros(31)
        low = capture_fixture(cameras.resolutions('shared_v1'))
        scene.validate_capture(q, q.copy(), *low, resolution_profile='shared_v1')
        with self.assertRaises(ValueError): scene.validate_capture(q, q.copy(), *low)
        with self.assertRaises(ValueError): scene.validate_capture(q, q.copy(), *capture_fixture(), resolution_profile='shared_v1')
        low[0]['head_rgb'][0, 0, 0] = 9
        with self.assertRaises(ValueError): scene.validate_capture(q, q.copy(), *low, resolution_profile='shared_v1')

    def test_real_onboard_adapter_passes_explicit_sizes_and_rejects_default_mismatch(self):
        sizes = cameras.resolutions('shared_v1')
        env, expected = environment_fixture('shared_v1')
        for sensor in env.robots[0].sensors.values():
            sensor.intrinsic_matrix = np.eye(3)
            size = sensor.image_width
            sensor.get_obs = lambda n=size: ({'rgb':np.full((n,n,3),123,dtype=np.uint8),
                                              'depth_linear':np.ones((n,n),dtype=np.float32)}, {})
        with self.assertRaises(ValueError): OnboardRGBD(env)
        adapter = OnboardRGBD(env, resolutions=sizes)
        model = SimpleNamespace(spec={'metadata':{'cameras':{v:{'width':n,'height':n} for v,n in sizes.items()}}})
        renders = []
        images, depths, receipt = adapter.read(model, render=lambda:renders.append('render'))
        self.assertEqual(len(renders),4)
        for view, size in sizes.items():
            self.assertEqual(images[view+'_rgb'].shape, (3,size,size))
            self.assertEqual(depths[view].shape, (size,size))
            self.assertEqual(receipt[view]['control_steps_in_capture'], 0)
            self.assertIs(adapter.sensors[view], env.robots[0].sensors[expected[view]['name']])
        env.step.assert_not_called(); env.load_observation_space.assert_not_called()

    def test_real_onboard_adapter_rejects_invalid_or_omitted_resolution_roles(self):
        for value in ({}, {'head':512}, {'head':512,'left_wrist':320,'right_wrist':320,'viewer':320},
                      {'head':True,'left_wrist':320,'right_wrist':320},
                      {'head':512.,'left_wrist':320,'right_wrist':320},
                      {'head':0,'left_wrist':320,'right_wrist':320},
                      {'head':512,'left_wrist':-1,'right_wrist':320}, [], 'shared_v1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OnboardRGBD(None, resolutions=value)

    def test_intrinsics_are_actual_camera_data_and_not_scaled_old_estimates(self):
        sensors = {v: SimpleNamespace(image_width=n, image_height=n,
            intrinsic_matrix=np.array([[n*.8, 0, n*.49], [0, n*.9, n*.51], [0, 0, 1.]]))
            for v, n in cameras.resolutions('shared_v1').items()}
        result = cameras.camera_intrinsics(sensors, np.asarray, 'shared_v1')
        self.assertEqual(result['head']['K'], sensors['head'].intrinsic_matrix.tolist())
        self.assertFalse(result['head']['scene_truth'])
        for change in ('size', 'nan', 'focal', 'principal', 'row', 'missing'):
            bad = deepcopy(sensors)
            if change == 'size': bad['head'].image_width = 720
            elif change == 'nan': bad['head'].intrinsic_matrix[0, 0] = np.nan
            elif change == 'focal': bad['head'].intrinsic_matrix[1, 1] = -1
            elif change == 'principal': bad['head'].intrinsic_matrix[0, 2] = 600
            elif change == 'row': bad['head'].intrinsic_matrix[2, 2] = 0
            else: del bad['head']
            with self.subTest(change=change), self.assertRaises(ValueError):
                cameras.camera_intrinsics(bad, np.asarray, 'shared_v1')


if __name__ == '__main__': unittest.main()
