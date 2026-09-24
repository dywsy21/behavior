from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/semantic_robot'))
import shared_camera_config as cameras
import probe_scene_preconfigured_cameras as variant
import probe_scene_startup as scene
from test_scene_startup_probe import profile


def config_fixture():
    return {'env_wrapper': {'_target_': cameras.ORIGINAL_WRAPPER},
            'mode': 'train', 'task': {'name': 'turning_on_radio'}, 'seed': 0,
            'partial_scene_load': True, 'max_steps': 4000,
            'robot': {'model': 'r1pro', 'name': 'robot_r1', 'obs_modalities': ['proprio', 'rgb'],
                      'eval': {'camera_sensor_names': {
                          'head': 'robot_r1:zed_link:Camera:0',
                          'left_wrist': 'robot_r1:left_realsense_link:Camera:0',
                          'right_wrist': 'robot_r1:right_realsense_link:Camera:0'}},
                      'grasping_mode': 'assisted', 'self_collisions': True,
                      'controller_config': {'base': {'name': 'JointController'}},
                      'reset_joint_pos': [1, 2, 3], 'action_normalize': False,
                      'sensor_config': {'VisionSensor': {
                          'sensor_kwargs': {'image_height': 1080, 'image_width': 1080}}}}}


class Camera:
    """Any accidental live setters, modality edits or renders fail the test."""
    def __init__(self, spec):
        self.name = spec['name']
        self._height, self._width = spec['height'], spec['width']
        self._load_config = {'image_height': self._height, 'image_width': self._width}
        self._modalities = {'rgb', 'depth_linear'}
        self._render_product = object()
    @property
    def image_height(self): return self._height
    @image_height.setter
    def image_height(self, value): raise AssertionError('Destructive live camera height setter')
    @property
    def image_width(self): return self._width
    @image_width.setter
    def image_width(self, value): raise AssertionError('Destructive live camera width setter')
    @property
    def modalities(self): return self._modalities
    def add_modality(self, value): raise AssertionError('Live modality modification')
    def remove_modality(self, value): raise AssertionError('Live modality modification')


class EnvironmentWrapper:
    def __init__(self, env): self.env = env
    def __getattr__(self, name): return getattr(self.env, name)


def environment_fixture(resolution_profile='full_v1'):
    robot = config_fixture()['robot']
    specs = cameras.camera_specs(robot['name'], robot['eval'], resolution_profile)
    live_robot = SimpleNamespace(name=robot['name'], sensors={s['name']: Camera(s) for s in specs.values()})
    env = SimpleNamespace(robots=[live_robot], _eval_robot_config=deepcopy(robot['eval']),
                          load_observation_space=Mock(side_effect=AssertionError('Premature reload')),
                          step=Mock(side_effect=AssertionError('No physics/control calls')))
    return env, specs


@contextmanager
def wrapper_module():
    module = ModuleType('omnigibson.envs.env_wrapper')
    module.EnvironmentWrapper = EnvironmentWrapper
    with patch.dict(sys.modules, {module.__name__: module}):
        yield


class PreconfiguredCameraTests(unittest.TestCase):
    def test_import_is_cpu_only_and_has_no_profile_side_effect(self):
        code = '''import sys
import probe_scene_startup as s
before=(s.PRECONFIGURE_CAMERAS,s.PATH_TRACING,s.supervisor.ENTRYPOINT)
import probe_scene_preconfigured_cameras, shared_camera_config
assert before==(s.PRECONFIGURE_CAMERAS,s.PATH_TRACING,s.supervisor.ENTRYPOINT)
assert not any(x in sys.modules for x in ('torch','omnigibson','isaacsim','carb'))
'''
        subprocess.run([sys.executable, '-c', code], cwd=scene.supervisor.REPO/'scripts/semantic_robot', check=True)

    def test_profile_keeps_original_renderer_and_all_resource_caps(self):
        with profile():
            base = scene.supervisor
            settings, budget = deepcopy(base.RUNTIME_SETTINGS), deepcopy(base.BUDGET_DETAILS)
            variant.configure_profile()
            self.assertTrue(scene.PRECONFIGURE_CAMERAS); self.assertTrue(scene.DISABLE_VIEWER)
            self.assertFalse(scene.PATH_TRACING)
            self.assertEqual(base.PROFILE_SETTINGS, {}); self.assertEqual(base.PROFILE_APP_CONFIG, {})
            self.assertEqual(base.RUNTIME_SETTINGS, settings)
            self.assertEqual(base.WALL_SECONDS, 600); self.assertEqual(base.AUXILIARY_MIB, 512)
            self.assertEqual(base.ENTRYPOINT, Path(variant.__file__).resolve())
            for key, value in budget.items(): self.assertEqual(base.BUDGET_DETAILS[key], value)
            self.assertTrue(base.SUCCESS_FIELDS['preconfigured_cameras_verified'])
            self.assertEqual(base.DEPENDENCIES[scene.OG/'omnigibson/utils/python_utils.py'],
                             '2d6ba4bc43b07a4e266f3b7048de2cd72efcb39f817acadeafb9488577a88869')
            scene.configure_profile()
            self.assertFalse(scene.PRECONFIGURE_CAMERAS)

    def test_transform_is_deep_copied_and_non_camera_config_byte_equivalent(self):
        config = config_fixture(); original = deepcopy(config)
        output, receipt = cameras.prepare_config(config)
        self.assertEqual(config, original)
        self.assertEqual(receipt['configured_camera_pixels'], 979200)
        self.assertEqual(receipt['original_camera_pixels'], 3499200)
        self.assertNotEqual(receipt['original_config_sha256'], receipt['effective_config_sha256'])
        self.assertEqual(output['env_wrapper'], {'_target_': cameras.PRECONFIGURED_WRAPPER})
        for spec in receipt['cameras'].values():
            entry = output['robot']['sensor_config'].pop(spec['config_key'])
            self.assertEqual(entry['modalities'], ['rgb', 'depth_linear'])
            self.assertEqual(entry['sensor_kwargs'], {'image_height': spec['height'], 'image_width': spec['width']})
        output['env_wrapper'] = original['env_wrapper']
        self.assertEqual(output, original)
        output['robot']['reset_joint_pos'][0] = 900
        self.assertEqual(config, original)

    def test_complete_class_kwargs_and_noise_are_copied_not_reset(self):
        config = config_fixture()
        entry = config['robot']['sensor_config']['VisionSensor']
        entry.update(enabled=True, noise_type='example', noise_kwargs={'a': [1, 2]})
        entry['sensor_kwargs'].update(focal_length=22.0, clipping_range=[.1, 10])
        output, receipt = cameras.prepare_config(config)
        for spec in receipt['cameras'].values():
            actual = output['robot']['sensor_config'][spec['config_key']]
            self.assertEqual(actual['noise_type'], entry['noise_type'])
            self.assertEqual(actual['noise_kwargs'], entry['noise_kwargs'])
            self.assertEqual(actual['sensor_kwargs']['clipping_range'], [.1, 10])
            self.assertIsNot(actual['noise_kwargs'], entry['noise_kwargs'])

    def test_changed_original_sensor_or_wrapper_contract_is_rejected(self):
        variants = []
        for value in ({'_target_': 'other'}, {'_target_': cameras.ORIGINAL_WRAPPER, 'extra': 1}):
            config = config_fixture(); config['env_wrapper'] = value; variants.append(config)
        for key, value in (('model', 'r1'), ('obs_modalities', ['all'])):
            config = config_fixture(); config['robot'][key] = value; variants.append(config)
        for key, value in (('image_width', 720), ('image_height', 1080.0)):
            config = config_fixture(); config['robot']['sensor_config']['VisionSensor']['sensor_kwargs'][key] = value
            variants.append(config)
        for key, value in (('enabled', False), ('modalities', ['rgb'])):
            config = config_fixture(); config['robot']['sensor_config']['VisionSensor'][key] = value; variants.append(config)
        config = config_fixture(); config['robot']['sensor_config']['zed_link:Camera:0'] = {}; variants.append(config)
        for config in variants:
            original = deepcopy(config)
            with self.subTest(config=config), self.assertRaises(ValueError): cameras.prepare_config(config)
            self.assertEqual(config, original)

    def test_metadata_roles_unique_and_robot_bound_but_not_hardcoded_links(self):
        robot = config_fixture()['robot']
        for changed in ({'head': 'robot_r1:foo:Camera:0'},
                        {**robot['eval']['camera_sensor_names'], 'head': 'robot_r1:left_realsense_link:Camera:0'},
                        {**robot['eval']['camera_sensor_names'], 'head': 'other:zed_link:Camera:0'},
                        {**robot['eval']['camera_sensor_names'], 'head': 'robot_r1:zed_link:Lidar:0'}):
            with self.assertRaises(ValueError): cameras.camera_specs(robot['name'], {'camera_sensor_names': changed})
        robot['eval']['camera_sensor_names']['head'] = 'robot_r1:renamed_head:Camera:0'
        spec = cameras.camera_specs(robot['name'], robot['eval'])
        self.assertEqual(spec['head']['config_key'], 'renamed_head:Camera:0')

    def test_wrapper_is_read_only_defers_space_reload_and_keeps_render_products(self):
        env, specs = environment_fixture()
        with wrapper_module():
            wrapper = cameras.wrap_preconfigured(env)
            self.assertIs(wrapper.env, env)
            self.assertIs(wrapper._behavior_deferred_space_reload, True)
            rows = cameras.validate_wrapper(wrapper, specs)
            self.assertEqual(rows['head']['initial_height'], 720)
            for _ in range(3): self.assertEqual(cameras.validate_wrapper(wrapper, specs), rows)
        env.load_observation_space.assert_not_called(); env.step.assert_not_called()

    def test_final_size_alone_cannot_hide_1080_initial_allocation(self):
        env, _ = environment_fixture()
        env.robots[0].sensors['robot_r1:zed_link:Camera:0']._load_config['image_height'] = 1080
        with wrapper_module(), self.assertRaisesRegex(ValueError, 'not created'): cameras.wrap_preconfigured(env)

    def test_missing_extra_wrong_size_or_privileged_modality_sensor_is_rejected(self):
        for change in ('missing', 'extra', 'wrong_size', 'modality', 'no_product', 'name'):
            env, specs = environment_fixture(); sensors = env.robots[0].sensors
            head = sensors[specs['head']['name']]
            if change == 'missing': sensors.pop(specs['head']['name'])
            elif change == 'extra': sensors['extra_camera'] = Camera({**specs['head'], 'name': 'extra_camera'})
            elif change == 'wrong_size': head._width = 1080
            elif change == 'modality': head._modalities.add('seg_instance')
            elif change == 'no_product': head._render_product = None
            else: head.name = 'other'
            with self.subTest(change=change), wrapper_module(), self.assertRaises(ValueError): cameras.wrap_preconfigured(env)

    def test_after_reset_rebuild_and_receipt_drift_are_rejected(self):
        for change in ('product', 'sensor', 'receipt', 'flag', 'expected'):
            env, specs = environment_fixture()
            with wrapper_module():
                wrapper = cameras.wrap_preconfigured(env)
                head = env.robots[0].sensors[specs['head']['name']]
                if change == 'product': head._render_product = object()
                elif change == 'sensor': env.robots[0].sensors[specs['head']['name']] = Camera(specs['head'])
                elif change == 'receipt': wrapper._behavior_camera_initialization['head']['initial_height'] = 1080
                elif change == 'flag': wrapper._behavior_deferred_space_reload = False
                else: specs['head']['name'] = 'other'
                with self.subTest(change=change), self.assertRaises(ValueError): cameras.validate_wrapper(wrapper, specs)

    def test_real_config_transform_precedes_constructor_and_original_deferred_reload(self):
        @dataclass(frozen=True)
        class Imports:
            Evaluator: object
            OmegaConf: object
        class OmegaConf:
            @staticmethod
            def to_container(config, *, resolve):
                assert resolve is True
                return deepcopy(config)
            @staticmethod
            def create(config): return deepcopy(config)
        original_config = config_fixture(); events = []; env, specs = environment_fixture()
        def constructor(config):
            self.assertEqual(config['robot']['sensor_config']['zed_link:Camera:0']['sensor_kwargs']['image_height'], 720)
            self.assertEqual(config['env_wrapper']['_target_'], cameras.PRECONFIGURED_WRAPPER)
            events.append('construct')
            wrapper = cameras.wrap_preconfigured(env)
            events.append('handles_initialized')
            # Existing installed chunk runner calls this after original init.
            env.load_observation_space = Mock(side_effect=lambda: events.append('space_reloaded'))
            if wrapper._behavior_deferred_space_reload: wrapper.env.load_observation_space()
            return SimpleNamespace(env=wrapper, reset=Mock(), load_task_instance=Mock())
        with profile(), tempfile.TemporaryDirectory() as folder, patch.object(scene.supervisor, 'OUTPUT', Path(folder)), \
             patch.object(scene, 'PRECONFIGURE_CAMERAS', True), wrapper_module():
            record = {}
            traced = scene.trace_session_imports(Imports(constructor, OmegaConf), record)
            evaluator = traced.Evaluator(original_config)
            evaluator.reset(); evaluator.load_task_instance(138); evaluator.reset()
            self.assertEqual(events, ['construct', 'handles_initialized', 'space_reloaded'])
            self.assertEqual(original_config, config_fixture())
            self.assertEqual(record['camera_configuration']['cameras'], specs)
            self.assertEqual(record['camera_initialization'], cameras.validate_wrapper(evaluator.env, specs))
            self.assertEqual(record['official_api_resets'], 2)
            with self.assertRaises(ValueError): traced.Evaluator(original_config)

    def test_camera_profile_refuses_combined_renderer_change_before_constructor(self):
        @dataclass(frozen=True)
        class Imports: Evaluator: object
        constructor = Mock()
        with profile(), patch.object(scene, 'PRECONFIGURE_CAMERAS', True), patch.object(scene, 'PATH_TRACING', True):
            traced = scene.trace_session_imports(Imports(constructor), {})
            with self.assertRaises(ValueError): traced.Evaluator(config_fixture())
        constructor.assert_not_called()


if __name__ == '__main__': unittest.main()
