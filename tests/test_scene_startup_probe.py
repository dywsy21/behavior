from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/semantic_robot'))
import probe_scene_startup as scene
import probe_simulator_startup as base
from test_simulator_startup_probe import fixture


@contextmanager
def profile():
    keys = ('ENTRYPOINT', 'OUTPUT', 'RUNTIME', 'WALL_SECONDS', 'SUCCESS_FIELDS',
            'BUDGET_DETAILS', 'DEPENDENCIES', 'FIXED_ENV', 'PROFILE_SETTINGS',
            'PROFILE_APP_CONFIG', 'RUNTIME_SETTINGS')
    with patch.multiple(base, **{k: getattr(base, k) for k in keys}), \
         patch.object(scene, 'PATH_TRACING', False), patch.object(scene, 'DISABLE_VIEWER', False):
        scene.configure_profile()
        yield


def capture_fixture():
    images, depths, receipts = {}, {}, {}
    for view, size in (('head', 720), ('left_wrist', 480), ('right_wrist', 480)):
        rgb = np.zeros((3, size, size), dtype=np.uint8); rgb[0] = 150
        images[view + '_rgb'] = rgb
        depths[view] = np.ones((size, size), dtype=np.float32)
        receipts[view] = {'valid_fraction': 1., 'shape': [size,size], 'modalities': ['rgb','depth_linear'],
            'depth_units': 'metres', 'depth_convention': 'distance_to_image_plane',
            'same_sensor_current_render': True, 'render_barrier_updates': 4,
            'control_steps_in_capture': 0, 'snapshot_id': 1,
            'rgb_sha256': hashlib.sha256(rgb.transpose(1,2,0).tobytes()).hexdigest(),
            'depth_sha256': hashlib.sha256(depths[view].tobytes()).hexdigest()}
    return images, depths, receipts


class SceneProbeTests(unittest.TestCase):
    def test_import_does_not_change_empty_profile_or_import_gpu_stack(self):
        script = '''import sys
import probe_simulator_startup as base
before = (base.ENTRYPOINT, base.OUTPUT, base.SUCCESS_FIELDS.copy())
import probe_scene_startup
assert before == (base.ENTRYPOINT, base.OUTPUT, base.SUCCESS_FIELDS)
assert not any(x in sys.modules for x in ('torch', 'isaacsim', 'omnigibson'))
'''
        subprocess.run([sys.executable, '-c', script], cwd=base.REPO / 'scripts/semantic_robot', check=True)

    def test_registered_scene_profile_keeps_original_resource_caps(self):
        with profile():
            self.assertEqual(base.WALL_SECONDS, 600)
            self.assertEqual(base.ENTRYPOINT, Path(scene.__file__).resolve())
            self.assertEqual(base.AUXILIARY_MIB, 512)
            self.assertEqual(base.BUDGET_DETAILS['official_api_resets'], 2)
            self.assertEqual(base.BUDGET_DETAILS['load_frozen_instance_calls'], 1)
            for key in ('robot_controls', 'expert_prefix_controls', 'old_policy_prefix_controls',
                        'model_calls', 'training_steps', 'shared_install_writes'):
                self.assertEqual(base.SUCCESS_FIELDS[key], 0)
            self.assertEqual(base.FIXED_ENV['OMNIGIBSON_GPU_ID'], '3')
            self.assertEqual(base.FIXED_ENV['BEHAVIOR_ACTION_STEPS'], '1')
            for path, digest in scene.EXTRA_DEPENDENCIES.items():
                self.assertEqual(base.DEPENDENCIES[path], digest)

    def test_launch_routes_to_scene_entrypoint_and_records_full_budget(self):
        with profile(), tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(base, 'OUTPUT', root/'run'), patch.object(base, 'RUNTIME', root/'cache'), \
                 patch.object(base, 'identity', return_value='f'*40), patch.object(base, 'snapshot', return_value=fixture()), \
                 patch.object(base.subprocess, 'Popen', return_value=Mock(pid=1234)) as popen, patch('builtins.print'):
                base.launch()
                self.assertEqual(popen.call_args.args[0], [str(base.PYTHON), str(Path(scene.__file__).resolve()), '--supervise'])
                budget = json.loads((root/'run/launch.json').read_text())['budget']
                self.assertEqual(budget['seconds'], 600)
                self.assertEqual(budget['rgbd_captures'], 1)
                self.assertEqual(budget['main_gpu_mib'], 4096)
                self.assertEqual(budget['runtime_free_mib'], 3072)
                with self.assertRaises(FileExistsError): base.launch()
                self.assertEqual(popen.call_count, 1)

    def test_supervisor_requires_all_scene_counts_and_exact_types(self):
        with profile():
            variants = [base.SUCCESS_FIELDS.copy()]
            for key in base.SUCCESS_FIELDS:
                changed = base.SUCCESS_FIELDS.copy(); changed.pop(key); variants.append(changed)
            changed = base.SUCCESS_FIELDS.copy(); changed['rgbd_captures'] = True; variants.append(changed)
            for record in variants:
                with self.subTest(record=record), tempfile.TemporaryDirectory() as folder:
                    output = Path(folder); (output/'worker.json').write_text(json.dumps(record))
                    child = Mock(pid=1234, returncode=0); child.poll.return_value = 0
                    with patch.object(base, 'OUTPUT', output), patch.object(base, 'identity', return_value='f'*40), \
                         patch.object(base, 'claim_stage'), patch.object(base, 'snapshot', return_value=fixture()), \
                         patch.object(base.subprocess, 'Popen', return_value=child) as popen, \
                         patch.object(base.signal, 'signal'), patch.object(base.signal, 'pthread_sigmask'):
                        if record == base.SUCCESS_FIELDS and type(record.get('rgbd_captures')) is int:
                            base.supervise()
                        else:
                            with self.assertRaises(RuntimeError): base.supervise()
                        self.assertEqual(popen.call_args.args[0][1:], [str(Path(scene.__file__).resolve()), '--worker'])

    def test_window_rejects_other_instance_split_seed_embodiment_and_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            robot = Path(folder)/'robot.yaml'; robot.write_bytes(b'robot')
            digest = hashlib.sha256(b'robot').hexdigest()
            good = dict(task_name='turning_on_radio', official_mode='train', instance_id=138, seed=0,
                        robot_config_path=str(robot), robot_config_sha256=digest)
            with patch.object(scene, 'ROBOT_SHA', digest):
                scene.validate_window(SimpleNamespace(**good))
                for key, value in (('task_name','other'), ('official_mode','public_test'), ('instance_id',137),
                                   ('instance_id',True), ('seed',1), ('seed',False), ('robot_config_sha256','0'*64)):
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        scene.validate_window(SimpleNamespace(**{**good, key:value}))
                robot.write_bytes(b'changed')
                with self.assertRaises(ValueError): scene.validate_window(SimpleNamespace(**good))

    def test_capture_rejects_motion_bad_depth_blank_rgb_or_resolution_change(self):
        images, depths, sensors = capture_fixture(); q = np.zeros(31)
        scene.validate_capture(q, q.copy(), images, depths, sensors)
        for changed in (q[:23], np.ones(31)*1e-4, np.ones(31)*np.nan):
            with self.assertRaises(ValueError): scene.validate_capture(q, changed, images, depths, sensors)
        for value in (0., .0999, float('nan'), 1.1):
            with self.assertRaises(ValueError):
                scene.validate_capture(q, q, images, depths, {**sensors, 'head': {'valid_fraction': value}})
        for bad in (np.zeros((3,720,720), dtype=np.uint8), images['head_rgb'].astype(np.float32),
                    images['head_rgb'][:,::2,::2]):
            with self.assertRaises(ValueError): scene.validate_capture(q, q, {**images, 'head_rgb':bad}, depths, sensors)
        with self.assertRaises(ValueError): scene.validate_capture(q, q, images, depths, {'head': sensors['head']})

    def test_depth_and_actual_byte_receipts_are_required_for_each_view(self):
        images, depths, sensors = capture_fixture(); q = np.zeros(31)
        for bad in ({}, {'head':depths['head']}, {**depths,'hidden_segmentation':depths['head']}):
            with self.assertRaises(ValueError): scene.validate_capture(q,q,images,bad,sensors)
        for view in depths:
            for bad in (depths[view][:10], depths[view].astype(np.float64), np.full_like(depths[view],np.nan)):
                with self.subTest(view=view), self.assertRaises(ValueError):
                    scene.validate_capture(q,q,images,{**depths,view:bad},sensors)
            missing=np.full_like(depths[view],np.nan)
            honest={**sensors,view:{**sensors[view],'valid_fraction':0.,
                'depth_sha256':hashlib.sha256(missing.tobytes()).hexdigest()}}
            with self.assertRaises(ValueError): scene.validate_capture(q,q,images,{**depths,view:missing},honest)
            for key,value in (('shape',[2,2]), ('rgb_sha256','0'*64), ('depth_sha256','0'*64),
                              ('same_sensor_current_render',False), ('render_barrier_updates',3),
                              ('control_steps_in_capture',1), ('snapshot_id',True), ('modalities',['rgb'])):
                changed={**sensors,view:{**sensors[view],key:value}}
                with self.subTest(view=view,key=key), self.assertRaises(ValueError):
                    scene.validate_capture(q,q,images,depths,changed)

    def test_session_trace_requires_exact_order_and_completed_not_attempted_calls(self):
        @dataclass(frozen=True)
        class Imports: Evaluator: object
        with tempfile.TemporaryDirectory() as folder, patch.object(base,'OUTPUT',Path(folder)):
            original=Mock(); record={}; constructor=Mock(return_value=original)
            imports=scene.trace_session_imports(Imports(constructor),record)
            evaluator=imports.Evaluator('cfg')
            with self.assertRaises(ValueError): evaluator.load_task_instance(138)
            evaluator.reset()
            with self.assertRaises(ValueError): evaluator.load_task_instance(137)
            evaluator.load_task_instance(138); evaluator.reset()
            self.assertEqual([e['name'] for e in record['official_api_events']], ['reset','load_task_instance','reset'])
            self.assertTrue(all(e['status']=='completed' for e in record['official_api_events']))
            self.assertEqual(record['official_api_resets'],2)
            self.assertEqual(record['load_frozen_instance_calls'],1)
            with self.assertRaises(ValueError): evaluator.reset()
            with self.assertRaises(ValueError): imports.Evaluator('again')
            self.assertEqual(original.reset.call_count,2); original.load_task_instance.assert_called_once_with(138)
            original.reset.side_effect=RuntimeError('failed reset'); failed={}
            other=scene.trace_session_imports(Imports(lambda cfg:original),failed).Evaluator('cfg')
            with self.assertRaises(RuntimeError): other.reset()
            self.assertEqual(failed['official_api_resets'],0)
            self.assertEqual(failed['official_api_events'][0]['status'],'started')
            with self.assertRaises(ValueError): other.load_task_instance(138)

    def test_owned_close_and_query_errors_cannot_leave_a_success_receipt(self):
        for failure_at in ('is_running', 'close'):
            with tempfile.TemporaryDirectory() as folder, patch.object(base, 'OUTPUT', Path(folder)):
                app = Mock(); app.is_running.return_value = True
                getattr(app, failure_at).side_effect = RuntimeError('native shutdown failed')
                record = {'phase': 'scene_observation_complete'}
                with self.assertRaisesRegex(RuntimeError, 'native shutdown failed'): scene.close_owned_app(app, record)
                self.assertEqual(json.loads((Path(folder)/'worker.json').read_text())['phase'], 'close_failed')

    def test_worker_captures_original_reset_without_actor_or_prefix_controls(self):
        self._exercise_worker(False)

    def test_pathtracing_worker_reapplies_native_overrides_and_original_og_mode_before_scene(self):
        self._exercise_worker(True)

    def test_pathtracing_worker_handles_observed_legacy_mode_before_scene(self):
        self._exercise_worker(True,native_render_mode='RaytracedLighting')

    def _exercise_worker(self, use_pathtracing, native_render_mode='RealTimePathTracing'):
        import shared_pathtracing as renderer
        images, depths, sensors = capture_fixture()
        factory = ModuleType('native_oracle_low_v1.official_factory'); factory.__file__ = str(scene.FACTORY)
        factory.DEFAULT_EVAL_ROOT = getattr(scene, 'EVAL_ROOT', Path('/mnt/sdc1/robodojo/behavior_eval'))
        window = SimpleNamespace(frozen_window=Mock(side_effect=AssertionError('No prefix replay')))
        factory.load_official_oracle_window = Mock(return_value=window)
        robot = SimpleNamespace(action_dim=23, get_joint_positions=Mock(return_value=np.zeros(31)))
        env = SimpleNamespace(robots=[robot], step=Mock(side_effect=AssertionError('No actor control')))
        original = SimpleNamespace(env=env, reset=Mock(), load_task_instance=Mock())
        @dataclass(frozen=True)
        class Imports:
            Evaluator: object
            gm: object
        @contextmanager
        def unlocked(): yield
        gm=SimpleNamespace(RENDER_VIEWER_CAMERA=True,unlocked=unlocked)
        imports = Imports(Mock(return_value=original),gm); factory._lazy_official_imports = Mock(return_value=imports)
        @contextmanager
        def original_session(window, *, gpu, imports):
            if use_pathtracing:
                og.launch(physics_dt=.008,device='cuda:3')
                renderer.read_checked(settings)  # The Environment now starts scene loading.
            evaluator=imports.Evaluator('cfg'); evaluator.reset(); evaluator.load_task_instance(138)
            yield SimpleNamespace(reset=evaluator.reset,evaluator=evaluator)
        factory.OfficialEvaluatorSession = Mock(side_effect=original_session)
        package = ModuleType('native_oracle_low_v1'); package.official_factory = factory
        og = ModuleType('omnigibson'); og.__path__ = []; og.app = None
        sim=SimpleNamespace(render=Mock(),scenes=[],viewer_camera=None)
        og.sim = None if use_pathtracing else sim
        source=Path(__file__).resolve() if use_pathtracing else scene.OG_SOURCE
        og_startup = ModuleType('omnigibson.simulator'); og_startup.__file__ = str(source)
        def original_launch(**kwargs):
            settings.set('/rtx/rendermode',native_render_mode)
            og.app=app; og.sim=sim
            return sim
        og.launch=og_startup._launch_simulator=Mock(side_effect=original_launch)
        og.simulator = og_startup
        sensor = SimpleNamespace(sensors={view: SimpleNamespace(image_width=d.shape[1], image_height=d.shape[0])
                                         for view, d in depths.items()})
        def read(model, *, render):
            for _ in range(4): render()
            return images, depths, sensors
        sensor.read = Mock(side_effect=read)
        onboard = ModuleType('semantic_robot.v2.onboard'); onboard.OnboardRGBD = Mock(return_value=sensor)
        backend = ModuleType('semantic_robot.og_backend'); backend.array = np.asarray
        onboard.__file__ = str(base.REPO/'src/semantic_robot/v2/onboard.py')
        backend.__file__ = str(base.REPO/'src/semantic_robot/og_backend.py')
        isaac = ModuleType('isaacsim'); isaac.SimulationApp = Mock()
        carb = ModuleType('carb'); carb.settings = ModuleType('carb.settings')
        values=base.RUNTIME_SETTINGS.copy()
        settings=SimpleNamespace(get=values.get,set=values.__setitem__)
        carb.settings.get_settings = Mock(return_value=settings)
        app = Mock(); app.is_running.return_value = False
        @contextmanager
        def startup(*args, **kwargs):
            kwargs['construct']()
            yield {'launch_calls':1, 'app_constructions':1, 'verified_copy_noops':[{},{}], 'shared_install_writes':0}
        modules = {m.__name__:m for m in (factory, package, og, og_startup, onboard, backend, isaac, carb, carb.settings)}
        with profile(), tempfile.TemporaryDirectory() as folder, patch.object(base, 'OUTPUT', Path(folder)), \
             patch.object(base, 'identity', return_value='f'*40), patch.object(base, 'claim_stage'), \
             patch.object(base, 'snapshot', return_value=fixture()), patch.object(scene, 'validate_window'), \
             patch.object(base.os, 'sched_getaffinity', return_value={72,73,74,75}), patch.object(base.os, 'sched_setaffinity'), \
             patch.object(base.signal, 'pthread_sigmask'), patch.dict(sys.modules, modules), \
             patch.object(scene.inspect, 'getfile', return_value=str(base.APP_SOURCE)), \
             patch.object(base, 'construct_app', return_value=app), patch.object(scene, 'private_og_startup', startup), \
             patch.object(sys, 'path', list(sys.path)), patch.object(scene,'OG_SOURCE',source), \
             patch.object(renderer,'_CONSUMED',set()), \
             patch.object(scene,'EXTRA_DEPENDENCIES',{**scene.EXTRA_DEPENDENCIES,
                 **({source:hashlib.sha256(source.read_bytes()).hexdigest()} if use_pathtracing else {})}):
            if use_pathtracing:
                import probe_scene_pathtracing
                probe_scene_pathtracing.configure_profile()
                base.OUTPUT=Path(folder)
                values.update(renderer.SETTINGS)
                values['/rtx/pathtracing/totalSpp']=4  # Native app reset ignores our CLI16.
            scene.scene_worker()
            record = json.loads((Path(folder)/'worker.json').read_text())
            self.assertEqual({k:record[k] for k in base.SUCCESS_FIELDS}, base.SUCCESS_FIELDS)
            self.assertEqual(len(record['artifacts_sha256']), 6)
            for name, digest in record['artifacts_sha256'].items():
                self.assertEqual(hashlib.sha256((Path(folder)/name).read_bytes()).hexdigest(), digest)
            if use_pathtracing:
                self.assertEqual(record['pathtracing_initial_settings'],renderer.SETTINGS)
                self.assertEqual(record['pathtracing']['original_render_mode'],native_render_mode)
                self.assertEqual(record['pathtracing_actual_settings'],renderer.SETTINGS)
                self.assertTrue(record['pathtracing_profile_verified'])
                og.launch.assert_called_once_with(physics_dt=.008,device='cuda:3')
        self.assertEqual(original.reset.call_count,2); original.load_task_instance.assert_called_once_with(138)
        env.step.assert_not_called()
        window.frozen_window.assert_not_called(); self.assertEqual(og.sim.render.call_count, 4)
        factory.OfficialEvaluatorSession.assert_called_once()
        self.assertEqual(factory.OfficialEvaluatorSession.call_args.args,(window,))
        self.assertEqual(factory.OfficialEvaluatorSession.call_args.kwargs['gpu'],3)


if __name__ == '__main__': unittest.main()
