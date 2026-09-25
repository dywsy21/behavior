from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import shared_pathtracing as renderer
import native_full_profile as native
import launch_h70 as diagnostic
from test_scene_pathtracing import fake_route, Settings


class ObservableSettings(Settings):
    def __init__(self):
        super().__init__()
        self.values.update(renderer.SETTINGS)
        self.callbacks = {}
        self.serial = 0
        self.writes = 0

    def subscribe_to_node_change_events(self, key, callback):
        self.serial += 1
        self.callbacks[self.serial] = (key, callback)
        return self.serial

    def unsubscribe_to_change_events(self, subscription):
        del self.callbacks[subscription]

    def set(self, key, value):
        self.writes += 1
        super().set(key, value)
        for path, callback in list(self.callbacks.values()):
            if path == key: callback(None, 'CHANGED')


class NativeRendererDiagnosticsTests(unittest.TestCase):
    def test_failure_values_are_recorded_before_assert_not_only_previous_pass(self):
        settings = ObservableSettings(); record = {}
        renderer.read_checked(settings, record)
        settings.set('/rtx/pathtracing/totalSpp', 4)
        with self.assertRaisesRegex(ValueError, 'totalSpp'):
            renderer.read_checked(settings, record)
        self.assertFalse(record['pathtracing_profile_verified'])
        self.assertEqual(record['pathtracing_differences'], {
            '/rtx/pathtracing/totalSpp': {'expected': 16, 'actual': 4, 'actual_type': 'int'}})
        self.assertEqual(record['pathtracing_actual_settings']['/rtx/pathtracing/totalSpp'], 4)

    def test_native_change_subscription_is_read_only_bounded_and_unsubscribed(self):
        settings = ObservableSettings(); record = {}; snapshots = []
        def write(name, data): snapshots.append((name, deepcopy(data)))
        with renderer.observe_changes(settings, record, write) as trace:
            self.assertEqual(settings.writes, 0)
            self.assertEqual(len(settings.callbacks), len(renderer.SETTINGS))
            settings.set('/rtx/pathtracing/totalSpp', 4)
            first = trace['events'][0]
            self.assertEqual(first['key'], '/rtx/pathtracing/totalSpp')
            self.assertEqual(first['actual'][first['key']], 4)
            self.assertTrue(any('test_native_change_subscription' in s for s in first['python_stack']))
            self.assertEqual(snapshots[-1][1]['events'][0], first)
            for i in range(64): settings.set('/rtx/pathtracing/totalSpp', i)
            self.assertEqual(len(trace['events']), 64)
            self.assertEqual(trace['dropped'], 1)
            self.assertEqual(settings.writes, 65)  # All writes are this test's external drift.
        self.assertFalse(settings.callbacks)
        self.assertEqual(snapshots[-1][1]['dropped'], 1)

    def test_trace_write_failure_is_retained_for_gate_failure(self):
        settings = ObservableSettings(); write = Mock(); record = {}
        with renderer.observe_changes(settings, record, write) as trace:
            write.side_effect = OSError('full')
            settings.set('/rtx/pathtracing/totalSpp', 4)
            self.assertIn('full', trace['errors'][0])
            write.side_effect = None
            record['pathtracing'] = {'launch_calls': 1, 'applied_before_scene': True}
            settings.values.update(renderer.SETTINGS)
            from types import SimpleNamespace
            og = SimpleNamespace(sim=SimpleNamespace(viewer_camera=None))
            with patch.object(renderer, 'get_settings', return_value=settings):
                with self.assertRaisesRegex(ValueError, 'trace incomplete'):
                    renderer.validate_after_scene(og, record)

    def test_partial_subscription_failure_removes_previous_subscriptions(self):
        settings = ObservableSettings()
        original = settings.subscribe_to_node_change_events
        def fail(key, cb):
            if settings.serial == 3: raise RuntimeError('subscription failed')
            return original(key, cb)
        settings.subscribe_to_node_change_events = fail
        with self.assertRaisesRegex(RuntimeError, 'subscription failed'):
            with renderer.observe_changes(settings, {}, Mock()): pass
        self.assertFalse(settings.callbacks)

    def test_original_error_written_inside_native_exit_and_not_overwritten(self):
        saved = []; record = {'phase':'loading_scene', 'official_api_resets':1}
        failure = ValueError('actual drift')
        def write(name, data): saved.append((name, deepcopy(data)))
        @contextmanager
        def sdk():
            try: yield
            finally:
                self.assertEqual(saved[0][0], 'native_failure.json')
                self.assertIn('actual drift', saved[0][1]['native_failure']['error'])
                raise SystemExit(0)  # Native shutdown can terminate before outer handlers.
        with self.assertRaises(SystemExit):
            with native.preserve_failure(record, write):
                with sdk():
                    with native.preserve_failure(record, write): raise failure
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0][1]['official_api_resets'], 1)
        self.assertNotIn('SystemExit', saved[0][1]['native_failure']['error'])

    def test_secondary_storage_error_never_replaces_original_exception(self):
        error = ValueError('original')
        with patch('sys.stderr'), self.assertRaises(ValueError) as caught:
            with native.preserve_failure({}, Mock(side_effect=OSError('full'))): raise error
        self.assertIs(caught.exception, error)

    def test_action_failure_is_not_relabelled_as_native_initialization_failure(self):
        record = {}; write = Mock(); entered = object(); exited = []
        @contextmanager
        def sdk():
            try: yield entered
            finally: exited.append(True)
        validate = Mock()
        with self.assertRaisesRegex(RuntimeError, 'action'):
            with native.initialized_session(sdk(), validate, record, write) as env:
                self.assertIs(env, entered)
                raise RuntimeError('action')
        validate.assert_called_once_with(entered, 1)
        write.assert_not_called(); self.assertNotIn('native_failure', record)
        self.assertEqual(exited, [True])

    def test_second_reset_drift_is_saved_before_native_exit(self):
        record = {'official_api_resets':1}; snapshots = []
        environment = Mock()
        @contextmanager
        def sdk():
            try: yield environment
            finally:
                self.assertEqual(snapshots[0]['official_api_resets'], 2)
                self.assertIn('reset PT changed', snapshots[0]['native_failure']['error'])
        def validate(env, count):
            if count == 2:
                record['official_api_resets'] = 2
                raise ValueError('reset PT changed')
        def write(name, data): snapshots.append(deepcopy(data))
        with self.assertRaisesRegex(ValueError, 'reset PT changed'):
            with native.initialized_session(sdk(), validate, record, write) as env:
                native.checked_reset(env, validate, record, write)
        environment.reset.assert_called_once()

    def test_initial_validation_drift_is_saved_before_sdk_exit(self):
        saved = []; record = {}
        @contextmanager
        def sdk():
            try: yield object()
            finally: self.assertIn('initial drift', saved[0]['native_failure']['error'])
        with self.assertRaisesRegex(ValueError, 'initial drift'):
            with native.initialized_session(sdk(), Mock(side_effect=ValueError('initial drift')),
                    record, lambda n,d: saved.append(deepcopy(d))):
                self.fail('must fail before caller')

    def test_late_action_phase_drift_drop_or_record_error_rejects_success_before_shutdown(self):
        from types import SimpleNamespace
        for fault in ('drift', 'recovered_drift', 'dropped', 'write_error'):
            with self.subTest(fault=fault):
                settings = ObservableSettings(); record = {'pathtracing': {
                    'launch_calls':1, 'applied_before_scene':True}}
                writes = []
                og = SimpleNamespace(sim=SimpleNamespace(viewer_camera=None))
                @contextmanager
                def sdk():
                    try: yield object()
                    finally: self.assertTrue(any(n == 'native_failure.json' for n,_ in writes))
                def write(n, d): writes.append((n, deepcopy(d)))
                def validate(env, resets): renderer.validate_after_scene(og, record)
                with patch.object(renderer, 'get_settings', return_value=settings), \
                        renderer.observe_changes(settings, record, write) as trace:
                    with self.assertRaises(ValueError):
                        with native.initialized_session(sdk(), validate, record, write):
                            validate(None, 2)  # Last reset validation has already passed.
                            if fault == 'dropped':
                                for i in range(65): settings.set('/rtx/pathtracing/spp', 4)
                            elif fault == 'write_error': trace['errors'].append('storage error')
                            else:
                                settings.set('/rtx/pathtracing/spp', 8)
                                if fault == 'recovered_drift': settings.set('/rtx/pathtracing/spp', 4)
                    self.assertFalse(record['pathtracing_profile_verified'])

    def test_original_empty_scene_boundary_owns_subscription_lifecycle(self):
        with fake_route() as (og, sim, _settings, record, write, digest):
            settings = ObservableSettings(); settings.values['/rtx/rendermode'] = 'RealTimePathTracing'
            with patch.object(renderer, 'get_settings', return_value=settings):
                with renderer.before_scene(og, sim, source_sha256=digest,
                        record=record, write=write, trace_write=Mock()):
                    og.launch()
                    self.assertEqual(len(settings.callbacks), len(renderer.SETTINGS))
                    settings.set('/rtx/pathtracing/spp', 8)
                    with self.assertRaises(ValueError): renderer.validate_after_scene(og, record)
                self.assertFalse(settings.callbacks)
                self.assertIs(og.launch, sim._launch_simulator)

    def test_diagnostic_ticket_is_distinct_and_caps_total_including_cold_start(self):
        from test_scene_startup_probe import profile
        gate = diagnostic.gate
        with profile(), patch.object(gate, 'ROOT'), patch.object(gate, 'RUNTIME'), patch.object(gate, 'WALL_SECONDS'):
            base = diagnostic.configure()
            self.assertEqual(gate.WALL_SECONDS, 900)
            self.assertEqual(gate.ROOT.name, 'h70_renderer_trace_v1')
            self.assertEqual(base.ENTRYPOINT, Path(diagnostic.__file__).resolve())
            self.assertIn('launch_h70.py', native.DEPENDENCY_FILES)
            args = gate.command(base)
            self.assertEqual(args[args.index('--mode')+1], 'gate')
            self.assertEqual(args[args.index('--prefix')+1], '0')

    def test_supervisor_reports_original_native_error_even_with_exit_zero_no_result(self):
        from test_native_full_gate import resources
        from test_scene_startup_probe import profile
        gate = diagnostic.gate
        with profile(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(gate, 'ROOT', root/'run'), patch.object(gate, 'RUNTIME', root/'runtime'):
                base = gate.configure(); (gate.ROOT/'gate').mkdir(parents=True)
                failure = {'native_failure': {'error':'ValueError(totalSpp)', 'phase':'loading_scene'},
                           'pathtracing_differences': {'totalSpp': {'actual':4, 'expected':16}}}
                (gate.ROOT/'gate/native_failure.json').write_text(json.dumps(failure))
                child = Mock(pid=123, returncode=0); child.poll.return_value = 0
                with patch.object(gate, 'identity', return_value='f'*40), \
                     patch.object(base, 'claim_stage'), patch.object(base, 'snapshot', return_value=resources()), \
                     patch.object(gate, 'stop_owned_group') as stop, \
                     patch.object(gate.os, 'sched_getaffinity', return_value={72,73,74,75}), \
                     patch.object(gate.os, 'sched_setaffinity'), \
                     patch.object(gate.subprocess, 'Popen', return_value=child):
                    with self.assertRaisesRegex(RuntimeError, 'totalSpp'): gate.supervise(base)
                    stop.assert_called_once_with(child)
                receipt = json.loads((gate.ROOT/'supervisor.json').read_text())
                self.assertEqual(receipt['status'], 'failed')
                self.assertEqual(receipt['native_failure'], failure['native_failure'])
                self.assertEqual(receipt['pathtracing_differences'], failure['pathtracing_differences'])

    def test_non_dlss_prerequisite_is_set_before_aa_and_never_touches_physics(self):
        settings = ObservableSettings()
        settings.values.update({'/rtx-transient/post/aa/limitedOps': True, '/rtx/post/aa/op': 3})
        observed = []; original = settings.set
        def set_and_check(key, value):
            if key == '/rtx/post/aa/op':
                self.assertIs(settings.get('/rtx-transient/post/aa/limitedOps'), False)
            observed.append(key); original(key, value)
        settings.set = set_and_check
        renderer.apply_settings(settings)
        self.assertEqual(set(observed), set(renderer.SETTINGS))
        self.assertEqual(settings.get('/physics/unchanged'), 42)
        self.assertIs(settings.get('/rtx-transient/post/aa/limitedOps'), False)
        settings.values['/rtx-transient/post/aa/limitedOps'] = True
        with self.assertRaisesRegex(ValueError, 'limitedOps'): renderer.read_checked(settings)

    def test_h71_distinct_ticket_keeps_control_gpu_caps_and_adds_only_aa_prerequisite(self):
        import launch_h71
        from test_scene_startup_probe import profile
        gate = launch_h71.gate
        with profile(), patch.object(gate, 'ROOT'), patch.object(gate, 'RUNTIME'), patch.object(gate, 'WALL_SECONDS'):
            base = launch_h71.configure()
            self.assertEqual(gate.ROOT.name, 'h71_aa_prerequisite_v1')
            self.assertEqual(gate.WALL_SECONDS, 1200)
            self.assertEqual(base.ENTRYPOINT, Path(launch_h71.__file__).resolve())
            self.assertIn('launch_h71.py', native.DEPENDENCY_FILES)
            args = gate.command(base)
            self.assertEqual(args[args.index('--max-controls')+1], '1536')
            self.assertEqual(args[args.index('--max-decisions')+1], '24')
            self.assertIn('--/rtx-transient/post/aa/limitedOps=false', base.app_configuration()['extra_args'])
            for path, digest in renderer.INSTALLED_DEPENDENCIES.items():
                self.assertEqual(base.DEPENDENCIES[path], digest)

    def test_installed_aa_implementation_missing_or_changed_rejects_before_gpu_import(self):
        import launch_h71
        from test_scene_startup_probe import profile
        gate = launch_h71.gate
        with profile(), patch.object(gate, 'ROOT'), patch.object(gate, 'RUNTIME'), patch.object(gate, 'WALL_SECONDS'):
            base = launch_h71.configure()
            dependencies = renderer.INSTALLED_DEPENDENCIES.copy()
            self.assertTrue(all(base.DEPENDENCIES[p] == d for p,d in dependencies.items()))
            with patch.object(base, 'DEPENDENCIES', dependencies), \
                 patch.object(base, 'PYTHON', Path(sys.executable)), \
                 patch.object(base.importlib.metadata, 'version', return_value='5.1.0.0'), \
                 patch.object(base.subprocess, 'check_output', return_value=''), \
                 patch.dict(base.os.environ, {'CUDA_VISIBLE_DEVICES':''}):
                for exists in (False, True):
                    with self.subTest(exists=exists), patch.object(Path, 'is_file', return_value=exists), \
                         patch.object(Path, 'read_bytes', return_value=b'wrong installation'):
                        with self.assertRaisesRegex(ValueError, 'dependency changed'): base.identity()


if __name__ == '__main__': unittest.main()
