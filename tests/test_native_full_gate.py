from copy import deepcopy
import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import launch_h69 as launch
import native_full_profile as native
from test_scene_startup_probe import profile
from test_preconfigured_cameras import config_fixture


def resources(pid=None):
    return {uuid: {'used_mib': 0, 'free_mib': 81152, 'processes': [] if pid is None else
            [{'pid': pid, 'type': 'G', 'used_mib': 400}]}
            for uuid in launch.scene.supervisor.GPU_UUIDS}


def result_fixture():
    return {'status': 'complete', 'task': 0, 'prefix_controls': 0, 'diagnostic_replay_controls': 0,
            'gate_ok': True, 'gate_failures': [], 'native_profile': native.NAME,
            'implementation_digest': 'digest', 'finger_kinematics': True, 'press_cycle_v1': True,
            'press_finger_asset_sha256': launch.ASSET_SHA, 'decisions': [{} for _ in range(24)], 'controls': 400,
            'model_calls': 0, 'contact_geometry': False, 'full_task_success_rate_claim': False,
            **{flag.replace('-', '_'): True for flag in launch.FLAGS if flag != 'structured-planning'}}


class NativeFullGateTests(unittest.TestCase):
    def test_generic_failure_is_preserved_even_when_native_shutdown_exits_zero(self):
        for exit_code in (0,1):
            with self.subTest(exit_code=exit_code),profile(),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                with patch.object(launch,'ROOT',root/'run'),patch.object(launch,'RUNTIME',root/'cache'):
                    base=launch.configure();(launch.ROOT/'gate').mkdir(parents=True)
                    primary={'error':'RuntimeError(USD_PRIMARY)','phase':'INITIAL_OBSERVATION'}
                    (launch.ROOT/'gate/failure.json').write_text(json.dumps(primary))
                    child=Mock(pid=123,returncode=exit_code);child.poll.return_value=exit_code
                    with patch.object(launch,'identity',return_value='f'*40), \
                         patch.object(base,'claim_stage'),patch.object(base,'snapshot',return_value=resources()), \
                         patch.object(launch,'stop_owned_group'), \
                         patch.object(launch.os,'sched_getaffinity',return_value={72,73,74,75}), \
                         patch.object(launch.os,'sched_setaffinity'), \
                         patch.object(launch.subprocess,'Popen',return_value=child):
                        with self.assertRaisesRegex(RuntimeError,'USD_PRIMARY'):launch.supervise(base)
                    saved=json.loads((launch.ROOT/'supervisor.json').read_text())
                    self.assertEqual(saved['primary_failure_file'],'failure.json')
                    self.assertEqual(saved['worker_failures']['failure.json'],primary)
                    self.assertEqual(saved['exit_code'],exit_code)

    def test_failure_reports_keep_both_files_and_fail_closed_on_malformed_receipts(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(launch,'ROOT',Path(folder)):
            root=Path(folder)/'gate';root.mkdir();receipt={}
            (root/'failure.json').write_text(json.dumps({'error':'ACTION_PRIMARY'}))
            (root/'native_failure.json').write_text(json.dumps({'native_failure':{'error':'NATIVE_SECONDARY'}}))
            with self.assertRaisesRegex(RuntimeError,'ACTION_PRIMARY'):launch.raise_worker_failure(receipt)
            self.assertEqual(len(receipt['worker_failures']),2)
            (root/'native_failure.json').write_text('{bad')
            with self.assertRaisesRegex(RuntimeError,'ACTION_PRIMARY'):launch.raise_worker_failure(receipt)
            self.assertIn('receipt_error',receipt['worker_failures']['native_failure.json'])
            (root/'failure.json').write_text('[]')
            with self.assertRaisesRegex(RuntimeError,'receipt_error'):launch.raise_worker_failure(receipt)

    def test_same_named_adapter_helper_cannot_replace_digest_bound_source(self):
        shadow = ModuleType('shared_pathtracing')
        shadow.__file__ = '/adapter/shared_pathtracing.py'
        shadow.__spec__ = native.importlib.util.spec_from_file_location('shared_pathtracing', shadow.__file__)
        with patch.dict(sys.modules, {'shared_pathtracing':shadow}), self.assertRaisesRegex(ValueError, 'shadowed'):
            native.checked_helpers()
        resolved = native.checked_helpers()
        self.assertEqual(Path(resolved['shared_pathtracing'].__file__).parent.resolve(), Path(native.__file__).parent.resolve())

    def test_manifest_expected_args_equal_actual_runner_parser(self):
        source = Path(native.__file__).parent/'run_v2.py'
        main = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        end = next(i for i,n in enumerate(main.body) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == 'parse_args')
        code = compile(ast.fix_missing_locations(ast.Module(body=main.body[:end+1], type_ignores=[])), str(source), 'exec')
        with profile():
            base = launch.configure(); ns = {'argparse':argparse}
            with patch.object(sys, 'argv', ['run_v2.py', *launch.command(base)[2:]]): exec(code, ns)
            self.assertEqual(vars(ns['args']), launch.expected_args(base))

    def test_import_is_cpu_only_and_does_not_reconfigure_profile(self):
        code = '''import sys
import probe_simulator_startup as base
before=(base.OUTPUT, base.RUNTIME, base.RUNTIME_SETTINGS.copy())
import native_full_profile, launch_h69
assert before==(base.OUTPUT, base.RUNTIME, base.RUNTIME_SETTINGS)
assert not any(k in sys.modules for k in ('torch','isaacsim','omnigibson','carb'))
'''
        subprocess.run([sys.executable, '-c', code], check=True, cwd=Path(native.__file__).parent)

    def test_command_is_bounded_original_start_gate_not_actor(self):
        with profile():
            base = launch.configure(); argv = launch.command(base)
            for flag, expected in {'--mode': 'gate', '--task': '0', '--prefix': '0', '--gpu': '3',
                    '--max-decisions': '24', '--max-controls': '1536', '--max-seconds': '1200',
                    '--native-profile': 'a100_full_v1', '--odometry-substep-controls': '6'}.items():
                self.assertEqual(argv[argv.index(flag)+1], expected)
            self.assertNotIn('--replay-prefix-spec', argv); self.assertNotIn('--uri', argv)
            self.assertTrue(launch.scene.PATH_TRACING)
            self.assertTrue(launch.scene.PRECONFIGURE_CAMERAS)
            self.assertEqual(launch.scene.CAMERA_RESOLUTION_PROFILE, 'full_v1')
            self.assertEqual(base.RUNTIME_SETTINGS['/renderer/activeGpu'], 3)

    def test_unknown_process_even_graphics_and_auxiliary_caps_reject(self):
        rows = resources(); launch.check_resources(rows)
        own = resources(42); launch.check_resources(own, rows, 42)
        for index in range(4):
            bad = deepcopy(own); key = list(bad)[index]
            bad[key]['processes'].append({'pid': 99, 'used_mib': 1, 'type': 'G'})
            if index >= 2:
                with self.assertRaises(RuntimeError): launch.check_resources(bad, rows, 42)
            else:
                launch.check_resources(bad, rows, 42)
            bad = deepcopy(own); bad[key]['used_mib'] = 24577 if index == 3 else 513
            bad[key]['processes'][0]['used_mib'] = bad[key]['used_mib']
            with self.assertRaises(RuntimeError): launch.check_resources(bad, rows, 42)
        with self.assertRaises(RuntimeError): launch.check_resources(own)
        bad = resources(); bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError): launch.check_resources(bad)

    def test_gpu_headroom_does_not_use_shared_training_pid_assumptions(self):
        rows = resources(); rows[list(rows)[3]]['free_mib'] = 32767
        with self.assertRaises(RuntimeError): launch.check_resources(rows)
        launch.check_resources(rows, resources(), 42)
        rows[list(rows)[3]]['free_mib'] = 8191
        with self.assertRaises(RuntimeError): launch.check_resources(rows, resources(), 42)

    def test_existing_teammate_on_gpu_zero_is_preserved_not_counted_as_our_memory(self):
        baseline = resources(); gpu0 = next(iter(baseline))
        baseline[gpu0].update(used_mib=12471, free_mib=68681,
            processes=[{'pid':3561374, 'type':'C', 'used_mib':12462}])
        launch.check_resources(baseline)
        current = deepcopy(baseline)
        current[gpu0]['used_mib'] += 400; current[gpu0]['free_mib'] -= 400
        current[gpu0]['processes'].append({'pid':42, 'type':'G', 'used_mib':400})
        launch.check_resources(current, baseline, 42)
        current[gpu0]['processes'][-1]['used_mib'] = 513
        with self.assertRaises(RuntimeError): launch.check_resources(current, baseline, 42)
        current[gpu0]['processes'][-1]['used_mib'] = 400
        current[gpu0]['processes'][0] = {'pid':3563718, 'type':'C', 'used_mib':30000}
        current[gpu0].update(used_mib=30400, free_mib=50752)
        launch.check_resources(current, baseline, 42)
        current[gpu0]['free_mib'] = 8191
        with self.assertRaises(RuntimeError): launch.check_resources(current, baseline, 42)

    def test_owned_gpu_helper_is_counted_in_aggregate_and_must_exit(self):
        baseline = resources(); current = resources(); gpu0 = next(iter(current))
        current[gpu0]['processes'] = [{'pid':42, 'type':'G', 'used_mib':300},
                                     {'pid':99, 'type':'C', 'used_mib':300}]
        with patch.object(launch.os, 'getsid', return_value=42):
            with self.assertRaisesRegex(RuntimeError, 'memory cap'): launch.check_resources(current, baseline, 42)
            current[gpu0]['processes'][0]['used_mib'] = 100
            launch.check_resources(current, baseline, 42)
            with self.assertRaisesRegex(RuntimeError, 'not been released'):
                launch.check_resources(current, baseline, 42, require_released=True)

    def test_exit_zero_not_enough_for_gate_pass(self):
        result = result_fixture(); launch.validate_result(result, 'digest')
        for key, value in [('gate_ok', False), ('gate_ok', 1), ('task', 3), ('prefix_controls', 1),
                ('diagnostic_replay_controls', 1), ('native_profile', 'original'),
                ('implementation_digest', 'old'), ('gate_failures', ['failure']),
                ('press_finger_asset_sha256', 'other'), ('decisions', [{}]*23), ('controls', 1537)]:
            bad = deepcopy(result); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): launch.validate_result(bad, 'digest')

    def test_launcher_reserves_once_and_records_exact_budget(self):
        with profile(), tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(launch, 'ROOT', root/'run'), patch.object(launch, 'RUNTIME', root/'cache'):
                base = launch.configure()
                with patch.object(launch, 'identity', return_value='f'*40), \
                     patch.object(base, 'snapshot', return_value=resources()), \
                     patch.object(launch.subprocess, 'Popen', return_value=Mock(pid=123)) as popen, patch('builtins.print'):
                    launch.launch(base)
                    data = json.loads((root/'run/launch.json').read_text())
                    self.assertEqual(data['budget']['primary_mib'], 24576)
                    self.assertEqual(data['budget']['wall_seconds'], 2400)
                    self.assertEqual(data['budget']['model_calls'], 0)
                    self.assertEqual(data['supervisor_pid'], 123)
                    self.assertNotIn('CUDA_VISIBLE_DEVICES', popen.call_args.kwargs['env'])
                    with self.assertRaises(FileExistsError): launch.launch(base)
                    self.assertEqual(popen.call_count, 1)

    def test_wall_watchdog_stops_only_owned_child(self):
        with profile(), tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(launch, 'ROOT', root/'run'), patch.object(launch, 'RUNTIME', root/'cache'):
                base = launch.configure(); launch.ROOT.mkdir()
                child = Mock(pid=123, returncode=-15); child.poll.return_value = None
                with patch.object(launch, 'identity', return_value='f'*40), \
                     patch.object(base, 'claim_stage'), patch.object(base, 'snapshot', return_value=resources()), \
                     patch.object(launch, 'stop_owned_group') as stop, \
                     patch.object(launch.os, 'sched_getaffinity', return_value={72,73,74,75}), \
                     patch.object(launch.os, 'sched_setaffinity'), \
                     patch.object(launch.subprocess, 'Popen', return_value=child), \
                     patch.object(launch.time, 'monotonic', side_effect=[0, 2401, 2402, 2403, 2404]):
                    with self.assertRaises(TimeoutError): launch.supervise(base)
                    stop.assert_called_once_with(child)
                data = json.loads((launch.ROOT/'supervisor.json').read_text())
                self.assertEqual(data['status'], 'failed'); self.assertEqual(data['worker_pid'], 123)
                self.assertEqual(data['exit_code'], -15)

    def test_worker_exiting_zero_after_wall_deadline_is_still_failure(self):
        with profile(), tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(launch, 'ROOT', root/'run'), patch.object(launch, 'RUNTIME', root/'cache'):
                base = launch.configure(); launch.ROOT.mkdir()
                child = Mock(pid=123, returncode=0); child.poll.return_value = 0
                with patch.object(launch, 'identity', return_value='f'*40), \
                     patch.object(base, 'claim_stage'), patch.object(base, 'snapshot', return_value=resources()), \
                     patch.object(launch, 'stop_owned_group') as stop, \
                     patch.object(launch.os, 'sched_getaffinity', return_value={72,73,74,75}), \
                     patch.object(launch.os, 'sched_setaffinity'), \
                     patch.object(launch.subprocess, 'Popen', return_value=child), \
                     patch.object(launch.time, 'monotonic', side_effect=[0, 2401, 2402, 2403, 2404]):
                    with self.assertRaisesRegex(TimeoutError, 'exited after'): launch.supervise(base)
                    stop.assert_called_once_with(child)
                self.assertEqual(json.loads((launch.ROOT/'supervisor.json').read_text())['status'], 'failed')

    def test_cleanup_targets_owned_group_even_after_its_leader_exits(self):
        child = Mock(pid=4321); child.poll.return_value = 0
        with patch.object(launch, 'owned_groups', return_value={4321}), \
             patch.object(launch.os, 'killpg', side_effect=[None, ProcessLookupError]) as kill:
            launch.stop_owned_group(child)
        self.assertEqual(kill.call_args_list[0].args, (4321, launch.signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (4321, 0))
        child.wait.assert_called_once()
        child.poll.return_value = None
        with patch.object(launch.os, 'getpgid', return_value=999), patch.object(launch.os, 'killpg') as kill:
            with self.assertRaises(RuntimeError): launch.stop_owned_group(child)
            kill.assert_not_called()
        with patch.object(launch.os, 'getpgid', side_effect=ProcessLookupError), \
             patch.object(launch, 'owned_groups', return_value={4321}), \
             patch.object(launch.os, 'killpg', side_effect=[None, ProcessLookupError]) as kill:
            launch.stop_owned_group(child)
            self.assertEqual(kill.call_args_list[0].args, (4321, launch.signal.SIGTERM))

    def test_cleanup_also_stops_owned_session_helpers_in_another_group(self):
        child = Mock(pid=4321); child.poll.return_value = 0
        events = []
        def kill(group, sig):
            events.append((group, sig))
            if sig == 0: raise ProcessLookupError
        with patch.object(launch, 'owned_groups', return_value={4321, 4322}), patch.object(launch.os, 'killpg', side_effect=kill):
            launch.stop_owned_group(child)
        self.assertEqual({group for group,sig in events if sig == launch.signal.SIGTERM}, {4321,4322})

    def test_cleanup_empty_session_never_signals_an_old_numeric_group_id(self):
        child = Mock(pid=4321); child.poll.return_value = 0
        with patch.object(launch, 'owned_groups', return_value=set()), patch.object(launch.os, 'killpg') as kill:
            launch.stop_owned_group(child)
        kill.assert_not_called(); child.wait.assert_called_once()

    def test_all_manifest_arguments_source_instance_and_zero_model_are_bound(self):
        with profile():
            base = launch.configure()
            manifest = {'code_commit': 'commit', 'implementation_digest': 'digest', 'instance': 138,
                'task': 0, 'task_name': 'turning_on_radio', 'split': 'train', 'seed': 0,
                'training_updates': 0, 'model_identity': None, 'actor_scene_truth': False,
                'prefix_is_expert_not_agent': False, 'diagnostic_replay_requested': False,
                'native_profile': 'a100_full_v1', 'args': launch.expected_args(base)}
            launch.validate_manifest(manifest, base, 'commit', 'digest')
            for key, value in [('code_commit','old'),('instance',139),('seed',True),('training_updates',1),
                               ('model_identity',{}),('actor_scene_truth',True)]:
                bad = deepcopy(manifest); bad[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    launch.validate_manifest(bad, base, 'commit', 'digest')
            for flag in launch.FLAGS:
                bad = deepcopy(manifest); bad['args'][flag.replace('-','_')] = False
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    launch.validate_manifest(bad, base, 'commit', 'digest')
                if flag != 'structured-planning':
                    result = result_fixture(); result[flag.replace('-','_')] = False
                    with self.assertRaises(ValueError): launch.validate_result(result, 'digest')
            result = result_fixture(); result['model_calls'] = 1
            with self.assertRaises(ValueError): launch.validate_result(result, 'digest')

    def test_camera_wrapper_copies_config_preserves_physics_and_constructs_once(self):
        @dataclass
        class Imports:
            Evaluator: object
            OmegaConf: object
        original = Mock(return_value=SimpleNamespace(env='env'))
        omega = SimpleNamespace(to_container=lambda c, **kw: deepcopy(c), create=lambda c: c)
        imports = Imports(original, omega); record = {}
        cfg = config_fixture(); untouched = deepcopy(cfg)
        with profile(), patch('shared_camera_config.validate_wrapper', return_value={'checked': True}), \
             patch.object(launch.scene.supervisor, 'write'):
            launch.configure()
            modified = native.prepare_imports(imports, record, instance_id=138)
            modified.Evaluator(cfg)
            with self.assertRaises(ValueError): modified.Evaluator(cfg)
        self.assertEqual(cfg, untouched); self.assertIs(imports.Evaluator, original)
        passed = original.call_args.args[0]
        self.assertEqual(passed['robot']['controller_config'], cfg['robot']['controller_config'])
        self.assertEqual(record['camera_configuration']['resolution_profile'], 'full_v1')
        self.assertEqual(original.call_count, 1)

    def test_runner_binds_profile_to_manifest_gate_and_actual_session(self):
        source = (Path(native.__file__).parent/'run_v2.py').read_text()
        self.assertIn('g.get("native_profile", "original")==args.native_profile', source)
        self.assertGreaterEqual(source.count('"native_profile":args.native_profile'), 2)
        self.assertIn('with session_factory(window,gpu=args.gpu)', source)
        self.assertIn('native_full_profile.session(official_factory', source)
        self.assertIn('from native_full_profile import DEPENDENCY_FILES', source)
        for name in native.DEPENDENCY_FILES:
            self.assertTrue((Path(native.__file__).parent/name).is_file())


if __name__ == '__main__': unittest.main()
