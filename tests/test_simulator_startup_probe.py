import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/semantic_robot'))
import probe_simulator_startup as probe


def fixture():
    return {u: {'used_mib': 73664, 'free_mib': 7489,
                'processes': [{'pid': p, 'type': 'C', 'used_mib': 73644}]}
            for u, p in probe.TRAINING.items()}


def xml_fixture(rows):
    body = []
    for u, row in rows.items():
        processes = ''.join('<process_info><pid>%s</pid><type>%s</type><used_memory>%s MiB</used_memory></process_info>'
            % (p['pid'], p['type'], p['used_mib']) for p in row['processes'])
        body.append('<gpu><uuid>%s</uuid><fb_memory_usage><used>%s MiB</used><free>%s MiB</free></fb_memory_usage><processes>%s</processes></gpu>'
                    % (u, row['used_mib'], row['free_mib'], processes))
    return '<nvidia_smi_log>' + ''.join(body) + '</nvidia_smi_log>'


class SimulatorProbeTests(unittest.TestCase):
    def test_xml_keeps_graphics_processes_and_exact_memory(self):
        rows = fixture()
        rows[probe.MAIN_GPU]['processes'].append({'pid': 1234, 'type': 'G', 'used_mib': 100})
        self.assertEqual(probe.parse_snapshot(xml_fixture(rows)), rows)
        with self.assertRaises(RuntimeError): probe.check_resources(rows)
        probe.check_resources(rows, fixture(), own_pid=1234)

    def test_xml_unknown_gpu_missing_units_and_invalid_pids_fail_closed(self):
        xml = xml_fixture(fixture())
        for bad in (xml.replace(probe.MAIN_GPU, 'unknown'), xml.replace('7489 MiB', 'N/A', 1),
                    xml.replace('<pid>3294349</pid>', '<pid>-1</pid>'), '<nvidia_smi_log/>'):
            with self.subTest(bad=bad[:35]), self.assertRaises(ValueError): probe.parse_snapshot(bad)

    def test_original_training_must_remain_and_no_unknown_process_allowed(self):
        baseline = fixture(); probe.check_resources(baseline)
        for u in probe.GPU_UUIDS:
            changed = fixture(); changed[u]['processes'] = []
            with self.assertRaises(RuntimeError): probe.check_resources(changed, baseline, 1234)
            changed = fixture(); changed[u]['processes'].append({'pid': 9876, 'type': 'G', 'used_mib': 1})
            with self.assertRaises(RuntimeError): probe.check_resources(changed, baseline, 1234)

    def test_preflight_and_runtime_reserves_cover_all_cards(self):
        for u in probe.GPU_UUIDS:
            changed = fixture(); changed[u]['free_mib'] = 7167
            with self.assertRaises(RuntimeError): probe.check_resources(changed)
            changed[u]['free_mib'] = 3071
            with self.assertRaises(RuntimeError): probe.check_resources(changed, fixture(), 1234)

    def test_incremental_and_process_memory_caps_independent_of_each_other(self):
        for u in probe.GPU_UUIDS:
            cap = 4096 if u == probe.MAIN_GPU else 512
            changed = fixture(); changed[u]['used_mib'] += cap
            changed[u]['free_mib'] -= cap
            changed[u]['processes'].append({'pid': 1234, 'type': 'C+G', 'used_mib': cap})
            probe.check_resources(changed, fixture(), 1234)
            more = copy.deepcopy(changed); more[u]['used_mib'] += 1
            with self.assertRaises(RuntimeError): probe.check_resources(more, fixture(), 1234)
            more = copy.deepcopy(changed); more[u]['processes'][-1]['used_mib'] += 1
            with self.assertRaises(RuntimeError): probe.check_resources(more, fixture(), 1234)

    def test_exact_runtime_settings_and_only_float32_precision_tolerance(self):
        probe.validate_settings(probe.RUNTIME_SETTINGS.copy())
        key = '/rtx-transient/resourcemanager/texturestreaming/memoryBudget'
        rounded = probe.RUNTIME_SETTINGS.copy(); rounded[key] = 0.009999999776482582
        probe.validate_settings(rounded)
        for k, v in ((key, 0.02), (key, float('nan')), (key, float('inf')), (key, None),
                     ('/rtx-transient/resourcemanager/enableTextureStreaming', 1),
                     ('/rtx-transient/resourcemanager/texturestreaming/streamingBudgetMB', 0),
                     ('/renderer/activeGpu', 0), ('/physics/cudaDevice', 2),
                     ('/renderer/multiGpu/enabled', True), ('/renderer/multiGpu/autoEnable', True),
                     ('/renderer/multiGpu/maxGpuCount', 2)):
            changed = probe.RUNTIME_SETTINGS.copy(); changed[k] = v
            with self.subTest(key=k, value=v), self.assertRaises(ValueError): probe.validate_settings(changed)

    def test_empty_app_has_explicit_gpu_and_bounded_threads_no_task(self):
        config = probe.app_configuration()
        self.assertEqual(config['active_gpu'], 3)
        self.assertEqual(config['physics_gpu'], 3)
        self.assertFalse(config['multi_gpu'])
        self.assertEqual(config['max_gpu_count'], 1)
        self.assertTrue(config['headless'])
        self.assertEqual(config['limit_cpu_threads'], 4)
        self.assertNotIn('open_usd', config)
        self.assertIn('--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=0.01', config['extra_args'])
        self.assertIn('--/app/extensions/registryEnabled=false', config['extra_args'])
        self.assertIn('--/renderer/multiGpu/autoEnable=false', config['extra_args'])
        self.assertIn('--/log/file=' + str(probe.OUTPUT/'kit.log'), config['extra_args'])
        env, dirs = probe.environment()
        self.assertNotIn('CUDA_VISIBLE_DEVICES', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertTrue(all(p.is_relative_to(probe.RUNTIME) for p in dirs))

    def test_constructor_has_only_private_portable_argv_and_restores_it(self):
        original = ['probe.py', '--worker', '--portable-root', '/unrelated']
        seen = []
        def factory(config, *, experience):
            seen.append(list(sys.argv))
            self.assertEqual(config, probe.app_configuration())
            self.assertEqual(experience, str(probe.EXPERIENCE))
            self.assertNotIn('--portable-root', config['extra_args'])
            return 'app'
        with patch.object(sys, 'argv', original):
            self.assertEqual(probe.construct_app(factory), 'app')
            self.assertIs(sys.argv, original)
            with self.assertRaises(RuntimeError):
                probe.construct_app(Mock(side_effect=RuntimeError('constructor failed')))
            self.assertIs(sys.argv, original)
        self.assertEqual(seen, [['probe.py', '--portable-root', str(probe.RUNTIME / 'portable')]])

    def test_stop_signals_only_the_owned_child(self):
        child = Mock(); child.poll.return_value = None
        probe.stop_owned(child)
        child.terminate.assert_called_once_with(); child.kill.assert_not_called()
        child = Mock(); child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('owned', 15), 0]
        probe.stop_owned(child)
        child.terminate.assert_called_once_with(); child.kill.assert_called_once_with()
        child = Mock(); child.poll.return_value = 0
        probe.stop_owned(child)
        child.terminate.assert_not_called(); child.kill.assert_not_called()

    def test_launch_is_exclusive_and_never_retries_existing_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(probe, 'OUTPUT', root / 'run'), patch.object(probe, 'RUNTIME', root / 'cache'), \
                 patch.object(probe, 'identity', return_value='f' * 40), \
                 patch.object(probe, 'snapshot', return_value=fixture()), \
                 patch.object(probe.subprocess, 'Popen', return_value=Mock(pid=1234)) as popen, patch('builtins.print'):
                probe.launch()
                self.assertEqual(popen.call_args.args[0][-1], '--supervise')
                budget = json.loads((root/'run/launch.json').read_text())['budget']
                self.assertEqual(budget['auxiliary_gpu_mib'], 512)
                self.assertEqual(budget['main_gpu_mib'], 4096)
                self.assertEqual(budget['runtime_free_mib'], 3072)
                with self.assertRaises(FileExistsError): probe.launch()
                self.assertEqual(popen.call_count, 1)

    def test_direct_internal_mode_missing_reservation_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            for mode in ('worker', 'supervisor'):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    probe.claim_stage(mode, 'f' * 40)

    def test_claim_binds_source_private_environment_and_one_time_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(probe, 'OUTPUT', root / 'run'), patch.object(probe, 'RUNTIME', root / 'cache'), \
                 patch.object(probe, 'identity', return_value='f' * 40), \
                 patch.object(probe, 'snapshot', return_value=fixture()), \
                 patch.object(probe.subprocess, 'Popen', return_value=Mock(pid=1234)) as popen, patch('builtins.print'):
                probe.launch()
                env = popen.call_args.kwargs['env']
                with patch.dict(os.environ, env, clear=True):
                    with self.assertRaises(ValueError): probe.claim_stage('supervisor', 'a' * 40)
                    with patch.dict(os.environ, {'XDG_CACHE_HOME': '/unrelated'}):
                        with self.assertRaises(ValueError): probe.claim_stage('supervisor', 'f' * 40)
                    with patch.dict(os.environ, {'H52_LAUNCH_TOKEN': 'b' * 48}):
                        with self.assertRaises(ValueError): probe.claim_stage('supervisor', 'f' * 40)
                    claim = probe.claim_stage('supervisor', 'f' * 40)
                    self.assertEqual(claim['pid'], os.getpid())
                    with self.assertRaises(FileExistsError): probe.claim_stage('supervisor', 'f' * 40)
                    with patch.object(probe.os, 'getppid', return_value=9999):
                        with self.assertRaises(ValueError): probe.claim_stage('worker', 'f' * 40)
                    expected = [str(probe.PYTHON), str(Path(probe.__file__).resolve()), '--supervise']
                    with patch.object(probe.os, 'getppid', return_value=os.getpid()), \
                         patch.object(Path, 'read_bytes', return_value=b'\0'.join(os.fsencode(a) for a in expected)+b'\0'):
                        probe.claim_stage('worker', 'f' * 40)
                        with self.assertRaises(FileExistsError): probe.claim_stage('worker', 'f' * 40)

    def test_supervisor_propagates_child_failure_and_after_exit_resource_failure(self):
        for returncode, bad_after, phase in ((7, False, 'updates_complete'),
                                           (0, True, 'updates_complete'),
                                           (0, False, 'app_ready'),
                                           (0, False, 'updates_complete')):
            with self.subTest(returncode=returncode, bad_after=bad_after, phase=phase), \
                 tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                (output / 'worker.json').write_text(json.dumps({'phase': phase, 'app_updates': 8}))
                after = fixture()
                if bad_after: after[probe.MAIN_GPU]['processes'] = []
                child = Mock(pid=1234, returncode=returncode); child.poll.return_value = returncode
                with patch.object(probe, 'OUTPUT', output), patch.object(probe, 'identity', return_value='f'*40), \
                     patch.object(probe, 'claim_stage'), patch.object(probe, 'snapshot', side_effect=[fixture(), after]), \
                     patch.object(probe.subprocess, 'Popen', return_value=child), \
                     patch.object(probe.signal, 'signal'), patch.object(probe.signal, 'pthread_sigmask'):
                    if returncode or bad_after or phase != 'updates_complete':
                        with self.assertRaises(RuntimeError): probe.supervise()
                        self.assertEqual(json.loads((output/'supervisor.json').read_text())['status'], 'failed')
                    else:
                        probe.supervise()
                        self.assertEqual(json.loads((output/'supervisor.json').read_text())['status'], 'completed')
                child.terminate.assert_not_called()

    def test_signal_after_spawn_cleans_up_the_owned_child(self):
        with tempfile.TemporaryDirectory() as folder:
            child = Mock(pid=1234, returncode=-15); child.poll.return_value = None
            with patch.object(probe, 'OUTPUT', Path(folder)), patch.object(probe, 'identity', return_value='f'*40), \
                 patch.object(probe, 'claim_stage'), patch.object(probe, 'snapshot', return_value=fixture()), \
                 patch.object(probe.subprocess, 'Popen', return_value=child), \
                 patch.object(probe.signal, 'signal') as handler, \
                 patch.object(probe.signal, 'pthread_sigmask', side_effect=[set(), InterruptedError('pending SIGTERM')]):
                with self.assertRaises(InterruptedError): probe.supervise()
                child.terminate.assert_called_once_with()
                self.assertGreaterEqual(handler.call_count, 4)
                self.assertEqual(json.loads((Path(folder)/'supervisor.json').read_text())['status'], 'failed')

    def test_worker_close_failure_overrides_updates_complete(self):
        app = Mock()
        app.is_running.return_value = True
        app.close.side_effect = RuntimeError('close failed')
        isaac = types.ModuleType('isaacsim'); isaac.SimulationApp = Mock()
        carb = types.ModuleType('carb'); carb_settings = types.ModuleType('carb.settings')
        settings = Mock(); settings.get.side_effect = probe.RUNTIME_SETTINGS.__getitem__
        carb_settings.get_settings = Mock(return_value=settings); carb.settings = carb_settings
        with tempfile.TemporaryDirectory() as folder, patch.object(probe, 'OUTPUT', Path(folder)), \
             patch.object(probe, 'identity', return_value='f'*40), patch.object(probe, 'claim_stage'), \
             patch.object(probe, 'snapshot', return_value=fixture()), \
             patch.object(probe.os, 'sched_getaffinity', return_value={72,73,74,75}), \
             patch.object(probe.os, 'sched_setaffinity'), patch.object(probe.signal, 'pthread_sigmask'), \
             patch.dict(sys.modules, {'isaacsim': isaac, 'carb': carb, 'carb.settings': carb_settings}), \
             patch.object(probe.inspect, 'getfile', return_value=str(probe.APP_SOURCE)), \
             patch.object(probe, 'construct_app', return_value=app):
            with self.assertRaisesRegex(RuntimeError, 'close failed'): probe.worker()
            record = json.loads((Path(folder)/'worker.json').read_text())
            self.assertEqual(record['phase'], 'close_failed')
            self.assertEqual(record['app_updates'], 8)


if __name__ == '__main__': unittest.main()
