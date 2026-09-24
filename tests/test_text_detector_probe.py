from copy import deepcopy
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import probe_text_detector as probe


class TextDetectorProbeTests(unittest.TestCase):
    def test_import_is_cpu_only(self):
        code = '''import sys
import probe_text_detector
assert not any(m in sys.modules for m in ('torch','isaacsim','omnigibson'))
'''
        subprocess.run([sys.executable,'-c',code],cwd=probe.REPO/'scripts/semantic_robot',check=True)

    def test_exact_budget_and_public_input_reference(self):
        spec = probe.read_spec()
        self.assertEqual(spec['max_calls'],12); self.assertEqual(spec['device'],'cpu')
        for key,value in (('max_calls',13),('max_seconds',1200),('device','cuda'),('box_threshold',.1),
                          ('views',['head']),('training_steps',1)):
            changed = deepcopy(spec); changed[key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'spec.json'; path.write_text(json.dumps(changed))
                with self.subTest(key=key), patch.object(probe,'SPEC',path), self.assertRaises(ValueError):
                    probe.read_spec()

    def test_query_is_original_public_target_not_a_handwritten_location(self):
        self.assertEqual(probe.query_text(' Handle of the refrigerator '),'handle of the refrigerator.')
        self.assertEqual(probe.query_text('Radio.'),'radio.')
        for value in ('', ' ', None, 1, 'x'*301):
            with self.subTest(value=value), self.assertRaises(ValueError):probe.query_text(value)

    def test_all_nine_model_hashes_and_exact_file_set_are_frozen(self):
        spec = probe.read_spec()
        variants = []
        for filename in spec['model_files']:
            changed = deepcopy(spec); changed['model_files'][filename] = '0'*64
            variants.append(changed)
        missing = deepcopy(spec); del missing['model_files']['config.json']; variants.append(missing)
        extra = deepcopy(spec); extra['model_files']['extra.json'] = '0'*64; variants.append(extra)
        for changed in variants:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'spec.json'; path.write_text(json.dumps(changed))
                with patch.object(probe,'SPEC',path), self.assertRaises(ValueError):probe.read_spec()

    def test_prepare_keeps_all_three_raw_views_and_no_previous_answer(self):
        raw = {view:np.full((4,5,3),i,dtype=np.uint8) for i,view in enumerate(('head','left_wrist','right_wrist'))}
        def load(case):
            return {'raw':raw,'goal':SimpleNamespace(target=case['goal']['target'])}, {'source':'test-public-pixels'}
        with patch.object(probe.source,'load_case',side_effect=load) as loader:
            queries = probe.prepare_inputs(probe.read_spec())
        self.assertEqual(loader.call_count,4); self.assertEqual(len(queries),12)
        self.assertEqual(len({q['id'] for q in queries}),12)
        self.assertEqual([q['view'] for q in queries[:3]],['head','left_wrist','right_wrist'])
        for q in queries:
            np.testing.assert_array_equal(np.asarray(q['image']),raw[q['view']])
            self.assertEqual(set(q),{'id','view','text','image','frame_binding','raw_pixels_sha256'})

    def test_detections_keep_every_prediction_without_clipping_or_best_only(self):
        value = {'scores':np.array([.45,.9]),'boxes':np.array([[-2,1,10,15],[2,4,6,8]]),
                 'text_labels':['radio','radio']}
        rows = probe.detections(value)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['box_xyxy_px'],[-2.,1.,10.,15.])
        self.assertEqual(rows[0]['score'],.45)
        self.assertEqual(probe.detections({'scores':np.empty(0),'boxes':np.empty((0,4)),'text_labels':[]}),[])

    def test_malformed_predictions_fail_instead_of_being_repaired(self):
        good = {'scores':np.array([.5]),'boxes':np.array([[1.,2.,3.,4.]]),'text_labels':['radio']}
        for key,value in (('scores',np.array([np.nan])),('scores',np.array([1.1])),
                          ('boxes',np.array([[3.,2.,1.,4.]])),('boxes',np.zeros((1,3))),
                          ('text_labels',[]),('text_labels',[1])):
            changed = deepcopy(good); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):probe.detections(changed)

    def test_child_environment_hides_gpu_without_mutating_parent(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2','HF_HUB_OFFLINE':'0'}):
            env = probe.cpu_environment()
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(env['HF_HUB_OFFLINE'],'1')
            self.assertEqual(env['PYTHONPATH'],str(probe.REPO/'src'))
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'2')

    def test_launch_is_single_use_with_durable_supervisor_and_600_second_budget(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe,'OUTPUT',Path(folder)/'run'), \
             patch.object(probe,'RUNTIME',Path(folder)/'runtime'), patch.object(probe,'identity',return_value='f'*40), \
             patch.object(probe.shared,'validate_model_files'), patch.object(probe,'prepare_inputs'), \
             patch.object(probe.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)), \
             patch.object(probe.subprocess,'Popen',return_value=Mock(pid=1234)) as popen, patch('builtins.print'):
            probe.launch()
            self.assertEqual(popen.call_args.args[0],
                             [str(probe.PYTHON),str(Path(probe.__file__).resolve()),'--supervisor'])
            self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            receipt = json.loads((probe.OUTPUT/'launch.json').read_text())
            self.assertEqual(receipt['source_commit'],'f'*40)
            self.assertEqual(receipt['model_calls'],12); self.assertEqual(receipt['gpu_allocations'],0)
            self.assertEqual(receipt['seconds'],600); self.assertEqual(receipt['cleanup_seconds'],15)
            with self.assertRaises(FileExistsError):probe.launch()
            self.assertEqual(popen.call_count,1)

    def test_stages_bind_to_reserved_supervisor_and_are_single_use(self):
        commit, token = 'f'*40, 'a'*48
        with tempfile.TemporaryDirectory() as folder, patch.object(probe,'OUTPUT',Path(folder)), \
             patch.dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',H58_LAUNCH_TOKEN=token), \
             patch.object(probe.os,'getpid',return_value=1234), \
             patch.object(probe.os,'getppid',return_value=1000):
            probe.write('launch.json',{'status':'supervisor_started','supervisor_pid':1234,
                'source_commit':commit,'spec_sha256':probe.shared.sha(probe.SPEC),
                'token_sha256':hashlib.sha256(token.encode()).hexdigest()})
            probe.claim_stage('supervisor',commit)
            with self.assertRaises(FileExistsError):probe.claim_stage('supervisor',commit)
            with patch.object(probe.os,'getpid',return_value=2222):
                with self.assertRaises(ValueError):probe.claim_stage('worker',commit)
                with patch.object(probe.os,'getppid',return_value=1234):
                    probe.claim_stage('worker',commit)
                    with self.assertRaises(FileExistsError):probe.claim_stage('worker',commit)
            with patch.dict(os.environ,H58_LAUNCH_TOKEN='b'*48), self.assertRaises(ValueError):
                probe.claim_stage('supervisor',commit)

    def test_supervisor_terminal_receipts_for_completion_failure_and_timeout(self):
        commit = 'f'*40
        original = json.loads((probe.REPO/probe.INPUT_SPEC).read_text())
        ids = [c['id']+'__'+v for c in original['cases'] for v in probe.read_spec()['views']]
        for mode in ('completed','partial','exit_error','missing','timeout_term','timeout_kill','spawn_failure',
                     'cleanup_error','pending_signal'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, \
                 patch.object(probe,'OUTPUT',Path(folder)), \
                 patch.object(probe,'identity',return_value=commit), patch.object(probe,'claim_stage'):
                child = Mock(pid=5432,returncode=0)
                child.poll.side_effect = lambda:child.returncode
                child.wait.return_value = 0
                result = {'status':'completed','pid':5432,'source_commit':commit,
                    'spec_sha256':probe.shared.sha(probe.SPEC),'cuda_initialized':False,
                    'queries':[{'id':id} for id in ids]}
                if mode == 'partial': result['queries'].pop()
                if mode == 'exit_error': child.returncode = 2
                if mode.startswith('timeout') or mode in ('cleanup_error','pending_signal'):
                    child.returncode = None
                    result['status'] = 'running'
                    timeout = subprocess.TimeoutExpired('owned worker',600)
                    if mode in ('timeout_term','pending_signal'):
                        child.wait.side_effect = [timeout,-15]
                        child.terminate.side_effect = lambda:setattr(child,'returncode',-15)
                    elif mode == 'cleanup_error':
                        child.wait.side_effect = timeout
                        child.terminate.side_effect = OSError('owned child terminate failed')
                    else:
                        child.wait.side_effect = [timeout,subprocess.TimeoutExpired('cleanup',7.5),-9]
                        child.kill.side_effect = lambda:setattr(child,'returncode',-9)
                if mode != 'missing':probe.write('result.json',result)
                original_handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
                def mask(how, sigs):
                    if mode == 'pending_signal' and how == signal.SIG_SETMASK:
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                    return set()
                with patch.object(probe.subprocess,'Popen',return_value=child,
                                  side_effect=OSError('spawn failed') if mode == 'spawn_failure' else None) as popen, \
                     patch.object(probe.signal,'pthread_sigmask',side_effect=mask):
                    if mode == 'pending_signal':child.wait.side_effect = [-15]
                    code = probe.supervisor()
                receipt = json.loads((probe.OUTPUT/'supervisor.json').read_text())
                self.assertEqual(receipt['status'],'completed' if mode == 'completed' else
                                 'timed_out' if mode.startswith('timeout') else
                                 'cleanup_failed' if mode == 'cleanup_error' else 'failed')
                self.assertEqual(code,0 if mode == 'completed' else 1)
                self.assertEqual(receipt['worker_reaped'],mode != 'cleanup_error')
                self.assertIn('finished_utc',receipt)
                for sig,handler in original_handlers.items():self.assertEqual(signal.getsignal(sig),handler)
                self.assertEqual(receipt['wall_limit_seconds'],600)
                self.assertEqual(receipt['cleanup_limit_seconds'],15)
                self.assertEqual(popen.call_args.args[0][-1],'--worker')
                self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                if mode.startswith('timeout') or mode == 'pending_signal':
                    self.assertTrue(receipt['terminate_sent']); child.terminate.assert_called_once()
                    self.assertEqual(receipt['exit_code'],-9 if mode == 'timeout_kill' else -15)
                    # Preserve the worker's last state, with terminal truth in supervisor.json.
                    self.assertEqual(json.loads((probe.OUTPUT/'result.json').read_text())['status'],'running')
                elif mode == 'cleanup_error':
                    child.terminate.assert_called_once(); self.assertIn('cleanup_error',receipt)
                else: child.terminate.assert_not_called()
                if mode == 'timeout_kill': child.kill.assert_called_once()
                else: child.kill.assert_not_called()


if __name__ == '__main__':unittest.main()
