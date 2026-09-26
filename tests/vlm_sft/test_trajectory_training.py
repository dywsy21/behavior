from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import trajectory_training as training
import launch_trajectory_benchmark as launcher
import prepare_trajectory_benchmark as cpu_prepare


def sample(length, targets):
    ids = (torch.arange(length) % 7).view(1, -1)
    labels = torch.full_like(ids, -100); labels[:, -targets:] = ids[:, -targets:]
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': labels}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = torch.nn.Embedding(7, 4); self.linear = torch.nn.Linear(4, 7)

    def forward(self, input_ids, logits_to_keep=None, **kwargs):
        from types import SimpleNamespace
        logits = self.linear(self.embedding(input_ids))
        if logits_to_keep is not None: logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


class TrainingTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_epoch_order_covers_all_windows_without_replacement(self):
        first = training.epoch_order(1000)
        self.assertEqual(set(first), set(range(1000)))
        self.assertEqual(first, training.epoch_order(1000))
        self.assertNotEqual(first, training.epoch_order(1000, epoch=1))
        self.assertNotEqual(first[:20], list(range(20)))
        for invalid in (True, 0, -1):
            with self.assertRaises(ValueError): training.epoch_order(invalid)

    def test_collation_keeps_audit_indices_outside_model_inputs(self):
        result = training.collate_training([(3, sample(7, 2)), (9, sample(9, 4))], pad_token_id=0)
        self.assertEqual(result['indices'], [3, 9]); self.assertEqual(result['target_tokens'], 6)
        self.assertEqual(result['total_tokens'], 16)
        self.assertNotIn('indices', result['batch'])
        self.assertTrue(torch.all(result['batch']['labels'][result['batch']['attention_mask'] == 0] == -100))

    def test_spawn_state_never_serializes_open_decoder(self):
        dataset = training.LazyTrainingDataset('/dataset', 'digest', '/model', 300)
        dataset._dataset = lambda: None  # Deliberately not pickleable decoder stand-in.
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(restored._dataset)
        self.assertEqual(len(restored), 300)

    def test_unequal_token_accumulation_matches_one_effective_batch_gradient(self):
        torch.manual_seed(3); model = TinyModel(); other = deepcopy(model)
        values = [sample(n, t) for n, t in ((7, 2), (9, 4), (11, 7), (8, 1))]
        chunks = [training.collate_training(list(enumerate(values[i:i + 2])), pad_token_id=0) for i in (0, 2)]
        actual_loss = training.accumulated_backward(model, chunks, move=lambda b: b)
        complete = training.collate_training(list(enumerate(values)), pad_token_id=0)['batch']
        expected_loss = training.supervised_loss(other, complete); expected_loss.backward()
        self.assertAlmostEqual(actual_loss, float(expected_loss), places=6)
        for actual, expected in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
        broken = deepcopy(chunks); broken[0]['target_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'token count'):
            training.accumulated_backward(TinyModel(), broken, move=lambda b: b)

    def make_corpus(self, root):
        sources = []
        for task in range(5):
            rows = [{'id': f't{task}_i{task}_e{task}_f{i:06d}', 'total_tokens': 1200 + 60 * task + i,
                     'target_tokens': 100 + i} for i in range(60)]
            rel = f'shards/train/episode_{task:06d}.parquet'; path = root / rel; path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path); data = path.read_bytes()
            sources.append({'split': 'train', 'task': task, 'instance': task, 'episode': task, 'samples': 60,
                'files': [{'kind': 'shards', 'path': rel, 'samples': 60, 'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()}]})
        # Deliberately nonexistent heldout files must never be read to plan TRAIN.
        sources.append({'split': 'validation', 'files': [{'path': 'do-not-read-heldout.parquet'}]})
        return {'sources': sources, 'samples_by_split': {'train': 300}}

    def test_plan_has_global_extremes_unique_rows_and_same_dataset_index_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest = self.make_corpus(root)
            plan = training.make_plan(root, manifest)
            self.assertEqual(plan['gate_indices'], [0, 299])
            self.assertEqual(len(plan['selected']), 256)
            self.assertEqual(len({r['index'] for r in plan['selected']}), 256)
            self.assertEqual(len({r['id'] for r in plan['selected']}), 256)
            self.assertEqual({r['task'] for r in plan['selected'][32:]}, set(range(5)))
            for row in plan['selected']:
                self.assertEqual(row['task'], row['index'] // 60)
                self.assertEqual(row['total_tokens'], 1200 + row['index'])
            self.assertEqual(plan, training.make_plan(root, manifest))
            (root / manifest['sources'][0]['files'][0]['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError): training.make_plan(root, manifest)

    def test_duplicate_train_instances_and_wrong_split_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = self.make_corpus(Path(folder))
            duplicate = deepcopy(manifest); duplicate['sources'].insert(0, duplicate['sources'][0])
            with self.assertRaises(ValueError): training.training_sources(duplicate)
            manifest['sources'][0]['files'][0]['path'] = 'shards/test/episode_000000.parquet'
            with self.assertRaises(ValueError): training.training_sources(manifest)

    def records(self):
        return [{'step': i + 1, 'indices': list(range(8 * i, 8 * i + 8)),
                 'compute_seconds': 100 if i < 4 else (2 if i == 4 else 4),
                 'end_to_end_seconds': 120 if i < 4 else 10} for i in range(32)]

    def test_capacity_excludes_warmup_uses_fastest_compute_not_slow_io(self):
        result = training.capacity_report(self.records(), 212500)
        self.assertEqual(result['timed_unique_windows'], 224)
        self.assertEqual(result['fastest_observed_compute_samples_per_second'], 4)
        self.assertEqual(result['one_unique_pass_seconds_at_fastest_observed_compute'], 53125)
        self.assertEqual(result['windows_for_three_hours_at_fastest_observed_compute'], 43200)
        self.assertTrue(result['single_unique_pass_supports_three_hours'])
        self.assertAlmostEqual(result['compute_samples_per_second'], 224 / 110)
        self.assertAlmostEqual(result['end_to_end_samples_per_second'], .8)
        self.assertFalse(training.capacity_report(self.records(), 300)['single_unique_pass_supports_three_hours'])

    def test_bad_capacity_accounting_rejected(self):
        broken = []
        x = self.records(); x[6]['indices'] = x[5]['indices']; broken.append(x)
        x = self.records(); x[6]['compute_seconds'] = float('nan'); broken.append(x)
        x = self.records(); x[6]['compute_seconds'] = 0; broken.append(x)
        x = self.records(); x[6]['end_to_end_seconds'] = 1; broken.append(x)
        x = self.records(); x[6]['indices'][0] = 999999; broken.append(x)
        broken.append(self.records()[:-1])
        for rows in broken:
            with self.assertRaises(ValueError): training.capacity_report(rows, 212500)

    def resources(self):
        return {gpu: {'used_mib': 0, 'free_mib': 81000, 'processes': []} for gpu in launcher.GPU_UUIDS}

    def test_guard_leaves_teammate_gpus_alone_but_requires_empty_gpu2(self):
        current = self.resources()
        current[launcher.GPU_UUIDS[0]] = {'used_mib': 74000, 'free_mib': 7000,
                                        'processes': [{'pid': 101, 'used_mib': 74000}]}
        launcher.check_resources(current)
        current[launcher.GPU_UUID]['processes'] = [{'pid': 102, 'used_mib': 100}]
        with self.assertRaisesRegex(RuntimeError, 'idle'): launcher.check_resources(current)
        current = self.resources(); current[launcher.GPU_UUID]['free_mib'] = 60000
        with self.assertRaises(RuntimeError): launcher.check_resources(current)

    def test_idle_context_cannot_bypass_wait_for_known_teammate_training(self):
        with (patch.object(launcher, 'PYTHON', Path(sys.executable)),
              patch.object(launcher, 'clean_commit', return_value='code'),
              patch.object(launcher, 'environment_identity', return_value={}),
              patch.object(launcher, 'checked_data'),
              patch.object(launcher, 'snapshot', return_value=self.resources()),
              patch.object(launcher, 'WAITING_TRAINING_PIDS', (os.getpid(),))):
            with self.assertRaisesRegex(RuntimeError, 'natural exit'): launcher.preflight()

    def test_cpu_prepare_refuses_gpu_visibility_before_reading_or_writing(self):
        with (patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2', 'HF_HUB_OFFLINE': '1'}),
              patch.object(cpu_prepare, 'checked_data') as read,
              patch.object(cpu_prepare, 'atomic_json') as write):
            with self.assertRaisesRegex(ValueError, 'CPU-only'): cpu_prepare.main()
            read.assert_not_called(); write.assert_not_called()

    def test_foreign_process_and_own_cross_gpu_or_oversize_rejected(self):
        base = self.resources(); current = deepcopy(base)
        current[launcher.GPU_UUID] = {'used_mib': 20000, 'free_mib': 61000,
                                     'processes': [{'pid': 99, 'used_mib': 20000}]}
        with patch.object(launcher, 'belongs_to_session', side_effect=lambda pid, sid: pid == sid):
            launcher.check_resources(current, base, 99)
            with self.assertRaises(RuntimeError): launcher.check_resources(current, base, 99, released=True)
            current[launcher.GPU_UUID]['processes'].append({'pid': 100, 'used_mib': 10})
            with self.assertRaisesRegex(RuntimeError, 'external'): launcher.check_resources(current, base, 99)
            # Cleanup only requires our session to leave; foreign jobs stay.
            current[launcher.GPU_UUID]['processes'] = [{'pid': 100, 'used_mib': 10}]
            launcher.check_resources(current, base, 99, released=True)
            current = deepcopy(base); current[launcher.GPU_UUIDS[1]]['processes'] = [{'pid': 99, 'used_mib': 1}]
            with self.assertRaises(RuntimeError): launcher.check_resources(current, base, 99)
            current = deepcopy(base); current[launcher.GPU_UUID]['used_mib'] = 40000
            with self.assertRaises(RuntimeError): launcher.check_resources(current, base, 99)

    def result_files(self, folder):
        records = self.records()
        for row in records: row.update(loss=1., gradient_norm=.5)
        selected = [{'index': i} for row in records for i in row['indices']]
        launch = {'model_identity': {'revision': 'fixed'}, 'environment': {'python': 'fixed'}}
        identity = {**launch, 'code_commit': 'code', 'config': training.CONFIG,
            'dataset_sha256': training.DATA_SHA, 'qa_sha256': training.QA_SHA,
            'old_adapter_loaded': False, 'online_executor_changed': False, 'physical_gpu': 2}
        plan = {'selected': selected, 'gate_indices': [0, 1], 'train_windows': 212500}
        gate = {'indices': [0, 1], 'left_padding': True, 'native': 1., 'custom': 1.}
        for name, data in (('sampling', plan), ('identity', identity), ('loss_gate', gate)):
            (folder / (name + '.json')).write_text(json.dumps(data))
        (folder / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        result = {'status': 'TRAINING_PIPELINE_BENCHMARKED', 'code_commit': 'code', 'config': training.CONFIG,
            'dataset_sha256': training.DATA_SHA, 'qa_sha256': training.QA_SHA,
            'optimizer_updates': 32, 'changed_lora_tensors': 2, 'unique_train_windows': 256,
            'checkpoint_writes': 0, 'training_eligible': False, 'three_hour_training_performed': False,
            'policy_success_rate_evaluated': False, 'wall_seconds': 1000., 'cuda_peak_reserved_mib': 12000.,
            'train_windows': 212500, 'capacity': training.capacity_report(records, 212500)}
        for name in ('sampling', 'identity', 'loss_gate', 'steps'):
            extension = '.jsonl' if name == 'steps' else '.json'
            result[name + '_sha256'] = hashlib.sha256((folder / (name + extension)).read_bytes()).hexdigest()
        (folder / 'result.json').write_text(json.dumps(result))
        return result, launch

    def test_supervisor_recomputes_capacity_and_requires_real_numeric_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); result, launch = self.result_files(root)
            self.assertEqual(launcher.verify_result(root, 'code', launch), result)
            altered = deepcopy(result); altered['capacity']['compute_samples_per_second'] *= 2
            (root / 'result.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'reproduced'): launcher.verify_result(root, 'code', launch)
            (root / 'result.json').write_text(json.dumps(result)); (root / 'failure.json').write_text('{}')
            with self.assertRaises(ValueError): launcher.verify_result(root, 'code', launch)

    def test_supervisor_rejects_wrong_code_weight_config_or_missing_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); result, launch = self.result_files(root)
            with self.assertRaises(ValueError): launcher.verify_result(root, 'othercode', launch)
            with self.assertRaises(ValueError): launcher.verify_result(root, 'code', {**launch, 'model_identity': {}})
            (root / 'steps.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'receipt changed'): launcher.verify_result(root, 'code', launch)

    def test_supervisor_external_arrival_stops_only_child_and_preserves_primary_error(self):
        for cleanup_fails in (False, True):
            with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder) / 'new'; baseline = self.resources(); foreign = deepcopy(baseline)
                foreign[launcher.GPU_UUID]['processes'] = [{'pid': 100, 'used_mib': 10}]
                child = Mock(pid=99, returncode=-15); child.poll.return_value = None
                stack.enter_context(patch.object(launcher, 'ROOT', root))
                stack.enter_context(patch.object(launcher, 'preflight', return_value=('code', {}, baseline)))
                stack.enter_context(patch.object(launcher, 'model_identity', return_value={}))
                stack.enter_context(patch.object(launcher, 'snapshot', side_effect=[baseline, foreign, baseline]))
                stack.enter_context(patch.object(launcher, 'belongs_to_session', side_effect=lambda pid, sid: pid == sid))
                start = stack.enter_context(patch.object(launcher.subprocess, 'Popen', return_value=child))
                stop = stack.enter_context(patch.object(launcher, 'stop_owned_group',
                    side_effect=RuntimeError('cleanup failure') if cleanup_fails else None))
                stack.enter_context(patch.object(launcher.os, 'sched_getaffinity', return_value=set(range(48, 56))))
                stack.enter_context(patch.object(launcher.os, 'sched_setaffinity'))
                stack.enter_context(patch.object(launcher.signal, 'pthread_sigmask', return_value=set()))
                stack.enter_context(patch.object(launcher.signal, 'signal', return_value=launcher.signal.SIG_DFL))
                with self.assertRaisesRegex(RuntimeError, 'external GPU2'):
                    launcher.run()
                stop.assert_called_once_with(child)
                self.assertTrue(start.call_args.kwargs['start_new_session'])
                saved = json.loads((root / 'supervisor.json').read_text())
                self.assertEqual(saved['worker_pid'], 99)
                self.assertIn('external GPU2', saved['error'])
                self.assertEqual(saved['status'], 'failed')
                self.assertEqual('cleanup_error' in saved, cleanup_fails)


if __name__ == '__main__': unittest.main()
