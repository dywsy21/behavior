import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import msgpack
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import aggregate, expected_cases, load_official_task_names
from wire import packb, unpackb
from summarize import summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [f'task{i}' for i in range(100)]

    def record(self, task='task0', instance=301, q=.5, success=False):
        return dict(task=task, instance_id=instance, rollout_id=0, success=success, q_score=dict(final=q))

    def test_exact_1000_disjoint_cases(self):
        self.assertEqual(len(expected_cases(self.tasks)), 1000)
        self.assertNotIn(('task0',1,0), expected_cases(self.tasks))
        self.assertNotIn(('task0',311,0), expected_cases(self.tasks))

    def test_task_names_are_canonical_not_legacy_instruction_text(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tasks.csv'
            p.write_text('Task ID,Task,task\n'+''.join(
                f'{i},task_{i},Instruction that is not the identifier\n' for i in range(100)))
            self.assertEqual(load_official_task_names(p)[0],'task_0')
            p.write_text('Task ID,Task\n0,turning_on_radio\n')
            with self.assertRaises(ValueError):load_official_task_names(p)

    def test_partial_missing_zero_explicit(self):
        report = aggregate(self.tasks,[self.record()])
        self.assertEqual(report['official_q_score'],.0005)
        self.assertEqual(report['completed_only_q'],.5)
        self.assertEqual(report['status'],'incomplete')
        self.assertEqual(report['missing'],999)

    def test_full_means_and_success_separate(self):
        rows = [self.record(t,i,.4,(i==301)) for t in self.tasks for i in range(301,311)]
        report = aggregate(self.tasks,rows)
        self.assertEqual(report['status'],'complete')
        self.assertAlmostEqual(report['official_q_score'],.4)
        self.assertAlmostEqual(report['official_sr'],.1)

    def test_duplicate_train_and_peak_q_not_accepted(self):
        for rows in [[self.record(),self.record()], [self.record(instance=0)],
                     [self.record(q=float('nan'))], [self.record(success=1)]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                aggregate(self.tasks,rows)

    def test_official_byte_key_decoder_compatibility(self):
        # This is the official unpack_data contract, not our own roundtrip.
        def official(value):
            if b'__ndarray__' in value:
                return np.ndarray(buffer=value[b'data'],dtype=np.dtype(value[b'dtype']),shape=value[b'shape'])
            return value
        chunk = np.arange(2*16*23,dtype=np.float32).reshape(2,16,23)
        decoded = msgpack.unpackb(packb(dict(action=chunk[:,0],action_chunk=chunk)),object_hook=official)
        np.testing.assert_array_equal(decoded['action'],decoded['action_chunk'][:,0])
        np.testing.assert_array_equal(unpackb(packb(chunk)),chunk)

    def test_native_sft_route_no_ppo_or_privileged_prepare(self):
        tree = ast.parse((Path(__file__).resolve().parents[1]/'native_engine.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='infer_native')
        text = ast.unparse(method)
        self.assertIn('policy.forward_inference', text)
        for forbidden in ['sample_stochastic_flow','sample_branch','A4DirectPPO','critic_remaining_fraction']:
            self.assertNotIn(forbidden,text)

    def test_live_poll_waits_for_stock_writer_but_final_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);metrics=root/'tasks/task0/json';metrics.mkdir(parents=True)
            package=root/'submission';package.mkdir()
            (root/'manifest.json').write_text(json.dumps(dict(tasks=self.tasks,source_commit='test',
                checkpoints={},development_notice='not blind')))
            (package/'submission_checklist.json').write_text(json.dumps(dict(metrics_json={},videos={})))
            (metrics/'task0_301_0.json').write_text('{"task":')
            self.assertEqual(summarize(root)['completed'],0)
            with self.assertRaises(json.JSONDecodeError):summarize(root,final=True)
            (metrics/'task0_301_0.json').write_text(json.dumps(self.record()))
            self.assertEqual(summarize(root)['completed'],0)
            with self.assertRaises(ValueError):summarize(root,final=True)


if __name__=='__main__': unittest.main()
