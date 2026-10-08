import ast
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from batch_core import validate_indices, needs_plan, stage_plans, commit_plans, partitions
from sparse_cache import indexed_cache_freeze


class Ledger:
    def __init__(self, name):
        self.task_name = name
        self.memory = 'old-' + name
        self.revision = 0

    def stage(self, event):
        if event.get('bad'):
            raise ValueError('Rejected event')
        return dict(installed_subgoal=event, event=event, proposal_id='one')

    def commit(self, proposal):
        self.memory = 'new-' + self.task_name
        self.revision += 1


def slot(name, chunks=0):
    return dict(ledger=Ledger(name), projection=None, context_id=None,
                chunks=chunks, planned_chunk=None)


def event(name):
    return dict(parent_goal=name, active_skills_semantic_json='["'+name+'"]',
                active_skills_text=name)


class BatchTests(unittest.TestCase):
    def test_production_reset_clears_cross_task_and_episode_memory(self):
        class ResetLedger:
            def __init__(self,task):
                self.memory=json.dumps(dict(task_name=task,issued_command_history=[],verified_world_facts=[]))
                self.revision=0
        stage=SOURCE.parents[1]/'rl/memlite_online/code/stage1_engine.py'
        begin=next(n for n in ast.walk(ast.parse(stage.read_text()))
                   if isinstance(n,ast.FunctionDef) and n.name=='begin')
        scope=dict(torch=torch,json=json,PlannerLedger=ResetLedger,
            Path=lambda path:SimpleNamespace(read_text=lambda:'\n'.join(
                json.dumps(dict(task_name=t)) for t in ['task_a','task_b'])))
        exec(compile(ast.Module(body=[begin],type_ignores=[]),'actual-stage1-begin','exec'),scope)
        base=type('NativeSFT',(),dict(begin=scope['begin'],prepare=lambda *a:'prepared'))
        node=next(n for n in ast.parse((SOURCE/'batched_engine.py').read_text()).body
                  if isinstance(n,ast.ClassDef) and n.name=='BatchedSFT')
        scope=dict(torch=torch,NativeSFT=base,json=json)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-batch-reset','exec'),scope)
        cls=scope['BatchedSFT'];engine=cls.__new__(cls)
        engine.trainer=None;engine.session_generation=0;engine.requests=0
        with tempfile.TemporaryDirectory() as directory:
            engine.output=Path(directory)/'no_checkpoints'
            for task in ['task_a','task_b','task_a']:
                meta=[dict(task=task,split='train_smoke',instance_id=i) for i in (1,2)]
                old_slots=getattr(engine,'slots',[])
                engine.begin(task,2,17,meta)
                self.assertTrue(all(s['chunks']==0 and s['projection'] is None for s in engine.slots))
                self.assertTrue(all(s is not old for s in engine.slots for old in old_slots))
                self.assertEqual(engine.task,task.replace('_',' '))
                valid=dict(task_name=engine.task,memlite_branch='low')
                self.assertEqual(engine.prepare('low',[dict(task=engine.task)],[valid]),'prepared')
                for projection in [valid|dict(task_name='another task'),valid|dict(memlite_branch='high')]:
                    with self.assertRaises(ValueError):
                        engine.prepare('low',[dict(task=engine.task)],[projection])
                engine.slots[0].update(chunks=97,projection={'old':True},planned_chunk=96,context_id='old')
                engine.slots[0]['ledger'].memory='DO NOT LEAK'
                engine.slots[0]['ledger'].revision=99
            self.assertEqual(engine.session_generation,3)
            lines=[json.loads(line) for line in (Path(directory)/'session_begin.jsonl').read_text().splitlines()]
            self.assertEqual([row['task'] for row in lines],['task_a','task_b','task_a'])
            with self.assertRaises(ValueError):engine.begin('task_a',2,17,[meta[0],meta[0]])
            with self.assertRaises(ValueError):engine.begin('task_b',2,17,meta)

    def test_sparse_cache_freeze_exact_rows_and_restore_helpers(self):
        class Cache:pass
        cache=Cache()
        original=torch.arange(24).reshape(4,2,3).float()
        for name in ('conv_states','recurrent_states'):
            setattr(cache,name,{0:original.clone(),1:None,3:original.clone()*2})
        helper=SimpleNamespace(_snapshot_sparse_states='original-snapshot',
                               _restore_sparse_states='original-restore')
        with indexed_cache_freeze(helper,Cache):
            self.assertIsNone(helper._snapshot_sparse_states(cache,torch.zeros(4,dtype=torch.bool)))
            saved=helper._snapshot_sparse_states(cache,torch.tensor([False,True,False,True]))
            for name in ('conv_states','recurrent_states'):
                for state in getattr(cache,name).values():
                    if state is not None:state.add_(100)
            helper._restore_sparse_states(cache,saved)
            for name in ('conv_states','recurrent_states'):
                for layer in (0,3):
                    wanted=original.clone()*(1 if layer==0 else 2)
                    wanted[[0,2]]+=100
                    torch.testing.assert_close(getattr(cache,name)[layer],wanted,rtol=0,atol=0)
        self.assertEqual(helper._snapshot_sparse_states,'original-snapshot')
        self.assertEqual(helper._restore_sparse_states,'original-restore')

    def test_indices_reject_aliases_bad_bounds_empty(self):
        for obs, indices in [([],[]), ([{}],[0,1]), ([{},{}],[0,0]), ([{}],[-1]),
                             ([{}],[4]), ([{}],[True])]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                validate_indices(obs,indices,4)
        validate_indices([{},{}],[3,1],4)

    def test_plan_atomic_and_noncontiguous_order(self):
        slots=[slot(str(i)) for i in range(4)]
        with self.assertRaises(ValueError):
            stage_plans(slots,[3,1],[event('three'),dict(bad=True)])
        self.assertTrue(all(s['ledger'].revision==0 for s in slots))
        prepared=stage_plans(slots,[3,1],[event('three'),event('one')])
        self.assertTrue(all(s['ledger'].revision==0 for s in slots))
        commit_plans(slots,prepared)
        self.assertEqual(slots[3]['projection']['parent_goal'],'three')
        self.assertEqual(slots[1]['projection']['parent_goal'],'one')
        self.assertEqual(slots[3]['projection']['memory'],'old-3')
        self.assertEqual(slots[3]['ledger'].memory,'new-3')
        self.assertIsNone(slots[0]['projection'])
        self.assertIsNone(slots[2]['projection'])

    def test_cardinality_cannot_silently_truncate(self):
        with self.assertRaises(ValueError):stage_plans([slot('0')],[0],[])

    def test_planner_period_and_already_planned_barrier(self):
        s=slot('0')
        self.assertTrue(needs_plan(s))
        s.update(projection={},planned_chunk=0)
        self.assertFalse(needs_plan(s))
        for chunk in range(1,8):
            s['chunks']=chunk
            self.assertFalse(needs_plan(s))
        s['chunks']=8
        self.assertTrue(needs_plan(s))
        s['planned_chunk']=8
        self.assertFalse(needs_plan(s))

    def test_partition_no_duplicate_padding_of_tail(self):
        self.assertEqual(partitions(list(range(10)),4),[[0,1,2,3],[4,5,6,7],[8,9]])
        self.assertEqual(partitions([7,9],4),[[7,9]])
        with self.assertRaises(ValueError):partitions([0,0],4)

    def load_noise(self):
        tree=ast.parse((SOURCE/'batched_engine.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='row_noise')
        scope=dict(contextmanager=contextmanager,torch=torch)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'noise','exec'),scope)
        return scope['row_noise']

    def test_noise_preserves_scalar_draw_shapes_and_restores(self):
        noise=self.load_noise()
        calls=[]
        class Helper:
            def _sample_noise(self,actions,dtype,embodiments):
                calls.append((tuple(actions.shape),embodiments))
                return torch.randn_like(actions)
        h=Helper();shape=torch.empty(4,32,27)
        torch.manual_seed(123)
        expected=torch.cat([h._sample_noise(shape[:1],torch.float32,['r1']) for _ in range(4)])
        calls.clear();torch.manual_seed(123);capture=[]
        with noise(h,capture=capture):
            actual=h._sample_noise(shape,torch.float32,['r1']*4)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(calls,[((1,32,27),['r1'])]*4)
        self.assertNotIn('_sample_noise',h.__dict__)
        torch.testing.assert_close(capture[0],expected)

    def test_noise_fixed_reorder_and_exception_restore(self):
        noise=self.load_noise()
        fn=lambda *a:(_ for _ in ()).throw(AssertionError('Must not draw'))
        helper=SimpleNamespace(_sample_noise=fn)
        fixed=torch.arange(4*32*27).reshape(4,32,27).float()
        with noise(helper,fixed=fixed[[3,1]]):
            actual=helper._sample_noise(torch.empty(2,32,27),torch.float32,None)
        torch.testing.assert_close(actual,fixed[[3,1]])
        self.assertIs(helper._sample_noise,fn)
        with self.assertRaises(ValueError),noise(helper,fixed=fixed):
            helper._sample_noise(torch.empty(1,32,27),torch.float32,None)
        self.assertIs(helper._sample_noise,fn)

    def test_real_dispatch_methods_route_noncontiguous_rows(self):
        # Execute the production class body, replacing only heavyweight model IO.
        tree=ast.parse((SOURCE/'batched_engine.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BatchedSFT')
        scope=dict(torch=torch,NativeSFT=object,validate_indices=validate_indices,
            needs_plan=needs_plan,time=__import__('time'),deepcopy=deepcopy,
            stage_plans=stage_plans,commit_plans=commit_plans,json=json,np=np)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'batched','exec'),scope)
        cls=scope['BatchedSFT'];engine=cls.__new__(cls)
        engine.mode='batch';engine.capture=False;engine.requests=0;engine.task='task';engine.trainer=None
        engine.slots=[slot(str(i)) for i in range(4)]
        high=[];low=[]
        def plan(obs,indices):
            high.append(list(indices));commit_plans(engine.slots,
                stage_plans(engine.slots,indices,[event(str(o['id'])) for o in obs]))
        def low_batch(obs,projections):
            low.append([p['parent_goal'] for p in projections])
            return np.stack([np.full((16,23),o['id']) for o in obs]),None
        engine.plan_batch=plan;engine.low_batch=low_batch
        actions,contexts=engine.infer_native([dict(id=3),dict(id=1)],[3,1])
        self.assertEqual(high,[[3,1]])
        self.assertEqual(low,[['3','1']])
        self.assertEqual([c['parent_goal'] for c in contexts],['3','1'])
        self.assertEqual(actions[:,0,0].tolist(),[3,1])
        self.assertEqual([s['chunks'] for s in engine.slots],[0,1,0,1])
        engine.slots[3]['chunks']=8
        engine.infer_native([dict(id=1),dict(id=3)],[1,3])
        self.assertEqual(high,[[3,1],[3]])
        self.assertEqual(low[-1],['1','3'])
        self.assertEqual([s['chunks'] for s in engine.slots],[0,2,0,9])


if __name__=='__main__':unittest.main()
