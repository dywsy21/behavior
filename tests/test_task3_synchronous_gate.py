import argparse
import ast
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import launch_h77 as launch
from test_synchronous_io_gate import SynchronousGateTests


@contextmanager
def configured():
    with ExitStack() as stack:
        for key in ('ROOT','RUNTIME','WALL_SECONDS','FLAGS','identity','validate_result','validate_manifest','command'):
            stack.enter_context(patch.object(launch.gate,key,getattr(launch.gate,key)))
        stack.enter_context(patch.object(launch.synchronous,'ORIGINAL_VALIDATE',launch.synchronous.ORIGINAL_VALIDATE))
        fake=SimpleNamespace(REPO=Path('/source'),PYTHON=Path('/python'))
        stack.enter_context(patch.object(launch.gate,'configure',return_value=fake))
        yield launch.configure()


class Task3GateTests(unittest.TestCase):
    def test_original_runtime_digest_stays_exactly_h75(self):
        from run_v2 import implementation_digest
        self.assertEqual(implementation_digest(),launch.DIGEST)

    def test_only_task_paths_change_and_actual_parser_matches_manifest(self):
        with configured() as base:
            command=launch.gate.command(base)
            self.assertEqual(command[command.index('--task')+1],'3')
            self.assertEqual(launch.gate.ROOT.name,'h77_task3_gate_v1')
            self.assertEqual(launch.gate.RUNTIME.name,'h77_task3_gate_v1')
            self.assertEqual(launch.gate.WALL_SECONDS,1200)
            self.assertEqual(base.ENTRYPOINT,Path(launch.__file__).resolve())
            self.assertEqual(command.count('--synchronous-io-v1'),1)
            source=Path(launch.__file__).with_name('run_v2.py')
            main=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
            end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
                     and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='parse_args')
            tree=ast.fix_missing_locations(ast.Module(body=main.body[:end+1],type_ignores=[]));ns={'argparse':argparse}
            with patch.object(sys,'argv',command[1:]):exec(compile(tree,str(source),'exec'),ns)
            self.assertEqual(vars(ns['args']),launch.gate.expected_args(base))

    def test_manifest_rejects_wrong_task_instance_model_and_budget(self):
        with configured() as base:
            manifest={'code_commit':'commit','implementation_digest':launch.DIGEST,'instance':242,'task':3,
                      'task_name':'cleaning_up_plates_and_food','split':'train','seed':0,'training_updates':0,
                      'model_identity':None,'actor_scene_truth':False,'prefix_is_expert_not_agent':False,
                      'diagnostic_replay_requested':False,'native_profile':'a100_full_v1','args':launch.gate.expected_args(base)}
            launch.validate_manifest(manifest,base,'commit',launch.DIGEST)
            for key,value in (('task',0),('instance',138),('seed',True),('training_updates',1),('actor_scene_truth',True)):
                broken=deepcopy(manifest);broken[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):launch.validate_manifest(broken,base,'commit',launch.DIGEST)

    def test_actual_task3_result_keeps_complete_clock_and_batch_validator(self):
        with configured(),tempfile.TemporaryDirectory() as temp,patch.object(launch.gate,'ROOT',Path(temp)):
            folder=Path(temp)/'gate';folder.mkdir();fixture=SynchronousGateTests()
            result,rows,sensors=fixture.fixture(folder)
            result.update(task=3,implementation_digest=launch.DIGEST)
            fixture.save(folder,result,rows,sensors)
            launch.gate.validate_result(result,launch.DIGEST)
            original=deepcopy(result)
            result['task']=0
            with self.assertRaises(ValueError):launch.gate.validate_result(result,launch.DIGEST)
            result=original
            rows[2]['after']['physics_index']=120
            fixture.save(folder,result,rows,sensors)
            with self.assertRaises(ValueError):launch.gate.validate_result(result,launch.DIGEST)


if __name__=='__main__':unittest.main()
