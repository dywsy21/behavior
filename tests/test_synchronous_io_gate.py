import argparse
import ast
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import launch_h73 as launch
from test_native_full_gate import result_fixture
from semantic_robot.v2.synchronous_io import VIEWS


class SynchronousGateTests(unittest.TestCase):
    def test_launch_keeps_registered_identity_flags_and_budgets(self):
        with ExitStack() as stack:
            for key in ('ROOT','RUNTIME','WALL_SECONDS','FLAGS','identity','validate_result'):
                stack.enter_context(patch.object(launch.gate,key,getattr(launch.gate,key)))
            fake=SimpleNamespace(REPO=Path('/source'),PYTHON=Path('/python'))
            stack.enter_context(patch.object(launch.gate,'configure',return_value=fake))
            base=launch.configure();args=launch.gate.command(base)
            self.assertEqual(args.count('--synchronous-io-v1'),1)
            self.assertEqual(args[args.index('--task')+1],'0')
            self.assertEqual(args[args.index('--prefix')+1],'0')
            self.assertEqual(launch.gate.WALL_SECONDS,1200)
            self.assertTrue(str(base.ENTRYPOINT).endswith('launch_h73.py'))
            source=Path(launch.__file__).with_name('run_v2.py')
            main=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
            end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
                     and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='parse_args')
            tree=ast.fix_missing_locations(ast.Module(body=main.body[:end+1],type_ignores=[]))
            ns={'argparse':argparse}
            with patch.object(sys,'argv',args[1:]):exec(compile(tree,str(source),'exec'),ns)
            self.assertEqual(vars(ns['args']),launch.gate.expected_args(base))

    def fixture(self,folder):
        result=result_fixture();result.update(synchronous_io_v1=True,controls=1)
        before={'simulation_time':1.,'physics_index':120}
        after={'simulation_time':1.+1/30,'physics_index':124}
        rows=[{'kind':'control','call':1,'before':before,'after':after,'completed':True,
               'actual_render_on_step':False,'render_requested':True},
              {'kind':'capture','capture':1,'before':after,'after':after,'completed':True,
               'references':{v:{'referenceTimeNumerator':124,'referenceTimeDenominator':120} for v in VIEWS}},
              {'kind':'close','before':after,'after':after,'completed':True}]
        sensors=[{'label':'one','cameras':{v:{'freshness_proof':'native_reference_time',
                   'native_time':{'reference_time_verified':True,'physics_index':124,
                                  'simulation_time':after['simulation_time'],'physics_ticks_in_capture':0}}
                   for v in VIEWS}}]
        return result,rows,sensors

    def save(self,folder,result,rows,sensors):
        raw=''.join(json.dumps(r)+'\n' for r in rows).encode()
        (folder/'native_io.jsonl').write_bytes(raw)
        (folder/'sensor_checks.json').write_text(json.dumps(sensors))
        result['native_io']={'controls':1,'attempts':1,'captures':1,'journal_sha256':hashlib.sha256(raw).hexdigest()}

    def test_completion_binds_every_control_capture_and_sensor_receipt(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(launch.gate,'ROOT',Path(tmp)):
            folder=Path(tmp)/'gate';folder.mkdir()
            result,rows,sensors=self.fixture(folder);self.save(folder,result,rows,sensors)
            launch.validate_result(result,'digest')
            variants=[lambda r,s:r[0]['after'].update(physics_index=120),
                      lambda r,s:r[0].update(completed=False),
                      lambda r,s:r[1]['references']['head'].update(referenceTimeNumerator=120),
                      lambda r,s:r[1].update(before={'simulation_time':2.,'physics_index':240}),
                      lambda r,s:r.pop(),
                      lambda r,s:s[0]['cameras']['head']['native_time'].update(physics_index=120),
                      lambda r,s:s.clear()]
            for change in variants:
                r,s=deepcopy(rows),deepcopy(sensors);change(r,s);self.save(folder,result,r,s)
                with self.subTest(change=change),self.assertRaises(ValueError):launch.validate_result(result,'digest')


if __name__=='__main__':unittest.main()
