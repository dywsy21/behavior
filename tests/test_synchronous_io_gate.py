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
from semantic_robot.v2.render_batch import make_receipts
sys.path.insert(0,str(Path(__file__).resolve().parent/'semantic_robot'))
from render_batch_fixture import batch_fixture, camera_receipts


class SynchronousGateTests(unittest.TestCase):
    def test_h75_is_a_new_single_bounded_run_with_digest_bound_launcher(self):
        import launch_h75
        from native_full_profile import DEPENDENCY_FILES
        with ExitStack() as stack:
            for key in ('ROOT','RUNTIME','WALL_SECONDS','FLAGS','identity','validate_result'):
                stack.enter_context(patch.object(launch.gate,key,getattr(launch.gate,key)))
            fake=SimpleNamespace(REPO=Path('/source'),PYTHON=Path('/python'))
            stack.enter_context(patch.object(launch.gate,'configure',return_value=fake))
            base=launch_h75.configure();args=launch.gate.command(base)
            self.assertEqual(launch.gate.ROOT.name,'h75_render_batch_v1')
            self.assertEqual(launch.gate.RUNTIME.name,'h75_render_batch_v1')
            self.assertEqual(launch.gate.WALL_SECONDS,1200)
            self.assertEqual(args.count('--synchronous-io-v1'),1)
            self.assertEqual(base.ENTRYPOINT,Path(launch_h75.__file__).resolve())
            self.assertIn('launch_h75.py',DEPENDENCY_FILES)

    def test_h74_has_new_paths_same_gate_budget_and_fixed_source_dependency(self):
        import launch_h74
        from native_full_profile import DEPENDENCY_FILES
        with ExitStack() as stack:
            for key in ('ROOT','RUNTIME','WALL_SECONDS','FLAGS','identity','validate_result'):
                stack.enter_context(patch.object(launch.gate,key,getattr(launch.gate,key)))
            fake=SimpleNamespace(REPO=Path('/source'),PYTHON=Path('/python'))
            stack.enter_context(patch.object(launch.gate,'configure',return_value=fake))
            base=launch_h74.configure();args=launch.gate.command(base)
            self.assertEqual(launch.gate.ROOT.name,'h74_graph_lifecycle_v1')
            self.assertEqual(launch.gate.RUNTIME.name,'h74_graph_lifecycle_v1')
            self.assertEqual(launch.gate.WALL_SECONDS,1200)
            self.assertEqual(args.count('--synchronous-io-v1'),1)
            self.assertEqual(args[args.index('--prefix')+1],'0')
            self.assertEqual(args[args.index('--max-controls')+1],'1536')
            self.assertEqual(base.ENTRYPOINT,Path(launch_h74.__file__).resolve())
            self.assertIn('launch_h74.py',DEPENDENCY_FILES)

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
        batch=batch_fixture(134)
        rows=[{'kind':'initialize','before':before,'after':before,'completed':True},
              {'kind':'prime','before':before,'after':before,'completed':True,'discarded':True,'batch':batch_fixture(132)},
              {'kind':'control','call':1,'before':before,'after':after,'completed':True,
               'actual_render_on_step':False,'render_requested':True},
              {'kind':'capture','capture':1,'before':after,'after':after,'completed':True,
               'baseline':batch_fixture(133),'batch':deepcopy(batch)},
              {'kind':'read','capture':1,'before':after,'after':after,'completed':True,'batch':deepcopy(batch),
               'buffer_hashes':{v:{'rgb_sha256':'a'*64,'depth_sha256':'b'*64} for v in VIEWS}},
              {'kind':'close','before':after,'after':after,'completed':True}]
        sensors=[{'label':'one','cameras':camera_receipts(make_receipts(batch,after,1))}]
        return result,rows,sensors

    def save(self,folder,result,rows,sensors):
        raw=''.join(json.dumps(r)+'\n' for r in rows).encode()
        (folder/'native_io.jsonl').write_bytes(raw)
        (folder/'sensor_checks.json').write_text(json.dumps(sensors))
        result['native_io']={'controls':1,'attempts':1,'captures':1,'reads':1,'primes':1,
                             'journal_sha256':hashlib.sha256(raw).hexdigest()}

    def test_completion_binds_every_control_capture_and_sensor_receipt(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(launch.gate,'ROOT',Path(tmp)):
            folder=Path(tmp)/'gate';folder.mkdir()
            result,rows,sensors=self.fixture(folder);self.save(folder,result,rows,sensors)
            launch.validate_result(result,'digest')
            variants=[lambda r,s:r[2]['after'].update(physics_index=120),
                      lambda r,s:r[0].update(completed=False),
                      lambda r,s:r[3]['batch']['cameras']['head']['frame'].update(rationalTimeOfSimNumerator=120),
                      lambda r,s:r[3].update(before={'simulation_time':2.,'physics_index':240}),
                      lambda r,s:r[3].update(baseline=batch_fixture(134)),
                      lambda r,s:r[4].update(batch=batch_fixture(135)),
                      lambda r,s:r[4]['buffer_hashes']['head'].update(rgb_sha256='c'*64),
                      lambda r,s:r.pop(4),
                      lambda r,s:r[1].update(discarded=False),
                      lambda r,s:r.pop(0),
                      lambda r,s:r.pop(),
                      lambda r,s:s[0]['cameras']['head']['native_time'].update(physics_index=120),
                      lambda r,s:s.clear()]
            for change in variants:
                r,s=deepcopy(rows),deepcopy(sensors);change(r,s);self.save(folder,result,r,s)
                with self.subTest(change=change),self.assertRaises(ValueError):launch.validate_result(result,'digest')


if __name__=='__main__':unittest.main()
