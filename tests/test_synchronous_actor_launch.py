import argparse
import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import launch_h79 as launch
from test_synchronous_io_gate import SynchronousGateTests


@contextmanager
def configured(root):
    with ExitStack() as stack:
        for key in ('ROOT','RUNTIME','WALL_SECONDS','FLAGS','identity','validate_result'):
            stack.enter_context(patch.object(launch.gate,key,getattr(launch.gate,key)))
        stack.enter_context(patch.object(launch.synchronous,'ORIGINAL_VALIDATE',launch.synchronous.ORIGINAL_VALIDATE))
        base=SimpleNamespace(REPO=launch.REPO,PYTHON=Path('/python'))
        stack.enter_context(patch.object(launch.gate,'configure',return_value=base))
        stack.enter_context(patch.object(launch,'ROOT',root))
        stack.enter_context(patch.object(launch,'RUNTIME',root/'runtime'))
        yield launch.configure()


class SynchronousActorLaunchTests(unittest.TestCase):
    def test_launch_prepares_private_leaf_before_supervisor_spawn(self):
        with tempfile.TemporaryDirectory() as tmp,configured(Path(tmp)/'run') as base,ExitStack() as stack:
            base.environment=lambda:({'SAFE':'only'},[launch.RUNTIME/'portable',launch.RUNTIME/'cache'])
            stack.enter_context(patch.object(launch,'preflight',return_value=('code',self.snapshot())))
            stack.enter_context(patch.object(launch,'model_identity',return_value={'exact':True}))
            sock=stack.enter_context(patch.object(launch.socket,'socket'))
            sock.return_value.__enter__.return_value.connect_ex.return_value=1
            def spawn(*args,**kwargs):
                receipt=launch.read(launch.ROOT/'launch.json')
                launch.check_runtime_preparation(receipt['runtime_preparation'])
                self.assertEqual(receipt['status'],'reserved')
                self.assertEqual(stat.S_IMODE((launch.RUNTIME/launch.SCREENSHOTS).stat().st_mode),0o700)
                return SimpleNamespace(pid=123)
            proc=stack.enter_context(patch.object(launch.subprocess,'Popen',side_effect=spawn))
            stack.enter_context(patch('builtins.print'))
            launch.launch(base)
            proc.assert_called_once()
            self.assertEqual(launch.read(launch.ROOT/'launch.json')['status'],'supervisor_started')

    def test_new_runtime_leaf_survives_kit_create_and_strict_tree_count(self):
        with tempfile.TemporaryDirectory() as tmp,configured(Path(tmp)):
            receipt=launch.prepare_runtime([launch.RUNTIME/'cache',launch.RUNTIME/'portable',launch.RUNTIME/'data'])
            leaf=launch.RUNTIME/launch.SCREENSHOTS
            self.assertEqual(stat.S_IMODE(leaf.stat().st_mode),0o700)
            leaf.mkdir(mode=0,parents=True,exist_ok=True)  # Kit's repeated creation cannot remove access.
            (leaf/'test.png').write_bytes(b'actual screenshot bytes')
            self.assertEqual(launch.check_runtime_preparation(receipt),receipt)
            def path(value):return Path(tmp) if str(value)=='/mnt/nvme_tmp' else Path(value)
            with patch('launch_h44.Path',side_effect=path):
                self.assertEqual(launch.tree_bytes(launch.RUNTIME,16),23)
                with self.assertRaisesRegex(RuntimeError,'Tree cap'):
                    launch.tree_bytes(launch.RUNTIME,1e-12)
                linked=launch.RUNTIME/'alias';linked.symlink_to(leaf)
                with self.assertRaisesRegex(RuntimeError,'Unexpected file/link'):
                    launch.tree_bytes(launch.RUNTIME,16)

    def test_existing_unreadable_runtime_and_outside_routes_are_never_repaired(self):
        with tempfile.TemporaryDirectory() as tmp,configured(Path(tmp)):
            outside=Path(tmp)/'outside'
            with self.assertRaises(ValueError):launch.prepare_runtime([outside])
            self.assertFalse(outside.exists());self.assertFalse(launch.RUNTIME.exists())
            launch.RUNTIME.mkdir(mode=0o700)
            leaf=launch.RUNTIME/launch.SCREENSHOTS;leaf.mkdir(parents=True)
            leaf.chmod(0)
            try:
                with self.assertRaises(FileExistsError):launch.prepare_runtime([])
                receipt={'schema':'h79-readable-kit-runtime-v1','runtime':str(launch.RUNTIME),'screenshots':str(leaf)}
                with self.assertRaises(ValueError):launch.check_runtime_preparation(receipt)
                self.assertEqual(stat.S_IMODE(leaf.stat().st_mode),0)
            finally:leaf.chmod(0o700)  # Only our disposable fixture, not production repair.

    def test_runtime_rejects_broken_alias_and_changed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp,configured(Path(tmp)):
            launch.RUNTIME.symlink_to(Path(tmp)/'missing',target_is_directory=True)
            with self.assertRaises(FileExistsError):launch.prepare_runtime([])
            launch.RUNTIME.unlink()
            receipt=launch.prepare_runtime([launch.RUNTIME/'portable'])
            with self.assertRaises(ValueError):launch.check_runtime_preparation({**receipt,'screenshots':'/unrelated'})
            leaf=launch.RUNTIME/launch.SCREENSHOTS
            leaf.rmdir();leaf.symlink_to(Path(tmp),target_is_directory=True)
            with self.assertRaises(ValueError):launch.check_runtime_preparation(receipt)

    def test_same_complete_harness_digest_and_explicit_actor_parser(self):
        from run_v2 import implementation_digest
        from semantic_robot.v2.run_budget import validate_run_budget
        self.assertEqual(implementation_digest(),launch.DIGEST)
        with configured(Path('/experiment')) as base:
            command=launch.commands(base)['actor']
            source=launch.REPO/'scripts/semantic_robot/run_v2.py'
            main=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
            end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
                     and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='parse_args')
            tree=ast.fix_missing_locations(ast.Module(body=main.body[:end+1],type_ignores=[]));ns={'argparse':argparse}
            with patch.object(sys,'argv',command[1:]):exec(compile(tree,str(source),'exec'),ns)
            self.assertEqual(vars(ns['args']),launch.expected_actor_args(base));validate_run_budget(ns['args'])
            self.assertEqual(command.count('--synchronous-io-v1'),1)
            self.assertEqual(command[command.index('--mode')+1],'agent')
            self.assertEqual(ns['args'].prefix,0);self.assertIsNone(ns['args'].replay_prefix_spec)
            self.assertEqual(len(ns['args'].gate_result),2)
            self.assertEqual(1+2*ns['args'].max_decisions+2+16+4,launch.MAX_CALLS)
            model=launch.commands(base)['model'];self.assertIn(str(launch.MODEL),model)
            self.assertEqual(model[model.index('--max-calls')+1],'215')

    def snapshot(self):
        return {u:{'used_mib':0,'free_mib':81152,'processes':[]} for u in launch.gate.scene.supervisor.GPU_UUIDS}

    def test_resource_ownership_caps_headroom_and_team_changes(self):
        gpus=list(self.snapshot());before=self.snapshot();state=deepcopy(before)
        def add(index,pid,used):
            row=state[gpus[index]];row['processes'].append({'pid':pid,'used_mib':used,'type':'C'});
            row['used_mib']+=used;row['free_mib']-=used
        with patch.object(launch,'belongs_to_session',side_effect=lambda p,s:p==s):
            launch.check_resources(before)
            add(0,99,12548);add(2,10,56000);add(2,20,400);add(3,20,6000)
            launch.check_resources(state,before,{'model':10,'actor':20})
            for index,pid,used in ((2,999,1),(3,999,1),(0,10,1),(2,20,113),(3,20,19000)):
                original=deepcopy(state);add(index,pid,used)
                with self.subTest(index=index,pid=pid),self.assertRaises(RuntimeError):
                    launch.check_resources(state,before,{'model':10,'actor':20})
                state=original
            with self.assertRaises(RuntimeError):launch.check_resources(state,before,{'model':10,'actor':20},released=True)
            state[gpus[0]]['free_mib']=8191
            with self.assertRaises(RuntimeError):launch.check_resources(state,before,{'model':10,'actor':20})
            with self.assertRaises(RuntimeError):launch.check_resources(state)

    def health(self):
        from semantic_robot.v2.structured_planning import schema_digest,DECODER_COMMIT
        return {'protocol':'semantic-v2','model':str(launch.MODEL),'revision':launch.REVISION,'code_commit':'code',
                'max_calls':215,'calls':0,'visible_devices':launch.gate.scene.supervisor.GPU_UUIDS[2],
                'dtype':'bfloat16','training_updates':0,'thinking':False,'transformers':'5.7.0','torch':'2.7.1+cu128',
                'image_max_side':640,'wrist_no_upsampling':True,'max_images':9,'backend':'transformers-sdpa',
                'planning_schema_sha256':schema_digest(),'structured_planning_schemas':['task_plan_v1','recovery_v1'],
                'structured_decoder':{'commit':DECODER_COMMIT},'finite_choice_kinds':['act','ground','reference']}

    def test_actual_model_health_requires_fresh_same_source_and_decoder(self):
        valid=self.health();launch.check_health(valid,'code',initial=True)
        for key,value in (('calls',1),('calls',True),('thinking',True),('max_calls',431),('training_updates',80),
                          ('code_commit','old'),('visible_devices','3'),('planning_schema_sha256','wrong'),
                          ('structured_decoder',{})):
            changed={**valid,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):launch.check_health(changed,'code',initial=True)
        launch.check_health({**valid,'calls':215},'code')
        with self.assertRaises(ValueError):launch.check_health({**valid,'calls':216},'code')

    def actor_fixture(self,folder):
        helper=SynchronousGateTests();r,rows,sensors=helper.fixture(folder)
        r.update(task=0,implementation_digest=launch.DIGEST,harness='grounded',gate_ok=False,
                 decisions=[{'decision':0}],official_success=False,terminal=False,final_goal_status={'unsatisfied':[0]},
                 wall_s=10.,stop_reason='DECISION_BUDGET_REACHED',model_calls=2)
        helper.save(folder,r,rows,sensors)
        return r,rows,sensors,helper

    def test_actor_terminal_keeps_complete_native_journal_validation(self):
        with tempfile.TemporaryDirectory() as tmp,configured(Path(tmp)):
            folder=Path(tmp)/'gate';folder.mkdir();r,rows,sensors,helper=self.actor_fixture(folder)
            launch.synchronous.validate_result(r,launch.DIGEST)
            for key,value in (('prefix_controls',448),('diagnostic_replay_controls',2),('controls',3073),('model_calls',216),
                              ('official_success','true'),('wall_s',float('nan')),('gate_ok',True)):
                with self.subTest(key=key),self.assertRaises(ValueError):launch.validate_actor_basic({**r,key:value},launch.DIGEST)
            rows[2]['after']['physics_index']=130;helper.save(folder,r,rows,sensors)
            with self.assertRaises(ValueError):launch.synchronous.validate_result(r,launch.DIGEST)

    def test_complete_actor_binds_manifest_reset_journal_and_model_ledger(self):
        with tempfile.TemporaryDirectory() as tmp,configured(Path(tmp)) as base:
            root=Path(tmp);folder=root/'gate';folder.mkdir();(root/'server').mkdir()
            r,_,_,_=self.actor_fixture(folder)
            (folder/'result.json').write_text(json.dumps(r))
            manifest={'code_commit':'code','implementation_digest':launch.DIGEST,'task':0,'instance':138,
                      'task_name':'turning_on_radio','split':'train','seed':0,'training_updates':0,
                      'actor_scene_truth':False,'prefix_is_expert_not_agent':False,'diagnostic_replay_requested':False,
                      'args':launch.expected_actor_args(base),'model_identity':self.health()}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            native={'official_api_events':[{'name':n,'instance':i,'status':'completed'}
                    for n,i in [('reset',None),('load_task_instance',138),('reset',None)]]}
            (folder/'native_profile.json').write_text(json.dumps(native))
            ledger=root/'server/calls.jsonl';ledger.write_text('{"call":1}\n{"call":2}\n')
            health={**self.health(),'calls':2}
            self.assertFalse(launch.validate_completion(base,'code',health)['official_success'])
            variants=[(folder/'manifest.json',{**manifest,'instance':242}),
                      (folder/'manifest.json',{**manifest,'args':{**manifest['args'],'prefix':448}}),
                      (folder/'native_profile.json',{'official_api_events':native['official_api_events'][:2]}),
                      (folder/'result.json',{**r,'model_calls':3})]
            for path,bad in variants:
                old=path.read_bytes();path.write_text(json.dumps(bad))
                with self.subTest(path=path),self.assertRaises(ValueError):launch.validate_completion(base,'code',health)
                path.write_bytes(old)
            ledger.write_text('{"call":1}\n{"error":"timeout","calls":2}\n')
            with self.assertRaises(ValueError):launch.validate_completion(base,'code',health)
            ledger.write_text('{"call":1}\n{"call":2}\n');(folder/'failure.json').write_text('{"error":"primary"}')
            with self.assertRaisesRegex(RuntimeError,'primary'):launch.validate_completion(base,'code',health)

    def test_model_pin_checks_every_file_and_rejects_optional_loading_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();(root/'configs/semantic_robot').mkdir(parents=True)
            (model/'download_receipt.json').write_text(json.dumps({'revision':launch.REVISION,'state':'complete'}))
            (model/'weight.safetensors').write_bytes(b'fixed')
            files={p.name:{'bytes':p.stat().st_size,'sha256':launch.sha(p)} for p in model.iterdir()}
            (root/'configs/semantic_robot/h79_model_identity.json').write_text(json.dumps({'path':str(model),'revision':launch.REVISION,'files':files}))
            with patch.object(launch,'REPO',root),patch.object(launch,'MODEL',model):
                launch.model_identity()
                (model/'adapter_config.json').write_text('{}')
                with self.assertRaises(ValueError):launch.model_identity()
                (model/'adapter_config.json').unlink();(model/'weight.safetensors').write_bytes(b'other')
                with self.assertRaises(ValueError):launch.model_identity()

    def test_model_load_failure_stops_owned_child_preserves_error(self):
        with tempfile.TemporaryDirectory() as tmp,configured(Path(tmp)) as base,ExitStack() as stack:
            base.claim_stage=Mock();base.snapshot=self.snapshot;base.environment=lambda:({'SAFE':'only'},[])
            fake=Mock(pid=100,returncode=9);fake.poll.return_value=9
            stack.enter_context(patch.object(launch,'preflight',return_value=('code',self.snapshot())))
            stack.enter_context(patch.object(launch,'model_identity',return_value={'exact':True}))
            stack.enter_context(patch.object(launch,'model_environment',return_value={'SAFE':'only'}))
            stack.enter_context(patch.object(launch,'tree_bytes',return_value=0))
            stack.enter_context(patch.object(launch.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
            stack.enter_context(patch.object(launch.os,'sched_getaffinity',return_value=set(range(80))))
            proc=stack.enter_context(patch.object(launch.subprocess,'Popen',return_value=fake))
            cleanup=stack.enter_context(patch.object(launch,'stop_owned_group'))
            runtime=launch.prepare_runtime([launch.RUNTIME/'portable'])
            launch.write('launch.json',{'commands':launch.commands(base),'model_identity':{'exact':True},
                                      'runtime_preparation':runtime})
            with self.assertRaisesRegex(RuntimeError,'Model exited before readiness'):launch.supervise(base)
            self.assertEqual(proc.call_count,1);cleanup.assert_called_once_with(fake)
            self.assertTrue(proc.call_args.kwargs['start_new_session'])
            receipt=launch.read(Path(tmp)/'supervisor.json')
            self.assertEqual(receipt['status'],'failed');self.assertEqual(receipt['exit_codes'],{'model':9})
            self.assertIn('after_exit',receipt);self.assertNotIn('actor_pid',receipt)


if __name__=='__main__':unittest.main()
