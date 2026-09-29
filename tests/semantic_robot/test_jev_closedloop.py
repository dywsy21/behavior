import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/semantic_robot'))
import launch_jev_closedloop as launch
from native_full_profile import configure_gpu


class ClosedLoopTests(unittest.TestCase):
    def test_valid_abstention_ends_without_fabricating_an_action(self):
        from run_v2 import select_feasible_action
        from semantic_robot.v2.jev_client import JevAbstained,JevError
        manager=SimpleNamespace(stop_reason=None)
        def abstain(*args): raise JevAbstained('no command selected')
        policy=SimpleNamespace(act_feasible=abstain,last_call={'result':{'choice':'abstain'}})
        action,receipt=select_feasible_action(policy,manager,None,None,())
        self.assertIsNone(action)
        self.assertEqual(manager.stop_reason,'JEV_ABSTAINED')
        self.assertEqual(receipt,policy.last_call)
        def failed(*args):raise JevError('network failure')
        policy.act_feasible=failed
        with self.assertRaises(JevError):select_feasible_action(policy,manager,None,None,())

    def test_wall_receipt_finite_and_bounded(self):
        for actor in (False,True):
            launch.validate_wall(0.,actor)
            launch.validate_wall(2405 if actor else 1205,actor)
            for bad in (None,True,float('nan'),float('inf'),-1,2406 if actor else 1206):
                with self.assertRaises(ValueError):launch.validate_wall(bad,actor)

    def test_inherited_key_environment_rejected_before_any_other_action(self):
        with patch.dict(os.environ,{'TYPESAFE_API_KEY':'dummy-do-not-inherit'}):
            with self.assertRaisesRegex(ValueError,'inherited TypeSafe'):launch.preflight(None)

    def test_separate_durable_jev_ledger(self):
        rows=[dict(call=1,model='jev-1.13.0',event='attempt'),
            dict(call=1,model='jev-1.13.0',event='validated',usage=dict(input_tokens=7,output_tokens=2))]
        result=dict(jev_requests=1,jev_input_tokens=7,jev_output_tokens=2)
        launch.validate_jev_ledger(rows,result)
        for bad in (rows[:1],rows[::-1],rows+rows):
            with self.assertRaises(ValueError):launch.validate_jev_ledger(bad,result)
        with self.assertRaises(ValueError):launch.validate_jev_ledger(rows,{**result,'jev_input_tokens':8})

    def setUp(self):
        self.uuids = launch.gate.scene.supervisor.GPU_UUIDS
        self.before = {u:dict(used_mib=400 if i==1 else 8000,free_mib=80738 if i==1 else 72000,
            processes=[dict(pid=11,used_mib=200),dict(pid=12,used_mib=200)] if i==1 else [])
            for i,u in enumerate(self.uuids)}

    def test_gpu_selection_updates_app_physics_runtime_and_env(self):
        base=SimpleNamespace(GPU_UUIDS=self.uuids,PROFILE_APP_CONFIG={},RUNTIME_SETTINGS={},FIXED_ENV={})
        with patch.dict(os.environ,{},clear=True):
            configure_gpu(base,1)
            self.assertEqual(base.PROFILE_APP_CONFIG,dict(active_gpu=1,physics_gpu=1))
            self.assertEqual(base.RUNTIME_SETTINGS['/renderer/activeGpu'],1)
            self.assertEqual(base.FIXED_ENV['OMNIGIBSON_GPU_ID'],'1')
            for bad in (True,0,2,-1,4):
                with self.assertRaises(ValueError): configure_gpu(base,bad)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}):
            with self.assertRaises(ValueError): configure_gpu(base,1)

    def test_external_jobs_preserved_but_no_large_gpu1_colocation(self):
        launch.check_resources(self.before)
        changed=copy.deepcopy(self.before);changed[self.uuids[1]]['processes'][0]['used_mib']=600
        with self.assertRaises(RuntimeError): launch.check_resources(changed)
        with self.assertRaises(RuntimeError): launch.check_resources(changed,self.before)
        changed=copy.deepcopy(self.before);changed[self.uuids[1]]['processes'].append(dict(pid=999,used_mib=1))
        with self.assertRaises(RuntimeError): launch.check_resources(changed,self.before)
        changed=copy.deepcopy(self.before);changed[self.uuids[0]]['processes'].append(dict(pid=999,used_mib=9000))
        launch.check_resources(changed,self.before)  # Team GPU may change normally.

    def test_own_memory_caps_and_cleanup(self):
        current=copy.deepcopy(self.before)
        current[self.uuids[1]]['processes'] += [dict(pid=101,used_mib=55000),dict(pid=102,used_mib=8000)]
        current[self.uuids[1]]['free_mib']=17700
        with patch.object(launch,'belongs_to_session',side_effect=lambda pid,sid:pid==sid):
            launch.check_resources(current,self.before,dict(model=101,actor=102))
            with self.assertRaises(RuntimeError): launch.check_resources(current,self.before,dict(model=101,actor=102),released=True)
            current[self.uuids[1]]['processes'][-1]['used_mib']=14337
            with self.assertRaises(RuntimeError): launch.check_resources(current,self.before,dict(model=101,actor=102))

    def test_exact_commands_and_defaults_for_both_modes(self):
        base=SimpleNamespace(PYTHON=Path('/python'),REPO=launch.REPO)
        with patch.object(launch.gate,'FLAGS',launch.FLAGS):
            for stage in launch.STAGES:
                root=launch.stage_root(stage)
                with patch.multiple(launch,STAGE=stage,ROOT=root,RUNTIME=launch.RUNTIME_PARENT/root.name):
                    args=launch.expected_args(base)
                    self.assertEqual(args['gpu'],1)
                    self.assertFalse(args['structured_planning'])
                    self.assertEqual(args['prefix'],0)
                    self.assertTrue(args['synchronous_io_v1'])
                    if stage.startswith('episode'):
                        self.assertEqual(args['controller'],'jev')
                        self.assertEqual(args['jev_max_calls'],208)
                        self.assertEqual(len(args['gate_result']),2)
                    else:
                        self.assertEqual(args['controller'],'vlm')
                        self.assertIsNone(args['typesafe_key_file'])
                        self.assertEqual(args['task'],int(stage[-1]))


if __name__=='__main__': unittest.main()
