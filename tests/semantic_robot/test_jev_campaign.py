import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semantic_robot.v2.jev_campaign import (BUDGET, CASE_IDS, REGISTRY, ROBOT_SHA,
    load_registry, validate_registry, case_identity, validate_window)
from semantic_robot.v2.jev_client import JevError
from semantic_robot.v2.jev_plan_audit import validate_actor_plan
from semantic_robot.v2.jev_control import validate_actor_ownership
from test_jev import client, Opener
from semantic_robot.v2.jev_policy import JevGroundedPolicy
import launch_jev_closedloop as launch

REPO = Path(__file__).resolve().parents[2]


class CampaignTests(unittest.TestCase):
    def test_six_fixed_distinct_train_instances_with_matching_plan_bytes(self):
        registry, sha = load_registry(REPO)
        self.assertEqual(sha, hashlib.sha256((REPO/REGISTRY).read_bytes()).hexdigest())
        self.assertEqual([c['instance'] for c in registry['cases']], [138,97,141,190,242,102])
        for key in CASE_IDS:
            case = case_identity(REPO,key)
            self.assertEqual(case['budget'],BUDGET)
            self.assertEqual(case['split'],'train'); self.assertEqual(case['seed'],0)
            with self.assertRaises(ValueError): case_identity(REPO,key,99)
        for mutate in (lambda r:r['cases'].pop(),lambda r:r['cases'].reverse(),
                       lambda r:r['cases'][1].update(instance=138),
                       lambda r:r['cases'][2].update(task_name='turning_on_radio'),
                       lambda r:r['cases'][2].update(window_name='../outside'),
                       lambda r:r.update(expert_prefix_controls=True),
                       lambda r:r['budget'].update(max_decisions=100),
                       lambda r:r['plans']['3'].update(path='/outside')):
            bad=copy.deepcopy(registry); mutate(bad)
            with self.assertRaises(ValueError): validate_registry(bad)

    def test_window_bytes_and_semantic_identity_both_checked(self):
        case=case_identity(REPO,'task1a')
        value=dict(kind='native_oracle_low_window',immutable=True,window_id=case['window_name'],
                   task_name=case['task_name'],instance_id=141,official_mode='train',seed=0,
                   robot_config_sha256=ROBOT_SHA)
        raw=json.dumps(value).encode(); fixture={**case,'window_sha256':hashlib.sha256(raw).hexdigest()}
        validate_window(fixture,raw)
        with self.assertRaises(ValueError): validate_window(fixture,raw+b' ')
        for key,bad in (('instance_id',190),('seed',True),('official_mode','public_test'),
                        ('robot_config_sha256','wrong'),('immutable',1),('task_name','wrong')):
            raw=json.dumps({**value,key:bad}).encode()
            with self.assertRaises(ValueError):
                validate_window({**fixture,'window_sha256':hashlib.sha256(raw).hexdigest()},raw)

    def test_new_commands_equal_real_parser_and_uniform_budget(self):
        source=REPO/'scripts/semantic_robot/run_v2.py'
        main=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='parse_args')
        code=compile(ast.fix_missing_locations(ast.Module(body=main.body[:end+1],type_ignores=[])),str(source),'exec')
        base=SimpleNamespace(PYTHON=Path('/python'),REPO=REPO)
        for stage in launch.MULTI_STAGES:
            with patch.multiple(launch,CAMPAIGN='multitask',STAGE=stage):
                root=launch.stage_root(stage)
                with patch.multiple(launch,ROOT=root,RUNTIME=launch.RUNTIME_PARENT/root.name), \
                        patch.object(launch.gate,'FLAGS',launch.FLAGS), \
                        patch.object(sys,'argv',launch.commands(base)['actor'][1:]):
                    ns={'argparse':argparse};exec(code,ns)
                    args=launch.expected_args(base)
                    self.assertEqual(vars(ns['args']),args)
                    if stage in CASE_IDS:
                        self.assertEqual(args['jev_eval_case'],stage)
                        self.assertEqual((args['max_decisions'],args['max_controls'],args['jev_max_calls']),(96,3072,288))
                        self.assertEqual(args['jev_task_plan'],str(REPO/case_identity(REPO,stage)['task_plan']['path']))
                        self.assertIn('215',launch.commands(base)['model'])
                    else: self.assertIsNone(args['jev_eval_case'])

    def test_missing_complete_proof_or_illegal_abstained_prefix_rejected(self):
        path=REPO/'configs/semantic_robot/jev_task1_slots.json'
        with self.assertRaises(JevError): validate_actor_plan(dict(stop_reason='JEV_ABSTAINED'),path,1)
        result=dict(stop_reason='JEV_PLAN_ABSTAINED',jev_plan_ownership=None,
                    planning_abstention=dict(selected_goal_ids=['goal_0_right','goal_1_right']))
        ledger=[dict(event='validated',call=i+1,selections=dict(next_goal=k))
                for i,k in enumerate(['goal_0_right','goal_1_right','abstain'])]
        self.assertTrue(validate_actor_plan(result,path,1,ledger)['legal_abstention_prefix_checked'])
        for keys in (['goal_1_right'],['goal_0_right','goal_1_left'],['goal_0_right','goal_2_left']):
            bad=copy.deepcopy(result);bad['planning_abstention']['selected_goal_ids']=keys
            with self.assertRaises((JevError,ValueError)): validate_actor_plan(bad,path,1,ledger)

    def test_action_abstention_still_bound_to_selected_goal(self):
        ledger=[]
        p=JevGroundedPolicy(SimpleNamespace(calls=0,identity={}),client(Opener([
            {'next_goal':'goal_0'},{'next_goal':'goal_1'}]),journal=ledger.append),task_id=0,
            task_plan=REPO/'configs/semantic_robot/jev_task0_plan.json')
        p.plan(0,p.plan_identity['official_instruction'],None)
        result=dict(stop_reason='JEV_ABSTAINED',jev_plan_ownership=p.plan_ownership,
            jev_requests=4,controls=1,prefix_controls=0,diagnostic_replay_controls=0,
            decisions=[dict(decision=0,accepted_before_motion=False,stop_reason='JEV_ABSTAINED',
                selection_source='Jev_all_world_search',jev_goal_authority=p.goal_authority(0,p.planned_goals[0]),
                jev_abstention=dict(call=4,question='command',choice='abstain'))])
        ledger.append(dict(call=4,event='validated',selections=dict(command='abstain')))
        validate_actor_ownership(result,ledger,[dict(control=1,safety_stop=True)])
        for tamper in ('missing','hand'):
            bad=copy.deepcopy(result)
            if tamper=='missing': del bad['decisions'][0]['jev_goal_authority']
            else: bad['decisions'][0]['jev_goal_authority']['goal']['hand']='left'
            with self.assertRaises(JevError): validate_actor_ownership(bad,ledger)

    def test_initial_safety_veto_is_a_failure_not_an_unowned_control(self):
        ledger=[]
        p=JevGroundedPolicy(SimpleNamespace(calls=0,identity={}),client(Opener([
            {'next_goal':'goal_0'},{'next_goal':'goal_1'}]),journal=ledger.append),task_id=0,
            task_plan=REPO/'configs/semantic_robot/jev_task0_plan.json')
        p.plan(0,p.plan_identity['official_instruction'],None)
        result=dict(stop_reason='MODEL_REQUESTED_SAFE_STOP',jev_plan_ownership=p.plan_ownership,
            jev_requests=2,controls=1,prefix_controls=0,diagnostic_replay_controls=0,
            decisions=[dict(decision=0,stop_reason='MODEL_REQUESTED_SAFE_STOP')])
        ticks=[dict(control=1,safety_stop=True)]
        audit=validate_actor_ownership(result,ledger,ticks)
        self.assertFalse(audit['control_integration_validated'])
        self.assertTrue(audit['safety_stopped_without_policy_action'])
        with self.assertRaises(JevError): validate_actor_ownership(result,ledger,[dict(control=1,decision=0)])
        with self.assertRaises(JevError): validate_actor_ownership({**result,'controls':2},ledger)


if __name__=='__main__': unittest.main()
