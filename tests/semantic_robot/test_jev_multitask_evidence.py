import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semantic_robot.v2.jev_campaign import CASE_IDS, case_identity, load_registry
from semantic_robot.v2.jev_client import JevAbstained
from semantic_robot.v2.jev_policy import JevGroundedPolicy
from semantic_robot.v2.jev_control import validate_actor_ownership
from semantic_robot.v2.jev_plan_audit import validate_actor_plan
from test_jev import Opener, client
from test_jev_task_slots import TRASH, FOOD
import probe_jev_multitask as probe
from summarize_jev_campaign import summarize

REPO=Path(__file__).resolve().parents[2]
KEYS={0:['goal_0','goal_1'],1:TRASH,3:FOOD}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def lines(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def policy_trial(task,abstain=False):
    registry,_=load_registry(REPO)
    ledger=[]
    keys=['abstain'] if abstain else KEYS[task]
    p=JevGroundedPolicy(SimpleNamespace(identity={},calls=0),
        client(Opener([dict(next_goal=k) for k in keys]),journal=ledger.append),
        task_id=task,task_plan=REPO/registry['plans'][str(task)]['path'])
    try: p.plan(task,p.plan_identity['official_instruction'],None)
    except JevAbstained: pass
    value=dict(stop_reason='JEV_PLAN_ABSTAINED' if abstain else 'MODEL_REQUESTED_SAFE_STOP',
        jev_plan_ownership=p.plan_ownership,
        planning_abstention=dict(selected_goal_ids=[],call=1,question='next_goal',choice='abstain',plan_complete=False) if abstain else None)
    return p,ledger,value


class EvidenceTests(unittest.TestCase):
    def test_api_admission_accepts_and_retains_all_valid_abstentions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            identity=dict(code_commit='frozen',implementation_digest='digest',registry_sha256=load_registry(REPO)[1],
                          probe_sha256='probe',model='jev-1.13.0')
            result=dict(schema='jev-multitask-contract-validation-v1',**identity,interface_validated=True,
                dry_run=False,new_controls=0,new_resets=0,training_updates=0,trials=[],api_requests=0)
            for i,(task,repeat) in enumerate((t,r) for t in (0,1,3) for r in range(3)):
                p,ledger,value=policy_trial(task,abstain=True)
                trial=dict(task_id=task,repeat=repeat,status='abstained_no_action',plan_result=value,accounting=p.accounting())
                path=folder/f'trial_{i:02d}.json'; journal=folder/f'calls_{i:02d}.jsonl'
                write(path,trial); lines(journal,ledger)
                result['trials'].append(dict(task_id=task,repeat=repeat,status=trial['status'],
                    trial_sha256=probe.sha(path),ledger_sha256=probe.sha(journal)))
                result['api_requests']+=p.client.calls
            write(folder/'result.json',result)
            with patch.object(probe,'source_identity',return_value=identity):
                receipt=probe.validate_evidence(REPO,folder)
                self.assertEqual(receipt['valid_abstentions'],9)
                self.assertEqual(receipt['completed_plans'],0)
                self.assertTrue(receipt['interface_validated'])
                bad=copy.deepcopy(result);bad['trials'].pop();write(folder/'result.json',bad)
                with self.assertRaises(ValueError):probe.validate_evidence(REPO,folder)
                write(folder/'result.json',result)
                (folder/'calls_00.jsonl').write_text('')
                with self.assertRaises(ValueError):probe.validate_evidence(REPO,folder)

    def episode(self,parent,key,*,abstain=False):
        case=case_identity(REPO,key);task=case['task_id']
        p,ledger,value=policy_trial(task,abstain=abstain)
        decisions=[] if abstain else [dict(decision=0,stop_reason='MODEL_REQUESTED_SAFE_STOP')]
        result=dict(**value,**p.accounting(),evaluation_case=case,task=task,status='complete',
            controller='jev',prefix_controls=0,diagnostic_replay_controls=0,official_success=False,
            decisions=decisions,controls=1,implementation_digest='digest')
        ticks=[dict(control=1,safety_stop=True)]
        ownership=validate_actor_ownership(result,ledger,ticks)
        ownership['registered_plan']=validate_actor_plan(result,REPO/case['task_plan']['path'],task,ledger)
        folder=parent/('jev_multi_20260930_'+key+'_v1')
        common=dict(stage=key,campaign='multitask',source_commit='frozen',implementation_digest='digest')
        write(folder/'launch.json',common)
        write(folder/'supervisor.json',dict(**common,status='completed',exit_codes={'actor':0},
            after_exit={'fixture':True},official_success=False))
        write(folder/'gate/result.json',result)
        write(folder/'gate/manifest.json',dict(evaluation_case=case,task=task,instance=case['instance'],
            implementation_digest='digest',code_commit='frozen',args=dict(jev_eval_case=key,controller='jev',prefix=0,
                max_decisions=96,max_controls=3072,max_seconds=2400,jev_max_calls=288)))
        write(folder/'decision_ownership.json',ownership)
        lines(folder/'gate/jev_calls.jsonl',ledger);lines(folder/'gate/steps.jsonl',ticks)
        return folder

    def test_all_registered_failures_count_no_old_or_missing_case_is_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            summary=summarize(root)
            self.assertEqual(summary['overall']['pending'],6)
            self.assertIsNone(summary['overall']['success_rate'])
            for i,key in enumerate(CASE_IDS): self.episode(root,key,abstain=i%2==0)
            summary=summarize(root)
            self.assertEqual(summary['overall']['completed'],6)
            self.assertEqual(summary['overall']['success_rate'],0)
            self.assertEqual([r['registered'] for r in summary['per_task'].values()],[2,2,2])
            self.assertFalse(any(r['control_integration_validated'] for r in summary['cases']))
            folder=root/'jev_multi_20260930_task1a_v1'
            bad=json.loads((folder/'gate/manifest.json').read_text());bad['args']['max_controls']=2048
            write(folder/'gate/manifest.json',bad)
            summary=summarize(root)
            self.assertEqual(summary['overall']['infrastructure_failures'],1)
            self.assertEqual(summary['overall']['completed'],5)
            self.assertIsNone(summary['overall']['success_rate'])


if __name__=='__main__': unittest.main()
