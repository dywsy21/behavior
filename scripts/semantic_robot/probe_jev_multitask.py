"""Finite text-only contract probe. Valid abstention is NOT an admission failure.

Three trials per task, no repeats/replacements, at most 54 requests. This tests
the exact live planning path, not task success or consistency of a preferred
plan. No simulator, GPU, observation, training or action execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from semantic_robot.v2.jev_campaign import load_registry
from semantic_robot.v2.jev_client import JevClient, JevAbstained, MODEL
from semantic_robot.v2.jev_policy import JevGroundedPolicy
from semantic_robot.v2.jev_plan_audit import validate_actor_plan


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def source_identity(repo):
    from run_v2 import implementation_digest
    if Path(repo).resolve() != REPO: raise ValueError('Probe uses its own frozen source')
    code = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'], text=True).strip():
        raise ValueError('Freeze clean source before API validation')
    _, registry_sha = load_registry(repo)
    return dict(code_commit=code, implementation_digest=implementation_digest(),
                registry_sha256=registry_sha, probe_sha256=sha(__file__), model=MODEL)


def validate_evidence(repo, folder):
    """Validate all nine trials, including rejected plans, against current source."""
    from launch_jev_closedloop import validate_jev_ledger
    folder = Path(folder)
    result = json.loads((folder/'result.json').read_text())
    identity = source_identity(repo)
    registry, _ = load_registry(repo)
    if (result.get('schema') != 'jev-multitask-contract-validation-v1'
            or any(result.get(k) != v for k,v in identity.items())
            or result.get('interface_validated') is not True or result.get('error')
            or result.get('dry_run') is not False
            or any(type(result.get(k)) is not int or result[k] != 0 for k in
                   ('new_controls','new_resets','training_updates'))):
        raise ValueError('Multi-task API evidence is incomplete or belongs to another source')
    rows = result.get('trials', [])
    if [(r.get('task_id'),r.get('repeat')) for r in rows] != [(t,i) for t in (0,1,3) for i in range(3)]:
        raise ValueError('Missing/reordered contract trials; do not select successful plans')
    requests = 0
    for i, row in enumerate(rows):
        path = folder/f'trial_{i:02d}.json'
        ledger_path = folder/f'calls_{i:02d}.jsonl'
        if row.get('trial_sha256') != sha(path) or row.get('ledger_sha256') != sha(ledger_path):
            raise ValueError('API trial/ledger bytes changed')
        trial = json.loads(path.read_text())
        if (trial.get('task_id') != row['task_id'] or trial.get('repeat') != row['repeat']
                or trial.get('status') not in ('complete','abstained_no_action')
                or trial.get('status') != row.get('status')):
            raise ValueError('Invalid contract trial outcome')
        ledger = [json.loads(x) for x in ledger_path.read_text().splitlines()]
        validate_jev_ledger(ledger, trial['accounting'])
        validate_actor_plan(trial['plan_result'], Path(repo)/registry['plans'][str(row['task_id'])]['path'],
                            row['task_id'], ledger)
        n = trial['accounting']['jev_requests']
        goal_count = {0:2,1:6,3:10}[row['task_id']]
        if not 1 <= n <= goal_count or trial['accounting']['jev_requests_without_validated_response'] != 0:
            raise ValueError('Contract trial call accounting/budget')
        requests += n
    if result.get('api_requests') != requests or not 9 <= requests <= 54:
        raise ValueError('Contract batch call accounting')
    return dict(path=str(folder/'result.json'), sha256=sha(folder/'result.json'),
                interface_validated=True, trials=9, api_requests=requests,
                completed_plans=sum(r['status']=='complete' for r in rows),
                valid_abstentions=sum(r['status']=='abstained_no_action' for r in rows),
                no_success_conditioned_admission=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--key-file', required=True)
    args = parser.parse_args()
    if 'TYPESAFE_API_KEY' in os.environ: raise ValueError('Use private key file only')
    identity = source_identity(REPO)
    registry, _ = load_registry(REPO)
    folder = Path(args.output); folder.mkdir(parents=True, exist_ok=False)
    result = dict(schema='jev-multitask-contract-validation-v1', **identity, interface_validated=False,
                  trials=[], dry_run=False, api_requests=0, new_controls=0, new_resets=0, training_updates=0,
                  max_requests=54, wall_seconds=1200, global_api_cap=None,
                  not_a_success_rate_evaluation=True)
    deadline = time.perf_counter()+1200
    try:
        for index, (task, repeat) in enumerate((t,i) for t in (0,1,3) for i in range(3)):
            path, journal_path = folder/f'trial_{index:02d}.json', folder/f'calls_{index:02d}.jsonl'
            with journal_path.open('x') as stream:
                def journal(row):
                    stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                client = JevClient(key_file=args.key_file, max_calls={0:2,1:6,3:10}[task], journal=journal)
                client.deadline = deadline
                policy = JevGroundedPolicy(SimpleNamespace(identity={},calls=0), client, task_id=task,
                    task_plan=REPO/registry['plans'][str(task)]['path'])
                policy.deadline = deadline
                try:
                    policy.plan(task, policy.plan_identity['official_instruction'], None)
                    status = 'complete'
                    plan_result = dict(stop_reason='PLAN_COMPLETE', jev_plan_ownership=policy.plan_ownership)
                except JevAbstained:
                    status = 'abstained_no_action'
                    selected = policy.last_call['result']['selected_goal_ids']
                    plan_result = dict(stop_reason='JEV_PLAN_ABSTAINED', jev_plan_ownership=None,
                        planning_abstention=dict(selected_goal_ids=selected, call=client.calls,
                            question='next_goal', choice='abstain', plan_complete=False))
                trial = dict(task_id=task,repeat=repeat,status=status,plan_result=plan_result,
                             accounting=policy.accounting(),receipt=policy.last_call)
                write(path, trial)
            row = dict(task_id=task,repeat=repeat,status=status,trial_sha256=sha(path),ledger_sha256=sha(journal_path))
            result['trials'].append(row); result['api_requests'] += client.calls
            print(json.dumps(row),flush=True)
        result['interface_validated'] = True
        write(folder/'result.json', result)
        validate_evidence(REPO, folder)
    except BaseException as error:
        result.update(interface_validated=False,error=str(error))
        raise
    finally:
        write(folder/'result.json', result)
        print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
