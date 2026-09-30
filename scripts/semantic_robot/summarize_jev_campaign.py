"""Report every preregistered slot, never a successful-episode subset."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from semantic_robot.v2.jev_campaign import CASE_IDS, case_identity, load_registry
from semantic_robot.v2.jev_control import validate_actor_ownership
from semantic_robot.v2.jev_plan_audit import validate_actor_plan
from launch_jev_closedloop import validate_jev_ledger


def read(path): return json.loads(path.read_text())


def summarize(parent, repo=REPO):
    parent,repo=Path(parent),Path(repo)
    registry,registry_sha=load_registry(repo)
    cases=[]; identities=set()
    for key in CASE_IDS:
        expected=case_identity(repo,key)
        folder=parent/(registry['campaign_id']+'_'+key+'_v1')
        row=dict(case_id=key,task_id=expected['task_id'],instance=expected['instance'],status='not_submitted',
                 official_success=None,output=str(folder))
        cases.append(row)
        if not folder.exists(): continue
        row['status']='pending'
        if not (folder/'supervisor.json').exists(): continue
        ended=read(folder/'supervisor.json')
        if ended.get('status')=='failed':
            row.update(status='infrastructure_failure',error=ended.get('error'));continue
        if ended.get('status')!='completed' or 'exit_codes' not in ended: continue
        try:
            result=read(folder/'gate/result.json'); manifest=read(folder/'gate/manifest.json')
            submitted=read(folder/'launch.json')
            if (result.get('evaluation_case')!=expected or manifest.get('evaluation_case')!=expected
                    or ended.get('stage')!=key or ended.get('campaign')!='multitask'
                    or submitted.get('stage')!=key or submitted.get('campaign')!='multitask'
                    or ended.get('exit_codes',{}).get('actor')!=0 or not ended.get('after_exit')
                    or manifest.get('task')!=expected['task_id'] or manifest.get('instance')!=expected['instance']
                    or result.get('controller')!='jev' or result.get('status')!='complete'
                    or type(result.get('official_success')) is not bool
                    or ended.get('official_success')!=result['official_success']
                    or any(result.get(k)!=0 for k in ('prefix_controls','diagnostic_replay_controls'))):
                raise ValueError('Case/outcome identity mismatch')
            args=manifest['args']
            if (args.get('jev_eval_case')!=key or args.get('controller')!='jev' or args.get('prefix')!=0
                    or (args.get('max_decisions'),args.get('max_controls'),args.get('max_seconds'),args.get('jev_max_calls'))!=(96,3072,2400,288)):
                raise ValueError('Unequal episode budget/reset protocol')
            digest=result['implementation_digest'];code=manifest['code_commit']
            if any(v.get('implementation_digest')!=digest for v in (manifest,submitted,ended)) or any(
                    v.get('source_commit')!=code for v in (submitted,ended)):
                raise ValueError('Episode source identity mismatch')
            identities.add((code,digest))
            ledger=[json.loads(x) for x in (folder/'gate/jev_calls.jsonl').read_text().splitlines()]
            ticks=[json.loads(x) for x in (folder/'gate/steps.jsonl').read_text().splitlines()]
            validate_jev_ledger(ledger,result)
            ownership=validate_actor_ownership(result,ledger,ticks)
            ownership['registered_plan']=validate_actor_plan(result,repo/expected['task_plan']['path'],expected['task_id'],ledger)
            if read(folder/'decision_ownership.json')!=ownership:
                raise ValueError('Ownership receipt mismatch')
            row.update(status='completed',official_success=result['official_success'],stop_reason=result['stop_reason'],
                controls=result['controls'],decisions=len(result['decisions']),jev_requests=result['jev_requests'],
                observer_requests=result['observer_calls'],control_integration_validated=ownership['control_integration_validated'],
                moving_actions=sum(r.get('accepted_before_motion') is True and r.get('action',{}).get('move')!='hold'
                                   for r in result['decisions']),
                jev_owned_actions=ownership['jev_owned_actions'])
        except (ValueError,KeyError,RuntimeError,OSError) as error:
            row.update(status='invalid_evidence',error=str(error))
    if len(identities)>1: raise ValueError('Do not mix source versions in one campaign rate')
    def counts(rows):
        done=[r for r in rows if r['status']=='completed']
        success=sum(r['official_success'] for r in done)
        return dict(registered=len(rows),completed=len(done),successes=success,
                    success_rate=success/len(rows) if len(done)==len(rows) else None,
                    infrastructure_failures=sum(r['status'] in ('infrastructure_failure','invalid_evidence') for r in rows),
                    pending=sum(r['status'] in ('pending','not_submitted') for r in rows))
    return dict(schema='jev-multitask-results-v1',registry_sha256=registry_sha,split='TRAIN development',seed=0,
                per_task={str(t):counts([r for r in cases if r['task_id']==t]) for t in (0,1,3)},
                overall=counts(cases),cases=cases,source_identities=[dict(commit=c,digest=d) for c,d in sorted(identities)],
                excludes_engineering_gates_and_old_jev04=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True)
    args=parser.parse_args()
    print(json.dumps(summarize(args.parent),ensure_ascii=False,indent=2,allow_nan=False))
