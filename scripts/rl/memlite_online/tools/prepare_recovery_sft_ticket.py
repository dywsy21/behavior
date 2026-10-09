"""CPU-only P2 readiness ticket. This tool NEVER starts training.

Model-graph, data, actual training and deployment gates are separate. This
summary never grants permission or requires an already-trained H0 before H0
itself can be proposed. It may incorporate bounded engineering run receipts.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha  # noqa: E402
from recovery_sft_data import require_training_pool  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True)
    parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--engineering-evidence',type=Path)
    parser.add_argument('--accepted-data-evidence',type=Path,
        help='SHA-pinned actual data/feature/start checks, separate from training authorization')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    recipe=json.loads(args.recipe.read_text())
    if (recipe['schema']!='memlite_recovery_preparation_recipe_v1'
            or recipe['maximum_updates_per_line']>1000 or recipe['maximum_event_passes']>5
            or recipe['maximum_wall_seconds_per_line']>14400 or not recipe['stop_on_first_limit']):
        raise ValueError('Unexpected recipe or expanded pilot budget')
    gates={}
    for line in ('H0','H1','L0'):
        pool=recipe[line]['pool']
        try:
            receipt,rows=require_training_pool(args.admission,pool,args.admission_sha256)
            gates[line]=dict(data_ready=True,rows=len(rows),coverage=receipt['pools'][pool]['coverage'])
        except ValueError as error:
            gates[line]=dict(data_ready=False,reason=str(error))
    engineering={}
    if args.engineering_evidence:
        for line,parent in (('L0','low'),('H1','high')):
            path=args.engineering_evidence/(line.lower()+'-trainer-v1-audit.json')
            evidence=json.loads(path.read_text())
            if (evidence['status']!='passed_cpu_reload' or evidence['component']!=line
                    or evidence['parent_sha256']!=recipe['parents'][parent]['sha256']
                    or evidence['adam_step']!=2 or evidence['rng_ranks']!=8 or evidence['formal_training']):
                raise ValueError('Wrong bounded training-chain receipt')
            engineering[line]=dict(training_chain_verified=True,evidence=str(path),sha256=file_sha(path))
        for key,name,status in (('H0_cache','h0-feature-cache-v1/receipt.json',None),
                                 ('H1_processor','feedback-processor-v1.json','passed_cpu_processor_contract')):
            path=args.engineering_evidence/name;evidence=json.loads(path.read_text())
            if evidence['optimizer_steps']!=0 or (status and evidence['status']!=status):
                raise ValueError('Wrong non-learning interface receipt')
            if key=='H0_cache' and (not evidence['diagnostic_only'] or evidence['high_sha256']!=recipe['parents']['high']['sha256']):
                raise ValueError('Diagnostic feature parent mismatch')
            engineering[key]=dict(interface_verified=True,diagnostic_only=True,evidence=str(path),sha256=file_sha(path))
    data_checks={}
    if args.accepted_data_evidence:
        accepted=json.loads(args.accepted_data_evidence.read_text())
        if (accepted['schema']!='recovery_accepted_data_preflight_v1'
                or accepted['admission_sha256']!=args.admission_sha256
                or accepted['optimizer_steps']!=0 or accepted['formal_training_authorized']):
            raise ValueError('Wrong accepted data preflight')
        for name,ref in accepted['checks'].items():
            path=Path(ref['path'])
            if file_sha(path)!=ref['sha256']:raise ValueError('Changed data preflight evidence: '+name)
            evidence=json.loads(path.read_text())
            if evidence.get('status') not in ref['accepted_statuses']:raise ValueError('Unpassed data gate: '+name)
            data_checks[name]=dict(verified=True,path=str(path),sha256=ref['sha256'])
        if set(data_checks)!={'full_read','processor','features','start_states','transfer'}:
            raise ValueError('Missing actual data/start/cache check')
        data_checks['receipt_sha256']=file_sha(args.accepted_data_evidence)
    remaining=[]
    if not all(v['data_ready'] for v in gates.values()):
        remaining.append('Accepted per-sample outcome/planner/action evidence; no whole-clip blanket approval')
    if not data_checks:remaining.append('Actual admitted-data processor, frozen causal cache, transfer and legal start checks')
    if not engineering:remaining.append('Full derivative trainer checkpoint/DDP/objective + per-run W&B acceptance')
    result=dict(schema='recovery_p2_preparation_ticket_v3',recipe_sha256=file_sha(args.recipe),
        admission_sha256=args.admission_sha256,node=recipe['preferred_node'],parents=recipe['parents'],
        data_gates=gates,engineering=engineering,accepted_data_checks=data_checks,
        technical_preparation_complete=not remaining,execution_ready=False,optimizer_steps=0,
        formal_training_authorized=False,remaining_preparation=remaining,
        separate_launch_requirements=['Independent team code review before merge; not performed by this single-agent preparation',
                                     'Explicit formal training run ticket; preparation is not authorization'],
        later_training_and_deployment=['Fit H0 on accepted member-conditioned features; held-group readiness calibration',
          'Generate actual OOF feedback aligned with verified H1 continuation; train each admitted line',
          'Revalidate observer if H1 backbone changes, integrate serving, measure overhead and closed-loop effects'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
