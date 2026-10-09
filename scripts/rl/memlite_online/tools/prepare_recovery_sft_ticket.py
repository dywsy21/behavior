"""CPU-only P2 readiness ticket. This tool NEVER starts training.

Model-graph and data gates are separate. Even a ready data pool still requires
a reviewed trainer/OOF/calibration integration receipt before formal launch.
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
    result=dict(schema='recovery_p2_preparation_ticket_v1',recipe_sha256=file_sha(args.recipe),
        admission_sha256=args.admission_sha256,node=recipe['preferred_node'],parents=recipe['parents'],
        data_gates=gates,execution_ready=False,optimizer_steps=0,
        remaining=['Accepted per-sample outcome/planner/action evidence; no whole-clip blanket approval',
                   'H0 member-conditioned feature cache and held-out calibration',
                   'H1 out-of-fold feedback aligned with REAL verified continuation',
                   'Full derivative trainer checkpoint/DDP/objective + per-run W&B acceptance',
                   'Fixed short TRAIN-dev legal start-state/replay acceptance; not RGB-only restoration'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
