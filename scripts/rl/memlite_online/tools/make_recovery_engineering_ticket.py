"""Prepare a TWO-update original-expert engineering ticket; no formal SFT."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--expert-index',type=Path,required=True)
    p.add_argument('--component',choices=('H1','L0'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    recipe=json.loads(a.recipe.read_text());root=Path(recipe['root'])
    files=dict(recipe=a.recipe.resolve(),expert_index=a.expert_index.resolve(),
               high_stats=root/'models/memlite-b-final-20260910/B-dataset-stats.json')
    ticket=dict(schema='recovery_sft_launch_v1',component=a.component,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        engineering_smoke=True,formal_training_authorized=False,maximum_updates=2,event_passes=1,
        wall_seconds=1800,world_size=8,micro_batch=1,global_batch=16,
        files={key:dict(path=str(path),sha256=file_sha(path)) for key,path in files.items()},
        scope='Original expert derivative graph/checkpoint/W&B test only; no recovery labels, no claimed policy improvement')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(ticket,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),sha256=file_sha(a.output),formal_training_authorized=False)))


if __name__=='__main__':main()
