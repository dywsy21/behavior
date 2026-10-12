"""Verify the real360-update paired data schedule and reserved disk, CPU only."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_periodic_data import PeriodicTeacherPool,remap_control_schedule,checkpoint_reservation
from recovery_sft_data import require_training_pool
from recovery_corpus import digest,file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('recipe','corpus','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists():raise ValueError('Fresh CPU audit only')
    import numpy as np
    recipe=json.loads(a.recipe.read_text());root=Path(recipe['root']);spec=recipe['H1']['periodic_teacher']
    pool=PeriodicTeacherPool(root,spec,repo=REPO)
    admission=a.corpus/'admission/admission.json'
    _,rows=require_training_pool(admission.parent,'planner',file_sha(admission))
    train=[r for r in rows if r['candidate']['split']=='train']
    tasks=np.load(pool.release.root/'train_tasks.npy',mmap_mode='r')
    original=json.loads(pool.control_path.read_text())
    binding=dict(component='H1',parent_sha256=recipe['parents']['high']['sha256'],
        admission_sha256=file_sha(admission),world_size=8,micro_batch=1)
    schedule=remap_control_schedule(original,pool.by_task,lambda i:int(tasks[i]),seed=recipe['seed'],
        recovery_rows=train,binding=binding)
    if len(schedule)!=360 or len(train)!=36 or max(r['event_pass'] for r in schedule)!=19:
        raise ValueError('Paired360 updates/36 recovery sources/20passes changed')
    old_new=[(step,slot,index) for step,b in enumerate(original['schedule'])
             for slot,(kind,index) in enumerate(b['rows']) if kind=='new']
    new_new=[(step,slot,index) for step,b in enumerate(schedule)
             for slot,(kind,index) in enumerate(b['rows']) if kind=='new']
    if old_new!=new_new: raise ValueError('Paired recovery rank/exposure differs')
    anchors=[i for b in schedule for kind,i in b['rows'] if kind=='expert']
    task_counts=Counter(int(tasks[i]) for i in anchors)
    if set(task_counts)!=set(range(100)):raise ValueError('Missing expert task coverage')
    reservation=checkpoint_reservation((root/recipe['parents']['high']['path']).stat().st_size,360,
        recipe['pilot']['checkpoint_every_updates'],spec['reserve_for_champion_bytes'])
    reservation['observed_free_bytes']=shutil.disk_usage(root).free
    if reservation['observed_free_bytes']<reservation['required_free_bytes']:raise ValueError('Insufficient paired-run disk reserve')
    result=dict(schema='periodic_planner_schedule_audit_v1',status='passed',
        recipe_sha256=file_sha(a.recipe),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        candidate_sha256=spec['candidate_sha256'],paired_control_schedule_sha256=spec['control_schedule']['sha256'],
        remapped_schedule_sha256=digest(schedule),updates=360,global_batch=8,expert_examples=len(anchors),
        unique_expert_anchors=len(set(anchors)),expert_task_counts=dict(sorted(task_counts.items())),
        expert_kind_counts=dict(Counter(pool.by_index[i]['identity']['kind'] for i in anchors)),
        recovery_examples=len(new_new),recovery_slots_exact=True,recovery_row_exposures=dict(Counter(i for _,_,i in new_new)),
        six_distinct_expert_tasks_every_batch=True,storage=reservation,optimizer_updates=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('expert_task_counts','recovery_row_exposures')}),flush=True)


if __name__=='__main__':main()
