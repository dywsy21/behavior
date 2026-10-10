"""Real TRAIN processor + unchanged event/expert/rank sampler acceptance.

CPU only. The supplement is still learner correction, never an expert archive.
The launch contract consumes this pinned receipt separately from the existing
all-row admission processor audit. Original DEV is never extended.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import bound,read,write_new
from recovery_corpus import digest,file_sha
from recovery_native_actions import NativeLearnerActionReader,SameEventLearnerSupplement
from recovery_sft_data import VerifiedRecoveryActionDataset,finite_mixture_schedule


def project(schedule,rows):
    return [dict(event_pass=b['event_pass'],new_events=b['new_events'],rows=[
        (kind,(rows[i]['candidate']['source_group'],rows[i]['approval']['event_id'])) if kind=='new'
        else (kind,i) for kind,i in b['rows']]) for b in schedule]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('recipe','corpus','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Frozen clean source and new output required')
    import torch
    from g05.utils.training.stage1_model import configuration
    torch.set_num_threads(2)
    recipe=read(a.recipe);root=Path(recipe['root']);spec=recipe['L0']['native_learner_supplement']
    manifest=bound(root,spec['manifest']);bound(root,spec['owner_review'])
    reader=NativeLearnerActionReader(manifest.parent,spec['manifest']['sha256'],spec['owner_review']['sha256'])
    names=read(root/recipe['expert_release']/'manifest.json')['task_names']
    config=configuration(root,'low',names)
    if file_sha(config['stats_path'])!=recipe['stats_sha256']:raise ValueError('Changed low stats')
    admission=a.corpus/'admission';admission_sha=file_sha(admission/'admission.json')
    base=VerifiedRecoveryActionDataset(admission,a.corpus/'raw',read(a.corpus/'audit/inventory.json'),
        config,split='train',admission_sha256=admission_sha)
    augmented=SameEventLearnerSupplement(base,reader)
    expert_index=read(root/'runs/recovery_sft_preparation_20261009/expert-anchor-index-v1.json')
    by_task={k:[r['candidate'] for r in v] for k,v in expert_index['rows'].items()}
    kwargs=dict(batch_size=recipe['pilot']['H1_L0_global_batch'],maximum_event_passes=recipe['pilot']['event_passes'],
        seed=recipe['seed'],allow_extended_event_fit=recipe['extended_event_fit'],
        anchor_selection_protocol=recipe['L0']['anchor_selection_protocol'])
    control=list(finite_mixture_schedule(base.rows,by_task,**kwargs))
    candidate=list(finite_mixture_schedule(augmented.rows,by_task,**kwargs))
    if project(control,base.rows)!=project(candidate,augmented.rows):
        raise ValueError('Supplement altered expert identities, event weighting or DDP rank positions')
    draws=Counter(augmented.rows[i]['candidate']['sample_id'] for b in candidate for kind,i in b['rows']
        if kind=='new' and i>=len(base))
    if not draws:raise ValueError('Registered schedule never exercises the supplement')
    samples=[]
    for i,row in enumerate(reader.rows):
        sample=augmented[len(base)+i];m=sample['samples']
        if (m['memlite_branch']!='low' or not m['low_action_supervision_mask']
                or m.get('outcome_supervision_mask',False) or sample['action_is_pad'].any()
                or not all(torch.isfinite(v).all() for v in sample['pixel_values'].values())):
            raise ValueError('Learner TRAIN branch/mask/image drift')
        samples.append(dict(sample_id=row['sample_id'],source_group=row['source_group'],
            observation_step=row['observation']['control_step'],actions_sha256=row['actions_sha256'],
            shape=list(sample['action'].shape),padding_indices=[7,8,17,18],train_masks_verified=True,
            scheduled_exposures=draws[row['sample_id']]))
    schedule_check=dict(unchanged_expert_event_rank_order=True,updates=len(candidate),
        original_train_anchors=len(base),augmented_train_anchors=len(augmented),
        independent_events=len({(r['candidate']['source_group'],r['approval']['event_id']) for r in base.rows}),
        common_projection_sha256=digest(project(control,base.rows)),
        learner_exposures=sum(draws.values()),unique_scheduled_learner_anchors=len(draws),
        original_dev_unchanged=True)
    result=dict(schema='native_learner_processor_audit_v1',status='passed',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        recipe_sha256=file_sha(a.recipe),admission_sha256=admission_sha,
        manifest_sha256=spec['manifest']['sha256'],owner_review_sha256=spec['owner_review']['sha256'],
        stats_sha256=recipe['stats_sha256'],training_processor_checked=True,samples=samples,
        same_event_schedule_check=schedule_check,optimizer_steps=0,oracle_inputs=False,training_launch_authorized=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);write_new(a.output,result)
    print(json.dumps(dict(status='passed',samples=len(samples),schedule=schedule_check,audit_sha256=file_sha(a.output))))


if __name__=='__main__':main()
