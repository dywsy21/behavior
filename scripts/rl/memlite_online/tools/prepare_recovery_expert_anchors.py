"""Pin a small task-/instance-diverse anchor index from ACCEPTED expert TRAIN.

No RGB decode, GPU, new label, data rewrite, or sample-count inflation. This
is a reproducible rehearsal source for the proposed 70/30 mixture, not BC
correction for a different rollout state.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha,group_key  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--protected-groups',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    start=time.monotonic()
    accepted=json.loads((args.release/'acceptance.json').read_text())
    manifest=json.loads((args.release/'manifest.json').read_text())
    if (accepted['status']!='ACCEPTED' or not all(accepted['gates'].values())
            or accepted['manifest_sha256']!=file_sha(args.release/'manifest.json')):
        raise ValueError('Unaccepted original expert source')
    for name in ('train_candidates.npy','train_tasks.npy','episodes.jsonl','episode_offsets.npy'):
        if file_sha(args.release/name)!=manifest['files'][name]:raise ValueError('Changed original expert index')
    protected=json.loads(args.protected_groups.read_text())
    if protected['manifest_sha256']!=accepted['manifest_sha256']:raise ValueError('Stale protected split')
    protected=set(protected['groups'])
    import numpy as np
    import torch
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    dataset=Stage1Dataset(args.release,None,'low','train')
    rng=np.random.default_rng(20261009)
    selected={}
    for task in np.unique(dataset.task_ids):
        candidates=np.flatnonzero(dataset.task_ids==task)
        proposals=rng.choice(candidates,size=min(512,len(candidates)),replace=False)
        rows=[];seen=set()
        for candidate in proposals:
            ep,frame,segment,_,_,_=dataset.locate(int(candidate))
            row=ep['row'];group=group_key(ep['task_name'],row['task_instance_id'])
            valid=min(32,row['length']-frame,ep['segments'][segment]['end']-frame)
            if ep['split']!='train' or group in protected:raise ValueError('Heldout rehearsal leakage')
            if valid!=32 or group in seen:continue
            seen.add(group)
            rows.append(dict(candidate=int(candidate),task=int(task),task_name=ep['task_name'],
                source_group=group,episode=int(row['episode_index']),frame=int(frame),valid_actions=valid))
            if len(rows)==32:break
        if len(rows)<8:raise ValueError('Insufficient distinct full-horizon TRAIN anchors for task '+str(task))
        selected[str(int(task))]=rows
    if len(selected)!=100 or torch.cuda.is_initialized():raise ValueError('Expected 100-task CPU-only index')
    result=dict(schema='recovery_expert_anchor_index_v1',manifest_sha256=accepted['manifest_sha256'],
        protected_groups_sha256=file_sha(args.protected_groups),release=str(args.release),
        seed=20261009,rows=selected,tasks=len(selected),samples=sum(len(v) for v in selected.values()),
        maximum_per_task=32,one_per_source_instance=True,all_full_horizon=True,seconds=time.monotonic()-start,
        rgb_decoded=False,new_recovery_labels=0,cuda_context_created=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
