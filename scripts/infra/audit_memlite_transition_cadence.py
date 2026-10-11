"""TRAIN-only metadata audit; creates no labels, samples or model updates."""
import argparse
from collections import Counter,defaultdict
import importlib.util
import json
from pathlib import Path
import time

from behavior_champion.release import AcceptedRelease, sha256

# Load the exact checked-in metadata module without executing g05.data's
# package initializer (which unnecessarily imports model/decoder dependencies).
_labels_spec=importlib.util.spec_from_file_location('cadence_stage1_labels',
    Path(__file__).resolve().parents[2]/'src/g05/data/memlite_stage1_labels.py')
_labels=importlib.util.module_from_spec(_labels_spec)
_labels_spec.loader.exec_module(_labels)
anchor_records=_labels.anchor_records


def analyze_episode(record):
    anchors=anchor_records(record['segments'],record['phase'])
    if not anchors:raise ValueError('No eligible annotated anchor')
    count=Counter(episodes=1,anchors=len(anchors))
    prior_frame=-1;coarse_previous='None';coarse_parent='None'
    first=anchors[0][0]
    for frame,segment,previous,parent,_ in anchors:
        if frame<=prior_frame:raise ValueError('Noncausal/repeated annotation clock')
        prior_frame=frame
        current=record['segments'][segment]
        if previous=='None':kind='initial'
        elif previous!=current['text']:kind='bundle_transition'
        else:kind='continuation'
        count[kind]+=1
        if previous!='None' and previous==current['text'] and parent!=current['parent']:
            count['parent_only_transition']+=1
        if (frame-first)%128==0:
            count['periodic128_eligible_calls']+=1
            count['periodic128_prior_differs_from_stride16']+=previous!=coarse_previous
            count['periodic128_parent_differs_from_stride16']+=parent!=coarse_parent
            if coarse_previous!='None':
                count['periodic128_noninitial_calls']+=1
                count['periodic128_bundle_transition']+=coarse_previous!=current['text']
            coarse_previous,coarse_parent=current['text'],current['parent']
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh audit result required')
    import numpy as np
    started=time.monotonic();accepted=AcceptedRelease(a.release)
    for name in ('train_candidates.npy','train_tasks.npy'):accepted.verify_file(name)
    candidates=np.load(a.release/'train_candidates.npy',mmap_mode='r')
    task_ids=np.load(a.release/'train_tasks.npy',mmap_mode='r')
    counts=Counter();tasks=defaultdict(Counter);cursor=0
    for serial,ep in accepted.episodes('train'):
        row=analyze_episode(ep);task=int(ep['row']['task_index']);n=row['anchors']
        observed=candidates[cursor:cursor+n]
        if (observed.shape!=(n,2) or not np.all(observed[:,0]==serial) or
                not np.array_equal(observed[:,1],np.arange(n)) or
                not np.all(task_ids[cursor:cursor+n]==task)):
            raise ValueError('Audit source/anchor/task differs from original training sampler')
        counts.update(row);tasks[task].update(row);cursor+=n
    eligible=accepted.acceptance['eligible_episodes'];quarantined=accepted.excluded_splits
    if (candidates.shape[0]!=counts['anchors'] or task_ids.shape!=(cursor,) or
            counts['episodes']!=eligible['train'] or counts['episodes']+quarantined['train']!=19000 or
            eligible['eval']+quarantined['eval']!=1000 or set(tasks)!=set(range(100))):
        raise ValueError('Audit does not exactly cover original protected TRAIN candidate universe')
    result=dict(status='train_metadata_transition_and_cadence_audit_completed',counts=dict(counts),
        tasks={str(k):dict(v) for k,v in sorted(tasks.items())},excluded_original_eval_episodes=1000,
        preserved_source_quarantines=dict(quarantined),original_candidates_all_rows_exact=True,
        source_manifest_sha256=sha256(a.release/'manifest.json'),
        bundle_transition_fraction=counts['bundle_transition']/counts['anchors'],
        periodic128_previous_mismatch_fraction=counts['periodic128_prior_differs_from_stride16']/counts['periodic128_eligible_calls'],
        seconds=time.monotonic()-started,optimizer_updates=0,new_training_samples=0,physical_controls=0,
        caveat='Annotation-command resampling diagnostic only. Fixed128 history is not an observed actor rollout, and changing refresh-count stride alone does not change previous intent/history. No success/failure labels are inferred.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'}),flush=True)


if __name__=='__main__':main()
