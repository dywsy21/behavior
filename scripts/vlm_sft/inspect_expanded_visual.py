"""Read-only H80 RAW inventory and deterministic TRAIN human-review selection.

Does not create labels, alter holdouts, or release a training dataset.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from prepare_visual_review import sha


def inventory(root):
    supervisor=json.loads((root/'supervisor.json').read_text())
    raw=root/'raw';manifest=json.loads((raw/'manifest.json').read_text());seal=json.loads((raw/'complete.json').read_text())
    if (supervisor['status']!='completed' or supervisor['exit_code']!=0 or
            (raw/'failure.json').exists() or supervisor['seal']!=seal or
            seal!={'status':'RAW_COMPLETE_LABELS_PENDING','manifest_sha256':sha(raw/'manifest.json'),'training_eligible':False} or
            sha(raw/'images.jsonl')!=manifest['images_manifest_sha256']):
        raise ValueError('Original complete RAW package and supervisor required')
    rows=[json.loads(line) for line in (raw/'images.jsonl').read_text().splitlines()]
    images=[];groups=defaultdict(list)
    for row in rows:
        if row['split']=='train':groups[(row['task'],row['instance'])].append(row)
        for view,receipt in row['image_receipts'].items():
            images.append({'id':row['id']+'_'+view,'split':row['split'],'task':row['task'],'instance':row['instance'],
                           'view':view,'path':row['images'][view],'png_sha256':receipt['png_sha256'],
                           'pixels':receipt['raw_pixels_sha256'],'dhash':receipt['difference_hash_256']})
    held={p['pixels'] for p in images if p['split']!='train'}
    counts=Counter(p['pixels'] for p in images if p['split']=='train')
    selections=[]
    for task in range(5):
        chosen=sorted((g for g in groups if g[0]==task),
                      key=lambda g:hashlib.sha256(f'h80-parent-review-41:{g}'.encode()).hexdigest())[:2]
        if len(chosen)!=2:raise ValueError('Insufficient task review coverage')
        for group in chosen:
            ordered=sorted(groups[group],key=lambda r:r['frame'])
            selected=[ordered[round(k*(len(ordered)-1)/4)] for k in range(5)]
            for row in selected:
                selections.append({k:row[k] for k in ('id','task','instance','episode','frame','split','images','image_receipts')})
    return {'status':'RAW_INVENTORY_ONLY','training_eligible':False,'manifest_sha256':sha(raw/'manifest.json'),
            'source_code_commit':manifest['code_commit'],'image_count':len(images),
            'images_by_split':dict(Counter(p['split'] for p in images)),
            'train_distinct_pixel_images':len(counts),'train_exact_duplicate_excess':sum(n-1 for n in counts.values()),
            'train_images_exactly_matching_heldout':sum(p['pixels'] in held for p in images if p['split']=='train'),
            'train_distinct_pixels_not_in_heldout':len(set(counts)-held),
            'train_unique_dhashes':len({p['dhash'] for p in images if p['split']=='train'}),
            'counts_task_view':dict(Counter(f'{p["task"]}:{p["view"]}' for p in images if p['split']=='train')),
            'review_states':selections,'review_images':len(selections)*3,'review_source_groups':10,
            'manual_review':'PENDING','near_duplicate_filter_applied':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(inventory(args.root)),flush=True)
