"""Prepare middle/late windows from the five existing OPEN TRAIN corrections.

No simulator, optimizer, automatic human approval, new source group, or CAL
access. Existing first-window permissions do NOT authorize the new windows.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--previous-review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    queue=json.loads((a.corpus/'review-queue.json').read_text())
    original=json.loads(a.previous_review.read_text())
    if original['schema']!='owner_local_recovery_review_v1' or 'role' in original:
        raise ValueError('Not the original TRAIN/dev review; never use calibration media')
    todo=[]
    for entry in original['branches']:
        matches=[q for q in queue if q['case']==entry['case'] and q['branch']==entry['branch']]
        if len(matches)!=1:raise ValueError('Source branch is ambiguous or missing')
        q=matches[0]
        if q['split']!='train' or not entry['action_steps']:continue
        if (q.get('mechanism')!='articulation' or not q['physical_recovery_candidate']
                or file_sha(Path(q['source_path'])/'manifest.json')!=entry['manifest_sha256']):
            raise ValueError('Original corrected TRAIN source changed')
        todo.append((q,entry))
    if len(todo)!=5:raise ValueError('Keep all five original OPEN TRAIN groups, no convenient subset')
    a.output.mkdir(parents=True);index=[]
    for q,entry in todo:
        path=a.output/q['case']
        subprocess.run([sys.executable,str(Path(__file__).with_name('review_articulation_recovery.py')),
            '--branch',q['source_path'],'--output',str(path),'--later-action-windows'],check=True)
        review=json.loads((path/'review.json').read_text())
        index.append(dict(case=q['case'],source_group=q['source_group'],split=q['split'],
            branch=q['branch'],source_path=q['source_path'],manifest_sha256=entry['manifest_sha256'],
            review_directory=str(path),review_sha256=file_sha(path/'review.json'),
            original_action_steps=entry['action_steps'],
            candidate_action_steps=review['later_action_window_candidates']['candidate_action_steps'],
            sheets=review['sheets'],panels=review['panels'],human_approved=False))
    result=dict(schema='later_action_coverage_review_v1',status='five_train_sources_pending_owner_review',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        previous_review_sha256=file_sha(a.previous_review),source_queue_sha256=file_sha(a.corpus/'review-queue.json'),
        rows=index,independent_new_groups=0,new_approved_windows=0,new_optimizer_updates=0,
        note='Later same-event action windows, not more independent recovery episodes. Existing DEV/CAL remain untouched.')
    (a.output/'review-index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(sources=len(index),candidate_windows=sum(len(x['candidate_action_steps']) for x in index),
        panels=sum(x['panels'] for x in index),index_sha256=file_sha(a.output/'review-index.json'))))


if __name__=='__main__':main()
