"""Review real issued RETRY targets separately from outcome-only approvals.

The selected branches are an explicit already-reviewed TRAIN cohort, not
chosen by model predictions. Show the decision image, the actual correction,
and a stable endpoint. This does NOT approve actions, outcomes, or plans.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_admission import local_file
from recovery_teacher_corpus import validate_branch,physical_proposal,PhysicalProposalIndex


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','corpus','outcome-decisions','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.output.exists():raise FileExistsError(a.output)
    prior=json.loads(a.outcome_decisions.read_text())
    if prior['schema']!='owner_local_recovery_review_v1':raise ValueError('Explicit reviewed cohort required')
    queue=json.loads((a.corpus/'review-queue.json').read_text())
    from PIL import Image,ImageDraw
    selected=[r for r in prior['branches'] if r['branch']!='clean']
    if len({r['case'] for r in selected})!=len(selected):raise ValueError('Duplicate source case')
    a.output.mkdir(parents=True);proposals=[];index=[]
    for ordinal,choice in enumerate(selected):
        matches=[q for q in queue if (q['case'],q['branch'])==(choice['case'],choice['branch'])]
        if len(matches)!=1 or matches[0]['split']!='train':raise ValueError('Only unchanged TRAIN cohort')
        q=matches[0];branch=local_file(root,Path(q['source_path'])/'manifest.json').parent
        m=json.loads((branch/'manifest.json').read_text())
        if (file_sha(branch/'manifest.json')!=choice['manifest_sha256'] or not m['physical_recovery_candidate']
                or m['failure'] is not None or file_sha(branch/'transitions.jsonl')!=m['transitions_sha256']
                or file_sha(branch/'plans.json')!=m['plans_sha256']):
            raise ValueError('Wrong/unverified corrective branch')
        rows=[json.loads(x) for x in (branch/'transitions.jsonl').read_text().splitlines()]
        plans=json.loads((branch/'plans.json').read_text());validate_branch(rows,m,plans)
        retries=[e for e in plans if e['decision']=='RETRY']
        if len(retries)!=1:raise ValueError('Require one actually issued retry')
        retry=retries[0]['control_step'];available=sorted(int(t) for t in m['anchors'] if int(t)<len(rows))
        if retry not in available:raise ValueError('No original decision image')
        if any(r['label_kind']!='same_state_local_teacher_candidate' for r in rows[retry:]):
            raise ValueError('Perturbation actions inside claimed corrective execution')
        idx=PhysicalProposalIndex(rows,m['arm'])
        labels={t:physical_proposal(rows,t,m['arm'],attempt_start=max(e['control_step'] for e in plans if e['control_step']<=t),index=idx)
                for t in available}
        success=next((t for t in available if t>retry and labels[t]=='SUCCEEDED'),None)
        if success is None or labels[available[-1]]!='SUCCEEDED':raise ValueError('Missing achieved/stable endpoint')
        frames=sorted({retry-4,retry,retry+4,retry+16,retry+32,success,available[-1]} & set(available))
        directory=a.output/(choice['case']+'--'+choice['branch']);directory.mkdir()
        evidence=[];sheets=[]
        for offset in range(0,len(frames),4):
            canvas=Image.new('RGB',(672,min(4,len(frames)-offset)*270),'white');draw=ImageDraw.Draw(canvas)
            for line,t in enumerate(frames[offset:offset+4]):
                draw.text((3,line*270+3),f'{ordinal:02d} {choice["case"]} / {m["arm"]} / t={t}',fill='black')
                draw.text((3,line*270+18),('ISSUED RETRY (TARGET)' if t==retry else 'execution context only')+f' / {labels[t]}',fill='black')
                hashes=m['anchors'][str(t)]['sha256']
                for col,camera in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
                    path=branch/'rgb'/f'{t:08d}'/(camera+'.jpg')
                    if file_sha(path)!=hashes[camera]:raise ValueError('Original RGB changed')
                    with Image.open(path) as im:
                        if im.size!=(224,224):raise ValueError('Unexpected original camera size')
                        canvas.paste(im.convert('RGB'),(col*224,line*270+40))
                evidence.append(dict(control_step=t,actual_physics=rows[t]['physical_before'],proposed_outcome=labels[t],
                    preceding_six=[dict(control_step=r['control_step'],physical_audit=r['physical_audit']) for r in rows[max(0,t-6):t]],
                    image_sha256=hashes,label_kind=rows[t]['label_kind']))
            path=directory/f'review-{offset//4:02d}.jpg';canvas.save(path,quality=95,subsampling=0)
            sheets.append(dict(path=path.name,sha256=file_sha(path)))
            index.append(dict(path=str(path.relative_to(a.output)),sha256=file_sha(path),case=choice['case'],arm=m['arm'],frames=frames[offset:offset+4]))
        material=dict(schema='offline_local_recovery_review_materials_v1',source=str(branch),
            manifest_sha256=file_sha(branch/'manifest.json'),transitions_sha256=m['transitions_sha256'],
            frames=frames,panels=3*len(frames),sheets=sheets,evidence=evidence,human_approved=False,
            actual_issued_retry=retries[0],plan_source_sha256=m['plans_sha256'],action_review_complete=False,
            review_scope='planner target only; sparse execution context cannot approve action windows')
        (directory/'review.json').write_text(json.dumps(material,indent=2)+'\n')
        proposals.append(dict(case=choice['case'],branch=choice['branch'],manifest_sha256=file_sha(branch/'manifest.json'),
            review_directory=str(directory.resolve().relative_to(root)),outcomes=[],action_steps=[],planner_steps=[retry],
            corrective_execution_visually_verified=False,notes='PENDING owner original-media and actual-plan review'))
    (a.output/'proposed-decisions.json').write_text(json.dumps(dict(schema='pending_planner_review_not_approval_v1',reviewer='',branches=proposals),indent=2)+'\n')
    receipt=dict(sheets=index,sources=len(selected),planner_candidates=len(selected),outcome_candidates=0,action_candidates=0,
        original_rgb_panels=sum(3*len(r['frames']) for r in index),training_ready=False,
        original_outcome_review_sha256=file_sha(a.outcome_decisions),selection='same preapproved TRAIN cohort; no model or eval/test selection')
    (a.output/'review-index.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='sheets'}))


if __name__=='__main__':main()
