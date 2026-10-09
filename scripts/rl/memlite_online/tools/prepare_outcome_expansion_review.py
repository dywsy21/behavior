"""Deterministic additional TRAIN outcome review, never automatic approval.

Each source contributes failed / genuinely progressing / recovered observations
and a same-clock clean-held counterexample. No action or planner targets are
released. Original RGB is SHA-checked and shown without generated imagery.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest,file_sha
from recovery_teacher_corpus import validate_branch,physical_proposal,PhysicalProposalIndex


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sources',type=int,default=24)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[4]
    if a.output.exists() or a.sources<=0:raise ValueError('Use a new review and positive source count')
    from PIL import Image,ImageDraw
    excluded=set()
    for p in (repo/'configs/recovery_sft').glob('*owner_review*.json'):
        d=json.loads(p.read_text());excluded.update(x['case'] for x in d.get('branches',[]) if 'case' in x)
    queue=json.loads((a.corpus/'review-queue.json').read_text())
    by_case={}
    for q in queue:
        if q['split']=='train' and q['case'] not in excluded:by_case.setdefault(q['case'],{})[q['branch']]=q
    eligible=[]
    for case,branches in by_case.items():
        if 'clean' not in branches:continue
        for name in ('open_gripper_joint_jitter','open_gripper'):
            q=branches.get(name)
            if (q and q['physical_recovery_candidate'] and q['failure'] is None
                    and all(q['proposed_outcomes'].get(k,0)>0 for k in ('SUCCEEDED','FAILED','IN_PROGRESS'))):
                eligible.append((digest(case),case,name,q['arm']));break
    # Interleave actual left/right arms, at most one new source per task.
    buckets={arm:sorted(x for x in eligible if x[3]==arm) for arm in ('left','right')}
    selected=[];tasks=set()
    while len(selected)<a.sources and any(buckets.values()):
        for arm in ('left','right'):
            while buckets[arm]:
                row=buckets[arm].pop(0);task=row[1].rsplit('_',1)[0]
                if task not in tasks:
                    selected.append(row);tasks.add(task);break
            if len(selected)==a.sources:break
    if len(selected)!=a.sources:raise ValueError('Insufficient distinct unreviewed TRAIN sources')
    a.output.mkdir(parents=True);proposals=[];sheets=[]
    for ordinal,(_,case,name,arm) in enumerate(selected):
        qs=by_case[case];q=qs[name]
        def load(q):
            branch=a.root/q['source_path'];m=json.loads((branch/'manifest.json').read_text())
            if file_sha(branch/'transitions.jsonl')!=m['transitions_sha256'] or file_sha(branch/'plans.json')!=m['plans_sha256']:
                raise ValueError('Changed closed branch evidence')
            rows=[json.loads(x) for x in (branch/'transitions.jsonl').read_text().splitlines()]
            plans=json.loads((branch/'plans.json').read_text());validate_branch(rows,m,plans)
            available=sorted(int(t) for t in m['anchors'] if int(t)<len(rows));idx=PhysicalProposalIndex(rows,m['arm'])
            labels={t:physical_proposal(rows,t,m['arm'],attempt_start=max(p['control_step'] for p in plans if p['control_step']<=t),index=idx) for t in available}
            return branch,m,rows,plans,labels
        branch,m,rows,plans,labels=load(q)
        retry=plans[1]['control_step']
        chosen=[]
        for label in ('FAILED','IN_PROGRESS','SUCCEEDED'):
            ts=[t for t,v in labels.items() if v==label and (t<retry if label=='FAILED' else t>retry)]
            if not ts:raise ValueError('Proposed class does not describe this causal attempt')
            chosen.append((min(ts),label))
        clean,cm,cr,cp,cl=load(qs['clean'])
        failed_t=chosen[0][0];clean_ts=[t for t,label in cl.items() if label=='SUCCEEDED']
        same_clock=min(clean_ts,key=lambda t:(abs(t-failed_t),t))
        samples=[(branch,m,rows,chosen,q),(clean,cm,cr,[(same_clock,'SUCCEEDED')],qs['clean'])]
        canvas=Image.new('RGB',(672,4*267),'white');draw=ImageDraw.Draw(canvas);line=0
        for branch,m,rows,points,source in samples:
            review=a.output/(case+'--'+source['branch']);review.mkdir();evidence=[]
            for t,label in points:
                draw.text((4,line*267+3),f'{ordinal+1:02d} {case} {source["branch"]}',fill='black')
                draw.text((4,line*267+19),f't={t} {arm} / CANDIDATE {label}',fill='black')
                sha=m['anchors'][str(t)]['sha256']
                for col,camera in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
                    path=branch/'rgb'/f'{t:08d}'/(camera+'.jpg')
                    if file_sha(path)!=sha[camera]:raise ValueError('Changed original RGB')
                    with Image.open(path) as im:
                        if im.size!=(224,224):raise ValueError('Unexpected original image shape')
                        canvas.paste(im.convert('RGB'),(224*col,267*line+40))
                evidence.append(dict(control_step=t,proposed_outcome=label,actual_physics=rows[t]['physical_before'],
                    preceding_six=[dict(control_step=r['control_step'],physical_audit=r['physical_audit']) for r in rows[t-6:t]],image_sha256=sha))
                line+=1
            proposals.append(dict(case=case,branch=source['branch'],manifest_sha256=file_sha(branch/'manifest.json'),
                review_directory=str(review.relative_to(a.root)),outcomes=[dict(control_step=t,value=label) for t,label in points],
                action_steps=[],planner_steps=[],corrective_execution_visually_verified=False,
                notes='PENDING owner original-media review; no positive BC/planner approval'))
            material=dict(schema='offline_local_recovery_review_materials_v1',source=str(branch),
                manifest_sha256=file_sha(branch/'manifest.json'),transitions_sha256=m['transitions_sha256'],
                frames=[t for t,_ in points],panels=len(points)*3,evidence=evidence,human_approved=False)
            (review/'material.pending.json').write_text(json.dumps(material,indent=2)+'\n')
        sheet=a.output/(f'source-{ordinal:02d}.jpg');canvas.save(sheet,quality=95,subsampling=0)
        sheet_sha=file_sha(sheet);sheets.append(dict(path=sheet.name,sha256=sheet_sha,case=case,arm=arm))
        for _,_,_,_,source in samples:
            review=a.output/(case+'--'+source['branch']);material=json.loads((review/'material.pending.json').read_text())
            # Both branches share one exact reviewed sheet, with separate
            # physical clocks and manifests; no cross-branch label reuse.
            material['sheets']=[dict(path='../'+sheet.name,sha256=sheet_sha)]
            (review/'review.json').write_text(json.dumps(material,indent=2)+'\n')
    (a.output/'proposed-decisions.json').write_text(json.dumps(dict(schema='pending_outcome_review_not_approval_v1',
        reviewer='',branches=proposals,selection='hash order, arm interleave, distinct new TRAIN tasks, no model predictions'),indent=2)+'\n')
    (a.output/'review-index.json').write_text(json.dumps(dict(sheets=sheets,inventory_sha256=file_sha(a.corpus/'audit/inventory.json'),
        anchors_sha256=file_sha(a.corpus/'audit/anchors.jsonl'),sources=len(selected),outcomes=4*len(selected),
        original_rgb_panels=12*len(selected),training_ready=False),indent=2)+'\n')
    print(json.dumps(dict(sources=len(selected),outcomes=4*len(selected),original_rgb_panels=12*len(selected),training_ready=False)))


if __name__=='__main__':main()
