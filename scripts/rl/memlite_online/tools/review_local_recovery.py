"""Lay out original camera frames and exact prior physics for owner review."""
import argparse
import json
from pathlib import Path
import sys

from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_teacher_corpus import validate_branch,physical_proposal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.branch/'manifest.json').read_text())
    rows=[json.loads(x) for x in (a.branch/'transitions.jsonl').read_text().splitlines()]
    plans=json.loads((a.branch/'plans.json').read_text());validate_branch(rows,manifest,plans)
    available=sorted(int(t) for t in manifest['anchors'] if int(t)<len(rows))
    # Every available RGB anchor over the entire proposed 32-control action
    # target, plus loss and stable endpoint. Never replicate missing frames.
    frames=sorted({0,available[-1],*([16,28] if 28 in available else [16]),
                   *(t for t in available if 32<=t<=64)} & set(available))
    evidence=[];sheets=[]
    for start in range(0,len(frames),4):
        sheet=Image.new('RGB',(672,min(4,len(frames)-start)*262),'#fafafa');draw=ImageDraw.Draw(sheet)
        for line,t in enumerate(frames[start:start+4]):
            row=rows[t];ctx=[v for v in plans if v['control_step']<=t][-1]
            label=physical_proposal(rows,t,manifest['arm'],attempt_start=ctx['control_step'])
            physics=row['physical_before'];arm=manifest['arm']
            title=f"t={t} {row['label_kind']} | {arm}: {physics['grasp'][arm]} aperture={physics['gripper_aperture'][arm]:.4f}"
            draw.text((4,line*262+3),title,fill='black')
            draw.text((4,line*262+18),f"proposed={label} | {a.branch.parent.name}/{a.branch.name}",fill='black')
            for col,camera in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
                path=a.branch/'rgb'/f'{t:08d}'/(camera+'.jpg')
                if file_sha(path)!=manifest['anchors'][str(t)]['sha256'][camera]:raise ValueError('Changed raw RGB')
                with Image.open(path) as image:sheet.paste(image.convert('RGB').resize((224,224)),(col*224,line*262+36))
            evidence.append(dict(control_step=t,actual_physics=physics,proposed_outcome=label,
                preceding_six=[dict(control_step=r['control_step'],physical_audit=r['physical_audit']) for r in rows[max(0,t-6):t]],
                image_sha256=manifest['anchors'][str(t)]['sha256']))
        name=f'review-{start//4:02d}.jpg';sheet.save(a.output/name,quality=95,subsampling=0)
        sheets.append(dict(path=name,sha256=file_sha(a.output/name)))
    result=dict(schema='offline_local_recovery_review_materials_v1',source=str(a.branch),
        manifest_sha256=file_sha(a.branch/'manifest.json'),transitions_sha256=file_sha(a.branch/'transitions.jsonl'),
        frames=frames,panels=3*len(frames),sheets=sheets,evidence=evidence,human_approved=False,
        action_review_start=32,action_review_end=64,physical_recovery_candidate=manifest['physical_recovery_candidate'])
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),frames=frames,panels=result['panels'],physical_recovery_candidate=manifest['physical_recovery_candidate'])))


if __name__=='__main__':main()
