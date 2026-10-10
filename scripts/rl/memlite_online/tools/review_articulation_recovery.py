"""Original RGB contact sheets at physical times; does NOT sign labels."""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_articulation_corpus import CLEAN,validate_articulation_branch
from recovery_action_coverage import later_action_windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    windows=p.add_mutually_exclusive_group()
    windows.add_argument('--correction-window',action='store_true',help='All recorded RGB in first 32 real corrective controls, plus causal result anchors')
    windows.add_argument('--later-action-windows',action='store_true',
        help='Review three clean later corrective windows; no automatic approval or new event count')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.branch/'manifest.json').read_text())
    if file_sha(a.branch/'transitions.jsonl')!=manifest['transitions_sha256']:raise ValueError('Changed actual trajectory')
    rows=[json.loads(l) for l in (a.branch/'transitions.jsonl').read_text().splitlines()]
    available=sorted(int(t) for t in manifest['anchors'] if int(t)<len(rows))
    retry=manifest.get('retry_control');wanted={0,available[-1],*range(0,len(rows),16)}
    if retry is not None:wanted.update(range(max(0,retry-16),retry+49,4))
    coverage=None
    if a.later_action_windows:
        plans=json.loads((a.branch/'plans.json').read_text())
        if file_sha(a.branch/'plans.json')!=manifest['plans_sha256']:raise ValueError('Changed issued history')
        seed=json.loads((a.branch.parent/'seed.json').read_text())
        proposals,_=validate_articulation_branch(rows,manifest,plans,seed)
        if (retry is None or not manifest['physical_recovery_candidate'] or manifest['failure'] is not None):
            raise ValueError('No completed physical correction for candidate BC review')
        succeeded=[t for t in available if t>retry+32 and proposals[t]=='SUCCEEDED']
        if not succeeded:raise ValueError('No verified later functional completion')
        coverage=later_action_windows(rows,available,retry_control=retry,first_success=succeeded[0],clean_label=CLEAN)
        frames=sorted(set(coverage['frames'])|{retry,succeeded[0],available[-1]})
    elif a.correction_window:
        if retry is None:raise ValueError('No actual correction for action-window review')
        frames=sorted({0,16,available[-1],*(t for t in available if retry-12<=t<=retry+32)} & set(available))
        endpoints=[t for t in available if t>=retry+32]
        if not endpoints:raise ValueError('No actual observation at/after the 32-control endpoint')
        frames=sorted(set(frames)|{endpoints[0]})
        # Always inspect a physically completed correction endpoint, but do
        # not automatically turn that endpoint into an extra BC target.
        succeeded=[t for t in available if t>retry+32 and rows[t]['observation_outcome_candidate']=='SUCCEEDED']
        if succeeded:frames=sorted(set(frames)|{succeeded[0]})
    else:
        frames=sorted({min(available,key=lambda v:abs(v-t)) for t in wanted if t<len(rows)})
    a.output.mkdir(parents=True);sheets=[];evidence=[]
    for start in range(0,len(frames),4):
        image=Image.new('RGB',(672,min(4,len(frames)-start)*272),'white');draw=ImageDraw.Draw(image)
        for line,t in enumerate(frames[start:start+4]):
            row=rows[t];physical=row['physical_before']
            draw.text((4,line*272+3),f"t={t} {row['teacher']['phase']} | Open={physical['open']} goal={physical['goal_predicate']}",fill='black')
            draw.text((4,line*272+19),f"joint={physical['object_joints']} fraction={physical.get('directed_open_fraction','legacy 5% only')}",fill='black')
            for col,camera in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
                path=a.branch/'rgb'/f'{t:08d}'/(camera+'.jpg')
                if file_sha(path)!=manifest['anchors'][str(t)]['sha256'][camera]:raise ValueError('Changed original RGB')
                with Image.open(path) as source:image.paste(source.convert('RGB'),(col*224,line*272+46))
            # This is current pre-action physics; never row[t]'s post-action outcome.
            evidence.append(dict(control_step=t,physical_before=physical,image_sha256=manifest['anchors'][str(t)]['sha256']))
        name=f'review-{start//4:02d}.jpg';image.save(a.output/name,quality=95,subsampling=0)
        sheets.append(dict(path=name,sha256=file_sha(a.output/name)))
    receipt=dict(schema='articulation_original_media_review_material_v1',manifest_sha256=file_sha(a.branch/'manifest.json'),
        transitions_sha256=file_sha(a.branch/'transitions.jsonl'),
        frames=frames,panels=len(frames)*3,sheets=sheets,evidence=evidence,human_approved=False)
    if coverage is not None:receipt['later_action_window_candidates']=coverage
    (a.output/'review.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(frames=frames,panels=len(frames)*3)))


if __name__=='__main__':main()
