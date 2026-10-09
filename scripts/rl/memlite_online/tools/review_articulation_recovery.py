"""Original RGB contact sheets at physical times; does NOT sign labels."""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.branch/'manifest.json').read_text())
    if file_sha(a.branch/'transitions.jsonl')!=manifest['transitions_sha256']:raise ValueError('Changed actual trajectory')
    rows=[json.loads(l) for l in (a.branch/'transitions.jsonl').read_text().splitlines()]
    available=sorted(int(t) for t in manifest['anchors'] if int(t)<len(rows))
    retry=manifest.get('retry_control');wanted={0,available[-1],*range(0,len(rows),16)}
    if retry is not None:wanted.update(range(max(0,retry-16),retry+49,4))
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
        frames=frames,panels=len(frames)*3,sheets=sheets,evidence=evidence,human_approved=False)
    (a.output/'review.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(frames=frames,panels=len(frames)*3)))


if __name__=='__main__':main()
