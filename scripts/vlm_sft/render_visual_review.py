"""Human-only contact sheets; these annotated/resized images NEVER train actors."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from PIL import Image,ImageDraw

from prepare_visual_review import atomic_json,sha


def raw_sheets(inventory,images_root,output):
    groups=defaultdict(list)
    for row in inventory['review_states']:
        if row['split']!='train':raise ValueError('This parent review must use TRAIN only')
        groups[(row['task'],row['instance'])].append(row)
    output.mkdir(exist_ok=False);pages=[]
    for (task,instance),rows in sorted(groups.items()):
        canvas=Image.new('RGB',(960,350*len(rows)),(235,235,235));draw=ImageDraw.Draw(canvas)
        for n,row in enumerate(sorted(rows,key=lambda r:r['frame'])):
            for j,view in enumerate(('head','left_wrist','right_wrist')):
                path=images_root/row['images'][view]
                if sha(path)!=row['image_receipts'][view]['png_sha256']:raise ValueError('Original review pixels changed')
                with Image.open(path) as im:image=im.convert('RGB').resize((320,320),Image.Resampling.LANCZOS)
                canvas.paste(image,(j*320,n*350+30))
                draw.text((j*320+3,n*350+3),row['id']+' '+view,fill=(0,0,0))
        path=output/f'task{task}_instance{instance}.jpg';canvas.save(path,quality=95)
        pages.append({'path':str(path),'task':task,'instance':instance,'states':[r['id'] for r in rows],
                      'sha256':sha(path)})
    atomic_json(output/'sheets.json',{'purpose':'human_raw_quality_review_only','training_eligible':False,
        'original_manifest_sha256':inventory['manifest_sha256'],'pages':pages})
    return pages


def teacher_sheets(predictions,images_root,output):
    rows=[r for r in predictions if r['phase']=='primary'];output.mkdir(exist_ok=False);pages=[]
    for offset in range(0,len(rows),9):
        group=rows[offset:offset+9];canvas=Image.new('RGB',(1080,420*3),(240,240,240));draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(group):
            x=(i%3)*360;y=(i//3)*420;path=images_root/'images'/(row['id']+'.png')
            if sha(path)!=row['png_sha256']:raise ValueError('Teacher review pixels changed')
            with Image.open(path) as im:image=im.convert('RGB').resize((360,360),Image.Resampling.LANCZOS)
            painter=ImageDraw.Draw(image)
            if row['format_valid']:
                for box in row['parsed']['boxes']:
                    painter.rectangle([round(v*.36) for v in box],outline=(255,40,40),width=3)
            canvas.paste(image,(x,y+60))
            pred=row['parsed']['visibility'] if row['format_valid'] else 'INVALID'
            for line,text in enumerate((row['id'],row['query'],f'Human {row["expected_visibility"]} / Teacher {pred}')):
                draw.text((x+3,y+line*18+3),text,fill=(0,0,0))
        path=output/f'teacher_{offset//9:02d}.jpg';canvas.save(path,quality=95)
        pages.append({'path':str(path),'ids':[r['id'] for r in group],'sha256':sha(path)})
    atomic_json(output/'sheets.json',{'purpose':'human_teacher_box_review_only','training_eligible':False,'pages':pages})
    return pages


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('raw','teacher'),required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--images-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.mode=='raw':pages=raw_sheets(json.loads(args.manifest.read_text()),args.images_root,args.output)
    else:pages=teacher_sheets([json.loads(line) for line in args.manifest.read_text().splitlines()],args.images_root,args.output)
    print(json.dumps(pages),flush=True)
