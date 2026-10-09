"""Render actual fresh failure observations for explicit owner start approval.

Review sheets are audit-only; never actor/training images. This creates no
approval, changes no simulator state, and independently hashes every file.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--remote-root',type=Path,required=True)
    p.add_argument('--local-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from PIL import Image,ImageOps,ImageDraw
    inventory=json.loads(a.inventory.read_text());candidates=[];rejected=[];sheets=[]
    if inventory['status']!='inventory_checked':raise ValueError('Incomplete cold inventory')
    a.output.mkdir(parents=True)
    for row in inventory['results']:
        directory=a.local_root/Path(row['directory']).relative_to(a.remote_root)
        path=directory/Path(row['receipt']).name
        if file_sha(path)!=row['sha256']:raise ValueError('Changed cold receipt')
        if row['status']!='accepted_failure_start':
            rejected.append(dict(case=row['case'],status=row['status'],receipt_sha256=row['sha256']));continue
        receipt=json.loads(path.read_text())
        if (receipt['status']!='accepted_failure_start' or not receipt['cold_process']
                or receipt['actor_oracle_inputs'] or receipt['optimizer_steps']!=0
                or receipt['continued_stable_grasp_controls']<64
                or receipt['failed_target_position_error_m']>.005
                or receipt['restored_failure_max_proprio_error']>1e-3):raise ValueError('Invalid cold evidence')
        snapshot=directory/'failure-start.pt'
        if file_sha(snapshot)!=receipt['failure_start_sha256']:raise ValueError('Changed saved failure world')
        candidates.append(dict(case=row['case'],source_group=receipt['source_group'],split=receipt['split'],
            directory=str(directory),remote_directory=row['directory'],receipt_sha256=row['sha256'],
            snapshot_sha256=receipt['failure_start_sha256'],source_branch_sha256=receipt['source_branch_sha256'],
            failed_position_error_m=receipt['failed_target_position_error_m'],
            continued_stable_grasp_controls=receipt['continued_stable_grasp_controls'],
            images={cam:file_sha(directory/(cam+'.png')) for cam in ('head','left_wrist','right_wrist')}))
    for offset in range(0,len(candidates),4):
        items=candidates[offset:offset+4];canvas=Image.new('RGB',(1080,390*len(items)),'white');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(items):
            draw.text((6,i*390+4),f"{row['case']} / {row['split']} / actual failure t=32",fill='black')
            for j,cam in enumerate(('head','left_wrist','right_wrist')):
                with Image.open(Path(row['directory'])/(cam+'.png')) as original:
                    original.load();thumb=ImageOps.contain(original.convert('RGB'),(360,360))
                canvas.paste(thumb,(j*360+(360-thumb.width)//2,i*390+25+(360-thumb.height)//2))
        path=a.output/f'starts-{offset//4:02d}.jpg';canvas.save(path,quality=95)
        sheets.append(dict(path=path.name,sha256=file_sha(path),cases=[r['case'] for r in items]))
    result=dict(schema='recovery_start_visual_review_v1',inventory_sha256=file_sha(a.inventory),
        candidates=candidates,rejected=rejected,sheets=sheets,manual_review_complete=False,
        scope='Cold-reproduced local failure starts; not full-task reset or actor evaluation')
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(candidates=len(candidates),rejected=len(rejected),sheets=len(sheets))))


if __name__=='__main__':main()
