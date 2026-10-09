"""Compare original demo observations with actual simulator prefix evidence.

Produces review materials and error statistics, never automatic labels or a
claim that a saved RGB/proprio vector constitutes a simulator snapshot.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',type=Path,required=True);p.add_argument('--sim',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.proposal/'manifest.json').read_text());ref=np.load(a.proposal/'prefix.npz')['state']
    rows=[json.loads(x) for x in (a.sim/'transitions.jsonl').read_text().splitlines()]
    prefix=[x for x in rows if x['before']['label'].startswith('expert-')]
    if len(prefix)!=352: raise ValueError('Incomplete expert prefix')
    states=np.array([x['before']['proprio'] for x in prefix]+[prefix[-1]['after']['proprio']])
    difference=np.abs(states-ref)
    targets=sorted({s['target'] for segment in manifest['source_episode']['segments'] if segment['start']<352
                    for s in json.loads(segment['semantic']) if s['verb']=='GRASP'})
    events={}
    for target in targets:
        holds={};longest={};first={}
        for t,row in enumerate(prefix):
            matches=[v for v in row['after']['physics'].values() if v['name']==target]
            if len(matches)!=1: continue
            for arm,state in matches[0]['grasp'].items():
                holds[arm]=holds.get(arm,0)+1 if state=='TRUE' else 0
                longest[arm]=max(longest.get(arm,0),holds[arm])
                if holds[arm]>=6: first.setdefault(arm,t+1)
        events[target]=dict(first_stable_grasp_observation=first,longest_consecutive_true=longest)
    sheets=[]
    for index,frames in enumerate(([0,128],[256,352])):
        sheet=Image.new('RGB',(1344,498),'white');draw=ImageDraw.Draw(sheet)
        for col,t in enumerate(frames):
            for r,(label,directory) in enumerate((('Original demo',a.proposal),('Actual official replay',a.sim/'frames'))):
                for c,camera in enumerate(('head','left_wrist','right_wrist')):
                    name=f'{t:06d}-{camera}_rgb.png' if r==0 else f'{t:06d}-{camera}.png'
                    with Image.open(directory/name) as im:
                        picture=im.convert('RGB').resize((224,224))
                    x=col*672+c*224;y=r*249
                    draw.text((x+2,y+3),f'{label} t={t} {camera}',fill='black')
                    sheet.paste(picture,(x,y+24))
        name=f'comparison-{index+1}.png';sheet.save(a.output/name);sheets.append(dict(path=name,sha256=file_sha(a.output/name)))
    result=dict(schema='expert_prefix_review_materials_v1',proposal_sha256=file_sha(a.proposal/'manifest.json'),
        simulator_transition_sha256=file_sha(a.sim/'transitions.jsonl'),actual_prefix_controls=len(prefix),
        first_state_abs_error=difference[0].tolist(),final_state_abs_error=difference[-1].tolist(),
        maximum_error_by_dimension=difference.max(0).tolist(),target_grasp_evidence=events,
        panels=24,images_resized_for_layout_only=True,training_approved=False,sheets=sheets)
    if (a.sim/'result.json').exists():
        sim=json.loads((a.sim/'result.json').read_text())
        result['branch_max_proprio_abs_error']=sim['replay_max_proprio_abs_error']
        result['snapshot_sha256']=sim['full_snapshot_sha256']
        result['branch_object_max_position_error']=max(
            float(np.max(np.abs(np.array(x['physics'][k]['position'])-np.array(y['physics'][k]['position']))))
            for x,y in zip(sim['branches'][0]['states'],sim['branches'][1]['states'])
            for k in x['physics'] if k in y['physics'])
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if 'error' not in k or not isinstance(v,list)},indent=2))


if __name__=='__main__':main()
