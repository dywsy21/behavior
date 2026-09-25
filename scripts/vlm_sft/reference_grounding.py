"""H82 teacher-only appearance references; no current-image answer is supplied."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw
from prepare_visual_review import sha
import visual_grounding as single

REPO=Path(__file__).resolve().parents[2]
SPEC=REPO/'configs/vlm_sft/h82_reference_gallery_v1.json'
RAW_REVIEW=REPO/'configs/vlm_sft/h80_parent_raw_review_v1.json'
VERSION='visual-grounding-train-reference-v1'
QUERIES=('the radio','the wastebasket','the plate holding food')
SYSTEM=single.SYSTEM+'''\nBefore CURRENT_RAW there are exactly three labeled reference images. The first two show the target's appearance from other training scenes; they are NOT the scene being labeled. The third is a negative example of the robot itself. Use them only to recognize appearance. Ignore any grippers/background in positive references. Locate targets ONLY in CURRENT_RAW, the FOURTH and LAST image. All output boxes are normalized ONLY to CURRENT_RAW. Never return a reference box or assume the target is present just because a reference shows it. An unmistakable robot shell is absent, not uncertain. Preserve uncertainty for genuinely unidentifiable fragments and plate backs whose food is not visible.'''


def crop_reference(root,ref):
    path=Path(root)/ref['path']
    if path.is_symlink() or sha(path)!=ref['png_sha256']:
        raise ValueError('Reference source PNG changed')
    with Image.open(path) as source:
        if list(source.size)!=ref['original_size']:raise ValueError('Reference dimensions changed')
        crop=ref['crop_xyxy'];w,h=source.size
        if (len(crop)!=4 or any(type(v) is not int for v in crop) or
                not 0<=crop[0]<crop[2]<=w or not 0<=crop[1]<crop[3]<=h):
            raise ValueError('Reference crop outside original image')
        image=source.convert('RGB').crop(crop)
    image.thumbnail((256,256),Image.Resampling.LANCZOS)
    return image


def validate_spec(spec,review):
    if (spec['schema']!='h82-reference-gallery-v1' or
            spec['approved_for']!='bounded_teacher_reference_calibration_only' or
            spec['training_eligible'] is not False or spec['teacher_only'] is not True or
            spec['source_manifest_sha256']!=review['manifest_sha256'] or
            review.get('train_only') is not True or
            review['approved_for']!='raw_candidate_integrity_and_teacher_reference_selection_only'):
        raise ValueError('Exact parent-approved TRAIN reference review required')
    reviewed={s['id']+'_'+v:(g['task'],g['instance'],digest)
        for g in review['groups'] for s in g['states'] for v,digest in s['png_sha256'].items()}
    refs=spec['references']
    if len(refs)!=7 or len({r['name'] for r in refs})!=7 or len({r['id'] for r in refs})!=7:
        raise ValueError('Exactly seven distinct reviewed references')
    for ref in refs:
        if (ref['split']!='train' or ref['role'] not in ('positive','negative') or
                reviewed.get(ref['id'])!=(ref['task'],ref['instance'],ref['png_sha256']) or
                ref['path']!='images/'+ref['id']+'.png'):
            raise ValueError('Unreviewed or non-TRAIN reference')
    for query in QUERIES:
        selected=[r for r in refs if r['query']==query]
        if len(selected)!=2 or any(r['role']!='positive' for r in selected):
            raise ValueError('Two positive references per query required')
    negative=[r for r in refs if r['role']=='negative']
    if len(negative)!=1 or negative[0]['query']!='*':raise ValueError('One shared robot-only negative required')
    return refs


def validate_sources(refs,plan,states):
    sources={(r['task'],r['instance']):r for r in plan['sources']}
    if len(sources)!=len(plan['sources']):raise ValueError('Duplicate raw source group')
    for ref in refs:
        match=re.fullmatch(r't(\d+)_i(\d+)_e(\d+)_f(\d+)_(head|left_wrist|right_wrist)',ref['id'])
        if match is None:raise ValueError('Unknown reference image ID')
        task,instance,episode,frame=map(int,match.groups()[:4]);view=match.group(5)
        source=sources.get((task,instance));state=states.get(ref['id'][:-(len(view)+1)])
        if (source is None or state is None or source['split']!='train' or state['split']!='train' or
                (ref['task'],ref['instance'])!=(task,instance) or source['episode']!=episode or
                frame not in source['selected_frames'] or
                (state['task'],state['instance'],state['episode'],state['frame'])!=(task,instance,episode,frame) or
                state['images'][view]!=ref['path'] or
                state['image_receipts'][view]['png_sha256']!=ref['png_sha256'] or
                state['image_receipts'][view]['resolution']!=ref['original_size']):
            raise ValueError('Reference differs from sealed TRAIN source/ledger')


@lru_cache(maxsize=1)
def gallery():
    spec=json.loads(SPEC.read_text());review=json.loads(RAW_REVIEW.read_text());refs=validate_spec(spec,review)
    root=Path(spec['raw_root'])
    if sha(root/'manifest.json')!=spec['source_manifest_sha256']:
        raise ValueError('Frozen H80 raw seal changed')
    manifest=json.loads((root/'manifest.json').read_text())
    if (sha(root/'source_plan.json')!=manifest['source_plan_sha256'] or
            sha(root/'images.jsonl')!=manifest['images_manifest_sha256']):
        raise ValueError('Sealed H80 source metadata changed')
    plan=json.loads((root/'source_plan.json').read_text());states={}
    wanted={re.sub(r'_(head|left_wrist|right_wrist)$','',r['id']) for r in refs}
    with (root/'images.jsonl').open() as ledger:
        for line in ledger:
            row=json.loads(line)
            if row['id'] in wanted:
                if row['id'] in states:raise ValueError('Duplicate reference state')
                states[row['id']]=row
    validate_sources(refs,plan,states)
    supervisor=json.loads((root.parent/'supervisor.json').read_text())
    if supervisor['status']!='completed' or supervisor['exit_code']!=0:
        raise ValueError('H80 raw collection must be complete')
    images={r['name']:crop_reference(root,r) for r in refs}
    for ref in refs:
        img=images[ref['name']]
        if (list(img.size)!=ref['reviewed_size'] or
                hashlib.sha256(img.tobytes()).hexdigest()!=ref['reviewed_pixels_sha256']):
            raise ValueError('Actual teacher crop differs from parent-reviewed pixels')
    identity={'spec_sha256':sha(SPEC),'parent_review_sha256':sha(RAW_REVIEW),
        'source_manifest_sha256':spec['source_manifest_sha256'],
        'references':[{'name':r['name'],'query':r['query'],'role':r['role'],'id':r['id'],
            'task':r['task'],'instance':r['instance'],'png_sha256':r['png_sha256'],
            'crop_xyxy':r['crop_xyxy'],'size':list(images[r['name']].size),
            'pixels_sha256':hashlib.sha256(images[r['name']].tobytes()).hexdigest()} for r in refs]}
    return refs,images,identity


def identity():return gallery()[2]


def messages(query,image,*,blind=False):
    if query not in QUERIES or blind:raise ValueError('H82 is a registered teacher-only non-blind calibration')
    msg,receipt=single.messages(query,image)
    refs,images,source=gallery()
    selected=[r for r in refs if r['query']==query]+[r for r in refs if r['role']=='negative']
    content=[]
    for index,ref in enumerate(selected):
        title=('POSITIVE_TARGET_REFERENCE_'+str(index+1) if ref['role']=='positive' else 'NEGATIVE_ROBOT_REFERENCE')
        content.extend([{'type':'text','text':title+': '+ref['description']},
                        {'type':'image','image':images[ref['name']].copy()}])
    content.extend(msg[1]['content'])
    receipt.update(protocol=VERSION,reference_spec_sha256=source['spec_sha256'],
        reference_inputs=[r for r in source['references'] if r['name'] in {s['name'] for s in selected}])
    return [{'role':'system','content':SYSTEM},{'role':'user','content':content}],receipt


def encode(processor,row,image):
    msg,receipt=messages(row['query'],image)
    prefix=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',enable_thinking=False)
    old=prefix['input_ids'].shape[1]
    if old>1800:raise ValueError('Reference grounding input budget exceeded')
    grid=prefix.get('image_grid_thw')
    if grid is None or grid.shape[0]!=4:raise ValueError('Exactly three references and one current image required')
    receipt.update(input_tokens=old,input_ids_sha256=hashlib.sha256(prefix['input_ids'].numpy().tobytes()).hexdigest())
    return prefix,receipt


def render_review(root,output):
    """Human-only crop sheet. Does not approve the spec or create model labels."""
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    spec=json.loads(SPEC.read_text());canvas=Image.new('RGB',(4*300,2*320),'white');draw=ImageDraw.Draw(canvas)
    receipts=[]
    for i,ref in enumerate(spec['references']):
        img=crop_reference(root,ref);img.save(output/(ref['name']+'.png'))
        x=(i%4)*300;y=(i//4)*320
        draw.text((x+5,y+4),ref['name']+' / '+ref['role'],fill='black');canvas.paste(img,(x+5,y+30))
        receipts.append({'name':ref['name'],'size':list(img.size),'pixels_sha256':hashlib.sha256(img.tobytes()).hexdigest()})
    canvas.save(output/'references.jpg');(output/'review.json').write_text(json.dumps(receipts,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render_review(args.review_root,args.output)
