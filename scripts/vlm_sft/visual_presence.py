"""Single-image visible identity probe; not a contact or completion policy.

This module supplies the identical image/text prefix to training, evaluation and
any later serving experiment. No episode, camera name, progress or history enters.
"""
import hashlib
import json
from pathlib import Path
import time
from collections import Counter

from PIL import Image
from modeling import eos_id
from prepare_visual_review import validate_collection, sha, CAMERAS, QUERIES

VERSION='visual-presence-v1'
REVIEW=Path(__file__).resolve().parents[2]/'configs/vlm_sft/h76_parent_visual_review_v1.json'
REVIEW_SHA='3159ed61cb56778343e360e282a9fbb68b64e5dd7d770d0346397fa1380dd136'
CLASSES={'P':'present','N':'absent','U':'uncertain'}
TARGETS={k:json.dumps({'visibility':v},separators=(',',':')) for k,v in CLASSES.items()}
SYSTEM='''Inspect only the supplied CURRENT_RAW image for the queried target. Return exactly one JSON object: {"visibility":"present"}, {"visibility":"absent"}, or {"visibility":"uncertain"}.
present: at least one visually identifiable target is visible, including identifiable partial views. absent: no identifiable target pixels are visible; this does NOT mean the object does not exist elsewhere or behind an occluder. uncertain: a candidate fragment or object is visible but its identity or a query qualifier cannot be confirmed from this image. If a plate is viewed from its back, do not assume food is on it. Robot parts are not task objects. Multiple matching objects still count as present; do not select a unique one. Do not infer holding, contact, motion, task progress, or success.'''


def load_rows(root):
    root=Path(root)
    source=validate_collection(root)
    if sha(REVIEW)!=REVIEW_SHA:raise ValueError('Exact parent-reviewed labels required')
    review=json.loads(REVIEW.read_text())
    if (review['source_manifest_sha256']!=sha(root/'review_manifest.json') or
            review['approved_for']!='bounded_visual_presence_pilot_only' or
            review['view_order']!=list(CAMERAS)):
        raise ValueError('Parent review does not approve these exact images')
    labels={r['id']:r for r in review['rows']}
    if len(labels)!=36 or len(review['rows'])!=36 or set(labels)!={r['id'] for r in source['rows']}:
        raise ValueError('Incomplete manual coverage')
    rows=[]
    for state in source['rows']:
        values=labels[state['id']]['labels']
        if len(values)!=3 or any(v not in CLASSES for v in values):raise ValueError('Missing/invalid visual label')
        for view,label in zip(CAMERAS,values):
            rows.append({'id':state['id']+'_'+view,'task':state['task'],'instance':state['instance'],
                         'view':view,'split':state['candidate_split'],'query':state['query'],
                         'path':str(root/state['images'][view]),'png_sha256':state['image_receipts'][view]['png_sha256'],
                         'label':label,'target':TARGETS[label]})
    if sum(r['split']=='visual_train' for r in rows)!=81 or sum(r['split']=='visual_validation' for r in rows)!=27:
        raise ValueError('Frozen instance split changed')
    return rows,{'protocol':VERSION,'manifest_sha256':sha(root/'review_manifest.json'),
                 'parent_review_sha256':REVIEW_SHA,'train_images':81,'validation_images':27,
                 'new_train_instances':9,'new_validation_instances':3,'action_labels':False}


def checked_image(row):
    if sha(Path(row['path']))!=row['png_sha256']:raise ValueError('Reviewed image changed')
    with Image.open(row['path']) as source:return source.convert('RGB')


def messages(query,image,*,blind=False):
    if query not in QUERIES.values():raise ValueError('Unregistered visual query')
    img=image.convert('RGB').copy()
    if img.size not in ((720,720),(480,480)):raise ValueError('Original onboard image required')
    img.thumbnail((640,640))  # Exactly serve_v2 sizing, including no wrist upsampling.
    if blind:img=Image.new('RGB',img.size,(128,128,128))
    msg=[{'role':'system','content':SYSTEM},{'role':'user','content':[
        {'type':'text','text':'CURRENT_RAW'},{'type':'image','image':img},
        {'type':'text','text':'Query: '+query}]}]
    receipt={'protocol':VERSION,'size':list(img.size),'blind':blind,
             'resized_pixels_sha256':hashlib.sha256(img.tobytes()).hexdigest()}
    return msg,receipt


def encode(processor,row,image,*,supervised=False,blind=False):
    import torch
    msg,receipt=messages(row['query'],image,blind=blind)
    prefix=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',enable_thinking=False)
    old=prefix['input_ids'].shape[1]
    if old>1800:raise ValueError('Visual context budget exceeded')
    receipt.update(input_tokens=old,input_ids_sha256=hashlib.sha256(prefix['input_ids'].numpy().tobytes()).hexdigest())
    if not supervised:return prefix,receipt
    if blind:raise ValueError('Blank images are evaluation-only, never supervised')
    target=row['target']
    if target!=TARGETS[row['label']]:raise ValueError('Unknown or inconsistent target')
    response=processor.tokenizer.encode(target,add_special_tokens=False)+[eos_id(processor)]
    if not 2<=len(response)<=32:raise ValueError('Unexpected target length')
    result={k:v.clone() for k,v in prefix.items()}
    result['input_ids']=torch.cat([prefix['input_ids'],torch.tensor([response],dtype=torch.long)],dim=1)
    for key in ('attention_mask','token_type_ids','mm_token_type_ids'):
        if key in result:
            value=1 if key=='attention_mask' else 0
            result[key]=torch.cat([result[key],torch.full((1,len(response)),value,dtype=result[key].dtype)],dim=1)
    labels=torch.full_like(result['input_ids'],-100);labels[:,old:]=result['input_ids'][:,old:]
    result['labels']=labels
    if not torch.equal(result['input_ids'][:,:old],prefix['input_ids']):raise RuntimeError('Training prefix changed')
    return result,receipt


def decode(model,processor,encoded):
    import torch
    inputs={k:v.to('cuda') for k,v in encoded.items() if k!='labels'}
    prefix=inputs['input_ids'].shape[1];eos=eos_id(processor);trie={}
    for target in TARGETS.values():
        node=trie
        for value in processor.tokenizer.encode(target,add_special_tokens=False):node=node.setdefault(value,{})
        node[eos]={}
    def allowed(_batch,ids):
        node=trie
        for value in ids[prefix:].tolist():node=node[value]
        return list(node) or [eos]
    torch.cuda.synchronize();start=time.monotonic()
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,
            prefix_allowed_tokens_fn=allowed,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    target=processor.tokenizer.decode(output[0,prefix:],skip_special_tokens=True).strip()
    reverse={v:k for k,v in TARGETS.items()}
    if target not in reverse or int(output[0,-1])!=eos:raise RuntimeError('Incomplete visual response')
    return {'prediction':reverse[target],'text':target,'latency_seconds':elapsed,'output_tokens':output.shape[1]-prefix}


def metrics(rows,predictions):
    if len({r['id'] for r in rows})!=len(rows) or set(predictions)!={r['id'] for r in rows}:
        raise ValueError('One prediction per exact heldout image required')
    confusion={k:{j:0 for j in CLASSES} for k in CLASSES}
    for row in rows:
        value=predictions[row['id']]
        if value not in CLASSES:raise ValueError('Unknown prediction')
        confusion[row['label']][value]+=1
    recalls={k:(v[k]/sum(v.values()) if sum(v.values()) else None) for k,v in confusion.items()}
    present=[v for v in recalls.values() if v is not None]
    return {'images':len(rows),'accuracy':sum(confusion[k][k] for k in CLASSES)/len(rows),
            'balanced_accuracy':sum(present)/len(present),'recall':recalls,'confusion':confusion,
            'full_task_success_rate_claim':False}


def train_prior_metrics(train,valid):
    majority=Counter(r['label'] for r in train).most_common(1)[0][0]
    key=lambda r:(r['query'],640 if r['view']=='head' else 480)
    priors={k:Counter(r['label'] for r in train if key(r)==k).most_common(1)[0][0] for k in {key(r) for r in train}}
    return {'majority_class_from_train':majority,'majority_baseline':metrics(valid,{r['id']:majority for r in valid}),
            'query_size_majority_baseline':metrics(valid,{r['id']:priors[key(r)] for r in valid})}
