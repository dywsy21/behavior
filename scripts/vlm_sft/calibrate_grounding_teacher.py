"""H81 bounded teacher calibration on already manually reviewed TRAIN images."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

from modeling import collate, eos_id, load_model
from prepare_visual_review import atomic_json,sha
from train_visual_presence import clean_commit,environment_identity,GPU_UUID,OVERLAY
from visual_presence import load_rows,checked_image,metrics
import visual_grounding as input_protocol
from visual_grounding import VERSION,decode_response

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h81_grounding_calibration_v1')
DATA=ROOT.parent/'h76_raw_review_v1'
MODEL_SPEC=REPO/'configs/semantic_robot/h79_model_identity.json'
CONFIG={'protocol':VERSION,'seed':41,'max_seconds':1800,'cleanup_seconds':30,'outer_seconds':1830,
        'cpu_ids':[56,57,58,59],'max_gpu_mib':71680,'allocator_mib':70656,
        'max_bytes':512*1024**2,'max_examples':97,'max_new_tokens':192,'primary_batch':4,'repeat_batch':8}
LABELS={'present':'P','absent':'N','uncertain':'U'}
PROFILE='h81'
BASE_CONFIG=CONFIG.copy()


def configure_profile(name):
    """Explicit CLI profile; historical H81 frozen worktrees are never changed."""
    global ROOT,CONFIG,PROFILE,input_protocol
    if name not in ('h81','h82','h83'):raise ValueError('Unregistered teacher profile')
    if name in ('h82','h83'):
        import reference_grounding as protocol
        ROOT=ROOT.parent/('h82_reference_calibration_v1' if name=='h82' else 'h83_dual_teacher_v1')
    else:
        import visual_grounding as protocol
        ROOT=ROOT.parent/'h81_grounding_calibration_v1'
    input_protocol=protocol;PROFILE=name
    CONFIG={**BASE_CONFIG,'protocol':protocol.VERSION}
    if name in ('h82','h83'):CONFIG['reference_spec_sha256']=sha(protocol.SPEC)
    if name=='h83':
        import dual_teacher_calibration as dual
        CONFIG.update(protocol=dual.VERSION,parent_review_sha256=sha(dual.REVIEW),max_seconds=2700,outer_seconds=2730,
                      max_examples=300,primary_batch=8,repeat_batch=0,max_bytes=1024**3)


def encode(processor,row,image):return input_protocol.encode(processor,row,image)


def input_messages(query,image):return input_protocol.messages(query,image)


def model_identity():
    spec=json.loads(MODEL_SPEC.read_text());base=Path(spec['path'])
    if spec['revision']!='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0':raise ValueError('Teacher revision changed')
    actual={p.name for p in base.iterdir() if p.is_file()}
    if actual!=set(spec['files']):raise ValueError('Teacher model file set changed')
    for name,item in spec['files'].items():
        path=base/name
        if path.is_symlink() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Teacher model identity changed: '+name)
    return spec


def training_rows():
    if PROFILE=='h83':
        from dual_teacher_calibration import load_rows as dual_rows
        return dual_rows()
    rows,identity=load_rows(DATA)
    selected=sorted((r for r in rows if r['split']=='visual_train'),key=lambda r:r['id'])
    if len(selected)!=81 or len({(r['task'],r['instance']) for r in selected})!=9:
        raise ValueError('Exactly 81 pre-reviewed TRAIN images only')
    # Repeat fixed hash-selected TRAIN cases to measure batching, not to search
    # for an answer or inflate independent-image counts.
    import hashlib
    repeated=sorted(selected,key=lambda r:hashlib.sha256(('h81-repeat-41:'+r['id']).encode()).hexdigest())[:16]
    if PROFILE=='h82':
        references=input_protocol.identity()
        groups={(r['task'],r['instance']) for r in selected}
        if any((r['task'],r['instance']) in groups for r in references['references']):
            raise ValueError('Reference instances overlap calibration sources')
        identity={**identity,'teacher_references':references}
    return selected,repeated,identity


def report(predictions,rows):
    valid={r['id']:LABELS[r['parsed']['visibility']] for r in predictions if r['format_valid']}
    wrong_format=[r['id'] for r in predictions if not r['format_valid']]
    truth={r['id']:r['label'] for r in rows}
    confusion={label:{p:0 for p in ('P','N','U','invalid')} for label in ('P','N','U')}
    for r in predictions:
        confusion[truth[r['id']]][LABELS[r['parsed']['visibility']] if r['format_valid'] else 'invalid']+=1
    by_class={k:{'true':sum(r['label']==k for r in rows),'predicted':sum(p==k for p in valid.values()),
                 'correct':sum(truth[i]==k and p==k for i,p in valid.items())} for k in LABELS.values()}
    for item in by_class.values():
        item['precision']=item['correct']/item['predicted'] if item['predicted'] else None
        item['recall']=item['correct']/item['true'] if item['true'] else None
    return {'total':len(rows),'format_valid':len(valid),'format_failures':wrong_format,
            'accuracy_including_invalid':sum(truth[i]==p for i,p in valid.items())/len(rows),
            'by_class':by_class,'confusion':confusion,'bbox_gold_available':False,'training_label_release':False,
            'full_task_success_rate_claim':False}


def run(output):
    if PROFILE=='h83':
        from dual_teacher_calibration import run as dual_run
        return dual_run(output)
    if output!=ROOT/'calibration' or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
        raise ValueError('Fixed calibration run and physical GPU2 only')
    output.mkdir(exist_ok=False);started=time.monotonic();examples=0
    def budget():
        if time.monotonic()-started>=CONFIG['max_seconds']:raise TimeoutError('Teacher calibration wall limit')
        if examples>CONFIG['max_examples']:raise RuntimeError('Teacher generation budget')
        if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>=CONFIG['max_bytes']:
            raise RuntimeError('Teacher output byte limit')
    try:
        launch=json.loads((ROOT/'launch.json').read_text());code=clean_commit()
        if launch['code_commit']!=code or launch['config']!=CONFIG:raise ValueError('Frozen teacher launch mismatch')
        import hashlib
        if hashlib.sha256(os.environ.get('H81_LAUNCH_TOKEN','').encode()).hexdigest()!=launch['token_sha256']:
            raise ValueError('Teacher supervisor token mismatch')
        environment=environment_identity();spec=model_identity();train,repeated,data=training_rows();budget()
        if spec!=launch['model_identity'] or data!=launch['dataset']:raise ValueError('Teacher launch identity changed')
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(41);random.seed(41)
        torch.cuda.set_per_process_memory_fraction(CONFIG['allocator_mib']*1024**2/torch.cuda.get_device_properties(0).total_memory)
        model,processor=load_model(spec['path']);model.requires_grad_(False);model.eval();eos=eos_id(processor);budget()
        atomic_json(output/'identity.json',{'code_commit':code,'config':CONFIG,'environment':environment,
            'model_identity':spec,'dataset':data,'train_ids':[r['id'] for r in train],
            'repeat_ids':[r['id'] for r in repeated],'training_updates':0,'new_controls':0})
        predictions=[];repeat_predictions=[];batches=[]
        with (output/'predictions.jsonl').open('x',buffering=1) as log:
            for phase,rows,batch_size in (('primary',train,CONFIG['primary_batch']),('repeat',repeated,CONFIG['repeat_batch'])):
                for offset in range(0,len(rows),batch_size):
                    budget();selected=rows[offset:offset+batch_size]
                    if examples+len(selected)>CONFIG['max_examples']:raise RuntimeError('Teacher example cap')
                    encoded=[encode(processor,r,checked_image(r)) for r in selected]
                    inputs={k:v.to('cuda') for k,v in collate([v[0] for v in encoded],processor.tokenizer.pad_token_id).items()}
                    prefix=inputs['input_ids'].shape[1];torch.cuda.synchronize();tick=time.monotonic()
                    examples+=len(selected)
                    with torch.inference_mode():
                        outputs=model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=CONFIG['max_new_tokens'],
                            eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id)
                    torch.cuda.synchronize();elapsed=time.monotonic()-tick;budget()
                    batches.append({'phase':phase,'ids':[r['id'] for r in selected],'seconds':elapsed,
                                    'batch_size':len(selected),'padded_input_tokens':prefix})
                    for row,(_,input_receipt),ids in zip(selected,encoded,outputs):
                        decoded=decode_response(processor,ids[prefix:].tolist())
                        value={'id':row['id'],'phase':phase,'query':row['query'],'expected_visibility':row['label'],
                            'png_sha256':row['png_sha256'],**input_receipt,**decoded,
                            'batch_seconds':elapsed,'training_eligible':False,'manual_box_review':'PENDING'}
                        log.write(json.dumps(value,separators=(',',':'))+'\n')
                        (predictions if phase=='primary' else repeat_predictions).append(value)
                    print(json.dumps({'phase':phase,'generated_examples':examples,'seconds':time.monotonic()-started,
                                      'batch_seconds':elapsed,'batch_size':len(selected)}),flush=True)
                    del inputs,outputs
        primary_by_id={p['id']:p for p in predictions}
        agreements=sum(p['parsed']==primary_by_id[p['id']]['parsed'] and p['format_valid'] for p in repeat_predictions)
        result={'status':'complete','code_commit':code,'config':CONFIG,'generated_examples':examples,
                'primary_unique_images':len(predictions),'repeated_images':len(repeat_predictions),
                'exact_repeat_agreements':agreements,'visibility':report(predictions,train),'batches':batches,
                'wall_seconds':time.monotonic()-started,'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                'predictions_sha256':sha(output/'predictions.jsonl'),'manual_box_review':'PENDING',
                'training_label_release':False,'training_updates':0,'new_controls':0}
        budget();atomic_json(output/'result.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        try:atomic_json(output/'failure.json',{'error':repr(error),'generated_examples':examples,
            'wall_seconds':time.monotonic()-started,'automatic_retry':False,'training_label_release':False})
        except BaseException as secondary:print('Secondary failure receipt: '+repr(secondary),flush=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--profile',choices=('h81','h82','h83'),default='h81')
    args=parser.parse_args();configure_profile(args.profile);run(args.output)
