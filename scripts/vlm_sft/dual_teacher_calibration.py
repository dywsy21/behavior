"""H83: selective agreement on fresh, parent-reviewed TRAIN source groups."""
from collections import Counter
from functools import lru_cache
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from prepare_visual_review import sha,atomic_json
from modeling import collate,eos_id,load_model
import reference_grounding as references
import visual_presence as presence
from visual_grounding import decode_response
import train_visual_presence as small

REPO=Path(__file__).resolve().parents[2]
REVIEW=REPO/'configs/vlm_sft/h83_parent_dual_teacher_review_v1.json'
VERSION='dual-teacher-grounding-selection-v1'
ADAPTER=Path('/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h78_presence_v1/training/adapter_0080')
ADAPTER_FILES={'adapter_model.safetensors':'0e1a239675ded85922faea3ab728cbd31646a6fb852ad1195622c3555349396f',
               'adapter_config.json':'4de2f8ce06bd216ffa31694dcb96f6c1d80f800478860207feff5c87f6ec62fd'}
LABELS={'present':'P','absent':'N','uncertain':'U'}


def critic_identity():
    base=small.model_identity()
    if {p.name for p in ADAPTER.iterdir() if p.is_file()}-set(ADAPTER_FILES)-{'README.md'}:
        raise ValueError('Unexpected critic adapter files')
    for name,digest in ADAPTER_FILES.items():
        p=ADAPTER/name
        if p.is_symlink() or sha(p)!=digest:raise ValueError('H78 critic adapter changed')
    return {'base':base,'adapter':str(ADAPTER),'adapter_files':ADAPTER_FILES,'optimizer_updates':80}


def validate_review(review,plan,states,first_review,reference_identity):
    if (review.get('schema')!='h83-dual-teacher-review-v1' or
            review.get('approved_for')!='bounded_dual_teacher_calibration_only' or
            review.get('training_eligible') is not False or review.get('teacher_outputs_seen_before_labeling') is not False or
            review.get('heldout_validation_or_test_used') is not False or
            review.get('views')!=['head','left_wrist','right_wrist']):
        raise ValueError('Exact pre-model parent review required')
    excluded=set(map(tuple,plan['source_identity']['historical_train_groups']))|{(g['task'],g['instance']) for g in first_review['groups']}
    if set(map(tuple,review['excluded_groups']))!=excluded:raise ValueError('Historical/references exclusion changed')
    groups={}
    for r in states.values():
        key=(r['task'],r['instance'])
        if r['split']=='train' and key not in excluded:groups.setdefault(key,[]).append(r)
    expected=[]
    for task in range(5):
        chosen=sorted((g for g in groups if g[0]==task),key=lambda g:hashlib.sha256(f'h80-parent-review-41:{g}'.encode()).hexdigest())[:2]
        if len(chosen)!=2:raise ValueError('Missing fresh task sources')
        for group in chosen:
            ordered=sorted(groups[group],key=lambda r:r['frame'])
            expected.extend(ordered[round(k*(len(ordered)-1)/4)]['id'] for k in range(5))
    selected=review['rows']
    if len(selected)!=50 or [r['id'] for r in selected]!=expected:raise ValueError('Fresh review schedule changed')
    reference_groups={(r['task'],r['instance']) for r in reference_identity['references']}
    selected_groups={(r['task'],r['instance']) for r in selected}
    if selected_groups & reference_groups or set(map(tuple,review['protect_groups_in_future_student_releases']))!=selected_groups:
        raise ValueError('Reference leakage or missing future student exclusion')
    queries={0:'the radio',1:'the wastebasket',2:'the radio',3:'the plate holding food',4:'the radio'}
    counts=Counter();rows=[]
    for r in selected:
        original=states[r['id']]
        if (any(r[k]!=original[k] for k in ('task','instance','episode','frame','split','images')) or
                r['split']!='train' or (r['task'],r['instance']) in excluded or
                r['query']!=queries[r['task']] or r['stratum']!=('core_targets' if r['task'] in (0,1,3) else 'cross_task_negative_control') or
                len(r['labels'])!=3 or any(label not in 'PNU' or len(label)!=1 for label in r['labels'])):
            raise ValueError('Unreviewed or altered raw state/query/label')
        for view,label in zip(review['views'],r['labels']):
            if r['png_sha256'][view]!=original['image_receipts'][view]['png_sha256']:
                raise ValueError('Gold source image differs from raw ledger')
            counts[label]+=1
            rows.append({'id':r['id']+'_'+view,'task':r['task'],'instance':r['instance'],'view':view,
                'query':r['query'],'label':label,'stratum':r['stratum'],'split':'label_quality_development',
                'path':str(Path(review['raw_root'])/r['images'][view]),'png_sha256':r['png_sha256'][view]})
    if counts!=review['counts'] or len(rows)!=150 or review['image_count']!=150 or (review['core_images'],review['control_images'])!=(90,60):
        raise ValueError('Review totals changed')
    return sorted(rows,key=lambda r:r['id'])


@lru_cache(maxsize=1)
def load_rows():
    review=json.loads(REVIEW.read_text());root=Path(review['raw_root']);refs=references.identity()
    if sha(root/'manifest.json')!=review['source_manifest_sha256'] or refs['source_manifest_sha256']!=review['source_manifest_sha256']:
        raise ValueError('Fresh gold is not from sealed H80')
    manifest=json.loads((root/'manifest.json').read_text())
    if sha(root/'source_plan.json')!=manifest['source_plan_sha256'] or sha(root/'images.jsonl')!=manifest['images_manifest_sha256']:
        raise ValueError('Sealed raw metadata changed')
    plan=json.loads((root/'source_plan.json').read_text())
    original=[json.loads(x) for x in (root/'images.jsonl').read_text().splitlines()]
    states={r['id']:r for r in original}
    if len(states)!=len(original):raise ValueError('Duplicate raw states')
    rows=validate_review(review,plan,states,json.loads(references.RAW_REVIEW.read_text()),refs)
    identity={'protocol':VERSION,'parent_review_sha256':sha(REVIEW),'manifest_sha256':review['source_manifest_sha256'],
        'core_images':90,'control_images':60,'counts':review['counts'],'teacher_references':refs,
        'critic_identity':critic_identity(),'future_student_excluded_groups':review['protect_groups_in_future_student_releases'],
        'training_label_release':False}
    return rows,[],identity


def presence_tokens(processor):
    eos=eos_id(processor)
    return {tuple(processor.tokenizer.encode(text,add_special_tokens=False)+[eos]):label for label,text in presence.TARGETS.items()}


def decode_presence(processor,generated):
    eos=eos_id(processor);pad=processor.tokenizer.pad_token_id;text='';error=None;parsed=None
    stop=generated.index(eos)+1 if eos in generated else len(generated);tokens=generated[:stop]
    try:
        if (type(generated) is not list or not 0<len(generated)<=32 or eos not in generated or
                any(type(i) is not int or not 0<=i<len(processor.tokenizer) for i in generated) or
                any(i!=pad for i in generated[stop:])):raise ValueError('Invalid or truncated presence tokens')
        label=presence_tokens(processor).get(tuple(tokens))
        if label is None:raise ValueError('Presence response differs from registered native-token target')
        text=processor.tokenizer.decode(tokens[:-1],skip_special_tokens=False).strip()
        if text!=presence.TARGETS[label]:raise ValueError('Unexpected presence tokenizer decode')
        parsed={'visibility':presence.CLASSES[label]}
    except (ValueError,TypeError) as caught:error=repr(caught)
    return {'text':text,'parsed':parsed,'format_valid':error is None,'error':error,'has_eos':eos in generated,
            'output_tokens':stop,'generated_token_ids':generated}


def constrained_presence(processor,prefix):
    trie={};eos=eos_id(processor)
    for sequence in presence_tokens(processor):
        node=trie
        for token in sequence:node=node.setdefault(token,{})
    def allowed(_batch,ids):
        node=trie
        for token in ids[prefix:].tolist():
            if token not in node:
                if int(token)==processor.tokenizer.pad_token_id:return [processor.tokenizer.pad_token_id]
                raise ValueError('Unregistered presence generation prefix')
            node=node[int(token)]
        return list(node) or [eos]
    return allowed


def agreement_report(grounding,critic,rows):
    by_id={r['id']:r for r in critic};proposals=[]
    for g in grounding:
        c=by_id[g['id']];a=LABELS[g['parsed']['visibility']] if g['format_valid'] else None
        b=LABELS[c['parsed']['visibility']] if c['format_valid'] else None
        selected=a==b and a in ('P','N')
        proposals.append({'id':g['id'],'label':a if selected else None,'accepted_candidate':selected,
            'reason':'strict_agreement' if selected else ('invalid' if a is None or b is None else 'uncertain_or_disagreement'),
            'boxes':g['parsed']['boxes'] if selected and a=='P' else [],'training_eligible':False})
    truth={r['id']:r for r in rows};stats={}
    for stratum in ('core_targets','cross_task_negative_control'):
        members=[p for p in proposals if truth[p['id']]['stratum']==stratum]
        accepted=[p for p in members if p['accepted_candidate']]
        classes={}
        for label in ('P','N'):
            chosen=[p for p in accepted if p['label']==label]
            correct=sum(truth[p['id']]['label']==label for p in chosen)
            classes[label]={'accepted':len(chosen),'correct':correct,'precision':correct/len(chosen) if chosen else None}
        stats[stratum]={'images':len(members),'accepted':len(accepted),'coverage':len(accepted)/len(members),
            'classes':classes,'errors':[p['id'] for p in accepted if p['label']!=truth[p['id']]['label']]}
    core=stats['core_targets']
    passed=core['coverage']>=.5 and all(core['classes'][k]['accepted']>=20 and core['classes'][k]['precision']>=.95 for k in ('P','N'))
    return {'strata':stats,'visibility_candidate_gate_passed':passed,'manual_box_gate':'PENDING',
            'training_label_release':False,'proposals':proposals}


def run(output,*,worker):
    cfg=worker.CONFIG;root=worker.ROOT
    if output!=root/'calibration' or os.environ.get('CUDA_VISIBLE_DEVICES')!=worker.GPU_UUID:
        raise ValueError('Registered H83 output and physical GPU2 only')
    output.mkdir(exist_ok=False);started=time.monotonic();examples=0
    def budget():
        if time.monotonic()-started>=cfg['max_seconds'] or examples>300:raise TimeoutError('H83 wall/example budget')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>=cfg['max_bytes']:raise RuntimeError('H83 artifact budget')
    try:
        launch=json.loads((root/'launch.json').read_text());code=worker.clean_commit();env=worker.environment_identity()
        if (launch['code_commit']!=code or launch['config']!=cfg or launch['environment']!=env or
                hashlib.sha256(os.environ.get('H81_LAUNCH_TOKEN','').encode()).hexdigest()!=launch['token_sha256']):
            raise ValueError('H83 launch identity mismatch')
        spec=worker.model_identity();rows,_,data=load_rows()
        if spec!=launch['model_identity'] or data!=launch['dataset']:raise ValueError('H83 data/models changed')
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(41);random.seed(41)
        torch.cuda.set_per_process_memory_fraction(cfg['allocator_mib']*1024**2/torch.cuda.get_device_properties(0).total_memory)
        atomic_json(output/'identity.json',{'code_commit':code,'config':cfg,'environment':env,'dataset':data,
            'model_identity':spec,'ids':[r['id'] for r in rows],'training_updates':0,'new_controls':0})
        predictions={};batches=[];unloads=[]
        with (output/'predictions.jsonl').open('x',buffering=1) as log:
            for phase in ('grounding','presence'):
                budget()
                if phase=='grounding':model,processor=load_model(spec['path']);encoder=references.encode;decoder=decode_response;max_tokens=192
                else:
                    if critic_identity()!=data['critic_identity']:raise ValueError('Critic changed before use')
                    model,processor=load_model(small.BASE,adapter=ADAPTER);encoder=presence.encode;decoder=decode_presence;max_tokens=32
                model.requires_grad_(False);model.eval();predictions[phase]=[];budget()
                for offset in range(0,len(rows),8):
                    selected=rows[offset:offset+8];budget()
                    if examples+len(selected)>300:raise RuntimeError('H83 generation cap')
                    encoded=[encoder(processor,r,presence.checked_image(r)) for r in selected]
                    inputs={k:v.to('cuda') for k,v in collate([v[0] for v in encoded],processor.tokenizer.pad_token_id).items()}
                    prefix=inputs['input_ids'].shape[1];extra={}
                    if phase=='presence':extra['prefix_allowed_tokens_fn']=constrained_presence(processor,prefix)
                    torch.cuda.synchronize();tick=time.monotonic();examples+=len(selected)
                    with torch.inference_mode():
                        output_ids=model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=max_tokens,
                            eos_token_id=eos_id(processor),pad_token_id=processor.tokenizer.pad_token_id,**extra)
                    torch.cuda.synchronize();elapsed=time.monotonic()-tick;budget()
                    batches.append({'phase':phase,'ids':[r['id'] for r in selected],'seconds':elapsed,
                                    'batch_size':len(selected),'padded_input_tokens':prefix})
                    for row,(_,receipt),ids in zip(selected,encoded,output_ids):
                        value={'id':row['id'],'phase':phase,'query':row['query'],'expected_visibility':row['label'],
                            'stratum':row['stratum'],'png_sha256':row['png_sha256'],**receipt,
                            **decoder(processor,ids[prefix:].tolist()),'batch_seconds':elapsed,
                            'training_eligible':False,'manual_box_review':'PENDING'}
                        log.write(json.dumps(value,separators=(',',':'))+'\n');predictions[phase].append(value)
                    print(json.dumps({'phase':phase,'generated_examples':examples,'seconds':time.monotonic()-started,
                        'batch_seconds':elapsed,'batch_size':len(selected)}),flush=True)
                    del inputs,output_ids
                del model,processor,encoded,ids;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();budget()
                memory={'phase':phase,'allocated_mib':torch.cuda.memory_allocated()/1024**2,
                        'reserved_mib':torch.cuda.memory_reserved()/1024**2}
                if memory['allocated_mib']>1024:raise RuntimeError('Previous teacher remains allocated before next phase')
                unloads.append(memory)
        result={'status':'complete','code_commit':code,'config':cfg,'generated_examples':examples,
            'unique_images':150,'wall_seconds':time.monotonic()-started,'batches':batches,
            'model_unloads':unloads,
            'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'predictions_sha256':sha(output/'predictions.jsonl'),
            'grounding':worker.report(predictions['grounding'],rows),'critic':worker.report(predictions['presence'],rows),
            'agreement':agreement_report(predictions['grounding'],predictions['presence'],rows),
            'training_label_release':False,'training_updates':0,'new_controls':0}
        budget();atomic_json(output/'result.json',result)
        print(json.dumps({'status':'complete','generated_examples':examples,'agreement':result['agreement']['strata'],
            'visibility_gate_passed':result['agreement']['visibility_candidate_gate_passed']}),flush=True)
    except BaseException as error:
        try:atomic_json(output/'failure.json',{'error':repr(error),'generated_examples':examples,
            'wall_seconds':time.monotonic()-started,'automatic_retry':False,'training_label_release':False})
        except BaseException as secondary:print('Failure receipt error: '+repr(secondary),flush=True)
        raise


def load_processor(path):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path,local_files_only=True)


def validate_result(root,code,cfg):
    import calibrate_grounding_teacher as worker
    folder=root/'calibration';launch=json.loads((root/'launch.json').read_text())
    if (folder/'failure.json').exists():raise ValueError('H83 worker failure recorded')
    result=json.loads((folder/'result.json').read_text());identity=json.loads((folder/'identity.json').read_text())
    rows,_,data=load_rows()
    if identity!={'code_commit':code,'config':cfg,'environment':launch['environment'],'dataset':data,
        'model_identity':launch['model_identity'],'ids':[r['id'] for r in rows],'training_updates':0,'new_controls':0}:
        raise ValueError('H83 identity receipt changed')
    if (result['status']!='complete' or result['code_commit']!=code or result['config']!=cfg or result['generated_examples']!=300 or
            result['unique_images']!=150 or result['training_label_release'] is not False or result['training_updates']!=0 or
            result['new_controls']!=0 or sha(folder/'predictions.jsonl')!=result['predictions_sha256']):
        raise ValueError('Incomplete H83 result')
    for key,limit in (('wall_seconds',cfg['max_seconds']),('cuda_peak_reserved_mib',cfg['max_gpu_mib'])):
        value=result[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=limit:raise ValueError('H83 budget exceeded')
    predictions=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
    if [(p['phase'],p['id']) for p in predictions]!=[(phase,r['id']) for phase in ('grounding','presence') for r in rows]:
        raise ValueError('H83 call schedule changed')
    by_phase={}
    for phase,path in (('grounding',launch['model_identity']['path']),('presence',small.BASE)):
        processor=load_processor(path)
        encoder=references.encode if phase=='grounding' else presence.encode
        decoder=decode_response if phase=='grounding' else decode_presence
        members=[p for p in predictions if p['phase']==phase];by_phase[phase]=members
        for row,p in zip(rows,members):
            _,receipt=encoder(processor,row,presence.checked_image(row));decoded=decoder(processor,p['generated_token_ids'])
            expected={'id':row['id'],'phase':phase,'query':row['query'],'expected_visibility':row['label'],
                'stratum':row['stratum'],'png_sha256':row['png_sha256'],**receipt,**decoded,
                'batch_seconds':p['batch_seconds'],'training_eligible':False,'manual_box_review':'PENDING'}
            if p!=expected:raise ValueError('H83 real input/token/response provenance differs')
        del processor
    if result['grounding']!=worker.report(by_phase['grounding'],rows) or result['critic']!=worker.report(by_phase['presence'],rows):
        raise ValueError('H83 raw confusion matrices differ')
    if result['agreement']!=agreement_report(by_phase['grounding'],by_phase['presence'],rows):raise ValueError('H83 selection statistics differ')
    expected_batches=[(phase,[r['id'] for r in rows[o:o+8]]) for phase in ('grounding','presence') for o in range(0,len(rows),8)]
    if [(b['phase'],b['ids']) for b in result['batches']]!=expected_batches:raise ValueError('H83 batch schedule changed')
    indexed={(p['phase'],p['id']):p for p in predictions}
    for b in result['batches']:
        members=[indexed[(b['phase'],i)] for i in b['ids']]
        if (set(b)!={'phase','ids','seconds','batch_size','padded_input_tokens'} or
                type(b['batch_size']) is not int or b['batch_size']!=len(members) or
                type(b['padded_input_tokens']) is not int or not 0<b['padded_input_tokens']<=1800 or
                type(b['seconds']) not in (float,int) or not math.isfinite(b['seconds']) or not 0<b['seconds']<=cfg['max_seconds'] or
                b['padded_input_tokens']!=max(p['input_tokens'] for p in members) or any(p['batch_seconds']!=b['seconds'] for p in members)):
            raise ValueError('Invalid H83 batch throughput')
    if sum(b['seconds'] for b in result['batches'])>result['wall_seconds']:raise ValueError('H83 generation exceeds total wall')
    unloads=result.get('model_unloads',[])
    if [u.get('phase') for u in unloads]!=['grounding','presence']:raise ValueError('Missing sequential teacher release proof')
    for u in unloads:
        if set(u)!={'phase','allocated_mib','reserved_mib'}:raise ValueError('Unexpected release receipt fields')
        for name,limit in (('allocated_mib',1024),('reserved_mib',cfg['max_gpu_mib'])):
            if type(u[name]) not in (int,float) or not math.isfinite(u[name]) or not 0<=u[name]<=limit:
                raise ValueError('Teacher model did not release its allocation')
    return result
