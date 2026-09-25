"""One fresh 2B visual component pilot, 80 total updates incl numerical gate."""
import argparse
import importlib.metadata
import importlib.util
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from modeling import collate,load_model,supervised_loss
from prepare_visual_review import atomic_json,sha
from visual_presence import VERSION,load_rows,checked_image,encode,decode,metrics,eos_id,train_prior_metrics

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/semantic_robot'))
from probe_shared_vlm import validate_model_files
BASE='/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917/models/Qwen3.5-2B'
WEIGHT_SHA='aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1'
GPU_UUID='GPU-3e4fda8c-536e-5899-e877-b8be97032fe0'
ROOT=Path('/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h78_presence_v1')
DATA=ROOT.parent/'h76_raw_review_v1'
OVERLAY='/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917/deps'
PACKAGES={'torch':'2.7.1+cu128','transformers':'5.7.0','peft':'0.18.0',
          'Pillow':'10.2.0','numpy':'1.26.4','safetensors':'0.8.0'}
CONFIG={'protocol':VERSION,'lora_rank':8,'lora_alpha':16,'lora_dropout':.05,
        'learning_rate':5e-5,'seed':41,'updates':80,'effective_batch':8,'microbatch':2,
        'wall_seconds':1800,'max_generated_calls':112,'max_bytes':1024**3,'max_gpu_mib':24576}


def model_pythonpath():return OVERLAY+':'+str(REPO/'src')


def environment_identity():
    if os.environ.get('PYTHONPATH')!=model_pythonpath() or sys.version_info[:3]!=(3,10,18):
        raise ValueError('Exact isolated HF overlay/interpreter required')
    actual={k:importlib.metadata.version(k) for k in PACKAGES}
    if actual!=PACKAGES:raise ValueError('Pinned VLM package versions changed')
    origins={k:importlib.util.find_spec(k).origin for k in ('torch','transformers','peft')}
    for name,path in origins.items():
        expected=Path(OVERLAY if name=='transformers' else '/mnt/sdc1/robodojo/GalaxeaVLA/.venv/lib/python3.10/site-packages')/name/'__init__.py'
        if Path(path)!=expected:raise ValueError('Unexpected package origin: '+name)
    return {'python':'3.10.18','packages':actual,'origins':origins,'pythonpath':model_pythonpath()}


def model_identity():
    path=REPO/'configs/semantic_robot/h45_shared_small_vlm_probe.json'
    spec=json.loads(path.read_text())
    if (spec['model']!=BASE or spec['revision']!='15852e8c16360a2fea060d615a32b45270f8a8fc' or
            spec['model_files'].get('model.safetensors-00001-of-00001.safetensors')!=WEIGHT_SHA):
        raise ValueError('Registered fresh base model identity changed')
    validate_model_files(Path(BASE),spec['model_files'])
    return {'path':BASE,'revision':spec['revision'],'model_files':spec['model_files'],'spec_sha256':sha(path)}


def clean_commit():
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Frozen clean Git source required')
    return subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()


def run(output):
    if output!=ROOT/'training' or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
        raise ValueError('Registered run and physical GPU2 only')
    launch=json.loads((ROOT/'launch.json').read_text());code=clean_commit()
    import hashlib
    if (launch['code_commit']!=code or launch['config']!=CONFIG or
            hashlib.sha256(os.environ.get('H78_LAUNCH_TOKEN','').encode()).hexdigest()!=launch['token_sha256']):
        raise ValueError('Missing exact supervisor launch receipt')
    output.mkdir(exist_ok=False);started=time.monotonic();steps=0;calls=0
    def budget():
        if time.monotonic()-started>=CONFIG['wall_seconds']:raise TimeoutError('Visual pilot wall budget')
        if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>=CONFIG['max_bytes']:
            raise RuntimeError('Visual pilot artifact budget')
        if shutil.disk_usage('/mnt/nvme_tmp').free<80*1024**3:raise RuntimeError('Disk reserve')
        if calls>CONFIG['max_generated_calls']:raise RuntimeError('Generated call budget')
    try:
        rows,dataset=load_rows(DATA);budget()
        base_identity=model_identity();environment=environment_identity()
        if base_identity!=launch['model_identity'] or environment!=launch['environment']:
            raise ValueError('Base model/environment differs from preflight')
        import torch,transformers,peft
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(41)
        torch.cuda.set_per_process_memory_fraction(23552*1024**2/torch.cuda.get_device_properties(0).total_memory)
        model,processor=load_model(BASE,train=True,cfg=CONFIG);budget()
        if any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if 'lora_B' in n):
            raise RuntimeError('Initial adapter is not fresh zero-effect LoRA')
        train=[r for r in rows if r['split']=='visual_train'];valid=[r for r in rows if r['split']=='visual_validation']
        atomic_json(output/'identity.json',{'code_commit':code,'config':CONFIG,'data':dataset,'environment':environment,
            'base_model_identity':base_identity,'base_model':BASE,'base_weight_sha256':WEIGHT_SHA,'base_config_sha256':sha(Path(BASE)/'config.json'),
            'tokenizer_sha256':sha(Path(BASE)/'tokenizer.json'),'torch':torch.__version__,
            'transformers':transformers.__version__,'peft':peft.__version__,'old_adapter_loaded':False,
            'train_counts':dict(Counter(r['label'] for r in train)),'physical_gpu':2,
            'started_unix':time.time(),'images_per_input':1,'action_or_success_labels':False})
        cache={};mask=[]
        for row in train:
            item,receipt=encode(processor,row,checked_image(row),supervised=True);budget()
            selected=item['labels'][item['labels']!=-100]
            if processor.tokenizer.decode(selected.tolist(),skip_special_tokens=True)!=row['target'] or int(selected[-1])!=eos_id(processor):
                raise RuntimeError('Assistant-only EOS mask failed')
            cache[row['id']]=item;mask.append({'id':row['id'],**receipt,'supervised_tokens':len(selected)})
        atomic_json(output/'mask_gate.json',mask)
        def batch(selected):
            return {k:v.to('cuda') for k,v in collate([cache[r['id']] for r in selected],processor.tokenizer.pad_token_id).items()}
        ordered=sorted(train,key=lambda r:cache[r['id']]['input_ids'].shape[1]);gate=[ordered[0],ordered[-1]]
        model.eval();b=batch(gate)
        with torch.no_grad():
            native=model(**b,use_cache=False).loss;custom=supervised_loss(model,b)
        if not torch.allclose(native,custom,rtol=1e-5,atol=1e-4):raise RuntimeError('Native/custom CE disagreement')
        atomic_json(output/'loss_gate.json',{'native':float(native),'custom':float(custom),
            'absolute_error':float((native-custom).abs()),'ids':[r['id'] for r in gate],
            'left_padding':bool((b['attention_mask']==0).any()),'all_labels_reviewed':True})
        del b,native,custom
        def predict(active,row,blind):
            nonlocal calls
            budget()
            if calls>=CONFIG['max_generated_calls']:raise RuntimeError('Generated call budget')
            x,receipt=encode(processor,row,checked_image(row),blind=blind)
            calls+=1;answer=decode(active,processor,x);budget()
            return {'id':row['id'],'expected':row['label'],**receipt,**answer}
        def evaluate(name,blind):
            model.eval();predictions=[predict(model,r,blind) for r in valid]
            result={'step':steps,'blind':blind,'predictions':predictions,
                    'metrics':metrics(valid,{r['id']:r['prediction'] for r in predictions})}
            atomic_json(output/(name+'.json'),result);return result['metrics']
        base=evaluate('base_images',False);base_blind=evaluate('base_gray',True)
        params=[p for p in model.parameters() if p.requires_grad]
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        optimizer=torch.optim.AdamW(params,lr=CONFIG['learning_rate'],betas=(.9,.95),weight_decay=.01)
        sampler=random.Random(41);checkpoints=[]
        def save(step):
            budget();folder=output/f'adapter_{step:04d}'
            if folder.exists():raise FileExistsError('No checkpoint replacement')
            model.save_pretrained(folder,safe_serialization=True);budget()
            checkpoints.append({'step':step,'path':str(folder),'weight_sha256':sha(folder/'adapter_model.safetensors'),
                                'config_sha256':sha(folder/'adapter_config.json')})
            return folder
        with (output/'steps.jsonl').open('x',buffering=1) as log:
            for step in range(1,CONFIG['updates']+1):
                budget();model.train();optimizer.zero_grad(set_to_none=True);losses=[];draws=[]
                for _ in range(CONFIG['effective_batch']//CONFIG['microbatch']):
                    selected=sampler.sample(train,CONFIG['microbatch']);draws.extend(r['id'] for r in selected)
                    loss=supervised_loss(model,batch(selected))
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    (loss/4).backward();losses.append(float(loss.detach()));budget()
                grad=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(grad) or float(grad)==0:raise RuntimeError('Invalid LoRA gradient')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError('Frozen model received gradient')
                optimizer.step();steps=step
                if torch.cuda.max_memory_reserved()>CONFIG['max_gpu_mib']*1024**2:raise RuntimeError('Own GPU memory cap')
                record={'step':step,'loss':sum(losses)/len(losses),'gradient_norm':float(grad),
                        'elapsed_seconds':time.monotonic()-started,'sample_ids':draws}
                log.write(json.dumps(record)+'\n')
                if step<=2 or step%10==0:print(json.dumps({k:v for k,v in record.items() if k!='sample_ids'}),flush=True)
                if step==2:
                    changed=sum(not torch.equal(p,before[n]) for n,p in model.named_parameters() if p.requires_grad)
                    if not changed:raise RuntimeError('Optimizer made no LoRA update')
                    del before;folder=save(step);model.eval()
                    x,_=encode(processor,gate[0],checked_image(gate[0]));inputs={k:v.to('cuda') for k,v in x.items()}
                    with torch.inference_mode():old=model(**inputs,use_cache=False,logits_to_keep=1).logits.float().cpu()
                    first=predict(model,gate[0],False)
                    restored,_=load_model(BASE,adapter=folder)
                    with torch.inference_mode():new=restored(**inputs,use_cache=False,logits_to_keep=1).logits.float().cpu()
                    second=predict(restored,gate[0],False)
                    if not torch.allclose(old,new,rtol=0,atol=1e-4) or first['prediction']!=second['prediction']:
                        raise RuntimeError('Saved adapter restore differs')
                    atomic_json(output/'restore_gate.json',{'passed':True,'changed_lora_tensors':changed,
                        'max_logit_error':float((old-new).abs().max()),'before':first,'restored':second})
                    del restored,old,new,inputs;torch.cuda.empty_cache()
        save(steps);model.eval();after=evaluate('finetuned_images',False);after_blind=evaluate('finetuned_gray',True);budget()
        atomic_json(output/'result.json',{'status':'complete','code_commit':code,'optimizer_updates':steps,
            'generated_calls':calls,'wall_seconds':time.monotonic()-started,'config':CONFIG,'dataset':dataset,'base_model_identity':base_identity,'environment':environment,
            'base':base,'base_gray':base_blind,'finetuned':after,'finetuned_gray':after_blind,
            **train_prior_metrics(train,valid),
            'checkpoints':checkpoints,'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
            'online_actor_evaluated':False,'full_task_success_rate_claim':False})
        print(json.dumps({'complete':True,'updates':steps,'base':base,'finetuned':after}),flush=True)
    except BaseException as error:
        try:atomic_json(output/'failure.json',{'error':repr(error),'optimizer_updates':steps,
            'generated_calls':calls,'wall_seconds':time.monotonic()-started,'automatic_retry':False})
        except BaseException as secondary:print('Failure receipt error: '+repr(secondary),flush=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
