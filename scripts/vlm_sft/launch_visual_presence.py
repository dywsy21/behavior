"""Exclusive, resource-guarded H78 training/evaluation supervisor; no retries."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from collections import Counter
import os
from pathlib import Path
import secrets
import random
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'semantic_robot'))
from launch_h69 import belongs_to_session,stop_owned_group
from probe_simulator_startup import snapshot,GPU_UUIDS
from train_visual_presence import ROOT,DATA,CONFIG,GPU_UUID,clean_commit,model_identity,environment_identity,model_pythonpath
from visual_presence import load_rows,metrics,messages,checked_image,TARGETS,VERSION,train_prior_metrics
from prepare_visual_review import atomic_json,sha

PYTHON=Path('/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python')
REPO=Path(__file__).resolve().parents[2]
H77=Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h77_task3_gate_v1/supervisor.json')


def check_resources(current,baseline=None,sid=None,*,released=False):
    if set(current)!=set(GPU_UUIDS):raise ValueError('Physical GPU set changed')
    for uuid,row in current.items():
        owned=[p for p in row['processes'] if belongs_to_session(p['pid'],sid)]
        if released and owned:raise RuntimeError('Own GPU session not released')
        if uuid==GPU_UUID:
            if baseline is None:
                if row['processes'] or row['free_mib']<32768:raise RuntimeError('GPU2 not available')
            elif any(not belongs_to_session(p['pid'],sid) for p in row['processes']):
                raise RuntimeError('Unknown GPU2 process; stop only our worker')
            if sum(p['used_mib'] for p in owned)>CONFIG['max_gpu_mib']:
                raise RuntimeError('Own GPU memory cap')
            if baseline is not None and row['used_mib']-baseline[uuid]['used_mib']>CONFIG['max_gpu_mib']:
                raise RuntimeError('GPU2 allocation delta cap')
        elif owned:
            raise RuntimeError('Training touched an unallocated GPU')
        if row['free_mib']<8192:raise RuntimeError('Shared GPU headroom')


def preflight():
    code=clean_commit()
    environment_identity()
    if Path(sys.executable)!=PYTHON:raise ValueError('Registered VLM interpreter required')
    ended=json.loads(H77.read_text())
    if ended.get('status') not in ('completed','failed') or not ended.get('after_exit'):
        raise RuntimeError('H77 is still active; do not conflict with its GPU2 exclusion')
    if any(belongs_to_session(p['pid'],ended.get('worker_pid')) for r in snapshot().values() for p in r['processes']):
        raise RuntimeError('H77 worker still owns a GPU')
    _rows,data=load_rows(DATA);before=snapshot();check_resources(before)
    return code,data,before


def command():
    return [str(PYTHON),str(REPO/'scripts/vlm_sft/train_visual_presence.py'),'--output',str(ROOT/'training')]


def launch():
    code,data,before=preflight()
    base_identity=model_identity()
    if ROOT.exists():raise FileExistsError('H78 already reserved; no implicit retry')
    ROOT.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',
             H78_LAUNCH_TOKEN=secrets.token_hex(24))
    env['PYTHONPATH']=model_pythonpath()  # Never inherit unrelated user overlays.
    for key,name in (('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor')):
        path=ROOT/'cache'/name;path.mkdir(parents=True);env[key]=str(path)
    receipt={'status':'reserved','code_commit':code,'config':CONFIG,'dataset':data,'baseline':before,'model_identity':base_identity,
             'utc':datetime.now(timezone.utc).isoformat(),'command':command(),'environment':environment_identity(),
             'launcher_sha256':sha(Path(__file__)),'token_sha256':hashlib.sha256(env['H78_LAUNCH_TOKEN'].encode()).hexdigest()}
    atomic_json(ROOT/'launch.json',receipt)
    with (ROOT/'supervisor.log').open('x') as log:
        child=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--supervise'],cwd=REPO,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(status='supervisor_started',supervisor_pid=child.pid)
    atomic_json(ROOT/'launch.json',receipt);print(json.dumps(receipt),flush=True)


def validate_result(result,code):
    def finite(value,lower=0,upper=1800):
        return type(value) in (int,float) and math.isfinite(value) and lower<=value<=upper
    def is_sha(value):
        return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')
    if ((ROOT/'training/failure.json').exists() or result.get('status')!='complete' or
            result.get('code_commit')!=code or type(result.get('optimizer_updates')) is not int or
            result['optimizer_updates']!=80 or result.get('generated_calls')!=110 or
            result.get('config')!=CONFIG or result.get('full_task_success_rate_claim') is not False or
            result.get('online_actor_evaluated') is not False or not finite(result.get('wall_seconds')) or
            not finite(result.get('cuda_peak_reserved_mib'),0,24576)):
        raise ValueError('Incomplete, failed or misidentified visual pilot')
    launch_receipt=json.loads((ROOT/'launch.json').read_text())
    if (result.get('dataset')!=launch_receipt['dataset'] or result.get('base_model_identity')!=launch_receipt['model_identity'] or
            result.get('environment')!=launch_receipt['environment']):
        raise ValueError('Result data/base binding changed')
    rows,_=load_rows(DATA);train=[r for r in rows if r['split']=='visual_train'];valid=[r for r in rows if r['split']=='visual_validation']
    train_by_id={r['id']:r for r in train};valid_by_id={r['id']:r for r in valid}
    folder=ROOT/'training';identity=json.loads((folder/'identity.json').read_text())
    exact={'code_commit':code,'config':CONFIG,'data':launch_receipt['dataset'],
           'base_model_identity':launch_receipt['model_identity'],'environment':launch_receipt['environment'],'old_adapter_loaded':False,
           'train_counts':dict(Counter(r['label'] for r in train)),'physical_gpu':2,
           'images_per_input':1,'action_or_success_labels':False}
    if any(json.dumps(identity.get(k),sort_keys=True)!=json.dumps(v,sort_keys=True) for k,v in exact.items()):
        raise ValueError('Worker identity does not bind data/model/config')
    mask=json.loads((folder/'mask_gate.json').read_text())
    if len(mask)!=81 or {r['id'] for r in mask}!=set(train_by_id):raise ValueError('Incomplete real mask gate')
    def check_image_receipt(p,row,blind):
        _,expected=messages(row['query'],checked_image(row),blind=blind)
        if any(type(p.get(k)) is not type(v) or p[k]!=v for k,v in expected.items()):
            raise ValueError('Wrong image/gray/size/protocol receipt')
        if not is_sha(p.get('input_ids_sha256')) or type(p.get('input_tokens')) is not int or not 1<=p['input_tokens']<=1800:
            raise ValueError('Missing exact input-prefix receipt')
    for row in mask:
        check_image_receipt(row,train_by_id[row['id']],False)
        if type(row.get('supervised_tokens')) is not int or not 2<=row['supervised_tokens']<=32:
            raise ValueError('Missing response/EOS supervision gate')
    loss=json.loads((folder/'loss_gate.json').read_text())
    if (not finite(loss.get('native'),0,1e6) or not finite(loss.get('custom'),0,1e6) or
            not math.isclose(loss['native'],loss['custom'],rel_tol=1e-5,abs_tol=1e-4) or
            not finite(loss.get('absolute_error'),0,1e6) or
            abs(loss['absolute_error']-abs(loss['native']-loss['custom']))>1e-8 or
            len(loss.get('ids',[]))!=2 or len(set(loss['ids']))!=2 or not set(loss['ids'])<=set(train_by_id) or
            loss.get('all_labels_reviewed') is not True or loss.get('left_padding') is not True):
        raise ValueError('Native/custom CE numerical gate failed')
    steps=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    if [r.get('step') for r in steps]!=list(range(1,81)):raise ValueError('Missing optimizer steps')
    previous=0;sampler=random.Random(41)
    for step in steps:
        expected_ids=[r['id'] for _ in range(4) for r in sampler.sample(train,2)]
        if (len(step.get('sample_ids',[]))!=8 or not set(step['sample_ids'])<=set(train_by_id) or
                step['sample_ids']!=expected_ids or
                not finite(step.get('loss'),0,1e6) or not finite(step.get('gradient_norm'),1e-30,1e30) or
                not finite(step.get('elapsed_seconds'),previous,1800)):
            raise ValueError('Invalid gradient, training IDs or step timing')
        previous=step['elapsed_seconds']
    if [c['step'] for c in result.get('checkpoints',[])]!=[2,80]:raise ValueError('Unexpected checkpoint sequence')
    for checkpoint in result['checkpoints']:
        folder=ROOT/'training'/f'adapter_{checkpoint["step"]:04d}'
        if (checkpoint['path']!=str(folder) or sha(folder/'adapter_model.safetensors')!=checkpoint['weight_sha256'] or
                sha(folder/'adapter_config.json')!=checkpoint['config_sha256']):
            raise ValueError('Checkpoint binding failed')
    restore=json.loads((ROOT/'training/restore_gate.json').read_text())
    if (restore.get('passed') is not True or type(restore.get('changed_lora_tensors')) is not int or
            restore['changed_lora_tensors']<=0 or not finite(restore.get('max_logit_error'),0,1e-4)):
        raise ValueError('Missing numerical restore gate')
    if (restore['before']['id'] not in train_by_id or restore['before']['id']!=restore['restored']['id'] or
            any(restore['before'][k]!=restore['restored'][k] for k in ('prediction','text','input_ids_sha256','resized_pixels_sha256'))):
        raise ValueError('Restore did not use same input/prediction')
    for key in ('before','restored'):
        check_image_receipt(restore[key],train_by_id[restore[key]['id']],False)
    def check_decode(p,row):
        if (p.get('expected')!=row['label'] or p.get('text')!=TARGETS.get(p.get('prediction')) or
                not finite(p.get('latency_seconds'),0,1800) or type(p.get('output_tokens')) is not int or
                not 2<=p['output_tokens']<=32):
            raise ValueError('Invalid decoded response, label or latency')
    for key in ('before','restored'):check_decode(restore[key],train_by_id[restore[key]['id']])
    prefixes={}
    for name,key,step,blind in (('base_images','base',0,False),('base_gray','base_gray',0,True),
                              ('finetuned_images','finetuned',80,False),('finetuned_gray','finetuned_gray',80,True)):
        report=json.loads((ROOT/'training'/f'{name}.json').read_text());predictions=report['predictions']
        if len(predictions)!=27 or report['step']!=step or report['blind'] is not blind:
            raise ValueError('Incomplete heldout predictions')
        if len({p['id'] for p in predictions})!=27:raise ValueError('Duplicated heldout predictions')
        for p in predictions:
            row=valid_by_id[p['id']];check_image_receipt(p,row,blind)
            check_decode(p,row)
            key_prefix=(p['id'],blind)
            if step==0:prefixes[key_prefix]=p['input_ids_sha256']
            elif prefixes[key_prefix]!=p['input_ids_sha256']:raise ValueError('Train/eval input prefix changed')
        actual=metrics(valid,{p['id']:p['prediction'] for p in predictions})
        if report['metrics']!=actual or result[key]!=actual:raise ValueError('Reported metric differs from predictions')
    if any(result.get(k)!=v for k,v in train_prior_metrics(train,valid).items()):
        raise ValueError('Training-only shortcut baseline misreported')
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>=CONFIG['max_bytes']:
        raise RuntimeError('Final artifacts exceeded budget')


def supervise():
    code,data,before=preflight();launch_receipt=json.loads((ROOT/'launch.json').read_text())
    if (launch_receipt['code_commit']!=code or launch_receipt['config']!=CONFIG or
            launch_receipt['dataset']!=data or launch_receipt['command']!=command() or
            hashlib.sha256(os.environ.get('H78_LAUNCH_TOKEN','').encode()).hexdigest()!=launch_receipt['token_sha256']):
        raise ValueError('Supervisor launch binding changed')
    if not {48,49,50,51}<=os.sched_getaffinity(0):raise ValueError('Registered CPU affinity unavailable')
    os.sched_setaffinity(0,{48,49,50,51});start=time.monotonic();child=None;failure=None
    receipt={'status':'starting','code_commit':code,'config':CONFIG,'samples':[],'baseline':before,'supervisor_pid':os.getpid()}
    def interrupted(sig,_frame):raise InterruptedError('H78 signal '+str(sig))
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
        try:
            with (ROOT/'worker.log').open('x') as log:
                child=subprocess.Popen(command(),cwd=REPO,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,restore_signals=True,
                    preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_UNBLOCK,set(previous)))
            receipt.update(status='running',worker_pid=child.pid)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
        while child.poll() is None:
            if time.monotonic()-start>=1800:raise TimeoutError('Complete pilot wall budget incl load/evaluation')
            current=snapshot();check_resources(current,before,child.pid)
            if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>=CONFIG['max_bytes']:
                raise RuntimeError('Complete experiment artifact budget')
            receipt['samples'].append({'elapsed':time.monotonic()-start,'gpus':current})
            atomic_json(ROOT/'supervisor.json',receipt);time.sleep(2)
        if child.returncode!=0 or (ROOT/'training/failure.json').exists():
            raise RuntimeError('Training/evaluation failed; inspect original worker failure receipt')
        result=json.loads((ROOT/'training/result.json').read_text());validate_result(result,code)
        if time.monotonic()-start>1800:raise TimeoutError('Result completed after wall budget')
        receipt['status']='completed'
    except BaseException as error:
        failure=error;receipt.update(status='failed',error=repr(error))
    finally:
        cleanup=time.monotonic()
        try:
            if child is not None:stop_owned_group(child)
            receipt['after_exit']=snapshot();check_resources(receipt['after_exit'],before,None if child is None else child.pid,released=True)
        except BaseException as error:
            failure=failure or error;receipt.update(status='failed',cleanup_error=repr(error))
        if time.monotonic()-cleanup>30:
            failure=failure or TimeoutError('Cleanup budget');receipt.update(status='failed',cleanup_overrun=True)
        receipt.update(seconds=time.monotonic()-start,exit_code=None if child is None else child.returncode)
        atomic_json(ROOT/'supervisor.json',receipt)
        for sig,handler in previous.items():signal.signal(sig,handler)
    if failure:raise failure


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch',action='store_true');modes.add_argument('--supervise',action='store_true')
    args=parser.parse_args();(launch if args.launch else supervise)()
