"""H81 one-shot supervised GPU2 teacher calibration; zero training/controls."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import shutil
import subprocess
import sys
import time

import calibrate_grounding_teacher as worker
from calibrate_grounding_teacher import (ROOT,REPO,CONFIG,GPU_UUID,model_identity,training_rows,report,
                                         clean_commit,environment_identity,input_messages)
from train_visual_presence import model_pythonpath
from prepare_visual_review import atomic_json,sha
from visual_presence import checked_image
from visual_grounding import decode_response
sys.path.insert(0,str(REPO/'scripts/semantic_robot'))
from launch_h69 import belongs_to_session,stop_owned_group
from probe_simulator_startup import snapshot,GPU_UUIDS

PYTHON=Path('/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python')
PROFILE='h81'


def configure_profile(name):
    global ROOT,CONFIG,PROFILE
    worker.configure_profile(name);ROOT=worker.ROOT;CONFIG=worker.CONFIG;PROFILE=name


def check_resources(current,baseline=None,sid=None,*,released=False):
    if set(current)!=set(GPU_UUIDS):raise ValueError('Physical GPU set changed')
    for uuid,row in current.items():
        owned=[p for p in row['processes'] if belongs_to_session(p['pid'],sid)]
        if released and owned:raise RuntimeError('Own teacher GPU resources not released')
        if uuid==GPU_UUID:
            if baseline is None and (row['processes'] or row['free_mib']<CONFIG['max_gpu_mib']+8192):
                raise RuntimeError('GPU2 unavailable for registered teacher cap')
            if baseline is not None:
                if any(not belongs_to_session(p['pid'],sid) for p in row['processes']):
                    raise RuntimeError('Unknown GPU2 process: stop only our worker')
                if (sum(p['used_mib'] for p in owned)>CONFIG['max_gpu_mib'] or
                        row['used_mib']-baseline[uuid]['used_mib']>CONFIG['max_gpu_mib']):
                    raise RuntimeError('Teacher GPU memory cap')
        elif owned:raise RuntimeError('Teacher touched an unallocated GPU')
        if row['free_mib']<8192:raise RuntimeError('Shared GPU headroom')


def command():
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(CONFIG['max_seconds'])+'s',str(PYTHON),
            str(REPO/'scripts/vlm_sft/calibrate_grounding_teacher.py'),'--profile',PROFILE,'--output',str(ROOT/'calibration')]


def load_processor(path):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path,local_files_only=True)


def validate_result(code):
    if PROFILE=='h83':
        from dual_teacher_calibration import validate_result as validate_dual
        return validate_dual(ROOT,code,CONFIG)
    folder=ROOT/'calibration';launch=json.loads((ROOT/'launch.json').read_text())
    if (folder/'failure.json').exists():raise ValueError('Teacher worker recorded failure')
    result=json.loads((folder/'result.json').read_text());identity=json.loads((folder/'identity.json').read_text())
    train,repeated,data=training_rows();rows=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
    if (result.get('status')!='complete' or result.get('code_commit')!=code or result.get('config')!=CONFIG or
            result.get('generated_examples')!=97 or result.get('primary_unique_images')!=81 or
            result.get('repeated_images')!=16 or result.get('training_label_release') is not False or
            result.get('manual_box_review')!='PENDING' or result.get('training_updates')!=0 or result.get('new_controls')!=0 or
            sha(folder/'predictions.jsonl')!=result.get('predictions_sha256')):
        raise ValueError('Incomplete bounded calibration')
    for key,limit in (('wall_seconds',1800),('cuda_peak_reserved_mib',CONFIG['max_gpu_mib'])):
        value=result.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=limit:
            raise ValueError('Teacher result budget failed')
    expected={'code_commit':code,'config':CONFIG,'environment':launch['environment'],
        'model_identity':launch['model_identity'],'dataset':data,'train_ids':[r['id'] for r in train],
        'repeat_ids':[r['id'] for r in repeated],'training_updates':0,'new_controls':0}
    if identity!=expected:raise ValueError('Teacher worker input identity changed')
    expected_keys={'id','phase','query','expected_visibility','png_sha256','protocol','size','blind','resized_pixels_sha256',
                   'input_tokens','input_ids_sha256','text','parsed','format_valid','error','has_eos','output_tokens',
                   'batch_seconds','training_eligible','manual_box_review','generated_token_ids'}
    if PROFILE=='h82':expected_keys|={'reference_spec_sha256','reference_inputs'}
    originals={r['id']:r for r in train};primary=[];repeats=[]
    expected_pairs=[('primary',r['id']) for r in train]+[('repeat',r['id']) for r in repeated]
    if [(r.get('phase'),r.get('id')) for r in rows]!=expected_pairs:raise ValueError('Duplicated or unregistered teacher calls')
    processor=load_processor(launch['model_identity']['path'])
    reference_receipts={}
    for p in rows:
        r=originals[p['id']];_,input_receipt=input_messages(r['query'],checked_image(r))
        if PROFILE=='h82':
            if r['id'] not in reference_receipts:
                _,reference_receipts[r['id']]=worker.encode(processor,r,checked_image(r))
            input_receipt=reference_receipts[r['id']]
        if (set(p)!=expected_keys or p['query']!=r['query'] or p['expected_visibility']!=r['label'] or
                p['png_sha256']!=r['png_sha256'] or p['training_eligible'] is not False or p['manual_box_review']!='PENDING' or
                any(p[k]!=v for k,v in input_receipt.items()) or type(p['input_tokens']) is not int or
                not 0<p['input_tokens']<=1800 or type(p['output_tokens']) is not int or not 0<p['output_tokens']<=192 or
                type(p['has_eos']) is not bool or type(p['format_valid']) is not bool or
                type(p['batch_seconds']) not in (float,int) or not math.isfinite(p['batch_seconds']) or not 0<p['batch_seconds']<=1800):
            raise ValueError('Teacher input or response provenance failed')
        digest=p['input_ids_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or set(digest)-set('0123456789abcdef'):
            raise ValueError('Missing exact token input receipt')
        decoded=decode_response(processor,p['generated_token_ids'])
        if any(p[k]!=v for k,v in decoded.items()):
            raise ValueError('Invalid decode falsely reported valid')
        (primary if p['phase']=='primary' else repeats).append(p)
    if result['visibility']!=report(primary,train):raise ValueError('Teacher metrics differ from raw outputs')
    by_id={r['id']:r for r in primary}
    if any(p['input_ids_sha256']!=by_id[p['id']]['input_ids_sha256'] for p in repeats):
        raise ValueError('Batch comparison changed unpadded input prefix')
    agreement=sum(p['format_valid'] and p['parsed']==by_id[p['id']]['parsed'] for p in repeats)
    if result.get('exact_repeat_agreements')!=agreement:raise ValueError('Batch agreement misreported')
    expected_batches=[]
    for phase,group,size in (('primary',train,4),('repeat',repeated,8)):
        for i in range(0,len(group),size):expected_batches.append((phase,[r['id'] for r in group[i:i+size]]))
    if [(b.get('phase'),b.get('ids')) for b in result.get('batches',[])]!=expected_batches:
        raise ValueError('Teacher batch schedule changed')
    indexed={(p['phase'],p['id']):p for p in rows}
    for batch in result['batches']:
        if (set(batch)!={'phase','ids','seconds','batch_size','padded_input_tokens'} or
                type(batch['batch_size']) is not int or batch['batch_size']!=len(batch['ids']) or
                type(batch['seconds']) not in (int,float) or not math.isfinite(batch['seconds']) or
                not 0<batch['seconds']<=1800 or type(batch['padded_input_tokens']) is not int or
                not 0<batch['padded_input_tokens']<=1800):
            raise ValueError('Invalid teacher batch throughput receipt')
        members=[indexed[(batch['phase'],idx)] for idx in batch['ids']]
        if (any(p['batch_seconds']!=batch['seconds'] for p in members) or
                max(p['input_tokens'] for p in members)!=batch['padded_input_tokens']):
            raise ValueError('Batch timing/padded tokens differ from row evidence')
    if sum(b['seconds'] for b in result['batches'])>result['wall_seconds']:
        raise ValueError('Generation exceeds total calibration time')
    return result


def preflight():
    if Path(sys.executable)!=PYTHON:raise ValueError('Registered VLM interpreter only')
    code=clean_commit();environment=environment_identity();_train,_repeat,data=training_rows()
    if not set(CONFIG['cpu_ids'])<=os.sched_getaffinity(0):raise ValueError('CPU allocation unavailable')
    if shutil.disk_usage('/mnt/nvme_tmp').free<80*1024**3:raise RuntimeError('Disk reserve')
    before=snapshot();check_resources(before)
    return code,environment,data,before


def launch():
    code,environment,data,before=preflight();identity=model_identity()
    if ROOT.exists():raise FileExistsError(PROFILE+' already reserved; no automatic retry')
    ROOT.mkdir(parents=True);token=secrets.token_hex(24)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONPATH=model_pythonpath(),PYTHONDONTWRITEBYTECODE='1',
             PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',H81_LAUNCH_TOKEN=token)
    for key,name in (('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor')):
        path=ROOT/'cache'/name;path.mkdir(parents=True);env[key]=str(path)
    receipt={'code_commit':code,'config':CONFIG,'environment':environment,'dataset':data,'baseline':before,
             'model_identity':identity,'command':command(),'utc':datetime.now(timezone.utc).isoformat(),
             'token_sha256':hashlib.sha256(token.encode()).hexdigest(),'status':'reserved'}
    atomic_json(ROOT/'launch.json',receipt)
    with (ROOT/'supervisor.log').open('x') as log:
        child=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--profile',PROFILE,'--supervise'],cwd=REPO,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(status='supervisor_started',supervisor_pid=child.pid)
    try:atomic_json(ROOT/'launch.json',receipt)
    except Exception as error:receipt['receipt_update_error']=repr(error)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('model_identity','dataset')}),flush=True)


def supervise():
    code,environment,data,before=preflight();launch_receipt=json.loads((ROOT/'launch.json').read_text())
    if (launch_receipt['code_commit']!=code or launch_receipt['config']!=CONFIG or launch_receipt['environment']!=environment or
            launch_receipt['dataset']!=data or launch_receipt['command']!=command() or
            hashlib.sha256(os.environ.get('H81_LAUNCH_TOKEN','').encode()).hexdigest()!=launch_receipt['token_sha256']):
        raise ValueError('Teacher supervisor identity mismatch')
    os.sched_setaffinity(0,set(CONFIG['cpu_ids']));start=time.monotonic();child=None;error=None
    receipt={'status':'starting','code_commit':code,'config':CONFIG,'baseline':before,'samples':[],
             'supervisor_pid':os.getpid(),'training_label_release':False}
    def interrupted(sig,_frame):raise InterruptedError('Teacher supervisor signal '+str(sig))
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
        try:
            with (ROOT/'worker.log').open('x') as log:
                child=subprocess.Popen(command(),cwd=REPO,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_UNBLOCK,set(previous)))
            receipt.update(status='running',worker_pid=child.pid)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
        while child.poll() is None:
            if time.monotonic()-start>=CONFIG['max_seconds']:raise TimeoutError('Teacher total wall budget')
            current=snapshot();check_resources(current,before,child.pid)
            if (sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>=CONFIG['max_bytes'] or
                    shutil.disk_usage('/mnt/nvme_tmp').free<80*1024**3):raise RuntimeError('Teacher artifact/disk budget')
            receipt['samples'].append({'seconds':time.monotonic()-start,'gpus':current})
            atomic_json(ROOT/'supervisor.json',receipt);time.sleep(2)
        if child.returncode!=0:raise RuntimeError('Teacher worker failed; inspect original failure receipt')
        validate_result(code)
        if time.monotonic()-start>CONFIG['max_seconds']:raise TimeoutError('Teacher acceptance wall budget')
        receipt['status']='completed'
    except BaseException as caught:error=caught;receipt.update(status='failed',error=repr(caught))
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        cleanup=time.monotonic()
        try:
            if child is not None:stop_owned_group(child)
            receipt['after_exit']=snapshot();check_resources(receipt['after_exit'],before,None if child is None else child.pid,released=True)
        except BaseException as caught:error=error or caught;receipt.update(status='failed',cleanup_error=repr(caught))
        receipt.update(seconds=time.monotonic()-start,cleanup_seconds=time.monotonic()-cleanup,
                       exit_code=None if child is None else child.returncode)
        if receipt['seconds']>CONFIG['outer_seconds'] or receipt['cleanup_seconds']>CONFIG['cleanup_seconds']:
            error=error or TimeoutError('Teacher cleanup budget');receipt.update(status='failed',cleanup_overrun=True)
        try:atomic_json(ROOT/'supervisor.json',receipt)
        except BaseException as caught:
            print(json.dumps({'primary_error':repr(error),'receipt_error':repr(caught)}),file=sys.stderr,flush=True);error=error or caught
        for sig,handler in previous.items():signal.signal(sig,handler)
    if error:raise error


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--supervise',action='store_true')
    parser.add_argument('--profile',choices=('h81','h82','h83'),default='h81')
    args=parser.parse_args();configure_profile(args.profile);supervise() if args.supervise else launch()
