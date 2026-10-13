"""Owned-GPU official evaluation of the registered terminal champion EMA."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE/'memlite_sft100'),str(HERE/'memlite_final')]
from common import ROOT,SIM_ROOT,atomic_json,sha256,aggregate
from completed_metrics import completed_metrics
from memlite_final.lifecycle import episode_portions
from memlite_final.worker import loader_lock,stop_child


def registration(job):
    job=Path(job).resolve();m=json.loads((job/'manifest.json').read_text())
    if (job.parent!=ROOT/'runs' or m['schema']!='champion_final_public100_run_v1'
            or m['source_commit']!=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
            or len(m['configs'])!=8):
        raise ValueError('Exact clean registered terminal-model source required')
    for i,cfg in enumerate(m['configs']):
        if cfg['gpu']!=i or sha256(cfg['path'])!=cfg['sha256']:raise ValueError('Changed model config')
    if sha256(m['terminal_path'])!=m['terminal_sha256']:raise ValueError('Changed terminal training receipt')
    return job,m


def select_preflight(job,m):
    candidates={}
    for size in (2,4):
        path=job/'preflight'/f'batch{size}'/'status.json'
        if not path.is_file():continue
        r=json.loads(path.read_text())
        if r['status']!='completed':continue
        expected={f'{task}-n{n}-i{ids[0]}' for task in ('turning_on_radio','clean_a_keyboard')
            for n,ids in episode_portions(size,preflight=True)}
        verification=r.get('verification',{})
        if (r['source_commit']!=m['source_commit'] or r['num_envs']!=size or r['preflight'] is not True
                or r['export_sha256']!=m['export_sha256'] or r['config_sha256']!=m['configs'][0]['sha256']
                or verification.get('weights_unchanged') is not True or verification.get('optimizer_updates')!=0
                or verification.get('source_commit')!=m['source_commit']
                or verification.get('config_sha256')!=m['configs'][0]['sha256']
                or len(r['completed_parts'])!=len(expected)
                or {Path(p).name for p in r['completed_parts']}!=expected
                or not r['control_steps_per_second']>0):
            raise ValueError('Incomplete/foreign actual terminal-EMA engineering receipt')
        candidates[size]=(r['control_steps_per_second'],dict(path=str(path),sha256=sha256(path)))
    if 2 not in candidates:raise ValueError('Actual terminal model batch2 preflight required')
    size=4 if 4 in candidates and candidates[4][0]>candidates[2][0] else 2
    return dict(source_commit=m['source_commit'],export_sha256=m['export_sha256'],num_envs=size,
        evidence={str(k):v[1] for k,v in candidates.items()},
        selection='Actual applied-controls/s over identical eight TRAIN cases, never public scores')


def worker(job,gpu,size,preflight):
    job,m=registration(job)
    if not 0<=gpu<8 or size not in (2,4) or (preflight and gpu!=0):raise ValueError('Registered GPU/slot scope required')
    cfg=m['configs'][gpu];policy=json.loads(Path(cfg['path']).read_text())
    if preflight:
        path=job/'preflight'/f'batch{size}'
        names={r['task']:r['task_id'] for r in m['cases']}
        tasks=[dict(task=t,task_id=names[t]) for t in ('turning_on_radio','clean_a_keyboard')]
    else:
        accepted=json.loads((job/'accepted-preflight.json').read_text())
        if (accepted['source_commit']!=m['source_commit'] or accepted['export_sha256']!=m['export_sha256']
                or accepted['num_envs']!=size):raise ValueError('Missing accepted frozen model engineering gate')
        path=job/'workers'/f'gpu_{gpu}';tasks=m['groups'][gpu]['tasks']
    path.mkdir(parents=True,exist_ok=False)
    lock=(SIM_ROOT/f'gpu_{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='loading',pid=os.getpid(),gpu=gpu,num_envs=size,preflight=preflight,
        source_commit=m['source_commit'],config_sha256=cfg['sha256'],export_sha256=m['export_sha256'],
        started=time.time(),completed_parts=[],control_steps=0,run_seconds=0.)
    def publish():state['updated']=time.time();atomic_json(path/'status.json',state)
    env={k:v for k,v in os.environ.items() if not k.startswith('RL_') and k!='JAX_PLATFORMS'}
    model_path=[str(REPO/'src'),str(ROOT/policy['reference']/'src'),str(ROOT/policy['openpi_reference']/'src'),
        str(ROOT/policy['openpi_reference']/'packages/openpi-client/src'),m['model_overlay']]
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=os.pathsep.join(model_path),PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    model=sim=loader=None
    try:
        loader=loader_lock(job)
        with (path/'model.log').open('x') as log:
            model=subprocess.Popen([m['model_python'],str(HERE/'serve_champion_official.py'),
                '--config',cfg['path'],'--output',str(path/'model'),'--profile-memory'],
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['model_pid']=model.pid;publish();deadline=time.monotonic()+1800
        while True:
            status=path/'model/status.json'
            if status.exists() and json.loads(status.read_text())['status']=='ready':break
            if model.poll() is not None:raise RuntimeError('Exact terminal EMA model loading failed')
            if time.monotonic()>deadline or (job/'STOP').exists():raise TimeoutError('Model loading timeout/stop')
            time.sleep(3)
        loader.close();loader=None
        for task in tasks:
            for count,indices in episode_portions(size,preflight=preflight):
                if (job/'STOP').exists():raise InterruptedError('Administrative stop')
                if shutil.disk_usage(job).free<150<<30:raise RuntimeError('Disk safety reserve')
                part=f"{task['task']}-n{count}-i{indices[0]}"
                output=path/'parts'/part if preflight else job/'parts'/part
                output.parent.mkdir(parents=True,exist_ok=True);loader=loader_lock(job)
                command=['bash',str(HERE/'memlite_sft100/launch_sim.sh'),str(HERE/'run_champion_task.py'),
                    '--task',task['task'],'--task-id',str(task['task_id']),'--policy-config',cfg['path'],
                    '--output',str(output),'--num-envs',str(count),'--indices',*map(str,indices)]
                if preflight:command.append('--train-smoke')
                # Model dependency overlay MUST NOT contaminate the simulator.
                sim_env=env|dict(EVAL_GPU=str(gpu),EVAL_SOURCE=str(HERE),PYTHONPATH=str(REPO/'src'))
                with (path/(part+'.log')).open('x') as log:
                    sim=subprocess.Popen(command,env=sim_env,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(path/(part+'-command.json'),dict(argv=command,official_commit=m['official_commit']))
                state.update(status='evaluating',sim_pid=sim.pid,part=part);publish()
                previous=None;last=time.monotonic()
                while sim.poll() is None:
                    if loader is not None and (output/'scene.ready').exists():loader.close();loader=None
                    if (job/'STOP').exists():raise InterruptedError('Administrative stop')
                    if model.poll() is not None:raise RuntimeError('Frozen model service exited')
                    service=json.loads((path/'model/status.json').read_text())
                    if service['status']=='failed_no_retry':raise RuntimeError('Frozen model failed')
                    events=path/'model/events.jsonl';progress=events.stat().st_size if events.exists() else 0
                    if progress!=previous:previous=progress;last=time.monotonic()
                    if time.monotonic()-last>1800:raise TimeoutError('No model progress for 30 minutes')
                    if shutil.disk_usage(job).free<150<<30:raise RuntimeError('Disk safety stop')
                    time.sleep(5)
                if loader is not None:loader.close();loader=None
                receipt=json.loads((output/'status.json').read_text())
                if sim.returncode or receipt['status']!='completed':raise RuntimeError('Official task batch incomplete')
                records=completed_metrics(output,final=True)
                videos=list((output/'videos').glob('*.mp4'))
                if len(records)!=len(indices) or len(videos)!=len(indices) or any(v.stat().st_size<1000 for v in videos):
                    raise ValueError('Incomplete original metrics/video outputs')
                for attempt in (output/'attempts').glob('*.json'):
                    r=json.loads(attempt.read_text());state['run_seconds']+=r['seconds']
                    state['control_steps']+=sum(r['actual_applied_controls'])
                state['completed_parts'].append(str(output));publish()
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed_no_retry',error=repr(error));raise
    finally:
        if loader is not None:loader.close()
        stop_child(sim)
        if model is not None and model.poll() is None:
            os.killpg(model.pid,signal.SIGTERM)
            try:model.wait(timeout=240)
            except subprocess.TimeoutExpired:stop_child(model)
        receipt=path/'model/result.json'
        if receipt.exists():state['verification']=json.loads(receipt.read_text())
        if state['status']=='completed' and not state.get('verification',{}).get('weights_unchanged'):
            state.update(status='failed_no_retry',error='Missing exact frozen model final digest')
        state['control_steps_per_second']=state['control_steps']/state['run_seconds'] if state['run_seconds'] else None
        state['finished']=time.time();publish();lock.close()
    if state['status']!='completed':raise RuntimeError('Worker final verification failed')


def report(job,*,final=False):
    job=Path(job);m=json.loads((job/'manifest.json').read_text());records=[];inventory=[]
    for part in sorted((job/'parts').glob('*')):
        for metric,row in completed_metrics(part,final=final):
            records.append(row);video=part/'videos'/(metric.stem+'.mp4')
            item=dict(task=row['task'],instance_id=row['instance_id'],rollout_id=row['rollout_id'],
                metrics=str(metric),video=str(video),video_exists=video.is_file())
            if final:
                if not video.is_file() or video.stat().st_size<1000:raise ValueError('Missing original video')
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                    '-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(video)],text=True))
                stream=probe['streams'][0]
                if int(stream['nb_frames'])!=row['steps'] or stream['r_frame_rate']!='30/1':
                    raise ValueError('Original video frames/control count differ')
                item.update(metrics_sha256=sha256(metric),video_sha256=sha256(video),video_bytes=video.stat().st_size,video_probe=stream)
            inventory.append(item)
    summary=aggregate(m['tasks'],records)
    summary.update(source_commit=m['source_commit'],export_sha256=m['export_sha256'],
        original_outputs_unmodified=True,optimizer_updates=0,final_model_submission_ready=False)
    atomic_json(job/'summary.json',summary)
    if final:
        if summary['status']!='complete':raise ValueError('Never seal incomplete1000 as complete')
        atomic_json(job/'inventory.json',inventory)
        text=f'''# Terminal champion EMA official evaluation

Frozen project commit: {m['source_commit']}; official: {m['official_commit']}.
EMA: {m['export_sha256']}, terminal completed updates: {m['completed_training_updates']}.
100 tasks x public instances301–310 x rollout0; unchanged official timeout,
physics, metrics, and 30fps videos. Policy seed17; simulator seed0. Instance301
was previously development, not blind. No public checkpoint selection or retries.
Each slot has a separate original-protocol connection, stage/RNG/action cache.
Server counters count delivered actions; separate post-step receipts count actual
controls. Onboard RGB/proprio/task inputs only; FM20, chunk20, native23 controls.

Reproduce with this frozen commit and verified export, register a NEW directory
using prepare_champion_full_eval.py with the exact terminal/export SHA, then run:
python scripts/eval/champion_full_eval.py worker --job NEW_RUN --gpu 0 --num-envs 2 --preflight
python scripts/eval/champion_full_eval.py worker --job NEW_RUN --gpu 0 --num-envs 4 --preflight
python scripts/eval/champion_full_eval.py supervise --job NEW_RUN

Original metrics are archived; original MP4s are separate and SHA-bound by
inventory.json. This is evaluation output collection, NOT a completed model
delivery or external submission. Single24GB Docker OR50 hosted policy ports,
external video upload and team declarations still require separate acceptance.
'''
        readme=job/'submission/README.md'
        with readme.open('x') as stream:stream.write(text)
        target=job/'submission/evaluation_outputs.zip'
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for item in inventory:archive.write(item['metrics'],'json/'+Path(item['metrics']).name)
            for name in ('rgb_wrapper.py','r1pro.yaml','README.md'):archive.write(job/'submission'/name,name)
            for name in ('champion_batched_client.py','run_champion_task.py'):archive.write(HERE/name,'transport/'+name)
            archive.write(HERE/'memlite_sft100/wire.py','transport/wire.py')
            for name in ('manifest.json','accepted-preflight.json','inventory.json','summary.json'):archive.write(job/name,name)
            archive.write(m['terminal_path'],'terminal-training.json')
            for cfg in m['configs']:archive.write(cfg['path'],f"configs/gpu_{cfg['gpu']}.json")
        atomic_json(job/'submission/package-status.json',dict(status='evaluation_outputs_only_not_final_model_delivery',
            metrics=1000,videos=1000,zip_path=str(target),zip_sha256=sha256(target),video_inventory=str(job/'inventory.json'),
            videos_uploaded=False,docker_24gb_validated=False,external_endpoint_50_ports_validated=False,
            team_declarations_confirmed=False,officially_submitted=False))
    return summary


def supervise(job):
    job,m=registration(job)
    if (job/'status.json').exists() or (job/'STOP').exists():raise ValueError('No implicit restart/reroll')
    accepted=select_preflight(job,m)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Do not displace active GPU work')
    lock=(job/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(job/'accepted-preflight.json',accepted)
    state=dict(status='running',pid=os.getpid(),source_commit=m['source_commit'],started=time.time(),
        num_envs=accepted['num_envs'],completed_gpus=[],failed_gpus={});children={}
    try:
        for gpu in range(8):
            with (job/f'gpu_{gpu}.log').open('x') as log:
                children[gpu]=subprocess.Popen([sys.executable,__file__,'worker','--job',str(job),
                    '--gpu',str(gpu),'--num-envs',str(accepted['num_envs'])],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['worker_pids']={str(k):c.pid for k,c in children.items()};atomic_json(job/'status.json',state)
        while children:
            if (job/'STOP').exists():raise InterruptedError('Administrative/safety stop')
            for gpu,child in list(children.items()):
                if child.poll() is None:continue
                p=job/'workers'/f'gpu_{gpu}'/'status.json'
                row=json.loads(p.read_text()) if p.exists() else dict(status='missing_status')
                if child.returncode or row['status']!='completed':
                    state['failed_gpus'][str(gpu)]=row;raise RuntimeError('Worker failed; no automatic reroll')
                state['completed_gpus'].append(gpu);del children[gpu]
            r=report(job);state.update(completed_episodes=r['completed'],updated=time.time());atomic_json(job/'status.json',state)
            if children:time.sleep(30)
        r=report(job,final=True);state.update(status='completed',q_score=r['official_q_score'],success_rate=r['official_sr'])
    except BaseException as error:
        state.update(status='needs_diagnosis',error=repr(error));(job/'STOP').touch();raise
    finally:
        deadline=time.monotonic()+300
        while any(c.poll() is None for c in children.values()) and time.monotonic()<deadline:time.sleep(3)
        state['finished']=time.time();atomic_json(job/'status.json',state);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('worker','supervise','report'))
    p.add_argument('--job',type=Path,required=True);p.add_argument('--gpu',type=int)
    p.add_argument('--num-envs',type=int,choices=(2,4));p.add_argument('--preflight',action='store_true')
    p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.command=='worker':worker(a.job,a.gpu,a.num_envs,a.preflight)
    elif a.command=='supervise':supervise(a.job)
    else:
        r=report(a.job,final=a.final);print(json.dumps({k:r[k] for k in ('status','completed','missing','official_q_score','official_sr')}))
