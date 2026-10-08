"""Bounded 2-GPU TRAIN benchmark, followed by actual numerical batch QA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from common import ROOT, SOURCE, REPO, G05, G05_COMMIT, MODEL_PYTHON, sha256, atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);a=p.parse_args()
    if a.job.parent!=ROOT/'runs' or a.job.exists():
        raise ValueError('New registered benchmark directory required')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Benchmark source must be frozen and clean')
    if subprocess.check_output(['git','-C',str(G05),'rev-parse','HEAD'],text=True).strip()!=G05_COMMIT:
        raise ValueError('Wrong G0.5 dependency')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPUs must be free before the bounded benchmark')
    prior=json.loads((a.prior/'manifest.json').read_text())
    if not (a.prior/'STOP').exists():
        raise ValueError('Prior evaluation is not stopped')
    for side,pin in prior['checkpoints'].items():
        for path_key,hash_key in [('path','sha256'),('config','config_sha256'),('stats_path','stats_sha256')]:
            if sha256(pin[path_key])!=pin[hash_key]:
                raise ValueError('Model/config/stats changed: '+side+'/'+path_key)
    a.job.mkdir();started=time.time();deadline=started+7200
    manifest=prior|dict(kind='train_only_speed_benchmark',source_commit=commit,source_path=str(SOURCE),
        owner='Codex/EVAL-BATCH-SPEED-10383',created=started,episodes=12,
        cases=[dict(task='turning_on_radio',instance_id=i,mode='train') for i in range(1,5)],
        budget=dict(gpus=[0,1],seconds=7200,steps_per_instance=512),
        configurations=['serial2','batch2','batch4'],cross_task_check=['turning_on_radio','clean_a_keyboard'],
        automatic_public_start=False)
    atomic_json(a.job/'manifest.json',manifest)
    state=dict(status='running',pid=os.getpid(),started=started,deadline=deadline,children={})
    atomic_json(a.job/'status.json',state)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    for key in list(env):
        if key.startswith('RL_'):env.pop(key)
    env['PYTHONPATH']=str(G05/'src')+':'+str(SOURCE)
    children=[]

    def launch(name,gpu,mode,num_envs,*,cross_task=False):
        job=a.job/name;job.mkdir();atomic_json(job/'manifest.json',manifest|dict(num_envs=num_envs,inference_mode=mode))
        command=[str(MODEL_PYTHON),str(SOURCE/'worker.py'),'--job',str(job),'--gpu',str(gpu),
            '--smoke','--inference-mode',mode,'--num-envs',str(num_envs),
            '--train-steps','128' if cross_task else '512','--capture-train-audit',
            '--cross-task-check' if cross_task else '--benchmark']
        with (job/'worker.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
        children.append((name,job,child));state['children'][name]=dict(pid=child.pid,gpu=gpu,command=command)
        atomic_json(a.job/'status.json',state)
        return child

    def wait(group):
        while any(child.poll() is None for child in group):
            if time.time()>deadline or (a.job/'STOP').exists():
                raise TimeoutError('Benchmark budget or requested stop')
            if any(child.poll() not in (None,0) for child in group):
                raise RuntimeError('Benchmark child failed')
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits'],text=True)
            with (a.job/'gpu_samples.jsonl').open('a') as log:
                log.write(json.dumps(dict(time=time.time(),sample=sample))+'\n')
            time.sleep(5)
        if any(child.returncode for child in group):raise RuntimeError('Benchmark child failed')

    try:
        wait([launch('serial2',0,'serial',2),launch('batch2',1,'batch',2)])
        wait([launch('batch4',1,'batch',4)])
        wait([launch('isolation',1,'batch',2,cross_task=True)])
        inputs=sorted((a.job/'serial2/smoke/gpu_0').glob('audit_input_*.pt'))
        inputs+=sorted((a.job/'batch4/smoke/gpu_1').glob('audit_input_*.pt'))
        isolation=sorted((a.job/'isolation/smoke/gpu_1').glob('audit_input_*.pt'))
        if len(inputs)!=4 or len(isolation)!=3:raise ValueError('Missing real TRAIN audit observations')
        inputs.append(isolation[-1])  # first observation after the actual A -> B task switch
        audit_env=env|dict(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                           TOKENIZERS_PARALLELISM='false')
        command=[str(MODEL_PYTHON),str(SOURCE/'audit_batch.py'),'--inputs',*[str(p) for p in inputs],
                 '--output',str(a.job/'numerical_audit')]
        with (a.job/'audit.log').open('x') as log:
            child=subprocess.Popen(command,env=audit_env,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        children.append(('audit',a.job/'numerical_audit',child))
        state['children']['audit']=dict(pid=child.pid,gpu=0,command=command)
        atomic_json(a.job/'status.json',state);wait([child])
        qa=json.loads((a.job/'numerical_audit/audit.json').read_text())
        report={}
        for name,gpu in [('serial2',0),('batch2',1),('batch4',1)]:
            job=a.job/name;run=job/'smoke'/f'gpu_{gpu}'
            worker=json.loads((run/'status.json').read_text())
            if worker['status']!='completed' or not worker['verification']['weights_unchanged']:
                raise ValueError('Missing benchmark completion/weight verification')
            requests=[json.loads(line) for line in (run/'inference.jsonl').read_text().splitlines()]
            attempts=[json.loads(p.read_text()) for p in job.glob('smoke_tasks/*/attempts/*.json')]
            records=[json.loads(p.read_text()) for p in job.glob('smoke_tasks/*/json/*.json')]
            steps=sum(r['steps'] for r in records);seconds=sum(r['run_seconds'] for r in attempts)
            report[name]=dict(instances=len(records),control_steps=steps,run_seconds=seconds,
                control_steps_per_second=steps/seconds,policy_seconds=sum(r['seconds'] for r in requests),
                requests=len(requests),verification=worker['verification'],
                high_requests=[r['timing'] for r in requests if r['timing'].get('high_batch_size',0)])
        atomic_json(a.job/'report.json',dict(configurations=report,numerical_gate=qa['engineering_gate_passed'],
            source_commit=commit,automatic_public_start=False))
        state.update(status='completed' if qa['engineering_gate_passed'] else 'numerical_gate_failed')
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        for name,job,child in children:
            if child.poll() is None:
                if name=='audit':
                    os.killpg(child.pid,__import__('signal').SIGTERM)
                else:(job/'STOP').touch()
        for _,_,child in children:
            if child.poll() is None:
                try:child.wait(timeout=240)
                except subprocess.TimeoutExpired:state['cleanup_needs_attention']=True
        state['finished']=time.time();atomic_json(a.job/'status.json',state)


if __name__=='__main__':main()
