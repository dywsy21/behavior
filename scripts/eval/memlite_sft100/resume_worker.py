"""Dynamic remaining-case work queue; one isolated model session per GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from common import ROOT,SOURCE,SIM_ROOT,MODEL_PYTHON,G05,atomic_json,sha256
from worker import acquire_loader,stop_child,anonymous_gib


def claim(job,parts,gpu):
    with (job/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (job/'STOP').exists():raise InterruptedError('Requested evaluation stop')
        for part in parts:
            path=job/'claims'/(part['id']+'.json')
            if not path.exists():
                atomic_json(path,dict(part=part,gpu=gpu,pid=os.getpid(),claimed=time.time(),
                                      automatic_retry=False))
                return part
    return None


def publish_part(job,part,output):
    """Link completed originals to the canonical results tree, never overwrite."""
    expected={(part['task'],i+301,0) for i in part['indices']}
    records=[]
    for path in sorted((output/'json').glob('*.json')):
        record=json.loads(path.read_text());key=(record['task'],record['instance_id'],record['rollout_id'])
        if key not in expected:raise ValueError('Wrong official part identity')
        if path.stem!=f'{key[0]}_{key[1]}_{key[2]}':raise ValueError('Official result filename mismatch')
        expected.remove(key)
        video=output/'videos'/(path.stem+'.mp4')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,width,height,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
        if int(probe['nb_frames'])!=record['steps'] or probe['r_frame_rate']!='30/1' or (
                probe['width'],probe['height'])!=(672,448):
            raise ValueError('Original media validation failed')
        records.append((path,video))
    if expected:raise ValueError('Missing part results')
    task=job/'tasks'/part['task']
    for kind in ('json','videos'):(task/kind).mkdir(parents=True,exist_ok=True)
    # Separate tasks may share a queue, never a memory/intent session. Part
    # instances are disjoint, so os.link's exclusive destination is the guard.
    for metrics,video in records:
        for original,kind in [(metrics,'json'),(video,'videos')]:
            target=task/kind/original.name
            os.link(original,target)
    atomic_json(output/'publication.json',dict(status='completed',published=time.time(),
        part=part,original_outputs_unmodified=True,
        metrics=[dict(path=str(m),sha256=sha256(m)) for m,_ in records]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args();job=a.job;gpu=a.gpu
    manifest=json.loads((job/'manifest.json').read_text())
    if manifest['kind']!='native_sft100_admin_resume' or manifest['inference_mode']!='batch':
        raise ValueError('Wrong resume plan')
    run=job/'workers'/f'gpu_{gpu}';run.mkdir(parents=True,exist_ok=False)
    lock=open(SIM_ROOT/f'gpu_{gpu}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='starting',pid=os.getpid(),gpu=gpu,source=manifest['source_commit'],completed_parts=[])
    def publish():state['updated']=time.time();atomic_json(run/'status.json',state)
    policy=sim=loader=None;port=19600+gpu
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('RL_'):env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        PYTHONPATH=str(G05/'src')+':'+str(SOURCE))
    try:
        loader=acquire_loader(job)
        with (run/'policy.log').open('x') as stream:
            policy=subprocess.Popen([str(MODEL_PYTHON),str(SOURCE/'serve.py'),'--run',str(run),
                '--port',str(port),'--inference-mode','batch'],env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state['policy_pid']=policy.pid;publish();deadline=time.time()+1200
        while not (run/'policy.ready').exists():
            if policy.poll() is not None:raise RuntimeError('Policy failed to load')
            if time.time()>deadline or (job/'STOP').exists():raise TimeoutError('Load timeout/stop')
            time.sleep(3)
        loader.close();loader=None
        while True:
            part=claim(job,manifest['parts'],gpu)
            if part is None:break
            output=job/'parts'/part['id'];output.mkdir(exist_ok=False)
            if shutil.disk_usage(job).free<150*2**30:raise RuntimeError('Disk reserve reached')
            loader=acquire_loader(job)
            command=['bash',str(SOURCE/'launch_sim.sh'),str(SOURCE/'run_task.py'),
                '--task',part['task'],'--output',str(output),'--policy-run',str(run),'--port',str(port),
                '--num-envs',str(part['num_envs']),'--indices',*[str(i) for i in part['indices']]]
            sim_env=env|dict(EVAL_GPU=str(gpu),EVAL_SOURCE=str(SOURCE))
            atomic_json(output/'command.json',dict(argv=command,part=part,source_commit=manifest['source_commit'],
                official_commit=manifest['official_commit'],environment_overrides={
                    key:sim_env[key] for key in ('EVAL_GPU','EVAL_SOURCE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
            with (output/'sim.log').open('x') as stream:
                sim=subprocess.Popen(command,env=sim_env,stdin=subprocess.DEVNULL,stdout=stream,
                                      stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='evaluating',part=part,sim_pid=sim.pid,started_part=time.time());publish()
            last_progress=time.time();last_request=None
            while sim.poll() is None:
                if loader is not None and (output/'scene.ready').exists():loader.close();loader=None
                if (job/'STOP').exists():raise InterruptedError('Requested evaluation stop')
                if policy.poll() is not None:raise RuntimeError('Policy exited')
                ps=json.loads((run/'policy_status.json').read_text())
                if ps['status']=='failed':raise RuntimeError('Policy failed: '+str(ps.get('error')))
                if ps.get('requests')!=last_request:last_request=ps.get('requests');last_progress=time.time()
                if time.time()-last_progress>1800:raise TimeoutError('No progress for 30 minutes')
                if anonymous_gib()>210:raise RuntimeError('Host RAM safety limit')
                if shutil.disk_usage(job).free<150*2**30:raise RuntimeError('Disk safety limit')
                time.sleep(5)
            if loader is not None:loader.close();loader=None
            if sim.returncode or json.loads((output/'status.json').read_text())['status']!='completed':
                raise RuntimeError('Official part did not complete: '+part['id'])
            publish_part(job,part,output)
            state['completed_parts'].append(part['id']);publish()
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        if loader is not None:loader.close()
        stop_child(sim)
        if policy is not None and policy.poll() is None:
            (run/'STOP').touch()
            try:policy.wait(timeout=180)
            except subprocess.TimeoutExpired:stop_child(policy)
        if (run/'evaluation_ack.json').exists():
            state['verification']=json.loads((run/'evaluation_ack.json').read_text())
        if state.get('status')=='completed' and not state.get('verification',{}).get('weights_unchanged'):
            state.update(status='failed',error='Missing exact post-run model fingerprints')
        state['finished']=time.time();publish();lock.close()
    if state['status']!='completed':raise RuntimeError(state.get('error','Failed verification'))


if __name__=='__main__':main()
