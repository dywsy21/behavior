"""One simulator process per ACKed short-skill episode; no failure retries.

This changes reset isolation, not skill reward, horizons, actor inputs or
on-policy accounting. Each episode remains individually identified and saved.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'scripts/eval/memlite_sft100'),str(Path(__file__).resolve().parents[1]/'code')]
from common import atomic_json
from recovery_corpus import file_sha
from skill_cold_worker import next_worker_action,require_service_hello
from skill_training_protocol import read_only_recipe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--gpu',type=int,choices=range(8),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--peer-collection',type=Path)
    p.add_argument('--peer-collection-commit')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    read_only_recipe(cfg)
    if bool(a.peer_collection)!=bool(a.peer_collection_commit):
        raise ValueError('Peer collection and frozen source commit must be bound together')
    peer_args=[]
    if a.peer_collection:
        if (not (a.peer_collection/'status.json').is_file() or len(a.peer_collection_commit)!=40
                or any(c not in '0123456789abcdef' for c in a.peer_collection_commit)):
            raise ValueError('Invalid owned collection peer')
        peer_args=['--peer-collection',str(a.peer_collection.resolve()),'--peer-collection-commit',a.peer_collection_commit]
    if (a.case not in cfg['cases'] or cfg.get('user_goal_authorized') is not True
            or cfg.get('simulator_reset_protocol') != 'fresh_process_each_episode_v1'):
        raise ValueError('Explicit case and fresh-reset recipe required')
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New worker and frozen clean source required')
    a.output.mkdir(parents=True)
    receipt=dict(status='starting',pid=os.getpid(),gpu=a.gpu,case=a.case,config_sha256=file_sha(a.config),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        completed_episodes=0,failed_attempts_retried=0,optimizer_on_this_host=False,started_unix=time.time())
    if a.peer_collection:
        receipt.update(peer_collection=str(a.peer_collection.resolve()),peer_collection_commit=a.peer_collection_commit)
    env=dict(os.environ,EVAL_GPU=str(a.gpu),EVAL_SOURCE=str(REPO/'scripts/eval/memlite_sft100'),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    try:
        # Read HELLO and close without sending any operation. A model load can
        # be slower than scene load; retry ONLY this pre-simulator transport,
        # never a job, physical restore or failed episode. Binding mismatch is
        # fatal and cannot be disguised as a temporarily unavailable service.
        from websockets.sync.client import connect
        from websockets.exceptions import ConnectionClosed,InvalidMessage
        from wire import unpackb
        receipt.update(status='waiting_service_before_simulator',transport_waits=0)
        while True:
            atomic_json(a.output/'status.json',dict(receipt,updated_unix=time.time()))
            try:
                with connect('ws://127.0.0.1:'+str(a.port),max_size=32<<20,
                             compression=None,ping_timeout=None,open_timeout=10) as socket:
                    require_service_hello(unpackb(socket.recv(timeout=30)),receipt['config_sha256'])
                receipt['service_hello_verified_before_first_simulator']=True
                break
            except (OSError,TimeoutError,ConnectionClosed,InvalidMessage):
                receipt['transport_waits']+=1
                time.sleep(2)
        while True:
            ordinal=receipt['completed_episodes']+1
            child=a.output.parent/(a.output.name+f'-episode-{ordinal:06d}')
            receipt.update(status='running_episode',child_output=str(child),updated_unix=time.time())
            atomic_json(a.output/'status.json',receipt)
            with (a.output/f'episode-{ordinal:06d}.log').open('x') as log:
                process=subprocess.Popen(['bash','scripts/eval/memlite_sft100/launch_sim.sh',
                    'scripts/rl/memlite_online/tools/collect_short_skill_rl.py',
                    '--config',str(a.config.resolve()),'--case',a.case,'--port',str(a.port),
                    '--output',str(child.resolve()),'--single-episode',*peer_args],cwd=REPO,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                receipt['child_pid']=process.pid;atomic_json(a.output/'status.json',receipt)
                code=process.wait()
            result_path=child/'result.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else None
            action=next_worker_action(code,result)
            with (a.output/'episodes.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(child=str(child),result_sha256=file_sha(result_path),
                    returncode=code,next_action=action))+'\n')
            if action=='finished':
                receipt.update(status='finished_service_window',updated_unix=time.time());break
            receipt['completed_episodes']+=1
    except BaseException as exc:
        receipt.update(status='failed_no_retry',error=repr(exc),updated_unix=time.time())
        atomic_json(a.output/'status.json',receipt);atomic_json(a.output/'result.json',receipt)
        raise
    atomic_json(a.output/'status.json',receipt);atomic_json(a.output/'result.json',receipt)


if __name__=='__main__':main()
