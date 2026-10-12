"""Persistent per-GPU batch server for one frozen MEM-Lite final-evaluation run."""
import argparse
import asyncio
import http
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/rl/memlite_online/code'),
    str(REPO/'scripts/eval'),str(REPO/'scripts/eval/memlite_sft100'),str(HERE)]
from common import atomic_json
from recovery_corpus import file_sha
from recovery_process_cache import isolated_cache_env
from wire import packb,unpackb


async def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('config','pack','run'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args()
    if a.run.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New output and immutable clean source required')
    a.run.mkdir(parents=True);os.environ.update(isolated_cache_env(a.run))
    state=dict(status='loading',pid=os.getpid(),config_sha256=file_sha(a.config),manifest_sha256=a.manifest_sha256,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),started=time.time(),optimizer_updates=0)
    atomic_json(a.run/'status.json',state)
    import torch
    import websockets
    from recovery_cuda_startup import initialize_cuda_backend
    initialize_cuda_backend(0)
    from models import LoadedFinalModels
    from batch_runtime import CausalBatchRuntime
    loaded=LoadedFinalModels(a.pack,a.manifest_sha256,a.config)
    def emit(kind,record):
        with (a.run/'events.jsonl').open('a') as f:f.write(json.dumps(dict(record,kind=kind),allow_nan=False)+'\n')
    runtime=CausalBatchRuntime(models=loaded.identity,calibration=loaded.calibration,planner=loaded.planner,
        adapter=loaded.adapter,observer_factory=loaded.observer.bind,sample=loaded.sample,emit=emit,run_id=a.run.name)
    gate=asyncio.Lock();stopped=asyncio.Event();owner=None
    def ack(rows):
        slots=[r['slot'] for r in rows]
        if len(set(slots))!=len(slots) or any(set(r)!={'slot','token','controls'} for r in rows):
            raise ValueError('Duplicate or malformed applied-action receipt')
        runtime.acknowledge({r['slot']:dict(token=r['token'],controls=r['controls']) for r in rows})
    def dispatch(request):
        if request.pop('config_sha256')!=state['config_sha256']:raise ValueError('Foreign policy revision')
        op=request.pop('op')
        if op=='begin' and set(request)=={'metadata','mode'}:
            runtime.begin(request['metadata'],request['mode']);return dict(begun=True)
        if op in ('infer','finish') and set(request)==({'observations','acks','active'} if op=='infer' else {'observations','acks'}):
            obs=request['observations']
            if len(obs)!=len(runtime.sessions):raise ValueError('Observation identity batch differs')
            ack(request['acks'])
            active=request['active'] if op=='infer' else []
            close=[i for i,w in runtime.sessions.items() if i not in active and not w.session.closed]
            runtime.close(close,obs)
            if op=='finish':return dict(steps=[w.session.feedback.control_step for w in runtime.sessions.values()])
            rows,timing=runtime.infer(obs,active)
            return dict(rows=[dict(slot=i,**row) for i,row in rows.items()],timing=timing)
        raise ValueError('Unknown or privileged request fields')
    async def handler(socket):
        nonlocal owner
        if owner is not None:
            await socket.close(code=1013,reason='One official batched evaluator per port');return
        owner=socket
        try:
            await socket.send(packb(dict(kind='memlite_final_ack_batch_v1',config_sha256=state['config_sha256'],
                action_dim=23,chunk_size=16,optimizer_updates=0)))
            async for payload in socket:
                async with gate:
                    result=await asyncio.to_thread(dispatch,unpackb(payload))
                    state.update(status='ready',updated=time.time(),requests=runtime.requests,
                        max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        max_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    atomic_json(a.run/'status.json',state)
                    await socket.send(packb(dict(ok=True,result=result)))
        except websockets.ConnectionClosed:
            if any(not w.session.closed for w in runtime.sessions.values()):
                state.update(status='failed',error='Connection lost with unclosed episodes')
                atomic_json(a.run/'status.json',state);stopped.set()
        except Exception as e:
            state.update(status='failed',error=repr(e));atomic_json(a.run/'status.json',state);traceback.print_exc()
            try:await socket.send(packb(dict(ok=False,error=repr(e))))
            except websockets.ConnectionClosed:pass
            stopped.set()
        finally:
            # A normal websocket close also ends async iteration without a
            # ConnectionClosed exception. It must not abandon live memory and
            # silently admit a replacement evaluator or reroll.
            if state.get('status')!='failed' and any(not w.session.closed for w in runtime.sessions.values()):
                state.update(status='failed',error='Client closed with unacknowledged or unfinished episodes')
                atomic_json(a.run/'status.json',state);stopped.set()
            owner=None
    async def watch():
        while not stopped.is_set():
            if (a.run/'STOP').exists():stopped.set();return
            await asyncio.sleep(2)
    def health(conn,request):
        if request.path=='/healthz':return conn.respond(http.HTTPStatus.OK,'OK\n')
    try:
        async with websockets.serve(handler,'127.0.0.1',a.port,max_size=256<<20,compression=None,
                ping_interval=None,ping_timeout=None,process_request=health):
            state.update(status='ready',loaded=time.time());atomic_json(a.run/'status.json',state)
            (a.run/'READY').touch();watcher=asyncio.create_task(watch())
            await stopped.wait();watcher.cancel()
    finally:
        receipt=await asyncio.to_thread(loaded.finish)
        atomic_json(a.run/'weights-final.json',receipt)
        if state['status']!='failed':state['status']='finished'
        state.update(finished=time.time(),verification=receipt);atomic_json(a.run/'status.json',state)


if __name__=='__main__':asyncio.run(main())
