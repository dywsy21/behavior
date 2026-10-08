"""Official reset/action/action_chunk websocket protocol, bound to localhost."""
import argparse
import asyncio
import http
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import websockets
from common import bootstrap, atomic_json, ROOT, DATA, load_official_task_names
bootstrap()
from wire import packb, unpackb
from a4_observation import behavior_obs_to_native_low
from batched_engine import BatchedSFT


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--inference-mode', choices=['serial','batch'], default='serial')
    p.add_argument('--capture-train-audit', action='store_true')
    args = p.parse_args()
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    state = dict(status='loading', pid=os.getpid(), optimizer_steps=0, started=time.time())
    atomic_json(run/'policy_status.json', state)
    engine = await asyncio.to_thread(BatchedSFT, run/'no_checkpoints',
                                   mode=args.inference_mode, capture=args.capture_train_audit)
    tasks = load_official_task_names(DATA/'2026-challenge-task-instances/metadata/B100_task_misc.csv')
    gate = asyncio.Lock()
    stopped = asyncio.Event()
    owner = None

    async def handler(socket):
        nonlocal owner
        if owner is not None:
            await socket.close(code=1013, reason='One batched evaluator per port'); return
        owner = socket
        initialized = False
        active_meta = None
        await socket.send(packb(dict(model='MEM-Lite stage1 SFT', native_fm=True, action_dim=23,
                                     action_chunk_size=16, optimizer_steps=0)))
        try:
            async for payload in socket:
                request = unpackb(payload)
                async with gate:
                    if request.get('reset') is True:
                        initialized = False
                        active_meta = json.loads((run/'current_batch.json').read_text())
                        if active_meta['mode'] not in ('train_smoke','public_test'):
                            raise ValueError('Unregistered evaluation split')
                        if args.capture_train_audit and active_meta['mode'] != 'train_smoke':
                            raise ValueError('No public-test audit captures or tuning')
                        continue  # The official reset protocol has NO reply.
                    if active_meta is None:
                        raise ValueError('Official reset must precede inference')
                    if request.get('__action_chunk_size__') != 16:
                        raise ValueError('Use --replay-action-chunk-size 16')
                    task_ids = np.asarray(request['task_id']).reshape(-1)
                    n = len(task_ids)
                    if n != len(active_meta['instance_ids']) or len(set(task_ids.tolist())) != 1:
                        raise ValueError('Batch task/instance cardinality mismatch')
                    task = tasks[int(task_ids[0])]
                    if task != active_meta['task']:
                        raise ValueError('Simulator task differs from frozen case registration')
                    # This explicit projection is the only input to the actor.
                    # cam_rel_poses, object state, rewards, success, poses etc.
                    # are never forwarded, even if supplied by a client.
                    native = [behavior_obs_to_native_low(
                        {k:v[i] for k,v in request.items() if k != '__action_chunk_size__'}, tasks)
                        for i in range(n)]
                    if not initialized:
                        await asyncio.to_thread(engine.begin, task, n, 17,
                            [dict(task=task, split=active_meta['mode'], instance_id=i)
                             for i in active_meta['instance_ids']])
                        initialized = True
                    for i, obs in enumerate(native):
                        obs['control_step'] = engine.slots[i]['chunks'] * 16
                    started = time.monotonic()
                    actions, contexts = await asyncio.to_thread(engine.infer_native, native, list(range(n)))
                    elapsed = time.monotonic()-started
                    with (run/'inference.jsonl').open('a') as f:
                        f.write(json.dumps(dict(batch=active_meta, native_fm=True, seconds=elapsed,
                            timing=engine.last_timing, inference_mode=args.inference_mode,
                            requests=engine.requests, contexts=contexts,
                            chunks=[s['chunks'] for s in engine.slots],
                            actions_min=float(actions.min()), actions_max=float(actions.max()),
                            image_shapes=[{k:list(v.shape) for k,v in o['images'].items()} for o in native]))+'\n')
                    state.update(status='evaluating', updated=time.time(), requests=engine.requests,
                        batch=active_meta, max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        max_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    atomic_json(run/'policy_status.json', state)
                    await socket.send(packb(dict(action=actions[:,0], action_chunk=actions,
                                                 server_timing=dict(infer_ms=elapsed*1000))))
        except websockets.ConnectionClosed:
            pass
        except Exception as error:
            state.update(status='failed', error=repr(error), updated=time.time())
            atomic_json(run/'policy_status.json', state)
            traceback.print_exc()
            stopped.set()
            try: await socket.send(str(error))
            except websockets.ConnectionClosed: pass
        finally:
            owner = None

    async def watcher():
        while not stopped.is_set():
            if (run/'STOP').exists():
                async with gate:
                    receipt = await asyncio.to_thread(engine.verify)
                    atomic_json(run/'evaluation_ack.json', receipt)
                    state.update(status='finished', finished=time.time(), receipt=receipt)
                    atomic_json(run/'policy_status.json', state)
                stopped.set(); return
            await asyncio.sleep(2)

    def health(connection, request):
        if request.path == '/healthz':
            return connection.respond(http.HTTPStatus.OK, 'OK\n')

    async with websockets.serve(handler, '127.0.0.1', args.port, compression=None,
                               max_size=512<<20, ping_timeout=None, process_request=health):
        state.update(status='ready', loaded=time.time(), max_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        atomic_json(run/'policy_status.json', state)
        (run/'policy.ready').touch()
        task = asyncio.create_task(watcher())
        await stopped.wait()
        task.cancel()


if __name__ == '__main__':
    asyncio.run(main())
