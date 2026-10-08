"""Serialize model work without blocking stop requests during GPU updates."""
from pathlib import Path
import argparse
import asyncio
import os
import sys
import time
import traceback
from bootstrap import bootstrap
bootstrap()

SNAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNAP / 'code'))
sys.path.append('/run/ti/behavior_stage3_20260930/tools')
from a4_wire import packb, unpackb
from checkpoint_io import atomic_json
from stage1_engine import Stage1Engine
import torch
import websockets


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    status = dict(status='loading', optimizer_updates=0, started=time.time(),
                  deadline=args.deadline, pid=os.getpid())
    atomic_json(run / 'policy_status.json', status)
    engine = await asyncio.to_thread(Stage1Engine, run / 'checkpoints')
    gate = asyncio.Lock()
    stopped = asyncio.Event()
    draining = False
    last_recovery = time.time()
    final_receipt = None

    def record(state=None, **fields):
        if state is not None:
            status['status'] = state
        status.update(fields, optimizer_updates=engine.trainer.update_count if engine.trainer else 0,
                      gpu_peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                      updated=time.time())
        atomic_json(run / 'policy_status.json', status)

    def dispatch(request):
        kind = request['kind']
        if kind == 'begin':
            if request.get('reward_protocol','legacy_two_task_v1') != engine.reward_protocol:
                raise ValueError('Collector/service reward protocol mismatch')
            if engine.reward_protocol=='shared_terminal_q_v1':
                from shared_reward_math import CONTROL_GAMMA
                if request.get('control_gamma') != CONTROL_GAMMA:
                    raise ValueError('Collector discount differs from shared reward')
            engine.begin(request['task'], request['num_envs'], request['seed'],request.get('episode_metadata'))
            return {}
        if kind == 'infer':
            return engine.infer(request['observations'], request['indices'])
        if kind == 'value':
            return {'values': engine.value(request['observations'], request['indices'])}
        if kind == 'update':
            # Terminal rewards are often in a short tail. Never drop valid
            # on-policy data based on batch length, including a single chunk.
            return engine.trainer.update(engine, request['experience_ids'],
                                         request['advantages'], request['returns'])
        raise ValueError('Unknown request: ' + str(kind))

    async def finish(reason):
        nonlocal draining, final_receipt
        draining = True
        async with gate:
            if final_receipt is not None:
                return final_receipt
            record('saving', stop_reason=reason)
            if engine.trainer is not None and engine.trainer.update_count:
                final_receipt = await asyncio.to_thread(
                    engine.trainer._save_checkpoint,
                    {'reason': reason, 'base_model': 'fduTristin/memlite-stage1'})
            else:
                final_receipt = {'updates': 0, 'no_update': True, 'latest': None}
            atomic_json(run / 'save_ack.json', final_receipt)
            record('finished', exit_result=final_receipt)
            return final_receipt

    async def handler(socket):
        nonlocal last_recovery
        await socket.send(packb({'kind': 'stage1_rl_v1'}))
        async for payload in socket:
            request = unpackb(payload)
            try:
                if request['kind'] == 'finish':
                    result = await finish('finish_rpc')
                    await socket.send(packb({'ok': True, 'response': result}))
                    stopped.set()
                    return
                if draining:
                    await socket.send(packb({'ok': False, 'error': 'service_draining'}))
                    continue
                async with gate:
                    if draining:
                        raise RuntimeError('service_draining')
                    record('updating' if request['kind'] == 'update' else 'working',
                           request_kind=request['kind'])
                    result = await asyncio.to_thread(dispatch, request)
                    if engine.trainer and engine.trainer.update_count and time.time() - last_recovery >= 3600:
                        receipt = await asyncio.to_thread(engine.trainer._save_checkpoint,
                                                         {'reason': 'hourly_recovery'})
                        record(recovery_checkpoint=receipt)
                        last_recovery = time.time()
                    record('training')
                # A disconnected collector must not discard an accepted update.
                try:
                    await socket.send(packb({'ok': True, 'response': result}))
                except websockets.ConnectionClosed:
                    return
            except Exception as error:
                record('failed', error=repr(error))
                traceback.print_exc()
                (run / 'STOP_SAVE').touch()
                try:
                    await socket.send(packb({'ok': False, 'error': repr(error)}))
                except websockets.ConnectionClosed:
                    pass
                return

    async def watchdog():
        while not stopped.is_set():
            if (run / 'STOP_SAVE').exists() or time.time() >= args.deadline - 240:
                try:
                    await finish('stop_file_or_deadline')
                except Exception as error:
                    record('save_failed', save_error=repr(error))
                    traceback.print_exc()
                finally:
                    stopped.set()
                return
            await asyncio.sleep(1)

    async with websockets.serve(handler, '127.0.0.1', args.port,
                                max_size=512 << 20, ping_timeout=None, close_timeout=10):
        record('ready')
        (run / 'policy.ready').touch()
        watcher = asyncio.create_task(watchdog())
        await stopped.wait()
        await watcher


asyncio.run(main())
