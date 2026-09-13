"""Private, single-session native-task AR500 prefix service, at most 16 calls."""
import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np

from native_ar_prefix_client import task_only_request


class NativeARSession:
    def __init__(self, *, identity, infer, ingress, seed_rng, publish_chunk):
        self.identity, self.infer, self.ingress = identity, infer, ingress
        self.seed_rng, self.publish_chunk = seed_rng, publish_chunk
        self.owner, self.calls = None, 0

    def begin(self, owner, request):
        if (self.owner is not None or request != dict(kind='begin', seed=17)
                or type(request.get('seed')) is not int):
            raise ValueError('Only one seed17 session; no reset/retry budget')
        self.seed_rng(17)
        self.owner = owner
        return dict(seed=17)

    def chunk(self, owner, request):
        if (self.owner is None or owner is not self.owner or self.calls >= 16
                or set(request) != {'kind', 'observation', 'execute_steps'}
                or request['kind'] != 'native_task_chunk'):
            raise ValueError('Wrong session, forbidden fields or exhausted native AR budget')
        task_only_request(request['observation'], request['execute_steps'])
        observed, admission = self.ingress.prepare(request['observation'], execute_steps=16)
        index = self.calls
        self.calls += 1  # A failed neural attempt still consumes the finite call budget.
        result = self.infer(observed)
        actions = result['actions']
        if (actions.shape != (16, 23) or actions.dtype != np.float32 or not np.isfinite(actions).all()
                or result['schema'].get('complete') is not True
                or result['schema'].get('rule_safe_clamp') is not False):
            raise ValueError('Invalid complete native AR/official23 output')
        self.publish_chunk(index, result, admission)
        self.ingress.commit(admission)
        return dict(actions=actions, actor_identity=self.identity,
            history_admission=dict(consumed_actions=admission['consumed_actions'], execute_steps=16,
                anchor_hashes={str(k): v for k, v in admission['anchor_hashes'].items()}),
            execution_start=0, execution_steps=16, schema=result['schema'], timing=result['timing'],
            teacher_or_planner_in_actor=False)


async def serve(manifest, output):
    import torch
    import websockets
    from g05.utils.websocket import packb, unpackb
    from native_ar_actor_runtime import load_native_actor, runtime_dependencies, raw_actions_to_wire
    from native_action_observations import infer_native_task_observation
    from probe_ar_execution_codec import publish

    model, adapter, receipt = load_native_actor()
    dependencies = runtime_dependencies()
    if receipt['checkpoint_sha256'] != manifest['checkpoint_sha256']:
        raise RuntimeError('Service loaded a different native checkpoint')
    identity = dict(kind='native_task_ar500_prefix_history6_service',
        checkpoint_sha256=receipt['checkpoint_sha256'], initial_action_count=448,
        prefix_window_sha256=manifest['window_sha256'], num_obs_steps=6, execution_start=0,
        max_chunks=16, static_format_forced=True, teacher_or_planner_in_actor=False,
        actual_load_receipt=receipt)
    publish(output / 'load_receipt.json', identity)
    def seed_rng(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    def infer(observed):
        result = infer_native_task_observation(model, adapter, observed, device='cuda:0', constrain_format=True)
        wire = raw_actions_to_wire(result['grouped_raw_action'], dependencies.bridge)
        if not np.array_equal(wire[:, 6], np.zeros(32, dtype=np.float32)):
            raise RuntimeError('Raw trunk channel3 left official-reset support; do not clamp it')
        return dict(actions=wire[:16].copy(), raw23_all32=wire.tolist(), schema=result['schema'],
            trace=result['schema_trace'], timing=result['timing'],
            normalized_action=result['generated']['action'].detach().cpu().tolist())
    def record(index, result, admission):
        saved = {key: value for key, value in result.items() if key != 'actions'}
        publish(output / f'chunk_{index:02d}.json', dict(**saved,
            history=dependencies.history.history_admission_to_wire(admission),
            teacher_or_planner_in_actor=False, physics_execution_not_certified_here=True))
    session = NativeARSession(identity=identity, infer=infer,
        ingress=dependencies.history.LowHistoryIngress(initial_action_count=448),
        seed_rng=seed_rng, publish_chunk=record)
    async def handler(ws):
        owner = object()
        await ws.send(packb(dict(mode='native_task_ar_history6', components=['native_task_actor'], identity=identity)))
        try:
            async for message in ws:
                request = unpackb(message)
                if request == {'kind': 'inspect'}:
                    response = dict(identity=identity, model_calls=session.calls, session_begun=session.owner is not None)
                elif isinstance(request, dict) and request.get('kind') == 'begin':
                    response = session.begin(owner, request)
                else:
                    response = session.chunk(owner, request)
                await ws.send(packb(dict(ok=True, response=response)))
        except Exception as exc:
            print(f'Native AR connection failed: {type(exc).__name__}: {exc}', flush=True)
            await ws.send(packb(dict(ok=False, error=f'{type(exc).__name__}: {exc}')))
        finally:
            if session.owner is owner:
                publish(output / 'session_closed.json', dict(actual_neural_attempts=session.calls,
                    history_last_count=session.ingress.last_count, automatic_retry=False))
    async with websockets.serve(handler, '127.0.0.1', manifest['port'], max_size=128 << 20, ping_timeout=None):
        print('native_task_ar_history6 websocket ready', flush=True)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    from run_native_ar_prefix_pilot import validate_manifest
    manifest = validate_manifest(args.manifest)
    output = args.manifest.parent / 'service'
    output.mkdir(exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(manifest['gpu_model'])
    # Establish the immutable G05 extension path before importing its public helpers.
    from action_training_runtime import bootstrap
    bootstrap()
    asyncio.run(serve(manifest, output))


if __name__ == '__main__':
    main()
