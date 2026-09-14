"""Three completed FM500 policies, one matched local512-control window each.

Reuse the unchanged, hash-pinned A4 service and official physics/history code.
Only weights differ between the three arms. This is a fixed-skill training-
instance diagnostic, not autonomous task success or new recovery training data.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
BASE = Path('/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913')
OUTPUT = BASE / 'fm_screen_prefix512_v1'
WORK = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910')
RUNTIME = WORK / 'native_a3_aligned_prefix_runtime_v2'
SOURCE = WORK / 'a3_history_serving_candidate_v1'
COMPOSITION = WORK / 'a3_serving_composition_v2'
ADAPTER = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1')
SERVICE = Path('/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_prefix_20260913/scripts/experiments/serve_low_fm_prefix.py')
SERVICE_SHA = 'c4bc099d1ac5d4d96e311498d6d8c8309900f8cb1e91c140be2f1d901057109f'
PYTHON = '/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python3.10'
BEHPY = '/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python'
PORT = 8786
SCREEN = {
    'fm_control_v1': 'def222a6674e6ac92e6ee982c22836b789240f1542c459d5de2111cd646a244e',
    'fm_beta_stratified_v2': '05ea17bcbbf62437b70e286ce58e6113f20cfe1dd7b79eb56010b73d1f7a6c1e',
    'fm_exec_weight2_v2': '101c4b32d7df3bacd529159474f5c1f1720ade2a93614b5d68ea43de5e7eca2f',
}
RECIPE = dict(environment_seed=0, policy_seed=17, prefix_controls=448, max_chunks=32,
              execute_steps=16, max_model_controls=512, total_model_controls=1536,
              gpu_model=1, gpu_simulator=1, optimizer_updates=0, extra_offline_generations=0,
              training_admissible=False, full_task_success_rate_claim=False, automatic_retry=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def imports():
    sys.path[:0] = [str(RUNTIME), str(WORK / 'c1_v2'), str(ADAPTER), str(SOURCE / 'src'), str(SOURCE)]


def clean_commit():
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Use a clean immutable screening checkout')
    return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()


def bound_window(window, context):
    from c1_feedback.official_factory_c1 import _verify_window_context
    _verify_window_context(window, context)
    if (context.split != 'train' or context.task_id != 0 or context.instance_id != 138
            or window.official_mode != 'train' or window.seed != 0
            or window.task_name != 'turning_on_radio' or window.execute_steps != 16
            or window.max_chunks != 80 or len(window.frozen_window().prefix_actions) != 448):
        raise ValueError('Only the original radio121/138/env0/448-prefix diagnostic is allowed')
    bounded = replace(window, max_chunks=32)
    _verify_window_context(bounded, context)
    return bounded


def validate_response(response):
    import numpy as np
    actions = np.asarray(response.get('actions'))
    alignment = response.get('inference_alignment_admission', {})
    expected = dict(padded_dimensions=[7, 8, 17, 18], real_control_dimensions=23,
                    image_history_frames=6, action_history_steps=0, action_execution_start_index=0,
                    expert_actions_used=False)
    if (actions.shape != (16, 23) or actions.dtype != np.float32 or not np.isfinite(actions).all()
            or any(alignment.get(key) != value for key, value in expected.items())):
        raise ValueError('Actual FM reply changed the23D/mask/history/execution0:16 contract')
    return response


def validate_hello(hello, manifest, name):
    identity = hello.get('identity', {})
    if (hello.get('mode') != 'native_skill_fm_history6' or hello.get('components') != ['low']
            or identity.get('kind') != 'formal_a2_native_skill_fm_prefix_history6_service'
            or identity.get('checkpoint_sha256') != SCREEN[name] or identity.get('checkpoint_step') != 500
            or identity.get('service_entry_sha256') != SERVICE_SHA
            or identity.get('initial_action_count') != 448
            or identity.get('prefix_window_sha256') != manifest['window_sha256']
            or identity.get('num_obs_steps') != 6
            or identity.get('history_context_kind') != 'actual_demo_prefix_then_policy'
            or not identity.get('checkpoint_load', {}).get('full_base_and_adapter_bitwise_equal')):
        raise ValueError('The actual private endpoint is not the full approved FM500 policy')
    return identity


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('Never overwrite or repeat the three registered physical runs')
    commit = clean_commit()
    prior_path = WORK / 'a3_aligned_radio_e121_l1_v2/manifest.json'
    proof = BASE / 'fm_screen_actions_v1/result.json'
    if (sha(prior_path) != '7fefd3894d84d8819508b152916664e8dbd35290904c6575f10b0304734d70b4'
            or sha(proof) != 'a9906fa5f4c254a1f5ce7ef0a9ce0456ebf49e8e8d96177f74dda7c51e1adca8'
            or not read(proof)['complete'] or sha(SERVICE) != SERVICE_SHA):
        raise RuntimeError('Original physics, completed FM comparison or frozen service changed')
    prior = read(prior_path)
    if prior['source_episode_index'] != 121 or prior['policy_seed'] != 17:
        raise ValueError('Wrong original paired source/seed')
    files = dict(prior['files'])
    for path, digest in files.items():
        if sha(path) != digest:
            raise RuntimeError('Original evaluator/history dependency changed: ' + path)
    imports()
    from c1_feedback.official_factory_c1 import load_c1_window_and_context
    window, context = load_c1_window_and_context(prior['window_path'], prior['context_path'])
    bound_window(window, context)
    for name, digest in SCREEN.items():
        root = BASE / name
        inspection, state = read(root / 'formal_checkpoint_inspection.json'), read(root / 'status.json')
        if (state['state'] != 'complete' or not inspection['passed'] or inspection['step'] != 500
                or inspection['checkpoint_sha256'] != digest or inspection['actual_adam_states'] != 504
                or not inspection['frozen_bitwise_unchanged']):
            raise RuntimeError('Only completed, verified500 policies may enter physics')
        for path in (root / 'formal_checkpoint_inspection.json', root / 'status.json',
                     root / 'formal/coordination_run_receipt.json', root / 'method_spec.json'):
            files[str(path)] = sha(path)
    for path in (HERE, SERVICE, prior_path, proof):
        files[str(path)] = sha(path)
    OUTPUT.mkdir()
    contexts = {}
    for name in SCREEN:
        directory = OUTPUT / name
        directory.mkdir()
        copied = read(prior['context_path'])
        copied['run_id'] = 'FM500-prefix512-' + name + '-20260914-v1'
        path = directory / 'event_context.json'
        write(path, copied)
        contexts[name], files[str(path)] = str(path), sha(path)
    manifest = dict(kind='three_FM500_matched_radio_prefix512_v1', commit=commit, files=files,
                    recipe=RECIPE, screen=SCREEN, contexts=contexts, window_path=prior['window_path'],
                    window_sha256=prior['window_sha256'], port=PORT,
                    subgoal_origin='same_frozen_supervised_GRASP_not_autonomous_planning',
                    old_a4_result_is_historical_reference_only=True)
    write(OUTPUT / 'manifest.json', manifest)
    return manifest


def validate():
    manifest = read(OUTPUT / 'manifest.json')
    if (manifest['commit'] != clean_commit() or manifest['recipe'] != RECIPE or manifest['screen'] != SCREEN
            or manifest['port'] != PORT or manifest['kind'] != 'three_FM500_matched_radio_prefix512_v1'):
        raise RuntimeError('Registered source, methods or finite budget changed')
    for path, digest in manifest['files'].items():
        if sha(path) != digest:
            raise RuntimeError('Registered dependency changed: ' + path)
    return manifest


def free_mib():
    return int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())


async def socket_gate(manifest, name):
    imports()
    import websockets
    from g05.utils.websocket import unpackb
    async with websockets.connect(f'ws://127.0.0.1:{PORT}', max_size=64 << 20, ping_timeout=None) as ws:
        identity = validate_hello(unpackb(await asyncio.wait_for(ws.recv(), timeout=60)), manifest, name)
    # Identity greeting only: no begin, no RNG reset, no neural invocation.
    write(OUTPUT / name / 'socket_gate.json', dict(passed=True, model_calls=0, identity=identity))


def supervise():
    manifest = validate()
    completed = []
    for name, digest in SCREEN.items():
        if free_mib() < 40 * 1024 or shutil.disk_usage(BASE).free < 120 * 1024**3:
            raise RuntimeError('Insufficient GPU/disk headroom; keep other jobs untouched')
        with socket.socket() as check:
            check.bind(('127.0.0.1', PORT))
        directory, formal = OUTPUT / name, BASE / name / 'formal'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        env.pop('PYTHONPATH', None)
        env.pop('MEMLITE_CANDIDATE_ROOT', None)
        command = [PYTHON, str(SERVICE), '--run', str(formal), '--checkpoint', str(formal / 'checkpoints/step_500.pt'),
            '--checkpoint-step', '500', '--checkpoint-sha', digest, '--runtime', str(RUNTIME),
            '--config', str(COMPOSITION / 'serving_config.yaml'), '--adapter', str(ADAPTER),
            '--output-dir', str(directory / 'service'), '--port', str(PORT), '--serving-source', str(SOURCE),
            '--snapshot', str(COMPOSITION / 'native_snapshot.json'), '--source-receipt', str(COMPOSITION / 'source_receipt.json'),
            '--initial-action-count', '448', '--prefix-window-sha', manifest['window_sha256'],
            '--inference-alignment-receipt', str(RUNTIME / 'inference_alignment_receipt.json')]
        with (directory / 'service.log').open('x') as stream:
            service = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        write(directory / 'service.launch.json', dict(pid=service.pid, command=command, started_unix=time.time()))
        try:
            for _ in range(180):
                if service.poll() is not None:
                    raise RuntimeError('Private FM service exited before readiness')
                if 'native_skill_fm_history6 websocket ready' in (directory / 'service.log').read_text():
                    break
                time.sleep(5)
            else:
                raise TimeoutError('Private service did not initialize within15 minutes')
            asyncio.run(socket_gate(manifest, name))
            if free_mib() < 25 * 1024:
                raise RuntimeError('Simulator needs25 GiB remaining; never evict training')
            sim_env = dict(env)
            sim_env.pop('CUDA_VISIBLE_DEVICES', None)
            sim_env.update(OMNI_KIT_ACCEPT_EULA='1', OMNIGIBSON_HEADLESS='1', OMNIGIBSON_NO_OMNI_LOGS='1',
                OMNIGIBSON_GPU_ID='1', BEHAVIOR_ACTION_STEPS='1',
                OMNIGIBSON_APPDATA_PATH=str(WORK / 'kit_c1_gpu1_appdata_v1'),
                LD_LIBRARY_PATH='/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/pymeshlab/lib')
            command = [BEHPY, str(HERE), 'simulate', '--arm', name]
            with (directory / 'rollout.log').open('x') as stream:
                child = subprocess.Popen(command, env=sim_env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                write(directory / 'rollout.launch.json', dict(pid=child.pid, command=command, started_unix=time.time()))
                if child.wait() != 0:
                    raise RuntimeError('Bounded physics run failed; no retry or extra arm')
            result = read(directory / 'actual_rollout/collection_result.json')
            if (result['status'] != 'complete' or result['prefix_actions_executed'] != 448
                    or not 0 < result['native_policy_actions_consumed'] <= 512
                    or result['low_service_identity']['checkpoint_sha256'] != digest
                    or len(result['actual_history_requests']) > 32 or result['training_admissible']):
                raise RuntimeError('Completed physics identity/budget failed')
            receipt = dict(complete=True, result_sha256=sha(directory / 'actual_rollout/collection_result.json'),
                          model_controls=result['native_policy_actions_consumed'], personal_review_pending=True,
                          full_task_success_rate_claim=False, automatic_retry=False)
            write(directory / 'completion.json', receipt)
            completed.append(dict(arm=name, **receipt))
        finally:
            if service.poll() is None:
                service.terminate()
                try:
                    service.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    service.kill()
                    service.wait()
            write(directory / 'private_service_exit.json', dict(pid=service.pid, returncode=service.returncode,
                                                               other_jobs_stopped=False))
    write(OUTPUT / 'completion.json', dict(complete=True, arms=completed, personal_review_pending=True,
          full_task_success_rate_claim=False, optimizer_updates=0, automatic_retry=False))


def simulate(name):
    if name not in SCREEN:
        raise ValueError('Unknown or unregistered FM500 arm')
    manifest = validate()
    imports()
    from c1_feedback.official_factory_c1 import load_c1_window_and_context
    from native_a2_prefix_client import NativeA2PrefixClient
    import a2_prefix_official_factory as factory
    original_client = NativeA2PrefixClient

    class CheckedClient(original_client):
        def validate_identity(self, hello):
            super().validate_identity(hello)
            return validate_hello(hello, manifest, name)

        def request(self, observation, installed, execute_steps):
            if len(self.requests) >= 32 or execute_steps != 16:
                raise ValueError('The sole physical session exhausted its32-chunk budget')
            return validate_response(super().request(observation, installed, execute_steps))

    # Process-local client admission only; original physics, recording and
    # neural service source files stay unchanged. Validate before env.step.
    factory.NativeA2PrefixClient = CheckedClient
    window, context = load_c1_window_and_context(manifest['window_path'], manifest['contexts'][name])
    window = bound_window(window, context)
    result = factory.run_official_a2_prefix_window(window=window, context=context, source_root=SOURCE,
        policy_uri=f'ws://127.0.0.1:{PORT}', expected_checkpoint_sha256=SCREEN[name],
        prefix_window_sha256=manifest['window_sha256'], output_dir=OUTPUT / name / 'actual_rollout',
        gpu=1, policy_seed=17)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'supervise', 'simulate'))
    parser.add_argument('--arm', choices=tuple(SCREEN))
    args = parser.parse_args()
    if (args.mode == 'simulate') != (args.arm is not None):
        parser.error('--arm is required only for simulate')
    if args.mode == 'start':
        lock = (BASE / 'fm_screen_prefix512_v1.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        command = [PYTHON, str(HERE), 'supervise']
        with (OUTPUT / 'supervisor.log').open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        receipt = dict(pid=child.pid, command=command, started_unix=time.time(),
                       manifest_sha256=sha(OUTPUT / 'manifest.json'), recipe=RECIPE)
        write(OUTPUT / 'launch.json', receipt)
        print(json.dumps(receipt), flush=True)
        return
    try:
        supervise() if args.mode == 'supervise' else simulate(args.arm)
    except BaseException:
        target = OUTPUT if args.mode == 'supervise' else OUTPUT / args.arm
        write(target / (args.mode + '_failure.json'), dict(failed=True, traceback=traceback.format_exc(),
                                                        automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
