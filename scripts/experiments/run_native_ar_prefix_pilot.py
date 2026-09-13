"""One native-task AR500 local physics pilot: original448 + at most256 controls.

The original GRASP context is evaluator-only. Actor receives the task and real
observations, not a fixed skill or outcome. Never retry/deploy or modify old runs.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = Path('/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913')
OUTPUT = BASE / 'ar_native_prefix_pilot_v1'
WORK = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910')
HISTORY = WORK / 'native_a3_aligned_prefix_runtime_v2'
ADAPTER = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1')
SIM_SOURCE = WORK / 'a3_history_serving_candidate_v1'  # Serialization/evaluator imports only, not the AR model.
PYTHON = '/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python3.10'
BEHPY = '/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python'
CHECKPOINT_SHA = '639e64aeeb251113b807751f234077595165e65e9dd9e3b66cd7c4661f9df963'
PORT = 8785


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def prepare():
    if OUTPUT.exists() or subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('New bounded output and clean pinned source required')
    original_path = WORK / 'a3_aligned_radio_e121_l1_v2/manifest.json'
    proof = BASE / 'ar_native_actor_wire_probe_v1/result.json'
    if (sha(original_path) != '7fefd3894d84d8819508b152916664e8dbd35290904c6575f10b0304734d70b4'
            or sha(proof) != '2570ba7d943b39a05953bc7bcc69b216709955dafb2d3eda6575e45062d82c75'
            or not read(proof)['complete']):
        raise RuntimeError('Required original physics and actual native actor/bridge evidence changed')
    original = read(original_path)
    files = dict(original['files'])
    for path, expected in files.items():
        if sha(path) != expected:
            raise RuntimeError('Preserved original evaluator/history/context changed: ' + path)
    window = read(original['window_path'])
    if (window['seed'] != 0 or window['official_mode'] != 'train' or window['instance_id'] != 138
            or window['task_name'] != 'turning_on_radio' or window['execute_steps'] != 16
            or original['source_episode_index'] != 121 or original['policy_seed'] != 17):
        raise RuntimeError('Only the registered radio train121/138/env0/policy17 pilot is allowed')
    OUTPUT.mkdir()
    context = read(original['context_path'])
    context['run_id'] = 'native-task-AR500-schema-radio-prefix256-20260913-v1'
    write(OUTPUT / 'event_context.json', context)
    for path in (original_path, proof, OUTPUT / 'event_context.json', Path(__file__),
                 HERE / 'native_ar_prefix_client.py', HERE / 'serve_native_ar_prefix.py',
                 HERE / 'native_ar_actor_runtime.py', HERE / 'native_action_observations.py',
                 *sorted((SIM_SOURCE / 'src/g05/utils/websocket').glob('*.py'))):
        files[str(path)] = sha(path)
    manifest = dict(kind='bounded_native_task_ar500_prefix_pilot_v1',
        code_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        files=files, window_path=original['window_path'], context_path=str(OUTPUT / 'event_context.json'),
        window_sha256=original['window_sha256'], checkpoint_sha256=CHECKPOINT_SHA,
        policy_seed=17, environment_seed=0, prefix_actions=448, max_chunks=16, execute_steps=16,
        max_model_controls=256, max_generations=16, gpu_model=1, gpu_simulator=1, port=PORT,
        static_format_forced=True, actor_conditioning='native_task', evaluator_subgoal_sent=False,
        training_admissible=False, optimizer_updates=0, automatic_retry=False, full_task_success_rate_claim=False)
    write(OUTPUT / 'manifest.json', manifest)
    return manifest


def validate_manifest(path):
    if Path(path) != OUTPUT / 'manifest.json':
        raise ValueError('Only the registered local pilot manifest is accepted')
    manifest = read(path)
    expected = dict(kind='bounded_native_task_ar500_prefix_pilot_v1', checkpoint_sha256=CHECKPOINT_SHA,
        policy_seed=17, environment_seed=0, prefix_actions=448, max_chunks=16, execute_steps=16,
        max_model_controls=256, max_generations=16, gpu_model=1, gpu_simulator=1, port=PORT,
        static_format_forced=True, actor_conditioning='native_task', evaluator_subgoal_sent=False,
        training_admissible=False, optimizer_updates=0, automatic_retry=False, full_task_success_rate_claim=False)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Pilot identity, observation boundary or finite budget changed')
    if (subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() != manifest['code_commit']
            or subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()):
        raise RuntimeError('Pilot source is not its clean immutable Git snapshot')
    for source, digest in manifest['files'].items():
        if sha(source) != digest:
            raise RuntimeError('Pinned pilot dependency changed: ' + source)
    return manifest


async def socket_gate(manifest):
    # No begin and no neural call: leave the sole budgeted actor session for physics.
    from action_training_runtime import bootstrap
    bootstrap()
    from g05.utils.websocket import packb, unpackb
    from native_ar_prefix_client import validate_native_identity
    import websockets
    async with websockets.connect(f'ws://127.0.0.1:{PORT}', max_size=128 << 20, ping_timeout=None) as ws:
        hello = unpackb(await asyncio.wait_for(ws.recv(), timeout=60))
        identity = validate_native_identity(hello, checkpoint_sha=CHECKPOINT_SHA,
            prefix_actions=448, window_sha=manifest['window_sha256'])
        await ws.send(packb(dict(kind='inspect')))
        reply = unpackb(await asyncio.wait_for(ws.recv(), timeout=60))
        if (reply.get('ok') is not True or reply['response']['identity'] != identity
                or reply['response']['model_calls'] != 0 or reply['response']['session_begun']):
            raise RuntimeError('Private socket identity gate must not consume a model call/session')
    write(OUTPUT / 'socket_gate.json', dict(passed=True, actual_model_calls=0, identity=identity))


def free_mib(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())


def supervise():
    manifest = validate_manifest(OUTPUT / 'manifest.json')
    if free_mib(1) < 40 * 1024 or shutil.disk_usage(BASE).free < 120 * 1024**3:
        raise RuntimeError('Not enough GPU/disk headroom; other jobs remain untouched')
    with socket.socket() as check:
        check.bind(('127.0.0.1', PORT))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None)
    command = [PYTHON, str(HERE / 'serve_native_ar_prefix.py'), '--manifest', str(OUTPUT / 'manifest.json')]
    with (OUTPUT / 'service.log').open('x') as stream:
        service = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    write(OUTPUT / 'service.launch.json', dict(pid=service.pid, command=command, started_unix=time.time()))
    try:
        for _ in range(180):
            if service.poll() is not None:
                raise RuntimeError('Private AR service exited before readiness')
            if 'native_task_ar_history6 websocket ready' in (OUTPUT / 'service.log').read_text():
                break
            time.sleep(5)
        else:
            raise TimeoutError('AR service initialization exceeded its 15-minute engineering gate')
        asyncio.run(socket_gate(manifest))
        if free_mib(1) < 25 * 1024:
            raise RuntimeError('Simulator needs 25 GiB remaining headroom; do not evict training')
        sim_env = dict(env)
        sim_env.pop('CUDA_VISIBLE_DEVICES', None)
        sim_env.update(OMNI_KIT_ACCEPT_EULA='1', OMNIGIBSON_HEADLESS='1', OMNIGIBSON_NO_OMNI_LOGS='1',
            OMNIGIBSON_GPU_ID='1', BEHAVIOR_ACTION_STEPS='1',
            OMNIGIBSON_APPDATA_PATH=str(WORK / 'kit_c1_gpu1_appdata_v1'),
            LD_LIBRARY_PATH='/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/pymeshlab/lib')
        command = [BEHPY, str(HERE / 'run_native_ar_prefix_pilot.py'), 'simulate']
        with (OUTPUT / 'rollout.log').open('x') as stream:
            simulation = subprocess.Popen(command, cwd=WORK, env=sim_env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
            write(OUTPUT / 'rollout.launch.json', dict(pid=simulation.pid, command=command, started_unix=time.time()))
            if simulation.wait() != 0:
                raise RuntimeError('Bounded native AR physics pilot failed; no automatic retry')
        result = read(OUTPUT / 'actual_rollout/collection_result.json')
        if (result['status'] != 'complete' or result['prefix_actions_executed'] != 448
                or not 0 < result['native_policy_actions_consumed'] <= 256
                or result['low_service_identity']['checkpoint_sha256'] != CHECKPOINT_SHA
                or any(row['evaluator_subgoal_sent'] for row in result['actual_history_requests'])):
            raise RuntimeError('Physics completion identity/budget/actor boundary failed')
        write(OUTPUT / 'completion.json', dict(complete=True,
            result_sha256=sha(OUTPUT / 'actual_rollout/collection_result.json'), personal_review_pending=True,
            full_task_success_rate_claim=False, automatic_extra_rollouts=False))
    finally:
        if service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()
        write(OUTPUT / 'private_service_exit.json', dict(pid=service.pid,
            returncode=service.returncode, other_jobs_stopped=False))


def simulate():
    manifest = validate_manifest(OUTPUT / 'manifest.json')
    sys.path[:0] = [str(HISTORY), str(WORK / 'c1_v2'), str(ADAPTER), str(SIM_SOURCE / 'src'), str(SIM_SOURCE)]
    from c1_feedback.official_factory_c1 import (load_c1_window_and_context, _verify_window_context,
        bootstrap_g05_source, install_frozen_semantic_subgoal, _load_physical_oracle_module,
        physical_stability, _make_evidence_writer, OfficialEvaluatorSession, TerminalAwareHarness, _DEFAULT_ORACLE_MODULE)
    from c1_feedback.feedback_c1 import C1EventRecorder, C1FeedbackDriver, _atomic_json
    from native_ar_prefix_client import make_client
    import numpy as np
    import imageio.v2 as imageio
    bootstrap_g05_source(SIM_SOURCE)
    window, context = load_c1_window_and_context(manifest['window_path'], manifest['context_path'])
    _verify_window_context(window, context)
    if window.seed != 0 or len(window.frozen_window().prefix_actions) != 448 or context.split != 'train':
        raise ValueError('Original seed/prefix/source boundary changed')
    # The physical evaluator keeps its original GRASP truth target; the AR
    # client intentionally discards installed_subgoal before any wire request.
    installed = install_frozen_semantic_subgoal(window.semantic_subgoal)
    bounded = replace(window.frozen_window(), max_chunks=16)
    oracle_module = _load_physical_oracle_module(_DEFAULT_ORACLE_MODULE)
    oracle = physical_stability.make_stable_scene_oracle(oracle_module, stable_frames=6)
    output = OUTPUT / 'actual_rollout'
    recorder = C1EventRecorder(output, context, history_action_stride=16)
    evidence_writer = _make_evidence_writer(module=oracle_module, output_dir=output, context=context,
        physical_wrapper_path=physical_stability.__file__)
    client = make_client(f'ws://127.0.0.1:{PORT}', source_root=SIM_SOURCE,
        expected_checkpoint_sha256=CHECKPOINT_SHA, prefix_actions=448,
        prefix_window_sha256=manifest['window_sha256'], seed=17)
    result = dict(kind='native_task_ar500_schema_local_prefix_pilot', status='running',
        actor_receives_evaluator_subgoal=False, actor_receives_physical_truth=False,
        training_admissible=False, full_task_success_rate_claim=False, source_episode=121,
        source_instance=138, static_format_forced=True, max_model_controls=256)
    with OfficialEvaluatorSession(window, gpu=1) as original:
        harness = TerminalAwareHarness(original)
        video = None
        try:
            video = imageio.get_writer(str(output / 'rollout.mp4'), fps=15, codec='libx264', quality=7, macro_block_size=2)
            def capture(frame):
                client.capture(frame)
                if frame.source_frame % 2 == 0:
                    video.append_data(np.moveaxis(frame.images['head_rgb'], 0, -1))
            recorder.on_capture = capture
            driver = C1FeedbackDriver(action_env=harness, physical_env=harness.evaluator.env,
                observation_provider=harness.observation, request_low_chunk=client.request,
                physical_oracle=oracle, recorder=recorder, context=context, evidence_writer=evidence_writer)
            harness.reset()
            result.update(driver.run(window=bounded, installed_subgoal=installed))
            result['status'] = 'complete'
        except BaseException:
            result.update(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            result.update(low_service_identity=client.identity, actual_history_requests=client.requests)
            pending_error = sys.exc_info()[0] is not None
            close_errors = []
            for close in (client.close, *([video.close] if video is not None else [])):
                try:
                    close()
                except BaseException:
                    close_errors.append(traceback.format_exc())
            if close_errors:
                result.update(status='failed', close_errors=close_errors)
            result['persisted_before_official_evaluator_shutdown'] = True
            _atomic_json(output / 'collection_result.json', result)
            if close_errors and not pending_error:
                raise RuntimeError('Native AR client/video did not close cleanly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'supervise', 'simulate'))
    args = parser.parse_args()
    if args.mode == 'start':
        lock = (BASE / 'native_ar_prefix_pilot_v1.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        command = [PYTHON, str(Path(__file__)), 'supervise']
        with (OUTPUT / 'supervisor.log').open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        receipt = dict(pid=child.pid, command=command, started_unix=time.time(),
            manifest_sha256=sha(OUTPUT / 'manifest.json'), max_model_controls=256, automatic_retry=False)
        write(OUTPUT / 'launch.json', receipt)
        print(json.dumps(receipt), flush=True)
        return
    try:
        supervise() if args.mode == 'supervise' else simulate()
    except BaseException:
        write(OUTPUT / f'{args.mode}_failure.json', dict(failed=True, traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
