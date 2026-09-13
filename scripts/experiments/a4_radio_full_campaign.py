"""Run exactly three full autonomous radio episodes, without replay or retries.

Owns one A4 low service, one unchanged B-final high service, and one serial
simulator. Existing services/source/data are read-only and remain untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from paired_a3_a4_actions import WORK, CHECKPOINTS, PYTHON, sha, read, write_new
from radio_full_manifest import build_manifest, validate_manifest, RUNTIME, HIGH_SHA

HERE = Path(__file__).resolve().parent
MAIN = Path('/mnt/sdc1/robodojo/GalaxeaVLA')
ADAPTER = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1')
BEHPY = '/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python'
SOURCE = WORK / 'a3_history_serving_candidate_v1'
COMPOSITION = WORK / 'a3_serving_composition_v2'
OUTPUT = Path('/mnt/sdc1/robodojo/behavior_dev/a4_radio_full_20260913_v1')
LOW_PORT, HIGH_PORT = 8783, 8784
HIGH_RUN = WORK / 'formal_b_parent_format_1500_v1'
LOW_RUN = CHECKPOINTS['A4'][0].parents[1]


def check_resources():
    for gpu, required in ((0, 35 * 1024), (1, 70 * 1024), (2, 25 * 1024)):
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        if free < required:
            raise RuntimeError(f'Insufficient GPU {gpu} headroom; no existing jobs stopped')
    if shutil.disk_usage(OUTPUT.parent).free < 120 * 1024**3:
        raise RuntimeError('Insufficient disk reserve')
    for port in (LOW_PORT, HIGH_PORT):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))


def model_env(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    for name in ('PYTHONPATH', 'MEMLITE_CANDIDATE_ROOT'):
        env.pop(name, None)
    env['LD_LIBRARY_PATH'] = str(MAIN / '.venv/lib/python3.10/site-packages/nvidia/npp/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    return env


def start_child(label, command, env, *, cwd=MAIN):
    with (OUTPUT / f'{label}.log').open('x') as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
    write_new(OUTPUT / f'{label}.launch.json', dict(pid=child.pid, command=command, started_unix=time.time()))
    return child


async def readiness(high, low):
    """Handshake actual services; inference probe uses original train observations."""
    import websockets
    import numpy as np
    sys.path[:0] = [str(RUNTIME), str(ADAPTER), str(SOURCE / 'src'), str(SOURCE)]
    from g05.utils.websocket import packb, unpackb
    from native_ab_phase_identity import validate_high_identity
    from native_b_session import PlannerHistory, CAMERAS
    from native_a2_history import LowHistoryIngress, WIDTHS, history_admission_to_wire
    manifest = read(OUTPUT / 'manifest.json')
    for label, child, port, expected_mode, expected_sha in (
            ('low', low, LOW_PORT, 'native_skill_fm_history6', CHECKPOINTS['A4'][1]),
            ('high', high, HIGH_PORT, 'native_b_high_only', HIGH_SHA)):
        for attempt in range(180):
            if child.poll() is not None:
                raise RuntimeError(f'Private {label} service exited; inspect log')
            try:
                async with websockets.connect(f'ws://127.0.0.1:{port}', open_timeout=2, max_size=64 << 20) as ws:
                    hello = unpackb(await asyncio.wait_for(ws.recv(), 10))
                    if hello['mode'] != expected_mode or hello['identity']['checkpoint_sha256'] != expected_sha:
                        raise ValueError('Actual endpoint has wrong model identity')
                    if label == 'high':
                        validate_high_identity(hello['identity'], checkpoint_sha=HIGH_SHA,
                            lineage_sha=manifest['high_phase_lineage_sha256'])
                    break
            except (OSError, TimeoutError, asyncio.TimeoutError):
                await asyncio.sleep(5)
        else:
            raise TimeoutError(f'Private {label} service failed to become ready')
    receipt = read(OUTPUT / 'low/load_receipt.json')
    if (receipt['checkpoint_step'] != 2500 or not receipt['checkpoint_load']['full_base_and_adapter_bitwise_equal']
            or receipt['service_entry_sha256'] != sha(HERE / 'serve_low_fm_full.py')):
        raise ValueError('A4 full checkpoint was not restored')
    original = WORK / 'demo_radio_e121_alignment_v2_persist'
    original_manifest = read(original / 'manifest.json')
    context = read(original_manifest['context']['path'])
    rows = []
    async with websockets.connect(f'ws://127.0.0.1:{LOW_PORT}', max_size=64 << 20, ping_timeout=None) as ws:
        await ws.recv()
        async def call(request):
            await ws.send(packb(request))
            reply = unpackb(await asyncio.wait_for(ws.recv(), 600))
            if reply.get('ok') is not True:
                raise ValueError('Actual full-service probe rejected: ' + str(reply))
            return reply['response']
        await call(dict(kind='begin', seed=17))
        history, ingress = PlannerHistory(), LowHistoryIngress()
        for frame in range(0, 112, 16):
            path = original / 'actual_replay/observations' / f'f{frame:08d}.npz'
            with np.load(path, allow_pickle=False) as obs:
                if obs['source_frame'].tolist() != [frame]:
                    raise ValueError('Wrong original observation clock')
                observation = dict(task=context['task_name'], frequency=30., embodiment_type='galaxea_r1pro',
                    images={key: obs[key].copy() for key in CAMERAS},
                    state={key: obs['state_' + key].copy() for key in WIDTHS})
            history.append(observation, frame)
            wire = history.request_observation()
            _, admission = ingress.prepare(wire, execute_steps=16)
            segment = next(s for s in original_manifest['segments'] if s['segment_start'] <= frame < s['segment_end'])
            goal = {key: segment[key] for key in ('parent_goal', 'active_skills_semantic_json')}
            reply = await call(dict(kind='native_low_chunk', observation=wire, installed_subgoal=goal, execute_steps=16))
            actions = np.asarray(reply['actions'])
            fix = reply['inference_alignment_admission']
            if (actions.shape != (16, 23) or actions.dtype != np.float32 or not np.isfinite(actions).all()
                    or reply['history_admission'] != history_admission_to_wire(admission)
                    or reply['low_component_identity']['checkpoint_sha256'] != CHECKPOINTS['A4'][1]
                    or fix['padded_dimensions'] != [7, 8, 17, 18] or fix['real_control_dimensions'] != 23
                    or fix['action_history_steps'] != 0 or fix['action_execution_start_index'] != 0
                    or fix['image_history_frames'] != 6 or fix['expert_actions_used']):
                raise ValueError('Actual low service clock/mask/action contract failed')
            ingress.commit(admission)
            rows.append(dict(frame=frame, source_observation_sha256=sha(path), history=reply['history_admission'], alignment=fix))
    write_new(OUTPUT / 'wire_probe.json', dict(passed=True, model_calls=7, physics_controls=0,
        rows=rows, checkpoint_sha256=CHECKPOINTS['A4'][1], probe_only_train_observations=True,
        probe_history_and_rng_reset_by_each_episode=True, training_admissible=False))


def summarize(rows):
    completed = [r for r in rows if r.get('returncode') == 0 and r.get('status') == 'complete']
    successes = sum(r['task_success'] is True for r in completed)
    return dict(completed_episodes=len(completed), successes=successes,
        success_rate=successes / len(completed) if completed else None,
        all_predeclared_episodes_complete=len(completed) == 3, planned_episodes=3)


def supervise():
    manifest = validate_manifest(read(OUTPUT / 'manifest.json'))
    check_resources()
    children = []
    rows = []
    status = 'failed'
    try:
        low_command = [str(PYTHON), str(HERE / 'serve_low_fm_full.py'), '--run', str(LOW_RUN),
            '--checkpoint', manifest['low_checkpoint_path'], '--checkpoint-step', '2500',
            '--checkpoint-sha', CHECKPOINTS['A4'][1], '--runtime', str(RUNTIME),
            '--config', str(COMPOSITION / 'serving_config.yaml'), '--adapter', str(ADAPTER),
            '--output-dir', str(OUTPUT / 'low'), '--port', str(LOW_PORT), '--serving-source', str(SOURCE),
            '--snapshot', str(COMPOSITION / 'native_snapshot.json'), '--source-receipt', str(COMPOSITION / 'source_receipt.json'),
            '--inference-alignment-receipt', str(RUNTIME / 'inference_alignment_receipt.json')]
        low = start_child('low', low_command, model_env(0))
        children.append(('low', low))
        high_command = [str(PYTHON), str(RUNTIME / 'serve_formal_b_native.py'), '--run', str(HIGH_RUN),
            '--checkpoint', manifest['high_checkpoint_path'], '--source', str(WORK / 'b_parent_format_serving_source_v2'),
            '--phase-lineage', manifest['high_phase_lineage_path'], '--output', str(OUTPUT / 'high'), '--port', str(HIGH_PORT)]
        high = start_child('high', high_command, model_env(2))
        children.append(('high', high))
        asyncio.run(readiness(high, low))
        sim_env = model_env(1)
        sim_env.pop('CUDA_VISIBLE_DEVICES', None)
        sim_env.update(OMNI_KIT_ACCEPT_EULA='1', OMNIGIBSON_HEADLESS='1', OMNIGIBSON_NO_OMNI_LOGS='1',
            OMNIGIBSON_GPU_ID='1', BEHAVIOR_ACTION_STEPS='1',
            OMNIGIBSON_APPDATA_PATH=str(WORK / 'kit_c1_gpu1_appdata_v1'),
            LD_LIBRARY_PATH='/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/pymeshlab/lib')
        for index, episode in enumerate(manifest['episodes']):
            instance = episode['instance_id']
            output = OUTPUT / f'instance_{instance}'
            command = [BEHPY, str(HERE / 'run_radio_full.py'), '--manifest', str(OUTPUT / 'manifest.json'),
                '--episode-index', str(index), '--runtime', str(RUNTIME), '--adapter', str(ADAPTER),
                '--source', str(SOURCE), '--output', str(output), '--gpu', '1',
                '--high-uri', f'ws://127.0.0.1:{HIGH_PORT}', '--low-uri', f'ws://127.0.0.1:{LOW_PORT}']
            print(json.dumps(dict(status='starting', instance_id=instance)), flush=True)
            simulation = start_child(f'instance_{instance}', command, sim_env, cwd=WORK)
            children.append((f'instance_{instance}', simulation))
            returncode = simulation.wait()
            result_path = output / 'result.json'
            result = read(result_path) if result_path.is_file() else {}
            row = dict(instance_id=instance, returncode=returncode, status=result.get('status'),
                task_success=result.get('task_success'), consumed_actions=result.get('consumed_actions'),
                elapsed_seconds=result.get('elapsed_seconds'), termination_reason=result.get('termination_reason'),
                result=str(result_path), result_sha256=sha(result_path) if result_path.is_file() else None)
            rows.append(row)
            write_new(OUTPUT / f'completed_{instance}.json', row)
            print(json.dumps(row), flush=True)
            if returncode != 0 or result.get('status') != 'complete':
                raise RuntimeError('Incomplete episode or infrastructure fault; no automatic retry or silent denominator change')
            if (result['teacher_prefix_actions'] != 0 or result['oracle_subgoals_used']
                    or result['low_identity']['checkpoint_sha256'] != CHECKPOINTS['A4'][1]
                    or result['high_identity']['checkpoint_sha256'] != HIGH_SHA
                    or result['consumed_actions'] > episode['max_steps']):
                raise ValueError('Actual episode violates frozen autonomous evaluation identity/budget')
        status = 'complete'
    except BaseException:
        write_new(OUTPUT / 'failure.json', dict(status='stopped_for_inspection', traceback=traceback.format_exc()))
        raise
    finally:
        exits = []
        for label, child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            exits.append(dict(label=label, pid=child.pid, returncode=child.returncode))
        write_new(OUTPUT / 'summary.json', dict(status=status, episodes=rows, **summarize(rows),
            manifest_sha256=sha(OUTPUT / 'manifest.json'), finished_unix=time.time(),
            private_child_exits=exits, other_jobs_stopped=False, training_admissible=False))


def start():
    if OUTPUT.exists():
        raise FileExistsError('Cohort already exists; inspect it, never overwrite or automatically retry')
    if subprocess.check_output(['git', '-C', str(HERE.parents[1]), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Use a clean pinned Git worktree')
    lock = (WORK / 'a3_aligned_simulator_gpu1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_resources()
    prep = read(WORK / 'a3_aligned_runtimes_v2_preparation.json')
    if (prep['status'] != 'complete' or prep['roots']['full']['path'] != str(RUNTIME)
            or prep['roots']['full']['files'] != {p.name: sha(p) for p in RUNTIME.glob('*.py')}):
        raise ValueError('Validated runtime was altered')
    commit = subprocess.check_output(['git', '-C', str(HERE.parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = build_manifest(commit)
    OUTPUT.mkdir()
    write_new(OUTPUT / 'manifest.json', manifest)
    command = [str(PYTHON), str(HERE / 'a4_radio_full_campaign.py'), 'supervise']
    with (OUTPUT / 'supervisor.log').open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(lock.fileno(),))
    write_new(OUTPUT / 'launch.json', dict(pid=child.pid, command=command, started_unix=time.time(),
        code_commit=commit, manifest_sha256=sha(OUTPUT / 'manifest.json'), max_policy_controls=9672,
        no_automatic_retry=True, other_jobs_stopped=False))
    print(json.dumps(dict(pid=child.pid, output=str(OUTPUT))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('start', 'supervise'))
    if parser.parse_args().operation == 'start':
        start()
    else:
        supervise()
