"""Foreground supervisor for one bounded H85 action-data training benchmark.

Wait for teammates to finish naturally. Never launch on an occupied GPU2;
never signal any process except the child session created by this invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

from prepare_visual_review import atomic_json, sha
from train_visual_presence import (GPU_UUID, clean_commit, model_identity, environment_identity,
                                  model_pythonpath)
from trajectory_training import CONFIG, DATA_SHA, QA_SHA, ROOT, checked_data, capacity_report

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/semantic_robot'))
from launch_h69 import belongs_to_session, stop_owned_group
from probe_simulator_startup import snapshot, GPU_UUIDS

PYTHON = Path('/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python')
WAITING_TRAINING_PIDS = (3641677, 3641678, 3641679, 3641680)


def check_resources(current, baseline=None, sid=None, *, released=False):
    if set(current) != set(GPU_UUIDS): raise ValueError('Registered GPU identities changed')
    for uuid, row in current.items():
        owned = [p for p in row['processes'] if belongs_to_session(p['pid'], sid)]
        if owned and (released or uuid != GPU_UUID): raise RuntimeError('Own session unreleased or touched another GPU')
        if uuid != GPU_UUID or released: continue  # Teammates' other GPUs are not ours to manage.
        if baseline is None:
            if row['processes'] or row['free_mib'] < 65536: raise RuntimeError('GPU2 must be idle with at least64GiB free')
        else:
            if any(not belongs_to_session(p['pid'], sid) for p in row['processes']):
                raise RuntimeError('New external GPU2 process; stop only our benchmark')
            if (row['used_mib'] - baseline[uuid]['used_mib'] > CONFIG['max_gpu_mib']
                    or sum(p['used_mib'] for p in owned) > CONFIG['max_gpu_mib']
                    or row['free_mib'] < 32768):
                raise RuntimeError('Own allocation cap or GPU2 reserve violated')


def verify_result(folder, code, launch):
    result = json.loads((folder / 'result.json').read_text())
    if ((folder / 'failure.json').exists() or result.get('status') != 'TRAINING_PIPELINE_BENCHMARKED'
            or result.get('code_commit') != code or result.get('config') != CONFIG
            or result.get('dataset_sha256') != DATA_SHA or result.get('qa_sha256') != QA_SHA
            or result.get('optimizer_updates') != CONFIG['updates'] or result.get('changed_lora_tensors', 0) <= 0
            or result.get('unique_train_windows') != CONFIG['updates'] * CONFIG['effective_batch']
            or result.get('checkpoint_writes') != 0 or result.get('training_eligible') is not False
            or result.get('three_hour_training_performed') is not False
            or result.get('policy_success_rate_evaluated') is not False):
        raise ValueError('Missing, incomplete, or wrong benchmark result')
    for name in ('sampling', 'steps', 'identity', 'loss_gate'):
        extension = '.jsonl' if name == 'steps' else '.json'
        if sha(folder / (name + extension)) != result[name + '_sha256']: raise ValueError('Benchmark receipt changed')
    for name, maximum in (('wall_seconds', CONFIG['wall_seconds']), ('cuda_peak_reserved_mib', CONFIG['max_gpu_mib'])):
        value = result.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError('Benchmark budget accounting invalid')
    plan = json.loads((folder / 'sampling.json').read_text())
    identity = json.loads((folder / 'identity.json').read_text())
    if (identity['code_commit'] != code or identity['config'] != CONFIG
            or identity['model_identity'] != launch['model_identity'] or identity['environment'] != launch['environment']
            or identity['dataset_sha256'] != DATA_SHA or identity['qa_sha256'] != QA_SHA
            or identity['old_adapter_loaded'] is not False or identity['online_executor_changed'] is not False
            or identity['physical_gpu'] != 2): raise ValueError('Model/data/training identity mismatch')
    records = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
    selected = plan['selected']
    if [i for r in records for i in r['indices']] != [r['index'] for r in selected]:
        raise ValueError('Actual training sample order differs from plan')
    if result['train_windows'] != 212500 or plan['train_windows'] != 212500:
        raise ValueError('Wrong full TRAIN size')
    if result['capacity'] != capacity_report(records, result['train_windows']):
        raise ValueError('Capacity report cannot be reproduced from actual steps')
    for r in records:
        for key in ('loss', 'gradient_norm'):
            if type(r[key]) not in (int, float) or not math.isfinite(r[key]) or r[key] < 0:
                raise ValueError('Invalid training numeric receipt')
        if r['gradient_norm'] == 0: raise ValueError('No action gradient')
    loss = json.loads((folder / 'loss_gate.json').read_text())
    if (loss['indices'] != plan['gate_indices'] or loss['left_padding'] is not True
            or not math.isfinite(loss['native']) or not math.isfinite(loss['custom'])
            or not math.isclose(loss['native'], loss['custom'], rel_tol=1e-5, abs_tol=1e-4)):
        raise ValueError('Native/custom CE gate invalid')
    return result


def preflight():
    if Path(sys.executable) != PYTHON: raise ValueError('Registered VLM interpreter required')
    code = clean_commit(); environment = environment_identity(); checked_data()
    before = snapshot(); check_resources(before)
    if any(Path(f'/proc/{pid}').exists() for pid in WAITING_TRAINING_PIDS):
        raise RuntimeError('Previously verified teammate training still live; wait for natural exit')
    if ROOT.exists(): raise FileExistsError('Preserve previous benchmark; no automatic retry')
    return code, environment, before


def run():
    started = time.monotonic(); code, environment, before = preflight()
    identity = model_identity()
    check_resources(snapshot())  # Identity hashing is not a GPU reservation.
    if not set(range(48, 56)) <= os.sched_getaffinity(0): raise ValueError('Registered CPU set unavailable')
    os.sched_setaffinity(0, set(range(48, 56)))
    ROOT.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
        H85_BENCHMARK_TOKEN=secrets.token_hex(24), PYTHONPATH=model_pythonpath())
    for key, name in (('CUDA_CACHE_PATH', 'cuda'), ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')):
        folder = ROOT / 'cache' / name; folder.mkdir(parents=True); env[key] = str(folder)
    command = [str(PYTHON), str(REPO / 'scripts/vlm_sft/benchmark_trajectory_training.py')]
    launch = {'code_commit': code, 'config': CONFIG, 'dataset_sha256': DATA_SHA, 'qa_sha256': QA_SHA,
        'model_identity': identity, 'environment': environment, 'baseline': before, 'command': command,
        'utc': datetime.now(timezone.utc).isoformat(), 'supervisor_pid': os.getpid(),
        'token_sha256': hashlib.sha256(env['H85_BENCHMARK_TOKEN'].encode()).hexdigest()}
    atomic_json(ROOT / 'launch.json', launch)
    receipt = {'status': 'starting', 'code_commit': code, 'supervisor_pid': os.getpid(), 'baseline': before}
    child = None; failure = None
    def interrupted(sig, frame): raise InterruptedError('Benchmark signal ' + str(sig))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set(previous))
        try:
            with (ROOT / 'worker.log').open('x') as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, restore_signals=True,
                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, set(previous)))
            receipt.update(status='running', worker_pid=child.pid)
        finally: signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        while child.poll() is None:
            if time.monotonic() - started >= CONFIG['wall_seconds']: raise TimeoutError('Entire benchmark wall budget')
            current = snapshot(); check_resources(current, before, child.pid)
            if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) >= CONFIG['max_bytes']:
                raise RuntimeError('Total artifact/cache cap')
            receipt.update(last_gpu_snapshot=current, elapsed_seconds=time.monotonic() - started)
            atomic_json(ROOT / 'supervisor.json', receipt); time.sleep(2)
        if child.returncode != 0: raise RuntimeError('Benchmark worker failed; inspect worker.log/failure.json')
        verify_result(ROOT / 'training', code, launch)
        if time.monotonic() - started > CONFIG['wall_seconds']: raise TimeoutError('Benchmark completed after budget')
        receipt.update(status='completed')
    except BaseException as exc:
        failure = exc; receipt.update(status='failed', error=repr(exc))
    finally:
        try:
            if child is not None: stop_owned_group(child)
            after = snapshot(); check_resources(after, before, None if child is None else child.pid, released=True)
            receipt['after_exit'] = after
        except BaseException as exc:
            failure = failure or exc; receipt.update(status='failed', cleanup_error=repr(exc))
        receipt.update(seconds=time.monotonic() - started, worker_exit_code=None if child is None else child.returncode)
        atomic_json(ROOT / 'supervisor.json', receipt)
        for sig, handler in previous.items(): signal.signal(sig, handler)
    if failure is not None: raise failure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight', action='store_true'); modes.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        code, environment, before = preflight()
        print(json.dumps({'status': 'AVAILABLE_NOT_LAUNCHED', 'code_commit': code, 'gpus': before}))
    else: run()
