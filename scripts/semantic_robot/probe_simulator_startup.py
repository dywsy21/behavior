"""H52b: one empty Kit startup with explicit renderer GPU isolation settings.

The existing H44 launcher and installed simulator sources are not modified.
Texture streaming limits are not a hard total-VRAM cap; the independent parent
also monitors all compute AND graphics processes and preserves GPU headroom.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
PYTHON = Path('/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python')
ISAAC = Path('/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/isaacsim')
EXPERIENCE = ISAAC / 'apps/omnigibson_5_1_0.kit'
OG_KIT = Path('/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson/omnigibson/omnigibson_5_1_0.kit')
APP_SOURCE = ISAAC / 'exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py'
DEPENDENCIES = {
    EXPERIENCE: '1aec3eda2c9841a060070c16305ea90c72c92a56cf0c973084b88f61b23f140d',
    OG_KIT: '1aec3eda2c9841a060070c16305ea90c72c92a56cf0c973084b88f61b23f140d',
    APP_SOURCE: '7cbaa6f00e935a6f14bf1c28ec0db089fd924e931f3b0deee07a822f9b7d0090',
}
OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260924/h52b_explicit_gpu_v1')
RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h52b_explicit_gpu_v1')
GPU_UUIDS = (
    'GPU-c0299c97-edd5-82e6-bec9-876e775ad9d1',
    'GPU-c4369c36-5c93-9920-dc4d-a3ec3594d51f',
    'GPU-3e4fda8c-536e-5899-e877-b8be97032fe0',
    'GPU-c67cdb9d-ec23-7ca0-dce9-14a61c24c46b',
)
TRAINING = dict(zip(GPU_UUIDS, (3294346, 3294347, 3294348, 3294349)))
MAIN_GPU = GPU_UUIDS[3]
WALL_SECONDS = 300
AUXILIARY_MIB = 512
SETTINGS = {
    '/rtx-transient/resourcemanager/enableTextureStreaming': True,
    '/rtx-transient/resourcemanager/texturestreaming/memoryBudget': 0.01,
    '/rtx-transient/resourcemanager/texturestreaming/streamingBudgetMB': 16,
}
GPU_SETTINGS = {'/renderer/activeGpu': 3, '/physics/cudaDevice': 3,
                '/renderer/multiGpu/enabled': False, '/renderer/multiGpu/autoEnable': False,
                '/renderer/multiGpu/maxGpuCount': 1}
RUNTIME_SETTINGS = {**SETTINGS, **GPU_SETTINGS}
ROUTES = {'CUDA_CACHE_PATH': 'cuda', 'TORCH_HOME': 'torch', 'TRITON_CACHE_DIR': 'triton',
          'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'XDG_CACHE_HOME': 'xdg', 'TMPDIR': 'tmp',
          'TMP': 'tmp', 'TEMP': 'tmp', '__GL_SHADER_DISK_CACHE_PATH': 'gl',
          'OMNIGIBSON_APPDATA_PATH': 'omnigibson', 'HF_HOME': 'hf'}
FIXED_ENV = {'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
             'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2', 'OMNI_KIT_ACCEPT_EULA': 'YES',
             'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}


def mib(text):
    if not isinstance(text, str) or not re.fullmatch(r'\d+ MiB', text.strip()):
        raise ValueError('Exact finite nvidia-smi MiB value required')
    return int(text.split()[0])


def parse_snapshot(xml):
    root = ET.fromstring(xml)
    rows = {}
    for node in root.findall('gpu'):
        uuid = node.findtext('uuid')
        if uuid in rows:
            raise ValueError('Duplicate GPU identity')
        processes = []
        for process in node.findall('processes/process_info'):
            pid = int(process.findtext('pid', '-1'))
            if pid <= 0:
                raise ValueError('Invalid GPU process identity')
            processes.append({'pid': pid, 'type': process.findtext('type'),
                              'used_mib': mib(process.findtext('used_memory'))})
        rows[uuid] = {'used_mib': mib(node.findtext('fb_memory_usage/used')),
                      'free_mib': mib(node.findtext('fb_memory_usage/free')),
                      'processes': processes}
    if set(rows) != set(GPU_UUIDS):
        raise ValueError('Exact four registered GPUs required')
    return rows


def snapshot():
    return parse_snapshot(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=5))


def check_resources(current, baseline=None, own_pid=None):
    if set(current) != set(TRAINING) or (baseline is not None and set(baseline) != set(TRAINING)):
        raise ValueError('Unknown GPU set')
    for uuid, row in current.items():
        expected = {TRAINING[uuid]}
        observed = {p['pid'] for p in row['processes']}
        if not expected <= observed or observed - expected - ({own_pid} if own_pid else set()):
            raise RuntimeError('Training identity changed or unknown compute/graphics process')
        if row['free_mib'] < (7168 if baseline is None else 3072):
            raise RuntimeError('Shared GPU reserve would be violated')
        if baseline is not None:
            cap = 4096 if uuid == MAIN_GPU else AUXILIARY_MIB
            if row['used_mib'] - baseline[uuid]['used_mib'] > cap:
                raise RuntimeError('Incremental shared GPU memory cap exceeded')
            if any(p['pid'] == own_pid and p['used_mib'] > cap for p in row['processes']):
                raise RuntimeError('Owned GPU process memory cap exceeded')


def identity():
    if Path(sys.executable).resolve() != PYTHON.resolve() or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Exact simulator Python and unremapped physical GPU IDs required')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean immutable source required')
    if importlib.metadata.version('isaacsim') != '5.1.0.0':
        raise ValueError('Frozen Isaac Sim version required')
    for path, digest in DEPENDENCIES.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Installed simulator/experience dependency changed: ' + str(path))
    return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()


def environment():
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('PYTHONPATH', None)
    env.update({name: str(RUNTIME / folder) for name, folder in ROUTES.items()})
    env.update(FIXED_ENV)
    native = str(PYTHON.parent.parent / 'lib/python3.11/site-packages/pymeshlab/lib')
    env['LD_LIBRARY_PATH'] = ':'.join([native, *[p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                                               if p and p != native]])
    return env, sorted({Path(env[key]) for key in ROUTES} | {RUNTIME / x for x in ('portable', 'cache', 'data')})


def claim_stage(mode, commit):
    """Single-use launch reservation; internal CLI modes cannot launch alone."""
    if mode not in ('supervisor', 'worker'):
        raise ValueError('Unknown launch stage')
    token = os.environ.get('H52_LAUNCH_TOKEN', '')
    if not re.fullmatch(r'[0-9a-f]{48}', token):
        raise ValueError('Missing launch reservation token')
    token_sha = hashlib.sha256(token.encode()).hexdigest()
    launch = json.loads((OUTPUT / 'launch.json').read_text())
    if (launch.get('token_sha256') != token_sha or launch.get('source_commit') != commit or
            launch.get('output') != str(OUTPUT) or launch.get('runtime') != str(RUNTIME)):
        raise ValueError('Launch reservation/source/path mismatch')
    expected, dirs = environment()
    for key in (*ROUTES, *FIXED_ENV, 'LD_LIBRARY_PATH'):
        if os.environ.get(key) != expected[key]:
            raise ValueError('Private runtime environment mismatch: ' + key)
    if os.environ.get('PYTHONPATH') or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Inherited source/GPU remapping forbidden')
    if any(not p.is_dir() or p.resolve() != p for p in (OUTPUT, RUNTIME, *dirs)):
        raise ValueError('Owned runtime/output directories must exist without redirection')
    if mode == 'worker':
        parent = json.loads((OUTPUT / 'supervisor.claim.json').read_text())
        if (parent.get('pid') != os.getppid() or parent.get('source_commit') != commit or
                parent.get('token_sha256') != token_sha):
            raise ValueError('Worker not spawned by the reserved supervisor')
        expected_args = [str(PYTHON), str(Path(__file__).resolve()), '--supervise']
        actual_args = Path('/proc', str(os.getppid()), 'cmdline').read_bytes().split(b'\0')
        if actual_args != [os.fsencode(a) for a in expected_args] + [b'']:
            raise ValueError('Unexpected supervisor command')
    claim = {'pid': os.getpid(), 'source_commit': commit, 'token_sha256': token_sha}
    with (OUTPUT / (mode + '.claim.json')).open('x') as file:
        json.dump(claim, file)
    return claim


def app_configuration():
    extra = [f'--{key}={str(value).lower()}' for key, value in SETTINGS.items()]
    extra += ['--/app/tokens/omni_global_cache=' + str(RUNTIME / 'cache'),
              '--/app/tokens/omni_global_data=' + str(RUNTIME / 'data'),
              '--/renderer/multiGpu/autoEnable=false', '--/app/extensions/registryEnabled=false',
              '--/log/level=info', '--/log/file=' + str(OUTPUT / 'kit.log'),
              '--/log/fileLogLevel=info', '--/log/outputStreamLevel=warning']
    return {'headless': True, 'multi_gpu': False, 'active_gpu': 3, 'physics_gpu': 3,
            'max_gpu_count': 1,
            'width': 320, 'height': 320, 'limit_cpu_threads': 4,
            'disable_viewport_updates': True, 'enable_crashreporter': False,
            'extra_args': extra}


def construct_app(factory):
    # SimulationApp forwards unknown sys.argv to Kit, and consults argv (not
    # extra_args) when deciding whether to append its default --portable flag.
    # Do not leak this script's --worker flag or inherit an alternative root.
    previous = sys.argv
    try:
        sys.argv = [previous[0], '--portable-root', str(RUNTIME / 'portable')]
        return factory(app_configuration(), experience=str(EXPERIENCE))
    finally:
        sys.argv = previous


def validate_settings(actual):
    if set(actual) != set(RUNTIME_SETTINGS):
        raise ValueError('Exact streaming setting keys required')
    for key, expected in RUNTIME_SETTINGS.items():
        value = actual[key]
        # Carb may round settings to float32. This is serialization precision,
        # not permission to change the registered one-percent budget.
        if type(value) is not type(expected) or ((not math.isfinite(value) or abs(value - expected) > 1e-8)
                if type(expected) is float else value != expected):
            raise ValueError('Runtime streaming settings differ from preregistration')


def write(name, data):
    # Parent/child may read receipts concurrently; never expose partial JSON.
    pending = OUTPUT / (name + '.tmp')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False))
    pending.replace(OUTPUT / name)


def worker():
    # Undo the parent's short Popen ownership mask before any GPU import.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    commit = identity()
    claim_stage('worker', commit)
    check_resources(snapshot())
    cores = {72, 73, 74, 75}
    if not cores <= os.sched_getaffinity(0):
        raise ValueError('Registered CPU affinity unavailable')
    os.sched_setaffinity(0, cores)
    record = {'source_commit': commit, 'pid': os.getpid(), 'phase': 'before_import',
              'task_loads': 0, 'simulator_resets': 0, 'robot_controls': 0, 'model_calls': 0,
              'training_steps': 0, 'app_updates': 0, 'success_rate': None,
              'app_config': app_configuration(), 'expected_settings': RUNTIME_SETTINGS,
              'dependencies': {str(k): v for k, v in DEPENDENCIES.items()}}
    write('worker.json', record)
    started = time.monotonic()
    app = None
    try:
        from isaacsim import SimulationApp
        actual_source = Path(inspect.getfile(SimulationApp)).resolve()
        if actual_source != APP_SOURCE.resolve():
            raise ValueError('Imported SimulationApp does not match frozen source')
        record['actual_app_source'] = str(actual_source)
        record['phase'] = 'constructing_app'; write('worker.json', record)
        app = construct_app(SimulationApp)
        import carb.settings
        settings = carb.settings.get_settings()
        actual = {key: settings.get(key) for key in RUNTIME_SETTINGS}
        validate_settings(actual)
        record.update(phase='app_ready', actual_settings=actual, startup_seconds=time.monotonic()-started)
        write('worker.json', record)
        for _ in range(8):
            if not app.is_running():
                raise RuntimeError('Kit stopped before bounded empty updates')
            app.update()
            record['app_updates'] += 1
            write('worker.json', record)
        record.update(phase='updates_complete', seconds=time.monotonic()-started)
        write('worker.json', record)
    except BaseException as error:
        record.update(phase='failed', error=repr(error), seconds=time.monotonic()-started)
        write('worker.json', record)
        raise
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as error:
                record.update(phase='close_failed', close_error=repr(error), seconds=time.monotonic()-started)
                write('worker.json', record)
                raise


def stop_owned(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(timeout=15)


def supervise():
    commit = identity()
    claim_stage('supervisor', commit)
    before = snapshot(); check_resources(before)
    env, _ = environment()
    started = time.monotonic()
    child, failure = None, None
    receipt = {'status': 'starting', 'supervisor_pid': os.getpid(), 'worker_pid': None,
               'baseline': before, 'samples': [], 'wall_limit_seconds': WALL_SECONDS}
    def interrupted(signum, frame):
        raise InterruptedError('H52 supervisor signal ' + str(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        # Defer these signals until Popen has returned and child ownership is
        # recorded, then deliver any pending signal inside this try/finally.
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(previous))
        try:
            with (OUTPUT / 'worker.log').open('x') as log:
                child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker'],
                    env=env, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            receipt.update(status='running', worker_pid=child.pid)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        write('supervisor.json', receipt)
        while child.poll() is None:
            if time.monotonic() - started >= WALL_SECONDS:
                raise TimeoutError('H52 empty-startup wall budget exhausted')
            current = snapshot()
            receipt['samples'].append({'elapsed': time.monotonic()-started, 'gpus': current})
            write('supervisor.json', receipt)
            check_resources(current, before, child.pid)
            time.sleep(0.25)
        if child.returncode != 0:
            raise RuntimeError('H52 worker failed with exit code ' + str(child.returncode))
        record = json.loads((OUTPUT / 'worker.json').read_text())
        if record.get('phase') != 'updates_complete' or record.get('app_updates') != 8:
            raise RuntimeError('H52 worker did not complete the registered empty updates')
        receipt.update(status='completed', exit_code=child.returncode)
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error))
    finally:
        try:
            if child is not None:
                stop_owned(child)
        except BaseException as error:
            failure = failure or error
            receipt.update(status='failed', cleanup_error=repr(error))
        receipt.update(seconds=time.monotonic()-started, exit_code=None if child is None else child.returncode)
        try:
            receipt['after_exit'] = snapshot()
            check_resources(receipt['after_exit'], before)
        except BaseException as error:
            failure = failure or error
            receipt.update(status='failed', after_exit_error=repr(error))
        try:
            write('supervisor.json', receipt)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if failure is not None:
        raise failure


def launch():
    commit = identity()
    before = snapshot(); check_resources(before)
    if RUNTIME.exists():
        raise FileExistsError('Never reuse an H52 runtime')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    env, dirs = environment()
    for path in dirs:
        path.mkdir(parents=True, exist_ok=False)
    env['H52_LAUNCH_TOKEN'] = secrets.token_hex(24)
    receipt = {'status': 'reserved', 'source_commit': commit, 'utc': datetime.now(timezone.utc).isoformat(),
               'token_sha256': hashlib.sha256(env['H52_LAUNCH_TOKEN'].encode()).hexdigest(),
               'output': str(OUTPUT), 'runtime': str(RUNTIME), 'gpu_before': before,
               'budget': {'seconds': WALL_SECONDS, 'app_updates': 8, 'task_loads': 0,
                          'simulator_resets': 0, 'robot_controls': 0, 'model_calls': 0,
                          'main_gpu_mib': 4096, 'auxiliary_gpu_mib': AUXILIARY_MIB,
                          'preflight_free_mib': 7168, 'runtime_free_mib': 3072},
               'app_config': app_configuration()}
    write('launch.json', receipt)
    with (OUTPUT / 'supervisor.log').open('x') as log:
        child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--supervise'],
            cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    receipt.update(status='supervisor_started', supervisor_pid=child.pid)
    write('launch.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    for mode in ('launch', 'supervise', 'worker'):
        group.add_argument('--' + mode, action='store_true')
    args = p.parse_args()
    (launch if args.launch else supervise if args.supervise else worker)()
