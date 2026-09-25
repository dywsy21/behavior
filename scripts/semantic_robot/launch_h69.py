"""One bounded H69 gate on the newly free GPU3; no automatic physical retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import time

import probe_scene_compatible_cameras as profile
import probe_scene_startup as scene

ROOT = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h69_native_gate_v1')
RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h69_native_gate_v1')
ASSET = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h64_finger_collision_asset_v1/result.json')
ASSET_SHA = '5f2cbf18b989eb22bc98665f48538c6cc306679d3be94ab4de46f2af7b4c6f04'
WALL_SECONDS = 2400
FLAGS = (
    'refine-grounding', 'visual-odometry', 'active-grasp-probe', 'grasp-motion',
    'robot-geometry-guards', 'approach-reorientation', 'approach-body-options',
    'approach-progress', 'held-object-inspection', 'persistent-grasp-tracks',
    'spatial-grasp-features', 'inspection-budget-aware', 'multicamera-inspection',
    'structured-planning', 'search-motion-recovery', 'odometry-self-exclusion',
    'gripper-completion-v1', 'odometry-match-refinement', 'workspace-posture',
    'near-contact-review', 'appearance-memory', 'carry-duration-v1',
    'joint-boundary-start-v1', 'approach-translation-preview', 'near-pose-gap',
    'finger-kinematics', 'press-cycle-v1',
)


def configure():
    profile.configure_profile()
    base = scene.supervisor
    base.OUTPUT, base.RUNTIME, base.ENTRYPOINT = ROOT, RUNTIME, Path(__file__).resolve()
    return base


def belongs_to_session(pid, session_id):
    if session_id is None: return False
    if pid == session_id: return True
    try: return os.getsid(pid) == session_id
    except ProcessLookupError: return False


def check_resources(current, baseline=None, own_pid=None, *, require_released=False):
    base = scene.supervisor
    if set(current) != set(base.GPU_UUIDS) or (baseline is not None and set(baseline) != set(current)):
        raise ValueError('Unknown physical GPU set')
    for uuid, row in current.items():
        cap = 24576 if uuid == base.GPU_UUIDS[3] else 512
        if baseline is None:
            # GPUs0/1 belong to teammates. Record, never signal their existing
            # processes; only our allocated GPUs2/3 must start unoccupied.
            if uuid in base.GPU_UUIDS[2:] and row['processes']:
                raise RuntimeError('Allocated GPU already occupied')
            if row['free_mib'] < (32768 if uuid == base.GPU_UUIDS[3] else 8704):
                raise RuntimeError('Preflight GPU reserve violated')
            continue
        existing = {p['pid'] for p in baseline[uuid]['processes']}
        allocated = uuid in base.GPU_UUIDS[2:]
        owned = {p['pid'] for p in row['processes'] if belongs_to_session(p['pid'], own_pid)}
        if require_released and owned:
            raise RuntimeError('Owned GPU session has not been released')
        if allocated and any(p['pid'] not in existing | owned for p in row['processes']):
            raise RuntimeError('Unknown compute/graphics process; stop only our worker')
        # Team-owned0/1 may start/end jobs normally. Their allocations are not
        # ours; enforce our exact PID and shared headroom, not a frozen roster.
        if ((allocated and row['used_mib'] - baseline[uuid]['used_mib'] > cap) or
                sum(p['used_mib'] for p in row['processes'] if p['pid'] in owned) > cap):
            raise RuntimeError('Registered GPU memory cap exceeded')
        if row['free_mib'] < 8192:
            raise RuntimeError('Registered GPU reserve violated')


def command(base):
    args = [str(base.PYTHON), str(base.REPO/'scripts/semantic_robot/run_v2.py'),
            '--output', str(ROOT/'gate'), '--mode', 'gate', '--harness', 'grounded',
            '--task', '0', '--gpu', '3', '--prefix', '0', '--max-decisions', '24',
            '--max-controls', '1536', '--max-seconds', '1200',
            '--odometry-estimator', 'rgbd_joint', '--odometry-substep-controls', '6',
            '--native-profile', 'a100_full_v1', '--native-runtime', str(RUNTIME),
            '--press-finger-surfaces', str(ASSET)]
    return args + ['--' + flag for flag in FLAGS]


def identity(base):
    commit = base.identity()
    if hashlib.sha256(ASSET.read_bytes()).hexdigest() != ASSET_SHA:
        raise ValueError('Fixed finger asset changed')
    if shutil.disk_usage('/mnt/nvme_tmp').free < 80 * 1024**3:
        raise RuntimeError('NVMe disk reserve below 80GiB')
    return commit


def expected_args(base):
    expected = {'uri': 'http://127.0.0.1:8907', 'expected_revision': None,
                'gate_result': [], 'replay_prefix_spec': None,
                'contact_geometry': False, 'budget_profile': 'pilot', 'synchronous_io_v1':False}
    integer = {'task', 'gpu', 'prefix', 'max_decisions', 'max_controls', 'max_seconds',
               'odometry_substep_controls'}
    parts = iter(command(base)[2:])
    for flag in parts:
        key = flag.removeprefix('--').replace('-', '_')
        value = True if flag[2:] in FLAGS else next(parts)
        expected[key] = int(value) if key in integer else value
    return expected


def validate_manifest(manifest, base, commit, digest):
    exact = {'code_commit': commit, 'implementation_digest': digest, 'instance': 138,
             'task': 0, 'task_name': 'turning_on_radio', 'split': 'train', 'seed': 0,
             'training_updates': 0, 'model_identity': None, 'actor_scene_truth': False,
             'prefix_is_expert_not_agent': False, 'diagnostic_replay_requested': False,
             'native_profile': 'a100_full_v1', 'args': expected_args(base)}
    # Canonical JSON compares nested scalar types too (True is not integer1).
    for key, value in exact.items():
        if key not in manifest or json.dumps(manifest[key], sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError('Exact H69 manifest mismatch: ' + key)


def validate_result(result, digest):
    exact = {'status': 'complete', 'task': 0, 'prefix_controls': 0, 'diagnostic_replay_controls': 0,
             'gate_ok': True, 'gate_failures': [], 'native_profile': 'a100_full_v1',
             'implementation_digest': digest, 'finger_kinematics': True, 'press_cycle_v1': True,
             'press_finger_asset_sha256': ASSET_SHA, 'model_calls': 0,
             'contact_geometry': False, 'full_task_success_rate_claim': False}
    exact.update({flag.replace('-', '_'): True for flag in FLAGS if flag != 'structured-planning'})
    if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in exact.items()):
        raise ValueError('Gate did not pass its exact source/profile and original criteria')
    if len(result.get('decisions', [])) != 24 or not 0 < result.get('controls', 0) <= 1536:
        raise ValueError('Incomplete or over-budget physical gate')


def owned_groups(session_id):
    groups = set()  # Never signal an unobserved/recycled old numeric group ID.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        pid = int(entry.name)
        try:
            if os.getsid(pid) == session_id: groups.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
    return groups


def stop_owned_group(child):
    """Stop only the process group/session created by this Popen call.

    The leader may exit while native descendants still hold GPU memory.
    Reap our direct child and wait for its group, not just the leader.
    """
    pgid = child.pid  # start_new_session=True binds the owned PGID/SID.
    if child.poll() is None:
        try: actual_pgid = os.getpgid(child.pid)
        except ProcessLookupError: actual_pgid = pgid  # Leader exited; descendants may remain.
        if actual_pgid != pgid:
            raise RuntimeError('Owned child no longer has its isolated process group')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # Reserve five seconds for the final nvidia-smi snapshot and one for
        # receipts within the separately registered30s cleanup allowance.
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            for group in owned_groups(pgid):
                try: os.killpg(group, sig)
                except ProcessLookupError: pass
            child.poll()
            alive = False
            for group in owned_groups(pgid):
                try: os.killpg(group, 0); alive = True
                except ProcessLookupError: pass
            if not alive:
                child.wait(timeout=1)
                return
            time.sleep(.1)
    raise RuntimeError('Owned process group did not disappear within cleanup budget')


def supervise(base):
    commit = identity(base)
    base.claim_stage('supervisor', commit)
    before = base.snapshot(); check_resources(before)
    cores = {72, 73, 74, 75}
    if not cores <= os.sched_getaffinity(0):
        raise ValueError('Registered CPU cores unavailable')
    os.sched_setaffinity(0, cores)
    env, _ = base.environment()
    receipt = {'status': 'starting', 'source_commit': commit, 'supervisor_pid': os.getpid(),
               'worker_pid': None, 'baseline': before, 'samples': [], 'wall_seconds': WALL_SECONDS}
    started, child, failure = time.monotonic(), None, None
    def interrupted(signum, _frame): raise InterruptedError('H69 signal ' + str(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        oldmask = signal.pthread_sigmask(signal.SIG_BLOCK, set(previous))
        try:
            with (ROOT/'worker.log').open('x') as log:
                child = subprocess.Popen(command(base), cwd=base.REPO, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, restore_signals=True,
                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, set(previous)))
            receipt.update(status='running', worker_pid=child.pid)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)
        base.write('supervisor.json', receipt)
        while child.poll() is None:
            if time.monotonic() - started >= WALL_SECONDS:
                raise TimeoutError('H69 total wall budget, including native initialization')
            current = base.snapshot()
            receipt['samples'].append({'elapsed': time.monotonic()-started, 'gpus': current})
            base.write('supervisor.json', receipt)
            check_resources(current, before, child.pid)
            time.sleep(2)
        failure_path = ROOT/'gate/native_failure.json'
        if failure_path.exists():
            native = json.loads(failure_path.read_text())
            receipt['native_failure'] = native.get('native_failure')
            receipt['pathtracing_differences'] = native.get('pathtracing_differences')
            raise RuntimeError('Native session failed: ' + str(receipt['native_failure']))
        if child.returncode != 0:
            raise RuntimeError('Gate worker exit code ' + str(child.returncode))
        if time.monotonic() - started > WALL_SECONDS:
            raise TimeoutError('Worker exited after registered total wall budget')
        from run_v2 import implementation_digest
        digest = implementation_digest()
        validate_manifest(json.loads((ROOT/'gate/manifest.json').read_text()), base, commit, digest)
        result = json.loads((ROOT/'gate/result.json').read_text())
        validate_result(result, digest)
        if time.monotonic() - started > WALL_SECONDS:
            raise TimeoutError('Gate validation exceeded total wall budget')
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error; receipt.update(status='failed', error=repr(error))
    finally:
        cleanup_started = time.monotonic()
        try:
            if child is not None: stop_owned_group(child)
        except BaseException as error:
            failure = failure or error
            receipt.update(status='failed', cleanup_error=repr(error))
        receipt.update(seconds=time.monotonic()-started, exit_code=None if child is None else child.returncode)
        try:
            receipt['after_exit'] = base.snapshot()
            check_resources(receipt['after_exit'], before,
                            None if child is None else child.pid, require_released=True)
        except BaseException as error:
            failure = failure or error; receipt.update(status='failed', after_exit_error=repr(error))
        if time.monotonic() - cleanup_started > 30:
            failure = failure or TimeoutError('Cleanup exceeded registered30s allowance')
            receipt.update(status='failed', cleanup_budget_exceeded=True)
        base.write('supervisor.json', receipt)
        for sig, handler in previous.items(): signal.signal(sig, handler)
    if failure is not None: raise failure


def launch(base):
    commit = identity(base)
    before = base.snapshot(); check_resources(before)
    if ROOT.exists() or RUNTIME.exists():
        raise FileExistsError('Never resubmit an existing physical gate or runtime')
    ROOT.mkdir(parents=True)
    env, folders = base.environment()
    for path in folders: path.mkdir(parents=True, exist_ok=False)
    env['H52_LAUNCH_TOKEN'] = secrets.token_hex(24)
    receipt = {'status': 'reserved', 'source_commit': commit, 'utc': datetime.now(timezone.utc).isoformat(),
               'output': str(ROOT), 'runtime': str(RUNTIME), 'command': command(base),
               'token_sha256': hashlib.sha256(env['H52_LAUNCH_TOKEN'].encode()).hexdigest(),
               'budget': {'wall_seconds': WALL_SECONDS, 'cleanup_seconds': 30, 'primary_mib': 24576,
                          'auxiliary_mib': 512, 'runtime_reserve_mib': 8192,
                          'decisions': 24, 'controls': 1536, 'model_calls': 0, 'training_updates': 0},
               'gpu_before': before}
    base.write('launch.json', receipt)
    with (ROOT/'supervisor.log').open('x') as log:
        child = subprocess.Popen([str(base.PYTHON), str(base.ENTRYPOINT), '--supervise'],
            cwd=base.REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(status='supervisor_started', supervisor_pid=child.pid)
    base.write('launch.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch', action='store_true'); modes.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    base = configure()
    (launch if args.launch else supervise)(base)
