"""One registered 544-control simulator-only probe, with owned GPU watchdog."""
import argparse
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from common import REPO, OUT, RUNTIME, SIM_PY, commit, save, sha, send, recv, sim_env
from rtx_paths import ROOT, host_guard, cores
from g05.rl.protocol import check_reset


def spawn_simulators(children, listener, secret):
    # Workers validate their inherited allowed mask before narrowing it. Pinning
    # this controller first would accidentally restrict BOTH children to 16–17.
    required = cores(0) | cores(1) | {16, 17}
    if not required <= os.sched_getaffinity(0):
        raise ValueError('Registered simulator/controller CPU partition unavailable')
    for w in (0, 1):
        env = sim_env(w)
        env['BEHAVIOR_RL_IPC_KEY'] = secret.hex()
        with (OUT/f'sim_{w}.stdout.log').open('x') as log:
            children.append(subprocess.Popen([SIM_PY,str(REPO/'scripts/rl/sim_worker.py'),
                '--worker',str(w),'--port',str(listener.address[1])], cwd=REPO, env=env,
                stdout=log, stderr=subprocess.STDOUT))
    os.sched_setaffinity(0, {16, 17})


def snapshot():
    xml = subprocess.check_output(['nvidia-smi','-q','-x'], text=True, timeout=10)
    cards = ET.fromstring(xml).findall('gpu')
    if len(cards) != 1 or cards[0].findtext('product_name') != 'NVIDIA GeForce RTX 4090':
        raise ValueError('Registered single RTX4090 required')
    gpu = cards[0]
    return dict(uuid=gpu.findtext('uuid'), name=gpu.findtext('product_name'),
        free_mib=int(gpu.findtext('fb_memory_usage/free').split()[0]),
        used_mib=int(gpu.findtext('fb_memory_usage/used').split()[0]),
        processes=[int(p.findtext('pid')) for p in gpu.findall('processes/process_info')])


def controller():
    import numpy as np
    host_guard()
    manifest = json.loads((OUT/'manifest.json').read_text())
    if commit() != manifest['source_commit']: raise ValueError('Frozen probe source changed')
    started = time.monotonic(); controls = 0
    children = []; connections = {}; pending = {}; latest = {}
    secret = os.urandom(32)
    listener = Listener(('127.0.0.1',0), authkey=secret)
    listener._listener._socket.settimeout(1200)

    def status(phase, **extra):
        save(OUT/'status.json', dict(phase=phase, pid=os.getpid(), controls=controls,
             pending_controls=sum(pending.values()), seconds=time.monotonic()-started, **extra), replace=True)

    def guard():
        if time.monotonic()-started >= manifest['max_active_wall_seconds']:
            raise TimeoutError('Original registered RTX probe wall budget')

    def collect(w):
        nonlocal controls
        reply = recv(connections[w], 900)
        if reply['actual_controls'] != pending[w]: raise ValueError('Unexpected terminal or partial replay')
        controls += reply['actual_controls']; del pending[w]; latest[w] = reply
        if reply['terminal'] or reply['success']: raise ValueError('First 128 demo controls became terminal')

    def reset():
        for w in (0, 1): send(connections[w], dict(op='reset'))
        checks = []
        for w in (0, 1):
            latest[w] = recv(connections[w], 900)
            spec = manifest['workers'][w]
            ref = json.loads(Path(manifest['reference_resets'][str(spec['instance'])]).read_text())
            diff = check_reset(ref, latest[w]['reset_state'], manifest['reset_tolerance'])
            if latest[w]['episode_controls'] or latest[w]['terminal'] or latest[w]['success']:
                raise ValueError('Noninitial replay reset')
            checks.append(dict(worker=w, max_difference=diff, episode=latest[w]['episode']))
        save(OUT/f'reset_{latest[0]["episode"]:03d}.json', checks)

    def replay(count, phase, parallel):
        nonlocal controls
        progress = {0:0, 1:0}; began = time.monotonic()
        while min(progress.values()) < count:
            guard()
            for w in (0, 1):
                chunk = actions[w][progress[w]:min(count,progress[w]+16)]
                if not len(chunk): continue
                if controls + sum(pending.values()) + len(chunk) > manifest['max_controls']:
                    raise ValueError('Registered control budget exhausted')
                pending[w] = len(chunk)
                send(connections[w], dict(op='step', actions=chunk, phase=phase))
                if not parallel:
                    n = pending[w]; collect(w); progress[w] += n
            if parallel:
                for w in (0, 1):
                    if w in pending:
                        n = pending[w]; collect(w); progress[w] += n
            status(phase, progress=progress)
        return dict(controls=sum(progress.values()), seconds=time.monotonic()-began)

    try:
        status('initializing')
        actions = {r['worker']:np.load(r['actions'], allow_pickle=False) for r in manifest['workers']}
        spawn_simulators(children, listener, secret)
        save(OUT/'pids.json', dict(controller=os.getpid(), simulators=[p.pid for p in children]))
        for _ in (0, 1):
            conn = listener.accept(); hello = recv(conn, 30); w = hello['hello']
            if w not in (0, 1) or w in connections: raise ValueError('Wrong worker connection')
            connections[w] = conn
        images = []
        for w in (0, 1):
            latest[w] = recv(connections[w], 1200)
            spec = manifest['workers'][w]
            ref = json.loads(Path(manifest['reference_resets'][str(spec['instance'])]).read_text())
            check_reset(ref, latest[w]['reset_state'], manifest['reset_tolerance'])
            if latest[w]['episode_controls'] or latest[w]['success'] or latest[w]['terminal']:
                raise ValueError('Unexpected initial state')
            p = OUT/f'worker_{w}/reset_000.png'
            images.append(dict(path=str(p), sha256=sha(p), instance=spec['instance']))
        save(OUT/'image_gate.json', dict(images=images, original_reset_matched=True))
        status('awaiting_parent_image_review', images=images)
        gate_start = time.monotonic()
        release = OUT/'human_release.json'
        while not release.exists():
            guard()
            if time.monotonic()-gate_start > manifest['max_human_gate_seconds']:
                raise TimeoutError('Image review not completed inside registered 300-second window')
            time.sleep(1)
        signed = json.loads(release.read_text())
        if (signed.get('accepted') is not True or signed.get('reviewer') != 'parent_agent'
                or {(r['path'],r['sha256']) for r in signed['images']} != {(r['path'],r['sha256']) for r in images}
                or any(sha(r['path']) != r['sha256'] for r in images)):
            raise ValueError('Image gate signature mismatch')
        warm = replay(16, 'benchmark_warmup', True)
        reset(); serial = replay(128, 'benchmark_serial', False)
        reset(); parallel = replay(128, 'benchmark_parallel', True)
        if controls != 544 or pending: raise ValueError('Incomplete registered matched workload')
        save(OUT/'throughput.json', dict(warmup=warm, serial=serial, parallel=parallel,
            speedup=serial['seconds']/parallel['seconds'], includes_per_chunk_observation=True,
            excludes_initialization=True, excludes_model_inference_and_training=True,
            physical_gpu_count=1, resident_simulators=2, controls=controls,
            a100_reference=manifest['original_a100_reference'],
            claim_scope='whole-machine simulator throughput; not complete RL acceleration'))
        for w in (0, 1): send(connections[w], dict(op='close'))
        closed = [recv(connections[w], 30) for w in (0, 1)]
        if sum(r['total_controls'] for r in closed) != controls or not all(r.get('closed') for r in closed):
            raise ValueError('Physical shutdown ledger mismatch')
        for p in children:
            if p.wait(timeout=30): raise ValueError('Simulator did not exit cleanly')
        save(OUT/'closed.json', dict(receipts=closed, exits=[p.returncode for p in children], controls=controls))
        status('completed')
    except BaseException as error:
        status('failed', error=repr(error))
        save(OUT/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                    controls=controls, pending_controls=sum(pending.values())))
        raise
    finally:
        for conn in connections.values(): conn.close()
        listener.close()
        for p in children:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=10)


def supervise():
    host_guard()
    manifest = json.loads((OUT/'manifest.json').read_text()); source = commit()
    if source != manifest['source_commit']: raise ValueError('Frozen source mismatch')
    initial = snapshot()
    if initial['used_mib'] > 512: raise RuntimeError('GPU already occupied')
    start = time.monotonic(); child = None
    result = dict(source_commit=source, pid=os.getpid(), status='starting', initial_gpu=initial)
    try:
        with (OUT/'controller.stdout.log').open('x') as log:
            child = subprocess.Popen([SIM_PY,__file__,'--controller'], cwd=REPO, env=dict(os.environ,
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result.update(status='running', controller=child.pid)
        save(OUT/'supervisor.json', result, replace=True)
        peak = 0
        while child.poll() is None:
            current = snapshot(); peak = max(peak, current['used_mib'])
            if current['uuid'] != initial['uuid'] or current['free_mib'] < 2048:
                raise RuntimeError('GPU identity or 2GiB reserve violation')
            for pid in set(current['processes'])-set(initial['processes']):
                try: owned = os.getpgid(pid) == child.pid
                except ProcessLookupError: continue
                if not owned: raise RuntimeError('Another GPU workload appeared; stop only this probe')
            if time.monotonic()-start >= manifest['max_active_wall_seconds']:
                raise TimeoutError('Registered 3600-second GPU budget reached')
            time.sleep(2)
        status = json.loads((OUT/'status.json').read_text())
        result.update(status='completed' if child.returncode == 0 and status['phase']=='completed' else 'failed',
                      exit_code=child.returncode, peak_used_mib=peak)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        result.update(seconds=time.monotonic()-start, final_gpu=snapshot())
        save(OUT/'supervisor.json', result, replace=True)


def launch():
    host_guard()
    manifest = json.loads((OUT/'manifest.json').read_text())
    if manifest['source_commit'] != commit(): raise ValueError('Frozen source mismatch')
    save(OUT/'launch_claim.json', dict(source_commit=commit(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    with (OUT/'supervisor.stdout.log').open('x') as log:
        p = subprocess.Popen([SIM_PY,__file__,'--supervise'], cwd=REPO,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor=p.pid, run=str(OUT))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('launch','supervise','controller'): group.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    (launch if args.launch else supervise if args.supervise else controller)()
