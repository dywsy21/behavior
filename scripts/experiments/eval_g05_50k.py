"""User-requested 50k checkpoint, using the unchanged paired FM rollout path.

Reuses the completed 100-window evaluation. No ranking, training, or retries.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import socket
import subprocess
import time
import traceback

import eval_g05_100k as core
import eval_g05_best as previous

STEP = 50000
ROOT = Path('/mnt/nvme_tmp/robodojo_g05_50k_eval_20260928/v1')
RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/eval_g05_50k_20260928_v1')
PORT = 8933
SIM_TASKS = (0, 1)
WALL_SECONDS = 3600
ROWS_SHA = 'bbadff3fb036e95ae957a92da9639a66de5579abbc82bad070a392f874b56997'
LOAD_FIELDS = ('checkpoint', 'checkpoint_bytes', 'load_receipts', 'config_sha256',
               'stats_sha256', 'action_tokenizer_sha256', 'dtype')


def configure():
    core.OUT, core.RUNTIME, core.PORT = ROOT, RUNTIME, PORT
    core.MODEL_CALL_LIMIT = sum((core.LIMITS[t]+15)//16 for t in SIM_TASKS)


def evidence():
    result, _, receipt, digest = previous.read_candidate(STEP)
    if digest != ROWS_SHA:
        raise ValueError('The previously evaluated 50k windows changed')
    return result, receipt


def check_load(actual, expected):
    if any(actual[k] != expected[k] for k in LOAD_FIELDS):
        raise ValueError('50k deployment differs from the validated offline load')


def check_identity(identity):
    if (identity['checkpoint_step'] != STEP or identity['checkpoint'] != str(core.checkpoint(STEP))
            or identity['port'] != PORT or identity['action_source'] != 'fm'
            or identity['robot_action_dim'] != 23 or identity['obs_steps'] != 1
            or identity['predicted_steps'] != 32 or identity['execute_steps'] != 16
            or identity['predict_cot'] or identity['memlite']):
        raise ValueError('Wrong checkpoint or inference contract')


def spawn(children, mode, task=None):
    name = 'actor' if task is None else f'sim_{task}'
    command = [core.MODEL_PY if task is None else core.SIM_PY,
               str(Path(__file__).resolve()), mode]
    if task is not None:
        command += ['--task', str(task)]
    env = core.model_env(0) if task is None else core.simulation_env(task)
    with (ROOT/f'{name}.log').open('x') as log:
        process = subprocess.Popen(command, env=env, cwd=core.REPO, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children.append(process)
    core.save(ROOT/f'{name}.launch.json', dict(pid=process.pid, mode=mode, task=task,
        step=STEP, gpu=0 if task is None else 3))
    return process


def supervise():
    children = []
    start = time.monotonic()
    state = dict(status='running', source_commit=core.commit(), checkpoint_step=STEP,
        selection_reason='explicit_user_request', training_updates=0,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        evidence()
        actor = spawn(children, 'serve')
        while not (ROOT/'ready.json').exists():
            if actor.poll() is not None:
                raise RuntimeError('50k actor failed before readiness')
            if time.monotonic()-start > 600:
                raise TimeoutError('Actor preparation exceeded ten minutes')
            time.sleep(2)
        identity = json.loads((ROOT/'identity.json').read_text())
        check_identity(identity)
        for task in SIM_TASKS:
            remaining = WALL_SECONDS-(time.monotonic()-start)
            if remaining < 60 or actor.poll() is not None:
                raise RuntimeError('No remaining budget or actor stopped')
            process = spawn(children, 'simulate', task)
            code = process.wait(timeout=min(2400, remaining))
            result_path = ROOT/f'task_{task}/result.json'
            if code or not result_path.is_file():
                raise RuntimeError('50k simulator did not complete: '+str(task))
            result = json.loads(result_path.read_text())
            if (result['checkpoint_sha256'] != identity['checkpoint_sha256']
                    or result['source_commit'] != state['source_commit']
                    or result['status'] != 'completed'
                    or not 0 < result['controls'] <= core.LIMITS[task]):
                raise ValueError('Invalid simulator completion receipt')
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for process in reversed(children):
            core.stop_owned(process)
        state.update(seconds=time.monotonic()-start,
                     owned_returncodes={str(p.pid):p.returncode for p in children})
        core.save(ROOT/'supervisor.json', state)


def launch():
    revision = core.commit()
    result, _ = evidence()
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    if len(used) != 4 or any(int(x) > 512 for x in used):
        raise RuntimeError('Expected idle GPUs after the superseded evaluation')
    if ROOT.exists() or RUNTIME.exists():
        raise FileExistsError('Do not reuse an existing run or runtime')
    with socket.socket() as connection:
        connection.bind(('127.0.0.1', PORT))
    subprocess.run([core.MODEL_PY, '-c', 'from torchcodec.decoders import VideoDecoder'],
                   env=core.model_env(0), check=True, timeout=60)
    ROOT.mkdir(parents=True, exist_ok=False)
    core.save(ROOT/'launch.json', dict(owner='Codex/EVAL-G05-50K', source_commit=revision,
        checkpoint=str(core.checkpoint(STEP)), checkpoint_step=STEP,
        selection_reason='explicit_user_request', reused_offline=str(previous.folder(STEP)),
        reused_rows_sha256=ROWS_SHA, reused_losses=result['mean'], new_offline_windows=0,
        simulator_tasks=SIM_TASKS, control_limits=[core.LIMITS[t] for t in SIM_TASKS],
        max_model_calls=core.MODEL_CALL_LIMIT, wall_seconds=WALL_SECONDS,
        training_updates=0, automatic_retry=False))
    with (ROOT/'supervisor.log').open('x') as log:
        process = subprocess.Popen([core.MODEL_PY, str(Path(__file__).resolve()), 'supervise'],
            env=core.model_env(0), cwd=core.REPO, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    core.save(ROOT/'supervisor_launch.json', dict(pid=process.pid))
    print(json.dumps(dict(output=str(ROOT), pid=process.pid)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'supervise', 'serve', 'simulate'))
    parser.add_argument('--task', type=int, choices=SIM_TASKS)
    args = parser.parse_args()
    configure()
    if args.mode == 'serve':
        _, expected = evidence()
        cfg, policy, processor = core.load_model(STEP)
        check_load(json.loads((ROOT/f'load_{STEP}.json').read_text()), expected)
        asyncio.run(core.serve(cfg, policy, processor, step=STEP))
    elif args.mode == 'simulate':
        if args.task is None:
            raise ValueError('Task required')
        core.simulate(args.task)
    elif args.mode == 'supervise':
        supervise()
    else:
        launch()


if __name__ == '__main__':
    main()
