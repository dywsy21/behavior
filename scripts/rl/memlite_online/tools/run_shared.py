"""Bounded 8-rank launcher and persistent health/effect monitor, TRAIN only."""
# ruff: noqa: E402 -- pin dependency paths before importing runtime modules.
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from bootstrap import bootstrap
SOURCE = bootstrap()
from checkpoint_io import atomic_json
from gpu_admission import check_gpu_available
from probe_task_catalog import stop_owned
from shared_monitor import TrainingMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = args.job
    ownership = (job/'supervisor.lock').open('a')
    fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (job/'manifest.json').exists():
        raise ValueError('Do not relaunch an existing run; prepare a fresh shared resume')
    manifest = json.loads((job/'manifest.prepared.json').read_text())
    manifest.update(started=time.time(), absolute_deadline=time.time() + manifest['hours'] * 3600)
    atomic_json(job/'manifest.json', manifest)
    state = dict(status='loading', pid=os.getpid(), started=time.time(), source=manifest['source_commit'],
                 deadline=manifest['absolute_deadline'], policies={}, collectors={})
    processes, locks = [], []
    tracking = None
    monitor = TrainingMonitor(job)
    env_base = os.environ | dict(RL_REWARD_PROTOCOL='shared_terminal_q_v1',
        RL_TRAINING_CONFIG=str(SOURCE/'training.json'),
        RL_G05_SOURCE='/run/ti/rl_memlite_stage1_20261006/code/g05_sft_6af1ab9',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_DEBUG='WARN')
    for key in list(env_base):
        if key.startswith('RL_') and key not in {'RL_REWARD_PROTOCOL', 'RL_TRAINING_CONFIG', 'RL_G05_SOURCE'}:
            del env_base[key]

    def stop_requested(signum, frame):
        (job/'STOP_TRAINING').touch()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)

    def save_state():
        state['updated'] = time.time()
        atomic_json(job/'status.json', state)

    def start(command, env, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process

    try:
        for rank in range(8):
            lock = open(f'/run/ti/behavior_stage3_20260930/gpu_{rank}.lock', 'a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
            check_gpu_available(rank)
        try:
            import wandb
            tracking = wandb.init(project='behavior-memlite-rl', name=job.name, dir=str(job),
                config=dict(manifest['training'], world_size=8, source_commit=manifest['source_commit'],
                            metric_scope='on_policy_TRAIN', model_parent='stage1_high48045_low98414'),
                settings=wandb.Settings(init_timeout=30))
            state['wandb_url'] = tracking.url
        except Exception as error:
            state['wandb_error'] = repr(error)
        policies = []
        for start_rank in range(0, 8, 2):
            for rank in range(start_rank, start_rank + 2):
                run = job/'policies'/f'gpu_{rank}'
                run.mkdir(parents=True)
                process = start(['bash', str(SOURCE/'tools/run_policy_python.sh'),
                    str(SOURCE/'tools/serve_shared_rl.py'), '--job', str(job), '--run', str(run),
                    '--rank', str(rank)], env_base | {'CUDA_VISIBLE_DEVICES': str(rank)}, run/'policy.log')
                policies.append(process)
                state['policies'][str(rank)] = process.pid
            save_state()
            while not all((job/'policies'/f'gpu_{r}'/'model.loaded').exists()
                          for r in range(start_rank, start_rank + 2)):
                if any(p.poll() is not None for p in policies) or (job/'ABORT').exists():
                    raise RuntimeError('Policy model loading failed')
                if time.time() - state['started'] > 1500:
                    raise TimeoutError('Policy loading budget exceeded')
                time.sleep(2)
        while not all((job/'policies'/f'gpu_{r}'/'policy.ready').exists() for r in range(8)):
            if any(p.poll() is not None for p in policies) or time.time() - state['started'] > 1800:
                raise RuntimeError('Shared initialization failed/timed out')
            time.sleep(2)
        collectors = []
        for rank in range(8):
            process = start([sys.executable, str(SOURCE/'tools/run_shared_collector.py'),
                '--job', str(job), '--rank', str(rank)], env_base,
                job/'collector_logs'/f'gpu_{rank}.log')
            collectors.append(process)
            state['collectors'][str(rank)] = process.pid
        state['status'] = 'training'
        save_state()
        last_monitor = 0.
        while True:
            if (job/'ABORT').exists():
                raise RuntimeError('A rank raised global ABORT; all peers stop')
            for rank, process in enumerate(policies):
                if process.poll() is not None and not (job/'policies'/f'gpu_{rank}'/'save_ack.json').exists():
                    raise RuntimeError(f'Policy rank {rank} exited without shared save')
            for rank, process in enumerate(collectors):
                if process.poll() is not None and process.returncode != 0:
                    raise RuntimeError(f'Collector rank {rank} failed')
            if all(p.poll() is not None for p in collectors):
                state['status'] = 'completed'
                break
            if shutil.disk_usage(job).free < manifest['disk_reserve_gib'] * 2**30:
                (job/'STOP_TRAINING').touch()
                state['stop_reason'] = 'disk_reserve'
            if time.time() >= manifest['absolute_deadline'] - 300:
                (job/'STOP_TRAINING').touch()
                state['stop_reason'] = '24h_budget'
            if time.time() > manifest['absolute_deadline'] + 60:
                raise TimeoutError('Shared stop grace expired; preserving last committed checkpoint')
            if time.time() - last_monitor >= 60:
                summary = monitor.refresh()
                state['progress'] = {k: summary[k] for k in ('controls', 'completed_episodes',
                    'task_coverage', 'successes', 'macro_q_covered', 'macro_sr_covered', 'versions')}
                with (job/'monitor_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(time=time.time(), **state['progress'])) + '\n')
                save_state()
                if tracking is not None:
                    fields = {k: summary[k] for k in ('controls', 'completed_episodes', 'task_coverage',
                        'successes', 'macro_q_covered', 'macro_sr_covered', 'completed_only_sr',
                        'episodes_with_q_increase', 'controls_per_wall_second') if summary[k] is not None}
                    latest = summary['latest_rank0_update']
                    if latest:
                        fields.update({key: latest[key] for key in ('update', 'accepted_actor_lr',
                            'actor_grad_norm', 'critic_grad_norm', 'actor_max_delta', 'communication_seconds')})
                        fields.update({'ppo/' + k: v for k, v in latest['post_update'].items()})
                    for task, metrics in summary['per_task'].items():
                        fields.update({f'train_task/{task}/{key}': value for key, value in metrics.items()})
                    tracking.log(fields)
                last_monitor = time.time()
            time.sleep(3)
        monitor.refresh()
    except Exception as error:
        state.update(status='failed', error=repr(error))
        (job/'ABORT').touch()
        raise
    finally:
        for process in reversed(processes):
            stop_owned(process)
        for lock in locks:
            lock.close()
        state['finished'] = time.time()
        save_state()
        if tracking is not None:
            tracking.finish(exit_code=0 if state['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
