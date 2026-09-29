"""Explicit opt-in paths for the single authorized RTX simulation speed probe."""
import os
from pathlib import Path
import socket

PROFILE = 'rtx4090_speed_v1'
ROOT = Path('/home/user/behavior_rl_speed_20260928')
REPO = Path(__file__).resolve().parents[2]
SDK = ROOT/'src/BEHAVIOR-1K'
ENV = ROOT/'envs/sim'
OUT = ROOT/'runs/sim_speed_v1'
RUNTIME = ROOT/'runtime/sim_speed_v1'
ADAPTER = REPO/'scripts/rl/rtx_legacy'
TEMPLATE = ROOT/'fixtures/window.json'
TASKS = ROOT/'fixtures/tasks.jsonl'


def host_guard():
    if (socket.gethostname() != 'teai-g1' or os.getuid() != 1000
            or os.environ.get('BEHAVIOR_RL_SIM_PROFILE') != PROFILE
            or os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('Explicit registered RTX host/user/profile and unremapped GPU required')


def cores(worker):
    if worker not in (0, 1): raise ValueError('Only two registered workers')
    return set(range(8*worker, 8*(worker+1)))


def validate_worker(spec, *, evaluation):
    if (evaluation or spec.get('evaluation_only') or spec['worker'] not in (0, 1)
            or spec['gpu'] != 0 or spec['split'] != 'train' or spec['seed'] != 0
            or spec['task'] != 'turning_on_radio' or spec['task_id'] != 0
            or spec['instance'] != (1, 138)[spec['worker']]
            or spec['episode'] != (0, 121)[spec['worker']]
            or Path(spec['actions']) != ROOT/'fixtures'/f'demo_{spec["episode"]}.npy'):
        raise ValueError('Outside fixed RTX TRAIN speed-probe scope')


def validate_phase(phase):
    if phase not in ('benchmark_warmup', 'benchmark_serial', 'benchmark_parallel'):
        raise ValueError('RTX speed probe cannot issue training or evaluation policy controls')


def environment(worker, pool=None):
    host_guard()
    if pool is not None: raise ValueError('RTX speed probe is not an E2 training/evaluation pool')
    import sys
    sys.path.insert(0, str(REPO/'scripts/semantic_robot'))
    import probe_simulator_startup as base
    runtime = RUNTIME/f'worker_{worker}'
    runtime.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(base.FIXED_ENV)
    env.update({k: str(runtime/v) for k, v in base.ROUTES.items()})
    for folder in set(base.ROUTES.values()) | {'portable', 'cache', 'data'}:
        (runtime/folder).mkdir(parents=True, exist_ok=True)
    (runtime/'portable/data/documents/Kit/shared/screenshots').mkdir(parents=True, mode=0o700)
    env.update(OMNIGIBSON_GPU_ID='0', OMNIGIBSON_HEADLESS='1', OMNIGIBSON_NO_OMNI_LOGS='0',
        OMNIGIBSON_DATA_PATH=str(ROOT/'datasets'), BEHAVIOR_ACTION_STEPS='1', MEMLITE_SIM_TRACE_PATH='',
        PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        LD_LIBRARY_PATH=str(ENV/'lib/python3.11/site-packages/pymeshlab/lib'))
    return env
