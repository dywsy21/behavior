"""Frozen paths and authenticated local IPC; never imported by production."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO)]
DATA = Path('/mnt/sdc1/robodojo/datasets/behavior2026_g05_tasks_0_4')
RAW = Path('/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos')
RUN = Path('/mnt/sdc1/robodojo/outputs/g05/r1pro/behavior5_nomem_bs8_4gpu_20260923T135319Z')
PARENT = RUN/'checkpoints/step_50000.pt'
PARENT_SHA = 'c465044b025a487c42fddce17d45059b314a547e1b672f27cbdbbb7cfe2f6d48'
MODEL_PY = '/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python'
SIM_PY = '/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python'
PREVIOUS = Path('/mnt/nvme_tmp/robodojo_g05_rl_20260928/e1')
OUT = Path('/mnt/nvme_tmp/robodojo_g05_rl_20260928/e1_v2')
PREVIOUS_RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e1')
RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e1_v2')
ADAPTER = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1')
TEMPLATE = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/c1_windows_v2_matched/c1v2-matched-t0-train-e121-f448-grasp/window.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def commit():
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Clean frozen Git source required')
    return subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()


def save(path, obj, *, replace=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('w') as f: json.dump(obj,f,indent=2,allow_nan=False,ensure_ascii=False)
        os.replace(tmp,path)
    else:
        with path.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False,ensure_ascii=False)


def send(conn, obj):
    from g05.utils.websocket import packb
    conn.send_bytes(packb(obj))


def recv(conn, timeout=900):
    from g05.utils.websocket import unpackb
    if not conn.poll(timeout): raise TimeoutError('Simulator IPC timeout')
    result=unpackb(conn.recv_bytes(64<<20))
    if 'error' in result: raise RuntimeError(result['error'])
    return result


def sim_env(worker):
    import shutil
    sys.path.insert(0,str(REPO/'scripts/semantic_robot'))
    import probe_simulator_startup as base
    runtime=RUNTIME/f'worker_{worker}'
    runtime.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ); env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(base.FIXED_ENV)
    env.update({k:str(runtime/v) for k,v in base.ROUTES.items()})
    for sub in set(base.ROUTES.values())|{'portable','cache','data'}:
        (runtime/sub).mkdir(parents=True,exist_ok=True)
    # Only immutable-ish compiler caches from VERIFIED DEAD previous workers;
    # real copies, never hardlinks/shared writable runtimes or old Kit settings.
    for sub in ('cuda','torch','triton','inductor'):
        source=PREVIOUS_RUNTIME/f'worker_{worker}'/sub
        if source.is_dir(): shutil.copytree(source,runtime/sub,dirs_exist_ok=True)
    (runtime/'portable/data/documents/Kit/shared/screenshots').mkdir(parents=True,mode=0o700)
    env.update(OMNIGIBSON_GPU_ID=str(worker+2), OMNIGIBSON_HEADLESS='1',OMNIGIBSON_NO_OMNI_LOGS='0',
        BEHAVIOR_ACTION_STEPS='1',MEMLITE_SIM_TRACE_PATH='',PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        LD_LIBRARY_PATH='/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/pymeshlab/lib')
    return env
