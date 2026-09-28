"""Isolated CPU-only simulator installation for the authorized RTX speed probe.

No simulator import, GPU job, dataset download, shared-env mutation or sudo.
The existing robo RL run is deliberately outside this process's scope.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

ROOT = Path('/home/user/behavior_rl_speed_20260928')
REPO = Path(__file__).resolve().parents[2]
SDK = ROOT / 'src/BEHAVIOR-1K'
ENV = ROOT / 'envs/sim'
OUT = ROOT / 'runs/bootstrap_v1'
SDK_COMMIT = '26f2c7ef7b9cf96bd0414f81e1e751e493762779'
WALL_SECONDS = 7200
RESERVE_BYTES = 200 * 1024**3


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    if socket.gethostname() != 'teai-g1' or os.getuid() != 1000:
        raise RuntimeError('This opt-in installation is only registered on teai-g1/user')
    if REPO != ROOT / 'src/behavior' or ENV.exists() or SDK.exists():
        raise RuntimeError('Expected new isolated paths; do not overwrite an environment')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain']).strip():
        raise RuntimeError('Frozen clean source required')
    if shutil.disk_usage(ROOT).free < RESERVE_BYTES + 100 * 1024**3:
        raise RuntimeError('Insufficient disk reserve before preparation')
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    allowed = sorted(os.sched_getaffinity(0))[:8]
    os.sched_setaffinity(0, allowed)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
               PYTHONUNBUFFERED='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_CACHE_DIR=str(ROOT / 'cache/pip'),
               OMNI_KIT_ACCEPT_EULA='YES', CUDA_VISIBLE_DEVICES='')
    base = dict(pid=os.getpid(), source_commit=source, sdk_commit=SDK_COMMIT,
                wall_limit_seconds=WALL_SECONDS, reserve_bytes=RESERVE_BYTES,
                cpu_affinity=allowed, gpu_launches=0, dataset_downloads=0,
                environment=str(ENV), root=str(ROOT))
    write(OUT / 'launch.json', base)

    def run(stage, args):
        write(OUT / 'status.json', dict(base, stage=stage, status='running',
                                       seconds=time.monotonic()-started))
        with (OUT / (stage+'.log')).open('x') as log:
            process = subprocess.Popen(args, cwd=REPO, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    if time.monotonic()-started >= WALL_SECONDS:
                        raise TimeoutError('Registered preparation time limit')
                    if shutil.disk_usage(ROOT).free < RESERVE_BYTES:
                        raise RuntimeError('Preparation disk reserve reached')
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError(f'{stage} failed, exit={process.returncode}; see stage log')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)

    try:
        run('sdk_init', ['git', 'init', str(SDK)])
        run('sdk_origin', ['git', '-C', str(SDK), 'remote', 'add', 'origin',
                           'https://github.com/StanfordVL/BEHAVIOR-1K.git'])
        run('sdk_fetch', ['git', '-C', str(SDK), 'fetch', '--depth=1', 'origin', SDK_COMMIT])
        run('sdk_checkout', ['git', '-C', str(SDK), 'checkout', '--detach', SDK_COMMIT])
        run('python_env', ['/home/user/miniconda3/bin/conda', 'create', '-y', '--prefix', str(ENV),
                          '--override-channels', '-c', 'conda-forge', 'python=3.11', 'pip',
                          'setuptools>=71,<81', 'wheel'])
        pip = [str(ENV/'bin/python'), '-m', 'pip', 'install']
        run('torch', [*pip, '--index-url', 'https://download.pytorch.org/whl/cu128',
                      'torch==2.7.0', 'torchvision==0.22.0', 'torchaudio==2.7.0', 'torchcodec==0.5'])
        run('numpy', [*pip, 'numpy==1.26.0'])
        run('bddl', [*pip, '-e', str(SDK/'bddl3')])
        run('omnigibson', [*pip, '--no-build-isolation', '-e', str(SDK/'OmniGibson')+'[eval]'])
        run('isaac', [*pip, 'isaacsim[all,extscache]==5.1.0',
                      '--extra-index-url', 'https://pypi.nvidia.com'])
        run('runtime_extra', [*pip, 'cffi==1.17.1', 'websockets==15.0.1',
                              'msgpack', 'msgpack-numpy', 'imageio-ffmpeg'])
        run('cpu_versions', [str(ENV/'bin/python'), '-c',
            'import importlib.metadata as m,json; '
            'print(json.dumps({p:m.version(p) for p in '
            '("omnigibson","bddl","isaacsim","torch","numpy")},sort_keys=True))'])
        write(OUT / 'status.json', dict(base, stage='installed_cpu_metadata_only',
              status='completed', seconds=time.monotonic()-started,
              actual_gpu_import_test=False, scene_or_speed_verified=False))
    except BaseException as error:
        write(OUT / 'status.json', dict(base, status='failed', error=repr(error),
                                       seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
