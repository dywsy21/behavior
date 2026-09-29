"""Bounded, hash-pinned public asset preparation, separate from active envs."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import stat
import subprocess
import time
import zipfile

ROOT = Path('/home/user/behavior_rl_speed_20260928')
REVISION = '9f0d57d465726976ed98138d3f8b8ca3e2186775'
FILES = [
    ('2026-challenge-task-instances', '2026-challenge-task-instances.zip', 108543852,
     'd4ac3d72dd585178e85d542f28d1b48233b886c80e196e2f9d3aa5993ba21f81'),
    ('omnigibson-robot-assets', 'omnigibson-robot-assets-3.8.2.zip', 641004353,
     '3d813b2181e0581cf2300a40892de70f8475fe59346ceeea4fb9bf7ff21ce126'),
    ('behavior-1k-assets', 'behavior-1k-assets-3.9.0.zip', 31457673073,
     '09e9fce600f841dc611aa96c0b1b9f9074f56f0e67898b37b39bd00c38a0095e'),
]
WALL = 7200
RESERVE = 200 * 1024**3


def safe_members(archive):
    members = archive.infolist()
    if sum(i.file_size for i in members) > 50 * 1024**3:
        raise ValueError('Unexpected expanded size')
    for member in members:
        name = PurePosixPath(member.filename)
        if (name.is_absolute() or '..' in name.parts or '\\' in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError('Unsafe archive member')
    return members


def main():
    if socket.gethostname() != 'teai-g1' or os.getuid() != 1000:
        raise ValueError('Only registered RTX host/user allowed')
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip():
        raise ValueError('Clean frozen source required')
    if shutil.disk_usage(ROOT).free < RESERVE + 100*1024**3:
        raise ValueError('Insufficient disk reserve')
    out = ROOT/'runs/assets_v1'
    out.mkdir(parents=True, exist_ok=False)
    datasets = ROOT/'datasets'
    datasets.mkdir(exist_ok=False)
    cache = ROOT/'cache/asset_zips'
    cache.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    base = dict(pid=os.getpid(), revision=REVISION, wall_limit_seconds=WALL,
                source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                reserve_bytes=RESERVE, gpu_launches=0, completed=[],
                source='hf-mirror.com mirror of behavior-1k/zipped-datasets; LFS hashes pinned',
                official_endpoint='https://huggingface.co/datasets/behavior-1k/zipped-datasets',
                robo_runtime_equivalence_verified=False)

    def write(**extra):
        temp = out/'status.pending.json'
        temp.write_text(json.dumps(dict(base, seconds=time.monotonic()-started, **extra), indent=2)+'\n')
        temp.replace(out/'status.json')

    def guard():
        if time.monotonic()-started >= WALL or shutil.disk_usage(ROOT).free < RESERVE:
            raise RuntimeError('Registered asset preparation time/disk budget reached')

    try:
        for dataset, filename, size, digest in FILES:
            guard()
            archive = cache/filename
            write(stage='download', file=filename, status='running')
            url = f'https://hf-mirror.com/datasets/behavior-1k/zipped-datasets/resolve/{REVISION}/{filename}'
            with (out/(dataset+'.download.log')).open('x') as log:
                p = subprocess.Popen(['curl', '-fL', '--connect-timeout', '20', '--retry', '2',
                    '--retry-max-time', '120', '--speed-limit', '1024', '--speed-time', '60',
                    '--max-time', str(max(1, int(WALL-(time.monotonic()-started)))),
                    '-C', '-', '-o', str(archive), url], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while p.poll() is None:
                        guard()
                        if archive.exists() and archive.stat().st_size > size:
                            raise ValueError('Download exceeds pinned file size')
                        time.sleep(2)
                    if p.returncode:
                        raise RuntimeError(f'Asset download failed: {filename}, exit={p.returncode}')
                finally:
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGTERM)
                        try: p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=10)
            if archive.stat().st_size != size:
                raise ValueError('Incomplete asset file')
            write(stage='hash', file=filename, status='running')
            h = hashlib.sha256()
            with archive.open('rb') as f:
                while block := f.read(8<<20):
                    guard(); h.update(block)
            if h.hexdigest() != digest:
                raise ValueError('Pinned LFS SHA mismatch')
            dest = datasets/dataset
            dest.mkdir(exist_ok=False)
            write(stage='extract', file=filename, status='running')
            with zipfile.ZipFile(archive) as z:
                for member in safe_members(z):
                    guard(); z.extract(member, dest)
            base['completed'].append(dict(dataset=dataset, archive=str(archive), bytes=size, sha256=digest))
        write(stage='assets_downloaded_and_hashed', status='completed')
    except BaseException as error:
        write(status='failed', error=repr(error))
        raise


if __name__ == '__main__': main()
