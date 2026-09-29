"""One explicit range-download continuation, debiting the original 2h budget."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import zipfile

from rtx_assets import ROOT, REVISION, FILES, WALL, RESERVE, safe_members


def ranges(start, size, chunk=16<<20):
    if not 0 <= start <= size or chunk < 1:
        raise ValueError('Invalid pinned byte interval')
    return [(lo, min(lo+chunk, size)-1) for lo in range(start, size, chunk)]


def main():
    if socket.gethostname() != 'teai-g1' or os.getuid() != 1000:
        raise ValueError('Registered RTX host only')
    prior = json.loads((ROOT/'runs/assets_v1/status.json').read_text())
    proc = Path(f'/proc/{prior["pid"]}/stat')
    if (prior['status'] != 'failed' or prior['completed'] or prior['revision'] != REVISION
            or (proc.exists() and proc.read_text().rsplit(')', 1)[1].split()[0] != 'Z')):
        raise ValueError('Only the stopped pre-extraction v1 download can resume')
    spent = math.ceil(prior['seconds'])
    if not 0 < spent < WALL-300:
        raise ValueError('Insufficient original preparation budget')
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip():
        raise ValueError('Clean frozen source required')
    out = ROOT/'runs/assets_v2'
    out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    started = time.monotonic()
    stop = threading.Event(); lock = threading.Lock(); processes = set()
    base = dict(pid=os.getpid(), revision=REVISION, prior_seconds=spent,
        remaining_seconds=WALL-spent, original_wall_seconds=WALL,
        source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        completed=[], gpu_launches=0, parallel_ranges=4, robo_runtime_equivalence_verified=False)

    def guard():
        if (stop.is_set() or spent+time.monotonic()-started >= WALL
                or shutil.disk_usage(ROOT).free < RESERVE):
            raise RuntimeError('Original download budget/reserve reached or another range failed')

    def write(**extra):
        tmp = out/'status.pending.json'
        tmp.write_text(json.dumps(dict(base, seconds=time.monotonic()-started,
                                      cumulative_seconds=spent+time.monotonic()-started, **extra), indent=2)+'\n')
        tmp.replace(out/'status.json')

    def download(url, chunk_path, lo, hi):
        guard()
        size = hi-lo+1
        if chunk_path.is_file() and chunk_path.stat().st_size == size:
            return
        with lock:
            guard()
            log = chunk_path.with_suffix('.log').open('a')
            p = subprocess.Popen(['curl', '-fLsS', '--connect-timeout', '15', '--max-time',
                str(min(180, max(1, int(WALL-spent-(time.monotonic()-started))))),
                '--speed-limit', '512', '--speed-time', '30', '--retry', '2', '--retry-max-time', '150',
                '--max-filesize', str(size), '--range', f'{lo}-{hi}', '-o', str(chunk_path),
                '-w', '%{http_code}', url], stdout=subprocess.PIPE, stderr=log, start_new_session=True)
            processes.add(p)
        try:
            while p.poll() is None:
                guard(); time.sleep(.5)
            code = p.stdout.read().decode()
            if p.returncode or not code.endswith('206') or chunk_path.stat().st_size != size:
                raise RuntimeError(f'Incomplete HTTP206 range {lo}-{hi}, exit={p.returncode}, code={code}')
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=5)
            with lock: processes.discard(p)
            p.stdout.close(); log.close()

    try:
        for dataset, filename, size, digest in FILES:
            guard()
            archive = ROOT/'cache/asset_zips'/filename
            start = archive.stat().st_size if archive.exists() else 0
            pending_ranges = ranges(start, size)
            chunks = out/(dataset+'.chunks'); chunks.mkdir(exist_ok=False)
            url = f'https://hf-mirror.com/datasets/behavior-1k/zipped-datasets/resolve/{REVISION}/{filename}'
            write(status='running', stage='parallel_ranges', file=filename, bytes_reused=start)
            executor = ThreadPoolExecutor(max_workers=4)
            try:
                futures = {executor.submit(download, url, chunks/f'{lo}-{hi}.part', lo, hi)
                           for lo, hi in pending_ranges}
                while futures:
                    guard()
                    done, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for job in done: job.result()
            except BaseException:
                stop.set()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            write(status='running', stage='assemble_and_hash', file=filename)
            with archive.open('ab') as target:
                for lo, hi in pending_ranges:
                    guard()
                    if target.tell() != lo: raise ValueError('Noncontiguous asset assembly')
                    with (chunks/f'{lo}-{hi}.part').open('rb') as source:
                        while block := source.read(8<<20): guard(); target.write(block)
            h = hashlib.sha256()
            with archive.open('rb') as source:
                while block := source.read(8<<20): guard(); h.update(block)
            if archive.stat().st_size != size or h.hexdigest() != digest:
                raise ValueError('Complete asset SHA/size mismatch')
            dest = ROOT/'datasets'/dataset
            dest.mkdir(exist_ok=False)
            write(status='running', stage='extract', file=filename)
            with zipfile.ZipFile(archive) as z:
                for member in safe_members(z): guard(); z.extract(member, dest)
            base['completed'].append(dict(dataset=dataset, bytes=size, sha256=digest))
        write(status='completed', stage='assets_downloaded_and_hashed')
    except BaseException as error:
        stop.set()
        write(status='failed', error=repr(error))
        raise


if __name__ == '__main__': main()
