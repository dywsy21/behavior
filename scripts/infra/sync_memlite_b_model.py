"""Copy the hash-verified B-final model-only export through local SSH.

Transfers immutable binary assets, never overlays an active code directory.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

import sync_memlite_a4_checkpoint as common

PARENT_SHA = 'd4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef'
SOURCE = '/mnt/sdc1/robodojo/behavior_dev/memhigh_a800_benchmark_20260930/weights-v1/B-final-model.pt'
DEST = '/data/workspace/wsy/behavior2026/models/memlite-b-final-20260910'


def validate_receipt(receipt):
    if (receipt.get('status') != 'complete' or receipt.get('source_sha256') != PARENT_SHA
            or receipt.get('model_only') is not True or receipt.get('optimizer_state_exported') is not False
            or receipt.get('cuda_initialized') is not False or receipt.get('path') != SOURCE
            or not isinstance(receipt.get('bytes'), int) or not 0 < receipt['bytes'] < 26593013632
            or not re.fullmatch('[0-9a-f]{64}', receipt.get('sha256', ''))):
        raise ValueError('Not the registered, validated B-final export receipt')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--lc3-control', type=Path, required=True)
    args = ap.parse_args()
    receipt = json.loads(args.receipt.read_text())
    validate_receipt(receipt)
    common.SOURCE, common.SIZE, common.SHA256 = SOURCE, receipt['bytes'], receipt['sha256']
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.work / 'sync.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parts = args.work / 'chunks-v1'
    parts.mkdir(exist_ok=True)
    if shutil.disk_usage(args.work).free < 2 * common.SIZE + 1024**3:
        raise RuntimeError('Insufficient local staging space')
    ssh = ['ssh', '-T', '-S', str(args.lc3_control), '-o', 'BatchMode=yes', 'lc3']
    subprocess.run(ssh + ['true'], timeout=20, check=True)
    started = time.monotonic()
    status = dict(state='downloading', parent_sha256=PARENT_SHA, sha256=common.SHA256,
                  bytes=common.SIZE, source=SOURCE, destination=DEST + '/B-final-model.pt')
    def update(**values):
        status.update(values, elapsed_seconds=time.monotonic()-started)
        common.publish_json(args.work / 'status.json', status)
        print(json.dumps(status), flush=True)
    try:
        entries = common.chunks(common.SIZE, 16 * 1024**2)
        completed = total = 0
        last = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(common.download_chunk, entry, parts) for entry in entries]
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                total += result['bytes']
                if time.monotonic()-last >= 20 or completed == len(entries):
                    update(completed_chunks=completed, total_chunks=len(entries), local_bytes=total)
                    last = time.monotonic()
        assembled = args.work / 'B-final-model.verified.pt'
        if assembled.exists():
            if assembled.stat().st_size != common.SIZE or common.digest(assembled) != common.SHA256:
                raise RuntimeError('Existing verified-name asset has wrong identity')
        else:
            update(state='assembling')
            partial = args.work / 'B-final-model.assembled.partial'
            with partial.open('wb') as target:
                for i, _, _ in entries:
                    with (parts / f'{i:05d}.bin').open('rb') as source:
                        shutil.copyfileobj(source, target, 8 * 1024**2)
            if partial.stat().st_size != common.SIZE or common.digest(partial) != common.SHA256:
                raise RuntimeError('Assembled B-final content hash failed')
            partial.rename(assembled)
        subprocess.run(ssh + ['mkdir -p ' + shlex.quote(DEST)], check=True, timeout=30)
        target = DEST + '/B-final-model.pt'
        remote_partial = target + '.partial'
        subprocess.run(ssh + ['test ! -e ' + shlex.quote(target)], check=True, timeout=30)
        update(state='uploading')
        with (args.work / 'upload.log').open('ab') as log:
            subprocess.run(['rsync', '--partial', '--append-verify', '--info=progress2', '--timeout=180',
                            '-e', shlex.join(ssh[:-1]), str(assembled), 'lc3:' + remote_partial],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        update(state='remote_verification')
        sha = subprocess.check_output(ssh + ['sha256sum ' + shlex.quote(remote_partial)],
                                      text=True, timeout=300).split()[0]
        if sha != common.SHA256:
            raise RuntimeError('Destination hash mismatch; original/partial retained')
        subprocess.run(ssh + ['test ! -e ' + shlex.quote(target) + ' && mv -- ' +
                             shlex.quote(remote_partial) + ' ' + shlex.quote(target)],
                       check=True, timeout=30)
        update(state='complete', destination_sha256=sha)
    except BaseException as error:
        update(state='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
